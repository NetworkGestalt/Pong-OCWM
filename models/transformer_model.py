import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RotaryEmbedding(nn.Module):
    """Applies RoPE to q/k tensors shaped (B, heads, seq_len, head_dim)."""
    def __init__(self, dim: int, max_seq_len: int = 128, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)    
        self.register_buffer("cos_cached", freqs.cos()) 
        self.register_buffer("sin_cached", freqs.sin()) 

    def forward(self, x: Tensor) -> Tensor:
        S = x.shape[2]
        cos = self.cos_cached[:S].unsqueeze(0).unsqueeze(0)   
        sin = self.sin_cached[:S].unsqueeze(0).unsqueeze(0)   
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        return torch.stack([x_even * cos - x_odd * sin,
                            x_even * sin + x_odd * cos], dim=-1).flatten(-2)


class ObjectTokenizer(nn.Module):
    """Generates a batch of tokens of shape (B, W, 6, d_model) for [obj1, obj2, obj3, obj4, actL, actR].
        Includes three types of embedding:
          - object_id_emb: left paddle / right paddle / ball / score
          - action_id_emb: left paddle / right paddle
          - action_embedding: up / down / still (shared by both paddles)"""

    def __init__(self,
                 vae_latent_dim: int = 32,
                 pos_dim: int = 2,
                 n_actions: int = 3,
                 d_model: int = 256):
        super().__init__()

        self.d_model = d_model
        self.num_objects = 4
        self.tokens_per_step = 6

        # Object projection: (latent + pos) -> d_model
        obj_in_dim = vae_latent_dim + pos_dim
        self.obj_layer_norm = nn.LayerNorm(obj_in_dim)
        self.obj_proj = nn.Linear(obj_in_dim, d_model)

        # Action embedding (0=down, 1=still, 2=up)
        self.action_embedding = nn.Embedding(n_actions, d_model)

        # Action identity embedding (0=left, 1=right)
        self.action_id_emb = nn.Embedding(2, d_model)
        self.register_buffer("action_sides", torch.tensor([0, 1], dtype=torch.long))

        # Object identity embedding (0=left_paddle, 1=right_paddle, 2=ball, 3=score)
        self.obj_id_emb = nn.Embedding(self.num_objects, d_model)
        self.register_buffer("object_ids", torch.tensor([0, 1, 2, 3], dtype=torch.long))

    def forward(self,
                latents: Tensor,            # (B, W, K, latent_dim)
                pos: Tensor,                # (B, W, K, 2)
                left_actions: Tensor,       # (B, W)
                right_actions: Tensor):     # (B, W)

        B, W, K, _ = latents.shape

        # Object tokens
        obj_features = torch.cat([latents, pos], dim=-1)         
        obj_features = self.obj_layer_norm(obj_features)
        obj_tokens = self.obj_proj(obj_features) + self.obj_id_emb(self.object_ids).view(1, 1, K, -1)    

        # Action tokens (content + identity)
        side_embs = self.action_id_emb(self.action_sides)       

        left_actions = left_actions.long() + 1      # shift action indicators {-1, 0, 1} -> {0, 1, 2}
        right_actions = right_actions.long() + 1
        left_tok = self.action_embedding(left_actions) + side_embs[0].view(1, 1, -1)    
        right_tok = self.action_embedding(right_actions) + side_embs[1].view(1, 1, -1)    

        act_tokens = torch.stack([left_tok, right_tok], dim=2)                      

        x = torch.cat([obj_tokens, act_tokens], dim=2)
        return x


class SpatiotemporalAttention(nn.Module):
    """Factorized Spatiotemporal Attention:
        - Spatial attention: contextualizes within each timestep.
        - Causal temporal attention: attends to previous contextualized states with RoPE."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Spatial attention (standard MHA)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # Temporal attention (with RoPE)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_q = nn.Linear(d_model, d_model)
        self.temporal_k = nn.Linear(d_model, d_model)
        self.temporal_v = nn.Linear(d_model, d_model)
        self.temporal_out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Feedforward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                nn.GELU(),
                                nn.Linear(4 * d_model, d_model))

    def _spatial_attn(self, x: Tensor) -> Tensor:
        """x: (B*W, S, D) -> (B*W, S, D)"""
        x_norm = self.spatial_norm(x)
        out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        return x + out

    def _temporal_attn(self, x: Tensor, causal_mask: Tensor = None) -> Tensor:
        """x: (B*S, W, D) -> (B*S, W, D)"""
        B, W, D = x.shape
        x_norm = self.temporal_norm(x)

        q = self.temporal_q(x_norm).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.temporal_k(x_norm).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.temporal_v(x_norm).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, W, D)
        return x + self.temporal_out(out)

    def forward(self, x: Tensor, causal_mask: Tensor = None) -> Tensor:
        B, W, S, D = x.shape

        # Spatial: attend within each timestep
        x = self._spatial_attn(x.reshape(B * W, S, D)).view(B, W, S, D)

        # Temporal: attend contextualized states across timesteps (per token slot)
        x = self._temporal_attn(x.permute(0, 2, 1, 3).reshape(B * S, W, D), causal_mask)
        x = x.view(B, S, W, D).permute(0, 2, 1, 3)

        # Feedforward
        x = x + self.ff(self.ff_norm(x))
        return x.contiguous()


class Transformer(nn.Module):
    def __init__(self,
                 tokenizer: nn.Module,
                 n_layers: int = 3,
                 d_model: int = 256,
                 n_heads: int = 4,
                 latent_dim: int = 32,
                 max_seq_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.tokenizer = tokenizer
        self.layers = nn.ModuleList([SpatiotemporalAttention(d_model=d_model, n_heads=n_heads, 
                                                              max_seq_len=max_seq_len, dropout=dropout)
                                     for _ in range(n_layers)])

        self.latent_head = nn.Linear(d_model, latent_dim)
        self.pos_head = nn.Linear(d_model, 2)

    def forward(self,
                latents: Tensor,          # (B, W, K, latent_dim)
                pos: Tensor,              # (B, W, K, 2)
                left_actions: Tensor,     # (B, W)
                right_actions: Tensor):   # (B, W)

        x = self.tokenizer(latents, pos, left_actions, right_actions)    

        # Causal mask for temporal attention (W, W)
        W = x.shape[1]
        causal_mask = torch.triu(torch.ones(W, W, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)      

        obj_tokens = x[:, :, :4, :]
        pred_latents = self.latent_head(obj_tokens)      
        pred_pos = self.pos_head(obj_tokens)          

        out = {"pred_latents": pred_latents,
               "pred_pos": pred_pos}

        return out
                  
