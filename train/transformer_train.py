import numpy as np
import torch
import torch.nn.functional as F
from vae_model import ObjectVAE
from transformer_model import Transformer

# ------------------------------
# Transformer Batch Sampler
# ------------------------------

def sample_transformer_batch(buffer: dict, batch_size: int, W: int = 16, H: int = 8,
                             seed: int | None = None) -> dict:
    """Sample a batch of (context window, self-forcing horizon) pairs from the buffer.
       For each sample, selects a random episode and timestep tp1, then returns:
          - Window: frames [tp1-W, ..., tp1-1] (context for prediction)
          - Horizon: frames [tp1, ..., tp1+H-1] (targets for autoregressive prediction)"""
    rng = np.random.default_rng(seed)
    N, T = buffer["crops"].shape[:2]
    B = int(batch_size)

    # Valid tp1 range [W, T-H] ensures both window and horizon fit within episode
    valid_tp1 = np.arange(W, T - H + 1)
    n_valid_per_ep = len(valid_tp1)
    n_total = N * n_valid_per_ep

    # Sample flat indices and convert to (episode, timestep) pairs
    flat_idx = rng.integers(0, n_total, size=B)
    ep_idx = (flat_idx // n_valid_per_ep).astype(np.int32)
    tp1_idx = (flat_idx % n_valid_per_ep + W).astype(np.int32)

    # Context window: [tp1-W, ..., tp1-1]
    window_offsets = np.arange(-W, 0)
    t_windows = tp1_idx[:, None] + window_offsets

    crops_win = buffer["crops"][ep_idx[:, None], t_windows].astype(np.float32) / 255.0
    pos_win = buffer["pos"][ep_idx[:, None], t_windows]
    left_a_win = buffer["left_a"][ep_idx[:, None], t_windows]
    right_a_win = buffer["right_a"][ep_idx[:, None], t_windows]

    # Prediction horizon: [tp1, tp1+1, ..., tp1+H-1]
    hor_offsets = np.arange(H)
    t_hor = tp1_idx[:, None] + hor_offsets

    crops_hor = buffer["crops"][ep_idx[:, None], t_hor].astype(np.float32) / 255.0
    pos_hor = buffer["pos"][ep_idx[:, None], t_hor]
    left_a_hor = buffer["left_a"][ep_idx[:, None], t_hor]
    right_a_hor = buffer["right_a"][ep_idx[:, None], t_hor]

    batch = {"crops_win": crops_win,      # (B, W, K, 3, H_img, W_img)
             "pos_win": pos_win,          # (B, W, K, 2)
             "left_a_win": left_a_win,    # (B, W)
             "right_a_win": right_a_win,  # (B, W)
             "crops_hor": crops_hor,      # (B, H, K, 3, H_img, W_img)
             "pos_hor": pos_hor,          # (B, H, K, 2)
             "left_a_hor": left_a_hor,    # (B, H)
             "right_a_hor": right_a_hor,  # (B, H)
             "ep_idx": ep_idx,
             "tp1_idx": tp1_idx}

    return batch

# ------------------------------
# Transformer Training Loop
# ------------------------------

def train_transformer(vae: ObjectVAE,
                      transformer: Transformer,
                      buffer: dict,
                      W: int = 16,
                      H: int = 8,
                      num_steps: int = 3000,
                      batch_size: int = 64,
                      lr: float = 3e-4,
                      pos_loss_weight: float = 1.0,
                      clip_grad_norm: float = 1.0,
                      gamma: float = 0.975,
                      print_every: int = 100,
                      save_path: str | None = None) -> tuple[Transformer, dict]:
    """Train the transformer dynamics model with self-forcing (autoregressive rollout loss).
       At each step, predicts H future frames by feeding predictions back as input. Loss is MSE on latents and positions, 
       with exponential decay (gamma^h) over the horizon to help stabilize gradients."""
    device = next(transformer.parameters()).device
    H = buffer["H"]

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    transformer.train()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr/10)

    training_log = {'total_loss': [], 'lat_loss': [], 'pos_loss': [], 'lr': []}
                        
    for step in range(1, num_steps + 1):
        batch_np = sample_transformer_batch(buffer=buffer, batch_size=batch_size, W=W, H=H)

        # Move batch arrays to device
        crops_win = torch.from_numpy(batch_np["crops_win"]).to(device) 
        crops_hor = torch.from_numpy(batch_np["crops_hor"]).to(device)         
        pos_win = torch.from_numpy(batch_np["pos_win"]).to(device)                 
        pos_hor = torch.from_numpy(batch_np["pos_hor"]).to(device)                 
        left_a_win = torch.from_numpy(batch_np["left_a_win"]).long().to(device)    
        right_a_win = torch.from_numpy(batch_np["right_a_win"]).long().to(device)  
        left_a_hor = torch.from_numpy(batch_np["left_a_hor"]).long().to(device)    
        right_a_hor = torch.from_numpy(batch_np["right_a_hor"]).long().to(device)  
      
        # Encode crops to latents using frozen VAE
        B, W, K = crops_win.shape[:3]
        flat_win = crops_win.view(B*W*K, *crops_win.shape[3:])
        flat_hor = crops_hor.view(B*H*K, *crops_hor.shape[3:])
        
        with torch.no_grad():
            mu_win, _ = vae.encode(flat_win)
            latents_win = mu_win.view(B, W, K, -1)   # (B, W, K, latent_dim)

            mu_hor, _ = vae.encode(flat_hor)
            latents_hor = mu_hor.view(B, H, K, -1)   

        # Autoregressive rollout over prediction horizon
        loss_lat = 0.0
        loss_pos = 0.0

        for h in range(H):
            out = transformer(latents=latents_win,
                              pos=pos_win,
                              left_actions=left_a_win,
                              right_actions=right_a_win)

            # Predict next step from last position in context
            pred_latents = out["pred_latents"][:, -1]    
            pred_pos = out["pred_pos"][:, -1]            

            # Ground truth for this horizon step
            target_latents = latents_hor[:, h]       
            target_pos = pos_hor[:, h]              

            # Exponentially decaying loss weight over horizon
            weight = gamma ** h     # gamma^0 = 1 for h=0
            loss_lat = loss_lat + weight * F.mse_loss(pred_latents, target_latents)
            loss_pos = loss_pos + weight * F.mse_loss(pred_pos, target_pos)

            # Self-forcing: slide window forward, append prediction as new context
            latents_win = torch.cat([latents_win[:, 1:], pred_latents.unsqueeze(1)], dim=1)
            pos_win = torch.cat([pos_win[:, 1:], pred_pos.unsqueeze(1)], dim=1)
            left_a_win = torch.cat([left_a_win[:, 1:], left_a_hor[:, h:h+1]], dim=1)
            right_a_win = torch.cat([right_a_win[:, 1:], right_a_hor[:, h:h+1]], dim=1)

        loss = loss_lat + pos_loss_weight * loss_pos

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=float(clip_grad_norm))

        optimizer.step()
        scheduler.step()

        training_log['total_loss'].append(loss.item())
        training_log['lat_loss'].append(loss_lat.item())
        training_log['pos_loss'].append(loss_pos.item())
        training_log['lr'].append(scheduler.get_last_lr()[0])

        if (step % print_every) == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step:5d} | loss {loss.item():.6f} | lat {loss_lat.item():.6f} | "
                  f"pos {loss_pos.item():.6f} | lr {current_lr:.2e}")

    if save_path:
        torch.save(transformer.state_dict(), save_path)

    return transformer, training_log
                        