import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from pong import Pong
from render import Renderer
from vae_model import ObjectVAE

# ------------------------------
# VAE Training Data Collector
# ------------------------------

def vae_data_loader(env: Pong, renderer: Renderer, batch_size: int) -> np.ndarray:
    """Generate random Pong frames and extract object crops for VAE training.
        Returns crops as (B, K, 3, H, W) float32 array normalized to [0, 1]."""
    all_crops = []
    
    for _ in range(batch_size):
        env.reset(seed=np.random.randint(1_000_000))
        env.step(left_action=None, right_action=None)   # burn-in one step to get a valid prev_state
    
        left, right, ball, score = renderer.render_crops(state=env.state, prev_state=env.prev_state)
        crops = np.stack([left, right, ball, score], axis=0)    # (K, H, W, 3)
        crops = crops.astype(np.float32) / 255.0
        all_crops.append(crops)

    crops = np.stack(all_crops, axis=0)                         # (B, K, H, W, 3)
    crops = np.transpose(crops, (0, 1, 4, 2, 3))                # (B, K, 3, H, W) for PyTorch convolutions

    return crops

# ------------------------------
# Loss Function
# ------------------------------

def bce_loss_logits(recon_logits: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, kl_weight: float = 1.0
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Compute VAE loss: reconstruction + KL divergence. Uses BCE for reconstruction 
        due to the bit-like appearance of Pong objects (found to work better than MSE)."""
   
    # Reconstruction loss: binary cross-entropy (averaged over batch)
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum') / x.size(0)

    # Closed-form KL divergence of q(z|x) from a Gaussian prior (mode-seeking)
    kl_div_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    loss = recon_loss + (kl_weight * kl_div_loss)
    
    return loss, recon_loss, kl_div_loss

# ------------------------------
# KL Annealing Schedule
# ------------------------------

def linear_warmup(step: int, total_steps: int, target_beta: float, ramp_proportion: float = 0.75) -> float:
    """Linearly ramp KL weight from 0 to target_beta over the first ramp_proportion of training."""
    ramp_steps = int(total_steps * ramp_proportion)
    
    if step < ramp_steps:
        return target_beta * (step / ramp_steps)
    else:
        return target_beta

# ------------------------------
# VAE Training Loop
# ------------------------------

def train_vae(num_steps: int = 3000,
              batch_size: int = 64,
              latent_dim: int = 32,
              target_kl_weight: float = 1.0,
              ramp_proportion: float = 0.8,
              lr: float = 1e-3,
              save_path: str = None) -> tuple[ObjectVAE, dict]:
    """Train VAE on randomly generated Pong object crops with KL annealing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  
    env = Pong()
    renderer = Renderer(env.settings)
                  
    vae = ObjectVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    training_log = {'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'kl_weight': []}
    
    vae.train()
    for step in range(1, num_steps + 1):
        crops = vae_data_loader(env, renderer, batch_size=batch_size)        
        B, K, C, H, W = crops.shape
        crops_flat = torch.from_numpy(crops).reshape(B*K, C, H, W).to(device)

        # Forward pass
        recon_logits, mu, logvar = vae(crops_flat)

        # Compute loss with annealed KL weight
        kl_weight = linear_warmup(step, num_steps, target_kl_weight, ramp_proportion)
        loss, rec_loss, kl_loss = bce_loss_logits(recon_logits, crops_flat, mu, logvar, kl_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_log['total_loss'].append(loss.item())
        training_log['recon_loss'].append(rec_loss.item())
        training_log['kl_loss'].append(kl_loss.item())
        training_log['kl_weight'].append(kl_weight)
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d} | Loss: {loss.item():.3f} | "
                  f"Recon: {rec_loss.item():.3f} | KL: {kl_loss.item():.3f} | "
                  f"KL_w: {kl_weight:.3f}")
    
    if save_path:
        torch.save(vae.state_dict(), save_path)
    
    return vae, training_log
                  

