import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pong import Pong
from render import Renderer
from vae_model import ObjectVAE

# Training Data Loader
def vae_data_loader(env, renderer, batch_size): 
    """Generate random Pong frames and extract object crops."""
    all_crops = []
    
    for _ in range(batch_size):
        env.reset(seed=np.random.randint(1_000_000))
        env.step()
        state = env.state
        prev_state = env.prev_state
    
        left, right, ball, score = renderer.render_crops(state=state, prev_state=prev_state)
        crops = np.stack([left, right, ball, score], axis=0)    # (K, H, W, 3)
        crops = crops.astype(np.float32) / 255.0
        all_crops.append(crops)

    crops = np.stack(all_crops, axis=0)             # (B, K, H, W, 3)
    crops = np.transpose(crops, (0, 1, 4, 2, 3))    # (B, K, 3, H, W)

    return crops
    

# Loss Function
def bce_loss_logits(recon_logits, x, mu, logvar, kl_weight=1.0):
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum') / x.size(0)
    
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    loss = recon_loss + (kl_weight * kl_div)
    
    return loss, recon_loss, kl_div


# KL Annealing Schedule
def linear_warmup(step, total_steps, target_beta, ramp_proportion=0.75):
    """Ramps from 0 to target_beta over the first ramp_proportion of training."""
    ramp_steps = int(total_steps * ramp_proportion)
    
    if step < ramp_steps:
        return target_beta * (step / ramp_steps)
    else:
        return target_beta


# VAE Training Loop
def train_vae(num_steps=3000, 
              batch_size=64, 
              latent_dim=32, 
              target_kl_weight=1.0, 
              ramp_proportion=0.8, 
              lr=1e-3, 
              save_path=None):
    env = Pong()
    renderer = Renderer(env.settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ObjectVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'kl_weight': []}
    
    vae.train()
    
    for step in range(1, num_steps + 1):

        crops, _ = vae_data_loader(env, renderer, batch_size=batch_size)        
        B, K, C, H, W = crops.shape
        crops_flat = torch.from_numpy(crops).reshape(B*K, C, H, W).to(device)
        
        recon_logits, mu, logvar = vae(crops_flat)
        
        kl_weight = linear_warmup(step, num_steps, target_kl_weight, ramp_proportion)
        loss, rec_loss, kl_loss = bce_loss_logits(recon_logits, crops_flat, mu, logvar, kl_weight)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['total_loss'].append(loss.item())
        history['recon_loss'].append(rec_loss.item())
        history['kl_loss'].append(kl_loss.item())
        history['kl_weight'].append(kl_weight)
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d} | Loss: {loss.item():.3f} | "
                  f"Recon: {rec_loss.item():.3f} | KL: {kl_loss.item():.3f} | "
                  f"KL_w: {kl_weight:.3f}")
    
    if save_path:
        torch.save(vae.state_dict(), save_path)
    
    return vae, history
