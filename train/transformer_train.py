import numpy as np
import torch
import torch.nn.functional as F
from pong import Pong
from render import Renderer

def collect_buffer(N: int, 
                   T: int = 300, 
                   W: int = 1,
                   H: int = 1,
                   seed: int | None = None,
                   print_every: int = 50):
    rng = np.random.default_rng(seed)
    env = Pong()
    renderer = Renderer(env.settings)

    paddle_left_x = env.settings["paddle_left_x"]
    paddle_right_x = env.settings["paddle_right_x"]
    score_center = env.settings["score_center"]
    resolution = env.settings["resolution"]

    crops_buf = None
    pos_buf = None
    left_a_buf = np.empty((N, T), dtype=np.int32)
    right_a_buf = np.empty((N, T), dtype=np.int32)

    for ep in range(N):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        env.step(left_action=None, right_action=None)

        for t in range(T):
            left_t, right_t, ball_t, score_t = renderer.render_crops(state=env.state, prev_state=env.prev_state)
            crops = np.stack([left_t, right_t, ball_t, score_t], axis=0)
            crops = np.transpose(crops, (0, 3, 1, 2))

            pos = np.array([[paddle_left_x, env.state["paddle_left_y"]],
                            [paddle_right_x, env.state["paddle_right_y"]],
                            [env.state["ball_x"], env.state["ball_y"]],
                            [score_center[0], score_center[1]]], dtype=np.float32)
            pos = pos / resolution * 2.0 - 1.0

            _, info = env.step(left_action=None, right_action=None)

            if crops_buf is None:
                crops_buf = np.empty((N, T) + crops.shape, dtype=np.uint8)
                pos_buf = np.empty((N, T) + pos.shape, dtype=np.float32)

            crops_buf[ep, t] = crops
            pos_buf[ep, t] = pos
            left_a_buf[ep, t] = int(info["left_action"])
            right_a_buf[ep, t] = int(info["right_action"])

        if ((ep + 1) % print_every) == 0 or ep == 0:
            print("episode:", ep + 1)

    buffer = {"crops": crops_buf,           # (N, T, K, 3, H_img, W_img) uint8
              "pos": pos_buf,               # (N, T, K, 2)
              "left_a": left_a_buf,         # (N, T)
              "right_a": right_a_buf,       # (N, T)
              "W": W,
              "H": H}
    return buffer


def sample_batch(buffer: dict, batch_size: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    W = buffer["W"]
    H = buffer["H"]
    N, T = buffer["crops"].shape[:2]
    B = int(batch_size)

    # Valid tp1 range: [W, T-H]
    valid_tp1 = np.arange(W, T - H + 1)
    n_valid_per_ep = len(valid_tp1)
    n_total = N * n_valid_per_ep

    flat_idx = rng.integers(0, n_total, size=B)
    ep_idx = (flat_idx // n_valid_per_ep).astype(np.int32)
    tp1_idx = (flat_idx % n_valid_per_ep + W).astype(np.int32)

    # Window: [tp1-W, ..., tp1-1]
    window_offsets = np.arange(-W, 0)
    t_windows = tp1_idx[:, None] + window_offsets

    crops_win = buffer["crops"][ep_idx[:, None], t_windows].astype(np.float32) / 255.0
    pos_win = buffer["pos"][ep_idx[:, None], t_windows]
    left_a_win = buffer["left_a"][ep_idx[:, None], t_windows]
    right_a_win = buffer["right_a"][ep_idx[:, None], t_windows]

    # Horizon: [tp1, tp1+1, ..., tp1+H-1]
    hor_offsets = np.arange(H)
    t_hor = tp1_idx[:, None] + hor_offsets

    crops_hor = buffer["crops"][ep_idx[:, None], t_hor].astype(np.float32) / 255.0
    pos_hor = buffer["pos"][ep_idx[:, None], t_hor]
    left_a_hor = buffer["left_a"][ep_idx[:, None], t_hor]
    right_a_hor = buffer["right_a"][ep_idx[:, None], t_hor]

    batch = {"crops_win": crops_win,
             "pos_win": pos_win,
             "left_a_win": left_a_win,
             "right_a_win": right_a_win,
             "crops_hor": crops_hor,
             "pos_hor": pos_hor,
             "left_a_hor": left_a_hor,
             "right_a_hor": right_a_hor,
             "ep_idx": ep_idx,
             "tp1_idx": tp1_idx}
    return batch


def train_transformer(vae,
                      transformer,
                      buffer,
                      num_steps: int = 3000,
                      batch_size: int = 64,
                      lr: float = 3e-4,
                      pos_loss_weight: float = 1.0,
                      clip_grad_norm: float = 1.0,
                      gamma: float = 0.975,
                      print_every: int = 100,
                      save_path = None):

    device = next(transformer.parameters()).device
    H = buffer["H"]

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    transformer.train()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr/10)

    for step in range(1, num_steps + 1):

        # Sample batch
        batch_np = sample_batch(buffer=buffer, batch_size=batch_size)

        crops_win = torch.from_numpy(batch_np["crops_win"]).to(device)             # (B, W, K, C, H_img, W_img)
        crops_hor = torch.from_numpy(batch_np["crops_hor"]).to(device)             # (B, H, K, C, H_img, W_img)
        pos_win = torch.from_numpy(batch_np["pos_win"]).to(device)                 # (B, W, K, 2)
        pos_hor = torch.from_numpy(batch_np["pos_hor"]).to(device)                 # (B, H, K, 2)
        left_a_win = torch.from_numpy(batch_np["left_a_win"]).long().to(device)    # (B, W)
        right_a_win = torch.from_numpy(batch_np["right_a_win"]).long().to(device)  # (B, W)
        left_a_hor = torch.from_numpy(batch_np["left_a_hor"]).long().to(device)    # (B, H)
        right_a_hor = torch.from_numpy(batch_np["right_a_hor"]).long().to(device)  # (B, H)

        # VAE: encode crops
        B, W, K = crops_win.shape[:3]
        flat_win = crops_win.view(B*W*K, *crops_win.shape[3:])
        flat_hor = crops_hor.view(B*H*K, *crops_hor.shape[3:])
        
        with torch.no_grad():
            mu_win, _ = vae.encode(flat_win)
            latents_win = mu_win.view(B, W, K, -1)   # (B, W, K, latent_dim)

            mu_hor, _ = vae.encode(flat_hor)
            latents_hor = mu_hor.view(B, H, K, -1)   

        # Autoregressive rollout loss
        loss_lat = 0.0
        loss_pos = 0.0

        for h in range(H):
            out = transformer(latents=latents_win,
                              pos=pos_win,
                              left_actions=left_a_win,
                              right_actions=right_a_win)

            pred_latents = out["pred_latents"][:, -1]    
            pred_pos = out["pred_pos"][:, -1]            

            target_latents = latents_hor[:, h]       
            target_pos = pos_hor[:, h]              

            weight = gamma ** h     # gamma^0 = 1 for h=0
            loss_lat = loss_lat + weight * F.mse_loss(pred_latents, target_latents)
            loss_pos = loss_pos + weight * F.mse_loss(pred_pos, target_pos)

            # Self-forcing: drop oldest, append prediction
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

        if (step % print_every) == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step:5d} | loss {loss.item():.6f} | lat {loss_lat.item():.6f} | pos {loss_pos.item():.6f} | lr {current_lr:.2e}")

    if save_path:
        torch.save(transformer.state_dict(), save_path)

    return transformer
                        
