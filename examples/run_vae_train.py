from vae_train import train_vae
import numpy as np

if __name__ == "__main__":
    buffer = dict(np.load("pong_buffer.npz"))

    vae, training_log = train_vae(buffer=buffer,
                                  num_steps=3000,
                                  batch_size=64,
                                  latent_dim=32,
                                  target_kl_weight=1.0,
                                  ramp_proportion=0.75,
                                  lr=1e-3,
                                  save_path="pong_vae.pth")

    # To load trained model:
    # import torch
    # from vae_model import ObjectVAE
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vae = ObjectVAE(latent_dim=32).to(device)
    # vae.load_state_dict(torch.load("pong_vae.pth", map_location=device, weights_only=True))
    # vae.eval()
  