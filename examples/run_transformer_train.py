import torch
from vae_model import ObjectVAE
from transformer_model import Transformer, ObjectTokenizer
from transformer_train import collect_buffer, train_transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained VAE
    vae = ObjectVAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load("pong_vae.pth", map_location=device, weights_only=True))
    vae.eval()

    # Collect training data
    W = 16    # causal context window length
    H = 8     # self-forcing (rollout loss) horizon
    buffer = collect_buffer(N=512, T=256, W=W, H=H, print_every=50)

    # Initialize transformer
    d_model = 256
    n_layers = 4
    n_heads = 4
    latent_dim = 32

    tokenizer = ObjectTokenizer(vae_latent_dim=latent_dim,
                                pos_dim=2,
                                n_actions=3,
                                d_model=d_model).to(device)

    transformer = Transformer(tokenizer=tokenizer,
                              n_layers=n_layers,
                              n_heads=n_heads,
                              d_model=d_model,
                              latent_dim=latent_dim,
                              max_seq_len=W + H).to(device)

    # Train
    transformer = train_transformer(vae=vae,
                                    transformer=transformer,
                                    buffer=buffer,
                                    num_steps=4000,
                                    batch_size=64,
                                    lr=3e-4,
                                    pos_loss_weight=1.0,
                                    clip_grad_norm=1.0,
                                    gamma=0.975,
                                    print_every=100,
                                    save_path="pong_transformer.pth")

    # To load trained model:
    # tokenizer = ObjectTokenizer(vae_latent_dim=32, pos_dim=2, n_actions=3, d_model=256).to(device)
    # transformer = Transformer(tokenizer=tokenizer, n_layers=4, n_heads=4, d_model=256, latent_dim=32, max_seq_len=24).to(device)
    # transformer.load_state_dict(torch.load("pong_transformer.pth", map_location=device, weights_only=True))
    # transformer.eval()
