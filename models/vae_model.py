import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        self.latent_dim = latent_dim

        # Encoder: input (B, 3, crop_size, crop_size)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)    
        self.enc_conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1)  
        self.enc_conv3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)  
        
        self.fc_flat_size = 192 * 5 * 5    # 4800
        self.fc_mu     = nn.Linear(self.fc_flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_flat_size, latent_dim)

        # Decoder 
        self.dec_fc = nn.Linear(latent_dim, self.fc_flat_size)

        self.dec_conv0 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)                         
        self.dec_conv1 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.dec_conv2 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)    

    def encode(self, x):
        x = F.leaky_relu(self.enc_conv1(x), 0.2)
        x = F.leaky_relu(self.enc_conv2(x), 0.2)
        x = F.leaky_relu(self.enc_conv3(x), 0.2)
        x_flat = x.reshape(-1, self.fc_flat_size)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def decode(self, z):
        z = F.leaky_relu(self.dec_fc(z), 0.2)
        z = z.reshape(-1, 192, 5, 5)
        z = F.leaky_relu(self.dec_conv0(z), 0.2)
        z = F.leaky_relu(self.dec_conv1(z), 0.2)
        return self.dec_conv2(z)  

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
