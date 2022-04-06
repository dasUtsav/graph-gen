import torch
from torch import nn
from config import config

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc_mu = nn.Linear(4 * config['hidden_size'], config['vae_latent_dim'])
        self.fc_var = nn.Linear(4 * config['hidden_size'], config['vae_latent_dim'])
        self.max_sample = True

    def get_mu_logvar(self, input):

        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return mu, log_var

    def reparameterize(self, mu, log_var):

        # reparameterization trick
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        # temperature reference: self.z_mean + tf.scalar_mul(self.z_temperature, epsilon * tf.exp(self.z_log_sigma))  # N(mu, I * sigma**2)
        z = config['vae_temp'] * eps * std + mu

        return z

    def forward(self, input):
        # Add VAE here

        mu, log_var = self.get_mu_logvar(input)

        z = self.reparameterize(mu, log_var)

        # KL loss
        kld = (-0.5 * torch.sum(log_var - torch.pow(mu, 2) - torch.exp(log_var) + 1, 1)).mean().squeeze()

        return z, kld