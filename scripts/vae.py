from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class VAE_scRNA(nn.Module):
    """
    Variational Autoencoder for scRNA-seq with superior biological clustering.

    Architecture:
    - Encoder: input_dim → 512 → 128 → latent_dim
    - Decoder: latent_dim → 128 → 512 → input_dim (symmetric)
    """
    def __init__(self, input_dim: int, latent_dim: int = 20):
        super(VAE_scRNA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        # Output head for gene reconstruction
        self.decoder_output = nn.Linear(512, input_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Returns:
        - recon_x: Reconstructed gene expression tensor.
        - mu: Latent mean tensor.
        - log_var: Latent log-variance tensor.
        """
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)

        dec_h = self.decoder(z)
        # Ensure positive reconstruction for MSE loss
        recon_x = F.softplus(self.decoder_output(dec_h))

        return recon_x, mu, log_var


# ---------- Loss functions ----------

def reconstruction_loss(x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
    """
    MSE reconstruction loss (sum over all entries).
    """
    return F.mse_loss(recon_x, x, reduction="sum")


def elbo_loss(
    x: torch.Tensor,
    recon_x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    ELBO loss used in pretraining:
    ELBO = Reconstruction Loss (MSE, sum) + beta * KL Divergence (sum).
    """
    recon = reconstruction_loss(x, recon_x)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl


def compute_vae_loss(
    batch_x_original: torch.Tensor,
    batch_x_reconstructed: torch.Tensor,
    batch_mu: torch.Tensor,
    batch_log_var: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Batch-mean VAE loss (for joint training):
    L_vae = MSE(x_orig, x_recon) + beta * KL(q(z|x) || p(z)).
    where KL = -0.5 * mean(1 + log(σ²) - μ² - σ²)
    
    During joint training, this keeps the VAE decoder calibrated while the policy
    networks learn trajectories. The VAE's role is purely dimensional reduction:
    compress 3000D gene expression → 20D latent space.
    
    Components:
    -----------
    1. Reconstruction Loss (MSE in gene space [3000D]):
       - Ensures VAE can encode/decode cells accurately
       - Operates on pointwise gene expression errors
       - Naturally scales to gene expression magnitudes
    
    2. KL Divergence (in latent space [20D]):
       - Regularizes latent distribution to match Gaussian prior N(0,I)
       - Encourages smooth latent representations
       - Naturally scales to latent dimensionality (no extra scaling needed)
    """
    recon_loss = F.mse_loss(batch_x_original, batch_x_reconstructed)
    kl_loss = -0.5 * torch.mean(
        1 + batch_log_var - batch_mu.pow(2) - batch_log_var.exp()
    )
    return recon_loss + beta * kl_loss


# ---------- Utilities ----------

def build_vae_dataloader(
    expression_data: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Wrap expression tensor into a TensorDataset/DataLoader for VAE training.
    """
    dataset: TensorDataset = TensorDataset(expression_data)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

def load_trained_vae(
    model_path: str,
    input_dim: int,
    latent_dim: int,
    device: torch.device,
) -> VAE_scRNA:
    """
    Load a pretrained VAE from state_dict path.
    """
    vae = VAE_scRNA(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state = torch.load(model_path, map_location=device)
    vae.load_state_dict(state)
    vae.eval()
    return vae


def get_vae_decoder(vae: VAE_scRNA) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return decoder mapping z -> reconstructed gene expression.
    """
    return lambda z: vae.decoder_output(vae.decoder(z))


def compute_latent_embeddings(
    vae: VAE_scRNA,
    expression_data: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute latent embeddings (mu) for all cells.
    """
    vae.to(device)
    vae.eval()
    with torch.no_grad():
        h = vae.encoder(expression_data.to(device))
        mu, _ = vae.fc_mu(h), vae.fc_log_var(h)
    return mu.cpu()