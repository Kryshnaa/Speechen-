import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes diffusion timesteps into continuous sinusoidal vectors.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionBlock(nn.Module):
    """
    A block in our U-Net that takes features, injects the time embedding, and processes them.
    Features: [batch, in_channels, H, W]
    Time Embedding: [batch, time_emb_dim]
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=256, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        # Convolutional layers
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Projection layer to map time embedding dimension to feature channel dimension
        self.time_mlp = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t_emb):
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        
        # Project time embedding and reshape to match feature shape [batch, channels, 1, 1]
        time_proj = self.time_mlp(t_emb)
        time_proj = time_proj.unsqueeze(-1).unsqueeze(-1)
        
        # Inject time embedding by addition
        x = x + time_proj
        
        x = self.relu(x)
        return x

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for Denoising Diffusion.
    Takes:
        x_t: current state [batch, 1, 256, 256]
        cond: conditioning noisy audio [batch, 1, 256, 256]
        t: timestep [batch]
    Returns:
        noise prediction: [batch, 1, 256, 256]
    """
    def __init__(self, time_emb_dim=256):
        super().__init__()
        
        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )
        
        # Encoder (Downsampling)
        # Input has 2 channels (x_t concatenated with cond noisy magnitude)
        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down1 = DiffusionBlock(2, 32, time_emb_dim, downsample=True)
        self.down2 = DiffusionBlock(32, 64, time_emb_dim, downsample=True)
        self.down3 = DiffusionBlock(64, 128, time_emb_dim, downsample=True)
        self.down4 = DiffusionBlock(128, 256, time_emb_dim, downsample=True)
        self.down5 = DiffusionBlock(256, 512, time_emb_dim, downsample=True)
        
        # Bottleneck (8x8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Upsampling with skip connections)
        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = DiffusionBlock(512, 256, time_emb_dim, downsample=False) # Skip Down4 (256) -> Concats to 512
        self.up2 = DiffusionBlock(512, 128, time_emb_dim, downsample=False) # Skip Down3 (128) -> Concats to 256
        self.up3 = DiffusionBlock(256, 64, time_emb_dim, downsample=False)  # Skip Down2 (64)  -> Concats to 128
        self.up4 = DiffusionBlock(128, 32, time_emb_dim, downsample=False)  # Skip Down1 (32)  -> Concats to 64
        
        # Final reconstruction layer
        # Output has 1 channel (predicting noise added to the target magnitude)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1) # Linear activation (no Sigmoid/Tanh) for noise prediction
        )

    def forward(self, x_t, cond, t):
        # Compute time embedding
        t_emb = self.time_mlp(t)
        
        # Concatenate current state and conditioning signal along the channel dimension
        inputs = torch.cat([x_t, cond], dim=1)
        
        # Downsampling path
        d1 = self.down1(inputs, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        d5 = self.down5(d4, t_emb)
        
        # Bottleneck
        b = self.bottleneck(d5)
        
        # Upsampling path with skip connections
        u1 = self.up1(b, t_emb)
        u1 = torch.cat([u1, d4], dim=1)
        
        u2 = self.up2(u1, t_emb)
        u2 = torch.cat([u2, d3], dim=1)
        
        u3 = self.up3(u2, t_emb)
        u3 = torch.cat([u3, d2], dim=1)
        
        u4 = self.up4(u3, t_emb)
        u4 = torch.cat([u4, d1], dim=1)
        
        out = self.final(u4)
        return out

class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) helper class.
    Handles forward process (noising) and reverse process (sampling).
    """
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        
        # Linear beta schedule
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Forward process coefficients
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        # Reverse process coefficients
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        # Shift alpha_bar to compute alpha_bar_{t-1}
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])
        self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: Add noise to clean x_0 at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, cond, t_val):
        """
        Reverse process step: Predict noise, estimate x_0, and sample x_{t-1}.
        """
        batch_size = x_t.shape[0]
        t = torch.full((batch_size,), t_val, dtype=torch.long, device=self.device)
        
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1)
        
        # Predict the noise using the conditional U-Net
        predicted_noise = model(x_t, cond, t)
        
        # Estimate the mean of x_{t-1}
        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
        
        if t_val == 0:
            # At step 0, return the clean estimate directly
            return mean
        else:
            # Otherwise, add Gaussian noise back
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            # Standard DDPM sampling
            x_prev = mean + torch.sqrt(variance) * noise
            
            # Value clipping trick: since target is normalized spectrogram [0, 1],
            # clipping helps keep the generated outputs stable during sampling.
            return torch.clamp(x_prev, 0.0, 1.0)

    @torch.no_grad()
    def sample_enhanced(self, model, cond):
        """
        Generates clean spectrogram from pure noise, guided by the noisy condition.
        """
        model.eval()
        # Start from random Gaussian noise
        x = torch.randn_like(cond)
        
        # Step backward from T-1 down to 0
        for t_val in reversed(range(self.T)):
            x = self.p_sample(model, x, cond, t_val)
            
        return x
