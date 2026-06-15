import torch
import torch.nn as nn

class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling Block: Conv2D -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block: ConvTranspose2D -> BatchNorm -> Dropout (optional) -> ReLU
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5, inplace=True))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.block(x)
        # Concatenate skip connection along the channel dimension
        return torch.cat([x, skip_input], dim=1)

class Generator(nn.Module):
    """
    U-Net Generator for mapping noisy spectrograms to clean spectrograms.
    Input size: [batch, 1, 256, 256]
    Output size: [batch, 1, 256, 256]
    """
    def __init__(self):
        super().__init__()
        
        # Encoder (Downsampling)
        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.down1 = UNetDownBlock(1, 32, use_bn=False)  # No BN on first layer
        self.down2 = UNetDownBlock(32, 64)
        self.down3 = UNetDownBlock(64, 128)
        self.down4 = UNetDownBlock(128, 256)
        self.down5 = UNetDownBlock(256, 512)
        self.down6 = UNetDownBlock(512, 512, use_bn=False)  # Bottleneck
        
        # Decoder (Upsampling with skip connections)
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = UNetUpBlock(512, 512, use_dropout=True)    # Concats Down5 -> 1024 channels
        self.up2 = UNetUpBlock(1024, 256, use_dropout=True)   # Concats Down4 -> 512 channels
        self.up3 = UNetUpBlock(512, 128)                      # Concats Down3 -> 256 channels
        self.up4 = UNetUpBlock(256, 64)                       # Concats Down2 -> 128 channels
        self.up5 = UNetUpBlock(128, 32)                       # Concats Down1 -> 64 channels
        
        # Output layer (Sigmoid maps outputs to [0, 1] range of normalized magnitude)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Downsampling path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # Upsampling path with skip connections
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        
        out = self.final(u5)
        return out

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator. Classifies NxN local patches as Real or Fake.
    Input size: [batch, 2, 256, 256] (Concatenation of condition + target)
    Output size: [batch, 1, 30, 30]
    """
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: 2 x 256 x 256 -> 32 x 128 x 128
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 x 128 x 128 -> 64 x 64 x 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 32 x 32 -> 256 x 31 x 31 (Stride 1)
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output patch predictions: 256 x 31 x 31 -> 1 x 30 x 30
            # Note: No Sigmoid at the end because we train using LSGAN (Least-Squares) loss, 
            # which works best directly on the logits.
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, condition, target):
        # Concatenate the conditioning input (noisy spectrogram) and the target (clean/fake spectrogram)
        x = torch.cat([condition, target], dim=1)
        return self.model(x)
