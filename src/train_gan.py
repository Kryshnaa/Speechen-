# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.models.gan import Generator, Discriminator
from src.utils import save_comparison_plot

def train_gan(args):
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Set device: prioritize MPS (Mac Apple Silicon) or CUDA over CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load DataLoaders
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit
    )

    # Instantiate Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers (Adam with typical GAN hyperparameters)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss Functions
    criterion_GAN = nn.MSELoss()  # LSGAN loss
    criterion_L1 = nn.L1Loss()    # Reconstruction loss

    print("Starting GAN Training...")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]")
        for batch_idx, batch in enumerate(loop):
            # Move to device
            clean_mag = batch["clean_mag"].to(device)
            noisy_mag = batch["noisy_mag"].to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate fake spectrograms
            fake_mag = generator(noisy_mag)
            
            # Classify real pairs: (noisy, clean)
            pred_real = discriminator(noisy_mag, clean_mag)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))
            
            # Classify fake pairs: (noisy, fake)
            pred_fake = discriminator(noisy_mag, fake_mag.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))
            
            # Total Discriminator loss
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Classify fake pairs again (with gradients flowing to G)
            pred_fake_for_G = discriminator(noisy_mag, fake_mag)
            
            # Adversarial Loss (want G to fool D)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G, device=device))
            
            # L1 Reconstruction Loss
            loss_G_L1 = criterion_L1(fake_mag, clean_mag)
            
            # Total Generator Loss: Adv + lambda * L1
            loss_G = loss_G_GAN + args.lambda_l1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            # Track statistics
            running_g_loss += loss_G.item()
            running_d_loss += loss_D.item()
            
            loop.set_postfix(G_Loss=loss_G.item(), D_Loss=loss_D.item())

        epoch_g_loss = running_g_loss / len(train_loader)
        epoch_d_loss = running_d_loss / len(train_loader)
        print(f"Epoch {epoch} summary - G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}")

        # Save sample spectrogram comparison from the first batch
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                # Take the first sample from the first batch
                sample_noisy = noisy_mag[0:1]
                sample_clean = clean_mag[0:1]
                sample_fake = generator(sample_noisy)
                
                sample_path = os.path.join(args.sample_dir, f"epoch_{epoch}.png")
                save_comparison_plot(sample_clean[0], sample_noisy[0], sample_fake[0], sample_path)
                print(f"Saved training progress spectrogram plot to {sample_path}")

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"gan_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f"Saved GAN checkpoint to {checkpoint_path}")

    # Run speaker separation and transcription pipeline automatically after training
    try:
        from src.speaker_separator_pipeline import run_pipeline
        class DummyArgs:
            mix = True
            audio_1 = "test_samples/p232_001_clean.wav"
            audio_2 = "test_samples/p232_002_clean.wav"
            input_audio = None
            sep_model = "speechbrain/sepformer-whamr16k"
            asr_model = "openai/whisper-tiny"
            output_dir = "test_samples/separation_test"
            
        print("\n--- Running Speaker Separation and Transcription Pipeline Check ---")
        run_pipeline(DummyArgs())
    except Exception as e:
        print(f"Failed to auto-run Speaker Separation pipeline: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN Speech Enhancement Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 reconstruction loss")
    parser.add_argument("--crop_size", type=int, default=256, help="Spectrogram crop size")
    parser.add_argument("--train_limit", type=int, default=1000, help="Limit number of training samples (None for all)")
    parser.add_argument("--test_limit", type=int, default=100, help="Limit number of testing samples (None for all)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples_gan", help="Directory to save training sample plots")
    parser.add_argument("--sample_interval", type=int, default=2, help="Interval (epochs) to save sample plots")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval (epochs) to save checkpoints")
    
    args = parser.parse_args()
    train_gan(args)
