# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
import os
import argparse
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.models.diffusion import ConditionalUNet, DDPM
from src.utils import save_comparison_plot

def train_diffusion(args):
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Set device
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

    # Instantiate model and DDPM helper
    model = ConditionalUNet().to(device)
    ddpm = DDPM(T=args.timesteps, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("Starting Diffusion Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]")
        for batch_idx, batch in enumerate(loop):
            clean_mag = batch["clean_mag"].to(device)
            noisy_mag = batch["noisy_mag"].to(device)
            
            # Sample random timesteps for each image in the batch
            batch_size = clean_mag.shape[0]
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            
            # Generate noised clean spectrogram x_t
            x_t, noise = ddpm.q_sample(clean_mag, t)
            
            # Predict noise added
            optimizer.zero_grad()
            pred_noise = model(x_t, noisy_mag, t)
            
            # Loss is MSE between added noise and predicted noise
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(Loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} summary - Loss: {epoch_loss:.5f}")

        # Save sample denoised plot
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            # Get an enhancement sample using the reverse loop
            sample_noisy = noisy_mag[0:1]
            sample_clean = clean_mag[0:1]
            sample_enhanced = ddpm.sample_enhanced(model, sample_noisy)
            
            sample_path = os.path.join(args.sample_dir, f"epoch_{epoch}.png")
            save_comparison_plot(sample_clean[0], sample_noisy[0], sample_enhanced[0], sample_path)
            print(f"Saved training progress spectrogram plot to {sample_path}")

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"diffusion_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved Diffusion checkpoint to {checkpoint_path}")

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
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Speech Enhancement Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion timesteps (T)")
    parser.add_argument("--crop_size", type=int, default=256, help="Spectrogram crop size")
    parser.add_argument("--train_limit", type=int, default=1000, help="Limit number of training samples (None for all)")
    parser.add_argument("--test_limit", type=int, default=100, help="Limit number of testing samples (None for all)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples_diffusion", help="Directory to save sample plots")
    parser.add_argument("--sample_interval", type=int, default=2, help="Interval (epochs) to save sample plots")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval (epochs) to save checkpoints")
    
    args = parser.parse_args()
    train_diffusion(args)
