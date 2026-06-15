import torch
import sys
import os

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dataset import get_dataloaders
from src.models.gan import Generator, Discriminator
from src.models.diffusion import ConditionalUNet, DDPM
from src.utils import reconstruct_waveform_from_spec, compute_snr

def run_sanity_check():
    print("=== STARTING SPEECH ENHANCEMENT PIPELINE SANITY CHECK ===")
    
    # 1. Test device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 2. Test dataset loading (limit train to 2 samples, test to 2 samples)
    try:
        train_loader, test_loader = get_dataloaders(batch_size=2, train_limit=2, test_limit=2)
        batch = next(iter(train_loader))
        print("Dataset loaded successfully!")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"noisy_mag shape: {batch['noisy_mag'].shape}") # Expected: [2, 1, 256, 256]
        print(f"clean_mag shape: {batch['clean_mag'].shape}") # Expected: [2, 1, 256, 256]
        print(f"noisy_phase shape: {batch['noisy_phase'].shape}") # Expected: [2, 256, 256]
        print(f"max_val: {batch['max_val']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    # 3. Test GAN Generator and Discriminator forward pass
    try:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        
        noisy_mag_dev = batch["noisy_mag"].to(device)
        clean_mag_dev = batch["clean_mag"].to(device)
        
        fake_mag = generator(noisy_mag_dev)
        print(f"Generator output shape: {fake_mag.shape}") # Expected: [2, 1, 256, 256]
        
        pred_real = discriminator(noisy_mag_dev, clean_mag_dev)
        pred_fake = discriminator(noisy_mag_dev, fake_mag)
        print(f"Discriminator output shape (real): {pred_real.shape}") # Expected: [2, 1, 30, 30]
        print(f"Discriminator output shape (fake): {pred_fake.shape}") # Expected: [2, 1, 30, 30]
        print("GAN models sanity check passed!")
    except Exception as e:
        print(f"Error checking GAN models: {e}")
        return False

    # 4. Test Diffusion forward pass and single-step sampling
    try:
        diffusion_model = ConditionalUNet().to(device)
        ddpm = DDPM(T=200, device=device)
        
        t = torch.randint(0, 200, (2,), device=device).long()
        x_t, noise = ddpm.q_sample(clean_mag_dev, t)
        print(f"Diffusion forward q_sample shape: {x_t.shape}") # Expected: [2, 1, 256, 256]
        
        pred_noise = diffusion_model(x_t, noisy_mag_dev, t)
        print(f"Diffusion model noise prediction shape: {pred_noise.shape}") # Expected: [2, 1, 256, 256]
        
        # Test single reverse step sampling
        x_prev = ddpm.p_sample(diffusion_model, x_t, noisy_mag_dev, 100)
        print(f"Diffusion p_sample output shape: {x_prev.shape}") # Expected: [2, 1, 256, 256]
        print("Diffusion model sanity check passed!")
    except Exception as e:
        print(f"Error checking Diffusion models: {e}")
        return False

    # 5. Test Audio Reconstruction and Metrics (STOI/PESQ/SNR)
    try:
        clean_wf = reconstruct_waveform_from_spec(
            batch["clean_mag"][0], batch["noisy_phase"][0], batch["max_val"][0].item()
        )
        noisy_wf = reconstruct_waveform_from_spec(
            batch["noisy_mag"][0], batch["noisy_phase"][0], batch["max_val"][0].item()
        )
        print(f"Reconstructed clean waveform shape: {clean_wf.shape}")
        print(f"Reconstructed noisy waveform shape: {noisy_wf.shape}")
        
        snr = compute_snr(clean_wf, noisy_wf)
        print(f"Sanity check SNR calculation: {snr:.2f} dB")
        
        print("Audio reconstruction sanity check passed!")
    except Exception as e:
        print(f"Error checking audio reconstruction: {e}")
        return False

    # 6. Test Speaker Separation and Transcription Pipeline
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
            
        print("\n--- Running Speaker Separation and Transcription Pipeline Sanity Check ---")
        run_pipeline(DummyArgs())
        print("Speaker Separation and Transcription pipeline check passed!")
    except Exception as e:
        print(f"Error checking Speaker Separation and Transcription pipeline: {e}")
        return False

    print("=== ALL SANITY CHECKS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_sanity_check()
    sys.exit(0 if success else 1)
