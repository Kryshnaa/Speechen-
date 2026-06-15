import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

from src.dataset import VoiceBankDataset
from src.models.gan import Generator
from src.models.diffusion import ConditionalUNet, DDPM
from src.utils import reconstruct_waveform_from_spec, compute_snr, compute_stoi_pesq

def enhance_and_evaluate(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load test dataset
    print(f"Loading {args.num_samples} test samples from VoiceBank-DEMAND...")
    test_dataset = VoiceBankDataset(split="test", crop_size=args.crop_size, limit=args.num_samples)

    # Load GAN Generator
    generator = Generator().to(device)
    gan_loaded = False
    if args.gan_checkpoint and os.path.exists(args.gan_checkpoint):
        try:
            checkpoint = torch.load(args.gan_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded GAN generator checkpoint from {args.gan_checkpoint}")
            gan_loaded = True
        except Exception as e:
            print(f"Error loading GAN checkpoint: {e}. Using random initialization.")
    else:
        print("No valid GAN checkpoint provided. Running with random initialization for pipeline verification.")

    # Load Diffusion Model
    diffusion_model = ConditionalUNet().to(device)
    ddpm = DDPM(T=args.timesteps, device=device)
    diffusion_loaded = False
    if args.diffusion_checkpoint and os.path.exists(args.diffusion_checkpoint):
        try:
            checkpoint = torch.load(args.diffusion_checkpoint, map_location=device)
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Diffusion model checkpoint from {args.diffusion_checkpoint}")
            diffusion_loaded = True
        except Exception as e:
            print(f"Error loading Diffusion checkpoint: {e}. Using random initialization.")
    else:
        print("No valid Diffusion checkpoint provided. Running with random initialization for pipeline verification.")

    generator.eval()
    diffusion_model.eval()

    # To store metrics
    results = []

    print("\nProcessing enhancement and evaluation...")
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        sample_id = sample["id"]
        print(f"\n--- Processing sample {idx+1}/{len(test_dataset)}: {sample_id} ---")
        
        # Prepare inputs
        noisy_mag = sample["noisy_mag"].unsqueeze(0).to(device) # [1, 1, F, T]
        clean_mag = sample["clean_mag"].unsqueeze(0).to(device) # [1, 1, F, T]
        noisy_phase = sample["noisy_phase"]                     # [F, T]
        max_val = sample["max_val"]
        
        # 1. GAN Enhancement
        with torch.no_grad():
            gan_mag_pred = generator(noisy_mag).squeeze(0) # [1, F, T]
            
        # 2. Diffusion Enhancement
        with torch.no_grad():
            diffusion_mag_pred = ddpm.sample_enhanced(diffusion_model, noisy_mag).squeeze(0) # [1, F, T]
            
        # Reconstruct audio waveforms
        # Clean Target
        clean_wf = reconstruct_waveform_from_spec(
            clean_mag.squeeze(0), noisy_phase, max_val
        )
        # Noisy Input
        noisy_wf = reconstruct_waveform_from_spec(
            noisy_mag.squeeze(0), noisy_phase, max_val
        )
        # GAN Denoised
        gan_wf = reconstruct_waveform_from_spec(
            gan_mag_pred, noisy_phase, max_val
        )
        # Diffusion Denoised
        diffusion_wf = reconstruct_waveform_from_spec(
            diffusion_mag_pred, noisy_phase, max_val
        )

        # Move to CPU for saving and metrics
        clean_wf_np = clean_wf.cpu().numpy()
        noisy_wf_np = noisy_wf.cpu().numpy()
        gan_wf_np = gan_wf.cpu().numpy()
        diffusion_wf_np = diffusion_wf.cpu().numpy()

        # Save Audio Files
        noisy_path = os.path.join(args.output_dir, f"{sample_id}_noisy.wav")
        clean_path = os.path.join(args.output_dir, f"{sample_id}_clean.wav")
        gan_path = os.path.join(args.output_dir, f"{sample_id}_gan.wav")
        diffusion_path = os.path.join(args.output_dir, f"{sample_id}_diffusion.wav")
        
        sf.write(noisy_path, noisy_wf_np, 16000)
        sf.write(clean_path, clean_wf_np, 16000)
        sf.write(gan_path, gan_wf_np, 16000)
        sf.write(diffusion_path, diffusion_wf_np, 16000)
        
        print(f"Saved audio files to {args.output_dir}")

        # Compute SNR Metrics
        snr_noisy = compute_snr(clean_wf_np, noisy_wf_np)
        snr_gan = compute_snr(clean_wf_np, gan_wf_np)
        snr_diff = compute_snr(clean_wf_np, diffusion_wf_np)

        # Compute STOI and PESQ Metrics
        stoi_noisy, pesq_noisy = compute_stoi_pesq(clean_path, noisy_path)
        stoi_gan, pesq_gan = compute_stoi_pesq(clean_path, gan_path)
        stoi_diff, pesq_diff = compute_stoi_pesq(clean_path, diffusion_path)

        print(f"Metrics - Noisy:    SNR={snr_noisy:.2f}dB, STOI={stoi_noisy:.3f}, PESQ={pesq_noisy:.3f}")
        print(f"Metrics - GAN:      SNR={snr_gan:.2f}dB, STOI={stoi_gan:.3f}, PESQ={pesq_gan:.3f}")
        print(f"Metrics - Diffusion: SNR={snr_diff:.2f}dB, STOI={stoi_diff:.3f}, PESQ={pesq_diff:.3f}")

        # Save Spectrogram Plot Comparison
        plt.figure(figsize=(18, 4))
        
        plt.subplot(1, 4, 1)
        plt.imshow(noisy_mag.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='inferno')
        plt.title(f"Noisy (PESQ: {pesq_noisy:.2f})")
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(1, 4, 2)
        plt.imshow(gan_mag_pred.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='inferno')
        plt.title(f"GAN Enhanced (PESQ: {pesq_gan:.2f})")
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(1, 4, 3)
        plt.imshow(diffusion_mag_pred.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='inferno')
        plt.title(f"Diffusion Enhanced (PESQ: {pesq_diff:.2f})")
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(1, 4, 4)
        plt.imshow(clean_mag.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='inferno')
        plt.title("Clean Target (Reference)")
        plt.colorbar(format='%+2.0f dB')
        
        plt.suptitle(f"Spectrogram Denoising Comparison (Sample: {sample_id})")
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"{sample_id}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved comparison plot to {plot_path}")

        results.append({
            "id": sample_id,
            "snr_noisy": snr_noisy, "snr_gan": snr_gan, "snr_diff": snr_diff,
            "stoi_noisy": stoi_noisy, "stoi_gan": stoi_gan, "stoi_diff": stoi_diff,
            "pesq_noisy": pesq_noisy, "pesq_gan": pesq_gan, "pesq_diff": pesq_diff
        })

    # Print Final Summary Table
    print("\n" + "="*80)
    print(" EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Sample ID':<15} | {'Metric':<6} | {'Noisy Input':<12} | {'GAN Enhanced':<12} | {'Diffusion Enhanced':<18}")
    print("-"*80)
    
    avg_snr_noisy, avg_snr_gan, avg_snr_diff = 0, 0, 0
    avg_stoi_noisy, avg_stoi_gan, avg_stoi_diff = 0, 0, 0
    avg_pesq_noisy, avg_pesq_gan, avg_pesq_diff = 0, 0, 0

    for r in results:
        print(f"{r['id']:<15} | {'SNR':<6} | {r['snr_noisy']:<12.2f} | {r['snr_gan']:<12.2f} | {r['snr_diff']:<18.2f}")
        print(f"{'':<15} | {'STOI':<6} | {r['stoi_noisy']:<12.3f} | {r['stoi_gan']:<12.3f} | {r['stoi_diff']:<18.3f}")
        print(f"{'':<15} | {'PESQ':<6} | {r['pesq_noisy']:<12.3f} | {r['pesq_gan']:<12.3f} | {r['pesq_diff']:<18.3f}")
        print("-"*80)
        
        avg_snr_noisy += r['snr_noisy']
        avg_snr_gan += r['snr_gan']
        avg_snr_diff += r['snr_diff']
        avg_stoi_noisy += r['stoi_noisy']
        avg_stoi_gan += r['stoi_gan']
        avg_stoi_diff += r['stoi_diff']
        avg_pesq_noisy += r['pesq_noisy']
        avg_pesq_gan += r['pesq_gan']
        avg_pesq_diff += r['pesq_diff']

    n = len(results)
    print(f"{'AVERAGE':<15} | {'SNR':<6} | {avg_snr_noisy/n:<12.2f} | {avg_snr_gan/n:<12.2f} | {avg_snr_diff/n:<18.2f}")
    print(f"{'':<15} | {'STOI':<6} | {avg_stoi_noisy/n:<12.3f} | {avg_stoi_gan/n:<12.3f} | {avg_stoi_diff/n:<18.3f}")
    print(f"{'':<15} | {'PESQ':<6} | {avg_pesq_noisy/n:<12.3f} | {avg_pesq_gan/n:<12.3f} | {avg_pesq_diff/n:<18.3f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Speech Enhancement Models")
    parser.add_argument("--gan_checkpoint", type=str, default="", help="Path to trained GAN checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, default="", help="Path to trained Diffusion checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples to evaluate")
    parser.add_argument("--crop_size", type=int, default=256, help="Spectrogram crop size")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion timesteps (T)")
    parser.add_argument("--output_dir", type=str, default="./test_samples", help="Directory to save comparison audio and plots")
    
    args = parser.parse_args()
    enhance_and_evaluate(args)
