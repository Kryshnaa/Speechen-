import os
import argparse
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchaudio
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from transformers import pipeline

# Suppress warnings for clean output
import warnings
warnings.filterwarnings("ignore")

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_audio(path, target_sr=16000):
    """
    Loads an audio file, resamples to target_sr, and converts to mono if needed.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, target_sr

def mix_waveforms(wf1, wf2):
    """
    Aligns and mixes two waveforms.
    """
    min_len = min(wf1.shape[1], wf2.shape[1])
    wf1_cut = wf1[:, :min_len]
    wf2_cut = wf2[:, :min_len]
    
    mixed = wf1_cut + wf2_cut
    max_val = mixed.abs().max()
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed

def compute_spectrogram(waveform, n_fft=512, hop_length=128):
    """
    Helper to compute log-magnitude spectrogram for visualization.
    """
    window = torch.hann_window(n_fft).to(waveform.device)
    stft_res = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True
    )
    magnitude = torch.abs(stft_res)
    # Log compression
    log_mag = torch.log1p(10.0 * magnitude) / np.log(11.0)
    return log_mag.cpu().numpy()

def plot_separation_results(mixed_wf, separated_wfs, sr, save_path):
    """
    Generates and saves a rich visualization of the mixture and separated sources.
    """
    num_sources = len(separated_wfs)
    fig, axes = plt.subplots(2, num_sources + 1, figsize=(5 * (num_sources + 1), 7))
    
    # 1. Waveforms
    # Mixed
    time_axis = np.linspace(0, mixed_wf.shape[1] / sr, num=mixed_wf.shape[1])
    axes[0, 0].plot(time_axis, mixed_wf.squeeze(0).cpu().numpy(), color='#7f8c8d', alpha=0.8)
    axes[0, 0].set_title("Mixed Audio Waveform", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Separated Waveforms
    colors = ['#1abc9c', '#3498db', '#e74c3c', '#9b59b6']
    for idx, sep_wf in enumerate(separated_wfs):
        color = colors[idx % len(colors)]
        axes[0, idx + 1].plot(time_axis, sep_wf.squeeze(0).cpu().numpy(), color=color, alpha=0.8)
        axes[0, idx + 1].set_title(f"Speaker {idx + 1} Waveform", fontsize=12, fontweight='bold')
        axes[0, idx + 1].set_xlabel("Time (s)")
        axes[0, idx + 1].grid(True, linestyle='--', alpha=0.5)
        
    # 2. Spectrograms
    # Mixed Spectrogram
    mixed_spec = compute_spectrogram(mixed_wf)
    im0 = axes[1, 0].imshow(mixed_spec, origin='lower', aspect='auto', cmap='magma')
    axes[1, 0].set_title("Mixed Spectrogram", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Time Frames")
    axes[1, 0].set_ylabel("Frequency Bins")
    fig.colorbar(im0, ax=axes[1, 0], format='%+2.0f')
    
    # Separated Spectrograms
    for idx, sep_wf in enumerate(separated_wfs):
        sep_spec = compute_spectrogram(sep_wf)
        im = axes[1, idx + 1].imshow(sep_spec, origin='lower', aspect='auto', cmap='magma')
        axes[1, idx + 1].set_title(f"Speaker {idx + 1} Spectrogram", fontsize=12, fontweight='bold')
        axes[1, idx + 1].set_xlabel("Time Frames")
        fig.colorbar(im, ax=axes[1, idx + 1], format='%+2.0f')
        
    plt.suptitle("Multi-Speaker Audio Separation Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved successfully to: {save_path}")

def run_pipeline(args):
    device = get_device()
    print(f"Using execution device: {device}")
    
    # Step 1: Handle input audio (mix or load existing)
    if args.mix:
        if not args.audio_1 or not args.audio_2:
            raise ValueError("To create a synthetic mixture, you must provide paths for both --audio_1 and --audio_2")
        print(f"Mixing speaker audio files:\n  1: {args.audio_1}\n  2: {args.audio_2}")
        wf1, sr1 = load_audio(args.audio_1)
        wf2, sr2 = load_audio(args.audio_2)
        mixed_wf = mix_waveforms(wf1, wf2)
        
        # Save mixture
        os.makedirs(args.output_dir, exist_ok=True)
        mixed_path = os.path.join(args.output_dir, "synthetic_mixture.wav")
        torchaudio.save(mixed_path, mixed_wf, sr1)
        print(f"Synthetic mixture saved to: {mixed_path}")
    else:
        if not args.input_audio:
            raise ValueError("You must specify either --input_audio or use --mix with --audio_1 and --audio_2")
        print(f"Loading input mixed audio from: {args.input_audio}")
        mixed_wf, sr = load_audio(args.input_audio)
        mixed_path = args.input_audio
        
    # Step 2: Separate speech sources
    print(f"Loading speech separation model: {args.sep_model}...")
    # pyrefly: ignore [missing-import]
    from speechbrain.inference.separation import SepformerSeparation
    
    # SpeechBrain has a bug with 'mps' device where self.device_type is not set.
    # We fallback to CPU for SpeechBrain if MPS is active.
    sb_device = "cpu" if device == "mps" else device
    run_opts = {"device": sb_device}
    separator = SepformerSeparation.from_hparams(
        source=args.sep_model,
        savedir=os.path.join("pretrained_models", args.sep_model.replace("/", "_")),
        run_opts=run_opts
    )
    
    print("Performing speaker separation (source separation)...")
    # SepformerSeparation.separate_file expects a file path
    # returns estimated sources of shape [batch, time, num_sources]
    est_sources = separator.separate_file(path=mixed_path)
    
    # Rescale or format output
    # est_sources shape: [1, time, 2]
    # Squeeze batch dimension: [time, 2]
    est_sources = est_sources.squeeze(0).cpu()
    num_sources = est_sources.shape[1]
    
    separated_paths = []
    separated_wfs = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx in range(num_sources):
        # Transpose to [channels, time] -> [1, time] for torchaudio
        sep_wf = est_sources[:, idx:idx+1].t()
        
        # Normalize waveform
        max_val = sep_wf.abs().max()
        if max_val > 0:
            sep_wf = sep_wf / max_val
            
        sep_path = os.path.join(args.output_dir, f"separated_speaker_{idx+1}.wav")
        # Save at separation model sample rate (usually 16k or 8k, SepFormer is 16k/8k)
        torchaudio.save(sep_path, sep_wf, 16000)
        
        separated_paths.append(sep_path)
        separated_wfs.append(sep_wf)
        print(f"Saved separated Speaker {idx+1} to: {sep_path}")
        
    # Step 3: Visualize Separated Waveforms & Spectrograms
    plot_path = os.path.join(args.output_dir, "separation_analysis.png")
    plot_separation_results(mixed_wf, separated_wfs, 16000, plot_path)
    
    # Step 4: Perform Speech-to-Text (Transcription) using Whisper
    print(f"\nLoading Automatic Speech Recognition (ASR) model: {args.asr_model}...")
    
    # device parameter in pipeline: device="cuda" or device="mps" or device="cpu"
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=device
    )
    
    print("\n" + "=" * 60)
    print(" TRANSCRIPTION RESULTS")
    print("=" * 60)
    
    transcripts = {}
    for idx, sep_path in enumerate(separated_paths):
        print(f"Transcribing Speaker {idx+1}...")
        asr_result = asr(sep_path)
        transcript_text = asr_result.get("text", "").strip()
        transcripts[f"Speaker {idx+1}"] = transcript_text
        print(f"\033[1mSpeaker {idx+1}:\033[0m {transcript_text}")
        
    print("=" * 60)
    
    # Save transcription text to file
    txt_path = os.path.join(args.output_dir, "transcripts.txt")
    with open(txt_path, "w") as f:
        f.write("=== TRANSCRIPTION RESULTS ===\n\n")
        for key, text in transcripts.items():
            f.write(f"{key}:\n{text}\n\n")
    print(f"Transcripts saved to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Speaker Separation and Transcription Pipeline")
    parser.add_argument("--mix", action="store_true", help="Mix two audio files synthetically for testing")
    parser.add_argument("--audio_1", type=str, help="Clean speech file for speaker 1 (used with --mix)")
    parser.add_argument("--audio_2", type=str, help="Clean speech file for speaker 2 (used with --mix)")
    parser.add_argument("--input_audio", type=str, help="Path to pre-mixed audio file to separate directly")
    parser.add_argument("--sep_model", type=str, default="speechbrain/sepformer-whamr16k", 
                        help="SpeechBrain separation model name")
    parser.add_argument("--asr_model", type=str, default="openai/whisper-tiny", 
                        help="HuggingFace model for transcription")
    parser.add_argument("--output_dir", type=str, default="./test_samples/separation", 
                        help="Output directory to save results")
    
    args = parser.parse_args()
    run_pipeline(args)
