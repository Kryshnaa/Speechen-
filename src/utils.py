import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt

def waveform_to_spectrogram(waveform, n_fft=512, hop_length=128, win_length=512):
    """
    Converts a 1D audio waveform into 2D magnitude and phase spectrograms.
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)  # Squeeze channel dim to make it 1D
        
    window = torch.hann_window(win_length).to(waveform.device)
    stft_res = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True
    )
    
    magnitude = torch.abs(stft_res)
    phase = torch.angle(stft_res)
    return magnitude, phase

def spectrogram_to_waveform(magnitude, phase, n_fft=512, hop_length=128, win_length=512):
    """
    Converts 2D magnitude and phase spectrograms back to a 1D audio waveform.
    """
    complex_spec = magnitude * torch.exp(1j * phase)
    window = torch.hann_window(win_length).to(complex_spec.device)
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )
    return waveform

def compress_magnitude(magnitude, C=10.0):
    """
    Applies log compression to map magnitude range to approximately [0, 1].
    """
    return torch.log1p(C * magnitude) / np.log(1.0 + C)

def decompress_magnitude(compressed_magnitude, C=10.0):
    """
    Reverses the log compression.
    """
    return (torch.exp(compressed_magnitude * np.log(1.0 + C)) - 1.0) / C

def prepare_spec_for_model(waveform, n_fft=512, hop_length=128, win_length=512, crop_size=256):
    """
    Extracts magnitude and phase, crops the frequency and time to crop_size,
    and log-compresses the magnitude.
    Returns:
        log_mag: compressed magnitude of shape [1, crop_size, crop_size] (batch/channel dim added)
        phase: cropped phase of shape [crop_size, crop_size]
        max_val: the original maximum value of the magnitude spectrogram (for unscaling)
    """
    magnitude, phase = waveform_to_spectrogram(waveform, n_fft, hop_length, win_length)
    
    # Scale by maximum magnitude to normalize range
    max_val = magnitude.max().item()
    if max_val == 0:
        max_val = 1e-8
    magnitude = magnitude / max_val
    
    # Crop to crop_size x crop_size
    mag_cropped = magnitude[:crop_size, :crop_size]
    phase_cropped = phase[:crop_size, :crop_size]
    
    # Log compress
    log_mag = compress_magnitude(mag_cropped)
    
    # Add channel dimension
    log_mag = log_mag.unsqueeze(0)
    
    return log_mag, phase_cropped, max_val

def reconstruct_waveform_from_spec(log_mag, phase, max_val, n_fft=512, hop_length=128, win_length=512, target_freq=257, target_time=257):
    """
    Reverses log-compression, pads the spectrogram back to original STFT dimensions,
    and runs ISTFT to reconstruct the 1D waveform.
    """
    # Remove channel dim
    if log_mag.ndim == 3:
        log_mag = log_mag.squeeze(0)
        
    # Decompress
    magnitude = decompress_magnitude(log_mag)
    
    # Unscale
    magnitude = magnitude * max_val
    
    # Ensure phase is on the same device as magnitude
    phase = phase.to(magnitude.device)
    
    # Pad back to target_freq and target_time
    F_curr, T_curr = magnitude.shape
    pad_f = target_freq - F_curr
    pad_t = target_time - T_curr
    
    magnitude_padded = torch.nn.functional.pad(magnitude, (0, pad_t, 0, pad_f), mode='constant', value=0.0)
    phase_padded = torch.nn.functional.pad(phase, (0, pad_t, 0, pad_f), mode='constant', value=0.0)
    
    # Inverse STFT
    waveform = spectrogram_to_waveform(magnitude_padded, phase_padded, n_fft, hop_length, win_length)
    return waveform

def compute_snr(clean_wf, enhanced_wf):
    """
    Computes Signal-to-Noise Ratio (SNR) in dB.
    """
    if isinstance(clean_wf, torch.Tensor):
        clean_wf = clean_wf.numpy()
    if isinstance(enhanced_wf, torch.Tensor):
        enhanced_wf = enhanced_wf.numpy()
        
    # Ensure shapes match
    min_len = min(len(clean_wf), len(enhanced_wf))
    clean_wf = clean_wf[:min_len]
    enhanced_wf = enhanced_wf[:min_len]
    
    noise = clean_wf - enhanced_wf
    signal_power = np.sum(clean_wf ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    return 10 * np.log10(signal_power / noise_power)

def compute_stoi_pesq(clean_path, enhanced_path):
    """
    Computes STOI and PESQ metrics using external packages.
    """
    import soundfile as sf
    from pystoi import stoi
    from pesq import pesq
    
    clean, sr1 = sf.read(clean_path)
    enhanced, sr2 = sf.read(enhanced_path)
    
    # Ensure they are at 16kHz
    assert sr1 == 16000 and sr2 == 16000, "Audio must be at 16kHz"
    
    # Align lengths
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # STOI
    stoi_score = stoi(clean, enhanced, sr1, extended=False)
    
    # PESQ (narrowband, 16000Hz sampling rate)
    try:
        pesq_score = pesq(sr1, clean, enhanced, 'wb') # Wideband
    except Exception as e:
        print(f"Error computing wideband PESQ, falling back to narrowband: {e}")
        try:
            pesq_score = pesq(sr1, clean, enhanced, 'nb') # Narrowband
        except Exception:
            pesq_score = 0.0
            
    return stoi_score, pesq_score

def save_comparison_plot(clean_mag, noisy_mag, enhanced_mag, save_path):
    """
    Plots and saves side-by-side log-magnitude spectrogram comparison.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_mag.cpu().numpy().squeeze(), origin='lower', aspect='auto', cmap='inferno')
    plt.title("Noisy Spectrogram")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_mag.cpu().numpy().squeeze(), origin='lower', aspect='auto', cmap='inferno')
    plt.title("Enhanced Spectrogram")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(clean_mag.cpu().numpy().squeeze(), origin='lower', aspect='auto', cmap='inferno')
    plt.title("Clean Spectrogram (Target)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
