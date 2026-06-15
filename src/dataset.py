import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from src.utils import waveform_to_spectrogram, compress_magnitude

class VoiceBankDataset(Dataset):
    def __init__(self, split="train", crop_size=256, target_length=32768, limit=None):
        """
        Custom PyTorch Dataset for VoiceBank-DEMAND-16k.
        Args:
            split: "train" or "test"
            crop_size: target dimensions of spectrogram (e.g. 256)
            target_length: target number of samples in the 1D waveform (32768 = 2.048s)
            limit: optionally limit the dataset size for faster training/prototyping
        """
        self.split = split
        self.crop_size = crop_size
        self.target_length = target_length
        
        print(f"Loading JacobLinCool/VoiceBank-DEMAND-16k ({split} split)...")
        # Load dataset from Hugging Face
        self.dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split=split)
        
        if limit is not None:
            # Select a subset for fast training
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
            
        print(f"Loaded {len(self.dataset)} samples for split: {split}")

    def _pad_or_crop(self, wf):
        if len(wf) < self.target_length:
            pad_len = self.target_length - len(wf)
            wf = torch.nn.functional.pad(wf, (0, pad_len))
        elif len(wf) > self.target_length:
            if self.split == "train":
                # Random crop during training to augment data
                start = np.random.randint(0, len(wf) - self.target_length)
            else:
                # Fixed crop during testing for consistency
                start = 0
            wf = wf[start:start + self.target_length]
        return wf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load 1D audio arrays
        clean_wf = torch.tensor(item["clean"]["array"], dtype=torch.float32)
        noisy_wf = torch.tensor(item["noisy"]["array"], dtype=torch.float32)
        
        # Pad or crop waveforms to fixed length
        clean_wf = self._pad_or_crop(clean_wf)
        noisy_wf = self._pad_or_crop(noisy_wf)
        
        # Get spectrograms (magnitude and phase)
        clean_mag, _ = waveform_to_spectrogram(clean_wf)
        noisy_mag, noisy_phase = waveform_to_spectrogram(noisy_wf)
        
        # Use the maximum magnitude of the noisy spectrogram as the scaling factor
        max_val = noisy_mag.max().item()
        if max_val == 0:
            max_val = 1e-8
            
        # Normalize both magnitudes by the SAME scale factor
        clean_mag_norm = clean_mag / max_val
        noisy_mag_norm = noisy_mag / max_val
        
        # Crop frequency and time to crop_size (e.g., 256x256)
        clean_mag_crop = clean_mag_norm[:self.crop_size, :self.crop_size]
        noisy_mag_crop = noisy_mag_norm[:self.crop_size, :self.crop_size]
        noisy_phase_crop = noisy_phase[:self.crop_size, :self.crop_size]
        
        # Apply log compression to make training stable
        clean_log_mag = compress_magnitude(clean_mag_crop)
        noisy_log_mag = compress_magnitude(noisy_mag_crop)
        
        # Add channel dimension: [1, crop_size, crop_size]
        clean_log_mag = clean_log_mag.unsqueeze(0)
        noisy_log_mag = noisy_log_mag.unsqueeze(0)
        
        return {
            "clean_mag": clean_log_mag,     # [1, F, T]
            "noisy_mag": noisy_log_mag,     # [1, F, T]
            "noisy_phase": noisy_phase_crop, # [F, T]
            "max_val": max_val,
            "id": item["id"]
        }

def get_dataloaders(batch_size=16, crop_size=256, target_length=32768, train_limit=None, test_limit=None):
    """
    Creates PyTorch DataLoaders for training and testing.
    """
    train_dataset = VoiceBankDataset(split="train", crop_size=crop_size, target_length=target_length, limit=train_limit)
    test_dataset = VoiceBankDataset(split="test", crop_size=crop_size, target_length=target_length, limit=test_limit)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    return train_loader, test_loader
