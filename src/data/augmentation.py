"""
Audio data augmentation for training
"""

import numpy as np
import torch
import librosa

class AudioAugmentation:
    """Data augmentation for audio features"""
    
    def __init__(self, 
                 time_stretch_range=(0.9, 1.1),
                 pitch_shift_range=(-1, 1),
                 noise_factor=0.003,
                 apply_prob=0.3):
        """
        Args:
            time_stretch_range: Range for time stretching (min, max)
            pitch_shift_range: Range for pitch shifting in semitones
            noise_factor: Amount of gaussian noise to add
            apply_prob: Probability of applying each augmentation
        """
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_factor = noise_factor
        self.apply_prob = apply_prob
    
    def time_stretch(self, features):
        """Stretch features in time dimension"""
        if np.random.random() > self.apply_prob:
            return features
        
        rate = np.random.uniform(*self.time_stretch_range)
        
        # Interpolate along time axis
        original_length = features.shape[1]
        new_length = int(original_length * rate)
        
        # Use numpy interpolation
        time_old = np.linspace(0, 1, original_length)
        time_new = np.linspace(0, 1, new_length)
        
        stretched = np.zeros((features.shape[0], new_length))
        for i in range(features.shape[0]):
            stretched[i] = np.interp(time_new, time_old, features[i])
        
        # Resize back to original length
        if new_length < original_length:
            # Pad
            pad_size = original_length - new_length
            stretched = np.pad(stretched, ((0, 0), (0, pad_size)), mode='edge')
        else:
            # Crop
            stretched = stretched[:, :original_length]
        
        return stretched
    
    def add_noise(self, features):
        """Add Gaussian noise"""
        if np.random.random() > self.apply_prob:
            return features
        
        noise = np.random.randn(*features.shape) * self.noise_factor
        return features + noise
    
    def frequency_mask(self, features, max_mask_size=8):
        """Mask random frequency bins (SpecAugment style)"""
        if np.random.random() > self.apply_prob:
            return features
        
        n_freq_bins = features.shape[0]
        mask_size = np.random.randint(1, min(max_mask_size, n_freq_bins // 4))
        mask_start = np.random.randint(0, n_freq_bins - mask_size)
        
        features_masked = features.copy()
        features_masked[mask_start:mask_start + mask_size, :] = 0
        
        return features_masked
    
    def time_mask(self, features, max_mask_size=20):
        """Mask random time frames (SpecAugment style)"""
        if np.random.random() > self.apply_prob:
            return features
        
        n_time_frames = features.shape[1]
        mask_size = np.random.randint(1, min(max_mask_size, n_time_frames // 4))
        mask_start = np.random.randint(0, n_time_frames - mask_size)
        
        features_masked = features.copy()
        features_masked[:, mask_start:mask_start + mask_size] = 0
        
        return features_masked
    
    def __call__(self, features):
        """Apply random augmentations"""
        # Convert to numpy if tensor
        if torch.is_tensor(features):
            features = features.numpy()
        
        # Remove batch/channel dimensions if present
        original_shape = features.shape
        if len(features.shape) == 3:  # (1, freq, time)
            features = features[0]
        
        # Apply augmentations
        features = self.time_stretch(features)
        features = self.add_noise(features)
        features = self.frequency_mask(features)
        features = self.time_mask(features)
        
        # Restore original shape
        if len(original_shape) == 3:
            features = features[np.newaxis, ...]
        
        return torch.FloatTensor(features)


class AugmentedGTZANDataset(torch.utils.data.Dataset):
    """GTZAN dataset with augmentation"""
    
    def __init__(self, base_dataset, augmentation=None, use_augmentation=True):
        """
        Args:
            base_dataset: Original GTZANDataset
            augmentation: AudioAugmentation instance
            use_augmentation: Whether to apply augmentation (False for val/test)
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.use_augmentation = use_augmentation
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        features, label = self.base_dataset[idx]
        
        if self.use_augmentation and self.augmentation is not None:
            features = self.augmentation(features)
        
        return features, label