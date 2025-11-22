import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

class GTZANDataset(Dataset):
    """GTZAN Dataset with pre-extracted features"""
    
    def __init__(self, split='train', feature_type='mfcc', 
                 features_dir='data/features', 
                 splits_dir='data/splits',
                 transform=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            feature_type: 'mfcc', 'melspec', or 'dwt'
            features_dir: Directory containing extracted features
            splits_dir: Directory containing split files
            transform: Optional transform to apply
        """
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        self.transform = transform
        
        # Load split
        split_file = Path(splits_dir) / f'{split}.json'
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.file_names = split_data['files']
        self.labels = split_data['labels']
        
        # Load genre mapping
        with open(Path(splits_dir) / 'genre_mapping.json', 'r') as f:
            self.genre_to_idx = json.load(f)
        
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
        
        print(f"Loaded {split} set: {len(self.file_names)} samples ({feature_type})")
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # Get file info
        file_name = self.file_names[idx]
        label = self.labels[idx]
        
        # Determine genre from label
        genre = self.idx_to_genre[label]
        
        # Load feature
        feature_file = self.features_dir / genre / f'{file_name}_{self.feature_type}.npy'
        feature = np.load(feature_file)
        
        # Convert to tensor and add channel dimension
        feature = torch.FloatTensor(feature).unsqueeze(0)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

def get_augmented_data_loaders(feature_type='mfcc', batch_size=64, num_workers=8):
    """Create data loaders with augmentation"""
    from src.data.augmentation import AudioAugmentation, AugmentedGTZANDataset
    
    # Create base datasets
    train_dataset = GTZANDataset(split='train', feature_type=feature_type)
    val_dataset = GTZANDataset(split='val', feature_type=feature_type)
    test_dataset = GTZANDataset(split='test', feature_type=feature_type)
    
    # Create augmentation
    augmentation = AudioAugmentation(
        time_stretch_range=(0.8, 1.2),
        pitch_shift_range=(-2, 2),
        noise_factor=0.005,
        apply_prob=0.5
    )
    
    # Wrap with augmentation (only for training!)
    train_dataset_aug = AugmentedGTZANDataset(
        train_dataset, 
        augmentation=augmentation, 
        use_augmentation=True
    )
    
    val_dataset_aug = AugmentedGTZANDataset(
        val_dataset, 
        augmentation=None, 
        use_augmentation=False
    )
    
    test_dataset_aug = AugmentedGTZANDataset(
        test_dataset, 
        augmentation=None, 
        use_augmentation=False
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset_aug,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset_aug,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader

def get_data_loaders(feature_type='mfcc', batch_size=64, num_workers=8):  # ✅ Increased defaults
    """Create train, val, and test data loaders - Optimized for M4"""
    
    train_dataset = GTZANDataset(split='train', feature_type=feature_type)
    val_dataset = GTZANDataset(split='val', feature_type=feature_type)
    test_dataset = GTZANDataset(split='test', feature_type=feature_type)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # ✅ Disabled for MPS
        persistent_workers=True if num_workers > 0 else False,  # ✅ Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None  # ✅ Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # ✅ Larger for validation (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # ✅ Larger for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset...\n")
    
    for feature_type in ['mfcc', 'melspec', 'dwt']:
        print(f"\nTesting {feature_type}...")
        train_loader, val_loader, test_loader = get_data_loaders(
            feature_type=feature_type, 
            batch_size=4,
            num_workers=0
        )
        
        # Get a batch
        features, labels = next(iter(train_loader))
        print(f"  Batch features shape: {features.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    
    print("\n✓ Dataset working correctly!")