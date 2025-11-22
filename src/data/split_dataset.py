import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def create_data_splits(features_dir='data/features', 
                       train_ratio=0.7, 
                       val_ratio=0.15, 
                       test_ratio=0.15,
                       random_state=42):
    """
    Create train/validation/test splits and save file lists
    """
    
    features_path = Path(features_dir)
    
    # Check if features directory exists
    if not features_path.exists():
        print(f"❌ Features directory not found: {features_path.absolute()}")
        print("\nPlease run: python run_preprocessing.py")
        return False
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    
    splits = {
        'train': {'files': [], 'labels': [], 'genres': []},
        'val': {'files': [], 'labels': [], 'genres': []},
        'test': {'files': [], 'labels': [], 'genres': []}
    }
    
    print("Creating data splits...")
    print(f"Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print(f"Features directory: {features_path.absolute()}\n")
    
    total_files = 0
    
    for genre in genres:
        genre_dir = features_path / genre
        
        if not genre_dir.exists():
            print(f"⚠️  {genre:12s}: directory not found, skipping")
            continue
        
        # Get all MFCC files (we'll use this as reference)
        mfcc_files = sorted(genre_dir.glob('*_mfcc.npy'))
        
        if not mfcc_files:
            print(f"⚠️  {genre:12s}: no files found, skipping")
            continue
        
        # Get base filenames (without _mfcc.npy)
        base_names = []
        for f in mfcc_files:
            # Remove _mfcc.npy to get base name
            base_name = f.stem.replace('_mfcc', '')
            base_names.append(base_name)
        
        labels = [genre_to_idx[genre]] * len(base_names)
        genres_list = [genre] * len(base_names)
        
        total_files += len(base_names)
        
        # If too few samples, put at least one in each split
        if len(base_names) < 10:
            print(f"⚠️  {genre:12s}: only {len(base_names)} files, may not split evenly")
            # Put all in train for now
            splits['train']['files'].extend(base_names)
            splits['train']['labels'].extend(labels)
            splits['train']['genres'].extend(genres_list)
            continue
        
        # Split: first into train and temp (val+test)
        train_names, temp_names, train_labels, temp_labels = train_test_split(
            base_names, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Split temp into val and test
        val_names, test_names, val_labels, test_labels = train_test_split(
            temp_names, temp_labels,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Add to splits
        splits['train']['files'].extend(train_names)
        splits['train']['labels'].extend(train_labels)
        splits['train']['genres'].extend([genre] * len(train_names))
        
        splits['val']['files'].extend(val_names)
        splits['val']['labels'].extend(val_labels)
        splits['val']['genres'].extend([genre] * len(val_names))
        
        splits['test']['files'].extend(test_names)
        splits['test']['labels'].extend(test_labels)
        splits['test']['genres'].extend([genre] * len(test_names))
        
        print(f"{genre:12s}: {len(base_names):3d} total | Train={len(train_names):3d}, Val={len(val_names):3d}, Test={len(test_names):3d}")
    
    # Check if we have any data
    if total_files == 0:
        print("\n❌ No files found! Please check your features directory.")
        return False
    
    # Save splits
    output_dir = Path('data/splits')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to: {output_dir.absolute()}")
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f'{split_name}.json'
        
        # Save without genres list (not needed for loading)
        save_data = {
            'files': split_data['files'],
            'labels': split_data['labels']
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"  ✓ {split_name:5s}.json: {len(split_data['files']):3d} samples")
    
    # Save genre mapping
    with open(output_dir / 'genre_mapping.json', 'w') as f:
        json.dump(genre_to_idx, f, indent=2)
    print(f"  ✓ genre_mapping.json")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {total_files}")
    print(f"Train: {len(splits['train']['files']):3d} ({100*len(splits['train']['files'])/total_files:.1f}%)")
    print(f"Val:   {len(splits['val']['files']):3d} ({100*len(splits['val']['files'])/total_files:.1f}%)")
    print(f"Test:  {len(splits['test']['files']):3d} ({100*len(splits['test']['files'])/total_files:.1f}%)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = create_data_splits()
    if not success:
        print("\n⚠️  Split creation failed. Check the error messages above.")
        exit(1)
    else:
        print("\n✓ Splits created successfully!")