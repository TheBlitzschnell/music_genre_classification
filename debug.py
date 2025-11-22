from pathlib import Path
import json

print("Debugging dataset issues...\n")

# Check 1: Features directory
print("="*60)
print("1. Checking features directory...")
print("="*60)
features_dir = Path('data/features')
if features_dir.exists():
    print(f"✓ Features directory exists: {features_dir.absolute()}")
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        genre_dir = features_dir / genre
        if genre_dir.exists():
            mfcc_files = list(genre_dir.glob('*_mfcc.npy'))
            print(f"  {genre:12s}: {len(mfcc_files)} files")
        else:
            print(f"  {genre:12s}: ❌ directory missing")
else:
    print(f"❌ Features directory not found: {features_dir.absolute()}")

# Check 2: Splits directory
print("\n" + "="*60)
print("2. Checking splits directory...")
print("="*60)
splits_dir = Path('data/splits')
if splits_dir.exists():
    print(f"✓ Splits directory exists: {splits_dir.absolute()}")
    
    for split in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split}.json'
        if split_file.exists():
            with open(split_file, 'r') as f:
                data = json.load(f)
            print(f"  {split:6s}.json: {len(data['files'])} samples")
        else:
            print(f"  {split:6s}.json: ❌ file missing")
    
    # Check genre mapping
    genre_file = splits_dir / 'genre_mapping.json'
    if genre_file.exists():
        with open(genre_file, 'r') as f:
            mapping = json.load(f)
        print(f"\n  Genre mapping: {list(mapping.keys())}")
    else:
        print(f"\n  ❌ genre_mapping.json missing")
else:
    print(f"❌ Splits directory not found: {splits_dir.absolute()}")

# Check 3: Try loading a sample file
print("\n" + "="*60)
print("3. Testing sample file loading...")
print("="*60)
try:
    import numpy as np
    sample_files = list(Path('data/features/blues').glob('*_mfcc.npy'))
    if sample_files:
        sample = np.load(sample_files[0])
        print(f"✓ Successfully loaded sample: {sample_files[0].name}")
        print(f"  Shape: {sample.shape}")
    else:
        print("❌ No sample files found")
except Exception as e:
    print(f"❌ Error loading sample: {e}")