import shutil
from pathlib import Path

def reorganize_dataset():
    """Move dataset from downloaded location to correct structure"""
    
    # Your current location
    source_dir = Path("src/data/data/raw/Data/genres_original")
    
    # Where it should be
    target_dir = Path("data/raw/genres")
    
    print("Reorganizing dataset...")
    print(f"From: {source_dir.absolute()}")
    print(f"To:   {target_dir.absolute()}\n")
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print("\nLet me check where your files actually are...")
        
        # Search for the dataset
        base = Path(".")
        for path in base.rglob("blues.00000.wav"):
            print(f"   Found audio in: {path.parent.parent}")
            actual_source = path.parent.parent
            
            if input(f"\nIs this correct? (yes/no): ").lower().startswith('y'):
                source_dir = actual_source
                break
        else:
            print("Dataset not found!")
            return False
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    total_moved = 0
    
    for genre in genres:
        source_genre = source_dir / genre
        target_genre = target_dir / genre
        
        if source_genre.exists():
            # Create target genre folder
            target_genre.mkdir(exist_ok=True)
            
            # Copy all .wav files
            wav_files = list(source_genre.glob("*.wav"))
            
            for wav_file in wav_files:
                target_file = target_genre / wav_file.name
                if not target_file.exists():
                    shutil.copy2(wav_file, target_file)
                    total_moved += 1
            
            print(f"✓ {genre:12s} - copied {len(wav_files)} files")
        else:
            print(f"❌ {genre:12s} - not found")
    
    print(f"\n{'='*60}")
    print(f"✓ Reorganization complete!")
    print(f"{'='*60}")
    print(f"Total files moved: {total_moved}")
    print(f"New location: {target_dir.absolute()}")
    
    # Optional: Clean up old location
    if input("\nDelete old nested directories? (yes/no): ").lower().startswith('y'):
        try:
            shutil.rmtree("src/data/data")
            print("✓ Cleaned up old directories")
        except Exception as e:
            print(f"Could not delete: {e}")
    
    return True

if __name__ == "__main__":
    reorganize_dataset()