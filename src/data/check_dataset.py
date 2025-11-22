from pathlib import Path

def check_dataset_status():
    """Check if dataset exists and show structure"""
    
    data_dir = Path("data/raw//Data/genres_original")
    
    if not data_dir.exists():
        print("❌ Dataset directory doesn't exist!")
        print(f"   Looking for: {data_dir.absolute()}")
        return False
    
    print("📁 Dataset directory exists!")
    print(f"   Location: {data_dir.absolute()}\n")
    
    # Check for genre folders
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    found_genres = []
    for genre in genres:
        genre_path = data_dir / genre
        if genre_path.exists():
            wav_files = list(genre_path.glob("*.wav"))
            if wav_files:
                found_genres.append(genre)
                print(f"✓ {genre:12s} - {len(wav_files)} files")
            else:
                print(f"⚠ {genre:12s} - folder exists but empty")
        else:
            print(f"❌ {genre:12s} - folder missing")
    
    if found_genres:
        print(f"\n✓ Found {len(found_genres)} genres with audio files!")
        return True
    else:
        print("\n❌ No audio files found!")
        return False

if __name__ == "__main__":
    has_data = check_dataset_status()
    
    if not has_data:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Download dataset using one of these methods:")
        print("   Option A: Kaggle (recommended)")
        print("   Option B: Manual download")
        print("   Option C: Create test dataset")
        print("\n2. See the instructions I provided earlier!")