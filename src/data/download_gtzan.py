import os
import zipfile
from pathlib import Path
import subprocess

def download_gtzan_kaggle():
    """Download GTZAN dataset from Kaggle"""
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading GTZAN from Kaggle...")
    
    # Download using Kaggle API
    try:
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
            "-p", "data/raw",
            "--unzip"
        ], check=True)
        
        print("✓ Dataset downloaded successfully!")
        
        # The dataset structure might be different, let's check
        print("\nDataset structure:")
        for item in output_dir.rglob("*"):
            if item.is_dir():
                print(f"  📁 {item.relative_to(output_dir)}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
        return False
    
    return True

if __name__ == "__main__":
    download_gtzan_kaggle()