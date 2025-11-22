#!/usr/bin/env python3
"""
Run this from project root: python run_preprocessing.py
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import pywt

# Ensure we can import from src
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13, n_mels=128, target_length=130):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.target_length = target_length
    
    def extract_mfcc(self, audio_path):
        """Extract MFCC features"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Resize to target length
        if mfcc.shape[1] < self.target_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.target_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :self.target_length]
        
        return mfcc
    
    def extract_melspectrogram(self, audio_path):
        """Extract Mel-spectrogram"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target length
        if mel_spec_db.shape[1] < self.target_length:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, self.target_length - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :self.target_length]
        
        return mel_spec_db
    
    def extract_dwt(self, audio_path, wavelet='db4', level=4):
        """Extract DWT coefficients"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        # Stack coefficients
        dwt_features = []
        for coeff in coeffs:
            # Resize each level
            if len(coeff) < self.target_length:
                coeff = np.pad(coeff, (0, self.target_length - len(coeff)), mode='constant')
            else:
                coeff = coeff[:self.target_length]
            dwt_features.append(coeff)
        
        dwt_features = np.array(dwt_features)
        
        # If we have fewer than 256 features, pad
        if dwt_features.shape[0] < 256:
            pad_size = 256 - dwt_features.shape[0]
            dwt_features = np.pad(dwt_features, ((0, pad_size), (0, 0)), mode='constant')
        else:
            dwt_features = dwt_features[:256, :]
        
        return dwt_features
    
    def extract_all_features(self, audio_path):
        """Extract all three feature types"""
        return {
            'mfcc': self.extract_mfcc(audio_path),
            'melspec': self.extract_melspectrogram(audio_path),
            'dwt': self.extract_dwt(audio_path)
        }

def preprocess_gtzan_dataset(raw_dir='data/raw/genres', output_dir='data/features'):
    """Extract features from all GTZAN audio files"""
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # Check if dataset exists
    if not raw_path.exists():
        print(f"❌ Dataset not found at: {raw_path.absolute()}")
        print("\nPlease ensure your dataset is at: data/raw/genres/")
        print("Each genre should have its own subdirectory with .wav files")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor()
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    total_files = 0
    total_processed = 0
    total_errors = 0
    
    print("="*60)
    print("GTZAN FEATURE EXTRACTION")
    print("="*60)
    print(f"Input:  {raw_path.absolute()}")
    print(f"Output: {output_path.absolute()}")
    print("="*60 + "\n")
    
    for genre in genres:
        genre_dir = raw_path / genre
        
        if not genre_dir.exists():
            print(f"⚠️  {genre:12s} - folder not found, skipping")
            continue
        
        output_genre_dir = output_path / genre
        output_genre_dir.mkdir(exist_ok=True)
        
        audio_files = list(genre_dir.glob('*.wav'))
        total_files += len(audio_files)
        
        if not audio_files:
            print(f"⚠️  {genre:12s} - no .wav files found")
            continue
        
        print(f"🎵 Processing {genre:12s} ({len(audio_files)} files)...")
        
        genre_errors = 0
        for audio_file in tqdm(audio_files, desc=f"   {genre}", leave=False):
            try:
                # Extract all features
                features = extractor.extract_all_features(str(audio_file))
                
                # Save each feature type
                base_name = audio_file.stem
                np.save(output_genre_dir / f'{base_name}_mfcc.npy', features['mfcc'])
                np.save(output_genre_dir / f'{base_name}_melspec.npy', features['melspec'])
                np.save(output_genre_dir / f'{base_name}_dwt.npy', features['dwt'])
                
                total_processed += 1
                
            except Exception as e:
                genre_errors += 1
                total_errors += 1
                print(f"\n   ❌ Error: {audio_file.name} - {e}")
        
        if genre_errors == 0:
            print(f"   ✓ {genre:12s} - {len(audio_files)} files processed")
        else:
            print(f"   ⚠️  {genre:12s} - {len(audio_files)-genre_errors}/{len(audio_files)} processed")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files found:      {total_files}")
    print(f"Successfully processed: {total_processed}")
    print(f"Errors:                 {total_errors}")
    if total_files > 0:
        print(f"Success rate:           {100*total_processed/total_files:.1f}%")
    print("="*60)
    
    if total_errors == 0 and total_processed > 0:
        print("\n✓ Feature extraction complete!")
    elif total_processed > 0:
        print(f"\n⚠️  Feature extraction complete with {total_errors} errors")
    else:
        print("\n❌ No features were extracted!")
    
    print(f"\nFeatures saved to: {output_path.absolute()}")
    
    # Verify output
    print("\nVerifying output...")
    for genre in genres:
        genre_path = output_path / genre
        if genre_path.exists():
            mfcc_count = len(list(genre_path.glob('*_mfcc.npy')))
            if mfcc_count > 0:
                print(f"  ✓ {genre:12s}: {mfcc_count} files")

if __name__ == "__main__":
    preprocess_gtzan_dataset()