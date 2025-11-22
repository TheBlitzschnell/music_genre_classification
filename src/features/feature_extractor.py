import librosa
import numpy as np
import pywt
from pathlib import Path

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
    
    def extract_dwt(self, audio_path, wavelet='db4', level=5):
        """Extract DWT coefficients with improved processing"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Normalize audio first
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        # Process coefficients
        dwt_features = []
        
        for i, coeff in enumerate(coeffs):
            # Take absolute value (magnitude)
            coeff = np.abs(coeff)
            
            # Downsample to target length using averaging (smoother than truncation)
            if len(coeff) > self.target_length:
                # Reshape and average
                n_segments = self.target_length
                segment_size = len(coeff) // n_segments
                coeff_downsampled = []
                for j in range(n_segments):
                    start = j * segment_size
                    end = start + segment_size
                    if end > len(coeff):
                        end = len(coeff)
                    coeff_downsampled.append(np.mean(coeff[start:end]))
                coeff = np.array(coeff_downsampled)
            elif len(coeff) < self.target_length:
                # Pad with zeros
                coeff = np.pad(coeff, (0, self.target_length - len(coeff)), mode='constant')
            
            dwt_features.append(coeff)
        
        # Stack all levels
        dwt_features = np.array(dwt_features)
        
        # Normalize each level independently
        for i in range(len(dwt_features)):
            level_mean = dwt_features[i].mean()
            level_std = dwt_features[i].std()
            if level_std > 0:
                dwt_features[i] = (dwt_features[i] - level_mean) / level_std
        
        # Pad to 256 channels if needed
        # Pad to 256 channels if needed
        if dwt_features.shape[0] < 256:
            pad_size = 256 - dwt_features.shape[0]
            # Repeat the features multiple times to fill space
            repeats_needed = (pad_size // dwt_features.shape[0]) + 1
            repeated_features = np.tile(dwt_features, (repeats_needed, 1))
            # Take exactly what we need
            dwt_features = np.vstack([dwt_features, repeated_features[:pad_size]])
    
        # Ensure exactly 256 rows
        dwt_features = dwt_features[:256, :]
    
        return dwt_features
    
    def extract_chroma(self, audio_path):
        """Extract Chroma features"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
        # Resize to target length
        if chroma.shape[1] < self.target_length:
            chroma = np.pad(chroma, ((0, 0), (0, self.target_length - chroma.shape[1])), mode='constant')
        else:
            chroma = chroma[:, :self.target_length]
    
        return chroma
    
    def extract_all_features(self, audio_path):
        """Extract all feature types"""
        return {
            'mfcc': self.extract_mfcc(audio_path),
            'melspec': self.extract_melspectrogram(audio_path),
            'dwt': self.extract_dwt(audio_path),
            'chroma': self.extract_chroma(audio_path)  # Add this if you want it
        }

# Test
if __name__ == "__main__":
    extractor = FeatureExtractor()
    features = extractor.extract_all_features("data/raw/genres/blues/blues.00000.wav")
    print(f"MFCC shape: {features['mfcc'].shape}")
    print(f"Mel-spectrogram shape: {features['melspec'].shape}")
    print(f"DWT shape: {features['dwt'].shape}")
    print(f"Chroma shape: {features['chroma'].shape}")