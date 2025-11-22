from pathlib import Path
import numpy as np

features_dir = Path('data/features')
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

print('Checking extracted features...\n')
for genre in genres:
    mfcc_files = list((features_dir / genre).glob('*_mfcc.npy'))
    melspec_files = list((features_dir / genre).glob('*_melspec.npy'))
    dwt_files = list((features_dir / genre).glob('*_dwt.npy'))
    print(f'{genre:12s}: MFCC={len(mfcc_files)}, MelSpec={len(melspec_files)}, DWT={len(dwt_files)}')

# Check a sample
sample = np.load(list((features_dir / 'blues').glob('*_mfcc.npy'))[0])
print(f'\nSample MFCC shape: {sample.shape} ✓')