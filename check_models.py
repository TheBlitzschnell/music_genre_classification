from pathlib import Path
import torch
import json

checkpoint_dir = Path('models/checkpoints')

print("="*60)
print("TRAINED MODELS CHECK")
print("="*60)

feature_types = ['mfcc', 'melspec', 'dwt']
models_found = []

for ft in feature_types:
    checkpoint_file = checkpoint_dir / f'{ft}_cnn_best.pth'
    
    if checkpoint_file.exists():
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        val_acc = checkpoint.get('val_acc', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        
        print(f"\n✓ {ft:10s}_cnn:")
        print(f"    File: {checkpoint_file.name}")
        print(f"    Val Acc: {val_acc:.2f}%")
        print(f"    Epoch: {epoch}")
        
        # Check history file
        history_file = checkpoint_dir / f'{ft}_cnn_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            print(f"    Trained epochs: {len(history['train_loss'])}")
            print(f"    Best val acc: {max(history['val_acc']):.2f}%")
        
        models_found.append(ft)
    else:
        print(f"\n❌ {ft:10s}_cnn: NOT FOUND")

print("\n" + "="*60)
print(f"SUMMARY: {len(models_found)}/3 models trained")
print("="*60)

if len(models_found) < 3:
    print("\nMissing models:")
    for ft in feature_types:
        if ft not in models_found:
            print(f"  - {ft}_cnn")
    print("\nTrain missing models with:")
    for ft in feature_types:
        if ft not in models_found:
            print(f"  python train_single_cnn.py --feature {ft} --epochs 30 --batch_size 32")
else:
    print("\n✓ All 3 models ready for ensemble evaluation!")
    print("\nRun: python evaluate_ensemble.py")