#!/usr/bin/env python3
"""
Ultimate training with all optimizations
"""

import argparse
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.improved_cnn_models import get_improved_model
from src.models.trainer import CNNTrainer
from src.data.dataset import get_augmented_data_loaders

def main():
    parser = argparse.ArgumentParser(description='Ultimate optimized training')
    parser.add_argument('--feature', type=str, required=True,
                       choices=['mfcc', 'melspec', 'chroma'],
                       help='Feature type')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (max)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"ULTIMATE {args.feature.upper()} CNN TRAINING")
    print(f"{'='*60}")
    print(f"✅ Data augmentation: ENABLED")
    print(f"✅ Learning rate scheduling: ENABLED")
    print(f"✅ Early stopping: ENABLED (patience=25)")
    print(f"✅ Batch size: {args.batch_size}")
    print(f"✅ Initial LR: {args.lr}")
    print(f"✅ Max epochs: {args.epochs}")
    print(f"✅ Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load augmented data
    print("Loading augmented data...")
    train_loader, val_loader, test_loader = get_augmented_data_loaders(
        feature_type=args.feature,
        batch_size=args.batch_size,
        num_workers=8
    )
    
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Augmentation: Time stretch, noise, masking")
    
    # Create model
    print(f"\nCreating improved {args.feature} model...")
    model = get_improved_model(
        model_type=args.feature,
        num_classes=10,
        dropout_rate=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Parameters: {total_params:,}")
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=0.0001,
        model_name=f'ultimate_{args.feature}_cnn'
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir='models/ultimate_checkpoints',
        early_stopping_patience=35  # More patience with LR scheduling
    )
    
    # Test
    print("\n📊 Testing on test set...")
    test_acc, predictions, labels = trainer.test(test_loader)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation: {trainer.best_val_acc:.2f}%")
    print(f"Test Accuracy:   {test_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Calculate improvement
    baseline = {
        'mfcc': 61.33,
        'melspec': 71.33,
        'chroma': 44.00
    }
    
    improvement = test_acc - baseline.get(args.feature, 0)
    print(f"📈 Improvement: {improvement:+.2f}% from baseline")

if __name__ == "__main__":
    main()