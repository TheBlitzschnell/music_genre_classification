#!/usr/bin/env python3
"""
Train a single CNN model
Usage: python train_single_cnn.py --feature mfcc --epochs 50
"""

import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.cnn_models import get_model
from src.models.trainer import CNNTrainer
from src.data.dataset import get_data_loaders

def main():
    parser = argparse.ArgumentParser(description='Train single CNN')
    parser.add_argument('--feature', type=str, default='mfcc',
                       choices=['mfcc', 'melspec', 'chroma'],
                       help='Feature type to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Feature type: {args.feature}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        feature_type=args.feature,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create model
    print(f"\nCreating {args.feature.upper()} CNN model...")
    model = get_model(model_type=args.feature, num_classes=10, dropout_rate=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        model_name=f'{args.feature}_cnn'
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=10
    )
    
    # Test
    print("\nEvaluating on test set...")
    test_acc, predictions, labels = trainer.test(test_loader)
    
    print(f"\n{'='*60}")
    print(f"Final Results")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()