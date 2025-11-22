#!/usr/bin/env python3
"""
Test if BHO training works
"""

import torch
from black_hole_optimizer import OptimizedCNN, BlackHoleOptimizer

def run_tests():
    # Test configuration
    test_config = {
        'num_blocks': 3,
        'filters_1': 64,
        'filters_2': 128,
        'filters_3': 256,
        'kernel_size': 3,
        'pool_size': 2,
        'fc_units': 256,
        'dropout_conv': 0.25,
        'dropout_fc': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32
    }

    print("Testing BHO components...")
    print("="*60)

    # Test 1: Model creation
    print("\n1. Testing model creation...")
    try:
        model = OptimizedCNN((13, 130), test_config)
        print("   ✓ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False

    # Test 2: Device
    print("\n2. Testing device...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"   Using: {device}")
    try:
        model.to(device)
        print("   ✓ Model moved to device")
    except Exception as e:
        print(f"   ✗ Device transfer failed: {e}")
        return False

    # Test 3: Data loading
    print("\n3. Testing data loading...")
    try:
        from src.data.dataset import get_data_loaders
        train_loader, val_loader, _ = get_data_loaders('mfcc', batch_size=32, num_workers=0)
        print(f"   ✓ Data loaded: {len(train_loader.dataset)} train samples")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False

    # Test 4: Forward pass
    print("\n4. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            for features, labels in train_loader:
                features = features.to(device)
                outputs = model(features)
                print(f"   ✓ Forward pass: {features.shape} → {outputs.shape}")
                break
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Training step
    print("\n5. Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"   ✓ Training step: loss={loss.item():.4f}")
            break
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Full BHO evaluation
    print("\n6. Testing BHO evaluation (1 epoch)...")
    try:
        bho = BlackHoleOptimizer('mfcc', population_size=2, max_iterations=1)
        fitness, eval_time, params = bho.train_and_evaluate(test_config, device=device, epochs=1)
        print(f"   ✓ BHO eval: fitness={fitness:.2f}%, time={eval_time:.1f}s")
        
        if fitness == 0.0:
            print("   ⚠️  Warning: Fitness is 0.0 - training may be failing silently")
    except Exception as e:
        print(f"   ✗ BHO evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)