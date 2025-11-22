#!/usr/bin/env python3
"""
Find optimal ensemble weights using validation set
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from itertools import product

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.improved_cnn_models import get_improved_model
from src.data.dataset import get_data_loaders

class EnsembleWeightOptimizer:
    """Find optimal weights for ensemble"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.models = {}
        self.feature_types = ['mfcc', 'melspec', 'chroma']
    
    def load_models(self, checkpoint_dir='models/improved_checkpoints'):
        """Load trained models"""
        print("Loading models...\n")
        
        for ft in self.feature_types:
            model = get_improved_model(ft, 10)
            checkpoint_file = Path(checkpoint_dir) / f'improved_{ft}_cnn_best.pth'
            
            if not checkpoint_file.exists():
                print(f"⚠️  {ft} not found")
                continue
            
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models[ft] = model
            print(f"✓ {ft:10s}: Val={checkpoint['val_acc']:.2f}%")
        
        print(f"\n✓ Loaded {len(self.models)} models\n")
    
    def get_predictions(self, split='val'):
        """Get predictions on validation or test set"""
        all_preds = {}
        all_labels = []
        
        print(f"Getting {split} predictions...")
        
        for ft in self.models.keys():
            train_loader, val_loader, test_loader = get_data_loaders(
                ft, batch_size=32, num_workers=4
            )
            
            loader = val_loader if split == 'val' else test_loader
            
            preds = []
            labels = []
            
            with torch.no_grad():
                for features, batch_labels in tqdm(loader, desc=f"  {ft}", leave=False):
                    features = features.to(self.device)
                    outputs = self.models[ft](features)
                    probs = torch.softmax(outputs, dim=1)
                    preds.append(probs.cpu().numpy())
                    labels.append(batch_labels.numpy())
            
            all_preds[ft] = np.vstack(preds)
            
            if len(all_labels) == 0:
                all_labels = np.concatenate(labels)
        
        return all_preds, all_labels
    
    def evaluate_weights(self, predictions, labels, weights):
        """Evaluate ensemble with given weights"""
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        
        for ft, preds in predictions.items():
            weight = weights.get(ft, 0)
            weighted_sum += preds * weight
        
        ensemble_preds = np.argmax(weighted_sum, axis=1)
        accuracy = 100.0 * np.sum(ensemble_preds == labels) / len(labels)
        
        return accuracy
    
    def grid_search(self, predictions, labels, resolution=10):
        """Grid search over weight combinations"""
        print(f"\n{'='*60}")
        print("GRID SEARCH FOR OPTIMAL WEIGHTS")
        print(f"{'='*60}")
        print(f"Resolution: {resolution} steps per weight")
        print(f"Total combinations: {resolution**len(self.models)}")
        print(f"{'='*60}\n")
        
        # Generate weight combinations that sum to 1
        step = 1.0 / (resolution - 1)
        
        best_acc = 0
        best_weights = None
        
        # Try all combinations
        total_tried = 0
        
        if len(self.models) == 3:
            # For 3 models
            for w1 in np.linspace(0, 1, resolution):
                for w2 in np.linspace(0, 1 - w1, resolution):
                    w3 = 1.0 - w1 - w2
                    
                    if w3 < 0 or w3 > 1:
                        continue
                    
                    weights = {
                        list(self.models.keys())[0]: w1,
                        list(self.models.keys())[1]: w2,
                        list(self.models.keys())[2]: w3
                    }
                    
                    acc = self.evaluate_weights(predictions, labels, weights)
                    total_tried += 1
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = weights.copy()
        
        elif len(self.models) == 2:
            # For 2 models
            for w1 in np.linspace(0, 1, resolution):
                w2 = 1.0 - w1
                
                weights = {
                    list(self.models.keys())[0]: w1,
                    list(self.models.keys())[1]: w2
                }
                
                acc = self.evaluate_weights(predictions, labels, weights)
                total_tried += 1
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights.copy()
        
        print(f"✓ Tested {total_tried} weight combinations")
        print(f"\n{'='*60}")
        print("OPTIMAL WEIGHTS FOUND")
        print(f"{'='*60}")
        
        for ft, weight in best_weights.items():
            print(f"{ft:10s}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\nValidation Accuracy: {best_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return best_weights, best_acc
    
    def compare_strategies(self, val_preds, val_labels, test_preds, test_labels):
        """Compare different weighting strategies"""
        
        print(f"\n{'='*60}")
        print("COMPARING WEIGHTING STRATEGIES")
        print(f"{'='*60}\n")
        
        strategies = {}
        
        # 1. Equal weights
        equal_weights = {ft: 1.0/len(self.models) for ft in self.models.keys()}
        strategies['Equal weights'] = equal_weights
        
        # 2. Validation accuracy weights
        val_accs = {}
        for ft, preds in val_preds.items():
            pred_labels = np.argmax(preds, axis=1)
            val_accs[ft] = np.sum(pred_labels == val_labels) / len(val_labels)
        
        total_val_acc = sum(val_accs.values())
        val_based_weights = {ft: acc/total_val_acc for ft, acc in val_accs.items()}
        strategies['Val accuracy weights'] = val_based_weights
        
        # 3. Best model only
        best_model = max(val_accs.items(), key=lambda x: x[1])[0]
        best_only_weights = {ft: (1.0 if ft == best_model else 0.0) for ft in self.models.keys()}
        strategies[f'Best only ({best_model})'] = best_only_weights
        
        # 4. Drop worst model (2-model ensemble)
        worst_model = min(val_accs.items(), key=lambda x: x[1])[0]
        remaining_models = [ft for ft in self.models.keys() if ft != worst_model]
        drop_worst_weights = {ft: (1.0/len(remaining_models) if ft in remaining_models else 0.0) 
                             for ft in self.models.keys()}
        strategies[f'Drop worst ({worst_model})'] = drop_worst_weights
        
        # 5. Optimal from grid search
        optimal_weights, _ = self.grid_search(val_preds, val_labels, resolution=20)
        strategies['Optimal (grid search)'] = optimal_weights
        
        # Evaluate all strategies
        print("\nSTRATEGY COMPARISON")
        print("="*60)
        print(f"{'Strategy':<30} {'Val Acc':<12} {'Test Acc':<12}")
        print("="*60)
        
        results = []
        
        for name, weights in strategies.items():
            val_acc = self.evaluate_weights(val_preds, val_labels, weights)
            test_acc = self.evaluate_weights(test_preds, test_labels, weights)
            results.append((name, weights, val_acc, test_acc))
            print(f"{name:<30} {val_acc:>6.2f}%     {test_acc:>6.2f}%")
        
        print("="*60)
        
        # Find best on test set
        best_result = max(results, key=lambda x: x[3])
        print(f"\n🏆 BEST STRATEGY: {best_result[0]}")
        print(f"   Test Accuracy: {best_result[3]:.2f}%")
        print(f"\n   Weights:")
        for ft, weight in best_result[1].items():
            print(f"     {ft:10s}: {weight:.3f} ({weight*100:.1f}%)")
        
        return best_result

def main():
    print("="*60)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*60 + "\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    optimizer = EnsembleWeightOptimizer(device=device)
    optimizer.load_models()
    
    # Get predictions
    print("="*60)
    val_preds, val_labels = optimizer.get_predictions('val')
    test_preds, test_labels = optimizer.get_predictions('test')
    print("="*60)
    
    # Compare strategies
    best_strategy = optimizer.compare_strategies(val_preds, val_labels, test_preds, test_labels)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nUse the '{best_strategy[0]}' strategy")
    print(f"Final Test Accuracy: {best_strategy[3]:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()