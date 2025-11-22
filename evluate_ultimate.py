#!/usr/bin/env python3
"""
Evaluate ultimate optimized ensemble with weighted voting
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.improved_cnn_models import get_improved_model
from src.data.dataset import get_data_loaders

class WeightedEnsembleEvaluator:
    """Weighted ensemble evaluation"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.models = {}
        self.feature_types = ['mfcc', 'melspec', 'chroma']
        
        # Optimal weights based on validation performance
        self.weights = {
            'mfcc': 0.37,     # 30%
            'melspec': 0.36,  # 60% (best model)
            'chroma': 0.27    # 10% (weakest)
        }
        
        # Genre mapping
        with open('data/splits/genre_mapping.json', 'r') as f:
            self.genre_to_idx = json.load(f)
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
    
    def load_models(self, checkpoint_dir='models/ultimate_checkpoints'):
        """Load all trained models"""
        checkpoint_path = Path(checkpoint_dir)
        
        print("Loading ultimate optimized models...")
        for feature_type in self.feature_types:
            model = get_improved_model(model_type=feature_type, num_classes=10)
            
            checkpoint_file = checkpoint_path / f'ultimate_{feature_type}_cnn_best.pth'
            
            if not checkpoint_file.exists():
                print(f"⚠️  {checkpoint_file} not found, skipping {feature_type}")
                continue
            
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models[feature_type] = model
            val_acc = checkpoint.get('val_acc', 0)
            print(f"  ✓ {feature_type:8s} (Val: {val_acc:.2f}%, Weight: {self.weights[feature_type]:.1%})")
        
        print(f"\n✓ Loaded {len(self.models)} models")
    
    def predict_weighted_ensemble(self, test_loaders):
        """Weighted ensemble prediction"""
        
        print("\nRunning weighted ensemble predictions...")
        
        all_predictions = {}
        all_labels = []
        
        # Get predictions from each model
        for feature_type, model in self.models.items():
            print(f"  Predicting with {feature_type}_cnn...")
            loader = test_loaders[feature_type]
            
            predictions = []
            labels = []
            
            with torch.no_grad():
                for features, batch_labels in tqdm(loader, desc=f"    {feature_type}", leave=False):
                    features = features.to(self.device)
                    outputs = model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions.append(probabilities.cpu().numpy())
                    labels.append(batch_labels.numpy())
            
            all_predictions[feature_type] = np.vstack(predictions)
            
            if len(all_labels) == 0:
                all_labels = np.concatenate(labels)
        
        # Weighted ensemble
        print("\n  Computing weighted ensemble...")
        weighted_preds = []
        for feature_type, preds in all_predictions.items():
            weight = self.weights[feature_type]
            weighted_preds.append(preds * weight)
        
        ensemble_probs = np.sum(weighted_preds, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Individual accuracies
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL ACCURACIES (Test Set)")
        print("="*60)
        
        individual_accs = {}
        for feature_type, preds in all_predictions.items():
            pred_labels = np.argmax(preds, axis=1)
            accuracy = 100.0 * np.sum(pred_labels == all_labels) / len(all_labels)
            individual_accs[feature_type] = accuracy
            print(f"{feature_type:12s}: {accuracy:.2f}%")
        
        # Ensemble accuracy
        ensemble_accuracy = 100.0 * np.sum(ensemble_preds == all_labels) / len(all_labels)
        
        print("="*60)
        print(f"{'WEIGHTED ENSEMBLE':12s}: {ensemble_accuracy:.2f}% 🎯")
        print("="*60)
        
        best_individual = max(individual_accs.values())
        improvement = ensemble_accuracy - best_individual
        print(f"\nImprovement: {improvement:+.2f}% over best individual")
        
        return ensemble_preds, all_labels, individual_accs, ensemble_accuracy
    
    def evaluate(self):
        """Full evaluation"""
        
        # Load test data
        test_loaders = {}
        for feature_type in self.feature_types:
            if feature_type in self.models:
                _, _, test_loader = get_data_loaders(
                    feature_type=feature_type,
                    batch_size=32,
                    num_workers=8
                )
                test_loaders[feature_type] = test_loader
        
        # Predict
        preds, labels, individual_accs, ensemble_acc = self.predict_weighted_ensemble(test_loaders)
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            labels, preds,
            target_names=self.idx_to_genre.values(),
            digits=2
        )
        print(report)
        
        # Save results
        results_path = Path('results/ultimate_ensemble_results.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'ensemble_accuracy': float(ensemble_acc),
                'individual_accuracies': {k: float(v) for k, v in individual_accs.items()},
                'weights': self.weights
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
        
        return ensemble_acc

def main():
    print("="*60)
    print("ULTIMATE WEIGHTED ENSEMBLE EVALUATION")
    print("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    evaluator = WeightedEnsembleEvaluator(device=device)
    evaluator.load_models()
    ensemble_acc = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Final Weighted Ensemble: {ensemble_acc:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()