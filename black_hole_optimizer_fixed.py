"""
Black Hole Optimization - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
import copy
from datetime import datetime
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import get_data_loaders

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class OptimizedCNN(nn.Module):
    """Flexible CNN with configurable architecture"""
    
    def __init__(self, input_shape, config):
        super(OptimizedCNN, self).__init__()
        
        in_channels = 1
        in_height, in_width = input_shape
        
        layers = []
        
        # Build convolutional blocks
        num_blocks = config['num_blocks']
        for i in range(num_blocks):
            out_channels = config[f'filters_{i+1}']
            kernel = config['kernel_size']
            
            # Double conv block
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Conv2d(out_channels, out_channels, kernel, padding=kernel//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            # Pooling
            pool_size = config.get('pool_size', 2)
            layers.append(nn.MaxPool2d(pool_size, pool_size))
            layers.append(nn.Dropout2d(config['dropout_conv']))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, config['fc_units']),
            nn.BatchNorm1d(config['fc_units']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_fc']),
            nn.Linear(config['fc_units'], 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BlackHoleOptimizer:
    """Black Hole Optimization"""
    
    def __init__(self, feature_type, population_size=12, max_iterations=15):
        self.feature_type = feature_type
        self.population_size = population_size
        self.max_iterations = max_iterations
        
        self.input_shapes = {
            'mfcc': (13, 130),
            'melspec': (128, 130),
            'chroma': (12, 130)
        }
        
        # Feature-specific search spaces
        if feature_type == 'mfcc':
            self.search_space = {
                'num_blocks': [3],
                'filters_1': [96, 128, 160],
                'filters_2': [192, 256, 320],
                'filters_3': [384, 512, 640],
                'filters_4': [512, 768],
                'kernel_size': [3, 5],
                'pool_size': [2],
                'fc_units': [384, 512, 768],
                'dropout_conv': [0.20, 0.25, 0.30],
                'dropout_fc': [0.45, 0.50, 0.55],
                'learning_rate': [0.0003, 0.0005, 0.0008, 0.001],
                'batch_size': [48, 64, 80]
            }
        elif feature_type == 'melspec':
            self.search_space = {
                'num_blocks': [4],
                'filters_1': [64, 96, 128],
                'filters_2': [128, 192, 256],
                'filters_3': [256, 384, 512],
                'filters_4': [512, 768, 1024],
                'kernel_size': [3, 5],
                'pool_size': [2],
                'fc_units': [512, 768, 1024],
                'dropout_conv': [0.15, 0.20, 0.25],
                'dropout_fc': [0.40, 0.50, 0.60],
                'learning_rate': [0.0003, 0.0005, 0.0008, 0.001],
                'batch_size': [32, 48, 64]
            }
        elif feature_type == 'chroma':
            self.search_space = {
                'num_blocks': [3],
                'filters_1': [128, 160, 192],
                'filters_2': [256, 320, 384],
                'filters_3': [512, 640, 768],
                'filters_4': [768, 1024],
                'kernel_size': [3, 5],
                'pool_size': [2],
                'fc_units': [384, 512, 768],
                'dropout_conv': [0.20, 0.25, 0.30],
                'dropout_fc': [0.35, 0.40, 0.45],
                'learning_rate': [0.0003, 0.0005, 0.0008],
                'batch_size': [64, 80, 96]
            }
        
        self.population = []
        self.fitness_scores = []
        self.black_hole = None
        self.black_hole_fitness = 0.0
        
        self.history = {
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_config': [],
            'evaluation_times': []
        }
    
    def random_config(self):
        """Generate random configuration"""
        config = {}
        for param, values in self.search_space.items():
            value = np.random.choice(values)
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            config[param] = value
        
        num_blocks = config['num_blocks']
        if num_blocks == 3 and 'filters_4' in config:
            config.pop('filters_4')
        
        return config
    
    def train_and_evaluate(self, config, device='mps', epochs=20):
        """Train and evaluate configuration"""
        start_time = time.time()
        
        try:
            train_loader, val_loader, _ = get_data_loaders(
                feature_type=self.feature_type,
                batch_size=config['batch_size'],
                num_workers=0
            )
            
            model = OptimizedCNN(
                input_shape=self.input_shapes[self.feature_type],
                config=config
            )
            model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=0.0001
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
            )
            
            best_val_acc = 0.0
            patience = 8
            patience_counter = 0
            
            for epoch in range(epochs):
                # Train
                model.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Validate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                val_acc = 100.0 * correct / total
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            elapsed_time = time.time() - start_time
            return best_val_acc, elapsed_time, total_params
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"    Error: {e}")
            return 0.0, elapsed_time, 0
    
    def distance(self, config1, config2):
        """Calculate distance between configs"""
        if config2 is None:
            return 1.0
        
        try:
            distances = []
            common_params = set(config1.keys()) & set(config2.keys())
            
            for param in common_params:
                dist = 0.0 if config1[param] == config2[param] else 1.0
                distances.append(dist)
            
            return np.mean(distances) if distances else 1.0
        except:
            return 1.0
    
    def move_toward_black_hole(self, star):
        """Move star toward black hole"""
        if self.black_hole is None:
            return star
        
        new_star = {}
        for param in self.search_space.keys():
            if param not in self.black_hole:
                new_star[param] = star.get(param, np.random.choice(self.search_space[param]))
                continue
            
            if np.random.random() < 0.7:
                new_star[param] = self.black_hole[param]
            else:
                new_star[param] = star.get(param, np.random.choice(self.search_space[param]))
        
        return new_star
    
    def optimize(self, device='mps', eval_epochs=20):
        """Run Black Hole Optimization"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f'models/bho_results/{self.feature_type}_{timestamp}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = save_dir / 'optimization.log'
        
        def log_print(msg):
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
        
        log_print("="*70)
        log_print(f"BLACK HOLE OPTIMIZATION - {self.feature_type.upper()}")
        log_print("="*70)
        log_print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print("="*70 + "\n")
        
        # Initialize
        self.population = [self.random_config() for _ in range(self.population_size)]
        self.fitness_scores = [0.0] * self.population_size
        
        total_evaluations = 0
        total_eval_time = 0
        
        # Main loop
        for iteration in range(self.max_iterations):
            log_print(f"\n{'='*70}")
            log_print(f"ITERATION {iteration + 1}/{self.max_iterations}")
            log_print(f"{'='*70}")
            
            iteration_start = time.time()
            
            # Evaluate all
            for i, config in enumerate(self.population):
                log_print(f"\nStar {i+1}/{self.population_size}:")
                
                fitness, eval_time, params = self.train_and_evaluate(config, device, epochs=eval_epochs)
                self.fitness_scores[i] = fitness
                total_evaluations += 1
                total_eval_time += eval_time
                
                log_print(f"  Fitness: {fitness:.2f}% | Time: {eval_time:.1f}s")
            
            # Find black hole
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.black_hole_fitness:
                self.black_hole = copy.deepcopy(self.population[best_idx])
                self.black_hole_fitness = self.fitness_scores[best_idx]
                log_print(f"\n🕳️  NEW BLACK HOLE: {self.black_hole_fitness:.2f}%")
            
            # Stats
            mean_fitness = np.mean(self.fitness_scores)
            iteration_time = time.time() - iteration_start
            
            log_print(f"\nSummary:")
            log_print(f"  Best: {self.black_hole_fitness:.2f}%")
            log_print(f"  Mean: {mean_fitness:.2f}%")
            log_print(f"  Time: {iteration_time/60:.1f} min")
            
            # Save history
            self.history['iteration'].append(iteration + 1)
            self.history['best_fitness'].append(float(self.black_hole_fitness))
            self.history['mean_fitness'].append(float(mean_fitness))
            self.history['best_config'].append(copy.deepcopy(self.black_hole))
            self.history['evaluation_times'].append(iteration_time)
            
            # Move stars (only after first iteration)
            if self.black_hole is not None and iteration > 0:
                total_fitness = sum(self.fitness_scores)
                event_horizon = self.black_hole_fitness / total_fitness if total_fitness > 0 else 0.1
                
                new_population = []
                absorbed = 0
                
                for i, star in enumerate(self.population):
                    if i == best_idx:
                        new_population.append(star)
                        continue
                    
                    dist = self.distance(star, self.black_hole)
                    
                    if dist < event_horizon:
                        new_population.append(self.random_config())
                        absorbed += 1
                    else:
                        new_population.append(self.move_toward_black_hole(star))
                
                self.population = new_population
                log_print(f"  Absorbed: {absorbed} stars")
            
            # Save checkpoint
            checkpoint_data = {
                'iteration': iteration + 1,
                'black_hole': convert_numpy_types(self.black_hole),  # ✅ Convert
                'black_hole_fitness': float(self.black_hole_fitness),
                'history': convert_numpy_types(self.history)  # ✅ Convert
            }
            with open(save_dir / 'checkpoint.json', 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        # Final results
        log_print("\n" + "="*70)
        log_print("COMPLETE")
        log_print("="*70)
        
        if self.black_hole is not None:
            log_print("\nBest Configuration:")
            for param, value in sorted(self.black_hole.items()):
                log_print(f"  {param:20s}: {value}")
            log_print(f"\nBest Accuracy: {self.black_hole_fitness:.2f}%")
        else:
            log_print("\nNo valid configuration found!")
        
        log_print("="*70)
        
        # Save final
        with open(save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'best_config': convert_numpy_types(self.black_hole),
                'best_fitness': float(self.black_hole_fitness),
                'history': convert_numpy_types(self.history),
                'feature_type': self.feature_type
            }, f, indent=2)
        
        return self.black_hole, self.black_hole_fitness, save_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, required=True,
                       choices=['mfcc', 'melspec', 'chroma'])
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--eval_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='mps')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    
    optimizer = BlackHoleOptimizer(
        feature_type=args.feature,
        population_size=args.population,
        max_iterations=args.iterations
    )
    
    best_config, best_fitness, save_dir = optimizer.optimize(
        device=device,
        eval_epochs=args.eval_epochs
    )
    
    print(f"\n✅ DONE! Best: {best_fitness:.2f}%")
    print(f"Results: {save_dir}")


if __name__ == "__main__":
    main()