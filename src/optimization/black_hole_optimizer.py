"""
Black Hole Optimization for CNN Hyperparameters
"""

import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
import copy

class BlackHoleOptimizer:
    """
    Black Hole Optimization Algorithm for hyperparameter tuning
    
    Inspired by black hole physics:
    - Stars = candidate hyperparameter configurations
    - Black hole = best configuration found
    - Stars move toward black hole (exploitation)
    - Stars near event horizon are replaced (exploration)
    """
    
    def __init__(self, 
                 search_space,
                 population_size=20,
                 max_iterations=30,
                 feature_type='mfcc'):
        """
        Args:
            search_space: Dict defining parameter ranges
            population_size: Number of candidate configurations
            max_iterations: Maximum optimization iterations
            feature_type: 'mfcc', 'melspec', or 'dwt'
        """
        self.search_space = search_space
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.feature_type = feature_type
        
        # Population of stars (configurations)
        self.population = []
        self.fitness_scores = []
        
        # Best configuration (black hole)
        self.black_hole = None
        self.black_hole_fitness = 0.0
        
        # History
        self.history = {
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': []
        }
    
    def _random_config(self):
        """Generate random configuration from search space"""
        config = {}
        for param_name, param_range in self.search_space.items():
            if isinstance(param_range, list):
                # Categorical choice
                config[param_name] = np.random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous range (min, max)
                if isinstance(param_range[0], int):
                    config[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    config[param_name] = np.random.uniform(param_range[0], param_range[1])
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")
        return config
    
    def initialize_population(self):
        """Initialize random population of configurations"""
        print(f"Initializing population of {self.population_size} configurations...")
        self.population = [self._random_config() for _ in range(self.population_size)]
        self.fitness_scores = [0.0] * self.population_size
    
    def evaluate_fitness(self, config, train_loader, val_loader, 
                        device='mps', epochs=10):
        """
        Evaluate fitness of a configuration
        
        Args:
            config: Hyperparameter configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            epochs: Number of training epochs (short for speed)
        
        Returns:
            Validation accuracy (fitness score)
        """
        from src.models.cnn_models import get_model
        from src.models.trainer import CNNTrainer
        
        # Create model with this configuration
        model = get_model(
            model_type=self.feature_type,
            num_classes=10,
            dropout_rate=config.get('dropout_rate', 0.5)
        )
        
        # Manually update architecture based on config
        # This is a simplified version - in practice, you'd rebuild the model
        # For now, we'll just train with different hyperparameters
        
        # Create trainer
        trainer = CNNTrainer(
            model=model,
            device=device,
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0001),
            model_name=f'bho_temp_{self.feature_type}'
        )
        
        # Train for limited epochs
        try:
            for epoch in range(epochs):
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)
            
            # Return validation accuracy as fitness
            return val_acc
            
        except Exception as e:
            print(f"Error training config: {e}")
            return 0.0
    
    def calculate_event_horizon(self):
        """Calculate event horizon radius"""
        # Event horizon: R = f(BH) / Σf(stars)
        total_fitness = sum(self.fitness_scores)
        if total_fitness == 0:
            return 0.1
        return self.black_hole_fitness / total_fitness
    
    def distance(self, star1, star2):
        """Calculate normalized distance between two configurations"""
        distances = []
        for param_name in self.search_space.keys():
            val1 = star1[param_name]
            val2 = star2[param_name]
            
            # Normalize by range
            param_range = self.search_space[param_name]
            if isinstance(param_range, list):
                # Categorical: 0 if same, 1 if different
                dist = 0.0 if val1 == val2 else 1.0
            else:
                # Numerical: normalize by range
                min_val, max_val = param_range
                if max_val > min_val:
                    dist = abs(val1 - val2) / (max_val - min_val)
                else:
                    dist = 0.0
            distances.append(dist)
        
        # Euclidean distance
        return np.sqrt(np.mean(np.array(distances) ** 2))
    
    def move_star_toward_black_hole(self, star):
        """Move star toward black hole"""
        new_star = {}
        rand = np.random.random()  # Random factor [0, 1]
        
        for param_name in self.search_space.keys():
            star_val = star[param_name]
            bh_val = self.black_hole[param_name]
            param_range = self.search_space[param_name]
            
            if isinstance(param_range, list):
                # Categorical: randomly move toward BH value
                if np.random.random() < 0.5:
                    new_star[param_name] = bh_val
                else:
                    new_star[param_name] = star_val
            else:
                # Numerical: interpolate toward BH
                new_val = star_val + rand * (bh_val - star_val)
                
                # Clip to valid range
                min_val, max_val = param_range
                new_val = np.clip(new_val, min_val, max_val)
                
                # Convert to int if original was int
                if isinstance(param_range[0], int):
                    new_val = int(round(new_val))
                
                new_star[param_name] = new_val
        
        return new_star
    
    def optimize(self, train_loader, val_loader, device='mps', 
                 evaluation_epochs=10, save_dir='models/bho_results'):
        """
        Run Black Hole Optimization
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            device: Device to use
            evaluation_epochs: Epochs per fitness evaluation
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print(f"BLACK HOLE OPTIMIZATION - {self.feature_type.upper()} CNN")
        print("="*60)
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Evaluation epochs: {evaluation_epochs}")
        print("="*60 + "\n")
        
        # Initialize population
        self.initialize_population()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Evaluate all stars
            print(f"\nEvaluating {self.population_size} configurations...")
            for i, star in enumerate(tqdm(self.population, desc="Fitness evaluation")):
                fitness = self.evaluate_fitness(
                    star, train_loader, val_loader,
                    device=device, epochs=evaluation_epochs
                )
                self.fitness_scores[i] = fitness
            
            # Find best (black hole)
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.black_hole_fitness:
                self.black_hole = copy.deepcopy(self.population[best_idx])
                self.black_hole_fitness = self.fitness_scores[best_idx]
                print(f"\n🕳️  New Black Hole found!")
                print(f"   Fitness: {self.black_hole_fitness:.2f}%")
            
            # Statistics
            mean_fitness = np.mean(self.fitness_scores)
            std_fitness = np.std(self.fitness_scores)
            
            print(f"\nIteration {iteration + 1} Summary:")
            print(f"  Best fitness:  {self.black_hole_fitness:.2f}%")
            print(f"  Mean fitness:  {mean_fitness:.2f}%")
            print(f"  Std fitness:   {std_fitness:.2f}%")
            
            # Update history
            self.history['iteration'].append(iteration + 1)
            self.history['best_fitness'].append(self.black_hole_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['std_fitness'].append(std_fitness)
            
            # Calculate event horizon
            event_horizon = self.calculate_event_horizon()
            print(f"  Event horizon: {event_horizon:.4f}")
            
            # Move stars and check event horizon
            new_population = []
            for i, star in enumerate(self.population):
                if i == best_idx:
                    # Keep black hole
                    new_population.append(star)
                    continue
                
                # Check if star crossed event horizon
                dist = self.distance(star, self.black_hole)
                
                if dist < event_horizon:
                    # Star absorbed - generate new random star
                    new_star = self._random_config()
                    print(f"  Star {i} absorbed, generating new star")
                else:
                    # Move star toward black hole
                    new_star = self.move_star_toward_black_hole(star)
                
                new_population.append(new_star)
            
            self.population = new_population
            
            # Save checkpoint
            checkpoint = {
                'iteration': iteration + 1,
                'black_hole': self.black_hole,
                'black_hole_fitness': self.black_hole_fitness,
                'history': self.history
            }
            
            checkpoint_path = save_path / f'{self.feature_type}_bho_checkpoint.json'
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        
        # Final results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best configuration found:")
        for param, value in self.black_hole.items():
            print(f"  {param}: {value}")
        print(f"\nBest fitness: {self.black_hole_fitness:.2f}%")
        print("="*60 + "\n")
        
        # Save final results
        results_path = save_path / f'{self.feature_type}_bho_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'best_config': self.black_hole,
                'best_fitness': self.black_hole_fitness,
                'history': self.history
            }, f, indent=2)
        
        print(f"✓ Results saved to: {results_path}")
        
        return self.black_hole, self.black_hole_fitness