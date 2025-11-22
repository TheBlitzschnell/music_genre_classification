import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json

class CNNTrainer:
    """Trainer for single CNN model"""
    
    def __init__(self, model, device='mps', learning_rate=0.001, 
                 weight_decay=0.0001, model_name='model'):
        """
        Args:
            model: PyTorch model
            device: 'mps', 'cuda', or 'cpu'
            learning_rate: Learning rate
            weight_decay: L2 regularization
            model_name: Name for saving checkpoints
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        
        # Move model to device
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0

        self.use_amp = (device == 'mps')
        if self.use_amp:
            self.scaler = GradScaler('mps')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for features, labels in pbar:
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('mps'):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                # Backward pass with AMP
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc='Validation'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              save_dir='models/checkpoints', early_stopping_patience=20):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            early_stopping_patience: Epochs to wait before early stopping
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")
        
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Update history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                
                checkpoint_path = save_path / f'{self.model_name}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
        
        # Save final model
        final_path = save_path / f'{self.model_name}_final.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }, final_path)
        
        # Save history
        history_path = save_path / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved to: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def test(self, test_loader):
        """Test the model"""
        self.model.eval()
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc='Testing'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * correct / total
        
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        
        return test_acc, all_predictions, all_labels