"""
Improved CNN architectures based on best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMFCC_CNN(nn.Module):
    """Improved MFCC CNN with deeper architecture and better regularization"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImprovedMFCC_CNN, self).__init__()
        
        # Input: (batch, 1, 13, 130)
        
        # Block 1 - Initial feature extraction
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Block 2 - Mid-level features
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Block 3 - High-level features
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv3b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class ImprovedMelSpec_CNN(nn.Module):
    """Improved Mel-Spectrogram CNN - deeper with residual-like connections"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImprovedMelSpec_CNN, self).__init__()
        
        # Input: (batch, 1, 128, 130)
        
        # Block 1
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Block 2
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        # Block 3
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Block 4
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class ImprovedChroma_CNN(nn.Module):
    """Improved Chroma CNN - optimized for small input (12×130)"""
    
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(ImprovedChroma_CNN, self).__init__()
        
        # Input: (batch, 1, 12, 130)
        
        # Block 1
        self.conv1a = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Block 2
        self.conv2a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Block 3
        self.conv3a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


def get_improved_model(model_type='mfcc', num_classes=10, dropout_rate=0.5):
    """Factory function for improved models"""
    
    if model_type.lower() == 'mfcc':
        return ImprovedMFCC_CNN(num_classes, dropout_rate)
    elif model_type.lower() in ['melspec', 'mel']:
        return ImprovedMelSpec_CNN(num_classes, dropout_rate)
    elif model_type.lower() == 'chroma':
        return ImprovedChroma_CNN(num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing improved architectures...\n")
    
    # MFCC
    mfcc_model = ImprovedMFCC_CNN()
    mfcc_input = torch.randn(4, 1, 13, 130)
    mfcc_output = mfcc_model(mfcc_input)
    print(f"Improved MFCC CNN:")
    print(f"  Input:  {mfcc_input.shape}")
    print(f"  Output: {mfcc_output.shape}")
    print(f"  Params: {sum(p.numel() for p in mfcc_model.parameters()):,}\n")
    
    # MelSpec
    mel_model = ImprovedMelSpec_CNN()
    mel_input = torch.randn(4, 1, 128, 130)
    mel_output = mel_model(mel_input)
    print(f"Improved MelSpec CNN:")
    print(f"  Input:  {mel_input.shape}")
    print(f"  Output: {mel_output.shape}")
    print(f"  Params: {sum(p.numel() for p in mel_model.parameters()):,}\n")
    
    # Chroma
    chroma_model = ImprovedChroma_CNN()
    chroma_input = torch.randn(4, 1, 12, 130)
    chroma_output = chroma_model(chroma_input)
    print(f"Improved Chroma CNN:")
    print(f"  Input:  {chroma_input.shape}")
    print(f"  Output: {chroma_output.shape}")
    print(f"  Params: {sum(p.numel() for p in chroma_model.parameters()):,}\n")
    
    print("✓ All improved models working!")