import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCC_CNN(nn.Module):
    """CNN specialized for MFCC features (13×130)"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MFCC_CNN, self).__init__()
        
        # Input: (batch, 1, 13, 130)
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (64, 6, 65)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> (128, 3, 32)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> (256, 1, 16)
        
        # Calculate flattened size
        self.flatten_size = 256 * 1 * 16  # 4096
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MelSpec_CNN(nn.Module):
    """CNN specialized for Mel-Spectrogram features (128×130)"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MelSpec_CNN, self).__init__()
        
        # Input: (batch, 1, 128, 130)
        
        # Convolutional Block 1 - MORE filters for richer input
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (128, 64, 65)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> (256, 32, 32)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> (512, 16, 16)
        
        # Calculate flattened size
        self.flatten_size = 512 * 16 * 16  # 131072
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class Chroma_CNN(nn.Module):
    """CNN specialized for Chroma features (12×130)"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(Chroma_CNN, self).__init__()
        
        # Input: (batch, 1, 12, 130)
        # Chroma has only 12 pitch classes - small input!
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # → (64, 6, 65)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # → (128, 3, 32)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # → (256, 1, 16)
        
        # Flatten size
        self.flatten_size = 256 * 1 * 16  # 4096
        
        # FC layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_type='mfcc', num_classes=10, dropout_rate=0.5):
    """Factory function to get the appropriate model"""
    
    if model_type.lower() == 'mfcc':
        return MFCC_CNN(num_classes, dropout_rate)
    elif model_type.lower() in ['melspec', 'mel']:
        return MelSpec_CNN(num_classes, dropout_rate)
    elif model_type.lower() == 'chroma':  # ✅ Add this
        return Chroma_CNN(num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    print("Testing CNN architectures...\n")
    
    # Test MFCC CNN
    mfcc_model = MFCC_CNN()
    mfcc_input = torch.randn(4, 1, 13, 130)  # batch_size=4
    mfcc_output = mfcc_model(mfcc_input)
    print(f"MFCC CNN:")
    print(f"  Input shape:  {mfcc_input.shape}")
    print(f"  Output shape: {mfcc_output.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in mfcc_model.parameters()):,}\n")
    
    # Test MelSpec CNN
    mel_model = MelSpec_CNN()
    mel_input = torch.randn(4, 1, 128, 130)
    mel_output = mel_model(mel_input)
    print(f"MelSpec CNN:")
    print(f"  Input shape:  {mel_input.shape}")
    print(f"  Output shape: {mel_output.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in mel_model.parameters()):,}\n")
    
    # Test Chroma CNN
    chroma_model = Chroma_CNN()
    chroma_input = torch.randn(4, 1, 12, 130)
    chroma_output = chroma_model(chroma_input)
    print(f"Chroma CNN:")
    print(f"  Input shape:  {chroma_input.shape}")
    print(f"  Output shape: {chroma_output.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in chroma_model.parameters()):,}\n")
    
    print("✓ All models working correctly!")