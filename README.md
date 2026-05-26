# 🎵 Music Genre Classification with Parallel CNN Ensemble

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced deep learning system for automatic music genre classification using parallel CNN ensemble with optimized architectures. Developed as part of a Speech Processing university course project.

## 🎯 Project Overview

This project implements a **parallel CNN ensemble system** that classifies music into 10 genres by combining three specialized neural networks, each processing different audio feature representations:

- **MFCC CNN**: Captures timbral and textural characteristics
- **Mel-Spectrogram CNN**: Analyzes melodic and harmonic content  
- **Chroma CNN**: Processes harmonic and pitch class information

**Final Performance**: 68.67% accuracy on GTZAN test set with weighted ensemble voting.

## 📊 Key Results

| Model | Individual Accuracy | Ensemble Contribution |
|-------|-------------------|---------------------|
| **Mel-Spectrogram** | 71.33% | 50% weight |
| **MFCC** | 61.33% | 35% weight |
| **Chroma** | 44.00% | 15% weight |
| **Weighted Ensemble** | **68.67%** | Final prediction |

### Per-Genre Performance

| Genre | Accuracy | Notes |
|-------|----------|-------|
| Classical | 100% | Perfect classification |
| Jazz | 87% | Strong performance |
| Metal | 87% | Distinctive patterns |
| Hip-hop | 73% | Clear rhythmic signature |
| Pop | 53% | Moderate overlap with Disco |
| Rock | 40% | Confused with Metal |

## 🏗️ Architecture

### Parallel CNN Design
```
Audio Input (30-second clips)
          |
    ┌─────┴─────┬──────────┐
    ↓           ↓          ↓
  MFCC      Mel-Spec    Chroma
(13×130)   (128×130)   (12×130)
    |           |          |
    ↓           ↓          ↓
CNN-1 (3)   CNN-2 (4)   CNN-3 (3)
  blocks      blocks     blocks
    |           |          |
    ↓           ↓          ↓
  61.33%      71.33%     44.00%
    |           |          |
    └─────┬─────┴──────────┘
          ↓
   Weighted Ensemble
     (35%, 50%, 15%)
          ↓
      68.67% Final
```

### Individual CNN Architecture

Each CNN features:
- **VGG-style double convolution blocks**
- Progressive filter expansion (64→128→256→512)
- Batch normalization for training stability
- Global average pooling to reduce parameters
- Dropout regularization (conv + FC layers)
- ReLU activation throughout

## 📁 Project Structure
```
music_genre_classification/
├── data/
│   ├── splits/              # Train/val/test splits
│   └── genre_mapping.json   # Genre label mapping
├── src/
│   ├── data/
│   │   ├── dataset.py       # PyTorch Dataset and DataLoaders
│   │   └── feature_extraction.py  # Audio feature extraction
│   ├── models/
│   │   ├── improved_cnn_models.py  # Final CNN architectures
│   │   └── trainer.py       # Training utilities
│   └── utils/
│       └── visualization.py # Plotting and analysis
├── models/                  # Saved model checkpoints (not in repo)
├── results/                 # Evaluation results
├── train_improved.py        # Training script
├── evaluate_ensemble.py     # Ensemble evaluation
├── optimize_ensemble_weights.py  # Weight optimization
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- PyTorch 2.0+
- 16GB RAM recommended
- GPU optional (MPS for Apple Silicon, CUDA for NVIDIA)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download GTZAN Dataset
```bash
# Download from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# Extract to data/genres/ directory
# Structure should be: data/genres/blues/, data/genres/classical/, etc.
```

### Extract Features
```bash
# Extract all audio features (MFCC, Mel-spectrogram, Chroma)
python src/data/feature_extraction.py
```

### Train Models
```bash
# Train individual models
python train_improved.py --feature mfcc --epochs 100
python train_improved.py --feature melspec --epochs 100
python train_improved.py --feature chroma --epochs 100
```

### Evaluate Ensemble
```bash
# Optimize ensemble weights and evaluate
python optimize_ensemble_weights.py
```

## 📈 Training Details

### Hyperparameters

**MFCC CNN:**
- Blocks: 3
- Filters: [128, 256, 512]
- Learning rate: 0.0005
- Batch size: 64
- Dropout: 0.5

**Mel-Spectrogram CNN:**
- Blocks: 4
- Filters: [64, 128, 256, 512]
- Learning rate: 0.0008
- Batch size: 48
- Dropout: 0.5

**Chroma CNN:**
- Blocks: 3
- Filters: [160, 320, 640]
- Learning rate: 0.0005
- Batch size: 80
- Dropout: 0.4

### Training Strategy

- Optimizer: Adam with weight decay (0.0001)
- Learning rate scheduling: ReduceLROnPlateau
- Early stopping: Patience of 15 epochs
- Data split: 70% train, 15% validation, 15% test
- Training time: ~30-40 minutes per model on M4

## 🔬 Technical Highlights

### Innovations & Optimizations

1. **Feature Diversity**: Three complementary audio representations
2. **Adaptive Architecture**: Depth varies by input size (MFCC: 3 blocks, MelSpec: 4 blocks)
3. **Weighted Ensemble**: Optimized through grid search (400+ combinations tested)
4. **Hardware Optimization**: MPS backend for Apple Silicon GPU acceleration

### Challenges Solved

1. **Tensor Dimension Collapse**: Prevented zero-sized tensors with depth constraints
2. **Python 3.13 Multiprocessing**: Resolved DataLoader compatibility issues
3. **Overfitting**: Applied progressive dropout and early stopping
4. **Feature Selection**: Pivoted from DWT (22% accuracy) to Chroma (44%)

## 📊 Experimental Results

### Improvement Timeline

| Phase | Best Individual | Ensemble | Key Change |
|-------|----------------|----------|------------|
| Baseline | 55% | 57% | Initial architecture |
| Improved CNNs | 71% | 68% | +VGG blocks, +depth |
| Weight Optimization | 71% | 68.67% | +Optimized voting |

### Confusion Analysis

Most common misclassifications:
- Rock → Metal (12%)
- Metal → Rock (10%)
- Blues → Jazz (7%)
- Pop → Disco (7%)

These confusions reflect genuine musical ambiguities rather than pure model errors.

## 🔮 Future Improvements

**Quick Wins (2-4 hours):**
- Focal loss for hard genres (Rock, Country)
- Remove underperforming Chroma, use MFCC+MelSpec
- Extended training (150-200 epochs)

**Medium Effort (1-2 days):**
- Attention mechanisms in MelSpec CNN
- Transfer learning from pre-trained audio models
- Gentler data augmentation (mixup, cutmix)

**Major Undertaking (1 week):**
- Audio Spectrogram Transformer (AST)
- Larger dataset (FMA: 100K tracks)
- Multi-task learning (genre + mood + instrument)

Expected improvement: 75-80% with these enhancements.

## 📚 Research Foundation

Based on comprehensive literature review of 30 research papers (2020-2025):

**Key References:**
- Teng Li (2024): *Optimizing Deep Learning with Black Hole Optimization* - 95.2% on GTZAN
- Yang et al. (2020): *Parallel Recurrent CNN* - 92.0% on GTZAN
- Zhang & Li (2025): *Parallel CNNs with Capuchin Search* - 96.1% on GTZAN

Our approach adapts these methodologies to consumer hardware constraints while maintaining research-grade rigor.

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.0+
- **Audio Processing**: Librosa 0.10+
- **Numerical Computing**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Optimization**: scikit-learn, custom implementations
- **GPU Acceleration**: Metal Performance Shaders (MPS) for Apple Silicon

## �� License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: GTZAN Genre Collection by G. Tzanetakis and P. Cook
- **Course**: Speech Processing (University Course)
- **Hardware**: Mac Mini M4 (10-core CPU, 16GB RAM)
- **Inspiration**: Recent advances in music information retrieval (MIR) research

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/yourusername/music-genre-classification/issues)
- Email: your.email@example.com

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@misc{music-genre-classification-2025,
  author = {Sina Sepehrazar},
  title = {Music Genre Classification with Parallel CNN Ensemble},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/TheBlitzschnell/music-genre-classification}
}
```

---

**⭐ Star this repository if you found it helpful!**
