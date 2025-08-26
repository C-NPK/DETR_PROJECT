# DETR Object Detection Project

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://c-npk.github.io/DETR_PROJECT/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow)](https://huggingface.co/transformers/)

A comprehensive implementation of DETR (Detection Transformer) for object detection using PyTorch and Hugging Face Transformers. This project focuses on training DETR models with the SHHS dataset and includes advanced training techniques such as early stopping and confidence dropout.

## 🌐 Project Website

Visit our [GitHub Pages website](https://c-npk.github.io/DETR_PROJECT/) for comprehensive documentation, setup guides, and usage examples.

## 🎯 Overview

DETR (Detection Transformer) revolutionizes object detection by treating it as a direct set prediction problem. Unlike traditional detection methods, DETR eliminates the need for hand-designed components like anchor generation and non-maximum suppression, instead leveraging the power of transformer architectures.

### Key Features

- 🧠 **Transformer-based Architecture**: Utilizes the power of attention mechanisms for accurate object detection
- 🔧 **Multiple Training Configurations**: Basic and advanced training modes with different optimization strategies
- 📊 **Comprehensive Monitoring**: Built-in validation tracking and performance metrics
- 🛡️ **Regularization Techniques**: Early stopping and confidence dropout for better generalization
- 💾 **Automatic Checkpointing**: Smart model saving based on validation performance
- 📈 **Loss Visualization**: Training and validation loss tracking with NumPy exports

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for training

### Installation

```bash
# Clone the repository
git clone https://github.com/C-NPK/DETR_PROJECT.git
cd DETR_PROJECT

# Install dependencies
pip install torch torchvision transformers pillow numpy
```

### Dataset Setup

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── Images/
│   └── Annotations/
├── val/
│   ├── Images/
│   └── Annotations/
└── test/
    ├── Images/
    └── Annotations/
```

### Training

**Basic Training:**
```bash
python train0.py
```

**Advanced Training with Early Stopping:**
```bash
python train2_earlystop_confdropout.py
```

## 📁 Project Structure

```
DETR_PROJECT/
├── docs/                          # GitHub Pages website
│   ├── index.html                 # Main website file
│   └── assets/                    # CSS, JS, and images
├── scripts/                       # Additional training scripts
├── train0.py                      # Basic DETR training script
├── train2_earlystop_confdropout.py # Advanced training with regularization
└── README.md                      # This file
```

## 🔧 Training Scripts

### train0.py - Basic Training

- Standard DETR training loop
- Best model saving based on validation loss
- Training and validation loss tracking
- Suitable for initial experiments

### train2_earlystop_confdropout.py - Advanced Training

- Early stopping mechanism with patience parameter
- Confidence dropout regularization
- Enhanced validation monitoring
- Production-ready training pipeline

## 📊 Model Outputs

After training, you'll find these files:

**Model Checkpoints:**
- `best_model0.pth` / `best_model2.pth` - Best performing models
- `last_model0.pth` / `last_model2.pth` - Final epoch models

**Training Metrics:**
- `model0_training_losses.npy` - Training loss history
- `model0_validation_losses.npy` - Validation loss history
- `model2_training_losses.npy` - Advanced training losses
- `model2_validation_losses.npy` - Advanced validation losses

## 🏗️ Architecture

This project uses the pre-trained DETR model from Facebook Research:
- **Base Model**: `facebook/detr-resnet-50`
- **Backbone**: ResNet-50
- **Framework**: PyTorch via Hugging Face Transformers

## 📚 Documentation

For detailed documentation, including:
- Step-by-step setup instructions
- Configuration options
- Training strategies
- Performance optimization tips
- Troubleshooting guide

Visit our [comprehensive documentation website](https://c-npk.github.io/DETR_PROJECT/).

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug? [Open an issue](https://github.com/C-NPK/DETR_PROJECT/issues)
2. **Feature Requests**: Have an idea? Share it with us
3. **Pull Requests**: Ready to contribute code? Fork and submit a PR
4. **Documentation**: Help improve our docs and examples

### Development Setup

```bash
# Fork the repository and clone your fork
git clone https://github.com/your-username/DETR_PROJECT.git
cd DETR_PROJECT

# Create a new branch for your feature
git checkout -b feature/your-feature-name

# Make your changes and commit
git commit -m "Add your feature"

# Push to your fork and submit a PR
git push origin feature/your-feature-name
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- [Project Website](https://c-npk.github.io/DETR_PROJECT/)
- [Original DETR Paper](https://arxiv.org/abs/2005.12872)
- [Hugging Face DETR Documentation](https://huggingface.co/docs/transformers/model_doc/detr)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📞 Support

Need help? Here are your options:

- 📖 Check our [documentation website](https://c-npk.github.io/DETR_PROJECT/)
- 🐛 [Open an issue](https://github.com/C-NPK/DETR_PROJECT/issues) for bugs
- 💡 [Start a discussion](https://github.com/C-NPK/DETR_PROJECT/discussions) for questions
- 📧 Contact the maintainers

---

⭐ **Star this repository if you find it helpful!**

Built with ❤️ for the computer vision community.