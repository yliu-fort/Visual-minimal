# Visual-Minimal

A clean and feature-complete visual classifier training framework: YAML configuration, deterministic training, experiment tracking and logging, unit tests, type checking and code style tools.

![CIFAR10 Classification Example](./examples/cifar10.png)

## Features

- 🎯 **Multi-Architecture Support**: ResNet, EfficientNet, Vision Transformer (ViT)
- 📊 **Complete Experiment Tracking**: TensorBoard and MLflow integration
- 🔧 **Flexible Configuration**: YAML configuration files supporting all training parameters
- 💾 **Smart Checkpointing**: Auto-save, resume training, checkpoint management
- 🎲 **Deterministic Training**: Reproducible experimental results
- 🧪 **Test Coverage**: Complete unit tests and type checking
- 🚀 **One-Click Deployment**: Quick deployment scripts for AWS EC2 GPU instances

## Quick Start

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m unittest
```

### Training Models

```bash
# Train CIFAR-10 classifier with default configuration
python -m visual.train --config configs/default.yaml

# Start TensorBoard to view training progress
tensorboard --logdir runs
```

## Configuration

All training parameters can be configured through the `configs/default.yaml` file:

- **Data Configuration**: Dataset selection, batch size, data augmentation
- **Model Configuration**: Model architecture, pretrained weights, number of classes
- **Training Configuration**: Learning rate, epochs, optimizer settings
- **Tracking Configuration**: TensorBoard and MLflow settings

### Supported Model Architectures

- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0` to `efficientnet_b7`
- **Vision Transformer**: `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`

### Supported Datasets

- **CIFAR-10**: 10-class color image classification
- **MNIST**: Handwritten digit recognition (grayscale images)

## Project Structure

```
Visual-minimal/
├── src/visual/           # Core source code
│   ├── train.py         # Training main program
│   ├── model.py         # Model definitions
│   ├── config.py        # Configuration management
│   ├── image_data.py    # Data loaders
│   ├── logger.py        # Logging
│   ├── checkpointing.py # Checkpoint management
│   └── seed.py          # Random seed setup
├── configs/             # Configuration files
├── scripts/             # Utility scripts
├── tests/               # Unit tests
└── examples/            # Examples and visualizations
```

## Code Quality Tools

### Code Formatting
```bash
black . && isort .
```

### Static Analysis
```bash
ruff check .
```

### Type Checking
```bash
mypy src tests
```

## Dataset Visualization

```bash
python scripts/visualise_datasets.py
```

After running, the following visualization images will be generated in the `examples/figures/` directory:
- gmm.png - Gaussian Mixture Model
- ring.png - Ring distribution
- two_moons.png - Two moons distribution
- concentric.png - Concentric circles distribution
- spiral.png - Spiral distribution
- checkerboard.png - Checkerboard distribution
- pinwheel.png - Pinwheel distribution
- swiss_roll2d.png - Swiss roll distribution

## Checkpoint Management

The framework provides intelligent checkpoint management features:

- **Auto-save**: Automatically save checkpoints by steps or epochs
- **Resume Training**: Support resuming from latest checkpoint or specified checkpoint
- **Checkpoint Cleanup**: Automatically keep the latest N checkpoints
- **Random State Saving**: Ensure completely reproducible training

## Experiment Tracking

### TensorBoard
```bash
tensorboard --logdir runs
```

### MLflow (Optional)
Enable MLflow tracking in the configuration file:
```yaml
tracking:
  mlflow:
    enabled: true
    tracking_uri: "file:./mlruns"
    experiment_name: "visual_cifar10"
```

## One-Click Deployment

### AWS EC2 GPU Instance
```bash
# NVIDIA GPU instance
./scripts/init_aws_ami.sh

# CPU instance
./scripts/init_aws.sh
```

### Vast.ai Deployment
```bash
./scripts/init_vastai.sh
```

## Development Guide

### Adding New Models
1. Add support for new model architectures in the `VisualClassifier` class in `src/visual/model.py`
2. Update model name options in the configuration file

### Adding New Datasets
1. Implement new data loader functions in `src/visual/image_data.py`
2. Add dataset selection logic in `src/visual/train.py`
3. Update the `DataCfg` class in the configuration file

### Running Tests
```bash
# Run all tests
python -m unittest discover tests

# Run specific tests
python -m unittest tests.test_config
python -m unittest tests.test_seed
```

## Dependencies

Main dependencies:
- `torch>=2.2` - PyTorch deep learning framework
- `torchvision>=0.17` - Computer vision tools
- `timm>=1.0.7` - Pretrained model library
- `tensorboard>=2.16` - Experiment tracking
- `mlflow>=2.12` - Model management (optional)

Development tools:
- `black>=24.3` - Code formatting
- `ruff>=0.4` - Code linting
- `isort>=5.13` - Import sorting
- `mypy>=1.10` - Type checking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome! Please ensure:

1. Code passes all tests
2. Follow the project's code style
3. Add appropriate documentation and tests

## Changelog

- **v1.0.0**: Initial version, supporting CIFAR-10 and MNIST datasets, multiple model architectures, complete experiment tracking functionality