# Installation Guide

This document provides detailed installation instructions for the `optimal_stopping` package.

## Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Latest version recommended
- **Git**: For cloning the repository

## Quick Installation

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/Daniel-Souza7/thesis-new-files.git
cd thesis-new-files

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e ".[full]"
```

### Option 2: Minimal Installation

For core functionality only (without real data, hyperparameter optimization, or TensorFlow):

```bash
pip install -e .
```

## Dependency Groups

The package supports multiple installation profiles:

| Profile | Command | Includes |
|---------|---------|----------|
| **Core** | `pip install -e .` | NumPy, PyTorch, SciPy, Matplotlib, Pandas |
| **Full** | `pip install -e ".[full]"` | Core + TensorFlow, yfinance, Optuna, fbm |
| **Development** | `pip install -e ".[dev]"` | Core + pytest, black, flake8 |

## Verifying Installation

After installation, verify the package is correctly installed:

```python
# Test basic import
import optimal_stopping
print(f"Version: {optimal_stopping.__version__}")

# Test algorithm import
from optimal_stopping.algorithms import RT, RLSM, LSM

# Test payoff import
from optimal_stopping.payoffs import BasketCall, MaxCall

# Test model import
from optimal_stopping.models import BlackScholes, Heston

print("Installation successful!")
```

## Platform-Specific Notes

### Linux/macOS

No additional steps required. Install as described above.

### Windows

If you encounter issues with PyTorch installation:

```bash
# Install PyTorch with CPU support
pip install torch torchvision torchaudio

# Then install the package
pip install -e ".[full]"
```

### GPU Support (Optional)

For CUDA-enabled GPU acceleration with PyTorch:

```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install the package
pip install -e ".[full]"
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'optimal_stopping'`

**Solution**: Ensure you installed with `-e` flag from the repository root:
```bash
cd thesis-new-files
pip install -e .
```

**Issue**: `ImportError: cannot import name 'fbm'`

**Solution**: The `fbm` package is required for Fractional Brownian Motion models:
```bash
pip install fbm
```

**Issue**: TensorFlow installation fails

**Solution**: TensorFlow is optional. Install without it:
```bash
pip install -e .  # Minimal installation without TensorFlow
```

**Issue**: yfinance fails to download data

**Solution**: This is often a network issue. Check your internet connection and firewall settings.

### Getting Help

If you encounter issues:

1. Check that you're using Python 3.8+: `python --version`
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Try installing in a fresh virtual environment
4. Open an issue on GitHub with your error message

## Uninstallation

To uninstall the package:

```bash
pip uninstall optimal_stopping
```

## Next Steps

After installation, see the [README.md](README.md) for usage instructions and examples.
