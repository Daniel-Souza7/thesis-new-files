# Pre-computed Path Datasets

This directory contains pre-computed Monte Carlo path datasets for reproducible experiments.

## Directory Structure

```
data/
├── stored_paths/          # HDF5 path files
│   ├── BS_1.h5            # BlackScholes, d=1
│   ├── BS_2.h5            # BlackScholes, d=2
│   ├── BS_7.h5            # BlackScholes, d=7
│   ├── BS_50.h5           # BlackScholes, d=50
│   ├── BS_500.h5          # BlackScholes, d=500
│   ├── RH_5.h5            # RoughHeston, d=5
│   ├── SBB_25.h5          # Real data (SBB), d=25
│   └── ...
└── README.md              # This file
```

## Dataset Specifications

| File | Model | d (stocks) | m (paths) | N (dates) | T (mat.) | r (drift) | σ (vol.) |
|------|-------|------------|-----------|-----------|----------|-----------|----------|
| BS_1.h5 | BlackScholes | 1 | 8.0M | 100 | 1.0 | 0.08 | 0.20 |
| BS_2.h5 | BlackScholes | 2 | 8.0M | 100 | 1.0 | 0.08 | 0.20 |
| BS_7.h5 | BlackScholes | 7 | 14.0M | 100 | 1.0 | 0.08 | 0.20 |
| BS_50.h5 | BlackScholes | 50 | 10.0M | 100 | 1.0 | 0.08 | 0.20 |
| BS_500.h5 | BlackScholes | 500 | 10.0M | 100 | 1.0 | 0.08 | 0.20 |
| RH_5.h5 | RoughHeston | 5 | 10.0M | 20 | 0.5 | 0.02 | - |
| SBB_25.h5 | RealData | 25 | 10.0M | 20 | 0.5 | 0.02 | Hist. |

## Usage

### Loading Pre-computed Paths

```python
from optimal_stopping.storage import StoredPathsModel

# Load BlackScholes d=50 paths
model = StoredPathsModel(
    path_file='data/stored_paths/BS_50.h5',
    nb_paths=1000000  # Use subset of stored paths
)

# Generate paths (loads from file)
paths = model.generate_paths()
print(f"Path shape: {paths.shape}")  # (1000000, 50, 101)
```

### Using with Algorithms

```python
from optimal_stopping.algorithms import RT
from optimal_stopping.payoffs import BasketCall
from optimal_stopping.storage import StoredPathsModel

# Load model
model = StoredPathsModel('data/stored_paths/BS_50.h5', nb_paths=1000000)

# Define payoff
payoff = BasketCall(strike=100)

# Price with RT
rt = RT(model=model, payoff=payoff)
price, time = rt.price()
```

## Generating New Datasets

To generate new path datasets:

```python
from optimal_stopping.storage.store_paths import generate_and_store

# Generate and store BlackScholes paths
generate_and_store(
    model_name='BlackScholes',
    nb_stocks=100,
    nb_paths=10000000,
    nb_dates=100,
    maturity=1.0,
    drift=0.05,
    volatility=0.2,
    output_file='data/stored_paths/BS_100_custom.h5'
)
```

## File Format

Datasets are stored in HDF5 format with GZIP compression (level 4):

```
file.h5
├── paths                 # (nb_paths, nb_stocks, nb_dates+1) float32
├── metadata/
│   ├── model_name        # e.g., "BlackScholes"
│   ├── nb_stocks         # Number of assets
│   ├── nb_paths          # Number of paths
│   ├── nb_dates          # Number of time steps
│   ├── maturity          # Time to maturity
│   ├── drift             # Risk-free rate
│   ├── volatility        # Volatility parameter
│   ├── seed              # Random seed for reproducibility
│   └── created_at        # Timestamp
```

## Downloading Datasets

Pre-computed datasets for thesis experiments are available at:
[Google Drive link - placeholder]

Download and extract to this directory to reproduce exact thesis results.

## Notes

- All paths use float32 precision (sufficient for pricing, 50% memory savings)
- Paths are generated with fixed random seeds for reproducibility
- Large files (>1GB) should be downloaded separately
- Use `nb_paths` parameter to load subsets of large datasets
