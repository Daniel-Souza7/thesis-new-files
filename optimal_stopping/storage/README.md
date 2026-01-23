# Path Storage & Data Management

This module provides utilities for generating, storing, and loading Monte Carlo paths for reproducible experiments. Pre-computed paths enable:

1. **Reproducibility**: Identical paths across algorithm comparisons
2. **Performance**: Skip expensive path generation (especially for RealData/RoughHeston)
3. **Memory efficiency**: HDF5 memory-mapping loads only requested data

## Module Structure

```
storage/
├── __init__.py          # Module exports
├── path_storage.py      # Core storage functions (store_paths, load_paths)
├── stored_model.py      # StoredPathsModel class
├── store_paths.py       # CLI tool for path generation
├── user_data_model.py   # UserDataModel for custom CSV data
└── stored_paths/        # HDF5 storage directory (auto-created)
```

## Quick Start

### 1. Store Paths (One-Time)

```python
from optimal_stopping.storage.path_storage import store_paths

# Store BlackScholes paths for d=50 stocks
storage_id = store_paths(
    stock_model='BlackScholes',
    nb_stocks=50,
    nb_paths=10000000,      # 10M paths
    nb_dates=100,
    maturity=1.0,
    spot=100,
    drift=0.08,
    volatility=0.2
)
# Output: Use in config: stock_models=['BlackScholesStored1737654321123']
```

### 2. Use Stored Paths (Many Times)

```python
from optimal_stopping.storage import StoredPathsModel

# Load pre-computed paths
model = StoredPathsModel(
    base_model='BlackScholes',
    storage_id='1737654321123',
    nb_stocks=50,
    nb_paths=1000000,       # Use subset
    nb_dates=100,
    maturity=1.0,
    spot=100
)

# Generate paths (loads from disk)
paths, _ = model.generate_paths()
```

### 3. Use in Experiment Configs

```python
# In experiments/configs/my_experiment.py
my_config = {
    'stock_models': ['BlackScholesStored1737654321123'],
    'nb_stocks': [50],
    'nb_paths': [1000000],
    # ... other parameters
}
```

---

## Core Components

### `store_paths()`

Generate and store paths to disk in HDF5 format.

```python
from optimal_stopping.storage.path_storage import store_paths

storage_id = store_paths(
    stock_model: str,          # Model name (e.g., 'BlackScholes', 'RealData')
    nb_stocks: int,            # Number of assets d
    nb_paths: int,             # Number of Monte Carlo paths m
    nb_dates: int,             # Number of time steps N
    maturity: float,           # Time to maturity T
    spot: float = 100.0,       # Initial price S_0
    custom_id: str = None,     # Custom ID (default: timestamp)
    **model_params             # Model-specific parameters
)
```

**Model-specific parameters:**

| Model | Parameters |
|-------|------------|
| BlackScholes | `drift`, `volatility`, `dividend`, `correlation` |
| Heston | `drift`, `volatility`, `mean`, `speed`, `correlation` |
| RoughHeston | `drift`, `volatility`, `mean`, `speed`, `correlation`, `hurst` |
| RealData | `tickers`, `start_date`, `end_date`, `exclude_crisis` |

### `load_paths()`

Load stored paths with validation.

```python
from optimal_stopping.storage.path_storage import load_paths

paths, variance_paths, metadata = load_paths(
    stock_model='BlackScholes',
    storage_id='1737654321123',
    nb_stocks=50,          # Must be ≤ stored
    nb_paths=1000000,      # Must be ≤ stored
    nb_dates=100,          # Must match stored
    maturity=1.0,          # Must match stored
    spot=110               # Can differ (paths rescaled)
)
```

**Validation rules:**

| Parameter | Rule |
|-----------|------|
| `nb_dates` | Must match stored |
| `maturity` | Must match stored |
| `nb_stocks` | Can be ≤ stored (subset) |
| `nb_paths` | Can be ≤ stored (subset) |
| `spot` | Can differ (paths rescaled by `spot/stored_spot`) |

### `list_stored_paths()`

List all stored path files with metadata.

```python
from optimal_stopping.storage.path_storage import list_stored_paths

stored = list_stored_paths()
# Output:
# 📚 Found 3 stored path file(s):
#
#   BlackScholesStored1737654321123
#     Model: BlackScholes
#     Stocks: 50, Paths: 10,000,000, Dates: 100
#     Maturity: 1.0 years, Spot: 100
#     Size: 234.5 MB
#     Created: 2025-01-23 15:30:00
```

### `delete_stored_paths()`

Delete a stored path file.

```python
from optimal_stopping.storage.path_storage import delete_stored_paths

delete_stored_paths('BlackScholesStored1737654321123')
# Output: 🗑️  Deleted: BlackScholesStored1737654321123
```

---

## `StoredPathsModel`

A `Model` wrapper that loads paths from storage instead of generating them.

```python
from optimal_stopping.storage import StoredPathsModel

model = StoredPathsModel(
    base_model='RealData',
    storage_id='1737654321123',
    nb_stocks=25,
    nb_paths=500000,
    nb_dates=20,
    maturity=0.5,
    spot=100,
    start_index=0           # Sliding window offset (for train/eval split)
)
```

### Sliding Window for Train/Eval Split

Use `start_index` to access different portions of stored paths:

```python
# Training: first 500K paths
model_train = StoredPathsModel(
    base_model='BlackScholes',
    storage_id='123',
    nb_paths=500000,
    start_index=0,           # Paths [0, 500000)
    ...
)

# Evaluation: next 500K paths
model_eval = StoredPathsModel(
    base_model='BlackScholes',
    storage_id='123',
    nb_paths=500000,
    start_index=500000,      # Paths [500000, 1000000)
    ...
)
```

---

## `UserDataModel`

Load custom price/return data from CSV files.

### CSV Format

```csv
date,ticker,price
2020-01-01,AAPL,300.00
2020-01-01,MSFT,160.00
2020-01-02,AAPL,302.50
2020-01-02,MSFT,161.20
```

### Usage

```python
from optimal_stopping.storage import UserDataModel

model = UserDataModel(
    data_file='my_data.csv',
    data_folder='data/user_data/',
    tickers=['AAPL', 'MSFT'],
    value_type='price',          # or 'return'
    drift_override=0.05,         # Override empirical (None = use empirical)
    volatility_override=0.2,     # Override empirical (None = use empirical)
    nb_stocks=2,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)

paths, _ = model.generate_paths()
```

### Features

- **Stationary Block Bootstrap**: Same methodology as `RealDataModel`
- **Automatic block length**: Estimated from autocorrelation decay
- **Drift/volatility control**: Use empirical or override
- **Multiple tickers**: Preserves cross-asset correlations

---

## CLI Tool

The `store_paths.py` CLI tool provides command-line access to storage functions.

### Store Paths

```bash
# BlackScholes
python -m optimal_stopping.storage.store_paths \
    --stock_model=BlackScholes \
    --nb_stocks=50 \
    --nb_paths=10000000 \
    --nb_dates=100 \
    --maturity=1.0 \
    --drift=0.08 \
    --volatility=0.2

# RoughHeston
python -m optimal_stopping.storage.store_paths \
    --stock_model=RoughHeston \
    --nb_stocks=5 \
    --nb_paths=1000000 \
    --nb_dates=20 \
    --maturity=0.5 \
    --drift=0.02 \
    --volatility=0.3 \
    --mean=0.04 \
    --speed=2.0 \
    --correlation=-0.7 \
    --hurst=0.1

# RealData
python -m optimal_stopping.storage.store_paths \
    --stock_model=RealData \
    --nb_stocks=25 \
    --nb_paths=1000000 \
    --nb_dates=20 \
    --maturity=0.5 \
    --start_date=2010-01-01 \
    --end_date=2024-01-01 \
    --drift=None \
    --volatility=None
```

### List Stored Paths

```bash
python -m optimal_stopping.storage.store_paths --list
```

### Delete Stored Paths

```bash
python -m optimal_stopping.storage.store_paths --delete=BlackScholesStored1737654321123
```

---

## HDF5 File Format

Paths are stored in HDF5 format with GZIP compression:

```
file.h5
├── paths                  # (nb_paths, nb_stocks, nb_dates+1) float32
├── variance_paths         # (nb_paths, nb_stocks, nb_dates+1) float32 [optional]
└── [attributes]           # Metadata
    ├── stock_model        # e.g., "BlackScholes"
    ├── storage_id         # e.g., "1737654321123"
    ├── nb_stocks          # 50
    ├── nb_paths           # 10000000
    ├── nb_dates           # 100
    ├── maturity           # 1.0
    ├── spot               # 100
    ├── timestamp          # Unix timestamp
    ├── creation_date      # "2025-01-23 15:30:00"
    └── param_*            # Model-specific parameters
```

### Memory Efficiency

- **Lazy loading**: HDF5 memory-mapping loads only requested data
- **Subsetting**: Request fewer stocks/paths than stored
- **Compression**: GZIP level 1 (fast compression, ~3x size reduction)
- **Chunking**: Auto-chunked for efficient partial reads

---

## Dynamic Model Registration

Stored models are automatically registered in `STOCK_MODELS`:

```python
from optimal_stopping.models.stock_model import STOCK_MODELS

# After storing paths, the model is auto-registered
print('BlackScholesStored1737654321123' in STOCK_MODELS)  # True

# Can be used like any other model
ModelClass = STOCK_MODELS['BlackScholesStored1737654321123']
model = ModelClass(nb_stocks=50, nb_paths=100000, ...)
```

This enables seamless integration with experiment configurations:

```python
# In config file
my_config = {
    'stock_models': ['BlackScholesStored1737654321123'],
    ...
}
```

---

## Best Practices

### 1. Store 2x Paths for Train/Eval Split

The CLI automatically doubles the requested path count:

```bash
# Requests 1M paths, stores 2M
python -m optimal_stopping.storage.store_paths \
    --nb_paths=1000000 ...
```

Then use `start_index` for distinct train/eval sets.

### 2. Use `nb_dates` and `maturity` Consistently

These must match exactly between storage and loading. Plan your experiments before storing.

### 3. Spot Rescaling for Moneyness Studies

Store paths at `spot=100`, then rescale at load time:

```python
# Stored at spot=100
model_itm = StoredPathsModel(..., spot=90)    # In-the-money
model_atm = StoredPathsModel(..., spot=100)   # At-the-money
model_otm = StoredPathsModel(..., spot=110)   # Out-of-the-money
```

### 4. Storage Location

Default: `optimal_stopping/storage/stored_paths/`

To change, modify `STORAGE_DIR` in `path_storage.py`.

---

## References

- HDF5 for Python (h5py): https://docs.h5py.org/
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303-1313.
