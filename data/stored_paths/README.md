# Stored Paths

This directory stores pre-generated stock paths for reuse across experiments. This feature provides:

- **Performance**: Skip expensive path generation (especially for RealData)
- **Reproducibility**: Use identical paths across multiple algorithm runs
- **Flexibility**: Store once, reuse with different strikes, payoffs, algorithms

## Quick Start

### 1. Store Paths

Generate and store paths using the CLI:

```bash
# Store RealData paths with empirical drift/volatility
python -m optimal_stopping.data.store_paths \
    --stock_model=RealData \
    --nb_stocks=50 \
    --nb_paths=100000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --drift=None \
    --volatilities=None

# Output will show the storage ID, e.g.:
# ✅ Success! Storage ID: 1732000000123
# 📝 Use in config: stock_models=['RealDataStored1732000000123']
```

### 2. Use Stored Paths in Configs

Reference the stored model in your config:

```python
test_config = _DefaultConfig(
    stock_models=['RealDataStored1732000000123'],  # Use stored ID
    nb_stocks=[10],      # Can use subset (≤50)
    nb_paths=[50000],    # Can use subset (≤100000)
    nb_dates=[252],      # Must match exactly
    maturity=[1.0],      # Must match exactly
    spots=[90, 100, 110],  # Can rescale!
    strikes=[100, 110],
    payoffs=['BasketCall', 'MaxCall'],
    algos=['RLSM', 'RFQI'],
    ...
)
```

## CLI Commands

### Store Paths

```bash
# RealData with empirical values
python -m optimal_stopping.data.store_paths \
    --stock_model=RealData \
    --nb_stocks=100 \
    --nb_paths=200000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --spot=100 \
    --drift=None \
    --volatilities=None

# RealData with specific tickers
python -m optimal_stopping.data.store_paths \
    --stock_model=RealData \
    --nb_stocks=5 \
    --nb_paths=50000 \
    --nb_dates=100 \
    --maturity=0.5 \
    --tickers AAPL MSFT GOOGL AMZN NVDA \
    --drift=None \
    --volatilities=None

# BlackScholes with correlation
python -m optimal_stopping.data.store_paths \
    --stock_model=BlackScholes \
    --nb_stocks=20 \
    --nb_paths=100000 \
    --nb_dates=100 \
    --maturity=0.5 \
    --drift=0.05 \
    --volatility=0.2 \
    --correlation=-0.3

# Heston model
python -m optimal_stopping.data.store_paths \
    --stock_model=Heston \
    --nb_stocks=10 \
    --nb_paths=100000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --drift=0.05 \
    --volatility=0.2
```

### List Stored Paths

```bash
python -m optimal_stopping.data.store_paths --list
```

Output:
```
📚 Found 3 stored path file(s):

  RealDataStored1732000000123
    Model: RealData
    Stocks: 50, Paths: 100,000, Dates: 252
    Maturity: 1.0 years, Spot: 100
    Size: 487.3 MB
    Created: 2024-11-17 12:30:45

  BlackScholesStored1732000001234
    Model: BlackScholes
    Stocks: 20, Paths: 100,000, Dates: 100
    Maturity: 0.5 years, Spot: 100
    Size: 195.2 MB
    Created: 2024-11-17 13:15:22
```

### Delete Stored Paths

```bash
python -m optimal_stopping.data.store_paths --delete=RealDataStored1732000000123
```

## Python API

You can also use the Python API directly:

```python
from optimal_stopping.data.path_storage import store_paths, list_stored_paths

# Store paths
storage_id = store_paths(
    stock_model='RealData',
    nb_stocks=50,
    nb_paths=100000,
    nb_dates=252,
    maturity=1.0,
    drift=(None,),
    volatilities=(None,),
)

# List stored paths
stored = list_stored_paths(verbose=True)

# Use in model directly
from optimal_stopping.data.stored_model import StoredPathsModel

model = StoredPathsModel(
    base_model='RealData',
    storage_id=storage_id,
    nb_stocks=10,  # Subset
    nb_paths=50000,  # Subset
    nb_dates=252,
    maturity=1.0,
    spot=100,
)

paths, variance_paths = model.generate_paths()
```

## Parameter Matching Rules

When using stored paths, the following rules apply:

### ✅ MUST Match Exactly
- `nb_dates` - Number of time steps
- `maturity` - Maturity in years

### ✅ CAN Use Subset
- `nb_stocks` - Can request ≤ stored (uses first N stocks)
- `nb_paths` - Can request ≤ stored (uses first N paths)

### ✅ CAN Rescale
- `spot` - Paths are automatically rescaled: `paths * (new_spot / stored_spot)`

### ⚠️ Other Parameters
Model-specific parameters (drift, volatility, tickers, etc.) are stored as metadata.
If they differ from what you request, a warning is issued but paths are still used.

## Storage Format

Paths are stored as HDF5 (`.h5`) files with:

- **Compression**: GZIP level 4 (good balance of size/speed)
- **Datasets**:
  - `paths`: Main price paths (nb_paths × nb_stocks × nb_dates+1)
  - `variance_paths`: Variance paths (optional, for Heston models)
- **Metadata**: All generation parameters stored as HDF5 attributes

## Examples

### Example 1: Store and Reuse RealData

```bash
# Step 1: Store paths (run once, takes ~2 minutes)
python -m optimal_stopping.data.store_paths \
    --stock_model=RealData \
    --nb_stocks=100 \
    --nb_paths=200000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --drift=None \
    --volatilities=None

# Output: Storage ID: 1732000000123
```

Then use in config:

```python
# configs.py
test_real_data_stored = _DefaultConfig(
    stock_models=['RealDataStored1732000000123'],
    nb_stocks=[10, 25, 50, 100],  # Try different dimensions
    nb_paths=[100000],  # Subset of stored
    nb_dates=[252],  # Must match
    maturity=[1.0],  # Must match
    spots=[90, 100, 110],  # Will rescale
    strikes=[90, 100, 110],
    payoffs=['BasketCall', 'BasketPut', 'MaxCall', 'MinPut'],
    algos=['RLSM', 'RFQI', 'LSM', 'FQI'],
    ...
)
```

Run multiple times (instant path loading):
```bash
python -m optimal_stopping.run.run_algo --configs=test_real_data_stored
```

### Example 2: Compare Real vs Synthetic

Store both real and synthetic data, then compare:

```bash
# Store RealData
python -m optimal_stopping.data.store_paths \
    --stock_model=RealData \
    --nb_stocks=50 \
    --nb_paths=100000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --drift=None \
    --volatilities=None
# ID: 1732000001111

# Store BlackScholes with matched parameters
python -m optimal_stopping.data.store_paths \
    --stock_model=BlackScholes \
    --nb_stocks=50 \
    --nb_paths=100000 \
    --nb_dates=252 \
    --maturity=1.0 \
    --drift=0.05 \
    --volatility=0.2 \
    --correlation=-0.3
# ID: 1732000002222
```

Config:
```python
test_comparison = _DefaultConfig(
    stock_models=[
        'RealDataStored1732000001111',
        'BlackScholesStored1732000002222',
    ],
    nb_stocks=[10, 25, 50],
    ...
)
```

## Tips

1. **Storage is cheap**: 100k paths × 50 stocks × 252 dates ≈ 500 MB
2. **Generate once, experiment many times**: Change payoffs, strikes, algorithms freely
3. **Reproducibility**: Share storage IDs with collaborators for exact replication
4. **Spot rescaling**: Store at spot=100, use for any spot price
5. **Subset flexibility**: Store large (e.g., 100 stocks, 200k paths), use subsets as needed

## Troubleshooting

### "Incompatible parameters for stored paths"

Check that:
- `nb_dates` matches exactly
- `maturity` matches exactly
- `nb_stocks` ≤ stored
- `nb_paths` ≤ stored

### "Stored paths not found"

Run `python -m optimal_stopping.data.store_paths --list` to see available IDs.

### Storage ID Changed

Storage IDs are timestamps. If you delete and recreate, the ID will change.
Update your configs with the new ID.

## File Location

Stored paths are saved in:
```
optimal_stopping/data/stored_paths/
├── RealData_1732000000123.h5
├── BlackScholes_1732000001234.h5
└── README.md (this file)
```

Each file is named: `{model}_{storage_id}.h5`
