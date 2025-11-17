# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python research project for pricing American-style options using reinforcement learning and Monte Carlo methods. It implements multiple algorithms for optimal stopping problems with a focus on financial derivatives.

**Key Features:**
- 408 option payoff types (34 base payoffs + 374 barrier variants)
- Multiple pricing algorithms (RLSM, RFQI, LSM, FQI, DOS, NLSM)
- Support for path-dependent and standard options
- Real market data integration via yfinance
- Multiple stochastic models (Black-Scholes, Heston, fractional Brownian motion, rough Heston, real data)
- HDF5-based path storage for memory efficiency
- Telegram notifications for long-running experiments

## Essential Commands

### Running Experiments

```bash
# Run algorithm with a specific configuration
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:ConfigName

# Examples of common configs:
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:_DefaultConfig
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_barrier_convergence
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_stored

# Run with custom flags
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:ConfigName --nb_jobs=8 --path_gen_seed=42
```

### Generating Output

```bash
# Generate Excel reports from results
python -m optimal_stopping.run.write_excel --configs=ConfigName

# Generate figures
python -m optimal_stopping.run.write_figures --config=ConfigName
```

### Testing

```bash
# Run basic payoff tests
python -m optimal_stopping.test_payoffs_basic

# Other test files (phase 1 and phase 2 tests)
python -m optimal_stopping.test_phase1
python -m optimal_stopping.test_phase2
```

### Data Management

```bash
# Store paths to HDF5 for memory efficiency
python -m optimal_stopping.data.store_paths

# Real data model downloads data automatically via yfinance
# Data is cached to avoid re-downloading
```

## Architecture

### Core Structure

```
optimal_stopping/
├── algorithms/          # Pricing algorithms
│   ├── standard/       # Standard option algorithms (RLSM, RFQI, LSM, FQI, DOS, NLSM)
│   └── path_dependent/ # Path-dependent algorithms (SRLSM, SRFQI)
├── data/               # Stochastic models and data management
│   ├── stock_model.py  # Base Model class and Black-Scholes
│   ├── real_data.py    # Real market data with stationary block bootstrap
│   ├── path_storage.py # HDF5 memory-mapped storage
│   └── store_paths.py  # Path generation and storage utilities
├── payoffs/            # 408 option payoff implementations
│   ├── payoff.py       # Base Payoff class with auto-registration
│   ├── barrier_wrapper.py # Handles all 11 barrier types
│   ├── basket_*.py     # Multi-asset options (simple, asian, rank, quantile, range/dispersion)
│   └── single_*.py     # Single-asset options (simple, asian, lookback, quantile, range)
├── run/                # Execution and output
│   ├── configs.py      # Experiment configurations
│   ├── run_algo.py     # Main execution engine
│   ├── write_excel.py  # Excel report generation
│   └── write_figures.py # Figure generation
└── utilities/          # Helper functions for data processing and plotting
```

### Algorithm Categories

**Standard Options** (use RLSM/RFQI):
- BasketCall, BasketPut, GeometricCall, MaxCall, MinPut
- DispersionCall, DispersionPut
- BestOfKCall, WorstOfKPut, RankWeightedBasket
- Simple Call/Put

**Path-Dependent Options** (use SRLSM/SRFQI):
- ALL barrier options (11 types: UO, DO, UI, DI, UODO, UIDI, UIDO, UODI, PTB, StepB, DStepB)
- Lookback options (fixed/floating strike)
- Asian options (fixed/floating strike)
- Range options
- Quantile options

### Key Design Patterns

**1. Payoff Auto-Registration System**

Payoffs automatically register themselves when defined via `__init_subclass__()`:

```python
from optimal_stopping.payoffs import Payoff

class MyPayoff(Payoff):
    abbreviation = "MyPay"
    is_path_dependent = False

    def eval(self, X):
        # X shape: (nb_paths, nb_stocks) for standard
        # X shape: (nb_paths, nb_stocks, nb_dates+1) for path-dependent
        return np.maximum(0, np.sum(X, axis=1) - self.strike)
```

Access via:
```python
from optimal_stopping.payoffs import get_payoff_class
MyPayoff = get_payoff_class('MyPayoff')  # or 'MyPay'
```

**2. Barrier Wrapper Pattern**

One `BarrierPayoff` class wraps ANY base payoff to create barrier variants. The factory function `create_barrier_payoff()` dynamically generates 11 barrier types for each base payoff.

**3. Algorithm Routing by Path-Dependency**

The `run_algo.py` automatically routes to the correct algorithm based on `payoff.is_path_dependent`:
- Path-dependent → Use SRLSM/SRFQI (full path history)
- Standard → Use RLSM/RFQI (current state only)

**4. Memory-Efficient Path Storage**

For large experiments, paths are stored in HDF5 format with memory-mapped access to avoid loading entire datasets into RAM. See `optimal_stopping/data/path_storage.py`.

## Configuration System

Configs are defined as dataclasses in `optimal_stopping/run/configs.py`:

```python
@dataclass
class MyConfig(_DefaultConfig):
    algos: Iterable[str] = ('RLSM', 'RFQI', 'SRLSM', 'SRFQI')
    nb_stocks: Iterable[int] = (5, 10)
    payoffs: Iterable[str] = ('MaxCall', 'UO_BasketCall')
    nb_paths: Iterable[int] = (10000,)
    nb_runs: int = 5
```

**Important Parameters:**
- `alpha`: Quantile level for quantile options (default: 0.95)
- `k`: Number of assets for best-of-k/worst-of-k (default: 2)
- `weights`: Custom weights for rank-weighted options (default: None = auto)
- `barriers`: Barrier level for barrier options
- `step_param1-4`: Bounds for step barrier random walk
- `use_path`: Set to True for storing paths in memory (use HDF5 for large experiments)

### When Adding New Parameters

If you add a parameter to `_DefaultConfig`, you MUST update:
1. `run/configs.py`: Add to `_DefaultConfig` dataclass
2. `run/run_algo.py`: Add to `_CSV_HEADERS` and `_run_algo()` signature
3. `utilities/read_data.py`: Add to `INDEX` list
4. `utilities/filtering.py`: Add to `FILTERS` mapping

## Stock Models

Available models in `optimal_stopping/data/stock_model.py`:
- `BlackScholes`: Geometric Brownian motion
- `Heston`: Stochastic volatility
- `FractionalBlackScholes`: Fractional Brownian motion
- `RoughHeston`: Rough volatility
- `RealDataModel`: Real market data with stationary block bootstrap

**RealDataModel Usage:**
```python
# Use empirical drift/volatility from real data
config = _DefaultConfig(
    stock_models=['RealData'],
    drift=(None,),        # Use historical drift
    volatilities=(None,), # Use historical volatility
)

# Override with specific values
config = _DefaultConfig(
    stock_models=['RealData'],
    drift=(0.05,),       # Force 5% drift
    volatilities=(0.2,), # Force 20% volatility
)
```

## Common Workflows

### Adding a New Payoff

1. Create file `optimal_stopping/payoffs/my_payoff.py`:
```python
from .payoff import Payoff
import numpy as np

class MyNewPayoff(Payoff):
    abbreviation = "MyPay"
    is_path_dependent = False

    def eval(self, X):
        return np.maximum(0, np.sum(X, axis=1) - self.strike)
```

2. Import in `optimal_stopping/payoffs/__init__.py`:
```python
from .my_payoff import MyNewPayoff
_BASE_PAYOFFS.append(MyNewPayoff)  # Auto-generates 11 barrier variants!
```

That's it! The payoff is now available in the registry with all barrier variants.

### Adding a New Algorithm

1. Create file in `optimal_stopping/algorithms/standard/` or `optimal_stopping/algorithms/path_dependent/`
2. Implement the class with methods:
   - `__init__(model, payoff, **kwargs)`
   - `price()` - returns (price, computation_time) tuple
3. Add to `_ALGOS` dict in `run/run_algo.py`

### Running Validation Tests

See `VALIDATION_TEST_PLAN.md` for comprehensive test suite:

```bash
# Run all validation configs
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_barrier_convergence
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_alpha_sensitivity
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_k_sensitivity
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_step_barriers
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_large_basket
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_in_barriers
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_payoff_ordering
```

## Output Structure

Results are saved to:
- `output/metrics_draft/`: CSV files with raw results (timestamped)
- `results/`: Excel reports and figures
- `optimal_stopping/data/stored_paths/`: HDF5 path storage files

CSV columns include: algo, model, payoff, price, duration, comp_time, greeks (delta, gamma, theta, rho, vega), and all config parameters.

## Critical Implementation Notes

### Discount Factor Bug Fix

The `disc_factor()` method in `stock_model.py:40-43` was fixed to use `self.rate` instead of `self.drift`. This is critical for correct pricing.

### Path-Dependency Flag

All algorithms check `payoff.is_path_dependent` to determine:
- How to evaluate payoffs (current state vs. full history)
- Which algorithm variant to use (standard vs. path-dependent)

### Barrier Types

11 barrier types are supported:
- **Knock-Out**: UO, DO, UODO, UODI
- **Knock-In**: UI, DI, UIDI, UIDO
- **Special**: PTB (partial time barrier), StepB (step barrier), DStepB (double step barrier)

Step barriers use cumulative random walk: `B(t) = B(0) + sum(U(step_param1, step_param2))`

### Memory Management

For experiments with >100k paths or >100 stocks:
1. Set `use_path=False` in config
2. Use `store_paths.py` to pre-generate and save paths to HDF5
3. Use `PathStorage` class for memory-mapped access
4. See recent commits on HDF5 optimization for performance

## Dependencies

Install via: `pip install -r requirements.txt`

Key dependencies:
- `numpy`, `scipy`: Numerical computing
- `torch`: Neural networks (for NLSM, DOS)
- `scikit-learn`: Regression utilities
- `pandas`: Data processing
- `h5py`: HDF5 storage
- `yfinance`: Real market data
- `fbm`: Fractional Brownian motion
- `telegram-notifications`: Remote job notifications

## Recent Major Changes

1. **Payoff System Restructure** (see `IMPLEMENTATION_SUMMARY.md`):
   - 408 payoffs implemented (34 base + 374 barrier variants)
   - Auto-registration system
   - Barrier wrapper pattern
   - 7 new parameters added

2. **Algorithm Restructure**:
   - Split into `standard/` and `path_dependent/` directories
   - RLSM/RFQI for standard options
   - SRLSM/SRFQI for path-dependent options
   - Proper path history handling

3. **HDF5 Memory Optimization**:
   - Memory-mapped access instead of loading into RAM
   - Progress feedback during save operations
   - Scalar value handling fixes

4. **Real Data Model**:
   - Stationary block bootstrap
   - Automatic optimal block length selection
   - Crisis period filtering
   - Empirical drift/volatility support
