# Data Module

> Stochastic process models and path generation for American option pricing.
> See [Main README](../../README.md) for repository overview.

---

## Overview

This module implements **6 stochastic process models** for generating underlying asset price paths:

| Model | Class | Key Features |
|-------|-------|--------------|
| **Black-Scholes** | `BlackScholes` | Geometric Brownian Motion with correlation |
| **Heston** | `Heston` | Stochastic volatility, leverage effect |
| **Fractional BM** | `FractionalBrownianMotion` | Long-memory dynamics |
| **Rough Heston** | `RoughHeston` | Rough volatility ($H \approx 0.1$) |
| **Real Data** | `RealDataModel` | Stationary Block Bootstrap from Yahoo Finance |
| **User Data** | `UserDataModel` | Custom CSV data with block bootstrap |

Additionally, the module provides infrastructure for **path storage and reuse** via HDF5 files.

---

## Directory Structure

```
data/
├── stock_model.py          # GBM, Heston, FBM, Rough Heston implementations
├── real_data.py            # Stationary Block Bootstrap from market data
├── user_data_model.py      # Custom CSV data model
├── path_storage.py         # HDF5 path management
├── stored_paths/           # Pre-generated path storage
│   └── README.md           # Path storage documentation
└── user_data/              # User-provided CSV data
    └── README.md           # User data documentation
```

---

## Stochastic Process Models

### 1. Black-Scholes (Geometric Brownian Motion)

**Location:** `stock_model.py`

The foundational model under the Black-Scholes framework. Asset price $S_t$ evolves according to:

$$dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t$$

with solution:

$$S_t = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$$

**Multi-Asset Extension:** For $d$ assets with correlation matrix $\rho$:

$$dS^i_t = (r - q_i) S^i_t \, dt + \sigma_i S^i_t \, dW^i_t, \quad d\langle W^i, W^j \rangle_t = \rho_{ij} \, dt$$

Correlated increments are generated via Cholesky decomposition: $\Delta W = L Z \sqrt{\Delta t}$ where $\rho = LL^\top$.

**Usage:**

```python
from optimal_stopping.data.stock_model import BlackScholes

model = BlackScholes(
    drift=0.05,              # Risk-free rate r
    volatility=0.2,          # Volatility sigma
    nb_stocks=50,            # Number of assets d
    nb_paths=1000000,        # Monte Carlo paths m
    nb_dates=100,            # Exercise dates N
    spot=100,                # Initial price S_0
    maturity=1.0,            # Time to maturity T (years)
    dividend=0.0,            # Dividend yield q
    correlation=0.3          # Cross-asset correlation rho
)

# Generate paths: shape (nb_paths, nb_stocks, nb_dates+1)
stock_paths = model.generate_paths()
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drift` | `float` | 0.05 | Risk-free rate $r$ |
| `volatility` | `float` | 0.2 | Volatility $\sigma$ |
| `nb_stocks` | `int` | 1 | Number of assets $d$ |
| `nb_paths` | `int` | 10000 | Monte Carlo paths $m$ |
| `nb_dates` | `int` | 50 | Exercise dates $N$ |
| `spot` | `float` | 100 | Initial price $S_0$ |
| `maturity` | `float` | 1.0 | Time to maturity $T$ |
| `dividend` | `float` | 0.0 | Dividend yield $q$ |
| `correlation` | `float` | 0.0 | Cross-asset correlation $\rho$ |

---

### 2. Heston Stochastic Volatility Model

**Location:** `stock_model.py`

Captures the volatility smile through mean-reverting stochastic variance:

$$dS_t = (r - q) S_t \, dt + \sqrt{v_t} S_t \, dW^S_t$$
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW^v_t$$

where $d\langle W^S, W^v \rangle_t = \rho \, dt$ (leverage effect).

**Discretization:** Full Truncation Euler-Maruyama scheme ensuring positivity:

$$v_{t+\Delta t} = v_t + \kappa(\theta - v_t^+) \Delta t + \xi \sqrt{v_t^+} \Delta W^v_t$$

**Usage:**

```python
from optimal_stopping.data.stock_model import Heston

model = Heston(
    drift=0.05,              # Risk-free rate r
    volatility=0.2,          # Initial volatility sqrt(v_0)
    mean=0.04,               # Long-run variance theta
    speed=2.0,               # Mean reversion speed kappa
    vol_of_vol=0.3,          # Vol of vol xi
    correlation=-0.7,        # Leverage effect rho
    nb_stocks=1,
    nb_paths=100000,
    nb_dates=50,
    spot=100,
    maturity=1.0
)

# Returns tuple: (stock_paths, variance_paths)
stock_paths, var_paths = model.generate_paths()
```

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean` | `float` | 0.04 | Long-run variance $\theta$ |
| `speed` | `float` | 2.0 | Mean reversion speed $\kappa$ |
| `vol_of_vol` | `float` | 0.3 | Volatility of volatility $\xi$ |

**Note:** For Heston, the state space is augmented to $(S_t, v_t)$, and algorithms receive both price and variance paths.

---

### 3. Fractional Brownian Motion

**Location:** `stock_model.py`

Models long-range dependence through the Hurst parameter $H \in (0, 1)$:

$$S_{t_{k+1}} = S_{t_k} \exp\left[(r - q) \Delta t + \sigma (B^H_{t_{k+1}} - B^H_{t_k})\right]$$

where $B^H_t$ is fractional Brownian motion with covariance:

$$\mathbb{E}[B^H_t B^H_s] = \frac{1}{2}\left(|t|^{2H} + |s|^{2H} - |t-s|^{2H}\right)$$

**Hurst Parameter Interpretation:**
- $H < 0.5$: Mean-reverting (roughness)
- $H = 0.5$: Standard Brownian motion
- $H > 0.5$: Trending/persistent

**Usage:**

```python
from optimal_stopping.data.stock_model import FractionalBrownianMotion

model = FractionalBrownianMotion(
    hurst=0.3,               # Hurst parameter H
    drift=0.05,
    volatility=0.2,
    nb_stocks=5,
    nb_paths=100000,
    nb_dates=50,
    spot=100,
    maturity=1.0
)

stock_paths = model.generate_paths()
```

---

### 4. Rough Heston Model

**Location:** `stock_model.py`

Combines stochastic volatility with rough fractional dynamics ($H \approx 0.1$):

$$dS_t = (r - q) S_t \, dt + \sqrt{v_t} S_t \, dW^S_t$$
$$v_t = v_0 + \frac{1}{\Gamma(H + \frac{1}{2})} \int_0^t (t-s)^{H-\frac{1}{2}} \left[\kappa(\theta - v_s) \, ds + \xi \sqrt{v_s} \, dW^v_s\right]$$

The singular kernel $(t-s)^{H-1/2}$ introduces long memory and roughness.

**Usage:**

```python
from optimal_stopping.data.stock_model import RoughHeston

model = RoughHeston(
    hurst=0.1,               # Roughness parameter H
    drift=0.02,
    volatility=0.2,
    mean=0.3,                # Long-run variance theta
    speed=0.15,              # Mean reversion kappa
    vol_of_vol=0.3,
    correlation=-0.7,
    nb_stocks=5,
    nb_paths=100000,
    nb_dates=20,
    spot=100,
    maturity=0.5
)

stock_paths, var_paths = model.generate_paths()
```

---

### 5. Real Data Model (Stationary Block Bootstrap)

**Location:** `real_data.py`

Generates synthetic price paths from historical market data using the Stationary Block Bootstrap [Politis & Romano, 1994], preserving empirical features:
- Autocorrelation structure
- Volatility clustering
- Fat tails
- Cross-asset correlations

**Data Source:** Yahoo Finance via `yfinance`

**Usage:**

```python
from optimal_stopping.data.real_data import RealDataModel

model = RealDataModel(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],  # Up to 250 S&P 500 stocks
    nb_paths=100000,
    nb_dates=50,
    maturity=0.5,
    start_date='2010-01-01',
    end_date='2024-01-01',
    exclude_crisis=True       # Exclude 2008, 2020 crisis periods
)

stock_paths = model.generate_paths()
```

**Key Features:**
- Automatic optimal block length selection [Patton, Politis & White, 2009]
- Risk-neutral drift adjustment (Empirical Martingale Correction)
- Support for up to 250 S&P 500 stocks with 15+ years history

**Risk-Neutral Adjustment:**

Historical returns $\tilde{r}^i_k$ are shifted to ensure risk-neutral drift:

$$S^i_{t_n} = S^i_0 \exp\left[\sum_{k=1}^n \left(\tilde{r}^i_k - \hat{\mu}_i + (r - q_i - \frac{1}{2}\hat{\sigma}^2_i) \Delta t\right)\right]$$

---

### 6. User Data Model

**Location:** `user_data_model.py`

Allows custom CSV data with the same block bootstrap methodology.

**Usage:**

```python
from optimal_stopping.data.user_data_model import UserDataModel

model = UserDataModel(
    user_data_file='my_stocks.csv',    # Relative to data/user_data/
    nb_paths=100000,
    nb_dates=50,
    maturity=0.5
)

stock_paths = model.generate_paths()
```

**CSV Format:**
```csv
Date,STOCK1,STOCK2,STOCK3
2020-01-02,100.5,50.2,75.8
2020-01-03,101.2,50.8,76.1
...
```

See `data/user_data/README.md` for detailed formatting requirements.

---

## Path Storage System

### Overview

Pre-generated paths can be stored and reused to:
- Ensure identical path sets across algorithm comparisons
- Avoid redundant computation for repeated experiments
- Enable exact reproducibility

**Storage Format:** HDF5 with GZIP compression

### Storing Paths

```bash
# Store paths via CLI
python -m optimal_stopping.data.store_paths \
    --model=BlackScholes \
    --nb_stocks=50 \
    --nb_paths=1000000 \
    --nb_dates=100 \
    --drift=0.05 \
    --volatility=0.2 \
    --spot=100 \
    --maturity=1.0
```

### Using Stored Paths

```python
from optimal_stopping.data.stored_model import StoredPathsModel

model = StoredPathsModel(
    storage_id='BlackScholes_50_1000000_100_1701234567',
    nb_paths=500000,         # Use subset of stored paths
    nb_stocks=25             # Use subset of stocks
)

stock_paths = model.generate_paths()
```

### Managing Stored Paths

```bash
# List all stored path sets
python -m optimal_stopping.data.store_paths --list

# Delete a stored path set
python -m optimal_stopping.data.store_paths --delete=storage_id
```

See `data/stored_paths/README.md` for comprehensive documentation.

---

## Configuration via `configs.py`

Data models are configured through the experiment configuration system:

```python
# In optimal_stopping/run/configs.py
from dataclasses import dataclass

@dataclass
class my_experiment(_DefaultConfig):
    # Model selection
    stock_models: tuple = ('BlackScholes',)

    # Common parameters
    nb_stocks: tuple = (5, 25, 50)
    nb_paths: tuple = (1000000,)
    nb_dates: tuple = (100,)
    maturities: tuple = (1.0,)
    spots: tuple = (100,)

    # Market parameters
    drift: tuple = (0.05,)
    volatilities: tuple = (0.2,)
    dividends: tuple = (0.0,)
    correlation: tuple = (0.3,)

    # Model-specific parameters
    # Heston
    mean: tuple = (0.04,)           # Long-run variance theta
    speed: tuple = (2.0,)           # Mean reversion kappa

    # Fractional/Rough
    hurst: tuple = (0.1,)           # Hurst parameter H
```

### Model-Specific Configurations

```python
# Black-Scholes with correlation
bs_correlated = _DefaultConfig(
    stock_models=('BlackScholes',),
    nb_stocks=(50,),
    correlation=(0.3,),
    drift=(0.05,),
    volatilities=(0.2,)
)

# Heston stochastic volatility
heston_config = _DefaultConfig(
    stock_models=('Heston',),
    nb_stocks=(1,),
    drift=(0.05,),
    volatilities=(0.2,),
    mean=(0.04,),
    speed=(2.0,),
    correlation=(-0.7,)           # Leverage effect
)

# Rough Heston
rough_heston = _DefaultConfig(
    stock_models=('RoughHeston',),
    hurst=(0.1,),
    mean=(0.3,),
    speed=(0.15,),
    correlation=(-0.7,)
)

# Real market data
real_data_config = _DefaultConfig(
    stock_models=('RealData',),
    nb_stocks=(25,),
    nb_dates=(50,)
)
```

---

## Output Format

All models return paths with shape `(nb_paths, nb_stocks, nb_dates+1)`:

```python
stock_paths = model.generate_paths()
# stock_paths.shape = (1000000, 50, 101)
#                      └── paths  └── stocks  └── time steps (t=0 to t=N)
```

For models with variance (Heston, Rough Heston):

```python
stock_paths, var_paths = model.generate_paths()
# stock_paths.shape = (nb_paths, nb_stocks, nb_dates+1)
# var_paths.shape = (nb_paths, nb_stocks, nb_dates+1)
```

---

## Pre-Computed Datasets

The thesis experiments use pre-computed path datasets available at:
[Google Drive Link](https://drive.google.com/drive/folders/thesis-datasets)

| Dataset ID | Model | $d$ | $m$ | $N$ | Parameters |
|------------|-------|-----|-----|-----|------------|
| `BS_1.h5` | BlackScholes | 1 | 8M | 100 | $r=0.08$, $\sigma=0.2$ |
| `BS_2.h5` | BlackScholes | 2 | 8M | 100 | $r=0.08$, $\sigma=0.2$ |
| `BS_7.h5` | BlackScholes | 7 | 14M | 100 | $r=0.08$, $\sigma=0.2$ |
| `BS_50.h5` | BlackScholes | 50 | 10M | 100 | $r=0.08$, $\sigma=0.2$ |
| `BS_500.h5` | BlackScholes | 500 | 10M | 100 | $r=0.08$, $\sigma=0.2$ |
| `RH_5.h5` | RoughHeston | 5 | 10M | 20 | $H=0.75$, $\kappa=0.15$ |
| `SBB_25.h5` | RealData | 25 | 10M | 20 | Historical S&P 500 |

---

## References

- [Black & Scholes, 1973] "The Pricing of Options and Corporate Liabilities"
- [Heston, 1993] "A Closed-Form Solution for Options with Stochastic Volatility"
- [Politis & Romano, 1994] "The Stationary Bootstrap"
- [Gatheral et al., 2018] "Volatility is Rough"
- [El Euch & Rosenbaum, 2019] "The Characteristic Function of Rough Heston Models"
