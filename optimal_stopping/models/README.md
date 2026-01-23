# Stochastic Process Models

This module implements stochastic process models for Monte Carlo path generation in American option pricing. All models generate price paths $\{S_{t_n}\}_{n=0}^N$ used by the pricing algorithms.

## Mathematical Framework

### Continuous-Time SDEs

The underlying asset price $S_t$ evolves according to stochastic differential equations (SDEs) of the general form:

$$dS_t = \mu(S_t, t) dt + \sigma(S_t, t) dW_t$$

where:
- $\mu(S_t, t)$: Drift function
- $\sigma(S_t, t)$: Diffusion (volatility) function
- $W_t$: Standard Brownian motion

For option pricing under the risk-neutral measure $\mathbb{Q}$, the drift becomes $\mu = r - q$ where $r$ is the risk-free rate and $q$ is the dividend yield.

### Discretization

Models discretize the SDE using the Euler-Maruyama scheme with time step $\Delta t = T/N$:

$$S_{t_{n+1}} = S_{t_n} + \mu(S_{t_n}, t_n) \Delta t + \sigma(S_{t_n}, t_n) \sqrt{\Delta t} \cdot Z_n$$

where $Z_n \sim \mathcal{N}(0, 1)$ are independent standard normals.

## Available Models

| Model | SDE | Use Case |
|-------|-----|----------|
| **BlackScholes** | $dS = (r-q)S dt + \sigma S dW$ | Standard equity options |
| **Heston** | $dS = (r-q)S dt + \sqrt{v}S dW^S$ | Stochastic volatility |
| **RoughHeston** | Fractional variance process | Rough volatility |
| **FractionalBrownianMotion** | Fractional Brownian motion | Long-memory effects |
| **RealDataModel** | Stationary Block Bootstrap | Historical data |

## Model Implementations

### BlackScholes (Geometric Brownian Motion)

**File:** `stock_model.py`

The standard GBM model with constant volatility:

$$dS_t = (r - q) S_t dt + \sigma S_t dW_t$$

**Closed-form solution:**

$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma W_T\right]$$

For multi-asset options ($d > 1$), the model supports correlated assets via Cholesky decomposition:

$$dS_t^{(i)} = (r - q) S_t^{(i)} dt + \sigma S_t^{(i)} \sum_{j=1}^i L_{ij} dW_t^{(j)}$$

where $L$ is the Cholesky factor of the correlation matrix $\rho$, i.e., $LL^\top = \rho$.

```python
from optimal_stopping.models import BlackScholes

model = BlackScholes(
    drift=0.05,           # Risk-free rate r
    volatility=0.2,       # Volatility σ
    dividend=0.0,         # Dividend yield q
    correlation=0.3,      # ρ for multi-asset (scalar or matrix)
    nb_stocks=10,         # Number of assets d
    nb_paths=100000,      # Monte Carlo paths m
    nb_dates=50,          # Exercise dates N
    maturity=1.0,         # Time to maturity T (years)
    spot=100,             # Initial spot price S_0
    dtype='float32'       # Precision ('float32' or 'float64')
)

# Generate paths: shape (m, d, N+1)
paths, _ = model.generate_paths()
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `drift` | float | Risk-free rate $r$ (or drift under $\mathbb{P}$) |
| `volatility` | float | Constant volatility $\sigma$ |
| `dividend` | float | Continuous dividend yield $q$ |
| `correlation` | float/array | Correlation $\rho$ (scalar for uniform, or $d \times d$ matrix) |
| `nb_stocks` | int | Number of assets $d$ |
| `nb_paths` | int | Number of Monte Carlo paths $m$ |
| `nb_dates` | int | Number of exercise dates $N$ |
| `maturity` | float | Time to maturity $T$ (in years) |
| `spot` | float | Initial spot price $S_0$ |
| `dtype` | str | Numerical precision |

---

### Heston (Stochastic Volatility)

**File:** `stock_model.py`

The Heston (1993) model with mean-reverting stochastic variance:

$$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$

$$dv_t = \kappa(\bar{v} - v_t) dt + \xi \sqrt{v_t} dW_t^v$$

where:
- $v_t$: Instantaneous variance
- $\kappa$: Mean reversion speed
- $\bar{v}$: Long-run variance (mean level)
- $\xi$: Volatility of variance (vol-of-vol)
- $\rho = \text{Corr}(dW_t^S, dW_t^v)$: Leverage correlation (typically negative)

**Feller condition** (ensures $v_t > 0$):

$$2\kappa\bar{v} \geq \xi^2$$

```python
from optimal_stopping.models import Heston

model = Heston(
    drift=0.05,           # Risk-free rate r
    volatility=0.3,       # Vol-of-vol ξ
    mean=0.04,            # Long-run variance v̄
    speed=1.5,            # Mean reversion κ
    correlation=-0.7,     # Leverage ρ (typically negative)
    nb_stocks=1,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100,
    dividend=0.0
)

# Returns (paths, variance_paths)
paths, var_paths = model.generate_paths()
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean` | float | Long-run variance $\bar{v}$ |
| `speed` | float | Mean reversion rate $\kappa$ |
| `volatility` | float | Vol-of-vol $\xi$ |
| `correlation` | float | Leverage correlation $\rho$ |

**Variants:**

- **HestonWithVar**: Same model but `generate_paths()` always returns variance paths (sets `return_var=True`).

---

### RoughHeston (Rough Volatility)

**File:** `stock_model.py`

The rough Heston model replaces the variance process with a fractional stochastic integral:

$$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$

$$v_t = v_0 + \frac{1}{\Gamma(H + \frac{1}{2})} \int_0^t (t-s)^{H-\frac{1}{2}} \left[\kappa(\bar{v} - v_s) ds + \xi \sqrt{v_s} dW_s^v\right]$$

where $H \in (0, 0.5)$ is the Hurst parameter controlling roughness. Smaller $H$ implies rougher variance paths.

**Key properties:**
- Non-Markovian: Future variance depends on full history
- Reproduces short-maturity smile steepening observed in markets
- Requires fine time discretization (`nb_steps_mult` parameter)

```python
from optimal_stopping.models import RoughHeston

model = RoughHeston(
    drift=0.02,
    volatility=0.3,       # Vol-of-vol ξ
    mean=0.04,            # Long-run variance v̄
    speed=1.5,            # Mean reversion κ
    correlation=-0.7,     # Leverage ρ
    hurst=0.25,           # Hurst parameter H ∈ (0, 0.5)
    nb_steps_mult=10,     # Fine discretization multiplier
    nb_stocks=5,
    nb_paths=100000,
    nb_dates=20,
    maturity=0.5,
    spot=100
)

paths, var_paths = model.generate_paths()
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hurst` | float | Hurst parameter $H \in (0, 0.5)$ |
| `nb_steps_mult` | int | Discretization multiplier (default: 10) |
| `v0` | float | Initial variance (default: $\bar{v}$) |

**Variants:**

- **RoughHestonWithVar**: Always returns variance paths.

---

### FractionalBrownianMotion

**File:** `stock_model.py`

Fractional Brownian motion (fBm) with Hurst parameter $H$:

$$B_H(t) = \int_0^t K_H(t, s) dW_s$$

where the kernel $K_H$ controls the correlation structure.

**Properties based on $H$:**
- $H = 0.5$: Standard Brownian motion (independent increments)
- $H > 0.5$: Persistent (trending) behavior
- $H < 0.5$: Anti-persistent (mean-reverting) behavior

```python
from optimal_stopping.models import FractionalBrownianMotion

model = FractionalBrownianMotion(
    drift=0.05,
    volatility=0.2,
    hurst=0.7,            # H > 0.5: persistence
    nb_stocks=1,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)

paths, _ = model.generate_paths()
```

**Variants:**

- **FractionalBlackScholes**: GBM driven by fBm instead of standard BM.
- **FractionalBrownianMotionPathDep**: Path-dependent representation for non-Markovian algorithms (1D only).

---

### RealDataModel (Stationary Block Bootstrap)

**File:** `real_data.py`

Generates paths using the Stationary Block Bootstrap (Politis & Romano, 1994) applied to historical market data. This preserves:
- Autocorrelation structure
- Volatility clustering
- Fat tails (kurtosis)
- Cross-asset correlations

```python
from optimal_stopping.models import RealDataModel

model = RealDataModel(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],  # Or None for auto-select
    start_date='2010-01-01',
    end_date='2024-01-01',
    exclude_crisis=False,     # Exclude 2008/2020 crisis periods
    only_crisis=False,        # Use only crisis periods
    drift_override=None,      # Override empirical drift (None = use historical)
    volatility_override=None, # Override empirical volatility
    avg_block_length=None,    # Auto-computed via Patton, Politis & White (2009)
    nb_stocks=25,
    nb_paths=100000,
    nb_dates=20,
    maturity=0.5,
    spot=100
)

paths, _ = model.generate_paths()
```

**Key Features:**

1. **Automatic ticker selection**: If `tickers=None`, selects top S&P 500 stocks by market cap and data availability.

2. **Optimal block length**: Uses the Patton, Politis & White (2009) algorithm to estimate optimal average block length from autocorrelation decay.

3. **Crisis filtering**:
   - `exclude_crisis=True`: Removes 2007-10 to 2009-06 and 2020-02 to 2020-05
   - `only_crisis=True`: Uses only crisis periods

4. **Drift/volatility control**:
   - `drift_override=None`: Use empirical historical drift
   - `drift_override=0.05`: Override to 5% annual drift
   - Same for `volatility_override`

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tickers` | list | Stock tickers (None = auto-select) |
| `start_date` | str | Historical data start (YYYY-MM-DD) |
| `end_date` | str | Historical data end |
| `exclude_crisis` | bool | Exclude crisis periods |
| `only_crisis` | bool | Use only crisis periods |
| `drift_override` | float | Override empirical drift |
| `volatility_override` | float | Override empirical volatility |
| `avg_block_length` | int | Average block length (None = auto) |

**Dependencies:** Requires `yfinance` package.

---

## Model Registry

Models are registered for dynamic loading via configuration files:

```python
from optimal_stopping.models import MODEL_REGISTRY, get_model

# Get model class by name
BlackScholes = get_model('BlackScholes')

# List available models
print(list(MODEL_REGISTRY.keys()))
# ['BlackScholes', 'Heston', 'HestonWithVar', 'FractionalBrownianMotion',
#  'FractionalBlackScholes', 'RoughHeston', 'RoughHestonWithVar', 'RealDataModel']
```

## Common Interface

All models inherit from the `Model` base class and implement:

```python
class Model:
    def __init__(self, drift, dividend, volatility, spot, nb_stocks,
                 nb_paths, nb_dates, maturity, name, dtype='float32', **kwargs):
        """Initialize model with common parameters."""

    def generate_paths(self, nb_paths=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate Monte Carlo paths.

        Returns:
            Tuple of (paths, auxiliary_data) where:
            - paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            - auxiliary_data: Model-specific (e.g., variance paths for Heston)
        """

    def disc_factor(self, date_begin, date_end) -> float:
        """Compute discount factor between two dates."""

    @property
    def dt(self) -> float:
        """Time step: maturity / nb_dates"""

    @property
    def df(self) -> float:
        """One-period discount factor: exp(-r * dt)"""
```

## Path Shape Convention

All models return paths with shape `(nb_paths, nb_stocks, nb_dates+1)`:

```
paths[i, j, n] = S_n^{(i,j)}
```

where:
- `i ∈ [0, nb_paths)`: Path index
- `j ∈ [0, nb_stocks)`: Asset index
- `n ∈ [0, nb_dates]`: Time index ($t_0 = 0$ to $t_N = T$)

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility. *The Review of Financial Studies*, 6(2), 327-343.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is Rough. *Quantitative Finance*, 18(6), 933-949.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic Block-Length Selection for the Dependent Bootstrap". *Econometric Reviews*, 28(4), 372-375.
