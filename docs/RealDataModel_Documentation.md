# RealDataModel: Real Market Data with Stationary Block Bootstrap

## Table of Contents
1. [Overview](#overview)
2. [Motivation: Why Real Data?](#motivation-why-real-data)
3. [Methodology: Stationary Block Bootstrap](#methodology-stationary-block-bootstrap)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Parameters and Limits](#parameters-and-limits)
7. [Advantages and Limitations](#advantages-and-limitations)
8. [Usage Examples](#usage-examples)
9. [References](#references)

---

## Overview

The `RealDataModel` is a stock price simulation framework that uses **real historical market data** instead of theoretical models like Black-Scholes. It employs the **stationary block bootstrap** method to generate synthetic price paths that preserve the empirical properties of actual market returns.

### Key Features
- ✅ Downloads real stock data via `yfinance` (Yahoo Finance API)
- ✅ Preserves **autocorrelation** (momentum, mean reversion patterns)
- ✅ Preserves **volatility clustering** (GARCH effects)
- ✅ Preserves **fat tails** and **skewness** (non-normal distributions)
- ✅ Maintains **realistic cross-asset correlations**
- ✅ Configurable time periods (exclude/include crisis periods)
- ✅ Supports up to **530 S&P 500 stocks** simultaneously

---

## Motivation: Why Real Data?

### Problems with Black-Scholes Assumptions

The Black-Scholes model assumes:
1. **Log-normal returns**: Returns follow a normal distribution
2. **Constant volatility**: Volatility doesn't change over time
3. **No autocorrelation**: Returns are independent (random walk)
4. **Continuous paths**: No jumps or gaps

### Reality of Financial Markets

Real markets exhibit:
1. **Fat tails**: Extreme events (crashes, rallies) occur more frequently than normal distribution predicts
2. **Volatility clustering**: High volatility periods cluster together (GARCH effects)
3. **Autocorrelation**: Returns show momentum (short-term) and mean reversion (long-term)
4. **Skewness**: Negative skew (crash risk) vs. positive skew (gradual rises)
5. **Time-varying correlations**: Asset correlations change during crises

### Impact on Option Pricing

These real-world features affect option values significantly:
- **Fat tails** → Higher out-of-the-money option prices
- **Volatility clustering** → Mispricing of volatility risk
- **Autocorrelation** → Path-dependent options (barriers, lookbacks) behave differently
- **Skewness** → Puts and calls priced asymmetrically

**RealDataModel addresses these issues** by using actual market behavior instead of theoretical assumptions.

---

## Methodology: Stationary Block Bootstrap

### What is Block Bootstrap?

The **stationary block bootstrap** (Politis & Romano, 1994) is a resampling technique designed for **time series data** with temporal dependencies.

#### Why Not Simple Bootstrap?

Standard bootstrap (Efron, 1979) samples individual observations independently:
```
Original data: [r₁, r₂, r₃, r₄, r₅, r₆, r₇, r₈]
Simple bootstrap: [r₃, r₁, r₇, r₂, r₈, r₄, r₅, r₆]  ❌ Destroys time dependencies!
```

**Problem**: This breaks autocorrelation, volatility clustering, and temporal patterns.

#### Block Bootstrap Solution

Sample **consecutive blocks** of data to preserve short-term dependencies:
```
Original data: [r₁, r₂, r₃, r₄, r₅, r₆, r₇, r₈]

Block 1: [r₃, r₄, r₅]  (length 3)
Block 2: [r₇, r₈, r₁]  (length 3, wraps around)
Block 3: [r₂, r₃]      (length 2)

Result: [r₃, r₄, r₅, r₇, r₈, r₁, r₂, r₃]  ✓ Preserves local structure!
```

### Stationary Block Bootstrap (SBB)

**Key innovation**: Block lengths are **random** (geometric distribution) instead of fixed.

#### Algorithm
1. Start at a random position in historical data
2. Draw block length `L` from geometric distribution: `P(L=k) = (1/p)(1-1/p)^(k-1)`
   - Average block length = `p` (e.g., 50 days)
3. Sample `L` consecutive returns (with wraparound)
4. Repeat until you have enough samples for one path
5. Repeat for multiple paths

#### Why Random Lengths?

**Fixed blocks** create artificial periodicity (every `L` steps, correlation resets).
**Random blocks** (SBB) maintain stationarity - statistical properties don't change over time.

### Visual Example

```
Historical returns (252 days):
[Day 1, Day 2, ..., Day 252]
      ↓↓↓  ↓↓↓↓     ↓↓↓↓↓↓
     Block 1   Block 2   Block 3  ...

Synthetic path (52 days for 3-month option):
Block 1 (len=15) + Block 2 (len=22) + Block 3 (len=15) = 52 days ✓

Each block preserves:
- Autocorrelation within the block
- Volatility clustering patterns
- Return distribution characteristics
```

---

## Mathematical Foundation

### Return Dynamics

Historical log-returns are computed as:
```
r_t,i = ln(S_t,i / S_{t-1,i})
```
where `S_t,i` is the price of stock `i` at time `t`.

### Bootstrap Sampling

For each synthetic path:
1. Generate bootstrap indices: `{τ₁, τ₂, ..., τ_T}` using SBB algorithm
2. Sample returns: `r*_t = r_{τ_t}` (bootstrapped returns)
3. Optionally adjust for target drift/volatility:
   ```
   r*_t,adj = (r*_t - μ_empirical) / σ_empirical × σ_target + μ_target
   ```
4. Reconstruct prices:
   ```
   S*_0 = S_0  (initial spot)
   S*_t = S*_{t-1} × exp(r*_t,adj)
   ```

### Block Length Selection

Optimal block length balances:
- **Too small** (e.g., L=1): Destroys autocorrelation (becomes simple bootstrap)
- **Too large** (e.g., L=252): Limited resampling diversity (just replays history)

**Automatic selection** (Politis & White, 2004):
```
L_opt ≈ lag where autocorrelation(lag) < 2/√n
```

For typical stocks:
- **Tech stocks** (AAPL, NVDA): L ≈ 5-20 days (momentum quickly decays)
- **Value stocks** (utilities): L ≈ 30-50 days (slower mean reversion)
- **Default**: L = 50 days (captures ~10 weeks of patterns)

### Correlation Preservation

For multi-asset baskets:
1. Download **joint** historical data for all stocks
2. Bootstrap **same indices** for all stocks: `{τ₁, τ₂, ..., τ_T}`
3. Sample multivariate returns: `r*_t = [r_{τ_t,1}, r_{τ_t,2}, ..., r_{τ_t,n}]`
4. **Result**: Cross-sectional correlation is automatically preserved!

**Why this works**: Sampling the same time periods keeps correlations intact.

---

## Implementation Details

### Data Source: yfinance

We use Yahoo Finance via `yfinance` library:
```python
import yfinance as yf
data = yf.download(['AAPL', 'MSFT', 'GOOGL'],
                   start='2010-01-01',
                   end='2024-01-01',
                   auto_adjust=True)  # Adjusts for splits/dividends
```

**Why Yahoo Finance?**
- ✓ Free and reliable
- ✓ Adjusted prices (handles stock splits, dividends)
- ✓ Long histories (>10 years for S&P 500)
- ✓ High-quality data for large caps

### Default Stock Universe

**530 S&P 500 stocks** organized by sector:
- **Technology** (70): AAPL, MSFT, NVDA, GOOGL, ...
- **Financials** (60): JPM, BAC, GS, BLK, REITs, ...
- **Healthcare** (50): JNJ, UNH, LLY, TMO, ...
- **Consumer** (90): WMT, AMZN, HD, COST, ...
- **Industrials** (80): BA, CAT, GE, UPS, ...
- **Energy** (35): XOM, CVX, SLB, ...
- **Materials** (35): LIN, NEM, FCX, ...
- **Utilities** (25): NEE, DUK, SO, ...
- **Real Estate** (20): PLD, AMT, SPG, ...
- **Communications** (30): GOOGL, META, DIS, VZ, ...

**Selection criteria**:
- Large-cap (high liquidity)
- Long trading history (available since 2010+)
- S&P 500 constituents (diversified, stable)

### Crisis Period Handling

**Two modes**:

1. **Exclude crisis** (`exclude_crisis=True`):
   - Removes 2008 financial crisis: Oct 2007 - Jun 2009
   - Removes COVID crash: Feb 2020 - May 2020
   - **Use case**: Normal market conditions, long-term pricing

2. **Only crisis** (`only_crisis=True`):
   - Keeps only crisis periods
   - **Use case**: Stress testing, tail risk analysis

3. **Default** (both `False`):
   - Uses all available data
   - **Use case**: Representative market behavior including extremes

### Drift and Volatility Adjustments

**Three scenarios**:

1. **Pure historical** (no drift/volatility in config):
   ```python
   # Uses empirical drift and volatility from data
   μ_empirical ≈ 19.9% annual  (FAANG+ stocks)
   σ_empirical ≈ 27.1% annual
   ```

2. **Config override** (drift=0.05 in config):
   ```python
   # Rescales returns to target drift/vol
   r_adjusted = (r - μ_empirical)/σ_empirical × σ_target + μ_target
   ```

3. **Explicit override** (drift_override parameter):
   ```python
   model = RealDataModel(drift_override=0.05, volatility_override=0.20)
   ```

**When to use each**:
- **Historical**: Study real market behavior
- **Config override**: Fair comparison with Black-Scholes (same drift)
- **Explicit override**: Scenario analysis (e.g., "What if drift = 0?")

---

## Parameters and Limits

### Constructor Parameters

| Parameter | Type | Default | Description | Limits |
|-----------|------|---------|-------------|--------|
| `tickers` | List[str] | S&P 500 (530) | Stock ticker symbols | Any valid Yahoo ticker |
| `start_date` | str | '2010-01-01' | Historical data start | Yahoo coverage (~1980+) |
| `end_date` | str | '2024-01-01' | Historical data end | Up to today |
| `exclude_crisis` | bool | False | Exclude 2008 & 2020 | True/False |
| `only_crisis` | bool | False | Only crisis periods | True/False |
| `drift_override` | float | None | Annual drift override | Any float (e.g., 0.05 = 5%) |
| `volatility_override` | float | None | Annual vol override | Any positive float |
| `avg_block_length` | int | Auto (~50) | Bootstrap block length | 5-100 days recommended |
| `nb_stocks` | int | len(tickers) | Number of stocks to use | 1-530 (uses first N) |
| `nb_paths` | int | Required | Paths to generate | 1-1,000,000+ |
| `nb_dates` | int | Required | Time steps per path | 1-10,000+ |
| `spot` | float | Required | Initial stock price | Any positive |
| `maturity` | float | Required | Option maturity (years) | Any positive |

### Computational Limits

| Dimension | Practical Limit | Notes |
|-----------|-----------------|-------|
| **Stocks** | 530 (S&P 500) | Could extend to 3,000+ (Russell 3000) |
| **Historical days** | 3,500+ (2010-2024) | More data = better estimates |
| **Paths** | 1,000,000+ | Memory: ~8 bytes × stocks × dates × paths |
| **Time steps** | 10,000+ | Daily steps for 40-year paths |
| **Block length** | 5-100 days | Auto-selected based on autocorrelation |

### Memory Requirements

For a typical setup:
- **nb_stocks = 50, nb_paths = 20,000, nb_dates = 252**
- Memory: 50 × 20,000 × 252 × 8 bytes ≈ **2 GB**

Formula: `Memory (GB) ≈ (nb_stocks × nb_paths × nb_dates × 8) / 1e9`

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Data download | O(stocks × days) | 30-60s for 530 stocks |
| Statistics calc | O(stocks × days) | <1s |
| Block length selection | O(stocks × lags²) | 1-5s |
| Path generation | O(paths × dates) | 1-2s for 20k paths |

**Bottleneck**: Data download (first time only)

---

## Advantages and Limitations

### Advantages ✅

1. **Realistic Market Behavior**
   - Captures actual fat tails, skewness, kurtosis
   - Preserves autocorrelation and GARCH effects
   - Real cross-asset correlations

2. **No Distributional Assumptions**
   - Model-free approach (non-parametric)
   - Works regardless of return distribution
   - Robust to non-normality

3. **Temporal Dependencies**
   - Block bootstrap preserves short-term patterns
   - Captures momentum and mean reversion
   - Volatility clustering maintained

4. **Flexibility**
   - Works with any number of assets (1-530)
   - Configurable time periods
   - Adjustable drift/volatility

5. **Empirical Foundation**
   - Based on 14+ years of actual data
   - Includes multiple market regimes
   - Crisis and normal periods

### Limitations ⚠️

1. **Stationarity Assumption**
   - **Assumes** market structure is stationary (doesn't change)
   - **Reality**: Regulations, technology, market structure evolve
   - **Impact**: May not predict unprecedented events

2. **Historical Dependence**
   - **Problem**: "Past performance ≠ future results"
   - **Risk**: If 2024+ markets differ structurally from 2010-2024
   - **Mitigation**: Use recent data, multiple periods

3. **No Structural Change**
   - **Cannot model**: Regime shifts, policy changes
   - **Example**: Fed rate changes, new regulations
   - **Alternative**: Use crisis-only or exclude-crisis modes

4. **Bootstrap Limitations**
   - **Finite sample**: Limited diversity from ~3,500 days
   - **Extreme tails**: Very rare events may be undersampled
   - **Mitigation**: Long historical period, multiple bootstraps

5. **Computational Cost**
   - **Download time**: 30-60s for 530 stocks (first run)
   - **Memory**: ~15 MB for returns, GB for paths
   - **Parallel inefficiency**: Each worker re-downloads

6. **No Fundamental Model**
   - **No economic structure**: Just statistical resampling
   - **Cannot extrapolate**: Beyond historical range
   - **Black box**: Hard to interpret "why" prices move

### When to Use RealDataModel

**Use when**:
- ✓ Testing algorithms on **realistic** market conditions
- ✓ Studying impact of **fat tails** and **volatility clustering**
- ✓ Pricing **path-dependent** options (barriers, lookbacks)
- ✓ Comparing **theoretical vs. empirical** pricing
- ✓ Stress testing with **crisis periods**

**Don't use when**:
- ✗ Need to model **future regime shifts** (interest rates, regulations)
- ✗ Require **analytical formulas** (use Black-Scholes, Heston)
- ✗ Working with **exotic assets** without historical data
- ✗ Need **fast** prototyping (download time matters)

---

## Usage Examples

### Example 1: Basic Usage (Historical Drift)

```python
from optimal_stopping.data.real_data import RealDataModel

# Use historical drift and volatility from 3 stocks
model = RealDataModel(
    nb_stocks=3,
    nb_paths=10000,
    nb_dates=52,     # ~3 months
    spot=100,
    maturity=0.25    # 3 months
)

paths, _ = model.generate_paths()
# paths.shape = (10000, 3, 53)  # 10k paths, 3 stocks, 53 time points
```

### Example 2: Override Drift (Fair Comparison with BS)

```python
# Use real correlations/tails, but match BS drift (5%) and vol (20%)
model = RealDataModel(
    drift=0.05,           # 5% annual drift (BS standard)
    volatility=0.20,      # 20% annual vol
    nb_stocks=5,
    nb_paths=20000,
    nb_dates=252,    # 1 year
    spot=100,
    maturity=1.0
)

paths, _ = model.generate_paths()
```

### Example 3: Crisis Period Analysis

```python
# Only use 2008 and 2020 crisis data
model = RealDataModel(
    only_crisis=True,    # Extreme market conditions
    nb_stocks=10,
    nb_paths=50000,
    nb_dates=126,    # 6 months
    spot=100,
    maturity=0.5
)

# Useful for stress testing barrier options
paths, _ = model.generate_paths()
```

### Example 4: Custom Tickers and Date Range

```python
# Use specific stocks from 2015-2024 (post-crisis period)
model = RealDataModel(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    start_date='2015-01-01',
    end_date='2024-01-01',
    exclude_crisis=True,      # Exclude COVID crash
    nb_stocks=5,
    nb_paths=10000,
    nb_dates=252,
    spot=100,
    maturity=1.0
)

paths, _ = model.generate_paths()
```

### Example 5: Large-Scale Basket (100 Stocks)

```python
# Price basket option on 100 stocks with realistic correlations
model = RealDataModel(
    nb_stocks=100,           # Uses first 100 from S&P 500 list
    nb_paths=20000,
    nb_dates=252,
    spot=100,
    maturity=1.0,
    drift=0.05,
    volatility=0.20
)

paths, _ = model.generate_paths()
# Correlation matrix is empirical from actual market data!
```

### Example 6: Using in Configs

```python
# In optimal_stopping/run/configs.py

real_vs_bs_vanilla = _FasterTable(
    stock_models=['BlackScholes', 'RealData'],  # Compare both

    algos=['RLSM', 'RFQI'],
    payoffs=['MaxCall', 'BasketCall', 'MinCall'],

    nb_stocks=[3, 5, 8],
    strikes=[100],
    spots=[100],
    volatilities=[0.2],    # Both models use 20% vol
    drift=[0.05],          # Both models use 5% drift

    nb_paths=[20000],
    nb_dates=[52],
    maturities=[0.25],

    nb_runs=10,
    representations=['TablePriceDuration'],
)
```

---

## References

### Academic Papers

1. **Politis, D. N., & Romano, J. P. (1994)**
   *"The stationary bootstrap"*
   Journal of the American Statistical Association, 89(428), 1303-1313.
   📖 **Original SBB paper** - Introduces random block lengths

2. **Politis, D. N., & White, H. (2004)**
   *"Automatic block-length selection for the dependent bootstrap"*
   Econometric Reviews, 23(1), 53-70.
   📖 **Optimal block length** - How to choose L automatically

3. **Efron, B. (1979)**
   *"Bootstrap methods: another look at the jackknife"*
   The Annals of Statistics, 7(1), 1-26.
   📖 **Original bootstrap** - Foundation for resampling methods

4. **Carlstein, E. (1986)**
   *"The use of subseries values for estimating the variance of a general statistic from a stationary sequence"*
   The Annals of Statistics, 14(3), 1171-1179.
   📖 **Block bootstrap** - Early fixed-length block approach

5. **Künsch, H. R. (1989)**
   *"The jackknife and the bootstrap for general stationary observations"*
   The Annals of Statistics, 17(3), 1217-1241.
   📖 **Moving block bootstrap** - Alternative fixed-length approach

### Textbooks

6. **Politis, D. N., Romano, J. P., & Wolf, M. (1999)**
   *"Subsampling"*
   Springer Series in Statistics.
   📚 Comprehensive treatment of resampling for time series

7. **Lahiri, S. N. (2003)**
   *"Resampling Methods for Dependent Data"*
   Springer.
   📚 Theory and applications of bootstrap for dependent data

### Software & Data

8. **yfinance Library**
   https://github.com/ranaroussi/yfinance
   🔧 Python interface to Yahoo Finance data

9. **Yahoo Finance**
   https://finance.yahoo.com/
   📊 Free historical stock price data

### Related Work in Finance

10. **Cont, R. (2001)**
    *"Empirical properties of asset returns: stylized facts and statistical issues"*
    Quantitative Finance, 1(2), 223-236.
    📊 **Stylized facts** - Fat tails, volatility clustering, etc.

11. **Bollerslev, T. (1986)**
    *"Generalized autoregressive conditional heteroskedasticity"*
    Journal of Econometrics, 31(3), 307-327.
    📊 **GARCH model** - Volatility clustering in returns

12. **Engle, R. F. (1982)**
    *"Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation"*
    Econometrica, 50(4), 987-1007.
    📊 **ARCH model** - Time-varying volatility

---

## Implementation Notes

### Code Structure

```
optimal_stopping/data/real_data.py
│
├── RealDataModel (main class)
│   ├── __init__()                    # Setup, download, process
│   ├── _get_default_tickers()        # S&P 500 list (530 stocks)
│   ├── _download_data()              # yfinance data retrieval
│   ├── _calculate_returns()          # Log returns
│   ├── _apply_crisis_filters()       # Remove/keep crisis periods
│   ├── _calculate_statistics()       # μ, σ, correlation
│   ├── _estimate_block_length()      # Automatic L from autocorr
│   ├── _stationary_bootstrap_indices() # SBB index generation
│   └── generate_paths()              # Main path generation
```

### Key Design Decisions

1. **Why stationary (not moving) block bootstrap?**
   - SBB has better statistical properties (stationarity)
   - Random lengths avoid artificial periodicity
   - Politis & Romano (1994) proven theory

2. **Why log returns (not simple returns)?**
   - Log returns are additive: `r_total = Σ r_t`
   - Preserve multiplicative price structure: `S_t = S_0 exp(Σ r_τ)`
   - Better for multivariate analysis (joint normality)

3. **Why wraparound sampling?**
   - Treats data as circular (no edge effects)
   - Fully utilizes all historical data
   - Standard in bootstrap literature

4. **Why automatic block length?**
   - Different stocks have different autocorrelation
   - Tech stocks: faster momentum decay → smaller L
   - Value stocks: slower mean reversion → larger L
   - Data-driven approach (Politis & White, 2004)

5. **Why 530 stocks?**
   - S&P 500 represents 80% of US market cap
   - High liquidity (tight spreads, low noise)
   - Long histories (survivorship bias minimal)
   - Sector diversity (realistic correlations)

---

## Comparison with Other Approaches

| Model | Autocorrelation | Vol Clustering | Fat Tails | Correlation | Flexibility | Speed |
|-------|-----------------|----------------|-----------|-------------|-------------|-------|
| **Black-Scholes** | ✗ | ✗ | ✗ | ✓ (input) | ★★★★★ | ★★★★★ |
| **Heston** | ✗ | ✓ | ~ | ✓ (input) | ★★★★☆ | ★★★★☆ |
| **GARCH** | ~ | ✓ | ✓ | ✓ (fit) | ★★★☆☆ | ★★★☆☆ |
| **Variance Gamma** | ✗ | ✗ | ✓ | ✓ (input) | ★★★☆☆ | ★★★★☆ |
| **RealData (SBB)** | ✓ | ✓ | ✓ | ✓ (empirical) | ★★★★☆ | ★★★☆☆ |

**RealDataModel unique advantages**:
- Only model that preserves **all four** empirical features
- **Non-parametric**: No distributional assumptions
- **Empirical correlations**: Automatically correct

**Trade-offs**:
- Slower (data download)
- Historical dependence
- No analytical formulas

---

## Future Extensions

Potential enhancements:

1. **File caching**: Cache downloaded data to avoid re-downloading
2. **International markets**: Add support for European, Asian stocks
3. **Intraday data**: Use minute/hourly bars for short-dated options
4. **Regime detection**: Auto-detect and sample from different market regimes
5. **Factor models**: Integrate Fama-French factors for correlation modeling
6. **GARCH overlay**: Combine SBB with GARCH for volatility forecasting

---

## Contact & Support

For questions about the implementation:
- See source code: `optimal_stopping/data/real_data.py`
- Check tests: `optimal_stopping/tests/test_real_data.py` (if available)
- Review examples: `optimal_stopping/run/configs.py` (real_vs_bs_* configs)

---

## Summary

The `RealDataModel` uses **stationary block bootstrap** on real market data to generate stock price paths that:
- ✅ Preserve autocorrelation, volatility clustering, fat tails
- ✅ Maintain empirical cross-asset correlations
- ✅ Reflect actual market behavior (not theoretical assumptions)
- ✅ Support 1-530 stocks simultaneously
- ✅ Enable fair comparison with Black-Scholes

**Use it when**: You need realistic market dynamics for option pricing, especially path-dependent options and large baskets.

**Trade-off**: Slower setup (data download) vs. realistic empirical properties.

**Bottom line**: RealDataModel bridges the gap between theoretical pricing models and actual market behavior.
