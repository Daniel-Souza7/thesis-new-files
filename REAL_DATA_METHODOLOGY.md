actual# Real Data Model: Methodology Specification

## Overview

This document describes the methodological choices made in implementing the `RealDataModel` class for simulating stock price paths using real market data. The model uses **stationary block bootstrap** (Politis & Romano, 1994) to generate realistic price paths that preserve empirical properties of financial time series.

---

## 1. Bootstrap Method: Stationary Block Bootstrap

### Choice: Stationary Block Bootstrap instead of Standard Moving Block Bootstrap

**Stationary Block Bootstrap (Politis & Romano, 1994):**
- Block lengths follow a **geometric distribution** with mean `p`
- Starting points are randomly selected from the entire historical sample
- Blocks can wrap around circularly

**Why not Moving Block Bootstrap (Künsch, 1989)?**
- Moving block bootstrap uses **fixed block lengths**, which can introduce artifacts at block boundaries
- Fixed blocks create dependence on the chosen block size parameter
- The resulting bootstrap sample has different statistical properties than the original data

**Rationale:**
The stationary block bootstrap was chosen because:
1. **Stationarity preservation**: The bootstrap sample is stationary if the original data is stationary (Politis & Romano, 1994, Theorem 1)
2. **No boundary effects**: Random block lengths eliminate artificial discontinuities at block boundaries
3. **Asymptotic validity**: Proven to provide consistent variance estimates for time series (Politis & Romano, 1994)
4. **Better mimics real dynamics**: The geometric distribution of block lengths better captures the mixing properties of financial returns

---

## 2. Block Length Selection: Automatic Data-Driven Selection

### Choice: Automatic block length estimation instead of Fixed block length

**Implementation (Lines 506-544):**
```python
def _estimate_block_length(self) -> int:
    """Estimate optimal block length from autocorrelation decay.
    Uses Politis & White (2004): Block length ≈ where autocorrelation falls below 2/√n
    """
```

The method:
1. Calculates autocorrelation function (ACF) for up to 100 lags
2. Finds where ACF falls below the significance threshold `2/√n` (Politis & White, 2004)
3. Uses the last significant lag as the optimal block length
4. Averages across multiple stocks and bounds between 5-50 days

**Why not Fixed block length?**
- A fixed block length (e.g., always 20 days) ignores the actual autocorrelation structure of the data
- Different stocks and time periods have different persistence levels
- Fixed lengths can be too short (destroying autocorrelation) or too long (introducing excessive persistence)

**Rationale:**
Automatic selection was chosen because:
1. **Data-adaptive**: Tailors block length to the actual autocorrelation decay rate in the historical sample
2. **Theoretically grounded**: Follows Politis & White (2004) optimal block length selection criterion
3. **Robust**: The 5-50 day bounds prevent pathological cases while allowing flexibility
4. **Preserves temporal dependence**: Ensures that short-term autocorrelation (e.g., momentum effects) is maintained in bootstrap samples

---

## 3. Return Calculation: Log Returns

### Choice: Log returns instead of Simple returns

**Implementation (Line 453):**
```python
self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
```

**Why not Simple returns?**
Simple returns: `r_simple = (S(t) - S(t-1)) / S(t-1)`

Problems:
- Not additive over time: `r_total ≠ r_1 + r_2 + ... + r_n`
- Asymmetric: A 50% gain followed by a 50% loss does not return to the original price
- Can exceed -100% (bounded below) but unbounded above

**Log returns:**
`r_log = ln(S(t) / S(t-1))`

**Rationale:**
Log returns were chosen because:
1. **Time additivity**: Log returns sum across time periods: `ln(S_T/S_0) = Σ r_log(t)`
2. **Symmetry**: A +10% log return and a -10% log return are symmetric
3. **Approximate normality**: Log returns are closer to normally distributed (especially for daily data)
4. **Correct path reconstruction** (Lines 611-613):
   ```python
   paths[path_idx, :, t + 1] = paths[path_idx, :, t] * np.exp(sampled_returns[t, :])
   ```
   This exponential transformation correctly reconstructs prices from log returns

---

## 4. Drift and Volatility: Empirical vs. Override

### Choice: Hybrid approach with optional override

**Default behavior: Use empirical statistics**
```python
self.empirical_drift_daily = np.mean(self.returns_array, axis=0)
self.empirical_drift_annual = np.mean(self.empirical_drift_daily) * 252
self.empirical_vol_daily = np.std(self.returns_array, axis=0)
self.empirical_vol_annual = np.mean(self.empirical_vol_daily) * np.sqrt(252)
```

**Optional override: User-specified drift/volatility** (Lines 605-608)
```python
if self.drift_override is not None or self.volatility_override is not None:
    # Demean and rescale
    sampled_returns = (sampled_returns - self.empirical_drift_daily) / self.empirical_vol_daily
    sampled_returns = sampled_returns * self.target_vol_daily + self.target_drift_daily
```

**Why not always use empirical statistics?**
- Historical drift may not reflect future expectations (e.g., using 2008 crisis data)
- Users may want to test sensitivity to different drift/volatility assumptions
- Benchmark comparisons require matching Black-Scholes parameters

**Why not always use overrides?**
- Loses realism from actual market behavior
- Misses regime-dependent dynamics (high vol in crashes, low vol in calm periods)
- Breaks correlation structure if naively rescaled

**Rationale:**
The hybrid approach was chosen because:
1. **Flexibility**: Researchers can use empirical values for realism OR specify parameters for controlled experiments
2. **Preserves autocorrelation**: The rescaling is applied **after** bootstrap sampling, so temporal dependence is maintained
3. **Preserves cross-sectional correlation**: The transformation is applied uniformly across stocks, maintaining the correlation matrix structure
4. **Clearly documented**: Override warnings are printed so users know which regime is active

---

## 5. Crisis Period Filtering

### Choice: Optional crisis filtering instead of Always include all data

**Options:**
1. **Default**: Use all historical data
2. **`exclude_crisis=True`**: Remove 2008 financial crisis (Oct 2007 - Jun 2009) and COVID crash (Feb 2020 - May 2020)
3. **`only_crisis=True`**: Use ONLY crisis periods

**Implementation (Lines 458-476):**
```python
def _apply_crisis_filters(self):
    if self.only_crisis:
        crisis_mask = (
            ((self.dates >= '2007-10-01') & (self.dates <= '2009-06-30')) |  # 2008 crisis
            ((self.dates >= '2020-02-01') & (self.dates <= '2020-05-31'))    # COVID crash
        )
        self.returns = self.returns[crisis_mask]
    elif self.exclude_crisis:
        # Exclude crisis periods
        ...
```

**Why not always include all data?**
- Crisis periods have fundamentally different statistical properties (fat tails, negative skewness, high correlation)
- Including crises may not represent "normal" market conditions if the goal is pricing in typical environments
- Fat tails from crises can dominate variance estimates

**Why not always exclude crises?**
- Crises ARE part of financial markets and should be modeled
- Stress testing requires crisis scenarios
- Excluding crises underestimates tail risk

**Rationale:**
Optional filtering was chosen because:
1. **Use case dependent**: Some applications need stress testing (include crises), others need typical behavior (exclude crises)
2. **Transparency**: Explicit date ranges are documented in code
3. **Research flexibility**: Enables comparative studies of crisis vs. normal periods
4. **Conservative defaults**: By default, ALL data is used (no hidden exclusions)

---

## 6. Data Quality: Coverage-Based Filtering

### Choice: Strict coverage requirements instead of Allow any available data

**Implementation (Lines 355-415):**
1. Calculate data coverage for each ticker: `coverage = prices.count() / total_rows`
2. Filter tickers with coverage ≥ 90% (relaxes to 80%, then 70% if needed)
3. Sort by coverage and select top N stocks
4. Drop all rows with ANY missing values across selected stocks

**Why not use all available data?**
- Different stocks have different listing dates and delisting events
- Missing data creates biases (survivorship bias, look-ahead bias)
- Sparse data leads to no common dates across large stock baskets

**Why not forward-fill or interpolate missing values?**
- Forward-filling introduces artificial autocorrelation
- Interpolation creates fake data points not based on actual trades
- Both methods distort the empirical correlation matrix

**Rationale:**
Strict filtering was chosen because:
1. **Data integrity**: Only use actual observed prices, not imputed values
2. **Consistent time windows**: All stocks have the same date range, enabling proper correlation estimation
3. **Scalability**: For large baskets (e.g., 100 stocks), overlapping date ranges are essential
4. **Transparent failures**: The code warns users about removed tickers rather than silently filling gaps

---

## 7. Correlation Preservation

### Choice: Sample returns jointly across stocks instead of Sample each stock independently

**Implementation (Lines 598-602):**
```python
# Generate bootstrap indices for this path
indices = self._stationary_bootstrap_indices(self.nb_dates)

# Sample returns using these indices (jointly across all stocks)
sampled_returns = self.returns_array[indices, :]  # Shape: (nb_dates, nb_stocks)
```

**Why not sample independently?**
Independent sampling:
```python
for stock in range(nb_stocks):
    indices_stock = bootstrap_indices()
    sampled_returns[:, stock] = returns[:, stock][indices_stock]
```
This would **destroy correlation structure** and create independent stock paths.

**Rationale:**
Joint sampling was chosen because:
1. **Realistic correlation**: Financial stocks are correlated, especially during market-wide events
2. **Multi-asset options**: Basket options, dispersion options, and correlation-dependent payoffs REQUIRE correlated paths
3. **Empirical preservation**: The bootstrap sample's correlation matrix approximates the historical correlation matrix
4. **Block structure**: Using the same block indices across stocks maintains cross-sectional dependence

---

## 8. Path Construction: Exponential Cumulative Returns

### Choice: Multiplicative path reconstruction instead of Additive returns

**Implementation (Lines 611-613):**
```python
for t in range(self.nb_dates):
    # Apply log returns: S(t+1) = S(t) * exp(r(t))
    paths[path_idx, :, t + 1] = paths[path_idx, :, t] * np.exp(sampled_returns[t, :])
```

**Why not additive?**
Additive reconstruction: `S(t+1) = S(t) + r(t)`
- Violates limited liability: Prices could go negative
- Inconsistent with log returns (if log returns are used, must exponentiate)
- Doesn't match geometric Brownian motion dynamics

**Rationale:**
Multiplicative reconstruction was chosen because:
1. **Consistency with log returns**: Exponentiating log returns correctly recovers price ratios
2. **Positive prices**: Exponential ensures `S(t) > 0` always
3. **Matches continuous-time models**: Geometric Brownian motion has `dS/S = μ dt + σ dW`, leading to exponential returns
4. **Correct compounding**: Multi-period returns compound multiplicatively, not additively

---

## 9. Data Source: yfinance with Auto-Adjusted Prices

### Choice: Adjusted close prices instead of Unadjusted close prices

**Implementation (Line 333):**
```python
data = yf.download(
    self.tickers,
    start=self.start_date,
    end=self.end_date,
    auto_adjust=True  # Adjust for splits and dividends
)
```

**Why not unadjusted prices?**
Unadjusted prices:
- Stock splits create artificial -50% "returns" overnight
- Dividend payments create artificial drops on ex-dividend dates
- Distorts statistical properties (fake volatility spikes)

**Adjusted prices:**
- Retroactively adjusts historical prices for all corporate actions
- Creates continuous price series as if no splits/dividends occurred
- Reflects total return (price appreciation + reinvested dividends)

**Rationale:**
Adjusted prices were chosen because:
1. **Statistical validity**: Returns reflect actual investor returns, not corporate actions
2. **Volatility accuracy**: No spurious volatility from splits
3. **Comparison across stocks**: All stocks on equal footing regardless of split history
4. **Standard practice**: Academic research and quant finance use adjusted prices by default

---

## Summary of Key Methodological Choices

| Aspect | Choice | Alternative | Reason |
|--------|--------|-------------|--------|
| Bootstrap method | Stationary block | Moving block | Preserves stationarity, no boundary effects |
| Block length | Data-driven (ACF) | Fixed | Adapts to actual autocorrelation structure |
| Return type | Log returns | Simple returns | Time-additive, symmetric, approximately normal |
| Drift/volatility | Empirical (override optional) | Always empirical or always override | Flexibility for different research goals |
| Crisis filtering | Optional | Always include or exclude | Use-case dependent (stress test vs. typical) |
| Missing data | Strict filtering | Interpolation | Data integrity, no fake values |
| Correlation | Joint sampling | Independent | Preserves cross-sectional dependence |
| Path reconstruction | Multiplicative (exp) | Additive | Ensures positive prices, correct compounding |
| Price data | Adjusted close | Unadjusted | Removes corporate action artifacts |

---

## References

- **Politis, D. N., & Romano, J. P. (1994).** "The stationary bootstrap." *Journal of the American Statistical Association*, 89(428), 1303-1313.
- **Politis, D. N., & White, H. (2004).** "Automatic block-length selection for the dependent bootstrap." *Econometric Reviews*, 23(1), 53-70.
- **Künsch, H. R. (1989).** "The jackknife and the bootstrap for general stationary observations." *The Annals of Statistics*, 17(3), 1217-1241.

---

## Implementation Location

File: `optimal_stopping/data/real_data.py`

Key methods:
- `_download_data()`: Lines 306-448
- `_calculate_returns()`: Lines 450-456
- `_apply_crisis_filters()`: Lines 458-476
- `_calculate_statistics()`: Lines 478-504
- `_estimate_block_length()`: Lines 506-544
- `_stationary_bootstrap_indices()`: Lines 546-578
- `generate_paths()`: Lines 580-615
