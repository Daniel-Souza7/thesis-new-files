# Payoff Library

This module implements **360 unique payoff structures** for American-style derivatives pricing. The library is constructed combinatorially from 30 base payoffs and 12 barrier conditions (including the "None" condition for vanilla options).

## Mathematical Framework

### Payoff Function

The payoff function $g: \mathbb{R}^d \to \mathbb{R}$ maps the underlying state to the immediate exercise value:

$$g(X_n) = \text{[payoff at time } n \text{ given state } X_n]$$

For standard options, $g$ depends only on current prices $S_n = (S_n^{(1)}, \ldots, S_n^{(d)})$.

For **path-dependent options**, $g$ depends on the full history:

$$g(X_n) = g(S_0, S_1, \ldots, S_n)$$

### Barrier Conditions

Barrier options modify the base payoff based on whether the underlying has crossed certain price levels:

$$g_{\text{barrier}}(X_n) = g(X_n) \cdot \mathbf{1}_{\{\text{barrier condition met}\}}$$

## Library Structure

| Category | Count | Formula Type |
|----------|-------|--------------|
| Base Payoffs | 30 | $g(S)$ |
| Barrier Conditions | 12 | Includes "None" |
| **Total Payoffs** | **360** | $30 \times 12$ |

```
payoffs/
├── __init__.py              # Auto-registration and exports
├── payoff.py                # Base Payoff class with registry
├── barrier_wrapper.py       # Barrier condition wrapper
├── basket_simple.py         # BasketCall, BasketPut, MaxCall, MinPut, etc.
├── basket_asian.py          # Asian basket options
├── basket_range_dispersion.py  # Dispersion payoffs
├── basket_rank.py           # Rank-weighted payoffs
├── single_simple.py         # Call, Put
├── single_lookback.py       # Lookback options
├── single_asian.py          # Single-asset Asian options
└── single_range.py          # Range options
```

## Base Payoffs (30)

### Basket Options ($d > 1$)

#### Simple Basket (6)

| Payoff | Formula | Abbreviation |
|--------|---------|--------------|
| **BasketCall** | $\left(\bar{S} - K\right)^+$ where $\bar{S} = \frac{1}{d}\sum_{i=1}^d S^{(i)}$ | `BskCall` |
| **BasketPut** | $\left(K - \bar{S}\right)^+$ | `BskPut` |
| **GeometricCall** | $\left(\tilde{S} - K\right)^+$ where $\tilde{S} = \left(\prod_{i=1}^d S^{(i)}\right)^{1/d}$ | `GeoCall` |
| **GeometricPut** | $\left(K - \tilde{S}\right)^+$ | `GeoPut` |
| **MaxCall** | $\left(\max_i S^{(i)} - K\right)^+$ | `MaxCall` |
| **MinPut** | $\left(K - \min_i S^{(i)}\right)^+$ | `MinPut` |

```python
from optimal_stopping.payoffs import BasketCall, MaxCall

# Simple basket call on average
basket_call = BasketCall(strike=100)

# Rainbow option on maximum
max_call = MaxCall(strike=100)
```

#### Asian Basket (4) — Path-Dependent

| Payoff | Formula | Abbreviation |
|--------|---------|--------------|
| **AsianFixedStrikeCall** | $\left(\frac{1}{N}\sum_{n=0}^N \bar{S}_n - K\right)^+$ | `AsianFi-BskCall` |
| **AsianFixedStrikePut** | $\left(K - \frac{1}{N}\sum_{n=0}^N \bar{S}_n\right)^+$ | `AsianFi-BskPut` |
| **AsianFloatingStrikeCall** | $\left(\bar{S}_N - \frac{1}{N}\sum_{n=0}^N \bar{S}_n\right)^+$ | `AsianFl-BskCall` |
| **AsianFloatingStrikePut** | $\left(\frac{1}{N}\sum_{n=0}^N \bar{S}_n - \bar{S}_N\right)^+$ | `AsianFl-BskPut` |

```python
from optimal_stopping.payoffs import AsianFixedStrikeCall

# Asian option on time-averaged basket
asian_call = AsianFixedStrikeCall(strike=100)
```

#### Dispersion & Range (4)

| Payoff | Formula | Description |
|--------|---------|-------------|
| **DispersionCall** | $\left(\sigma_d - K\right)^+$ where $\sigma_d = \text{std}(S^{(1)}, \ldots, S^{(d)})$ | Cross-sectional volatility |
| **DispersionPut** | $\left(K - \sigma_d\right)^+$ | |
| **MaxDispersionCall** | $\left(\max_i S^{(i)} - \min_i S^{(i)} - K\right)^+$ | Price range |
| **MaxDispersionPut** | $\left(K - (\max_i S^{(i)} - \min_i S^{(i)})\right)^+$ | |

#### Rank-Based (4)

| Payoff | Formula | Parameters |
|--------|---------|------------|
| **BestOfKCall** | $\left(\frac{1}{k}\sum_{j=1}^k S_{(j)} - K\right)^+$ | `k`: top $k$ performers |
| **WorstOfKPut** | $\left(K - \frac{1}{k}\sum_{j=d-k+1}^d S_{(j)}\right)^+$ | `k`: bottom $k$ performers |
| **RankWeightedBasketCall** | $\left(\sum_{j=1}^k w_j S_{(j)} - K\right)^+$ | `k`, `weights` |
| **RankWeightedBasketPut** | $\left(K - \sum_{j=1}^k w_j S_{(j)}\right)^+$ | `k`, `weights` |

where $S_{(1)} \geq S_{(2)} \geq \ldots \geq S_{(d)}$ are order statistics (sorted prices).

```python
from optimal_stopping.payoffs import BestOfKCall, RankWeightedBasketCall

# Average of top 3 performers
best3_call = BestOfKCall(strike=100, k=3)

# Custom weights on top 2
weighted_call = RankWeightedBasketCall(strike=100, k=2, weights=[0.7, 0.3])
```

### Single-Asset Options ($d = 1$)

#### Vanilla (2)

| Payoff | Formula | Abbreviation |
|--------|---------|--------------|
| **Call** | $(S - K)^+$ | `Call` |
| **Put** | $(K - S)^+$ | `Put` |

#### Lookback (4) — Path-Dependent

| Payoff | Formula | Abbreviation |
|--------|---------|--------------|
| **LookbackFixedCall** | $\left(\max_{0 \leq n \leq N} S_n - K\right)^+$ | `LBFi-Call` |
| **LookbackFixedPut** | $\left(K - \min_{0 \leq n \leq N} S_n\right)^+$ | `LBFi-Put` |
| **LookbackFloatCall** | $\left(S_N - \min_{0 \leq n \leq N} S_n\right)^+$ | `LBFl-Call` |
| **LookbackFloatPut** | $\left(\max_{0 \leq n \leq N} S_n - S_N\right)^+$ | `LBFl-Put` |

```python
from optimal_stopping.payoffs import LookbackFloatCall

# Floating strike lookback (always positive payoff)
lookback = LookbackFloatCall(strike=100)
```

#### Asian Single (4) — Path-Dependent

| Payoff | Formula | Abbreviation |
|--------|---------|--------------|
| **AsianFixedStrikeCall_Single** | $\left(\frac{1}{N}\sum_{n=0}^N S_n - K\right)^+$ | `AsianFi-Call` |
| **AsianFixedStrikePut_Single** | $\left(K - \frac{1}{N}\sum_{n=0}^N S_n\right)^+$ | `AsianFi-Put` |
| **AsianFloatingStrikeCall_Single** | $\left(S_N - \frac{1}{N}\sum_{n=0}^N S_n\right)^+$ | `AsianFl-Call` |
| **AsianFloatingStrikePut_Single** | $\left(\frac{1}{N}\sum_{n=0}^N S_n - S_N\right)^+$ | `AsianFl-Put` |

#### Range (2) — Path-Dependent

| Payoff | Formula |
|--------|---------|
| **RangeCall_Single** | $\left(\max_n S_n - \min_n S_n - K\right)^+$ |
| **RangePut_Single** | $\left(K - (\max_n S_n - \min_n S_n)\right)^+$ |

---

## Barrier Conditions (12)

Barriers transform any base payoff into a path-dependent conditional payoff.

### Single Barriers (4)

| Type | Name | Condition | Payoff Survives If |
|------|------|-----------|-------------------|
| **UO** | Up-and-Out | Knock-out | $\max_{n,i} S_n^{(i)} < B$ |
| **DO** | Down-and-Out | Knock-out | $\min_{n,i} S_n^{(i)} > B$ |
| **UI** | Up-and-In | Knock-in | $\max_{n,i} S_n^{(i)} \geq B$ |
| **DI** | Down-and-In | Knock-in | $\min_{n,i} S_n^{(i)} \leq B$ |

### Double Barriers (4)

| Type | Name | Condition |
|------|------|-----------|
| **UODO** | Double Knock-Out | $B_L < \min_{n,i} S_n^{(i)}$ AND $\max_{n,i} S_n^{(i)} < B_U$ |
| **UIDI** | Double Knock-In | $\min_{n,i} S_n^{(i)} \leq B_L$ OR $\max_{n,i} S_n^{(i)} \geq B_U$ |
| **UIDO** | Up-In-Down-Out | $\max_{n,i} S_n^{(i)} \geq B_U$ AND $\min_{n,i} S_n^{(i)} > B_L$ |
| **UODI** | Up-Out-Down-In | $\max_{n,i} S_n^{(i)} < B_U$ AND $\min_{n,i} S_n^{(i)} \leq B_L$ |

### Time-Varying Barriers (3)

| Type | Name | Description |
|------|------|-------------|
| **PTB** | Partial-Time Barrier | Barrier only active during $[T_1, T_2] \subset [0, T]$ |
| **StepB** | Step Barrier | Barrier grows at risk-free rate: $B(t) = B_0 e^{rt}$ |
| **DStepB** | Double Step Barrier | Both upper and lower barriers grow at risk-free rate |

### None (Vanilla)

The 12th condition is **None** (no barrier), yielding the vanilla base payoff.

### Usage

```python
from optimal_stopping.payoffs import get_payoff_class, create_barrier_payoff, BasketCall

# Method 1: Get pre-registered barrier payoff by name
UO_BasketCall = get_payoff_class('UO_BasketCall')
payoff = UO_BasketCall(strike=100, barrier=120)

# Method 2: Create barrier payoff dynamically
barrier_class = create_barrier_payoff(BasketCall, 'UO')
payoff = barrier_class(strike=100, barrier=120)

# Double barrier example
UODO_MaxCall = get_payoff_class('UODO_MaxCall')
payoff = UODO_MaxCall(strike=100, barrier_up=130, barrier_down=80)

# Step barrier (grows at risk-free rate)
StepB_BasketPut = get_payoff_class('StepB_BasketPut')
payoff = StepB_BasketPut(strike=100, barrier=90, rate=0.05, maturity=1.0)
```

---

## Payoff Registry

All 360 payoffs are auto-registered and can be retrieved by name:

```python
from optimal_stopping.payoffs import get_payoff_class, list_payoffs, _PAYOFF_REGISTRY

# Get payoff by name or abbreviation
BasketCall = get_payoff_class('BasketCall')
BasketCall = get_payoff_class('BskCall')  # Same result

# List all registered payoffs
all_payoffs = list_payoffs()
print(f"Total payoffs: {len(all_payoffs)}")  # 360

# Access registry directly
print(list(_PAYOFF_REGISTRY.keys())[:10])
```

### Payoff Summary

```python
from optimal_stopping.payoffs import print_payoff_summary

print_payoff_summary()
# Output:
# 📊 Payoff Registry Summary:
#    Base payoffs:    30
#    Barrier payoffs: 330
#    Total payoffs:   360
```

---

## Common Interface

All payoffs inherit from the `Payoff` base class:

```python
class Payoff:
    is_path_dependent: bool  # True if payoff needs full path history
    abbreviation: str        # Short name (e.g., "BskCall")

    def __init__(self, strike, **kwargs):
        """Initialize with strike and optional parameters."""

    def __call__(self, stock_paths) -> np.ndarray:
        """
        Evaluate payoff for all paths at all timesteps.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            payoffs: Array of shape (nb_paths, nb_dates+1)
        """

    def eval(self, X) -> np.ndarray:
        """
        Evaluate payoff for given stock prices.

        Args:
            X: Shape (nb_paths, nb_stocks) for standard options
               Shape (nb_paths, nb_stocks, nb_dates+1) for path-dependent

        Returns:
            Array of shape (nb_paths,)
        """
```

---

## Using Payoffs with Algorithms

```python
from optimal_stopping.algorithms import RT
from optimal_stopping.models import BlackScholes
from optimal_stopping.payoffs import MaxCall

# Configure model
model = BlackScholes(
    drift=0.05,
    volatility=0.2,
    nb_stocks=10,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)

# Define payoff
payoff = MaxCall(strike=100)

# Price with RT algorithm
rt = RT(model=model, payoff=payoff)
price, std_error = rt.price()
```

### Barrier Option Example

```python
from optimal_stopping.payoffs import get_payoff_class

# Up-and-Out Basket Call
UO_BasketCall = get_payoff_class('UO_BasketCall')
payoff = UO_BasketCall(strike=100, barrier=120)

# Use with any algorithm
rt = RT(model=model, payoff=payoff)
price, std_error = rt.price()
```

---

## Path-Dependent Payoffs

Path-dependent payoffs require the full price history up to the current time. The algorithms handle this automatically.

**Path-dependent payoffs:**
- All barrier payoffs (11 types × 30 base = 330)
- Asian options (8 total)
- Lookback options (4 total)
- Range options (2 total)

**Markovian payoffs (not path-dependent):**
- Simple basket (6)
- Rank-based (4)
- Vanilla call/put (2)

---

## Extending the Library

### Creating Custom Payoffs

```python
from optimal_stopping.payoffs import Payoff
import numpy as np

class DigitalCall(Payoff):
    """Binary/Digital Call: pays 1 if S > K, else 0"""
    abbreviation = "DigCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        basket = np.mean(X, axis=1)
        return (basket > self.strike).astype(float)

# Auto-registered! Now available via:
# get_payoff_class('DigitalCall')
```

### Creating Custom Barrier Payoffs

```python
from optimal_stopping.payoffs import create_barrier_payoff

# Create all barrier variants for custom payoff
for barrier_type in ['UO', 'DO', 'UI', 'DI']:
    barrier_class = create_barrier_payoff(DigitalCall, barrier_type)
    # Now UO_DigitalCall, DO_DigitalCall, etc. are available
```

---

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Zhang, P. G. (1998). *Exotic Options: A Guide to Second Generation Options*. World Scientific.
