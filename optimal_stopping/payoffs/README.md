# Payoffs Module

> Comprehensive library of 360 option payoff structures for American option pricing.
> See [Main README](../../README.md) for repository overview.

---

## Overview

This module implements **360 unique payoff structures** through combinatorial construction:

$$\text{Total Payoffs} = 30 \text{ base payoffs} \times (1 + 11 \text{ barrier types}) = 360$$

The library covers:
- **Vanilla options** (Call, Put)
- **Multi-asset baskets** (Arithmetic, Geometric, Rainbow)
- **Path-dependent exotics** (Asian, Lookback, Range)
- **Correlation products** (Dispersion, Best-of-K)
- **Barrier modifications** (Knock-in, Knock-out, Double, Step)

---

## Directory Structure

```
payoffs/
├── __init__.py                 # Registry & auto-generation system
├── payoff.py                   # Base Payoff class with auto-registration
├── barrier_wrapper.py          # BarrierPayoff wrapper (11 barrier types)
│
├── single_simple.py            # Call, Put
├── single_lookback.py          # Lookback options (4 variants)
├── single_asian.py             # Asian options single-asset (4 variants)
├── single_range.py             # Range options (2 variants)
│
├── basket_simple.py            # Basket, Geometric, Max, Min (6 variants)
├── basket_asian.py             # Asian basket options (4 variants)
├── basket_range_dispersion.py  # Dispersion options (4 variants)
└── basket_rank.py              # Rank-based options (4 variants)
```

---

## Quick Reference

### Listing Available Payoffs

```python
from optimal_stopping.payoffs import list_payoffs, get_payoff_class

# List all 360 payoffs
all_payoffs = list_payoffs()
print(f"Total payoffs: {len(all_payoffs)}")  # 360

# Filter by category
vanilla = [p for p in all_payoffs if 'Call' in p or 'Put' in p]
barriers = [p for p in all_payoffs if '-' in p]
path_dep = [p for p in all_payoffs if 'Asian' in p or 'Lookback' in p]
```

### Using Payoffs

```python
# Get payoff class by name
BasketCall = get_payoff_class('BasketCall')
payoff = BasketCall(strike=100)

# Evaluate payoff on stock paths
# stock_prices shape: (nb_paths, nb_stocks) for non-path-dependent
# stock_paths shape: (nb_paths, nb_stocks, nb_dates+1) for path-dependent
payoff_values = payoff.eval(stock_prices)
```

---

## Base Payoffs (30 Total)

### Category I: Single-Asset Payoffs (16)

#### Vanilla Options (2)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Call | `Call` | $(S_T - K)^+$ | No |
| Put | `Put` | $(K - S_T)^+$ | No |

```python
from optimal_stopping.payoffs import Call, Put

call = Call(strike=100)
put = Put(strike=100)
```

#### Lookback Options (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Lookback Fixed Call | `LookbackFixedCall` | $(M_T - K)^+$ | Yes |
| Lookback Fixed Put | `LookbackFixedPut` | $(K - m_T)^+$ | Yes |
| Lookback Float Call | `LookbackFloatCall` | $(S_T - m_T)^+$ | Yes |
| Lookback Float Put | `LookbackFloatPut` | $(M_T - S_T)^+$ | Yes |

where $M_T = \max_{t \leq T} S_t$ and $m_T = \min_{t \leq T} S_t$.

```python
from optimal_stopping.payoffs import LookbackFixedCall, LookbackFloatPut

lookback_call = LookbackFixedCall(strike=100)
lookback_put = LookbackFloatPut(strike=100)
```

#### Asian Options - Single Asset (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Asian Fixed Call | `AsianFixedStrikeCall_Single` | $(A_T - K)^+$ | Yes |
| Asian Fixed Put | `AsianFixedStrikePut_Single` | $(K - A_T)^+$ | Yes |
| Asian Float Call | `AsianFloatingStrikeCall_Single` | $(S_T - A_T)^+$ | Yes |
| Asian Float Put | `AsianFloatingStrikePut_Single` | $(A_T - S_T)^+$ | Yes |

where $A_T = \frac{1}{N}\sum_{k=1}^{N} S_{t_k}$ is the discrete arithmetic average.

```python
from optimal_stopping.payoffs import AsianFixedStrikeCall_Single

asian_call = AsianFixedStrikeCall_Single(strike=100)
```

#### Range Options (2)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Range Call | `RangeCall_Single` | $(R_T - K)^+$ | Yes |
| Range Put | `RangePut_Single` | $(K - R_T)^+$ | Yes |

where $R_T = M_T - m_T$ is the price range.

---

### Category II: Multi-Asset Payoffs (14)

#### Central Tendency Baskets (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Basket Call | `BasketCall` | $(\bar{S}_T - K)^+$ | No |
| Basket Put | `BasketPut` | $(K - \bar{S}_T)^+$ | No |
| Geometric Call | `GeometricCall` | $(G_T - K)^+$ | No |
| Geometric Put | `GeometricPut` | $(K - G_T)^+$ | No |

where $\bar{S}_T = \frac{1}{d}\sum_{i=1}^{d} S^i_T$ and $G_T = \left(\prod_{i=1}^{d} S^i_T\right)^{1/d}$.

```python
from optimal_stopping.payoffs import BasketCall, GeometricPut

basket_call = BasketCall(strike=100)
geometric_put = GeometricPut(strike=100)
```

#### Rainbow Options (2)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Max Call | `MaxCall` | $(S^{\max}_T - K)^+$ | No |
| Min Put | `MinPut` | $(K - S^{\min}_T)^+$ | No |

where $S^{\max}_T = \max_i S^i_T$ and $S^{\min}_T = \min_i S^i_T$.

```python
from optimal_stopping.payoffs import MaxCall, MinPut

max_call = MaxCall(strike=100)
min_put = MinPut(strike=100)
```

#### Dispersion Options (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Dispersion Call | `DispersionCall` | $(\sigma_{\text{disp}} - K)^+$ | No |
| Dispersion Put | `DispersionPut` | $(K - \sigma_{\text{disp}})^+$ | No |
| Max Dispersion Call | `MaxDispersionCall` | $(S^{\max} - S^{\min} - K)^+$ | No |
| Max Dispersion Put | `MaxDispersionPut` | $(K - (S^{\max} - S^{\min}))^+$ | No |

where $\sigma_{\text{disp}} = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(S^i_T - \bar{S}_T)^2}$.

```python
from optimal_stopping.payoffs import MaxDispersionCall

dispersion = MaxDispersionCall(strike=10)  # Strike on spread
```

#### Rank-Based Options (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Best-of-K Call | `BestOfKCall` | $\left(\frac{1}{k}\sum_{i=1}^{k} S_{(i)} - K\right)^+$ | No |
| Worst-of-K Put | `WorstOfKPut` | $\left(K - \frac{1}{k}\sum_{i=d-k+1}^{d} S_{(i)}\right)^+$ | No |
| Rank Weighted Call | `RankWeightedBasketCall` | $\left(\sum_{i=1}^{d} w_i S_{(i)} - K\right)^+$ | No |
| Rank Weighted Put | `RankWeightedBasketPut` | $\left(K - \sum_{i=1}^{d} w_i S_{(i)}\right)^+$ | No |

where $S_{(1)} \geq S_{(2)} \geq \ldots \geq S_{(d)}$ are order statistics (descending).

```python
from optimal_stopping.payoffs import BestOfKCall, RankWeightedBasketCall

# Best of top 3 assets (equal weight)
best_of_3 = BestOfKCall(strike=100, k=3)

# Custom rank weights: 50% best, 30% second, 20% third
rank_weighted = RankWeightedBasketCall(strike=100, weights=(0.5, 0.3, 0.2))
```

---

### Category III: Asian Basket Options (4)

| Payoff | Class | Formula | Path-Dependent |
|--------|-------|---------|----------------|
| Asian Fixed Basket Call | `AsianFixedStrikeCall` | $(\bar{A}_T - K)^+$ | Yes |
| Asian Fixed Basket Put | `AsianFixedStrikePut` | $(K - \bar{A}_T)^+$ | Yes |
| Asian Float Basket Call | `AsianFloatingStrikeCall` | $(\bar{S}_T - \bar{A}_T)^+$ | Yes |
| Asian Float Basket Put | `AsianFloatingStrikePut` | $(\bar{A}_T - \bar{S}_T)^+$ | Yes |

where $\bar{A}_T = \frac{1}{N}\sum_{k=1}^{N} \bar{S}_{t_k}$ is the time-averaged basket mean.

---

## Barrier Types (11)

Any base payoff can be modified with barrier conditions using the naming convention `{BARRIER}-{PAYOFF}`:

### Single Barriers (4)

| Code | Name | Condition | Effect |
|------|------|-----------|--------|
| `UO` | Up-and-Out | $\exists t: S_t > B$ | Knock-out (worthless) |
| `DO` | Down-and-Out | $\exists t: S_t < B$ | Knock-out (worthless) |
| `UI` | Up-and-In | $\exists t: S_t > B$ | Activates option |
| `DI` | Down-and-In | $\exists t: S_t < B$ | Activates option |

```python
# Up-and-Out Basket Call
UO_BasketCall = get_payoff_class('UO-BasketCall')
payoff = UO_BasketCall(strike=100, barrier=120)

# Down-and-In Min Put
DI_MinPut = get_payoff_class('DI-MinPut')
payoff = DI_MinPut(strike=100, barrier=80)
```

### Double Barriers (4)

| Code | Name | Condition |
|------|------|-----------|
| `UODO` | Double Knock-Out | Out if $S_t > B_U$ OR $S_t < B_L$ |
| `UIDI` | Double Knock-In | In when $S_t > B_U$ OR $S_t < B_L$ |
| `UIDO` | Up-In-Down-Out | Activate on up-touch, deactivate on down-touch |
| `UODI` | Up-Out-Down-In | Deactivate on up-touch, activate on down-touch |

```python
# Double Knock-Out Basket Call (corridor option)
UODO_BasketCall = get_payoff_class('UODO-BasketCall')
payoff = UODO_BasketCall(strike=100, barrier_up=150, barrier_down=70)
```

### Time-Dependent Barriers (3)

| Code | Name | Description |
|------|------|-------------|
| `PTB` | Partial Time Barrier | Barrier active only during $[T_1, T_2]$ |
| `StepB` | Step Barrier | Time-varying barrier $B(t)$ |
| `DStepB` | Double Step Barrier | Two time-varying barriers |

```python
# Partial Time Barrier: active from 25% to 75% of maturity
PTB_BasketCall = get_payoff_class('PTB-BasketCall')
payoff = PTB_BasketCall(strike=100, barrier=120,
                        step_param1=0.25, step_param2=0.75)

# Step Barrier: linear growth from B_start to B_end
StepB_MaxCall = get_payoff_class('StepB-MaxCall')
payoff = StepB_MaxCall(strike=100, barrier=100,
                       step_param1=100, step_param2=150)  # B(0)=100, B(T)=150
```

---

## Configuration via `configs.py`

### Basic Payoff Configuration

```python
from dataclasses import dataclass

@dataclass
class my_experiment(_DefaultConfig):
    # Payoff selection
    payoffs: tuple = ('BasketCall', 'BasketPut', 'MaxCall', 'MinPut')

    # Strike price
    strikes: tuple = (90, 100, 110)  # OTM, ATM, ITM
```

### Barrier Option Configuration

```python
@dataclass
class barrier_experiment(_DefaultConfig):
    # Barrier payoffs use prefix notation
    payoffs: tuple = (
        'UO-BasketCall',      # Up-and-Out
        'DI-MinPut',          # Down-and-In
        'UODO-GeometricCall'  # Double barrier
    )

    # Single barrier level
    barriers: tuple = (120,)

    # Double barrier levels
    barriers_up: tuple = (150,)
    barriers_down: tuple = (70,)
```

### Rank-Based Configuration

```python
@dataclass
class rank_experiment(_DefaultConfig):
    payoffs: tuple = ('BestOfKCall', 'WorstOfKPut', 'RankWeightedBasketCall')

    # Best/Worst of K parameter
    k: tuple = (3, 5)

    # Rank weights (must sum to 1)
    weights: tuple = ((0.5, 0.3, 0.2), (0.4, 0.3, 0.2, 0.1))
```

### Step Barrier Configuration

```python
@dataclass
class step_barrier_experiment(_DefaultConfig):
    payoffs: tuple = ('StepB-BasketCall', 'DStepB-MinPut')

    # Step barrier parameters
    step_param1: tuple = (100,)   # Start level or T1
    step_param2: tuple = (150,)   # End level or T2
    step_param3: tuple = (80,)    # For double step: lower start
    step_param4: tuple = (60,)    # For double step: lower end
```

---

## Mathematical Reference

### Barrier Payoff Formula

For a base payoff $V_{\text{base}}$ and barrier condition $\mathcal{C}$:

$$V_T = V_{\text{base}} \cdot \mathbf{1}_{\mathcal{C}}$$

**Knock-Out Conditions:**
- Up-and-Out: $\mathcal{C} = \{\tau_U > T\}$ where $\tau_U = \inf\{t : S_t > B_U\}$
- Down-and-Out: $\mathcal{C} = \{\tau_L > T\}$ where $\tau_L = \inf\{t : S_t < B_L\}$
- Double Knock-Out: $\mathcal{C} = \{\tau_U > T\} \cap \{\tau_L > T\}$

**Knock-In Conditions:**
- Up-and-In: $\mathcal{C} = \{\tau_U \leq T\}$
- Down-and-In: $\mathcal{C} = \{\tau_L \leq T\}$
- Double Knock-In: $\mathcal{C} = \{\tau_U \leq T\} \cup \{\tau_L \leq T\}$

### Monitoring Convention

For multi-asset payoffs, barrier monitoring uses the **basket mean** by default:
- $S_t = \bar{S}_t = \frac{1}{d}\sum_{i=1}^{d} S^i_t$

For single-asset payoffs ($d=1$):
- $S_t = S^1_t$

---

## Implementation Details

### Auto-Registration System

All payoffs are automatically registered via the `__init_subclass__` metaclass:

```python
# In payoff.py
class Payoff:
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Payoff._registry[cls.__name__] = cls
```

### Barrier Auto-Generation

Barrier variants are generated automatically in `__init__.py`:

```python
BARRIER_TYPES = ['UO', 'DO', 'UI', 'DI', 'UODO', 'UIDI', 'UIDO', 'UODI',
                 'PTB', 'StepB', 'DStepB']

# Auto-generate all combinations
for base_payoff in BASE_PAYOFFS:
    for barrier_type in BARRIER_TYPES:
        name = f"{barrier_type}-{base_payoff}"
        # Dynamic class creation with BarrierPayoff wrapper
```

### Creating Custom Payoffs

To add a new payoff:

1. Create a class inheriting from `Payoff`:

```python
# In payoffs/my_payoffs.py
from optimal_stopping.payoffs.payoff import Payoff

class MyCustomPayoff(Payoff):
    is_path_dependent = False  # or True for path-dependent

    def __init__(self, strike, my_param=1.0):
        self.strike = strike
        self.my_param = my_param

    def eval(self, stock_prices):
        """
        Args:
            stock_prices: (nb_paths, nb_stocks) for non-path-dependent
                         (nb_paths, nb_stocks, nb_dates+1) for path-dependent

        Returns:
            payoffs: (nb_paths,) array
        """
        if self.is_path_dependent:
            # Use full path history
            final_prices = stock_prices[:, :, -1]
        else:
            final_prices = stock_prices

        basket_mean = final_prices.mean(axis=1)
        return np.maximum(0, self.my_param * basket_mean - self.strike)
```

2. Import in `payoffs/__init__.py`:

```python
from optimal_stopping.payoffs.my_payoffs import MyCustomPayoff
```

The payoff is automatically registered and barrier variants are generated.

---

## Payoff Index

A complete mathematical index of all 360 payoffs with LaTeX formulas is available in:
- `payoffs_index.tex` (LaTeX source)
- [Online Documentation](#) (rendered PDF)

---

## Path-Dependent vs. Non-Path-Dependent

| Property | Non-Path-Dependent | Path-Dependent |
|----------|-------------------|----------------|
| `is_path_dependent` | `False` | `True` |
| Input shape | `(nb_paths, nb_stocks)` | `(nb_paths, nb_stocks, nb_dates+1)` |
| Examples | BasketCall, MaxCall | AsianCall, LookbackCall |
| Algorithm support | All algorithms | RT, SRLSM, RRLSM |

**Detection:**
```python
payoff = get_payoff_class('AsianFixedStrikeCall')(strike=100)
print(payoff.is_path_dependent)  # True

payoff = get_payoff_class('BasketCall')(strike=100)
print(payoff.is_path_dependent)  # False
```

---

## References

- [Hull, 2018] "Options, Futures, and Other Derivatives" - Chapters on exotic options
- [Zhang, 1998] "Exotic Options: A Guide to Second Generation Options"
- [Rubinstein & Reiner, 1991] "Breaking Down the Barriers" - Barrier option analytics
