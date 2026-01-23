# Algorithms Module

> Implementation of optimal stopping algorithms for American option pricing.
> See [Main README](../../README.md) for repository overview.

---

## Overview

This module implements **18 algorithms** organized into four categories:

| Category | Location | Algorithms | Description |
|----------|----------|------------|-------------|
| **Standard** | `standard/` | RT, RLSM, RFQI, LSM, FQI, EOP | Non-path-dependent options |
| **Path-Dependent** | `path_dependent/` | SRLSM, SRFQI, RRLSM | Barrier, Asian, lookback options |
| **Deep Learning** | `deep_neural_networks/` | DOS, NLSM | Trainable neural networks |
| **Experimental** | `testing/` | Various | Research prototypes |

---

## Directory Structure

```
algorithms/
├── standard/                    # Primary algorithms
│   ├── rt.py                   # RT: Randomized Thesis (proposed)
│   ├── rlsm.py                 # RLSM: Randomized LSM
│   ├── rfqi.py                 # RFQI: Randomized FQI
│   ├── lsm.py                  # LSM: Least Squares Monte Carlo
│   ├── fqi.py                  # FQI: Fitted Q-Iteration
│   └── eop.py                  # EOP: European Option Price
│
├── path_dependent/              # Path-dependent variants
│   ├── srlsm.py                # Special RLSM for barriers
│   ├── srfqi.py                # Special RFQI for barriers
│   └── rrlsm.py                # Recurrent RLSM (Echo State Networks)
│
├── deep_neural_networks/        # Deep learning approaches
│   ├── DOS.py                  # Deep Optimal Stopping
│   └── NLSM.py                 # Neural LSM
│
├── testing/                     # Experimental algorithms
│   ├── stochastic_mesh.py      # Broadie-Glasserman method
│   ├── randomized_stochastic_mesh1.py
│   ├── randomized_stochastic_mesh2.py
│   ├── zap_q.py                # ZapQ learning
│   ├── rzapq.py                # Randomized ZapQ
│   ├── dkl.py                  # Deep Kernel Learning LSM
│   ├── rdkl.py                 # Randomized DKL
│   └── SRFQI_RBF.py            # SRFQI with RBF kernels
│
└── utils/                       # Shared utilities
    ├── randomized_neural_networks.py   # RNN implementations
    ├── neural_networks.py              # Trainable networks
    ├── basis_functions.py              # Polynomial bases
    └── utilities.py                    # Helper functions
```

---

## Algorithm Reference

### RT (Randomized Thesis) - **Proposed Algorithm**

**Location:** `standard/rt.py`

The RT algorithm is the main contribution of this thesis. It extends RLSM to handle **both** path-dependent and non-path-dependent options with a unified architecture.

**Key Features:**
- Universal handling of all 360 payoff types
- Dimension-adaptive neuron allocation
- Feedforward path-dependent processing
- Non-negativity constraint on continuation values

**Mathematical Formulation:**

$$\hat{c}_n(x) = \max\left(0, \beta_n^\top \phi(x)\right)$$

where $\phi(x) = (\sigma(Wx + b)^\top, 1)^\top \in \mathbb{R}^K$ is the random feature map.

**Usage:**

```python
from optimal_stopping.algorithms.standard.rt import RT
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import BasketCall

model = BlackScholes(drift=0.05, volatility=0.2, nb_stocks=50,
                     nb_paths=100000, nb_dates=50, spot=100, maturity=1.0)
payoff = BasketCall(strike=100)

pricer = RT(
    model=model,
    payoff=payoff,
    hidden_size=75,              # Number of hidden neurons K
    activation='leakyrelu',      # Activation function
    use_payoff_as_input=True,    # Payoff augmentation
    train_ITM_only=True,         # ITM filtering
    dropout=0.0                  # Dropout probability
)

price, path_gen_time = pricer.price(train_eval_split=2)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | 20 | Number of hidden neurons $K$ |
| `activation` | `str` | `'leakyrelu'` | Activation: `'relu'`, `'tanh'`, `'elu'`, `'leakyrelu'` |
| `use_payoff_as_input` | `bool` | `True` | Include $g(x)$ in feature map |
| `use_barrier_as_input` | `bool` | `False` | Include barrier levels as input |
| `train_ITM_only` | `bool` | `True` | Train only on ITM paths |
| `dropout` | `float` | 0.0 | Dropout probability $\in [0, 1]$ |
| `factors` | `tuple` | `(1., 1.)` | Weight scaling factors |

---

### RLSM (Randomized Least Squares Monte Carlo)

**Location:** `standard/rlsm.py`

The baseline randomized neural network algorithm from [Herrera et al., 2021].

**Key Differences from RT:**
- Fixed architecture (no dimension adaptation)
- Designed for non-path-dependent options only
- No non-negativity constraint by default

**Usage:**

```python
from optimal_stopping.algorithms.standard.rlsm import RLSM

pricer = RLSM(
    model=model,
    payoff=payoff,
    hidden_size=20,
    activation='leakyrelu',
    ridge_coeff=0.0              # L2 regularization coefficient
)
price, _ = pricer.price()
```

---

### SRLSM (Special RLSM for Path-Dependent Options)

**Location:** `path_dependent/srlsm.py`

Explicit path-dependent variant that passes full trajectory history for payoff evaluation.

**Usage:**

```python
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.payoffs import get_payoff_class

UO_BasketCall = get_payoff_class('UO-BasketCall')
barrier_payoff = UO_BasketCall(strike=100, barrier=120)

pricer = SRLSM(
    model=model,
    payoff=barrier_payoff,
    hidden_size=30,
    activation='elu'             # ELU recommended for barriers
)
price, _ = pricer.price()
```

---

### RRLSM (Recurrent Randomized LSM)

**Location:** `path_dependent/rrlsm.py`

Uses Echo State Networks (ESN) to encode path history into a recurrent hidden state.

**Note:** As documented in thesis Section 4.6, RT outperforms RRLSM by an average of 12.1% on path-dependent contracts due to its feedforward decomposition avoiding recurrent state compression losses.

---

### LSM (Least Squares Monte Carlo)

**Location:** `standard/lsm.py`

Classical polynomial basis method from [Longstaff & Schwartz, 2001].

**Variants:**
- `LSM` - Degree-2 monomials (default)
- `LeastSquarePricerDeg1` - Degree-1 monomials
- `LeastSquarePricerLaguerre` - Weighted Laguerre polynomials

**Usage:**

```python
from optimal_stopping.algorithms.standard.lsm import LSM

pricer = LSM(model=model, payoff=payoff)
price, _ = pricer.price()
```

**Scalability Note:** LSM becomes computationally prohibitive for $d > 50$ due to polynomial basis explosion. The number of basis functions scales as $K = \binom{d+2}{2} = O(d^2)$.

---

### DOS (Deep Optimal Stopping)

**Location:** `deep_neural_networks/DOS.py`

Deep learning approach using trainable neural networks with policy gradient optimization.

**Characteristics:**
- Fast inference
- No convergence guarantees
- May converge to local minima

**Usage:**

```python
from optimal_stopping.algorithms.deep_neural_networks.DOS import DOS

pricer = DOS(
    model=model,
    payoff=payoff,
    hidden_size=40,
    nb_epochs=30
)
price, _ = pricer.price()
```

---

### NLSM (Neural Least Squares Monte Carlo)

**Location:** `deep_neural_networks/NLSM.py`

Combines LSM structure with trainable neural networks for continuation value regression.

---

### EOP (European Option Price)

**Location:** `standard/eop.py`

Computes European option prices by exercising all paths at maturity. Used as a benchmark when early exercise is provably suboptimal (e.g., calls with positive drift).

**Usage:**

```python
from optimal_stopping.algorithms.standard.eop import EOP

pricer = EOP(model=model, payoff=payoff)
european_price, _ = pricer.price()
```

---

## Utility Modules

### Randomized Neural Networks (`utils/randomized_neural_networks.py`)

Implements the frozen random weight architectures:

- **`Reservoir2`** (PyTorch): Main implementation with configurable layers, activations, dropout
- **`randomRNN`** (PyTorch): Recurrent variant for RRLSM
- **`Reservoir`** (NumPy): Legacy implementation (deprecated)

**Supported Activations:**
- `'relu'` - Rectified Linear Unit
- `'leakyrelu'` - Leaky ReLU (slope=0.01)
- `'tanh'` - Hyperbolic tangent
- `'elu'` - Exponential Linear Unit

### Basis Functions (`utils/basis_functions.py`)

Polynomial basis implementations for classical LSM:

- **`BasisFunctions`** - Degree-2 monomials
- **`BasisFunctionsDeg1`** - Degree-1 monomials
- **`BasisFunctionsLaguerre`** - Weighted Laguerre polynomials

---

## Algorithm Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                  ALGORITHM SELECTION GUIDE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Is your payoff path-dependent?                             │
│  (barriers, Asian, lookback, range)                         │
│                                                             │
│     NO ──────────────────────┐                              │
│                              │                              │
│     YES ─────┐               │                              │
│              ▼               ▼                              │
│        ┌─────────┐     ┌─────────┐                          │
│        │   RT    │     │   RT    │  ◄── Recommended         │
│        │ (univ.) │     │  RLSM   │                          │
│        │ SRLSM   │     │  LSM    │                          │
│        └─────────┘     └─────────┘                          │
│                                                             │
│  Need deep learning comparison?                             │
│     YES → DOS (fast) or NLSM (accurate)                     │
│                                                             │
│  Need classical baseline?                                   │
│     YES → LSM (polynomial basis)                            │
│                                                             │
│  Dimension > 50?                                            │
│     YES → RT or RLSM only (LSM infeasible)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration via `configs.py`

All algorithms are instantiated through the configuration system. Key hyperparameters:

```python
# In optimal_stopping/run/configs.py
@dataclass
class my_experiment(_DefaultConfig):
    algos: tuple = ('RT', 'RLSM', 'LSM')

    # Randomized NN hyperparameters
    hidden_size: tuple = (50, 75, 100)        # Grid search over K
    activation: tuple = ('leakyrelu', 'elu')  # Grid search activations
    dropout: tuple = (0.0,)
    ridge_coeff: tuple = (0.0,)               # L2 regularization
    use_payoff_as_input: tuple = (True,)
    train_ITM_only: tuple = (True,)

    # Deep NN hyperparameters
    nb_epochs: tuple = (30,)                  # Training epochs
```

---

## Performance Characteristics

| Algorithm | Time Complexity | Memory | Convergence | Best For |
|-----------|-----------------|--------|-------------|----------|
| **RT** | $O(m \cdot N \cdot K^2)$ | Low | Guaranteed | Universal |
| **RLSM** | $O(m \cdot N \cdot K^2)$ | Low | Guaranteed | Non-path-dep. |
| **SRLSM** | $O(m \cdot N \cdot K^2)$ | Medium | Guaranteed | Path-dependent |
| **LSM** | $O(m \cdot N \cdot K^2)$ | High | Guaranteed | Low-dim baseline |
| **DOS** | $O(m \cdot N \cdot E \cdot H^2)$ | Low | Not guaranteed | Fast approximation |
| **NLSM** | $O(m \cdot N \cdot E \cdot H^2)$ | Low | Not guaranteed | DL comparison |

Where: $m$ = paths, $N$ = dates, $K$ = basis size, $E$ = epochs, $H$ = hidden size.

---

## Adding New Algorithms

To implement a new algorithm:

1. Create a new file in the appropriate subdirectory
2. Implement a class with the following interface:

```python
class MyAlgorithm:
    def __init__(self, model, payoff, **kwargs):
        """
        Args:
            model: Stock model instance
            payoff: Payoff instance
            **kwargs: Algorithm-specific hyperparameters
        """
        self.model = model
        self.payoff = payoff
        # Initialize algorithm

    def price(self, train_eval_split=2):
        """
        Compute option price.

        Args:
            train_eval_split: Ratio for train/eval split

        Returns:
            tuple: (price, path_generation_time)
        """
        # Implementation
        return price, path_gen_time
```

3. Register in `configs.py` by adding to the `ALGO_DICT` mapping

---

## References

- [Herrera et al., 2021] "Optimal Stopping via Randomized Neural Networks"
- [Longstaff & Schwartz, 2001] "Valuing American Options by Simulation"
- [Tsitsiklis & Van Roy, 2001] "Regression Methods for Pricing Complex American-Style Options"
- [Becker et al., 2019] "Deep Optimal Stopping"
