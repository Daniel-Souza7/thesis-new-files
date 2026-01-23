# Optimal Stopping via Randomized Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Repository** for the Master's Thesis:
> *"Enhancing Randomized Neural Networks for Pricing Complex High-Dimensional American Equity Derivatives"*
> Daniel Salvador de Melo e Souza, January 2026
> Master in Quantitative Methods in Finance

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Optimal Stopping Problem](#the-optimal-stopping-problem)
3. [The RT Algorithm](#the-rt-algorithm)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Configuration System](#configuration-system)
8. [Running Experiments](#running-experiments)
9. [Implemented Algorithms](#implemented-algorithms)
10. [Payoff Library](#payoff-library)
11. [Stochastic Process Models](#stochastic-process-models)
12. [Citation](#citation)
13. [Acknowledgements](#acknowledgements)

---

## Introduction

The valuation of American-style derivatives constitutes a fundamental challenge in quantitative finance due to the early exercise feature embedded in these contracts. Unlike European options, which can only be exercised at maturity, American contracts permit exercise at any point during the asset's life, thereby transforming the pricing problem into an **optimal stopping problem** often without closed-form solution.

This repository implements the **RT algorithm** (Randomized Thesis), a production-grade randomized neural network framework for American option pricing that combines:

- **Provable convergence guarantees** (inherited from the convex optimization structure)
- **Linear dimensional scaling** (circumventing the curse of dimensionality)
- **Benchmark-level accuracy** (achieving <0.1% relative error up to d=500 dimensions)

The implementation builds upon the foundational work of [Herrera et al. (2021)](https://arxiv.org/abs/2104.13669) on Randomized Least Squares Monte Carlo (RLSM), introducing several structural enhancements documented in the thesis.

### Relationship to the Thesis

This repository serves as the computational companion to the thesis, providing:

- Full implementation of all algorithms benchmarked in Chapter 4
- The 360-instrument payoff library constructed combinatorially (Section 3.3)
- Five stochastic process models for underlying asset dynamics (Section 3.2)
- Reproducible experimental infrastructure with pre-computed path datasets
- Hyperparameter optimization framework (Appendix B)

---

## The Optimal Stopping Problem

### Mathematical Formulation

Consider a filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}_n)_{n=0}^N, \mathbb{P})$ satisfying the usual conditions. Let $(X_n)_{n=0}^N$ be an $\mathbb{R}^d$-valued Markov process representing the evolution of $d$ underlying assets.

The holder of an American option seeks to maximize the expected discounted payoff by choosing an optimal exercise time. The value at time $n$ in state $x$ is given by the **Snell envelope**:

$$U_n(x) = \max\left( g(x), \mathbb{E}\left[\alpha U_{n+1}(X_{n+1}) \mid X_n = x\right] \right)$$

with terminal condition $U_N(x) = g(x)$, where:
- $g: \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ is the payoff function
- $\alpha = e^{-r\Delta t}$ is the one-period discount factor
- $r$ is the risk-free interest rate
- $\Delta t = T/N$ is the time step

The key quantity is the **continuation value**:

$$c_n(x) = \mathbb{E}\left[\alpha U_{n+1}(X_{n+1}) \mid X_n = x\right]$$

The optimal exercise strategy is characterized by the stopping rule:

$$\tau^* = \min\{n \in \{0,1,\ldots,N\} : g(X_n) \geq c_n(X_n)\}$$

### The Curse of Dimensionality

Traditional numerical methods face severe scalability limitations:

| Method | Complexity | Practical Limit |
|--------|------------|-----------------|
| Binomial/Trinomial Trees | $O(M^d)$ | $d \leq 3$ |
| Finite Difference Methods | $O(N \cdot M^{3d})$ | $d \leq 3$ |
| LSM with Polynomials | $O\binom{d+q}{q}$ | $d \leq 50$ |
| **Randomized Neural Networks** | $O(K \cdot d)$ | $d \leq 500+$ |

---

## The RT Algorithm

### Randomized Neural Network Architecture

The RT algorithm approximates continuation values using a single-hidden-layer neural network with **frozen random weights**:

$$\hat{c}_n(x) = \beta_n^\top \phi(x) = \beta_{n,0} + \sum_{j=1}^{K-1} \beta_{n,j} \sigma(w_j^\top x + b_j)$$

where:
- $\sigma(\cdot)$ is a nonlinear activation function (e.g., LeakyReLU, ELU, Tanh)
- $W = [w_1, \ldots, w_{K-1}]^\top \in \mathbb{R}^{(K-1) \times d}$ are **frozen** random weights
- $b = (b_1, \ldots, b_{K-1})^\top \in \mathbb{R}^{K-1}$ are **frozen** random biases
- $\beta_n \in \mathbb{R}^K$ are the **trainable** output weights (learned via OLS)

This architecture transforms the non-convex deep learning optimization into a **convex linear regression** problem, retaining theoretical convergence guarantees.

### Key Enhancements over RLSM

The RT algorithm introduces several structural improvements:

1. **Dimension-Adaptive Neuron Allocation** (Eq. 3.1 in thesis):
   - Low dimensions ($d \leq 9$): $K = \max(2d, 5)$
   - Medium dimensions ($10 \leq d \leq 49$): $K = 1.5d$
   - High dimensions ($d \geq 500$): $K = 1.2d$

2. **Adaptive Activation Function Selection**:
   - LeakyReLU for smooth payoffs
   - ELU for path-dependent and non-smooth payoffs

3. **Payoff-Informed Feature Augmentation**:
   - Including $g(x)$ as an additional input to the random feature map

4. **Non-Negativity Constraints**:
   - Enforcing $\hat{c}_n(x) \geq 0$ to prevent spurious early exercise

5. **Feedforward Path-Dependent Handling**:
   - Explicit payoff tracking instead of recurrent state compression

---

## Repository Structure

```
optimal_stopping/
├── algorithms/                 # Algorithm implementations
│   ├── standard/              # Non-path-dependent algorithms
│   │   ├── rlsm.py           # Randomized LSM (baseline)
│   │   ├── rt.py             # RT algorithm (proposed)
│   │   ├── lsm.py            # Classical LSM
│   │   ├── fqi.py            # Fitted Q-Iteration
│   │   ├── rfqi.py           # Randomized FQI
│   │   └── eop.py            # European Option Price
│   ├── path_dependent/        # Path-dependent variants
│   │   ├── srlsm.py          # Special RLSM for barriers
│   │   ├── srfqi.py          # Special RFQI for barriers
│   │   └── rrlsm.py          # Recurrent RLSM
│   ├── deep_neural_networks/  # Deep learning approaches
│   │   ├── DOS.py            # Deep Optimal Stopping
│   │   └── NLSM.py           # Neural LSM
│   └── utils/                 # Neural network utilities
│
├── data/                       # Stochastic process models
│   ├── stock_model.py         # GBM, Heston, FBM, Rough Heston
│   ├── real_data.py           # Stationary Block Bootstrap
│   ├── user_data_model.py     # Custom CSV data
│   ├── stored_paths/          # Pre-generated path storage
│   └── path_storage.py        # HDF5 path management
│
├── payoffs/                    # 360-instrument payoff library
│   ├── payoff.py              # Base class with auto-registration
│   ├── barrier_wrapper.py     # Barrier option wrapper
│   ├── basket_*.py            # Multi-asset payoffs
│   └── single_*.py            # Single-asset payoffs
│
├── run/                        # Execution infrastructure
│   ├── configs.py             # Experiment configurations
│   ├── run_algo.py            # Main execution script
│   ├── write_excel.py         # Results aggregation
│   └── plot_convergence.py    # Visualization tools
│
├── optimization/               # Hyperparameter optimization
│   ├── hyperopt.py            # Bayesian optimization
│   └── search_spaces.py       # Parameter search spaces
│
└── utilities/                  # Analysis and plotting tools
```

See individual folder READMEs for detailed documentation.

---

## Installation

### Requirements

- Python 3.8 or higher
- 8+ GB RAM (recommended: 32+ GB for high-dimensional problems)
- CUDA-compatible GPU (optional, for deep learning methods)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Daniel-Souza7/thesis-new-files.git
   cd thesis-new-files
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "from optimal_stopping.payoffs import list_payoffs; print(f'Loaded {len(list_payoffs())} payoffs')"
   ```
   Expected output: `Loaded 360 payoffs`

### Dependencies

Core packages include:
- `numpy`, `scipy`, `scikit-learn` - Scientific computing
- `torch`, `tensorflow` - Neural network backends
- `optuna` - Bayesian hyperparameter optimization
- `pandas`, `yfinance`, `h5py` - Data handling
- `matplotlib`, `plotly` - Visualization

---

## Quick Start

### Example 1: Price a 50-Dimensional Basket Call with RT

```python
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import BasketCall
from optimal_stopping.algorithms.standard.rt import RT

# Define market parameters
model = BlackScholes(
    drift=0.08,           # Risk-free rate r
    volatility=0.2,       # Volatility sigma
    nb_stocks=50,         # Dimension d
    nb_paths=100000,      # Monte Carlo paths m
    nb_dates=100,         # Exercise dates N
    spot=100,             # Initial price S_0
    maturity=1.0          # Time to maturity T
)

# Define payoff: max(0, mean(S) - K)
payoff = BasketCall(strike=100)

# Initialize and run RT algorithm
pricer = RT(
    model=model,
    payoff=payoff,
    hidden_size=75,                # K = 1.5 * 50
    activation='leakyrelu',
    use_payoff_as_input=True,
    train_ITM_only=True
)

price, path_gen_time = pricer.price(train_eval_split=2)
print(f"Option Price: ${price:.3f}")
```

### Example 2: Run Experiment via Configuration

The recommended approach uses the configuration system:

```bash
# Run a predefined configuration
python -m optimal_stopping.run.run_algo --configs=fast_test

# Export results to Excel
python -m optimal_stopping.run.write_excel --configs=fast_test
```

---

## Configuration System

### Overview

All experiments are defined in `optimal_stopping/run/configs.py` using a declarative dataclass structure. Each parameter is specified as an **iterable**, enabling automatic grid search over all combinations.

### Creating a Configuration

```python
# In configs.py
from dataclasses import dataclass

@dataclass
class my_experiment(_DefaultConfig):
    # Algorithm selection
    algos: tuple = ('RT', 'RLSM', 'LSM')

    # Problem specification
    payoffs: tuple = ('BasketCall', 'BasketPut')
    nb_stocks: tuple = (5, 25, 50)

    # Market parameters
    drift: tuple = (0.05,)
    volatilities: tuple = (0.2,)
    spots: tuple = (100,)
    strikes: tuple = (100,)
    maturities: tuple = (1.0,)

    # Simulation parameters
    nb_paths: tuple = (500000,)
    nb_dates: tuple = (50,)
    nb_runs: int = 5

    # Algorithm hyperparameters
    hidden_size: tuple = (75,)
    activation: tuple = ('leakyrelu',)
    use_payoff_as_input: tuple = (True,)
    train_ITM_only: tuple = (True,)

    # Precision
    dtype: tuple = ('float32',)
```

### Key Configuration Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `algos` | `tuple[str]` | Algorithms to run | `('RT', 'RLSM')` |
| `payoffs` | `tuple[str]` | Payoff structures | `('BasketCall',)` |
| `stock_models` | `tuple[str]` | Stochastic models | `('BlackScholes',)` |
| `nb_stocks` | `tuple[int]` | Number of assets $d$ | `(5, 50, 500)` |
| `nb_paths` | `tuple[int]` | Monte Carlo paths $m$ | `(1000000,)` |
| `nb_dates` | `tuple[int]` | Exercise dates $N$ | `(100,)` |
| `hidden_size` | `tuple[int]` | Hidden neurons $K$ | `(75,)` |
| `activation` | `tuple[str]` | Activation function | `('leakyrelu',)` |
| `barriers` | `tuple[float]` | Barrier level $B$ | `(120,)` |
| `barriers_up` | `tuple[float]` | Upper barrier $B_U$ | `(150,)` |
| `barriers_down` | `tuple[float]` | Lower barrier $B_L$ | `(70,)` |

### Barrier Option Configuration

For barrier options, use the barrier prefix naming convention:

```python
barrier_experiment = _DefaultConfig(
    algos=('RT', 'SRLSM'),
    payoffs=('UO-BasketCall', 'DI-MinPut'),  # Barrier prefix
    barriers=(120,),                          # Single barrier level
    # For double barriers:
    # payoffs=('UODO-BasketCall',),
    # barriers_up=(150,),
    # barriers_down=(70,)
)
```

---

## Running Experiments

### Basic Execution

```bash
# Run experiment
python -m optimal_stopping.run.run_algo --configs=my_experiment

# With parallel processing (uses all CPUs - 1)
python -m optimal_stopping.run.run_algo --configs=my_experiment --nb_jobs=8

# Filter by algorithm
python -m optimal_stopping.run.run_algo --configs=my_experiment --algos=RT,RLSM

# Filter by dimension
python -m optimal_stopping.run.run_algo --configs=my_experiment --nb_stocks=50
```

### Results Processing

```bash
# Aggregate results to Excel
python -m optimal_stopping.run.write_excel --configs=my_experiment

# Generate convergence plots
python -m optimal_stopping.run.plot_convergence --configs=my_experiment
```

### Output Structure

Results are saved to `optimal_stopping/run/results/{config_name}/`:
```
results/my_experiment/
├── my_experiment.csv           # Raw results
├── my_experiment_summary.xlsx  # Aggregated statistics
└── figures/                    # Generated plots
```

### Greeks Calculation

```bash
# Compute option Greeks (delta, gamma, theta, vega, rho)
python -m optimal_stopping.run.run_algo --configs=my_experiment \
    --compute_greeks \
    --greeks_method=central \
    --eps=0.01
```

---

## Implemented Algorithms

### Primary Algorithms (Benchmarked in Thesis)

| Algorithm | Class | Description | Reference |
|-----------|-------|-------------|-----------|
| **RT** | `RT` | Randomized Thesis algorithm (proposed) | Thesis Ch. 3 |
| **RLSM** | `RLSM` | Randomized Least Squares Monte Carlo | [Herrera et al., 2021] |
| **RFQI** | `RFQI` | Randomized Fitted Q-Iteration | [Herrera et al., 2021] |
| **LSM** | `LSM` | Least Squares Monte Carlo | [Longstaff & Schwartz, 2001] |
| **FQI** | `FQI` | Fitted Q-Iteration | [Tsitsiklis & Van Roy, 2001] |
| **DOS** | `DOS` | Deep Optimal Stopping | [Becker et al., 2019] |
| **NLSM** | `NLSM` | Neural Least Squares Monte Carlo | [Lapeyre & Lelong, 2021] |
| **EOP** | `EOP` | European Option Price (benchmark) | - |

### Path-Dependent Variants

| Algorithm | Class | Description |
|-----------|-------|-------------|
| **SRLSM** | `SRLSM` | Special RLSM for path-dependent options |
| **SRFQI** | `SRFQI` | Special RFQI for path-dependent options |
| **RRLSM** | `RRLSM` | Recurrent RLSM with Echo State Networks |

### Algorithm Selection Guide

```
Is your payoff path-dependent (barriers, Asian, lookback)?
├── No  → Use RT (universal, handles both)
└── Yes → Use RT or SRLSM
          ├── RT: Feedforward handling (recommended)
          └── SRLSM: Explicit path-dependent design

Need deep learning comparison?
├── Yes → DOS (fast) or NLSM (more accurate)
└── No  → RT or RLSM

Need classical baseline?
└── Yes → LSM (polynomial basis)
```

---

## Payoff Library

The repository implements **360 unique payoff structures** through combinatorial construction:

$$\text{Total} = 30 \text{ base payoffs} \times (1 + 11 \text{ barrier types}) = 360$$

### Base Payoffs (30 Total)

#### Single-Asset Payoffs (16)
| Category | Payoffs | Mathematical Form |
|----------|---------|-------------------|
| Vanilla | `Call`, `Put` | $(S_T - K)^+$, $(K - S_T)^+$ |
| Lookback Fixed | `LookbackFixedCall`, `LookbackFixedPut` | $(M_T - K)^+$, $(K - m_T)^+$ |
| Lookback Float | `LookbackFloatCall`, `LookbackFloatPut` | $(S_T - m_T)^+$, $(M_T - S_T)^+$ |
| Asian Fixed | `AsianFixedStrikeCall_Single`, `...Put_Single` | $(A_T - K)^+$, $(K - A_T)^+$ |
| Asian Float | `AsianFloatingStrikeCall_Single`, `...Put_Single` | $(S_T - A_T)^+$, $(A_T - S_T)^+$ |
| Range | `RangeCall_Single`, `RangePut_Single` | $(R_T - K)^+$, $(K - R_T)^+$ |

where $M_T = \max_{t \leq T} S_t$, $m_T = \min_{t \leq T} S_t$, $A_T = \frac{1}{T}\sum_{k=1}^T S_{t_k}$, $R_T = M_T - m_T$.

#### Multi-Asset Payoffs (14)
| Category | Payoffs | Mathematical Form |
|----------|---------|-------------------|
| Basket | `BasketCall`, `BasketPut` | $(\bar{S}_T - K)^+$, $(K - \bar{S}_T)^+$ |
| Geometric | `GeometricCall`, `GeometricPut` | $(G_T - K)^+$, $(K - G_T)^+$ |
| Rainbow | `MaxCall`, `MinPut` | $(S^{\max}_T - K)^+$, $(K - S^{\min}_T)^+$ |
| Dispersion | `DispersionCall`, `MaxDispersionCall` | $(\sigma_{\text{disp}} - K)^+$, $(S^{\max} - S^{\min} - K)^+$ |
| Rank-Based | `BestOfKCall`, `WorstOfKPut` | Best/worst $k$ assets |
| Asian Basket | `AsianFixedStrikeCall`, `AsianFloatingStrikeCall` | Time-averaged basket |

where $\bar{S}_T = \frac{1}{d}\sum_{i=1}^d S^i_T$, $G_T = (\prod_{i=1}^d S^i_T)^{1/d}$.

### Barrier Types (11)

| Code | Name | Condition |
|------|------|-----------|
| `UO` | Up-and-Out | Knock-out if $S_t > B$ |
| `DO` | Down-and-Out | Knock-out if $S_t < B$ |
| `UI` | Up-and-In | Activates when $S_t > B$ |
| `DI` | Down-and-In | Activates when $S_t < B$ |
| `UODO` | Double Knock-Out | Out if $S_t > B_U$ or $S_t < B_L$ |
| `UIDI` | Double Knock-In | In when $S_t > B_U$ or $S_t < B_L$ |
| `UIDO` | Up-In-Down-Out | Activate on up, deactivate on down |
| `UODI` | Up-Out-Down-In | Deactivate on up, activate on down |
| `PTB` | Partial Time Barrier | Barrier active only in $[T_1, T_2]$ |
| `StepB` | Step Barrier | Time-varying barrier $B(t)$ |
| `DStepB` | Double Step Barrier | Two time-varying barriers |

### Usage Examples

```python
from optimal_stopping.payoffs import get_payoff_class, list_payoffs

# List all available payoffs
all_payoffs = list_payoffs()
print(f"Total payoffs: {len(all_payoffs)}")  # 360

# Get a specific payoff class
BasketCall = get_payoff_class('BasketCall')
payoff = BasketCall(strike=100)

# Barrier variant (using naming convention)
UO_BasketCall = get_payoff_class('UO-BasketCall')
barrier_payoff = UO_BasketCall(strike=100, barrier=120)

# In configs.py
my_config = _DefaultConfig(
    payoffs=('BasketCall', 'UO-BasketCall', 'UODO-MinPut'),
    barriers=(120,),
    barriers_up=(150,),
    barriers_down=(70,)
)
```

---

## Stochastic Process Models

### Implemented Models (6)

| Model | Class | Key Parameters | Use Case |
|-------|-------|----------------|----------|
| **Black-Scholes** | `BlackScholes` | `drift`, `volatility`, `correlation` | Baseline GBM |
| **Heston** | `Heston` | `mean_reversion`, `vol_of_vol`, `correlation` | Stochastic volatility |
| **Fractional BM** | `FractionalBrownianMotion` | `hurst` | Long-memory dynamics |
| **Rough Heston** | `RoughHeston` | `hurst`, `vol_of_vol` | Rough volatility |
| **Real Data** | `RealDataModel` | `tickers`, historical period | Empirical validation |
| **User Data** | `UserDataModel` | `user_data_file` | Custom datasets |

### Configuration Examples

```python
# Black-Scholes (GBM) with correlation
bs_config = _DefaultConfig(
    stock_models=('BlackScholes',),
    drift=(0.05,),
    volatilities=(0.2,),
    correlation=(0.3,),  # Cross-asset correlation
    nb_stocks=(50,)
)

# Heston stochastic volatility
heston_config = _DefaultConfig(
    stock_models=('Heston',),
    drift=(0.05,),
    volatilities=(0.2,),      # Initial volatility v_0
    mean=(0.04,),             # Long-run variance theta
    speed=(2.0,),             # Mean reversion kappa
    correlation=(-0.7,)       # Leverage effect rho
)

# Rough Heston
rough_config = _DefaultConfig(
    stock_models=('RoughHeston',),
    hurst=(0.1,),             # Roughness parameter H
    mean=(0.04,),
    speed=(0.3,)
)

# Real market data (Stationary Block Bootstrap)
real_config = _DefaultConfig(
    stock_models=('RealData',),
    # Downloads from Yahoo Finance automatically
)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{souza2026randomized,
  author  = {Souza, Daniel Salvador de Melo e},
  title   = {Enhancing Randomized Neural Networks for Pricing Complex
             High-Dimensional American Equity Derivatives},
  school  = {University of Coimbra},
  year    = {2026},
  type    = {Master's Thesis},
  note    = {Master in Quantitative Methods in Finance}
}
```

This work builds upon:

```bibtex
@article{herrera2021optimal,
  author  = {Herrera, Calypso and Krach, Florian and Ruigrok, Pierre and Teichmann, Josef},
  title   = {Optimal Stopping via Randomized Neural Networks},
  journal = {arXiv preprint arXiv:2104.13669},
  year    = {2021}
}
```

---

## Acknowledgements

This thesis was developed under the supervision of Professor Helder Sebastiao and Professor Pedro Godinho at the Faculty of Economics, University of Coimbra. Special thanks to the authors of [Herrera et al., 2021] for providing the foundational RLSM implementation, and to Florian Krach for his openness to communication.

Computational resources were provided by the research center CeBER (Centre for Business and Economics Research).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Links

- [Thesis PDF](#) (coming soon)
- [Pre-computed Datasets](https://drive.google.com/drive/folders/thesis-datasets)
- [Interactive Optimal Stopping Game](https://optimal-stopping-game.streamlit.app)
- [Original RLSM Repository](https://github.com/HeKrRuTe/OptStopRandNN)
