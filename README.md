# RT Algorithm: Randomized Neural Networks for High-Dimensional American Option Pricing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository implements the **RT (Randomized Thesis) algorithm**, a production-grade randomized neural network framework for pricing American-style derivatives in high-dimensional settings. The framework accompanies the thesis:

> **"The RT Algorithm: Randomized Neural Networks for High-Dimensional American Option Pricing"**

## The Optimal Stopping Problem

The valuation of American-style derivatives requires solving an **optimal stopping problem**: determining when to exercise an option to maximize its expected discounted payoff. Unlike European options (exercisable only at maturity), American options permit early exercise, transforming pricing into a dynamic programming problem.

Mathematically, the option value at time $n$ in state $x$ is given by the **Snell envelope**:

$$U_n(x) = \max\left\{ g(x), \mathbb{E}\left[\alpha U_{n+1}(X_{n+1}) \mid X_n = x\right] \right\}$$

where:
- $g(x)$ is the immediate exercise payoff
- $\alpha = e^{-r\Delta t}$ is the one-period discount factor
- The expectation represents the **continuation value** $c_n(x)$

The fundamental challenge lies in estimating continuation values $c_n(x)$ across high-dimensional state spaces, where traditional methods (lattice, finite differences) scale exponentially with dimension.

## The RT Algorithm

The RT algorithm addresses this challenge using **randomized neural networks**: single-hidden-layer networks with frozen random weights, where only the output layer is trained via linear regression.

### Key Features

- **Dimension-adaptive neuron allocation**: Hidden layer size scales with problem complexity
- **Activation function selection**: Tailored to payoff regularity (LeakyReLU vs ELU)
- **Payoff-informed feature augmentation**: Uses $g(x)$ as an input hint
- **Non-negativity constraints**: Enforces $\hat{c}_n(x) \geq 0$
- **Feedforward path-dependent handling**: Explicit tracking instead of recurrent networks

### Convergence Guarantees

Unlike deep neural networks with non-convex optimization, RT inherits the convexity of linear regression, enabling provable convergence as sample paths $m \to \infty$ and neurons $K \to \infty$.

## Repository Structure

```
thesis-new-files/
├── optimal_stopping/           # Core library (pip installable)
│   ├── algorithms/             # Pricing algorithms
│   │   ├── core/               # RT, RLSM, LSM, FQI, RFQI, EOP
│   │   ├── deep/               # DOS, NLSM (deep learning baselines)
│   │   ├── recurrent/          # RRLSM, SRLSM, SRFQI (path-dependent)
│   │   ├── experimental/       # Research algorithms (SM, ZAPQ, DKL)
│   │   └── utils/              # Neural network utilities
│   ├── models/                 # Stochastic process models
│   │   ├── stock_model.py      # GBM, Heston, FBM, Rough Heston
│   │   └── real_data.py        # Stationary Block Bootstrap
│   ├── payoffs/                # 360 payoff structures
│   │   ├── basket_*.py         # Multi-asset payoffs
│   │   ├── single_*.py         # Single-asset payoffs
│   │   └── barrier_wrapper.py  # Barrier conditions
│   ├── storage/                # Path caching utilities
│   ├── run/                    # Execution scripts
│   └── utilities/              # Analysis and plotting tools
├── experiments/                # Thesis experiment configurations
│   ├── configs/                # Configuration files
│   └── results/                # Output directory
├── optimization/               # Hyperparameter optimization (Optuna)
├── data/                       # Pre-computed path datasets
└── docs/                       # Documentation and LaTeX files
```

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Daniel-Souza7/thesis-new-files.git
cd thesis-new-files

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[full]"
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Usage

### Basic Example: Pricing a Basket Call Option

```python
from optimal_stopping.algorithms import RT
from optimal_stopping.models import BlackScholes
from optimal_stopping.payoffs import BasketCall

# Define the payoff
payoff = BasketCall(strike=100)

# Configure the model
model = BlackScholes(
    drift=0.05,           # Risk-free rate
    volatility=0.2,       # Volatility
    nb_stocks=10,         # Number of assets
    nb_paths=100000,      # Monte Carlo paths
    nb_dates=50,          # Exercise dates
    maturity=1.0,         # Time to maturity (years)
    spot=100,             # Initial spot price
)

# Generate paths
stock_paths = model.generate_paths()

# Price with RT algorithm
rt = RT(
    payoff=payoff,
    model=model,
    hidden_size=20,       # Or use adaptive sizing
    activation='leakyrelu'
)
price, std_error = rt.price(stock_paths)

print(f"Option Price: ${price:.4f} +/- ${std_error:.4f}")
```

### Running Experiments via Command Line

The main entry point for experiments is `optimal_stopping/run/run_algo.py`:

```bash
# Run with a predefined configuration
python -m optimal_stopping.run.run_algo --config thesis_basket_call

# Run with custom parameters (modify configs.py or use flags)
python -m optimal_stopping.run.run_algo --config my_config
```

See [optimal_stopping/run/README.md](optimal_stopping/run/README.md) for detailed usage.

### Using Pre-computed Paths

For reproducibility, use pre-computed path datasets:

```python
from optimal_stopping.storage import StoredPathsModel

# Load pre-computed paths
model = StoredPathsModel(
    path_file='data/stored_paths/BS_50.h5',
    nb_paths=1000000
)
stock_paths = model.generate_paths()
```

### Configuration Files

Experiments are configured via Python files in `experiments/configs/`:

```python
# experiments/configs/my_experiment.py
from experiments.configs.defaults import DefaultConfig

my_config = DefaultConfig(
    algos=['RT', 'RLSM', 'LSM'],
    stock_models=['BlackScholes'],
    payoffs=['BasketCall', 'BasketPut'],
    nb_stocks=[10, 50, 100],
    nb_paths=[1000000],
    nb_dates=[100],
    nb_runs=10,
    drift=0.05,
    volatility=0.2,
)
```

Run with:
```bash
python -m experiments.runners.run_algo --config my_config
```

## Implemented Algorithms

### Core Algorithms (Thesis)

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **RT** | Randomized Thesis algorithm (proposed) | This thesis |
| **RLSM** | Randomized Least Squares Monte Carlo | Herrera et al. (2021) |
| **RFQI** | Randomized Fitted Q-Iteration | Herrera et al. (2021) |
| **LSM** | Least Squares Monte Carlo | Longstaff & Schwartz (2001) |
| **FQI** | Fitted Q-Iteration | Tsitsiklis & Van Roy (2001) |
| **EOP** | European Option Price (benchmark) | - |

### Deep Learning Baselines

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **DOS** | Deep Optimal Stopping | Becker et al. (2019) |
| **NLSM** | Neural Least Squares Monte Carlo | Becker et al. (2020) |

### Path-Dependent Extensions

| Algorithm | Description |
|-----------|-------------|
| **RRLSM** | Recurrent RLSM (Echo State Networks) |
| **SRLSM** | Special RLSM for path-dependent payoffs |
| **SRFQI** | Special RFQI for path-dependent payoffs |

## Payoff Library (360 Instruments)

The framework implements **360 unique payoff structures** constructed combinatorially:

### Base Payoffs (30)

**Basket Options (d > 1):**
- Simple: BasketCall, BasketPut, GeometricCall, GeometricPut
- Extrema: MaxCall, MinPut (Rainbow options)
- Asian: Fixed/Floating strike variants
- Dispersion: DispersionCall, MaxDispersionCall
- Rank-based: BestOfK, WorstOfK, RankWeighted

**Single Asset Options (d = 1):**
- Vanilla: Call, Put
- Lookback: Fixed/Floating strike
- Asian: Fixed/Floating strike
- Range: RangeCall, RangePut

### Barrier Conditions (12)

| Type | Condition |
|------|-----------|
| None | No barrier (vanilla) |
| UO | Up-and-Out |
| DO | Down-and-Out |
| UI | Up-and-In |
| DI | Down-and-In |
| UODO | Double knock-out |
| UIDI | Double knock-in |
| UIDO | Up-in, Down-out |
| UODI | Up-out, Down-in |
| PTB | Partial-time barrier |
| StepB | Step barrier |
| DStepB | Double step barrier |

### Using Barrier Options

```python
from optimal_stopping.payoffs import BasketCall
from optimal_stopping.payoffs.barrier_wrapper import apply_barrier

# Create base payoff
base_payoff = BasketCall(strike=100)

# Apply Up-and-Out barrier
barrier_payoff = apply_barrier(
    base_payoff,
    barrier_type='UO',
    barrier_level=120
)
```

## Stochastic Models

### Geometric Brownian Motion (GBM)

$$dS_t = (r - q)S_t dt + \sigma S_t dW_t$$

```python
from optimal_stopping.models import BlackScholes

model = BlackScholes(
    drift=0.05,
    volatility=0.2,
    dividend=0.0,
    correlation=0.3,  # For multi-asset
    nb_stocks=10,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)
```

### Heston Stochastic Volatility

$$dS_t = (r-q)S_t dt + \sqrt{v_t} S_t dW^S_t$$
$$dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t} dW^v_t$$

```python
from optimal_stopping.models import Heston

model = Heston(
    drift=0.05,
    volatility=0.2,      # Initial volatility
    mean_reversion=1.5,  # kappa
    long_run_var=0.04,   # theta
    vol_of_vol=0.3,      # xi
    correlation=-0.7,    # rho (leverage effect)
    nb_stocks=1,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)
```

### Fractional Brownian Motion

```python
from optimal_stopping.models import FractionalBrownianMotion

model = FractionalBrownianMotion(
    drift=0.05,
    volatility=0.2,
    hurst=0.7,  # H > 0.5: persistence, H < 0.5: roughness
    nb_stocks=1,
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    spot=100
)
```

### Stationary Block Bootstrap (Real Data)

```python
from optimal_stopping.models import RealDataModel

model = RealDataModel(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    start_date='2010-01-01',
    end_date='2024-01-01',
    nb_paths=100000,
    nb_dates=50,
    maturity=1.0,
    drift=0.05  # Risk-neutral adjustment
)
```

## Hyperparameter Optimization

The `optimization/` module provides Bayesian hyperparameter tuning via Optuna:

```python
from optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    algo_name='RT',
    payoff='MaxCall',
    model='BlackScholes',
    method='tpe',        # Tree-structured Parzen Estimator
    timeout=3600,        # 1 hour
    n_trials=100
)

best_params = optimizer.optimize()
print(f"Best hidden_size: {best_params['hidden_size']}")
print(f"Best activation: {best_params['activation']}")
```

See [optimization/README.md](optimization/README.md) for detailed documentation.

## Reproducing Thesis Results

### Chapter 4 Experiments

To reproduce the benchmark results from Chapter 4:

```bash
# Table 4.2: Algorithmic comparison across dimensions
python -m experiments.runners.run_algo --config thesis_chapter4_table2

# Table 4.3: MaxCall activation function validation
python -m experiments.runners.run_algo --config thesis_chapter4_table3

# Table 4.5-4.7: Barrier option validation
python -m experiments.runners.run_algo --config thesis_chapter4_barriers

# Table 4.8: Path-dependent performance
python -m experiments.runners.run_algo --config thesis_chapter4_path_dependent
```

### Using Pre-computed Datasets

For exact reproduction, download pre-computed path datasets:

1. Download from [Google Drive link] (placeholder)
2. Extract to `data/stored_paths/`
3. Run experiments with `--use_stored_paths` flag

## Visualization

### Convergence Plots

```bash
python -m optimal_stopping.run.plot_convergence --config my_convergence_config
```

### Exercise Boundary Videos

```bash
python -m optimal_stopping.run.create_video --config my_video_config
```

### Comparison Tables

```bash
python -m optimal_stopping.utilities.comparison_table \
    --results results/ \
    --output tables/comparison.tex
```

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{souza2025rt,
  title={The RT Algorithm: Randomized Neural Networks for High-Dimensional American Option Pricing},
  author={Souza, Daniel},
  year={2025},
  school={[Your University]}
}
```

## References

- Herrera, C., Krach, F., Ruigrok, P., & Teichmann, J. (2021). Optimal stopping via randomized neural networks. *arXiv preprint arXiv:2104.13669*.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113-147.
- Becker, S., Cheridito, P., & Jentzen, A. (2019). Deep optimal stopping. *Journal of Machine Learning Research*, 20(74), 1-25.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the codebase of Herrera et al. (2021), with significant extensions for the RT algorithm, payoff library, and experimental infrastructure.
