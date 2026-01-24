# Run Module

> Experiment execution infrastructure and configuration system.
> See [Main README](../../README.md) for repository overview.

---

## Overview

This module provides the complete infrastructure for running experiments:

| Component | File | Purpose |
|-----------|------|---------|
| **Configuration** | `configs.py` | Declarative experiment specification |
| **Execution** | `run_algo.py` | Main experiment runner |
| **Results** | `write_excel.py` | CSV/Excel aggregation |
| **Visualization** | `plot_convergence.py` | Convergence plots |
| **Video** | `create_video.py` | Exercise policy animations |
| **Hyperopt** | `run_hyperopt.py` | Hyperparameter optimization |

---

## Directory Structure

```
run/
├── configs.py              # Experiment configurations (4000+ lines)
├── run_algo.py             # Main execution script
├── run_hyperopt.py         # Hyperparameter optimization
├── write_excel.py          # Results aggregation to Excel
├── write_figures.py        # Paper-quality figure generation
├── plot_convergence.py     # Convergence analysis plots
├── create_video.py         # Exercise policy video generation
└── results/                # Output directory for results
    └── {config_name}/      # Per-experiment results
        ├── *.csv           # Raw results
        ├── *.xlsx          # Aggregated statistics
        └── figures/        # Generated plots
```

---

## Configuration System

### Overview

All experiments are defined in `configs.py` using Python dataclasses. Each parameter is specified as an **iterable**, enabling automatic grid search over all combinations.

### Basic Configuration Structure

Configurations are created by instantiating `_DefaultConfig` with parameter overrides. Any parameter not specified uses its default value (see default values in table below).

```python
# In configs.py

my_experiment = _DefaultConfig(
    # Algorithm selection
    algos=['RT', 'RLSM', 'LSM'],

    # Problem specification
    payoffs=['BasketCall', 'BasketPut'],
    stock_models=['BlackScholes'],

    # Dimensions to test
    nb_stocks=[5, 25, 50, 100],

    # Market parameters (override defaults if needed)
    drift=[0.05],
    volatilities=[0.2],
    correlation=[0.0],
    dividends=[0.0],

    # Option parameters
    spots=[100],
    strikes=[100],
    maturities=[1.0],

    # Simulation parameters (default: 8M paths)
    nb_paths=[1000000],
    nb_dates=[100],
    nb_runs=5,              # Repetitions per configuration

    # Algorithm hyperparameters
    hidden_size=[None],     # None = use dimension heuristic
    activation=['leakyrelu'],
    use_payoff_as_input=[True],
    train_ITM_only=[True],
    dropout=[0.0],

    # Precision
    dtype=['float32'],
)
```

**Neuron Heuristic:** When `hidden_size=[None]`, the RT algorithm uses:
- $d \in [1, 9]$: $K = \max(2d, 5)$
- $d \in [10, 49]$: $K = 1.5d$
- $d \in [50, 99]$: $K = 1.4d$
- $d \in [100, 249]$: $K = 1.3d$
- $d \in [250, 499]$: $K = 1.25d$
- $d \geq 500$: $K = 1.2d$

### Grid Search Behavior

The execution system computes the Cartesian product of all iterable parameters:

```python
# This configuration:
algos = ('RT', 'RLSM')
nb_stocks = (5, 50)
payoffs = ('BasketCall', 'BasketPut')

# Generates 2 × 2 × 2 = 8 experimental configurations
```

### Complete Parameter Reference

#### Algorithm Selection

| Parameter | Type | Description |
|-----------|------|-------------|
| `algos` | `tuple[str]` | Algorithm names: `'RT'`, `'RLSM'`, `'SRLSM'`, `'LSM'`, `'DOS'`, `'NLSM'`, `'EOP'` |

#### Problem Specification

| Parameter | Type | Description |
|-----------|------|-------------|
| `payoffs` | `tuple[str]` | Payoff names from the 360-instrument library |
| `stock_models` | `tuple[str]` | `'BlackScholes'`, `'Heston'`, `'FractionalBrownianMotion'`, `'RoughHeston'`, `'RealData'` |
| `nb_stocks` | `tuple[int]` | Number of underlying assets $d$ |

#### Market Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `drift` | `tuple[float]` | Risk-free rate $r$ |
| `volatilities` | `tuple[float]` | Volatility $\sigma$ |
| `dividends` | `tuple[float]` | Dividend yield $q$ |
| `correlation` | `tuple[float]` | Cross-asset correlation $\rho$ |
| `hurst` | `tuple[float]` | Hurst parameter $H$ (for FBM/Rough Heston) |
| `mean` | `tuple[float]` | Long-run variance $\theta$ (Heston) |
| `speed` | `tuple[float]` | Mean reversion speed $\kappa$ (Heston) |

#### Option Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `spots` | `tuple[float]` | Initial spot price $S_0$ |
| `strikes` | `tuple[float]` | Strike price $K$ |
| `maturities` | `tuple[float]` | Time to maturity $T$ (years) |

#### Simulation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nb_paths` | `tuple[int]` | Number of Monte Carlo paths $m$ |
| `nb_dates` | `tuple[int]` | Number of exercise dates $N$ |
| `nb_runs` | `int` | Number of repetitions per configuration |
| `dtype` | `tuple[str]` | Numerical precision: `'float32'`, `'float64'` |

#### Algorithm Hyperparameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_size` | `tuple[int]` | Hidden layer neurons $K$ |
| `activation` | `tuple[str]` | Activation: `'relu'`, `'leakyrelu'`, `'tanh'`, `'elu'` |
| `use_payoff_as_input` | `tuple[bool]` | Include payoff $g(x)$ in features |
| `use_barrier_as_input` | `tuple[bool]` | Include barrier levels in features |
| `train_ITM_only` | `tuple[bool]` | Train only on ITM paths |
| `dropout` | `tuple[float]` | Dropout probability $\in [0, 1]$ |
| `ridge_coeff` | `tuple[float]` | L2 regularization coefficient |
| `nb_epochs` | `tuple[int]` | Training epochs (for iterative algorithms) |

#### Barrier Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `barriers` | `tuple[float]` | Single barrier level $B$ |
| `barriers_up` | `tuple[float]` | Upper barrier $B_U$ |
| `barriers_down` | `tuple[float]` | Lower barrier $B_L$ |

#### Rank-Based Payoff Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `tuple[int]` | Best/Worst of $k$ parameter |
| `weights` | `tuple[tuple]` | Rank weights (must sum to 1) |

#### Step Barrier Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `step_param1` | `tuple[float]` | Start level or $T_1$ |
| `step_param2` | `tuple[float]` | End level or $T_2$ |
| `step_param3` | `tuple[float]` | Lower start (double step) |
| `step_param4` | `tuple[float]` | Lower end (double step) |

#### Hyperparameter Optimization

| Parameter | Type | Description |
|-----------|------|-------------|
| `enable_hyperopt` | `bool` | Enable Bayesian optimization |
| `hyperopt_method` | `str` | `'tpe'` or `'random'` |
| `hyperopt_timeout` | `float` | Optimization timeout (seconds) |
| `hyperopt_n_trials` | `int` | Maximum number of trials |
| `hyperopt_fidelity_factor` | `int` | Path reduction factor |
| `hyperopt_variance_penalty` | `float` | Variance penalty $\lambda$ |

---

## Running Experiments

### Basic Execution

```bash
# Run a configuration
python -m optimal_stopping.run.run_algo --configs=my_experiment

# Run multiple configurations
python -m optimal_stopping.run.run_algo --configs=exp1,exp2,exp3
```

### Command-Line Options

```bash
python -m optimal_stopping.run.run_algo --configs=my_experiment \
    --nb_jobs=8 \                    # Parallel workers (default: all CPUs - 1)
    --algos=RT,RLSM \                # Filter algorithms
    --nb_stocks=50 \                 # Filter dimensions
    --print_errors \                 # Print full stack traces
    --path_gen_seed=42 \             # Reproducible path generation
    --train_eval_split=2 \           # Train/eval split ratio
    --DEBUG                          # Debug mode
```

### Greeks Calculation

```bash
python -m optimal_stopping.run.run_algo --configs=my_experiment \
    --compute_greeks \
    --greeks_method=central \        # 'central', 'forward', 'backward', 'regression'
    --eps=0.01 \                     # Finite difference epsilon
    --poly_deg=3                     # Polynomial degree for regression method
```

Computes: Delta ($\Delta$), Gamma ($\Gamma$), Theta ($\Theta$), Vega ($\mathcal{V}$), Rho ($\rho$).

### Upper Bound Computation

```bash
python -m optimal_stopping.run.run_algo --configs=my_experiment \
    --compute_upper_bound
```

---

## Results Processing

### Export to Excel

```bash
# Aggregate results to Excel with statistics
python -m optimal_stopping.run.write_excel --configs=my_experiment
```

**Output:** `results/my_experiment/my_experiment_summary.xlsx`

Contains:
- Mean prices per configuration
- Standard deviations
- Min/Max values
- Execution times
- Relative errors (when benchmark available)

### Generate Convergence Plots

```bash
python -m optimal_stopping.run.plot_convergence --configs=my_experiment
```

Generates:
- Price convergence over Monte Carlo paths
- 95% confidence intervals (t-distribution)
- Algorithm comparison plots

### Generate Paper-Quality Figures

```bash
python -m optimal_stopping.run.write_figures --configs=my_experiment
```

---

## Example Configurations

### Dimensional Scalability Study

```python
dimensional_study = _DefaultConfig(
    # Replicate Table 4.2 from thesis
    algos=['RT', 'RLSM', 'LSM', 'DOS', 'NLSM', 'EOP'],
    payoffs=['BasketCall'],
    stock_models=['BlackScholes'],

    nb_stocks=[1, 2, 7, 50, 500],
    nb_paths=[10000000],
    nb_dates=[100],
    nb_runs=5,

    drift=[0.08],
    volatilities=[0.2],
    spots=[100],
    strikes=[100],
    maturities=[1.0],

    hidden_size=[None],  # Use heuristic
    activation=['leakyrelu'],
)
```

### Barrier Option Study

```python
barrier_study = _DefaultConfig(
    # Barrier option monotonicity validation
    algos=['RT', 'SRLSM'],
    payoffs=['UO-BasketCall', 'DO-MaxCall', 'DI-MinPut'],
    stock_models=['BlackScholes'],

    nb_stocks=[5, 25],
    nb_paths=[1000000],
    nb_dates=[50],

    # Barrier sweep
    barriers=[80, 90, 100, 110, 120, 130, 140, 150],

    spots=[100],
    strikes=[100],
)
```

### Path-Dependent Comparison

```python
path_dependent_study = _DefaultConfig(
    # RT vs RRLSM on path-dependent options
    algos=['RT', 'RRLSM', 'SRLSM'],
    payoffs=[
        'LookbackFixedCall',
        'LookbackFloatPut',
        'AsianFixedStrikeCall',
        'AsianFloatingStrikePut',
        'UI-MinPut',
        'DO-MaxCall',
    ],

    nb_stocks=[1, 5, 25],
    activation=['elu'],  # ELU for path-dependent
)
```

### Activation Function Study

```python
activation_study = _DefaultConfig(
    # Validate activation function selection
    algos=['RT'],
    payoffs=['MaxCall', 'MinPut', 'BasketCall'],

    nb_stocks=[5, 25, 250],

    # Grid over activations
    activation=['relu', 'leakyrelu', 'tanh', 'elu'],
    hidden_size=[50, 100, 150],
)
```

### Real Data Validation

```python
real_data_study = _DefaultConfig(
    # Validation with historical market data
    algos=['RT', 'RLSM'],
    payoffs=['BasketCall', 'BasketPut', 'MaxCall'],
    stock_models=['RealData'],

    nb_stocks=[5, 10, 25],
    nb_paths=[500000],
    nb_dates=[50],
    maturities=[0.5],
)
```

---

## Output Structure

```
results/
└── my_experiment/
    ├── my_experiment.csv              # Raw results (all runs)
    ├── my_experiment_summary.xlsx     # Aggregated statistics
    ├── figures/
    │   ├── convergence_RT.png         # Per-algorithm convergence
    │   ├── comparison_d50.png         # Cross-algorithm comparison
    │   └── ...
    └── videos/
        └── exercise_policy.mp4        # Optional: exercise animations
```

### CSV Output Format

| Column | Description |
|--------|-------------|
| `algo` | Algorithm name |
| `payoff` | Payoff name |
| `nb_stocks` | Dimension $d$ |
| `nb_paths` | Monte Carlo paths $m$ |
| `price` | Computed option price |
| `time` | Execution time (seconds) |
| `run_id` | Run number (1 to `nb_runs`) |
| `hidden_size` | Hidden layer size $K$ |
| `activation` | Activation function |
| ... | All configuration parameters |

---

## Parallel Execution

### Multi-Core Processing

```bash
# Use 8 CPU cores
python -m optimal_stopping.run.run_algo --configs=my_experiment --nb_jobs=8

# Use all available cores minus 1 (default)
python -m optimal_stopping.run.run_algo --configs=my_experiment

# Sequential execution (for debugging)
python -m optimal_stopping.run.run_algo --configs=my_experiment --nb_jobs=1
```

### Telegram Notifications

The system supports Telegram notifications for long-running experiments. Configure in `configs.py`:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

---

## Hyperparameter Optimization

### Running Optimization

```bash
python -m optimal_stopping.run.run_hyperopt --configs=my_experiment
```

### Configuration

```python
hpo_experiment = _DefaultConfig(
    algos=['RT'],
    payoffs=['BasketCall'],
    nb_stocks=[50],
    nb_paths=[1000000],

    # Enable hyperparameter optimization
    enable_hyperopt=True,
    hyperopt_method='tpe',           # Bayesian optimization
    hyperopt_timeout=3600,           # 1 hour
    hyperopt_n_trials=100,
    hyperopt_fidelity_factor=4,      # Use 1/4 of paths
    hyperopt_variance_penalty=0.1,
    hyperopt_output_dir='hpo_results',
)
```

See `optimization/README.md` for detailed hyperparameter optimization documentation.

---

## Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Reduce path count
nb_paths: tuple = (100000,)  # Instead of 1000000

# Use float32 precision
dtype: tuple = ('float32',)
```

**Slow Execution:**
```python
# Reduce grid size
nb_stocks: tuple = (50,)     # Single dimension
activation: tuple = ('leakyrelu',)  # Single activation
```

**Reproducibility:**
```bash
# Set random seed
python -m optimal_stopping.run.run_algo --configs=my_experiment --path_gen_seed=42
```

**Algorithm Not Found:**
```python
# Check algorithm name spelling in configs.py
algos: tuple = ('RT', 'RLSM')  # Correct
algos: tuple = ('rt', 'rlsm')  # Incorrect (case-sensitive)
```

---

## References

- Thesis Chapter 4: Experimental methodology
- Thesis Chapter 5: Software implementation details
- Thesis Appendix B: Hyperparameter optimization framework
