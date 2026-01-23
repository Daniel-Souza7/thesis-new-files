# Experiment Execution

This module provides the main entry points for running pricing experiments, hyperparameter optimization, and visualization.

## Module Structure

```
run/
├── run_algo.py          # Main experiment runner
├── run_hyperopt.py      # Hyperparameter optimization
├── configs.py           # Experiment configurations
├── plot_convergence.py  # Convergence plots
├── create_video.py      # Exercise boundary videos
├── write_figures.py     # LaTeX table generation
└── write_excel.py       # Excel output
```

## Quick Start

### Run Experiments via Command Line

```bash
# Run with default configuration
python -m optimal_stopping.run.run_algo

# Run specific algorithms
python -m optimal_stopping.run.run_algo --algos=RT,RLSM,LSM

# Run specific dimensions
python -m optimal_stopping.run.run_algo --nb_stocks=10,50,100

# Run with debugging
python -m optimal_stopping.run.run_algo --DEBUG=True --print_errors=True
```

### Run with Thesis Configurations

```bash
# Table 4.2: Algorithmic comparison across dimensions
python -m optimal_stopping.run.run_algo --configs=thesis_table_4_2

# Table 4.3: MaxCall activation function validation
python -m optimal_stopping.run.run_algo --configs=thesis_table_4_3

# Tables 4.5-4.7: Barrier options
python -m optimal_stopping.run.run_algo --configs=thesis_barriers
```

---

## `run_algo.py` — Main Experiment Runner

The primary entry point for running pricing experiments.

### Command-Line Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--algos` | list | None | Algorithms to run (e.g., `RT,RLSM,LSM`) |
| `--nb_stocks` | list | None | Dimensions to test (e.g., `10,50,100`) |
| `--nb_jobs` | int | CPU-1 | Number of parallel workers |
| `--print_errors` | bool | False | Print detailed error messages |
| `--path_gen_seed` | int | None | Random seed for reproducibility |
| `--train_eval_split` | int | 2 | Train/eval split ratio |
| `--DEBUG` | bool | False | Enable debug mode (no parallelization) |
| `--generate_pdf` | bool | False | Generate LaTeX tables |
| `--compute_greeks` | bool | False | Compute option Greeks |
| `--compute_upper_bound` | bool | False | Compute upper bound estimate |

### Available Algorithms

```python
# Core algorithms (thesis)
'RT'        # Randomized Thesis algorithm (proposed)
'RLSM'      # Randomized Least Squares Monte Carlo
'RFQI'      # Randomized Fitted Q-Iteration
'LSM'       # Classical Least Squares Monte Carlo
'FQI'       # Fitted Q-Iteration
'EOP'       # European Option Price (benchmark)

# Deep learning baselines
'DOS'       # Deep Optimal Stopping
'NLSM'      # Neural Least Squares Monte Carlo

# Path-dependent algorithms
'SRLSM'     # Special RLSM (for barriers, lookbacks)
'SRFQI'     # Special RFQI (for barriers, lookbacks)
'RRLSM'     # Recurrent RLSM (Echo State Networks)

# Experimental algorithms
'SM'        # Stochastic Mesh
'RSM1'      # Randomized Stochastic Mesh v1
'RSM2'      # Randomized Stochastic Mesh v2
'ZAPQ'      # Zap Q-learning
'DKL'       # Deep Kernel Learning
```

### Algorithm Selection Rules

| Payoff Type | Recommended Algorithms |
|-------------|----------------------|
| Standard (BasketCall, MaxCall, etc.) | RT, RLSM, RFQI, LSM, FQI |
| Barrier (UO_*, DO_*, etc.) | RT, SRLSM, SRFQI |
| Lookback | RT, SRLSM, SRFQI |
| Asian | RT, SRLSM, SRFQI |

**Note:** RT is universal—it works with both standard and path-dependent payoffs.

---

## `configs.py` — Experiment Configurations

Define experiment parameters using the `_DefaultConfig` dataclass.

### Configuration Parameters

```python
@dataclass
class _DefaultConfig:
    # Algorithms
    algos: Iterable[str] = ('RT', 'RLSM', 'LSM', ...)

    # Model parameters
    stock_models: Iterable[str] = ('BlackScholes',)
    drift: Iterable[float] = (0.05,)
    volatilities: Iterable[float] = (0.2,)
    correlation: Iterable[float] = (0.0,)

    # Payoff parameters
    payoffs: Iterable[str] = ('MaxCall',)
    strikes: Iterable[float] = (100,)
    spots: Iterable[float] = (100,)
    barriers: Iterable[float] = (100000,)  # No barrier if very large

    # Monte Carlo parameters
    nb_stocks: Iterable[int] = (10,)
    nb_paths: Iterable[int] = (100000,)
    nb_dates: Iterable[int] = (50,)
    maturities: Iterable[float] = (1.0,)
    nb_runs: int = 10

    # Algorithm hyperparameters
    hidden_size: Iterable[int] = (20,)
    activation: Iterable[str] = ('leakyrelu',)
    train_ITM_only: Iterable[bool] = (True,)
    use_payoff_as_input: Iterable[bool] = (False,)

    # Output
    representations: Iterable[str] = ('TablePriceDuration',)
```

### Creating Custom Configurations

```python
# In configs.py
my_experiment = _DefaultConfig(
    algos=['RT', 'RLSM', 'LSM'],
    stock_models=['BlackScholes'],
    payoffs=['BasketCall', 'MaxCall'],
    nb_stocks=[10, 50, 100],
    nb_paths=[1000000],
    nb_dates=[100],
    nb_runs=5,
    drift=(0.05,),
    volatilities=(0.2,),
    spots=(90, 100, 110),  # ITM, ATM, OTM
    strikes=(100,),
    hidden_size=(20,),
    activation=('leakyrelu',),
)
```

### Example Configurations (Thesis)

```python
# Table 4.2: Dimension scaling
thesis_table_4_2 = _DefaultConfig(
    algos=['RT', 'RLSM', 'LSM', 'DOS', 'NLSM', 'FQI', 'EOP'],
    nb_stocks=[1, 2, 7, 50, 500],
    payoffs=['BasketCall'],
    nb_paths=[8000000, 8000000, 14000000, 10000000, 10000000],
    ...
)

# Barrier options
thesis_barriers = _DefaultConfig(
    algos=['RT', 'SRLSM'],
    payoffs=['UO_BasketCall', 'DO_MaxCall', 'UI_BasketPut'],
    barriers=[80, 90, 100, 110, 120],
    ...
)
```

---

## `run_hyperopt.py` — Hyperparameter Optimization

Bayesian hyperparameter optimization using Optuna.

### Usage

```bash
# Run hyperparameter optimization
python -m optimal_stopping.run.run_hyperopt \
    --algo=RT \
    --payoff=MaxCall \
    --nb_stocks=50 \
    --timeout=3600 \
    --n_trials=100
```

### Configuration

```python
# In configs.py
my_config = _DefaultConfig(
    algos=['RT'],
    payoffs=['MaxCall'],
    nb_stocks=[50],

    # Hyperopt settings
    enable_hyperopt=True,
    hyperopt_method='tpe',        # Tree-structured Parzen Estimator
    hyperopt_timeout=3600,        # 1 hour
    hyperopt_n_trials=100,
    hyperopt_fidelity_factor=4,   # Use nb_paths/4 for speed
    hyperopt_variance_penalty=0.1,
    hyperopt_output_dir='hyperopt_results',
)
```

### Tuned Parameters

| Parameter | Search Range | Description |
|-----------|-------------|-------------|
| `hidden_size` | [5, 200] | Number of hidden neurons |
| `activation` | ['relu', 'tanh', 'elu', 'leakyrelu'] | Activation function |
| `ridge_coeff` | [1e-6, 1e-1] | Regularization coefficient |
| `train_ITM_only` | [True, False] | Filter OTM paths |

---

## `plot_convergence.py` — Convergence Analysis

Generate Monte Carlo convergence plots.

### Usage

```bash
python -m optimal_stopping.run.plot_convergence \
    --results_file=output/metrics_draft/results.csv \
    --output_dir=figures/convergence/
```

### Output

- Price convergence vs. number of paths
- Standard error reduction
- Comparison across algorithms

---

## `create_video.py` — Exercise Boundary Visualization

Create animated visualizations of the exercise boundary evolution.

### Usage

```bash
python -m optimal_stopping.run.create_video \
    --config=video_testing2 \
    --output_dir=videos/
```

### Output

MP4 video showing:
- Stock price evolution
- Exercise boundary at each time step
- Optimal stopping decisions

---

## Output Format

### CSV Output

Results are saved to `output/metrics_draft/TIMESTAMP.csv`:

```csv
algo,model,payoff,drift,volatility,nb_stocks,nb_paths,nb_dates,spot,strike,maturity,price,duration,time_path_gen,comp_time,...
RT,BlackScholes,BasketCall,0.05,0.2,50,1000000,100,100,100,1.0,12.3456,45.2,10.1,35.1,...
```

### Key Columns

| Column | Description |
|--------|-------------|
| `algo` | Algorithm name |
| `model` | Stock model |
| `payoff` | Payoff name |
| `price` | Estimated option price |
| `duration` | Total runtime (seconds) |
| `time_path_gen` | Path generation time |
| `comp_time` | Computation time (excl. path gen) |
| `exercise_time` | Mean exercise time |

---

## Parallel Execution

The runner uses `joblib` for parallel execution:

```python
# Default: use all CPUs minus 1
python -m optimal_stopping.run.run_algo

# Custom parallelization
python -m optimal_stopping.run.run_algo --nb_jobs=8

# Debug mode (no parallelization)
python -m optimal_stopping.run.run_algo --DEBUG=True
```

### Combination Grid

Experiments iterate over all combinations of parameters:

```python
# This creates 3 × 3 × 5 × 10 = 450 tasks
my_config = _DefaultConfig(
    algos=['RT', 'RLSM', 'LSM'],           # 3 algorithms
    nb_stocks=[10, 50, 100],               # 3 dimensions
    payoffs=['BasketCall', 'MaxCall', ...], # 5 payoffs
    nb_runs=10                              # 10 runs each
)
```

---

## Reproducibility

### Setting Seeds

```bash
# Fixed seed for reproducibility
python -m optimal_stopping.run.run_algo --path_gen_seed=42

# Different seeds per run (default behavior)
python -m optimal_stopping.run.run_algo
```

### Using Stored Paths

For exact reproducibility, use pre-computed paths:

```python
# In configs.py
my_config = _DefaultConfig(
    stock_models=['BlackScholesStored1737654321123'],
    ...
)
```

---

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `nb_paths` or use stored paths
2. **Import Error**: Ensure package is installed (`pip install -e .`)
3. **Algorithm Error**: Check algorithm-payoff compatibility

### Debug Mode

```bash
# Enable verbose error output
python -m optimal_stopping.run.run_algo --DEBUG=True --print_errors=True
```

---

## References

- Configuration system: `configs.py`
- Algorithm registry: `optimal_stopping/algorithms/__init__.py`
- Payoff registry: `optimal_stopping/payoffs/__init__.py`
- Model registry: `optimal_stopping/models/__init__.py`
