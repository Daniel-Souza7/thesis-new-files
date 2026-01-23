# Experiments

This directory contains experiment configurations and results for reproducing thesis experiments.

## Directory Structure

```
experiments/
├── configs/           # Thesis experiment configurations
│   ├── thesis_chapter4.py    # Chapter 4 benchmark experiments
│   └── custom_example.py     # Template for custom experiments
├── results/           # Output directory for experiment results
│   ├── *.csv          # Price estimates and metrics
│   └── *.xlsx         # Aggregated results
└── README.md          # This file
```

## Running Experiments

### Using Pre-defined Configurations

The main execution script is located at `optimal_stopping/run/run_algo.py`:

```bash
# Run thesis Chapter 4 experiments
python -m optimal_stopping.run.run_algo --config thesis_basket_call

# Run with specific parameters
python -m optimal_stopping.run.run_algo \
    --algos RT RLSM LSM \
    --payoff BasketCall \
    --nb_stocks 50 \
    --nb_paths 1000000
```

### Configuration File Format

Experiment configurations are Python files that define parameter combinations:

```python
# experiments/configs/my_experiment.py

from optimal_stopping.run.configs import _DefaultConfig

my_config = _DefaultConfig(
    # Algorithms to compare
    algos=['RT', 'RLSM', 'LSM', 'DOS', 'NLSM'],

    # Stochastic models
    stock_models=['BlackScholes'],

    # Payoff structures
    payoffs=['BasketCall', 'BasketPut', 'MaxCall'],

    # Dimensions to test
    nb_stocks=[1, 7, 50, 500],

    # Monte Carlo parameters
    nb_paths=[10000000],
    nb_dates=[100],
    nb_runs=10,

    # Market parameters
    drift=0.08,
    volatility=0.2,
    spot=100,
    strike=100,
    maturity=1.0,
)
```

### Output Format

Results are saved to `experiments/results/` as CSV files with columns:
- `algo`: Algorithm name
- `payoff`: Payoff type
- `nb_stocks`: Number of assets
- `price`: Option price estimate
- `std_err`: Standard error
- `time`: Computation time (seconds)
- `exercise_time`: Mean exercise time

## Reproducing Thesis Results

### Table 4.2: Algorithmic Comparison

```bash
python -m optimal_stopping.run.run_algo --config thesis_table_4_2
```

This runs basket call pricing across dimensions d = {1, 2, 7, 50, 500} for all algorithms.

### Table 4.3: MaxCall Activation Validation

```bash
python -m optimal_stopping.run.run_algo --config thesis_table_4_3
```

Compares RT (ELU) vs RLSM (LeakyReLU) on MaxCall options.

### Tables 4.5-4.7: Barrier Option Validation

```bash
python -m optimal_stopping.run.run_algo --config thesis_barriers
```

Tests barrier convergence and monotonicity.

### Table 4.8: Path-Dependent Performance

```bash
python -m optimal_stopping.run.run_algo --config thesis_path_dependent
```

Compares RT vs RRLSM on lookback, Asian, and exotic options.

## Using Pre-computed Paths

For exact reproducibility, use pre-computed path datasets:

1. Download datasets from [link placeholder]
2. Place in `data/stored_paths/`
3. Add `--use_stored_paths` flag:

```bash
python -m optimal_stopping.run.run_algo --config thesis_table_4_2 --use_stored_paths
```

## Visualization

After running experiments, generate plots:

```bash
# Convergence plots
python -m optimal_stopping.run.plot_convergence \
    --results experiments/results/my_experiment.csv \
    --output experiments/results/convergence.png

# Comparison tables (LaTeX)
python -m optimal_stopping.utilities.comparison_table \
    --results experiments/results/ \
    --output experiments/results/comparison.tex
```

## Creating Videos

Visualize exercise boundaries:

```bash
python -m optimal_stopping.run.create_video \
    --algo RT \
    --payoff BasketPut \
    --output experiments/results/exercise_boundary.mp4
```
