# Hyperparameter Optimization for Optimal Stopping Algorithms

This module provides automated hyperparameter tuning for reinforcement learning-based American option pricing algorithms (RLSM, SRLSM, RFQI, SRFQI).

## Overview

Traditional hyperparameter selection relies on manual trial-and-error, which is:
- Time-consuming and doesn't scale
- Suboptimal (misses better configurations)
- Not reproducible or systematic

This module automates the process using **Bayesian Optimization (TPE)** via Optuna, providing:
- Systematic search through hyperparameter space
- Multi-fidelity optimization (faster trials with reduced paths)
- Automatic result logging and visualization
- Reproducible experiments with metadata tracking

## Key Features

### Supported Algorithms
- **RLSM** (Randomized Least Squares Monte Carlo) - single layer
- **SRLSM** (Special RLSM for path-dependent options) - single layer
- **RFQI** (Randomized Fitted Q-Iteration) - supports 1-4 layers

### Hyperparameters Optimized

**For RLSM/SRLSM** (single layer networks):
- `hidden_size`: Number of neurons (6-512)
- `activation`: Activation function ('relu', 'tanh', 'elu')
- `dropout`: Dropout probability (0.0-0.5) **⚠️ Experimental: See note below**
- `ridge_coeff`: Regularization (1e-4 to 10.0, log scale)

**For RFQI** (multi-layer networks):
- `hidden_size`: Neurons per layer (6-512)
- `num_layers`: Number of hidden layers (1-4)
- `activation`: Activation function ('relu', 'tanh', 'elu')
- `dropout`: Dropout between layers (0.0-0.5)
- `ridge_coeff`: Regularization (1e-4 to 10.0, log scale)

### Optimization Methods

1. **TPE (Tree-structured Parzen Estimator)** - Default, recommended
   - Bayesian optimization using probabilistic models
   - More sample-efficient than random search
   - Adaptively explores promising regions

2. **Random Search** - Baseline for comparison
   - Uniform random sampling
   - Good for establishing baselines

## Quick Start

### 1. Define a Configuration

In `optimal_stopping/run/configs.py`:

```python
test_hyperopt = _DefaultConfig(
    algos=['RLSM'],
    stock_models=['BlackScholes'],
    payoffs=['MaxCall'],
    nb_stocks=[2],
    nb_paths=[50000],  # Full paths for final training
    nb_dates=[20],
    enable_hyperopt=True,  # Enable optimization
    hyperopt_method='tpe',  # Use Bayesian optimization
    hyperopt_timeout=1200,  # 20 minutes
    hyperopt_fidelity_factor=4,  # Use nb_paths/4 during optimization
)
```

### 2. Run Optimization

```bash
python -m optimal_stopping.run.run_hyperopt --config test_hyperopt
```

### 3. Results

Results are saved to `hyperopt_results/` with:
- **SQLite database**: Full Optuna study for advanced analysis
- **JSON summary**: Machine-readable metadata and best parameters
- **Text summary**: Human-readable report with git hash, config, results
- **Visualizations**:
  - Optimization history (objective value over trials)
  - Parameter importances (which hyperparameters matter most)
  - Slice plots (where good parameters cluster)
  - Parallel coordinate plots (parameter relationships)

### 4. Use Optimized Parameters

Update your config with the best hyperparameters:

```python
production_config = _DefaultConfig(
    algos=['RLSM'],
    hidden_size=[128],  # From optimization results
    activation='relu',   # Custom activation (passed to algorithm)
    dropout=0.15,       # Custom dropout (passed to algorithm)
    # ... rest of config
)
```

## How It Works

### Three-Phase Workflow

**Phase 1: Hyperparameter Optimization (40% of time)**
- Uses reduced-fidelity evaluation (nb_paths/4 for speed)
- Optuna explores hyperparameter space using TPE
- Each trial trains and evaluates with candidate hyperparameters
- Objective: **Maximize validation price** (higher lower bound = better policy)
- Variance penalty: Prefer stable estimates over noisy ones

**Phase 2: Final Training (30% of time)**
- Train final model with optimized hyperparameters
- Uses full nb_paths for high-quality training
- Standard train/eval split

**Phase 3: Evaluation (30% of time)**
- Evaluate final model on test set
- Report unbiased performance metrics

### Objective Function

```
objective = validation_price - variance_penalty * std(price)
```

Why maximize price (not minimize error vs reference)?
- In optimal stopping, we compute **lower bounds** on the true option price
- Higher lower bound = better policy
- We don't need a reference price for optimization!
- Reference prices are only needed for final thesis validation

### Multi-Fidelity Optimization

During Phase 1, we use `nb_paths / fidelity_factor` to speed up trials:
- Default: `fidelity_factor=4` → Use 1/4 of full paths
- 4x faster trials → Explore more configurations
- Hyperparameter rankings stay similar with fewer paths

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_hyperopt` | bool | False | Enable hyperparameter optimization |
| `hyperopt_method` | str | 'tpe' | Optimization method ('tpe' or 'random') |
| `hyperopt_timeout` | float | 1200 | Timeout in seconds (20 min) |
| `hyperopt_n_trials` | int | None | Number of trials (None = run until timeout) |
| `hyperopt_fidelity_factor` | int | 4 | Path reduction factor (1/4 of nb_paths) |
| `hyperopt_variance_penalty` | float | 0.1 | Variance penalty weight |
| `hyperopt_output_dir` | str | 'hyperopt_results' | Output directory |

## Advanced Usage

### Programmatic API

```python
from optimal_stopping.optimization import HyperparameterOptimizer
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.payoffs import MaxCall

# Define problem
payoff = MaxCall(strike=100)
problem_config = {
    'model_params': {'drift': 0.06, 'volatility': 0.2, ...},
    'payoff': payoff,
    'nb_paths_full': 50000,
    'nb_dates': 20,
    'maturity': 1.0,
}

# Create optimizer
optimizer = HyperparameterOptimizer(
    algo_name='RLSM',
    algo_class=RLSM,
    model_class=BlackScholes,
    problem_config=problem_config,
    method='tpe',
    timeout=1200
)

# Run optimization
best_params = optimizer.optimize()
print(f"Best hyperparameters: {best_params}")
```

### Analyzing Results

```python
import optuna

# Load study from database
study = optuna.load_study(
    study_name='RLSM_tpe_20250122_143021',
    storage='sqlite:///hyperopt_results/RLSM_tpe_20250122_143021.db'
)

# Get best trial
best_trial = study.best_trial
print(f"Best value: {best_trial.value}")
print(f"Best params: {best_trial.params}")

# Analyze all trials
import pandas as pd
df = study.trials_dataframe()
print(df[['number', 'value', 'params_hidden_size', 'params_activation']])

# Custom visualizations
import optuna.visualization as vis
fig = vis.plot_contour(study, params=['hidden_size', 'dropout'])
fig.show()
```

### Custom Search Spaces

Edit `optimal_stopping/optimization/search_spaces.py`:

```python
CUSTOM_SEARCH_SPACE = {
    'hidden_size': ('int', 10, 256),  # Narrower range
    'activation': ('categorical', ['relu', 'elu']),  # Fewer choices
    'dropout': ('float', 0.0, 0.3),  # Lower max dropout
    'ridge_coeff': ('float', 0.001, 1.0, 'log'),
}
```

## Important Notes

### ⚠️ Dropout in Randomized (Frozen) Neural Networks

**Context:** RLSM/SRLSM/RFQI use **Extreme Learning Machines** (randomized neural networks) where weights are randomly initialized and then **frozen** - they are never trained via backpropagation.

**The Concern:** Dropout is traditionally a regularization technique for trainable networks. When applied to frozen networks:
- During training: Dropout randomly zeros features in the basis function matrix H
- This acts more like "data augmentation" than traditional dropout regularization
- May destabilize the ridge regression solution rather than improve it
- Standard dropout behavior (train vs eval mode) may not apply as intended

**Recommendation:**
- Include dropout in the search space to test empirically
- Monitor optimization results: if dropout=0.0 is consistently selected, it confirms dropout is not helpful for this architecture
- **Ridge regularization (`ridge_coeff`) is the primary regularization mechanism** - focus on optimizing this parameter

**Why it's still in the search space:**
- Some research suggests dropout on frozen random features can work
- Let the optimizer decide empirically whether it helps
- If harmful, TPE will learn to avoid non-zero dropout values

## Troubleshooting

### Issue: Optimization finds poor hyperparameters

**Solutions:**
1. Increase timeout: `hyperopt_timeout=3600` (1 hour)
2. Check if multi-fidelity approximation is too aggressive: `hyperopt_fidelity_factor=2`
3. Verify problem is well-specified (correct payoff, model, etc.)
4. Try random search baseline to verify TPE is working

### Issue: Visualizations not generating

**Solution:** Install dependencies:
```bash
pip install plotly kaleido
```

### Issue: Out of memory during optimization

**Solutions:**
1. Reduce `nb_paths`: Use smaller paths for both optimization and final training
2. Increase `hyperopt_fidelity_factor`: Use even fewer paths during optimization
3. Use float32 dtype: `dtype='float32'`

### Issue: Trials timing out

**Solution:** Increase per-trial timeout is not directly supported, but you can:
1. Reduce problem size (fewer paths, dates)
2. Use simpler algorithms (RLSM instead of RFQI)

## Implementation Details

### Modified Algorithm Classes

All algorithms (RLSM, SRLSM, RFQI) now accept:
- `activation`: String or torch.nn.Module ('relu', 'tanh', 'elu', 'leakyrelu')
- `dropout`: Float (0.0-1.0), probability of dropping neurons
- `num_layers`: Int (RFQI only, 1-4 layers)

### Reservoir2 Neural Network

The `Reservoir2` class (randomized neural network) has been extended to support:
- Multiple layers: `num_layers=1,2,3,4`
- Different activations: Configured via string or module
- Dropout between layers: Standard PyTorch dropout

### Constraints

- **RLSM/SRLSM**: Always use `num_layers=1` (single layer architecture)
- **RFQI**: Supports `num_layers=1-4` (multi-layer architecture)
- `nb_epochs` for RFQI: Can be optimized, but early stopping is recommended instead

## References

- Optuna: https://optuna.readthedocs.io/
- TPE Algorithm: Bergstra et al. (2011), "Algorithms for Hyper-Parameter Optimization"
- RLSM/RFQI: Herrera et al. (2021), "Optimal stopping via randomized neural networks"

## Future Enhancements

- [ ] Integrate early stopping for RFQI (monitor validation loss per epoch)
- [ ] Support for additional algorithms (LSM, FQI with polynomial bases)
- [ ] Parallel trial execution (multiple GPUs/processes)
- [ ] Transfer learning (warm-start from similar problems)
- [ ] Auto-determination of nb_paths based on variance targets
