"""
Standalone hyperparameter optimization script.

This script runs hyperparameter optimization for a specified algorithm and problem configuration.
Results are saved to the hyperopt_results directory with visualizations and metadata.

Usage:
    python -m optimal_stopping.run.run_hyperopt --config test_hyperopt

Or programmatically:
    from optimal_stopping.run import run_hyperopt
    best_params = run_hyperopt.optimize_config('test_hyperopt')
"""

import sys
import os
from absl import app, flags

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from optimal_stopping.optimization import HyperparameterOptimizer
from optimal_stopping.data import stock_model
from optimal_stopping.payoffs import _PAYOFF_REGISTRY
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
from optimal_stopping.utilities import configs_getter

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "test_hyperopt",
                    "Name of the config to use from configs.py")
flags.DEFINE_string("algo", None,
                    "Override algorithm (optional, uses config default otherwise)")
flags.DEFINE_string("output_dir", None,
                    "Override output directory (optional)")


# Algorithm mapping
_ALGOS = {
    "RLSM": RLSM,
    "SRLSM": SRLSM,
    "RFQI": RFQI,
    "SRFQI": SRFQI,
}

# Stock model mapping
_STOCK_MODELS = stock_model.STOCK_MODELS

# Payoff mapping
_PAYOFFS = _PAYOFF_REGISTRY


def optimize_config(config_name, algo_override=None, output_dir_override=None):
    """
    Run hyperparameter optimization for a given config.

    Args:
        config_name: Name of config from configs.py
        algo_override: Optional algorithm name to override config
        output_dir_override: Optional output directory override

    Returns:
        dict: Best hyperparameters found
    """
    # Get config
    config_dict = dict(configs_getter.get_configs())
    if config_name not in config_dict:
        raise ValueError(
            f"Config '{config_name}' not found. Available configs: {list(config_dict.keys())}"
        )

    config = config_dict[config_name]

    # Check if hyperopt is enabled
    if not getattr(config, 'enable_hyperopt', False):
        raise ValueError(
            f"Hyperparameter optimization is not enabled in config '{config_name}'. "
            f"Set enable_hyperopt=True in the config."
        )

    # Extract single values (hyperopt works on one problem at a time)
    algo_name = algo_override or config.algos[0]
    stock_model_name = config.stock_models[0]
    payoff_name = config.payoffs[0]
    nb_stocks = config.nb_stocks[0]
    nb_paths = config.nb_paths[0]
    nb_dates = config.nb_dates[0]
    maturity = config.maturities[0]
    spot = config.spots[0]
    strike = config.strikes[0]
    volatility = config.volatilities[0]
    drift = config.drift[0]
    dividend = config.dividends[0]
    correlation = config.correlation[0]
    barrier = config.barriers[0]
    barrier_up = config.barriers_up[0] if hasattr(config, 'barriers_up') else None
    barrier_down = config.barriers_down[0] if hasattr(config, 'barriers_down') else None
    dtype = config.dtype[0]
    train_ITM_only = config.train_ITM_only[0]
    use_payoff_as_input = config.use_payoff_as_input[0]
    use_barrier_as_input = config.use_barrier_as_input[0]

    # Get classes
    algo_class = _ALGOS.get(algo_name)
    if algo_class is None:
        raise ValueError(
            f"Algorithm '{algo_name}' not supported for hyperopt. "
            f"Supported: {list(_ALGOS.keys())}"
        )

    stock_model_class = _STOCK_MODELS.get(stock_model_name)
    if stock_model_class is None:
        raise ValueError(f"Stock model '{stock_model_name}' not found")

    payoff_class = _PAYOFFS.get(payoff_name)
    if payoff_class is None:
        raise ValueError(f"Payoff '{payoff_name}' not found")

    # Create payoff instance with all barrier parameters
    payoff_kwargs = {
        'strike': strike,
        'barrier': barrier,
        'spot': spot,
    }
    if barrier_up is not None:
        payoff_kwargs['barrier_up'] = barrier_up
    if barrier_down is not None:
        payoff_kwargs['barrier_down'] = barrier_down

    payoff = payoff_class(**payoff_kwargs)

    # Build problem config
    model_params = {
        'drift': drift,
        'volatility': volatility,
        'spot': spot,
        'dividend': dividend,
        'nb_stocks': nb_stocks,
        'correlation': correlation,
        'dtype': dtype,
    }

    problem_config = {
        'model_params': model_params,
        'payoff': payoff,
        'nb_paths_full': nb_paths,
        'nb_dates': nb_dates,
        'maturity': maturity,
        'train_eval_split': 2,
        'train_ITM_only': train_ITM_only,
        'use_payoff_as_input': use_payoff_as_input,
        'use_barrier_as_input': use_barrier_as_input,
    }

    # Get hyperopt settings
    method = config.hyperopt_method
    timeout = config.hyperopt_timeout
    n_trials = config.hyperopt_n_trials
    variance_penalty = config.hyperopt_variance_penalty
    fidelity_factor = config.hyperopt_fidelity_factor
    output_dir = output_dir_override or config.hyperopt_output_dir

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Config: {config_name}")
    print(f"Algorithm: {algo_name}")
    print(f"Stock Model: {stock_model_name}")
    print(f"Payoff: {payoff_name}")
    print(f"Problem: {nb_stocks} stocks, {nb_dates} dates, {nb_paths} paths")
    print(f"Method: {method.upper()}")
    print(f"Timeout: {timeout}s")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        algo_name=algo_name,
        algo_class=algo_class,
        model_class=stock_model_class,
        problem_config=problem_config,
        method=method,
        timeout=timeout,
        n_trials=n_trials,
        variance_penalty=variance_penalty,
        fidelity_factor=fidelity_factor,
        output_dir=output_dir
    )

    # Run optimization
    best_params = optimizer.optimize()

    # Get nb_epochs_used for RFQI/SRFQI if available
    nb_epochs_used = None
    if hasattr(optimizer.study.best_trial, 'user_attrs') and 'nb_epochs_used' in optimizer.study.best_trial.user_attrs:
        nb_epochs_used = optimizer.study.best_trial.user_attrs['nb_epochs_used']

    # Round float values to 3 decimals for display
    def round_value(v):
        if isinstance(v, float):
            return round(v, 3)
        return v

    rounded_params = {k: round_value(v) for k, v in best_params.items()}
    if nb_epochs_used is not None:
        rounded_params['nb_epochs'] = nb_epochs_used

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best hyperparameters: {rounded_params}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    # Return both best_params and nb_epochs_used
    return best_params, nb_epochs_used


def main(argv):
    """Main entry point for command-line usage."""
    del argv  # Unused

    best_params, nb_epochs_used = optimize_config(
        config_name=FLAGS.config,
        algo_override=FLAGS.algo,
        output_dir_override=FLAGS.output_dir
    )

    # Round float values to 3 decimals for display
    def round_value(v):
        if isinstance(v, float):
            return round(v, 3)
        return v

    rounded_params = {k: round_value(v) for k, v in best_params.items()}
    if nb_epochs_used is not None:
        rounded_params['nb_epochs'] = nb_epochs_used

    print("\nTo use these hyperparameters in your experiments:")
    print("1. Update your config in configs.py with the best values:")
    for param, value in rounded_params.items():
        print(f"   {param} = {value}")
    print("\n2. Run your experiments with the optimized parameters")


if __name__ == '__main__':
    app.run(main)
