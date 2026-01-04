"""
Objective function for hyperparameter optimization.

For optimal stopping problems, we MAXIMIZE the validation price (lower bound estimate).
The highest lower bound represents the best policy.

We also penalize variance to prefer stable estimates.
"""

import numpy as np
import time


def evaluate_objective(algo_class, model_class, hyperparams, problem_config,
                        variance_penalty=0.1, n_runs=3, fidelity_factor=4):
    """
    Evaluate objective function for a given set of hyperparameters.

    Objective: Maximize (validation_price - variance_penalty * std_price)

    Args:
        algo_class: Algorithm class (RLSM, SRLSM, RFQI, etc.)
        model_class: Stock model class (BlackScholes, Heston, etc.)
        hyperparams: Dictionary of hyperparameters to evaluate
        problem_config: Dictionary with problem specification
            {
                'model_params': dict,  # Parameters for stock model
                'payoff': payoff object,
                'nb_paths_full': int,  # Full nb_paths for final training
                'nb_dates': int,
                'maturity': float,
                'train_eval_split': int,
                'train_ITM_only': bool,
                'use_payoff_as_input': bool,
                'use_barrier_as_input': bool,
            }
        variance_penalty: Weight for variance penalty (default: 0.1)
        n_runs: Number of runs to average over (default: 1)
        fidelity_factor: Reduction factor for nb_paths during optimization (default: 4)

    Returns:
        float: Objective value (higher is better)
        dict: Metrics dictionary with detailed results
    """
    # Use reduced fidelity (fewer paths) for faster evaluation during optimization
    nb_paths_opt = problem_config['nb_paths_full'] // fidelity_factor

    # Build model with reduced paths
    model_params = problem_config['model_params'].copy()
    model_params['nb_paths'] = nb_paths_opt
    model_params['nb_dates'] = problem_config['nb_dates']
    model_params['maturity'] = problem_config['maturity']

    # Run multiple evaluations to estimate variance
    prices = []
    times = []

    for run in range(n_runs):
        # Create fresh model instance
        model = model_class(**model_params)

        # Create algorithm instance with current hyperparameters
        algo_params = {
            'model': model,
            'payoff': problem_config['payoff'],
            'hidden_size': hyperparams.get('hidden_size', 50),
            'train_ITM_only': problem_config.get('train_ITM_only', True),
            'use_payoff_as_input': problem_config.get('use_payoff_as_input', True),
            'use_barrier_as_input': problem_config.get('use_barrier_as_input', False),
            'activation': hyperparams.get('activation', 'leakyrelu'),
            'dropout': hyperparams.get('dropout', 0.0),
            'ridge_coeff': hyperparams.get('ridge_coeff', 0.0),  # ✅ FIXED: Now actually passed!
        }

        # Add algorithm-specific parameters
        if 'num_layers' in hyperparams:
            # RFQI/SRFQI supports multiple layers
            algo_params['num_layers'] = hyperparams['num_layers']

        if 'nb_epochs' in hyperparams:
            # RFQI/SRFQI uses iterative training
            algo_params['nb_epochs'] = hyperparams['nb_epochs']

        if 'early_stopping_callback' in hyperparams:
            # RFQI/SRFQI supports early stopping
            algo_params['early_stopping_callback'] = hyperparams['early_stopping_callback']

        # Create algorithm instance
        algo = algo_class(**algo_params)

        # Price the option (uses train/eval split internally)
        t_start = time.time()
        price, time_path_gen = algo.price(
            train_eval_split=problem_config.get('train_eval_split', 2)
        )
        comp_time = time.time() - t_start

        prices.append(price)
        times.append(comp_time)

        # Track epochs used (for RFQI/SRFQI with early stopping)
        if hasattr(algo, '_epochs_used'):
            if 'epochs_used' not in locals():
                epochs_used = []
            epochs_used.append(algo._epochs_used)

    # Compute statistics
    mean_price = np.mean(prices)
    std_price = np.std(prices) if n_runs > 1 else 0.0
    mean_time = np.mean(times)

    # Objective: Maximize price (lower bound) while penalizing variance
    # Higher price = better lower bound = better policy
    objective_value = mean_price - variance_penalty * std_price

    # Detailed metrics for logging
    metrics = {
        'mean_price': mean_price,
        'std_price': std_price,
        'mean_time': mean_time,
        'objective': objective_value,
        'n_runs': n_runs,
        'nb_paths_used': nb_paths_opt,
    }

    # Add epochs used for RFQI/SRFQI
    if 'epochs_used' in locals():
        metrics['nb_epochs_used'] = int(np.mean(epochs_used))  # Average across runs
        metrics['nb_epochs_std'] = float(np.std(epochs_used)) if len(epochs_used) > 1 else 0.0

    return objective_value, metrics


def evaluate_objective_with_early_stopping(algo_class, model_class, hyperparams,
                                             problem_config, variance_penalty=0.1,
                                             n_runs=1, fidelity_factor=4,
                                             early_stopping_config=None):
    """
    Evaluate objective for algorithms that support early stopping (RFQI, SRFQI).

    Integrates early stopping callback into the algorithm training loop.

    Args:
        algo_class: Algorithm class (RFQI, SRFQI, etc.)
        model_class: Stock model class
        hyperparams: Dictionary of hyperparameters
        problem_config: Problem specification
        variance_penalty: Weight for variance penalty
        n_runs: Number of runs to average
        fidelity_factor: Reduction factor for nb_paths
        early_stopping_config: Dict with early stopping parameters
            {
                'patience': int,
                'min_delta': float,
                'max_epochs': int,
            }

    Returns:
        float: Objective value
        dict: Metrics dictionary
    """
    from .early_stopping import EarlyStopping

    # If early stopping config provided, set up callback
    if early_stopping_config is not None:
        # Set max_epochs as nb_epochs
        hyperparams = hyperparams.copy()
        hyperparams['nb_epochs'] = early_stopping_config.get('max_epochs', 100)

        # Create early stopping callback
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 5),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            mode='max'  # Maximize validation score
        )
        hyperparams['early_stopping_callback'] = early_stopping

    # Use standard evaluation with early stopping integrated
    return evaluate_objective(
        algo_class, model_class, hyperparams, problem_config,
        variance_penalty, n_runs, fidelity_factor
    )
