"""
Hyperparameter search space definitions for optimal stopping algorithms.

Defines the search spaces for hyperparameter optimization.
Each algorithm has a specific search space based on its architecture.
"""

# Default search space for randomized neural network algorithms
# Used for: RLSM, SRLSM, LSM, FQI
DEFAULT_SEARCH_SPACE = {
    'hidden_size': ('int', 6, 512),  # Number of neurons per layer
    'activation': ('categorical', ['relu', 'tanh', 'elu', 'leakyrelu', 'softplus', 'gelu']),  # Activation function
}

RLSM_SEARCH_SPACE = DEFAULT_SEARCH_SPACE.copy()


def get_search_space(algo_name):
    """
    Get the appropriate search space for a given algorithm.

    Args:
        algo_name: Algorithm name ('RLSM', 'SRLSM', 'LSM', 'FQI', etc.)

    Returns:
        dict: Search space specification
    """
    # All algorithms use the default search space
    return DEFAULT_SEARCH_SPACE.copy()


def suggest_hyperparameter(trial, name, spec):
    """
    Suggest a hyperparameter value using Optuna trial.

    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Parameter specification tuple

    Returns:
        Suggested parameter value
    """
    param_type = spec[0]

    if param_type == 'int':
        low, high = spec[1], spec[2]
        return trial.suggest_int(name, low, high)

    elif param_type == 'float':
        low, high = spec[1], spec[2]
        log = (spec[3] == 'log') if len(spec) > 3 else False
        return trial.suggest_float(name, low, high, log=log)

    elif param_type == 'categorical':
        choices = spec[1]
        return trial.suggest_categorical(name, choices)

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
