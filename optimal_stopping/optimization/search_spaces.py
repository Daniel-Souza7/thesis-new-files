"""
Hyperparameter search space definitions for optimal stopping algorithms.

Defines the search spaces for hyperparameter optimization.
Each algorithm has a specific search space based on its architecture.
"""

# Default search space for randomized neural network algorithms
# Used for: RLSM, SRLSM, RFQI, SRFQI, LSM, FQI
DEFAULT_SEARCH_SPACE = {
    'hidden_size': ('int', 6, 512),  # Number of neurons per layer
    'activation': ('categorical', ['relu', 'tanh', 'elu']),  # Activation function
    'dropout': ('float', 0.0, 0.5),  # Dropout probability
    'ridge_coeff': ('float', 1e-4, 10.0, 'log'),  # Regularization coefficient (log scale)
}

# Extended search space for RFQI (supports multiple layers)
RFQI_SEARCH_SPACE = {
    'hidden_size': ('int', 6, 512),  # Number of neurons per layer
    'num_layers': ('int', 1, 4),  # Number of hidden layers (RFQI only)
    'activation': ('categorical', ['relu', 'tanh', 'elu']),  # Activation function
    'dropout': ('float', 0.0, 0.5),  # Dropout probability between layers
    'ridge_coeff': ('float', 1e-4, 10.0, 'log'),  # Regularization coefficient (log scale)
}

# Search space for RLSM/SRLSM (single layer only)
RLSM_SEARCH_SPACE = {
    'hidden_size': ('int', 6, 512),  # Number of neurons
    'activation': ('categorical', ['relu', 'tanh', 'elu']),  # Activation function
    'dropout': ('float', 0.0, 0.5),  # Dropout (less effect for single layer)
    'ridge_coeff': ('float', 1e-4, 10.0, 'log'),  # Regularization coefficient (log scale)
}


def get_search_space(algo_name):
    """
    Get the appropriate search space for a given algorithm.

    Args:
        algo_name: Algorithm name ('RLSM', 'SRLSM', 'RFQI', 'SRFQI', etc.)

    Returns:
        dict: Search space specification
    """
    if algo_name.upper() in ['RFQI', 'SRFQI']:
        return RFQI_SEARCH_SPACE.copy()
    elif algo_name.upper() in ['RLSM', 'SRLSM']:
        return RLSM_SEARCH_SPACE.copy()
    else:
        # Default for other algorithms
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
