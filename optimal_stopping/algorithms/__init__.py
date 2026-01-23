"""
Pricing algorithms for American option valuation.

This package provides implementations of various algorithms for pricing
American-style derivatives via optimal stopping:

Core Algorithms (Thesis):
    - RT: Randomized Thesis algorithm (proposed)
    - RLSM: Randomized Least Squares Monte Carlo
    - RFQI: Randomized Fitted Q-Iteration
    - LSM: Least Squares Monte Carlo
    - FQI: Fitted Q-Iteration
    - EOP: European Option Price benchmark

Deep Learning Baselines:
    - DOS: Deep Optimal Stopping
    - NLSM: Neural Least Squares Monte Carlo

Recurrent (Path-Dependent):
    - RRLSM: Recurrent RLSM
    - SRLSM: Special RLSM
    - SRFQI: Special RFQI

Experimental:
    - SM, RSM1, RSM2: Stochastic mesh methods
    - ZAPQ, RZAPQ: Zap Q-learning
    - DKL, RDKL: Deep kernel learning

Example usage:
    >>> from optimal_stopping.algorithms import RT, RLSM, LSM
    >>> from optimal_stopping.models import BlackScholes
    >>> from optimal_stopping.payoffs import BasketCall
    >>>
    >>> model = BlackScholes(nb_stocks=10, nb_paths=100000, ...)
    >>> payoff = BasketCall(strike=100)
    >>> rt = RT(model=model, payoff=payoff)
    >>> price, std_err = rt.price(model.generate_paths())
"""

# Core algorithms (thesis main methods)
from optimal_stopping.algorithms.core import (
    RT,
    RLSM,
    RFQI,
    LSM,
    FQI,
    EOP,
)

# Deep learning baselines
from optimal_stopping.algorithms.deep import (
    DOS,
    NLSM,
)

# Recurrent algorithms for path-dependent options
from optimal_stopping.algorithms.recurrent import (
    RRLSM,
    SRLSM,
    SRFQI,
)

# Algorithm registry for dynamic loading
ALGORITHM_REGISTRY = {
    # Core
    'RT': RT,
    'RLSM': RLSM,
    'RFQI': RFQI,
    'LSM': LSM,
    'FQI': FQI,
    'EOP': EOP,
    # Deep
    'DOS': DOS,
    'NLSM': NLSM,
    # Recurrent
    'RRLSM': RRLSM,
    'SRLSM': SRLSM,
    'SRFQI': SRFQI,
}

# Path-dependent algorithm names (for automatic routing)
PATH_DEPENDENT_ALGOS = {'RRLSM', 'SRLSM', 'SRFQI'}

# All public exports
__all__ = [
    # Core
    'RT',
    'RLSM',
    'RFQI',
    'LSM',
    'FQI',
    'EOP',
    # Deep
    'DOS',
    'NLSM',
    # Recurrent
    'RRLSM',
    'SRLSM',
    'SRFQI',
    # Registry
    'ALGORITHM_REGISTRY',
    'PATH_DEPENDENT_ALGOS',
]


def get_algorithm(name: str):
    """
    Get algorithm class by name.

    Args:
        name: Algorithm name (e.g., 'RT', 'RLSM', 'LSM')

    Returns:
        Algorithm class

    Raises:
        ValueError: If algorithm name is not recognized
    """
    if name not in ALGORITHM_REGISTRY:
        available = ', '.join(sorted(ALGORITHM_REGISTRY.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return ALGORITHM_REGISTRY[name]
