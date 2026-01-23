"""
Optimal Stopping - American Option Pricing with Randomized Neural Networks.

This package provides implementations of the RT algorithm and other methods
for pricing American-style derivatives in high-dimensional settings.

Modules:
    algorithms: Pricing algorithms (RT, RLSM, LSM, DOS, etc.)
    models: Stochastic process models (BlackScholes, Heston, etc.)
    payoffs: Payoff structures (360 instruments)
    storage: Path storage and data management

Quick Start:
    >>> from optimal_stopping.algorithms import RT
    >>> from optimal_stopping.models import BlackScholes
    >>> from optimal_stopping.payoffs import BasketCall
    >>>
    >>> model = BlackScholes(nb_stocks=10, nb_paths=100000, ...)
    >>> payoff = BasketCall(strike=100)
    >>> rt = RT(model=model, payoff=payoff)
    >>> price, time = rt.price()
"""

__version__ = "1.0.0"
__author__ = "Daniel Souza"

# Convenience imports
from optimal_stopping.algorithms import (
    RT,
    RLSM,
    RFQI,
    LSM,
    FQI,
    EOP,
    DOS,
    NLSM,
    RRLSM,
    SRLSM,
    SRFQI,
    ALGORITHM_REGISTRY,
)

from optimal_stopping.models import (
    BlackScholes,
    Heston,
    HestonWithVar,
    FractionalBrownianMotion,
    RoughHeston,
    RoughHestonWithVar,
    MODEL_REGISTRY,
)

from optimal_stopping.payoffs import (
    get_payoff_class,
    _PAYOFF_REGISTRY,
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    # Algorithms
    'RT',
    'RLSM',
    'RFQI',
    'LSM',
    'FQI',
    'EOP',
    'DOS',
    'NLSM',
    'RRLSM',
    'SRLSM',
    'SRFQI',
    'ALGORITHM_REGISTRY',
    # Models
    'BlackScholes',
    'Heston',
    'HestonWithVar',
    'FractionalBrownianMotion',
    'RoughHeston',
    'RoughHestonWithVar',
    'MODEL_REGISTRY',
    # Payoffs
    'get_payoff_class',
    '_PAYOFF_REGISTRY',
]
