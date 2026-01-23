"""
Stochastic process models for asset price simulation.

This module provides implementations of various stochastic models used
to generate Monte Carlo paths for option pricing:

- BlackScholes: Geometric Brownian Motion (standard model)
- Heston: Stochastic volatility model
- HestonWithVar: Heston model returning variance paths
- FractionalBrownianMotion: Non-Markovian fractional model
- RoughHeston: Rough volatility Heston model
- RoughHestonWithVar: Rough Heston returning variance paths
- RealDataModel: Stationary Block Bootstrap from historical data

Example usage:
    >>> from optimal_stopping.models import BlackScholes
    >>>
    >>> model = BlackScholes(
    ...     drift=0.05,
    ...     volatility=0.2,
    ...     nb_stocks=10,
    ...     nb_paths=100000,
    ...     nb_dates=50,
    ...     maturity=1.0,
    ...     spot=100
    ... )
    >>> paths = model.generate_paths()
"""

from optimal_stopping.models.stock_model import (
    BlackScholes,
    Heston,
    HestonWithVar,
    FractionalBrownianMotion,
    FractionalBlackScholes,
    RoughHeston,
    RoughHestonWithVar,
)
from optimal_stopping.models.real_data import RealDataModel

# Model registry for dynamic loading
MODEL_REGISTRY = {
    'BlackScholes': BlackScholes,
    'Heston': Heston,
    'HestonWithVar': HestonWithVar,
    'FractionalBrownianMotion': FractionalBrownianMotion,
    'FractionalBlackScholes': FractionalBlackScholes,
    'RoughHeston': RoughHeston,
    'RoughHestonWithVar': RoughHestonWithVar,
    'RealDataModel': RealDataModel,
}

__all__ = [
    # Core models
    'BlackScholes',
    'Heston',
    'HestonWithVar',
    'FractionalBrownianMotion',
    'FractionalBlackScholes',
    'RoughHeston',
    'RoughHestonWithVar',
    'RealDataModel',
    # Registry
    'MODEL_REGISTRY',
]


def get_model(name: str):
    """
    Get model class by name.

    Args:
        name: Model name (e.g., 'BlackScholes', 'Heston')

    Returns:
        Model class

    Raises:
        ValueError: If model name is not recognized
    """
    if name not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]
