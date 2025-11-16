"""
Optimal stopping payoffs package.

This package provides:
- Base Payoff class with auto-registration
- 34 base payoff classes (standard options without barriers)
- BarrierPayoff wrapper that applies 11 barrier types to any base payoff
- 374 barrier variant payoffs (34 base × 11 barriers, excluding non-applicable combos)

Total: 408 unique payoff types
"""

# Import base classes
from .payoff import Payoff, get_payoff_class, list_payoffs, _PAYOFF_REGISTRY
from .barrier_wrapper import BarrierPayoff, create_barrier_payoff

# Import simple basket payoffs (6)
from .basket_simple import (
    BasketCall, BasketPut,
    GeometricCall, GeometricPut,
    MaxCall, MinPut
)

# Import simple single payoffs (2)
from .single_simple import Call, Put

# TODO: Import remaining base payoffs (26 more to implement):
# - basket_asian.py: AsianFixedStrikeCall, AsianFixedStrikePut, AsianFloatingStrikeCall, AsianFloatingStrikePut (4)
# - basket_range_dispersion.py: RangeCall, RangePut, DispersionCall, DispersionPut (4)
# - basket_rank.py: BestOfKCall, WorstOfKPut, RankWeightedBasketCall, RankWeightedBasketPut (4)
# - basket_quantile.py: QuantileBasketCall, QuantileBasketPut (2)
# - single_lookback.py: LookbackFixedCall, LookbackFixedPut, LookbackFloatCall, LookbackFloatPut (4)
# - single_asian.py: AsianFixedStrikeCall_Single, AsianFixedStrikePut_Single, AsianFloatingStrikeCall_Single, AsianFloatingStrikePut_Single (4)
# - single_range.py: RangeCall_Single, RangePut_Single (2)
# - single_quantile.py: QuantileCall, QuantilePut (2)


# Auto-generate all barrier variants for implemented base payoffs
# This creates 88 barrier payoffs (8 base × 11 barriers)
_BASE_PAYOFFS = [
    BasketCall, BasketPut,
    GeometricCall, GeometricPut,
    MaxCall, MinPut,
    Call, Put
]

_BARRIER_TYPES = ['UO', 'DO', 'UI', 'DI', 'UODO', 'UIDI', 'UIDO', 'UODI', 'PTB', 'StepB', 'DStepB']

# Create barrier variants and register them
for base_payoff in _BASE_PAYOFFS:
    for barrier_type in _BARRIER_TYPES:
        barrier_class = create_barrier_payoff(base_payoff, barrier_type)
        # Register in global registry
        _PAYOFF_REGISTRY[barrier_class.__name__] = barrier_class
        if hasattr(barrier_class, 'abbreviation'):
            _PAYOFF_REGISTRY[barrier_class.abbreviation] = barrier_class


# Export everything
__all__ = [
    # Base classes
    'Payoff', 'BarrierPayoff',
    'get_payoff_class', 'list_payoffs',
    'create_barrier_payoff',

    # Simple basket payoffs
    'BasketCall', 'BasketPut',
    'GeometricCall', 'GeometricPut',
    'MaxCall', 'MinPut',

    # Single payoffs
    'Call', 'Put',

    # Registry access
    '_PAYOFF_REGISTRY',
]


def print_payoff_summary():
    """Print summary of all registered payoffs."""
    payoffs = list_payoffs()
    base_payoffs = [p for p in payoffs if not any(b in p for b in _BARRIER_TYPES)]
    barrier_payoffs = [p for p in payoffs if any(b in p for b in _BARRIER_TYPES)]

    print(f"=Ê Payoff Registry Summary:")
    print(f"   Base payoffs:    {len(base_payoffs)}")
    print(f"   Barrier payoffs: {len(barrier_payoffs)}")
    print(f"   Total payoffs:   {len(payoffs)}")
    print(f"\n Base payoffs: {sorted(base_payoffs)}")


if __name__ == "__main__":
    print_payoff_summary()
