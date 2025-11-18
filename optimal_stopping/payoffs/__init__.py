"""
Optimal stopping payoffs package.

This package provides:
- Base Payoff class with auto-registration
- 30 base payoff classes (standard options without barriers)
- BarrierPayoff wrapper that applies 11 barrier types to any base payoff
- 330 barrier variant payoffs (30 base × 11 barriers)

Total: 360 unique payoff types
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

# Import basket Asian payoffs (4)
from .basket_asian import (
    AsianFixedStrikeCall, AsianFixedStrikePut,
    AsianFloatingStrikeCall, AsianFloatingStrikePut
)

# Import basket range & dispersion payoffs (4)
from .basket_range_dispersion import (
    RangeCall, RangePut,
    DispersionCall, DispersionPut
)

# Import basket rank payoffs (4)
from .basket_rank import (
    BestOfKCall, WorstOfKPut,
    RankWeightedBasketCall, RankWeightedBasketPut
)

# Import simple single payoffs (2)
from .single_simple import Call, Put

# Import single lookback payoffs (4)
from .single_lookback import (
    LookbackFixedCall, LookbackFixedPut,
    LookbackFloatCall, LookbackFloatPut
)

# Import single Asian payoffs (4)
from .single_asian import (
    AsianFixedStrikeCall_Single, AsianFixedStrikePut_Single,
    AsianFloatingStrikeCall_Single, AsianFloatingStrikePut_Single
)

# Import single range payoffs (2)
from .single_range import (
    RangeCall_Single, RangePut_Single
)


# Auto-generate all barrier variants for all 30 base payoffs
# This creates 330 barrier payoffs (30 base × 11 barriers)
_BASE_PAYOFFS = [
    # Simple basket (6)
    BasketCall, BasketPut,
    GeometricCall, GeometricPut,
    MaxCall, MinPut,

    # Basket Asian (4)
    AsianFixedStrikeCall, AsianFixedStrikePut,
    AsianFloatingStrikeCall, AsianFloatingStrikePut,

    # Basket Range & Dispersion (4)
    RangeCall, RangePut,
    DispersionCall, DispersionPut,

    # Basket Rank (4)
    BestOfKCall, WorstOfKPut,
    RankWeightedBasketCall, RankWeightedBasketPut,

    # Simple single (2)
    Call, Put,

    # Single Lookback (4)
    LookbackFixedCall, LookbackFixedPut,
    LookbackFloatCall, LookbackFloatPut,

    # Single Asian (4)
    AsianFixedStrikeCall_Single, AsianFixedStrikePut_Single,
    AsianFloatingStrikeCall_Single, AsianFloatingStrikePut_Single,

    # Single Range (2)
    RangeCall_Single, RangePut_Single,
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

    # Basket Asian
    'AsianFixedStrikeCall', 'AsianFixedStrikePut',
    'AsianFloatingStrikeCall', 'AsianFloatingStrikePut',

    # Basket Range & Dispersion
    'RangeCall', 'RangePut',
    'DispersionCall', 'DispersionPut',

    # Basket Rank
    'BestOfKCall', 'WorstOfKPut',
    'RankWeightedBasketCall', 'RankWeightedBasketPut',

    # Single payoffs
    'Call', 'Put',

    # Single Lookback
    'LookbackFixedCall', 'LookbackFixedPut',
    'LookbackFloatCall', 'LookbackFloatPut',

    # Single Asian
    'AsianFixedStrikeCall_Single', 'AsianFixedStrikePut_Single',
    'AsianFloatingStrikeCall_Single', 'AsianFloatingStrikePut_Single',

    # Single Range
    'RangeCall_Single', 'RangePut_Single',

    # Registry access
    '_PAYOFF_REGISTRY',
]


def print_payoff_summary():
    """Print summary of all registered payoffs."""
    payoffs = list_payoffs()
    base_payoffs = [p for p in payoffs if not any(b in p for b in _BARRIER_TYPES)]
    barrier_payoffs = [p for p in payoffs if any(b in p for b in _BARRIER_TYPES)]

    print(f"\n📊 Payoff Registry Summary:")
    print(f"   Base payoffs:    {len(base_payoffs)}")
    print(f"   Barrier payoffs: {len(barrier_payoffs)}")
    print(f"   Total payoffs:   {len(payoffs)}")

    # Count by category
    basket_simple = [p for p in base_payoffs if any(x in p for x in ['Basket', 'Geometric', 'Max', 'Min']) and 'Asian' not in p and 'Range' not in p and 'Rank' not in p]
    basket_asian = [p for p in base_payoffs if 'Asian' in p and ('Bsk' in p or 'Basket' in p or 'Fixed' in p or 'Floating' in p)]
    basket_other = [p for p in base_payoffs if any(x in p for x in ['Range', 'Disp', 'Best', 'Worst', 'Rank'])]

    single_simple = [p for p in base_payoffs if p in ['Call', 'Put']]
    single_lookback = [p for p in base_payoffs if 'Lookback' in p or 'LB' in p]
    single_asian = [p for p in base_payoffs if 'Asian' in p and 'Single' in p]
    single_range = [p for p in base_payoffs if 'Range' in p and 'Single' in p]

    print(f"\n Base Payoffs Breakdown:")
    print(f"   Basket Simple:         {len(basket_simple)}")
    print(f"   Basket Asian:          {len(basket_asian)}")
    print(f"   Basket Range/Disp/Rank:{len(basket_other)}")
    print(f"   Single Simple:         {len(single_simple)}")
    print(f"   Single Lookback:       {len(single_lookback)}")
    print(f"   Single Asian:          {len(single_asian)}")
    print(f"   Single Range:          {len(single_range)}")

    expected_total = 30 + 30 * 11  # 30 base + 330 barriers
    print(f"\n Expected total: {expected_total}")
    print(f" Actual total:   {len(payoffs)}")

    if len(payoffs) >= expected_total:
        print(f" All {expected_total} payoffs successfully registered!")
    else:
        print(f" Missing {expected_total - len(payoffs)} payoffs")


if __name__ == "__main__":
    print_payoff_summary()
