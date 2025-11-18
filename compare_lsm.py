"""
Compare LSM implementations: Your implementation vs Debug version
"""

import numpy as np
import sys
from pathlib import Path

# Add optimal_stopping to path
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs.single_simple import Put
from debug_lsm import generate_gbm_paths, lsm_american_option


def compare_lsm_implementations():
    """Compare your LSM vs debug LSM."""

    # Common parameters
    nb_paths = 10000
    nb_stocks = 1
    nb_dates = 10
    maturity = 1.0
    spot = 100.0
    strike = 100.0
    drift = 0.06
    volatility = 0.2
    rate = 0.02  # drift - 0.04
    train_ITM_only = True

    print("=" * 80)
    print("LSM IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Paths:          {nb_paths:,}")
    print(f"  Stocks:         {nb_stocks}")
    print(f"  Time steps:     {nb_dates}")
    print(f"  Maturity:       {maturity}")
    print(f"  Spot:           {spot}")
    print(f"  Strike:         {strike}")
    print(f"  Drift:          {drift}")
    print(f"  Volatility:     {volatility}")
    print(f"  Rate:           {rate}")
    print(f"  Train ITM only: {train_ITM_only}")

    # Set same seed
    np.random.seed(42)

    # ========================================
    # YOUR LSM IMPLEMENTATION
    # ========================================
    print("\n" + "=" * 80)
    print("RUNNING YOUR LSM IMPLEMENTATION")
    print("=" * 80)

    # Create model and payoff
    model = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_stocks=nb_stocks,
        nb_paths=nb_paths,
        nb_dates=nb_dates,
        spot=spot,
        dividend=0.0,
        maturity=maturity
    )

    payoff = Put(strike=strike)

    # Create LSM pricer
    lsm = LeastSquaresPricer(
        model=model,
        payoff=payoff,
        train_ITM_only=train_ITM_only
    )

    # Price
    price_yours, time_gen = lsm.price()
    ex_time_yours = lsm.get_exercise_time()

    print(f"Price:              {price_yours:.4f}")
    print(f"Exercise Time:      {ex_time_yours:.4f}")

    # ========================================
    # DEBUG LSM IMPLEMENTATION
    # ========================================
    print("\n" + "=" * 80)
    print("RUNNING DEBUG LSM IMPLEMENTATION")
    print("=" * 80)

    # Reset seed to generate same paths
    np.random.seed(42)

    # Generate paths
    dt = maturity / nb_dates
    S = generate_gbm_paths(nb_paths, nb_dates, spot, rate, volatility, dt, d=0.0)

    # Price
    price_debug, ex_time_debug, _, _ = lsm_american_option(S, strike, rate, dt, option_type='put')

    print(f"Price:              {price_debug:.4f}")
    print(f"Exercise Time:      {ex_time_debug:.4f}")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    price_diff = abs(price_yours - price_debug)
    ex_time_diff = abs(ex_time_yours - ex_time_debug)

    print(f"Price Difference:         {price_diff:.4f} ({100*price_diff/price_debug:.2f}%)")
    print(f"Exercise Time Difference: {ex_time_diff:.4f}")

    if price_diff < 0.1:
        print("\n✓ Prices are very close!")
    elif price_diff < 0.5:
        print("\n⚠ Prices are reasonably close")
    else:
        print("\n✗ Prices differ significantly")

    if ex_time_diff < 0.05:
        print("✓ Exercise times are very close!")
    elif ex_time_diff < 0.1:
        print("⚠ Exercise times are reasonably close")
    else:
        print("✗ Exercise times differ significantly")

    print("=" * 80)


if __name__ == "__main__":
    compare_lsm_implementations()
