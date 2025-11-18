"""
Compare LSM implementations: Your implementation vs Debug version
Tests both PUT and CALL options
"""

import numpy as np
import sys
from pathlib import Path

# Add optimal_stopping to path
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs.single_simple import Put, Call
from debug_lsm import generate_gbm_paths, lsm_american_option


def compare_single_option(option_type, nb_paths, nb_stocks, nb_dates, maturity,
                          spot, strike, drift, volatility, rate, train_ITM_only, seed):
    """Compare your LSM vs debug LSM for a single option type."""

    print("\n" + "=" * 80)
    print(f"TESTING AMERICAN {option_type.upper()} OPTION")
    print("=" * 80)

    # Set same seed
    np.random.seed(seed)

    # ========================================
    # YOUR LSM IMPLEMENTATION
    # ========================================
    print("\n" + "-" * 80)
    print("YOUR LSM IMPLEMENTATION")
    print("-" * 80)

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

    if option_type == 'put':
        payoff = Put(strike=strike)
    else:  # call
        payoff = Call(strike=strike)

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
    print("\n" + "-" * 80)
    print("DEBUG LSM IMPLEMENTATION")
    print("-" * 80)

    # Reset seed to generate same paths
    np.random.seed(seed)

    # Generate paths using DRIFT (not rate) to match current implementation
    dt = maturity / nb_dates
    S = generate_gbm_paths(nb_paths, nb_dates, spot, drift, volatility, dt, d=0.0)

    # Price using RATE for discounting
    price_debug, ex_time_debug, _, _ = lsm_american_option(S, strike, rate, dt, option_type=option_type)

    print(f"Price:              {price_debug:.4f}")
    print(f"Exercise Time:      {ex_time_debug:.4f}")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "-" * 80)
    print(f"{option_type.upper()} OPTION COMPARISON")
    print("-" * 80)

    price_diff = abs(price_yours - price_debug)
    ex_time_diff = abs(ex_time_yours - ex_time_debug)

    print(f"Price Difference:         {price_diff:.4f} ({100*price_diff/price_debug:.2f}%)")
    print(f"Exercise Time Difference: {ex_time_diff:.4f}")

    if price_diff < 0.1:
        price_status = "✓ Prices are very close!"
    elif price_diff < 0.5:
        price_status = "⚠ Prices are reasonably close"
    else:
        price_status = "✗ Prices differ significantly"

    if ex_time_diff < 0.05:
        ex_time_status = "✓ Exercise times are very close!"
    elif ex_time_diff < 0.1:
        ex_time_status = "⚠ Exercise times are reasonably close"
    else:
        ex_time_status = "✗ Exercise times differ significantly"

    print(f"\n{price_status}")
    print(f"{ex_time_status}")

    return {
        'option_type': option_type,
        'price_yours': price_yours,
        'price_debug': price_debug,
        'price_diff': price_diff,
        'ex_time_yours': ex_time_yours,
        'ex_time_debug': ex_time_debug,
        'ex_time_diff': ex_time_diff
    }


def compare_lsm_implementations():
    """Compare your LSM vs debug LSM for both PUT and CALL options."""

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
    nb_runs = 10  # Number of runs to average over

    print("=" * 80)
    print("LSM IMPLEMENTATION COMPARISON - PUT AND CALL OPTIONS")
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
    print(f"  Number of runs: {nb_runs}")

    # Store results across runs
    put_results_all = []
    call_results_all = []

    for run in range(nb_runs):
        seed = 42 + run  # Different seed for each run
        print(f"\n{'-' * 80}")
        print(f"RUN {run + 1}/{nb_runs} (seed={seed})")
        print(f"{'-' * 80}")

        # Compare PUT option
        put_results = compare_single_option(
            'put', nb_paths, nb_stocks, nb_dates, maturity,
            spot, strike, drift, volatility, rate, train_ITM_only, seed
        )
        put_results_all.append(put_results)

        # Compare CALL option
        call_results = compare_single_option(
            'call', nb_paths, nb_stocks, nb_dates, maturity,
            spot, strike, drift, volatility, rate, train_ITM_only, seed
        )
        call_results_all.append(call_results)

    # ========================================
    # AGGREGATE STATISTICS
    # ========================================
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS ACROSS ALL RUNS")
    print("=" * 80)

    # PUT statistics
    put_prices_yours = [r['price_yours'] for r in put_results_all]
    put_prices_debug = [r['price_debug'] for r in put_results_all]
    put_ex_times_yours = [r['ex_time_yours'] for r in put_results_all]
    put_ex_times_debug = [r['ex_time_debug'] for r in put_results_all]

    print(f"\nPUT Option (n={nb_runs} runs):")
    print(f"  Your LSM Price:      {np.mean(put_prices_yours):.4f} ± {np.std(put_prices_yours):.4f}")
    print(f"  Debug LSM Price:     {np.mean(put_prices_debug):.4f} ± {np.std(put_prices_debug):.4f}")
    print(f"  Price Difference:    {np.mean([r['price_diff'] for r in put_results_all]):.4f} ± {np.std([r['price_diff'] for r in put_results_all]):.4f}")
    print(f"")
    print(f"  Your LSM ExTime:     {np.mean(put_ex_times_yours):.4f} ± {np.std(put_ex_times_yours):.4f}")
    print(f"  Debug LSM ExTime:    {np.mean(put_ex_times_debug):.4f} ± {np.std(put_ex_times_debug):.4f}")
    print(f"  ExTime Difference:   {np.mean([r['ex_time_diff'] for r in put_results_all]):.4f} ± {np.std([r['ex_time_diff'] for r in put_results_all]):.4f}")

    # CALL statistics
    call_prices_yours = [r['price_yours'] for r in call_results_all]
    call_prices_debug = [r['price_debug'] for r in call_results_all]
    call_ex_times_yours = [r['ex_time_yours'] for r in call_results_all]
    call_ex_times_debug = [r['ex_time_debug'] for r in call_results_all]

    print(f"\nCALL Option (n={nb_runs} runs):")
    print(f"  Your LSM Price:      {np.mean(call_prices_yours):.4f} ± {np.std(call_prices_yours):.4f}")
    print(f"  Debug LSM Price:     {np.mean(call_prices_debug):.4f} ± {np.std(call_prices_debug):.4f}")
    print(f"  Price Difference:    {np.mean([r['price_diff'] for r in call_results_all]):.4f} ± {np.std([r['price_diff'] for r in call_results_all]):.4f}")
    print(f"")
    print(f"  Your LSM ExTime:     {np.mean(call_ex_times_yours):.4f} ± {np.std(call_ex_times_yours):.4f}")
    print(f"  Debug LSM ExTime:    {np.mean(call_ex_times_debug):.4f} ± {np.std(call_ex_times_debug):.4f}")
    print(f"  ExTime Difference:   {np.mean([r['ex_time_diff'] for r in call_results_all]):.4f} ± {np.std([r['ex_time_diff'] for r in call_results_all]):.4f}")

    # ========================================
    # FINAL ASSESSMENT
    # ========================================
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)

    put_price_diff_mean = np.mean([r['price_diff'] for r in put_results_all])
    put_ex_diff_mean = np.mean([r['ex_time_diff'] for r in put_results_all])
    call_price_diff_mean = np.mean([r['price_diff'] for r in call_results_all])
    call_ex_diff_mean = np.mean([r['ex_time_diff'] for r in call_results_all])

    print(f"\nPUT Option:")
    if put_price_diff_mean < 0.1:
        print(f"  ✓ Prices match very well (diff={put_price_diff_mean:.4f})")
    else:
        print(f"  ✗ Prices differ significantly (diff={put_price_diff_mean:.4f})")

    if put_ex_diff_mean < 0.05:
        print(f"  ✓ Exercise times match very well (diff={put_ex_diff_mean:.4f})")
    else:
        print(f"  ✗ Exercise times differ significantly (diff={put_ex_diff_mean:.4f})")

    print(f"\nCALL Option:")
    if call_price_diff_mean < 0.1:
        print(f"  ✓ Prices match very well (diff={call_price_diff_mean:.4f})")
    else:
        print(f"  ✗ Prices differ significantly (diff={call_price_diff_mean:.4f})")

    if call_ex_diff_mean < 0.05:
        print(f"  ✓ Exercise times match very well (diff={call_ex_diff_mean:.4f})")
    else:
        print(f"  ✗ Exercise times differ significantly (diff={call_ex_diff_mean:.4f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_lsm_implementations()
