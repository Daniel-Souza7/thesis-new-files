#!/usr/bin/env python3
"""
Quick factor test - reduced grid for faster testing.

Tests key factor combinations on one simple and one complex problem.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.data.stock_model import BlackScholes, Heston
from optimal_stopping.payoffs import get_payoff_class, LookbackFixedPut
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI


def quick_test():
    """Quick test with reduced parameter grid."""
    print("=" * 80)
    print("Quick Factor Influence Test")
    print("=" * 80)

    # Reduced grid for quick testing
    activation_slopes = [0.8, 1.0, 1.2]  # 3 values instead of 6
    input_scales = [0.7, 1.0, 1.3]        # 3 values instead of 5

    # Simple problem: Barrier put
    print("\nSetting up Simple Problem (Barrier Put)...")
    model_simple = BlackScholes(
        spot=100.0, drift=0.05, dividend=0.0, volatility=0.2, maturity=1.0,
        nb_stocks=1, nb_paths=5000, nb_dates=30, seed=42
    )
    DownAndOutPut = get_payoff_class('DO-Put')
    payoff_simple = DownAndOutPut(strike=100.0, barrier=80.0)

    # Complex problem: Lookback with Heston
    print("Setting up Complex Problem (Lookback Put with Heston)...")
    model_complex = Heston(
        drift=0.05,           # Stock drift
        volatility=0.3,       # Vol-of-vol (xi)
        mean=0.04,           # Long-term variance mean (v_bar, also initial variance)
        speed=2.0,           # Mean reversion speed (kappa)
        correlation=-0.7,    # Correlation (rho)
        nb_stocks=1,
        nb_paths=5000,
        nb_dates=30,
        spot=100.0,          # Initial stock price
        maturity=1.0,
        dividend=0.0,
        seed=42
    )
    payoff_complex = LookbackFixedPut(strike=100.0)

    results = []

    # Test SRLSM
    print("\n" + "=" * 80)
    print("Testing SRLSM")
    print("=" * 80)

    print("\nSimple Problem:")
    for act_slope in activation_slopes:
        for inp_scale in input_scales:
            factors = (act_slope, inp_scale)
            print(f"  factors={factors}... ", end='', flush=True)

            try:
                algo = SRLSM(
                    model=model_simple, payoff=payoff_simple,
                    hidden_size=20, factors=factors, train_ITM_only=True
                )
                t0 = time.time()
                lower, upper, _ = algo.price_upper_lower_bound(train_eval_split=2)
                comp_time = time.time() - t0
                gap = (upper - lower) / lower * 100

                results.append({
                    'Algorithm': 'SRLSM', 'Problem': 'Simple',
                    'Act_Slope': act_slope, 'Inp_Scale': inp_scale,
                    'Lower': lower, 'Upper': upper, 'Gap_%': gap, 'Time_s': comp_time
                })
                print(f"Gap: {gap:.2f}%, Time: {comp_time:.2f}s")
            except Exception as e:
                print(f"Failed: {str(e)[:40]}")

    print("\nComplex Problem:")
    for act_slope in activation_slopes:
        for inp_scale in input_scales:
            factors = (act_slope, inp_scale)
            print(f"  factors={factors}... ", end='', flush=True)

            try:
                algo = SRLSM(
                    model=model_complex, payoff=payoff_complex,
                    hidden_size=20, factors=factors, train_ITM_only=True
                )
                t0 = time.time()
                lower, upper, _ = algo.price_upper_lower_bound(train_eval_split=2)
                comp_time = time.time() - t0
                gap = (upper - lower) / lower * 100

                results.append({
                    'Algorithm': 'SRLSM', 'Problem': 'Complex',
                    'Act_Slope': act_slope, 'Inp_Scale': inp_scale,
                    'Lower': lower, 'Upper': upper, 'Gap_%': gap, 'Time_s': comp_time
                })
                print(f"Gap: {gap:.2f}%, Time: {comp_time:.2f}s")
            except Exception as e:
                print(f"Failed: {str(e)[:40]}")

    # Test SRFQI
    print("\n" + "=" * 80)
    print("Testing SRFQI")
    print("=" * 80)

    print("\nSimple Problem:")
    for act_slope in activation_slopes:
        for inp_scale in input_scales:
            factors = (act_slope, inp_scale)
            print(f"  factors={factors}... ", end='', flush=True)

            try:
                algo = SRFQI(
                    model=model_simple, payoff=payoff_simple,
                    nb_epochs=20, hidden_size=20, factors=factors, train_ITM_only=True
                )
                t0 = time.time()
                lower, upper, _ = algo.price_upper_lower_bound(train_eval_split=2)
                comp_time = time.time() - t0
                gap = (upper - lower) / lower * 100

                results.append({
                    'Algorithm': 'SRFQI', 'Problem': 'Simple',
                    'Act_Slope': act_slope, 'Inp_Scale': inp_scale,
                    'Lower': lower, 'Upper': upper, 'Gap_%': gap, 'Time_s': comp_time
                })
                print(f"Gap: {gap:.2f}%, Time: {comp_time:.2f}s")
            except Exception as e:
                print(f"Failed: {str(e)[:40]}")

    print("\nComplex Problem:")
    for act_slope in activation_slopes:
        for inp_scale in input_scales:
            factors = (act_slope, inp_scale)
            print(f"  factors={factors}... ", end='', flush=True)

            try:
                algo = SRFQI(
                    model=model_complex, payoff=payoff_complex,
                    nb_epochs=20, hidden_size=20, factors=factors, train_ITM_only=True
                )
                t0 = time.time()
                lower, upper, _ = algo.price_upper_lower_bound(train_eval_split=2)
                comp_time = time.time() - t0
                gap = (upper - lower) / lower * 100

                results.append({
                    'Algorithm': 'SRFQI', 'Problem': 'Complex',
                    'Act_Slope': act_slope, 'Inp_Scale': inp_scale,
                    'Lower': lower, 'Upper': upper, 'Gap_%': gap, 'Time_s': comp_time
                })
                print(f"Gap: {gap:.2f}%, Time: {comp_time:.2f}s")
            except Exception as e:
                print(f"Failed: {str(e)[:40]}")

    # Analyze results
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for algo in ['SRLSM', 'SRFQI']:
        print(f"\n{algo}:")
        print("-" * 80)
        for prob in ['Simple', 'Complex']:
            subset = df[(df['Algorithm'] == algo) & (df['Problem'] == prob)]
            if len(subset) == 0:
                continue

            print(f"\n  {prob} Problem:")
            print(f"    Average Gap: {subset['Gap_%'].mean():.2f}% ± {subset['Gap_%'].std():.2f}%")
            print(f"    Best Gap: {subset['Gap_%'].min():.2f}%")

            # Best configuration
            best_idx = subset['Gap_%'].idxmin()
            best = subset.loc[best_idx]
            print(f"    Best config: act_slope={best['Act_Slope']:.1f}, inp_scale={best['Inp_Scale']:.1f}")

    # Factor influence
    print("\n" + "=" * 80)
    print("FACTOR INFLUENCE")
    print("=" * 80)

    print("\nActivation Slope Influence:")
    print(df.groupby('Act_Slope')['Gap_%'].agg(['mean', 'std']).to_string())

    print("\nInput Scale Influence:")
    print(df.groupby('Inp_Scale')['Gap_%'].agg(['mean', 'std']).to_string())

    # Save results
    df.to_csv('factor_quick_test_results.csv', index=False)
    print(f"\nResults saved to: factor_quick_test_results.csv")

    return df


if __name__ == "__main__":
    np.random.seed(42)
    df = quick_test()
    print("\n" + "=" * 80)
    print("Quick test complete!")
    print("=" * 80)
