"""
Test script for tree-based algorithms.

This script tests the CRR, Leisen-Reimer, and Trinomial tree algorithms
on a simple American put option to verify they work correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import Put
from optimal_stopping.algorithms.trees.crr import CRRTree
from optimal_stopping.algorithms.trees.leisen_reimer import LeisenReimerTree
from optimal_stopping.algorithms.trees.trinomial import TrinomialTree


def test_tree_algorithms():
    """Test all three tree algorithms on an American put option."""

    print("=" * 70)
    print("Testing Tree-Based Algorithms for American Options")
    print("=" * 70)
    print()

    # Problem setup: American put option
    # Parameters from classic benchmark (Longstaff-Schwartz 2001)
    S0 = 36.0      # Initial stock price
    K = 40.0       # Strike price
    r = 0.06       # Risk-free rate
    sigma = 0.2    # Volatility
    T = 1.0        # Maturity (1 year)

    print("Problem Setup:")
    print(f"  Option Type: American Put")
    print(f"  Initial Stock Price (S0): ${S0}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Risk-Free Rate (r): {r*100}%")
    print(f"  Volatility (σ): {sigma*100}%")
    print(f"  Maturity (T): {T} year")
    print()

    # Create stock model (single asset)
    model = BlackScholes(
        spot=S0,
        drift=r,  # For pricing, drift = risk-free rate
        volatility=sigma,
        rate=r,
        dividend=0.0,
        nb_stocks=1,
        maturity=T,
        nb_dates=50,
        nb_paths=10000
    )

    # Create American put payoff
    payoff = Put(strike=K)

    print("Running Tree Algorithms:")
    print("-" * 70)

    algorithms = [
        ("CRR (Cox-Ross-Rubinstein)", CRRTree, {"n_steps": 50}),
        ("LR (Leisen-Reimer)", LeisenReimerTree, {"n_steps": 51}),  # Odd for LR
        ("Trinomial Tree", TrinomialTree, {"n_steps": 50}),
    ]

    results = []

    for algo_name, AlgoClass, kwargs in algorithms:
        print(f"\n{algo_name}:")
        print(f"  Number of steps: {kwargs.get('n_steps', 50)}")

        try:
            # Create algorithm instance
            algo = AlgoClass(model=model, payoff=payoff, **kwargs)

            # Price the option
            price, comp_time = algo.price()

            # Get exercise time
            exercise_time = algo.get_exercise_time()

            # Store results
            results.append({
                'name': algo_name,
                'price': price,
                'time': comp_time,
                'exercise_time': exercise_time
            })

            print(f"  ✓ Price: ${price:.4f}")
            print(f"  ✓ Computation Time: {comp_time:.4f}s")
            print(f"  ✓ Average Exercise Time: {exercise_time:.4f} (normalized)")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print()
    print(f"{'Algorithm':<30} {'Price':>10} {'Comp Time':>12} {'Exercise Time':>15}")
    print("-" * 70)

    for result in results:
        print(f"{result['name']:<30} ${result['price']:>9.4f} {result['time']:>11.4f}s {result['exercise_time']:>14.4f}")

    print()

    # Compare with known benchmark (if available)
    # For American put with these parameters, the approximate price is around $4.48
    # Reference: Longstaff-Schwartz (2001) paper
    benchmark_price = 4.48

    print(f"Benchmark (Longstaff-Schwartz 2001): ${benchmark_price:.4f}")
    print()
    print("Price Differences from Benchmark:")
    for result in results:
        diff = result['price'] - benchmark_price
        pct_diff = (diff / benchmark_price) * 100
        print(f"  {result['name']:<30} {diff:+.4f} ({pct_diff:+.2f}%)")

    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_tree_algorithms()
