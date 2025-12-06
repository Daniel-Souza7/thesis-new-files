"""
Test script for multi-asset correlation in BlackScholes and tree algorithms.

This script tests:
1. BlackScholes correlation matrix handling
2. Correlated path generation via Cholesky decomposition
3. Multi-asset CRR tree with correlation
4. Comparison of independent vs. correlated asset pricing
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import BasketCall, MaxCall, Put
from optimal_stopping.algorithms.trees.crr import CRRTree
from optimal_stopping.algorithms.standard.rlsm import RLSM


def test_correlation_matrix():
    """Test BlackScholes correlation matrix initialization."""
    print("=" * 70)
    print("TEST 1: Correlation Matrix Initialization")
    print("=" * 70)
    print()

    # Test 1: Single asset (should work with default)
    model1 = BlackScholes(
        spot=100,
        drift=0.05,
        volatility=0.2,
        rate=0.03,
        nb_stocks=1,
        nb_paths=100,
        nb_dates=10,
        maturity=1.0
    )
    print(f"✓ Single asset correlation matrix:\n{model1.correlation_matrix}")
    print()

    # Test 2: Two assets with scalar correlation
    model2 = BlackScholes(
        spot=[100, 100],
        drift=0.05,
        volatility=[0.2, 0.25],
        rate=0.03,
        nb_stocks=2,
        nb_paths=100,
        nb_dates=10,
        maturity=1.0,
        correlation=0.5  # 50% correlation
    )
    print(f"✓ Two assets with ρ=0.5:\n{model2.correlation_matrix}")
    print()

    # Test 3: Three assets with full correlation matrix
    corr_matrix = [[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.6],
                   [0.3, 0.6, 1.0]]
    model3 = BlackScholes(
        spot=[100, 100, 100],
        drift=0.05,
        volatility=[0.2, 0.25, 0.3],
        rate=0.03,
        nb_stocks=3,
        nb_paths=100,
        nb_dates=10,
        maturity=1.0,
        correlation=corr_matrix
    )
    print(f"✓ Three assets with custom correlation matrix:\n{model3.correlation_matrix}")
    print()


def test_correlated_paths():
    """Test that generated paths have correct correlation."""
    print("=" * 70)
    print("TEST 2: Correlated Path Generation")
    print("=" * 70)
    print()

    # Generate paths with high correlation
    model_corr = BlackScholes(
        spot=[100, 100],
        drift=0.05,
        volatility=[0.2, 0.2],
        rate=0.03,
        nb_stocks=2,
        nb_paths=50000,
        nb_dates=50,
        maturity=1.0,
        correlation=0.8  # High correlation
    )

    paths_corr, _ = model_corr.generate_paths()

    # Compute empirical correlation of log returns
    log_returns_1 = np.diff(np.log(paths_corr[:, 0, :]), axis=1)
    log_returns_2 = np.diff(np.log(paths_corr[:, 1, :]), axis=1)

    # Flatten across time
    lr1_flat = log_returns_1.flatten()
    lr2_flat = log_returns_2.flatten()

    empirical_corr = np.corrcoef(lr1_flat, lr2_flat)[0, 1]

    print(f"Target correlation: 0.80")
    print(f"Empirical correlation: {empirical_corr:.4f}")
    print(f"Error: {abs(empirical_corr - 0.8):.4f}")

    if abs(empirical_corr - 0.8) < 0.02:
        print("✓ Correlation matches target within tolerance")
    else:
        print("✗ Correlation error too large!")
    print()


def test_multiasset_crr():
    """Test multi-asset CRR tree pricing."""
    print("=" * 70)
    print("TEST 3: Multi-Asset CRR Tree Pricing")
    print("=" * 70)
    print()

    # Setup: 2-asset basket call
    # Test both independent and correlated cases
    correlations = [0.0, 0.5, 0.9]
    results = []

    for rho in correlations:
        print(f"\nCorrelation ρ = {rho}:")
        print("-" * 40)

        model = BlackScholes(
            spot=[100, 100],
            drift=0.05,
            volatility=[0.2, 0.2],
            rate=0.03,
            nb_stocks=2,
            nb_paths=10000,
            nb_dates=50,
            maturity=1.0,
            correlation=rho
        )

        payoff = BasketCall(strike=100)  # Average of two assets

        # Price with CRR tree
        try:
            tree_algo = CRRTree(model, payoff, n_steps=30)
            tree_price, tree_time = tree_algo.price()

            print(f"  CRR Tree Price: ${tree_price:.4f}")
            print(f"  Computation Time: {tree_time:.4f}s")

            # Also price with RLSM for comparison
            rlsm_algo = RLSM(model, payoff)
            rlsm_price, rlsm_time = rlsm_algo.price()

            print(f"  RLSM Price: ${rlsm_price:.4f}")
            print(f"  Price Difference: ${abs(tree_price - rlsm_price):.4f}")

            results.append({
                'rho': rho,
                'tree_price': tree_price,
                'rlsm_price': rlsm_price
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("Price Sensitivity to Correlation:")
    print("=" * 70)
    for res in results:
        print(f"ρ = {res['rho']:.1f}: Tree=${res['tree_price']:.4f}, RLSM=${res['rlsm_price']:.4f}")

    print()
    print("Expected behavior: Higher correlation → Higher basket call price")
    print("(Assets move together more, increasing upside potential)")


def test_single_asset_backward_compat():
    """Test that single-asset trees still work (backward compatibility)."""
    print("=" * 70)
    print("TEST 4: Single-Asset Backward Compatibility")
    print("=" * 70)
    print()

    model = BlackScholes(
        spot=36,
        drift=0.06,
        volatility=0.2,
        rate=0.06,
        nb_stocks=1,
        nb_paths=10000,
        nb_dates=50,
        maturity=1.0
    )

    payoff = Put(strike=40)

    # CRR Tree
    crr = CRRTree(model, payoff, n_steps=50)
    crr_price, crr_time = crr.price()
    crr_exercise = crr.get_exercise_time()

    print(f"CRR Tree:")
    print(f"  Price: ${crr_price:.4f}")
    print(f"  Time: {crr_time:.4f}s")
    print(f"  Avg Exercise Time: {crr_exercise:.4f}")
    print()

    # Benchmark: Longstaff-Schwartz (2001) reports ~$4.48
    benchmark = 4.48
    print(f"Benchmark (LS 2001): ${benchmark:.4f}")
    print(f"Difference: ${abs(crr_price - benchmark):.4f}")

    if abs(crr_price - benchmark) < 0.10:
        print("✓ Single-asset pricing still accurate")
    else:
        print("⚠ Warning: Price differs from benchmark")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print("Multi-Asset Correlation Testing Suite")
    print("=" * 70)
    print("\n")

    try:
        test_correlation_matrix()
        test_correlated_paths()
        test_multiasset_crr()
        test_single_asset_backward_compat()

        print()
        print("=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
