"""
PHASE 1 TEST SUITE - Single Payoff Verification

Tests UpAndOutBasketCall to ensure correct implementation before
extending to all 60 payoff types.

Tests:
1. Barrier Knockout: barrier <= spot → price ≈ 0
2. Extreme Barrier: barrier = 10000 → matches vanilla BasketCall
3. Algorithm Protection: RLSM correctly rejects path-dependent options
4. SRLSM Basic Functionality: Can price barrier options correctly
5. Price Comparison: Barrier price < Vanilla price (barrier reduces value)
"""

import sys
import numpy as np

# Import your actual modules
from optimal_stopping.data import stock_model
from optimal_stopping.payoffs.standard import BasketCall
from optimal_stopping.payoffs.barriers import UpAndOutBasketCall
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM


def test_barrier_knockout():
    """
    Test that barrier <= spot gives price ≈ 0.

    When the barrier is at or below the initial spot price,
    the option should be immediately knocked out.
    """
    print("\n" + "="*70)
    print("TEST 1: Barrier Knockout")
    print("="*70)
    print("\nTesting that barriers at or below spot price give zero value...")

    model = stock_model.BlackScholes(
        drift=0.05,
        volatility=0.2,
        nb_stocks=5,
        nb_paths=10000,
        nb_dates=9,
        spot=100,
        dividend=0.1,
        maturity=3
    )

    results = []

    # Test barriers below and at spot
    for barrier in [80, 90, 95, 100]:
        payoff = UpAndOutBasketCall(strike=100, barrier=barrier)
        pricer = SRLSM(model, payoff, hidden_size=100, factors=(1., 1.))
        price, _ = pricer.price()

        status = "✓ PASS" if price < 0.01 else "❌ FAIL"
        results.append((barrier, price, status))
        print(f"  Barrier={barrier:3d}, Spot=100: Price={price:.6f} ... {status}")

    # Overall test result
    all_passed = all(r[2] == "✓ PASS" for r in results)
    print("\n" + "-"*70)
    if all_passed:
        print("✓ TEST 1 PASSED: All knocked-out barriers give zero value")
    else:
        print("❌ TEST 1 FAILED: Some barriers incorrectly have non-zero value")
    print("="*70)

    return all_passed


def test_extreme_barrier_matches_vanilla():
    """
    Test that barrier=10000 matches vanilla BasketCall.

    With an extremely high barrier, the barrier option should
    behave identically to a vanilla option (barrier never hit).
    """
    print("\n" + "="*70)
    print("TEST 2: Extreme Barrier vs Vanilla")
    print("="*70)
    print("\nComparing barrier option (B=10000) with vanilla option...")

    model = stock_model.BlackScholes(
        drift=0.15,
        volatility=0.2,
        nb_stocks=5,
        nb_paths=50000,  # More paths for better convergence
        nb_dates=9,
        spot=100,
        dividend=0.1,
        maturity=3
    )

    # Vanilla basket call with RLSM
    print("\n  Pricing vanilla BasketCall with RLSM...")
    vanilla = BasketCall(strike=100)
    pricer_vanilla = RLSM(model, vanilla, hidden_size=100, factors=(1., 1.))
    price_vanilla, _ = pricer_vanilla.price()
    print(f"  Vanilla price: {price_vanilla:.4f}")

    # Barrier with extreme barrier using SRLSM
    print("\n  Pricing UpAndOutBasketCall (B=10000) with SRLSM...")
    barrier = UpAndOutBasketCall(strike=100, barrier=10000)
    pricer_barrier = SRLSM(model, barrier, hidden_size=100, factors=(1., 1.))
    price_barrier, _ = pricer_barrier.price()
    print(f"  Barrier price: {price_barrier:.4f}")

    # Compare
    diff = abs(price_vanilla - price_barrier)
    rel_diff = diff / price_vanilla * 100 if price_vanilla > 0 else float('inf')

    print("\n" + "-"*70)
    print(f"  Absolute difference: {diff:.4f}")
    print(f"  Relative difference: {rel_diff:.2f}%")

    # Success if difference is small (< 0.5 absolute or < 5% relative)
    status = "✓ PASS" if (diff < 0.5 or rel_diff < 5.0) else "❌ FAIL"
    print(f"\n  Status: {status}")

    if status == "✓ PASS":
        print("✓ TEST 2 PASSED: Extreme barrier matches vanilla within tolerance")
    else:
        print("❌ TEST 2 FAILED: Extreme barrier differs significantly from vanilla")

    print("="*70)

    return status == "✓ PASS"


def test_algorithm_protection():
    """
    Test that RLSM rejects path-dependent options.

    The algorithm should raise a ValueError when trying to use
    RLSM with a path-dependent payoff.
    """
    print("\n" + "="*70)
    print("TEST 3: Algorithm Protection")
    print("="*70)
    print("\nTesting that RLSM correctly rejects path-dependent options...")

    model = stock_model.BlackScholes(
        drift=0.05,
        volatility=0.2,
        nb_stocks=5,
        nb_paths=10000,
        nb_dates=9,
        spot=100,
        dividend=0.1,
        maturity=3
    )

    barrier = UpAndOutBasketCall(strike=100, barrier=110)

    print("\n  Attempting to create RLSM with barrier option...")

    try:
        pricer = RLSM(model, barrier)  # Should raise ValueError
        print("  ❌ FAIL: RLSM incorrectly accepted path-dependent option")
        passed = False
    except ValueError as e:
        print(f"  ✓ PASS: RLSM correctly rejected path-dependent option")
        print(f"          Error message: '{e}'")
        passed = True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        passed = False

    print("\n" + "-"*70)
    if passed:
        print("✓ TEST 3 PASSED: Algorithm protection working correctly")
    else:
        print("❌ TEST 3 FAILED: Algorithm protection not working")
    print("="*70)

    return passed


def test_srlsm_basic_functionality():
    """
    Test that SRLSM can price a barrier option without errors.

    This is a sanity check to ensure the algorithm runs and
    produces a reasonable price.
    """
    print("\n" + "="*70)
    print("TEST 4: SRLSM Basic Functionality")
    print("="*70)
    print("\nTesting that SRLSM can price barrier options...")

    model = stock_model.BlackScholes(
        drift=0.05,
        volatility=0.2,
        nb_stocks=5,
        nb_paths=20000,
        nb_dates=9,
        spot=100,
        dividend=0.1,
        maturity=3
    )

    # Test a reasonable barrier option
    print("\n  Pricing UpAndOutBasketCall (K=100, B=120) with SRLSM...")
    barrier = UpAndOutBasketCall(strike=100, barrier=120)

    try:
        pricer = SRLSM(model, barrier, hidden_size=100, factors=(1., 1.))
        price, time_gen = pricer.price()

        print(f"  Price: {price:.4f}")
        print(f"  Path generation time: {time_gen:.4f}s")

        # Sanity checks
        checks = []

        # Price should be non-negative
        if price >= 0:
            print("  ✓ Price is non-negative")
            checks.append(True)
        else:
            print(f"  ❌ Price is negative: {price}")
            checks.append(False)

        # Price should be less than spot (upper bound)
        if price < model.spot:
            print("  ✓ Price is less than spot (reasonable upper bound)")
            checks.append(True)
        else:
            print(f"  ❌ Price {price} exceeds spot {model.spot}")
            checks.append(False)

        # Price should be less than vanilla (barrier reduces value)
        vanilla = BasketCall(strike=100)
        pricer_vanilla = RLSM(model, vanilla, hidden_size=100, factors=(1., 1.))
        price_vanilla, _ = pricer_vanilla.price()

        if price <= price_vanilla:
            print(f"  ✓ Barrier price ({price:.4f}) ≤ vanilla price ({price_vanilla:.4f})")
            checks.append(True)
        else:
            print(f"  ❌ Barrier price ({price:.4f}) > vanilla price ({price_vanilla:.4f})")
            checks.append(False)

        passed = all(checks)

    except Exception as e:
        print(f"  ❌ FAIL: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    print("\n" + "-"*70)
    if passed:
        print("✓ TEST 4 PASSED: SRLSM functioning correctly")
    else:
        print("❌ TEST 4 FAILED: SRLSM has issues")
    print("="*70)

    return passed


def test_srlsm_rejects_standard():
    """
    Test that SRLSM rejects standard (non-path-dependent) options.

    The algorithm should raise a ValueError when trying to use
    SRLSM with a standard payoff.
    """
    print("\n" + "="*70)
    print("TEST 5: SRLSM Protection (Rejects Standard Options)")
    print("="*70)
    print("\nTesting that SRLSM correctly rejects standard options...")

    model = stock_model.BlackScholes(
        drift=0.05,
        volatility=0.2,
        nb_stocks=5,
        nb_paths=10000,
        nb_dates=9,
        spot=100,
        dividend=0.1,
        maturity=3
    )

    vanilla = BasketCall(strike=100)

    print("\n  Attempting to create SRLSM with standard option...")

    try:
        pricer = SRLSM(model, vanilla)  # Should raise ValueError
        print("  ❌ FAIL: SRLSM incorrectly accepted standard option")
        passed = False
    except ValueError as e:
        print(f"  ✓ PASS: SRLSM correctly rejected standard option")
        print(f"          Error message: '{e}'")
        passed = True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        passed = False

    print("\n" + "-"*70)
    if passed:
        print("✓ TEST 5 PASSED: SRLSM protection working correctly")
    else:
        print("❌ TEST 5 FAILED: SRLSM protection not working")
    print("="*70)

    return passed


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*70)
    print(" "*15 + "PHASE 1 TEST SUITE")
    print(" "*10 + "Testing: UpAndOutBasketCall")
    print("="*70)

    results = {}

    # Run each test
    print("\nRunning tests...")

    try:
        results['knockout'] = test_barrier_knockout()
    except Exception as e:
        print(f"❌ Test 'knockout' failed with exception: {e}")
        results['knockout'] = False

    try:
        results['extreme_barrier'] = test_extreme_barrier_matches_vanilla()
    except Exception as e:
        print(f"❌ Test 'extreme_barrier' failed with exception: {e}")
        results['extreme_barrier'] = False

    try:
        results['rlsm_protection'] = test_algorithm_protection()
    except Exception as e:
        print(f"❌ Test 'rlsm_protection' failed with exception: {e}")
        results['rlsm_protection'] = False

    try:
        results['srlsm_basic'] = test_srlsm_basic_functionality()
    except Exception as e:
        print(f"❌ Test 'srlsm_basic' failed with exception: {e}")
        results['srlsm_basic'] = False

    try:
        results['srlsm_protection'] = test_srlsm_rejects_standard()
    except Exception as e:
        print(f"❌ Test 'srlsm_protection' failed with exception: {e}")
        results['srlsm_protection'] = False

    # Summary
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED ✅")
        print("\nThe implementation is correct. You can proceed to:")
        print("  1. Add remaining barrier options")
        print("  2. Add lookback options")
        print("  3. Implement SRFQI")
        print("  4. Extend to all 60 payoff types")
    else:
        print("❌ SOME TESTS FAILED ❌")
        print("\nDo NOT proceed until all tests pass.")
        print("Review the failed tests and fix the issues.")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)