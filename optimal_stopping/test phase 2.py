"""
TEST SUITE FOR RFQI AND SRFQI

Tests the Randomized Fitted Q-Iteration algorithms for both
standard and path-dependent options.

Tests:
1. RFQI can price vanilla BasketCall
2. SRFQI can price UpAndOutBasketCall
3. RFQI rejects path-dependent options
4. SRFQI rejects standard options
5. Barrier prices are less than vanilla prices
"""

import sys
import numpy as np

from optimal_stopping.data import stock_model
from optimal_stopping.payoffs.standard import BasketCall
from optimal_stopping.payoffs.barriers import UpAndOutBasketCall
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
# Also import RLSM/SRLSM for comparison
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM


def test_rfqi_vanilla():
    """Test that RFQI can price vanilla BasketCall."""
    print("\n" + "=" * 70)
    print("TEST 1: RFQI Vanilla Pricing")
    print("=" * 70)
    print("\nTesting that RFQI can price standard options...")

    model = stock_model.BlackScholes(
        drift=0.05, volatility=0.2, nb_stocks=5,
        nb_paths=20000, nb_dates=9, spot=100,
        dividend=0.1, maturity=3
    )

    try:
        payoff = BasketCall(strike=100)
        pricer = RFQI(model, payoff, nb_epochs=10, hidden_size=100, factors=(1., 1.))
        price, time_gen = pricer.price()

        print(f"  Price: {price:.4f}")
        print(f"  Path generation time: {time_gen:.4f}s")

        # Sanity checks
        checks = []

        if price >= 0:
            print("  ✓ Price is non-negative")
            checks.append(True)
        else:
            print(f"  ❌ Price is negative: {price}")
            checks.append(False)

        if price < model.spot:
            print("  ✓ Price is less than spot")
            checks.append(True)
        else:
            print(f"  ❌ Price exceeds spot")
            checks.append(False)

        # Compare with RLSM
        pricer_rlsm = RLSM(model, payoff, hidden_size=100, factors=(1., 1.))
        price_rlsm, _ = pricer_rlsm.price()

        diff = abs(price - price_rlsm)
        print(f"  RFQI price: {price:.4f}, RLSM price: {price_rlsm:.4f}, diff: {diff:.4f}")

        passed = all(checks)

    except Exception as e:
        print(f"  ❌ FAIL: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    print("\n" + "-" * 70)
    if passed:
        print("✓ TEST 1 PASSED: RFQI functioning correctly")
    else:
        print("❌ TEST 1 FAILED: RFQI has issues")
    print("=" * 70)

    return passed


def test_srfqi_barrier():
    """Test that SRFQI can price UpAndOutBasketCall."""
    print("\n" + "=" * 70)
    print("TEST 2: SRFQI Barrier Pricing")
    print("=" * 70)
    print("\nTesting that SRFQI can price barrier options...")

    model = stock_model.BlackScholes(
        drift=0.05, volatility=0.2, nb_stocks=5,
        nb_paths=20000, nb_dates=9, spot=100,
        dividend=0.1, maturity=3
    )

    try:
        payoff = UpAndOutBasketCall(strike=100, barrier=120)
        pricer = SRFQI(model, payoff, nb_epochs=10, hidden_size=100, factors=(1., 1.))
        price, time_gen = pricer.price()

        print(f"  Price: {price:.4f}")
        print(f"  Path generation time: {time_gen:.4f}s")

        # Sanity checks
        checks = []

        if price >= 0:
            print("  ✓ Price is non-negative")
            checks.append(True)
        else:
            print(f"  ❌ Price is negative: {price}")
            checks.append(False)

        # Compare with vanilla
        vanilla = BasketCall(strike=100)
        pricer_vanilla = RFQI(model, vanilla, nb_epochs=10, hidden_size=100, factors=(1., 1.))
        price_vanilla, _ = pricer_vanilla.price()

        if price <= price_vanilla:
            print(f"  ✓ Barrier price ({price:.4f}) ≤ vanilla price ({price_vanilla:.4f})")
            checks.append(True)
        else:
            print(f"  ❌ Barrier price ({price:.4f}) > vanilla price ({price_vanilla:.4f})")
            checks.append(False)

        # Compare with SRLSM
        pricer_srlsm = SRLSM(model, payoff, hidden_size=100, factors=(1., 1.))
        price_srlsm, _ = pricer_srlsm.price()

        diff = abs(price - price_srlsm)
        print(f"  SRFQI price: {price:.4f}, SRLSM price: {price_srlsm:.4f}, diff: {diff:.4f}")

        passed = all(checks)

    except Exception as e:
        print(f"  ❌ FAIL: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    print("\n" + "-" * 70)
    if passed:
        print("✓ TEST 2 PASSED: SRFQI functioning correctly")
    else:
        print("❌ TEST 2 FAILED: SRFQI has issues")
    print("=" * 70)

    return passed


def test_rfqi_protection():
    """Test that RFQI rejects path-dependent options."""
    print("\n" + "=" * 70)
    print("TEST 3: RFQI Protection")
    print("=" * 70)
    print("\nTesting that RFQI correctly rejects path-dependent options...")

    model = stock_model.BlackScholes(
        drift=0.05, volatility=0.2, nb_stocks=5,
        nb_paths=10000, nb_dates=9, spot=100,
        dividend=0.1, maturity=3
    )

    barrier = UpAndOutBasketCall(strike=100, barrier=110)

    print("\n  Attempting to create RFQI with barrier option...")

    try:
        pricer = RFQI(model, barrier)  # Should raise ValueError
        print("  ❌ FAIL: RFQI incorrectly accepted path-dependent option")
        passed = False
    except ValueError as e:
        print(f"  ✓ PASS: RFQI correctly rejected path-dependent option")
        print(f"          Error message: '{e}'")
        passed = True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        passed = False

    print("\n" + "-" * 70)
    if passed:
        print("✓ TEST 3 PASSED: RFQI protection working correctly")
    else:
        print("❌ TEST 3 FAILED: RFQI protection not working")
    print("=" * 70)

    return passed


def test_srfqi_protection():
    """Test that SRFQI rejects standard options."""
    print("\n" + "=" * 70)
    print("TEST 4: SRFQI Protection")
    print("=" * 70)
    print("\nTesting that SRFQI correctly rejects standard options...")

    model = stock_model.BlackScholes(
        drift=0.05, volatility=0.2, nb_stocks=5,
        nb_paths=10000, nb_dates=9, spot=100,
        dividend=0.1, maturity=3
    )

    vanilla = BasketCall(strike=100)

    print("\n  Attempting to create SRFQI with standard option...")

    try:
        pricer = SRFQI(model, vanilla)  # Should raise ValueError
        print("  ❌ FAIL: SRFQI incorrectly accepted standard option")
        passed = False
    except ValueError as e:
        print(f"  ✓ PASS: SRFQI correctly rejected standard option")
        print(f"          Error message: '{e}'")
        passed = True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        passed = False

    print("\n" + "-" * 70)
    if passed:
        print("✓ TEST 4 PASSED: SRFQI protection working correctly")
    else:
        print("❌ TEST 4 FAILED: SRFQI protection not working")
    print("=" * 70)

    return passed


def test_barrier_knockout_srfqi():
    """Test that SRFQI gives zero for knocked-out barriers."""
    print("\n" + "=" * 70)
    print("TEST 5: SRFQI Barrier Knockout")
    print("=" * 70)
    print("\nTesting that barriers at or below spot give zero value...")

    model = stock_model.BlackScholes(
        drift=0.05, volatility=0.2, nb_stocks=5,
        nb_paths=10000, nb_dates=9, spot=100,
        dividend=0.1, maturity=3
    )

    results = []

    for barrier in [80, 90, 100]:
        payoff = UpAndOutBasketCall(strike=100, barrier=barrier)
        pricer = SRFQI(model, payoff, nb_epochs=5, hidden_size=100, factors=(1., 1.))
        price, _ = pricer.price()

        status = "✓ PASS" if price < 0.01 else "❌ FAIL"
        results.append((barrier, price, status))
        print(f"  Barrier={barrier:3d}, Spot=100: Price={price:.6f} ... {status}")

    all_passed = all(r[2] == "✓ PASS" for r in results)
    print("\n" + "-" * 70)
    if all_passed:
        print("✓ TEST 5 PASSED: All knocked-out barriers give zero value")
    else:
        print("❌ TEST 5 FAILED: Some barriers incorrectly have non-zero value")
    print("=" * 70)

    return all_passed


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RFQI/SRFQI TEST SUITE")
    print("=" * 70)

    results = {}

    # Run each test
    print("\nRunning tests...")

    try:
        results['rfqi_vanilla'] = test_rfqi_vanilla()
    except Exception as e:
        print(f"❌ Test 'rfqi_vanilla' failed with exception: {e}")
        results['rfqi_vanilla'] = False

    try:
        results['srfqi_barrier'] = test_srfqi_barrier()
    except Exception as e:
        print(f"❌ Test 'srfqi_barrier' failed with exception: {e}")
        results['srfqi_barrier'] = False

    try:
        results['rfqi_protection'] = test_rfqi_protection()
    except Exception as e:
        print(f"❌ Test 'rfqi_protection' failed with exception: {e}")
        results['rfqi_protection'] = False

    try:
        results['srfqi_protection'] = test_srfqi_protection()
    except Exception as e:
        print(f"❌ Test 'srfqi_protection' failed with exception: {e}")
        results['srfqi_protection'] = False

    try:
        results['srfqi_knockout'] = test_barrier_knockout_srfqi()
    except Exception as e:
        print(f"❌ Test 'srfqi_knockout' failed with exception: {e}")
        results['srfqi_knockout'] = False

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED ✅")
        print("\nRFQI and SRFQI are working correctly!")
        print("You now have 4 algorithms:")
        print("  - RLSM (standard) + SRLSM (path-dependent)")
        print("  - RFQI (standard) + SRFQI (path-dependent)")
    else:
        print("❌ SOME TESTS FAILED ❌")
        print("\nReview the failed tests and fix the issues.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)