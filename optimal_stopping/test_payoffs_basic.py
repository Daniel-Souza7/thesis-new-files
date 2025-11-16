"""
Basic tests for new payoff implementation.

Run with: python -m optimal_stopping.test_payoffs_basic
"""

import numpy as np
from optimal_stopping.payoffs import (
    BasketCall, BasketPut, GeometricCall, GeometricPut, MaxCall, MinPut,
    Call, Put,
    get_payoff_class, list_payoffs,
    create_barrier_payoff, _PAYOFF_REGISTRY
)


def test_base_payoffs():
    """Test that base payoffs work correctly."""
    print("\n=== Testing Base Payoffs ===")

    # Create test data
    np.random.seed(42)
    nb_paths = 100
    nb_stocks = 3
    strike = 100

    # Test data: (nb_paths, nb_stocks)
    S = np.random.uniform(80, 120, (nb_paths, nb_stocks))

    # Test BasketCall
    payoff = BasketCall(strike)
    result = payoff.eval(S)
    assert result.shape == (nb_paths,), f"Wrong shape: {result.shape}"
    assert np.all(result >= 0), "Payoff should be non-negative"
    print(f"✅ BasketCall: mean={result.mean():.2f}, max={result.max():.2f}")

    # Test GeometricCall
    payoff = GeometricCall(strike)
    result = payoff.eval(S)
    assert result.shape == (nb_paths,)
    print(f"✅ GeometricCall: mean={result.mean():.2f}")

    # Test Call (single asset)
    S_single = np.random.uniform(80, 120, (nb_paths, 1))
    payoff = Call(strike)
    result = payoff.eval(S_single)
    assert result.shape == (nb_paths,)
    print(f"✅ Call: mean={result.mean():.2f}")

    print("✅ All base payoffs passed!")


def test_auto_registration():
    """Test that payoffs are auto-registered."""
    print("\n=== Testing Auto-Registration ===")

    # Check that payoffs are registered
    assert 'BasketCall' in _PAYOFF_REGISTRY
    assert 'BskCall' in _PAYOFF_REGISTRY  # Abbreviation
    assert 'Call' in _PAYOFF_REGISTRY

    print(f"✅ {len(_PAYOFF_REGISTRY)} payoffs registered")

    # Test get_payoff_class
    payoff_cls = get_payoff_class('BasketCall')
    assert payoff_cls == BasketCall

    payoff_cls = get_payoff_class('BskCall')  # By abbreviation
    assert payoff_cls == BasketCall

    print("✅ Auto-registration working!")


def test_barrier_payoffs():
    """Test that barrier payoffs work."""
    print("\n=== Testing Barrier Payoffs ===")

    # Create test data with full path
    np.random.seed(42)
    nb_paths = 100
    nb_stocks = 3
    nb_dates = 10
    strike = 100
    barrier = 110

    # Generate random walk: (nb_paths, nb_stocks, nb_dates+1)
    S = np.zeros((nb_paths, nb_stocks, nb_dates + 1))
    S[:, :, 0] = 100  # Start at 100
    for t in range(1, nb_dates + 1):
        S[:, :, t] = S[:, :, t-1] * np.exp(np.random.normal(0, 0.01, (nb_paths, nb_stocks)))

    # Test Up-and-Out Basket Call
    UO_BskCall = get_payoff_class('UO_BasketCall')
    payoff = UO_BskCall(strike, barrier=barrier)
    assert payoff.is_path_dependent == True

    result = payoff.eval(S)
    assert result.shape == (nb_paths,)
    assert np.all(result >= 0)

    # Count how many paths hit the barrier
    max_reached = np.max(S, axis=(1, 2))
    hit_barrier = (max_reached >= barrier).sum()
    paid_out = (result > 0).sum()

    print(f"✅ UO-BskCall:")
    print(f"   Paths hitting barrier: {hit_barrier}/{nb_paths}")
    print(f"   Paths paying out: {paid_out}/{nb_paths}")
    print(f"   Mean payoff: {result.mean():.2f}")

    # Test Down-and-In Put
    DI_Put = get_payoff_class('DI_Put')
    payoff = DI_Put(strike, barrier=90)
    result = payoff.eval(S[:, :1, :])  # Single stock

    print(f"✅ DI-Put: mean={result.mean():.2f}")

    print("✅ All barrier payoffs passed!")


def test_payoff_registry_size():
    """Check how many payoffs are registered."""
    print("\n=== Payoff Registry Size ===")

    all_payoffs = list_payoffs()
    base_payoffs = [p for p in all_payoffs if '_' not in p or 'Call' not in p]
    barrier_payoffs = [p for p in all_payoffs if any(b in p for b in ['UO_', 'DO_', 'UI_', 'DI_', 'UODO_', 'UIDI_', 'UIDO_', 'UODI_', 'PTB_', 'StepB_', 'DStepB_'])]

    print(f"Total registered: {len(all_payoffs)}")
    print(f"Base payoffs: {len(base_payoffs)}")
    print(f"Barrier payoffs: {len(barrier_payoffs)}")

    expected_base = 8  # 6 basket + 2 single
    expected_barrier = 8 * 11  # 8 base × 11 barrier types

    assert len(base_payoffs) >= expected_base, f"Expected at least {expected_base} base payoffs, got {len(base_payoffs)}"
    assert len(barrier_payoffs) >= expected_barrier, f"Expected at least {expected_barrier} barrier payoffs, got {len(barrier_payoffs)}"

    print(f"✅ Expected counts verified!")


if __name__ == "__main__":
    print("="*60)
    print("PAYOFF IMPLEMENTATION TEST SUITE")
    print("="*60)

    try:
        test_auto_registration()
        test_base_payoffs()
        test_barrier_payoffs()
        test_payoff_registry_size()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
