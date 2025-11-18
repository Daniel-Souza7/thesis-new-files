# Code Fix Required

## Bug #1: Barrier Wrapper Line 69 - Fix for Path-Dependent Payoffs

### Location
**File:** `/home/user/thesis-new-files/optimal_stopping/payoffs/barrier_wrapper.py`
**Method:** `BarrierPayoff.eval()`
**Lines:** 53-75

### Current (Broken) Code

```python
def eval(self, X):
    """
    Evaluate barrier payoff.

    Args:
        X: Array of shape (nb_paths, nb_stocks, nb_dates+1) - FULL path history

    Returns:
        Array of shape (nb_paths,) with payoff at maturity
    """
    nb_paths, nb_stocks, nb_dates = X.shape

    # Propagate initial prices to base payoff for normalization
    self.base_payoff.initial_prices = X[:, :, 0]

    # Evaluate base payoff at maturity
    base_value = self.base_payoff.eval(X[:, :, -1])  # <-- BUG HERE

    # Check barrier conditions
    barrier_active = self._check_barrier(X)

    # Apply barrier logic (active means payoff survives for Out, or activates for In)
    return base_value * barrier_active
```

### Problem
Line 69 `self.base_payoff.eval(X[:, :, -1])` extracts only the final timestep, reducing the array from 3D to 2D. This breaks path-dependent base payoffs that need the full path history.

### Fixed Code

```python
def eval(self, X):
    """
    Evaluate barrier payoff.

    Args:
        X: Array of shape (nb_paths, nb_stocks, nb_dates+1) - FULL path history

    Returns:
        Array of shape (nb_paths,) with payoff at maturity
    """
    nb_paths, nb_stocks, nb_dates = X.shape

    # Propagate initial prices to base payoff for normalization
    self.base_payoff.initial_prices = X[:, :, 0]

    # Evaluate base payoff - pass full path for path-dependent, final timestep for standard
    if self.base_payoff.is_path_dependent:
        # Path-dependent payoffs need the full path history
        base_value = self.base_payoff.eval(X)
    else:
        # Standard payoffs only need the final price
        base_value = self.base_payoff.eval(X[:, :, -1])

    # Check barrier conditions
    barrier_active = self._check_barrier(X)

    # Apply barrier logic (active means payoff survives for Out, or activates for In)
    return base_value * barrier_active
```

### Why This Fix Works

1. **For standard (non-path-dependent) payoffs:**
   - `X[:, :, -1]` extracts the final price: shape `(nb_paths, nb_stocks)`
   - These payoffs expect 2D input, so it works as before
   - Example: `BasketCall.eval()` operates on current prices only

2. **For path-dependent base payoffs:**
   - Pass the full `X`: shape `(nb_paths, nb_stocks, nb_dates+1)`
   - These payoffs expect 3D input to calculate over time/space
   - Example: `MaxDispersionCall.eval()` calculates `max - min` over all stocks and times

3. **For barrier evaluation:**
   - `self._check_barrier(X)` always receives full path (unchanged)
   - Barriers operate on the complete path history, which is correct

### Test Verification

After applying this fix, all of these should work:

```python
import numpy as np
from optimal_stopping.payoffs import get_payoff_class

X = np.ones((10, 3, 4)) * 100.0
for i in range(10):
    X[i, :, :] = X[i, :, :] + np.random.randn(3, 4) * 5

# These will now work:
test_payoffs = [
    'UO_MaxDispersionCall',
    'StepB_MaxDispersionCall',
    'UO_AsianFixedStrikeCall',
    'DO_AsianFixedStrikePut',
    'UI_AsianFloatingStrikeCall',
]

for payoff_name in test_payoffs:
    payoff_class = get_payoff_class(payoff_name)
    payoff = payoff_class(
        strike=20.0,
        barrier=120.0,
        barrier_up=120.0,
        barrier_down=80.0,
        rate=0.02,
        maturity=1.0,
    )
    result = payoff(X)  # Should work now!
    assert result.shape == (10, 4)
    print(f"✓ {payoff_name}: SUCCESS")
```

### Impact

This single change will fix **220 broken payoffs**:
- All 11 barrier types × 20 multi-asset path-dependent base payoffs
- Examples:
  - MaxDispersionCall variants (2 base payoffs × 11 barriers = 22)
  - AsianFixedStrikeCall variants (4 base payoffs × 11 barriers = 44)
  - AsianFloatingStrikeCall variants (4 base payoffs × 11 barriers = 44)
  - And 7 more types of multi-asset path-dependent payoffs

---

## Bug #2: Documentation/Design Issue

No code change needed. This is a consistency issue where:
- Single-asset payoffs have defensive dimension-handling code
- Multi-asset payoffs don't

After fixing Bug #1, consider:
1. Documenting that barrier wrapper properly handles both payoff types
2. Adding tests for all barrier-payoff combinations
3. Consider whether to add defensive code to all path-dependent payoffs for consistency

### Example of Defensive Code (already in LookbackFixedCall)

```python
def eval(self, X):
    """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
    if X.ndim == 3:
        X = X[:, 0, :]  # Extract single stock
    
    max_price = np.max(X, axis=1)
    return np.maximum(0, max_price - self.strike)
```

This defensive approach allows the payoff to accept both 2D and 3D input, making it more robust.

---

## Verification Steps

1. Apply the fix to `barrier_wrapper.py` line 69-71
2. Run unit tests on barrier payoffs
3. Verify the 220 previously broken payoffs now work
4. Run the full test suite to ensure no regressions

