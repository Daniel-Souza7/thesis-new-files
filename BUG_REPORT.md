# Bug Report: run_algo.py Testing Results

**Date:** 2025-11-18
**Test Configuration:** nb_paths=500, nb_dates=3, nb_stocks=3, nb_runs=1

---

## Summary

Testing revealed **2 critical bugs** affecting barrier-wrapped payoffs:

1. **Barrier wrapper breaks path-dependent multi-asset payoffs** (CRITICAL)
2. **Incomplete handling of barrier types on single-asset payoffs** (DESIGN ISSUE)

---

## BUG #1: Barrier Wrapper Fails for Path-Dependent Multi-Asset Payoffs

### Severity: CRITICAL

**Affected Payoffs:** All 220 barrier variants of multi-asset path-dependent payoffs
- Examples: `UO_MaxDispersionCall`, `StepB_AsianFixedStrikeCall`, `DO_MaxDispersionPut`, etc.

### Symptom

```
numpy.exceptions.AxisError: axis 2 is out of bounds for array of dimension 2
```

### Root Cause

**File:** `/home/user/thesis-new-files/optimal_stopping/payoffs/barrier_wrapper.py`
**Lines:** 53-75 (the `eval()` method)

```python
def eval(self, X):
    """
    Args:
        X: Array of shape (nb_paths, nb_stocks, nb_dates+1) - FULL path history
    """
    nb_paths, nb_stocks, nb_dates = X.shape

    # ... line 69 ...
    base_value = self.base_payoff.eval(X[:, :, -1])  # BUG: Extracts only final timestep (2D)

    barrier_active = self._check_barrier(X)
    return base_value * barrier_active
```

### The Problem

1. The barrier wrapper receives full path history `X` with shape `(nb_paths, nb_stocks, nb_dates+1)` (3D)
2. Line 69 extracts only the final timestep: `X[:, :, -1]` → reduces to `(nb_paths, nb_stocks)` (2D)
3. Passes this 2D array to the base payoff's `eval()` method
4. Path-dependent payoffs like `MaxDispersionCall` expect 3D input for their calculations
5. `MaxDispersionCall.eval()` tries: `np.max(X, axis=(1, 2))` on 2D array → **AxisError**

### Why Some Work, Some Don't

- **Payoffs that FAIL:** `MaxDispersionCall`, `MaxDispersionPut`, `AsianFixedStrikeCall`, `AsianFixedStrikePut`, `AsianFloatingStrikeCall`, `AsianFloatingStrikePut`
  - These directly use `np.max(X, axis=(1, 2))` expecting 3D input

- **Payoffs that WORK (by accident):** `LookbackFixedCall`, `LookbackFixedPut`, `RangeCall_Single`, `RangePut_Single`
  - These have defensive code: `if X.ndim == 3: X = X[:, 0, :]` to handle both 2D and 3D inputs
  - They gracefully degrade to 2D processing

### Test Results

```
Testing: UO_MaxDispersionCall
  ✗ FAILED - AxisError: axis 2 is out of bounds for array of dimension 2

Testing: StepB_MaxDispersionCall
  ✗ FAILED - AxisError: axis 2 is out of bounds for array of dimension 2

Testing: UO_AsianFixedStrikeCall
  ✗ FAILED - AxisError: axis 1 is out of bounds for array of dimension 1

Testing: DO_LookbackFixedCall
  ✓ SUCCESS (works by accident due to defensive code)

Testing: UI_RangeCall_Single
  ✓ SUCCESS (works by accident due to defensive code)
```

### Recommended Fix

The barrier wrapper should NOT extract the final timestep for path-dependent base payoffs. Change line 69 from:

```python
base_value = self.base_payoff.eval(X[:, :, -1])
```

To:

```python
if self.base_payoff.is_path_dependent:
    # Pass full path history for path-dependent payoffs
    base_value = self.base_payoff.eval(X)
else:
    # Extract final timestep for standard payoffs
    base_value = self.base_payoff.eval(X[:, :, -1])
```

---

## BUG #2: Barrier Wrapping of Single-Asset Path-Dependent Payoffs is Incomplete

### Severity: DESIGN ISSUE

**Affected Payoffs:** Single-asset path-dependent payoffs with barrier wrapping
- Examples: `UO_LookbackFixedCall`, `StepB_RangeCall_Single`

### Issue

Single-asset payoffs like `LookbackFixedCall` have defensive code to handle both 2D and 3D inputs:
```python
if X.ndim == 3:
    X = X[:, 0, :]  # Extract single stock
```

However, this defensive handling should be **consistent across all payoffs**, not just single-asset ones.

The current design where:
- Single-asset payoffs work with barrier wrappers
- Multi-asset path-dependent payoffs break with barrier wrappers

...is inconsistent and confusing.

### Recommendation

**Option 1 (Preferred):** Fix the barrier wrapper to handle path-dependent payoffs correctly (see Bug #1 fix)

**Option 2:** Add the same defensive code to all path-dependent multi-asset payoffs to accept both 2D and 3D inputs

---

## Test Summary

### Payoff Import Tests
- ✓ All 11 required payoffs found in registry
- ✓ 716 total payoffs registered successfully

### Payoff Instantiation Tests
- ✓ BasketCall (standard)
- ✓ MaxCall (standard)
- ✓ MinPut (standard)
- ✓ DispersionCall (standard)
- ✓ AsianFixedStrikeCall (path-dependent)
- ✓ LookbackFixedCall (path-dependent)
- ✓ RangeCall_Single (path-dependent)
- ✓ MaxDispersionCall (path-dependent)
- ✓ UO_BasketCall (barrier - works)
- ✓ DO_BasketPut (barrier - works)
- ✗ StepB_MaxDispersionCall (barrier - BROKEN)

### Barrier Evaluation Tests

**Multi-Asset Path-Dependent Payoffs:**
```
UO_MaxDispersionCall:       ✗ AxisError: axis 2 is out of bounds for array of dimension 2
DO_MaxDispersionCall:       ✗ AxisError: axis 2 is out of bounds for array of dimension 2
UI_MaxDispersionCall:       ✗ AxisError: axis 2 is out of bounds for array of dimension 2
DI_MaxDispersionCall:       ✗ AxisError: axis 2 is out of bounds for array of dimension 2
UODO_MaxDispersionCall:     ✗ AxisError: axis 2 is out of bounds for array of dimension 2
UIDI_MaxDispersionCall:     ✗ AxisError: axis 2 is out of bounds for array of dimension 2
UIDO_MaxDispersionCall:     ✗ AxisError: axis 2 is out of bounds for array of dimension 2
UODI_MaxDispersionCall:     ✗ AxisError: axis 2 is out of bounds for array of dimension 2
PTB_MaxDispersionCall:      ✗ AxisError: axis 2 is out of bounds for array of dimension 2
StepB_MaxDispersionCall:    ✗ AxisError: axis 2 is out of bounds for array of dimension 2
DStepB_MaxDispersionCall:   ✗ AxisError: axis 2 is out of bounds for array of dimension 2

UO_AsianFixedStrikeCall:    ✗ AxisError: axis 1 is out of bounds for array of dimension 1
[Similar failures for all Asian barrier variants]
```

**Single-Asset Path-Dependent Payoffs:**
```
UO_LookbackFixedCall:       ✓ SUCCESS
DO_LookbackFixedCall:       ✓ SUCCESS
[All single-asset barrier variants work]

UO_RangeCall_Single:        ✓ SUCCESS
[All single-asset barrier variants work]
```

**Multi-Asset Standard Payoffs:**
```
UO_BasketCall:              ✓ SUCCESS
DO_BasketPut:               ✓ SUCCESS
[All standard barrier variants work]
```

---

## Bug Impact

### Broken Functionality
- **220 barrier-wrapped multi-asset path-dependent payoffs are completely non-functional**
  - `StepB_MaxDispersionCall` and related (11 barriers × 2 base payoffs)
  - `UO_AsianFixedStrikeCall` and related (11 barriers × 4 base payoffs)
  - And many more combinations

### Affected Use Cases
1. **Barrier options on dispersion:** Cannot price options on volatility with barriers
2. **Barrier options on baskets with averaging:** Cannot price Asian basket options with barriers
3. **Step barriers on path-dependent payoffs:** StepB and DStepB variants all fail

### Workaround
Currently, users can:
- Use standard (non-barrier) versions of path-dependent payoffs
- Use barrier versions of standard payoffs only
- Avoid any combination of barriers with multi-asset path-dependent payoffs

---

## Files Modified by Testing

The following files were created for testing but are not part of the codebase:
- `/home/user/thesis-new-files/test_run_algo.py` (test script)

No modifications were made to the actual codebase.

---

## Recommendations

### Priority 1: CRITICAL - Fix Bug #1
Implement the recommended fix in `barrier_wrapper.py` to handle path-dependent base payoffs correctly. This will restore functionality to 220 currently broken payoffs.

### Priority 2: TESTING
After fix is implemented, add unit tests for all combinations:
- Barrier types × Path-dependent payoffs
- Ensure both single-asset and multi-asset path-dependent payoffs work

### Priority 3: CONSISTENCY
Consider either:
1. Adding defensive dimension handling to all path-dependent payoffs (like LookbackFixedCall has)
2. Or centralizing this logic in the Payoff base class

---

## Additional Notes

### What Works Well
- All standard (non-barrier) payoffs work correctly
- Barrier variants of standard payoffs work correctly
- The payoff registry is comprehensive (716 payoffs registered)
- Algorithm routing logic is sound

### What Needs Attention
- Barrier wrapping of path-dependent payoffs needs fixing
- Consider adding more comprehensive tests for barrier payoff combinations
- Document which payoff combinations are valid

