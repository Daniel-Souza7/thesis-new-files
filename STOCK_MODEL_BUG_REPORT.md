# Stock Model Bug Report

## Summary
Found **3 critical bugs** and **1 potential issue** in stock model implementations.

---

## BUG 1: FractionalBlackScholes - Missing drift_fct() and diffusion_fct() Implementation

**Severity:** CRITICAL

**Location:** `/home/user/thesis-new-files/optimal_stopping/data/stock_model.py` lines 168-199

**Issue:**
FractionalBlackScholes.generate_one_path() calls `self.diffusion_fct()` and `self.drift_fct()` (lines 193, 196) but does NOT implement these methods. The base Model class has them as NotImplementedError.

**Code:**
```python
class FractionalBlackScholes(Model):
    def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        # ... initialization ...

    def generate_one_path(self):
        # Line 193: Calls self.diffusion_fct() - NOT IMPLEMENTED!
        diffusion = self.diffusion_fct(previous_spots, k * self.dt)
        # Line 196: Calls self.drift_fct() - NOT IMPLEMENTED!
        + self.drift_fct(previous_spots, k * self.dt) * self.dt
```

**Error:**
```
NotImplementedError: Subclasses must implement diffusion_fct()
```

**Expected Behavior:**
FractionalBlackScholes should implement drift_fct() and diffusion_fct() similar to BlackScholes:
```python
def drift_fct(self, x, t):
    return self.drift * x

def diffusion_fct(self, x, t, v=0):
    return self.volatility * x
```

**Fix:**
Add these methods to FractionalBlackScholes class:
```python
def drift_fct(self, x, t):
    del t
    return self.drift * x

def diffusion_fct(self, x, t, v=0):
    del t
    return self.volatility * x
```

---

## BUG 2: RealDataModel - Missing 'name' Parameter in super().__init__()

**Severity:** CRITICAL

**Location:** `/home/user/thesis-new-files/optimal_stopping/data/real_data.py` lines 54-121

**Issue:**
RealDataModel calls `super().__init__(**kwargs)` but doesn't include the required 'name' parameter that Model.__init__() expects.

**Model.__init__ Signature (line 24):**
```python
def __init__(self, drift, dividend, volatility, spot, nb_stocks,
             nb_paths, nb_dates, maturity, name, risk_free_rate=None, **keywords):
```

**RealDataModel.__init__ Call (line 121):**
```python
super().__init__(**kwargs)  # Missing 'name' parameter!
```

**Error:**
```
TypeError: Model.__init__() missing 2 required positional arguments: 'dividend' and 'name'
```

**Comparison with BlackScholes (correct implementation, line 79-82):**
```python
super(BlackScholes, self).__init__(
    drift=drift, dividend=dividend, volatility=volatility,
    nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
    spot=spot, maturity=maturity, name="BlackScholes", **keywords)
```

Note: BlackScholes explicitly passes `name="BlackScholes"`

**Fix:**
Modify RealDataModel.__init__ to:
1. Accept dividend parameter (currently missing)
2. Pass 'name' to super().__init__()

**Current Code (line 120-121):**
```python
# Initialize base class
super().__init__(**kwargs)
```

**Fixed Code:**
```python
# Set name for base class
kwargs['name'] = 'RealData'

# Initialize base class
super().__init__(**kwargs)
```

Or better, add dividend parameter explicitly:
```python
def __init__(
    self,
    tickers: Optional[List[str]] = None,
    start_date: str = '2010-01-01',
    end_date: str = '2024-01-01',
    exclude_crisis: bool = False,
    only_crisis: bool = False,
    drift_override: Optional[float] = None,
    volatility_override: Optional[float] = None,
    avg_block_length: Optional[int] = None,
    cache_data: bool = True,
    dividend: float = 0.0,  # ADD THIS
    **kwargs
):
    # ... existing code ...

    # Set dividend in kwargs if not already present
    if 'dividend' not in kwargs:
        kwargs['dividend'] = dividend
    kwargs['name'] = 'RealData'

    super().__init__(**kwargs)
```

---

## BUG 3: RealDataModel - Drift/Volatility Tuple Conversion Issue

**Severity:** HIGH

**Location:** `/home/user/thesis-new-files/optimal_stopping/data/real_data.py` lines 115-118

**Issue:**
When configs pass `drift=(None,)` or `volatilities=(None,)` (as tuples), the code at lines 115-118 doesn't properly convert them to float before passing to Model.__init__():

```python
if 'drift' not in kwargs or kwargs.get('drift') is None:
    kwargs['drift'] = 0.05
```

The condition `kwargs.get('drift') is None` evaluates to False when drift=(None,) (a tuple), so the default is not applied. Then kwargs['drift']=(None,) is passed to Model.__init__(), which expects a float.

**Problem Flow:**
1. Config passes: `drift=(None,)` (tuple)
2. Line 115 check: `kwargs.get('drift') is None` → `(None,) is None` → False
3. Line 116 NOT executed: `kwargs['drift']` stays as `(None,)`
4. Line 121: `super().__init__(**kwargs)` called with `drift=(None,)`
5. Model.__init__ line 27: `self.drift = (None,) - dividend` → TypeError

**Error:**
```
TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
```

**Fix:**
Better tuple unpacking logic. After lines 90-111, explicitly convert tuples to floats:

```python
# After line 111, add:
# Ensure drift and volatility are floats, not tuples
if isinstance(kwargs.get('drift'), (tuple, list)):
    kwargs['drift'] = kwargs['drift'][0] if len(kwargs['drift']) > 0 else 0.05
if isinstance(kwargs.get('volatility'), (tuple, list)):
    kwargs['volatility'] = kwargs['volatility'][0] if len(kwargs['volatility']) > 0 else 0.2

# Then ensure they're not None
if kwargs.get('drift') is None:
    kwargs['drift'] = 0.05
if kwargs.get('volatility') is None:
    kwargs['volatility'] = 0.2
```

---

## POTENTIAL ISSUE: RoughHeston Variable Handling

**Severity:** MEDIUM (Not a bug, but fragile)

**Location:** `/home/user/thesis-new-files/optimal_stopping/data/stock_model.py` lines 436-460

**Issue:**
The `get_frac_var()` method works with both 1D and 2D variance arrays. The handling is correct but fragile:

```python
def get_frac_var(self, vars, dZ, step, la, thet, vol):
    v0 = vars[0]  # Works for both 1D and 2D
    times = (...)
    if len(vars.shape) == 2:
        times = np.repeat(np.expand_dims(times, 1), vars.shape[1], axis=1)
    # ... rest of code
```

If someone changes the calling code to pass 3D arrays, this will fail silently or produce wrong results. Consider adding assertions:

```python
assert len(vars.shape) in [1, 2], f"vars must be 1D or 2D, got shape {vars.shape}"
```

---

## Test Results Summary

### Passing Tests
- ✓ BlackScholes with nb_stocks=1
- ✓ BlackScholes with nb_stocks=5
- ✓ Heston with nb_stocks=1
- ✓ Heston with nb_stocks=5
- ✓ RoughHeston with nb_stocks=1
- ✓ RoughHeston with nb_stocks=5
- ✓ Discount factor correctly uses rate (not drift)

### Failing Tests
- ✗ FractionalBlackScholes (all Hurst parameters) - Missing drift_fct/diffusion_fct
- ✗ RealDataModel with drift=None - Missing 'name' parameter
- ✗ RealDataModel with drift=0.05 - Missing 'name' parameter
- ✗ RealDataModel with drift=(None,) - Missing 'name' parameter

---

## Recommended Fix Priority

1. **CRITICAL - Do First:**
   - Add drift_fct() and diffusion_fct() to FractionalBlackScholes
   - Add 'name' parameter to RealDataModel super().__init__()

2. **HIGH - Do Second:**
   - Improve tuple conversion logic in RealDataModel (lines 115-118)
   - Add dividend parameter to RealDataModel.__init__()

3. **MEDIUM - Do Third:**
   - Add shape assertions to RoughHeston.get_frac_var()

---

## Test Script
A comprehensive test script was created at: `/home/user/thesis-new-files/test_stock_models.py`

Run with: `python test_stock_models.py`

This script tests:
- BlackScholes with nb_stocks=1 and nb_stocks=5
- Heston with stochastic volatility
- FractionalBlackScholes with different Hurst parameters
- RoughHeston with rough volatility
- RealDataModel with empirical and specified drift/volatility
- Discount factor calculation (rate vs drift)
