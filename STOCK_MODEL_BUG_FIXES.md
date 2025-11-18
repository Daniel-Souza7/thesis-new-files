# Stock Model Bug Fixes - Implementation Guide

## Overview
3 critical bugs found in stock models that must be fixed before experiments can run:
1. **FractionalBlackScholes**: Missing drift_fct() and diffusion_fct()
2. **RealDataModel**: Missing 'name' parameter in super().__init__()
3. **RealDataModel**: Improper tuple-to-float conversion for drift/volatility

---

## BUG #1: FractionalBlackScholes Missing Methods

### File
`/home/user/thesis-new-files/optimal_stopping/data/stock_model.py`

### Current Problem (Lines 168-199)
FractionalBlackScholes calls `self.drift_fct()` and `self.diffusion_fct()` in generate_one_path() (lines 193, 196) but doesn't implement these methods. The base Model class raises NotImplementedError.

### Error Trace
```
File "/optimal_stopping/data/stock_model.py", line 193, in generate_one_path
    diffusion = self.diffusion_fct(previous_spots, k * self.dt)
File "/optimal_stopping/data/stock_model.py", line 57, in diffusion_fct
    raise NotImplementedError("Subclasses must implement diffusion_fct()")
NotImplementedError: Subclasses must implement diffusion_fct()
```

### Fix
Add these two methods to FractionalBlackScholes class after __init__:

```python
class FractionalBlackScholes(Model):
    def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        super(FractionalBlackScholes, self).__init__(
            drift=drift, dividend=dividend, volatility=volatility,
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, name="FractionalBlackScholes", **keywords
        )
        self.hurst = hurst
        self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method='cholesky')

    # ADD THESE TWO METHODS:
    def drift_fct(self, x, t):
        """Drift function for SDE."""
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        """Diffusion function for SDE."""
        del t
        return self.volatility * x

    def generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates+1) with prices."""
        # ... rest of existing code ...
```

**Why this works:**
- BlackScholes uses the same SDE: dS = mu*S*dt + sigma*S*dW
- FractionalBlackScholes uses the same drift and diffusion functions, just with fractional Brownian motion instead of standard BM
- The `del t` removes the unused parameter (required to match base class signature)

---

## BUG #2: RealDataModel Missing 'name' Parameter

### File
`/home/user/thesis-new-files/optimal_stopping/data/real_data.py`

### Current Problem (Lines 54-121)
RealDataModel.__init__() calls `super().__init__(**kwargs)` but 'name' is not in kwargs. Model.__init__() requires it.

### Model.__init__ Signature (Line 24 in stock_model.py)
```python
def __init__(self, drift, dividend, volatility, spot, nb_stocks,
             nb_paths, nb_dates, maturity, name, risk_free_rate=None, **keywords):
```

### Current RealDataModel Code (Line 121)
```python
# Initialize base class
super().__init__(**kwargs)  # Missing 'name'!
```

### Error
```
TypeError: Model.__init__() missing 2 required positional arguments: 'dividend' and 'name'
```

### Fix Option 1 (Minimal - Add 'name' only)
After line 120, add:

```python
# Set model name for base class
kwargs['name'] = 'RealData'

# Initialize base class
super().__init__(**kwargs)
```

### Fix Option 2 (Better - Add dividend parameter too)
Modify the __init__ signature and add dividend to kwargs:

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
    dividend: float = 0.0,  # ADD THIS PARAMETER
    **kwargs
):
    # ... existing lines 86-120 ...

    # Add dividend to kwargs if not present
    if 'dividend' not in kwargs:
        kwargs['dividend'] = dividend

    # Set model name for base class
    kwargs['name'] = 'RealData'

    # Initialize base class
    super().__init__(**kwargs)
```

**Recommended:** Use Fix Option 2 because:
- Makes dividend explicit and configurable
- Matches other model implementations (BlackScholes, Heston)
- More consistent API

---

## BUG #3: RealDataModel Improper Tuple Conversion

### File
`/home/user/thesis-new-files/optimal_stopping/data/real_data.py`

### Current Problem (Lines 115-118)
When configs pass `drift=(None,)` or `volatilities=(None,)`, the condition check doesn't work properly:

```python
if 'drift' not in kwargs or kwargs.get('drift') is None:
    kwargs['drift'] = 0.05
```

The check `kwargs.get('drift') is None` is False when drift=(None,) (a tuple), so defaults aren't applied.

### Why It's a Problem
1. Config passes: `drift=(None,)` (tuple from itertools.product)
2. Line 115: `kwargs.get('drift') is None` evaluates to `(None,) is None` → False
3. Default NOT applied, kwargs['drift'] stays as (None,)
4. Line 121: Model.__init__ receives drift=(None,)
5. Model.__init__ line 27: `self.drift = (None,) - dividend` → TypeError

### Fix
Replace lines 115-118 with better tuple handling:

```python
# Base Model class requires drift and volatility for calculations
# If None, provide sensible defaults (will use empirical values later)

# Ensure drift is a float (not a tuple)
drift_val = kwargs.get('drift')
if isinstance(drift_val, (tuple, list)):
    kwargs['drift'] = drift_val[0] if len(drift_val) > 0 else None

if kwargs.get('drift') is None:
    kwargs['drift'] = 0.05

# Ensure volatility is a float (not a tuple)
vol_val = kwargs.get('volatility')
if isinstance(vol_val, (tuple, list)):
    kwargs['volatility'] = vol_val[0] if len(vol_val) > 0 else None

if kwargs.get('volatility') is None:
    kwargs['volatility'] = 0.2
```

**Why this works:**
- Explicitly extracts tuple elements before checking for None
- Ensures kwargs['drift'] and kwargs['volatility'] are always floats
- Handles both (None,) → None and (0.05,) → 0.05 cases

---

## Implementation Checklist

### Priority 1 (Critical - Must Fix Now)
- [ ] Add drift_fct() and diffusion_fct() to FractionalBlackScholes (Lines 176-180 approx)
- [ ] Add 'name' parameter to RealDataModel super().__init__() call (Line 121)

### Priority 2 (High - Should Fix Soon)
- [ ] Add dividend parameter to RealDataModel.__init__() signature
- [ ] Improve tuple conversion logic in RealDataModel (Lines 115-118)

### Priority 3 (Medium - Nice to Have)
- [ ] Add shape assertions to RoughHeston.get_frac_var() for robustness

---

## Verification Steps

After applying fixes, run the test suite:

```bash
python test_stock_models.py
```

**Expected Results After Fixes:**
```
TEST: BlackScholes
1. Testing BlackScholes with nb_stocks=1...
   ✓ Paths shape: (10, 1, 11)
   ✓ PASS

2. Testing BlackScholes with nb_stocks=5...
   ✓ Paths shape: (10, 5, 11)
   ✓ PASS

TEST: Heston
1. Testing Heston with nb_stocks=1...
   ✓ Stock paths shape: (5, 1, 11)
   ✓ PASS

TEST: FractionalBlackScholes
1. Testing FractionalBlackScholes with Hurst=0.3...
   ✓ Paths shape: (5, 1, 11)  // Previously FAILED
   ✓ PASS

TEST: RoughHeston
1. Testing RoughHeston with nb_stocks=1...
   ✓ Stock paths shape: (3, 1, 11)
   ✓ PASS

TEST: RealDataModel
1. Testing RealDataModel with empirical drift/volatility...
   ✓ Paths shape: (2, 2, 11)  // Previously FAILED
   ✓ PASS

2. Testing RealDataModel with specified drift/volatility...
   ✓ Paths shape: (2, 2, 11)  // Previously FAILED
   ✓ PASS

TEST SUITE COMPLETE - ALL TESTS PASSED
```

---

## Related Code References

### BlackScholes Implementation (Correct Example)
File: `/home/user/thesis-new-files/optimal_stopping/data/stock_model.py` Lines 76-127

Shows the correct pattern:
```python
class BlackScholes(Model):
    def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        super(BlackScholes, self).__init__(
            drift=drift, dividend=dividend, volatility=volatility,
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, name="BlackScholes", **keywords)

    def drift_fct(self, x, t):
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        del t
        return self.volatility * x
```

### Heston Implementation (Correct Example)
File: `/home/user/thesis-new-files/optimal_stopping/data/stock_model.py` Lines 266-282

Shows correct parameter passing:
```python
class Heston(Model):
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, dividend=dividend, name="Heston", **kwargs
        )
```

---

## Summary

| Bug | File | Line | Severity | Status |
|-----|------|------|----------|--------|
| Missing drift_fct/diffusion_fct | stock_model.py | 168-199 | CRITICAL | Unfixed |
| Missing 'name' parameter | real_data.py | 121 | CRITICAL | Unfixed |
| Improper tuple conversion | real_data.py | 115-118 | HIGH | Unfixed |
| Shape assertion missing | stock_model.py | 436-460 | MEDIUM | Unfixed |

All bugs are localized and straightforward to fix. Estimated time: 15-20 minutes for all fixes.
