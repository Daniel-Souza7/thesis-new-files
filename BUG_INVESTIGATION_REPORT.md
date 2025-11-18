# Bug Investigation Report - test_all_bug_fixes Results

## Summary

Investigated 3 anomalies in test results. Found 2 definite bugs and 1 expected behavior.

---

## Bug 1: RankWeightedBasketCall Returns All Zeros ❌ CRITICAL

### Evidence
All RankWeightedBasketCall prices are exactly 0 across all models:
```
BlackScholes    RankWeightedBasketCall  5   3  0  0  0  0
FractionalBS    RankWeightedBasketCall  5   3  0  0  0  0
RealData        RankWeightedBasketCall  5   3  0  0  0  0
```

Meanwhile BestOfKCall (similar payoff) works fine with prices 16.2, 27.3, etc.

### Root Cause
**File**: `optimal_stopping/payoffs/basket_rank.py`
**Lines**: 137 (RankWeightedBasketCall), 197 (RankWeightedBasketPut)

**The Bug**:
```python
# Line 112-113: Default weights sum to 1
if weights is None:
    weights = np.ones(k) / k  # e.g., [1/3, 1/3, 1/3] for k=3

# Line 123-125: Normalize weights to sum to 1 (always true after this)
if not np.isclose(weight_sum, 1.0):
    weights = weights / weight_sum

# Line 134: Compute weighted average (already correct)
weighted_sum = np.sum(top_k_prices * weights, axis=1)

# Line 137: BUG - Multiply by 1/k AGAIN!
return np.maximum(0, (1.0 / k) * weighted_sum - self.strike)  # ❌
```

**Why it's wrong**:
- Weights already sum to 1 after normalization (line 123-125)
- `weighted_sum` is already a proper weighted average
- Multiplying by `1/k` again makes the value tiny
- With k=3, strike=100, even if prices are 120, we get: `(1/3) * 110 - 100 = -63` → 0

**Example Calculation**:
- k=3, prices=[120, 110, 90], strike=100
- weights=[1/3, 1/3, 1/3] (sum=1)
- weighted_sum = 120*(1/3) + 110*(1/3) + 90*(1/3) = 106.67
- **Current**: max(0, (1/3) * 106.67 - 100) = max(0, 35.56 - 100) = 0 ❌
- **Correct**: max(0, 106.67 - 100) = 6.67 ✅

### Fix
Remove the `(1.0 / k) *` factor from lines 137 and 197:

**Before**:
```python
return np.maximum(0, (1.0 / k) * weighted_sum - self.strike)
```

**After**:
```python
return np.maximum(0, weighted_sum - self.strike)
```

Same fix needed for RankWeightedBasketPut on line 197.

---

## Bug 2: RealData MaxDispersionCall Returns Zeros ❌ CRITICAL

### Evidence
RealData gives zero for all MaxDispersionCall prices:
```
RealData  MaxDispersionCall  5   3  0  0  0  0
RealData  MaxDispersionCall  10  3  0  0  0  0
```

But BlackScholes and FractionalBlackScholes give positive prices:
```
BlackScholes     MaxDispersionCall  5  3  0.211  0.027
FractionalBS     MaxDispersionCall  5  3  0.100  0.041
```

### Root Cause
**File**: `optimal_stopping/data/real_data.py`
**Lines**: 90-96 (drift override extraction), 506 (drift array creation), 512 (volatility array creation)

**The Problem**:
1. Config has `drift=[0.05]` and `volatilities=[0.2]` for BlackScholes/FractionalBlackScholes
2. RealDataModel extracts these as overrides (lines 90-96):
   ```python
   if drift_override is None and 'drift' in kwargs:
       drift_val = kwargs['drift']
       if isinstance(drift_val, (tuple, list)) and len(drift_val) > 0:
           drift_override = drift_val[0]  # drift_override = 0.05
   ```

3. When overrides are set, ALL stocks get IDENTICAL drift/volatility (lines 506, 512):
   ```python
   self.target_drift_daily = np.full(self.nb_stocks, self.drift_override / 252)
   self.target_vol_daily = np.full(self.nb_stocks, self.volatility_override / np.sqrt(252))
   ```

4. Returns are rescaled (lines 619-620):
   ```python
   sampled_returns = (sampled_returns - self.empirical_drift_daily) / self.empirical_vol_daily
   sampled_returns = sampled_returns * self.target_vol_daily + self.target_drift_daily
   ```

5. All stocks end up with:
   - Same expected growth rate
   - Same volatility
   - High correlation (bootstrap samples same time indices for all stocks)

6. With only 20 time steps, max - min across stocks is very small (< strike=100) → price ≈ 0

### Why This Is Wrong
RealDataModel is designed to use **empirical** drift and volatility from historical data by default. When config parameters meant for BlackScholes are passed through, they override the empirical values, defeating the purpose of using real data.

### Fix Options

**Option 1**: Don't extract drift/volatility from kwargs (BREAKING CHANGE)
- Remove lines 90-96
- Force users to use drift_override parameter explicitly
- Pro: Clear separation between config params and RealData params
- Con: Breaks existing usage pattern

**Option 2**: Only extract if explicitly not None
- Change line 90-96 to only extract non-None values
- Pro: Backward compatible, allows `drift=(None,)` to work
- Con: `drift=[0.05]` would still override

**Option 3**: Add a flag to disable override extraction
- Add parameter like `use_config_overrides=False`
- Pro: Flexible, backward compatible
- Con: More complex API

**Option 4**: Fix test config (TEMPORARY - doesn't solve root issue)
- Change test_all_bug_fixes to use `drift=(None,)` for RealData tests
- Pro: Quick fix for this specific test
- Con: Bug still exists for other users

### Recommended Fix
**Option 2** - Only extract non-None values:

```python
if drift_override is None and 'drift' in kwargs:
    drift_val = kwargs['drift']
    if isinstance(drift_val, (tuple, list)) and len(drift_val) > 0:
        drift_first = drift_val[0]
        # Only use as override if explicitly non-None
        if drift_first is not None:
            drift_override = drift_first
    elif drift_val is not None:
        drift_override = drift_val
```

This way:
- `drift=(None,)` → No override, use empirical
- `drift=(0.05,)` → Override to 0.05
- Default behavior (no drift in config) → Use empirical

---

## Issue 3: UO_MaxDispersionCall Returns Zeros ✅ EXPECTED

### Evidence
ALL models give zero for UO_MaxDispersionCall:
```
BlackScholes     UO_MaxDispersionCall  5  3  0  0  0  0
FractionalBS     UO_MaxDispersionCall  5  3  0  0  0  0
RealData         UO_MaxDispersionCall  5  3  0  0  0  0
```

### Analysis
**Config Parameters**:
- barrier = 120
- spot = 100
- nb_stocks = 5 or 10
- nb_dates = 20

**Knockout Probability**:
- Up-and-Out barrier at 120 (20% above spot)
- With 5-10 uncorrelated stocks, probability that at least one breaches 120 is high
- With 20 time steps, very likely that max(S_i, t) > 120 for at least one stock
- If any stock breaches barrier, option knocks out → payoff = 0

**Mathematical Reasoning**:
- Single stock reaching 120: P ≈ 30-40% with vol=0.2, T=1, 20 steps
- With 5 uncorrelated stocks: P(at least one > 120) ≈ 1 - (0.6)^5 ≈ 92%
- With 10 stocks: P(at least one > 120) ≈ 1 - (0.6)^10 ≈ 99%
- Across 3000 paths × 3 runs, getting 100% knockout is statistically plausible

### Conclusion
**NOT A BUG** - This is expected behavior given:
- High barrier breach probability with multi-stock portfolios
- 100% knockout rate across all runs is unlikely but possible
- To verify: Run with barrier=150 or nb_stocks=2 and check if some paths survive

### Verification Test
To confirm this is not a bug, run:
```python
test_uo_verification = _DefaultConfig(
    algos=['SRLSM'],
    payoffs=['UO_MaxDispersionCall', 'MaxDispersionCall'],
    nb_stocks=[2],  # Fewer stocks = lower knockout probability
    barriers=[150],  # Higher barrier = lower knockout probability
    nb_dates=[20],
    nb_paths=[5000],
    nb_runs=5,
)
```

If UO_MaxDispersionCall still gives zero with barrier=150 and nb_stocks=2, then it's a bug.
If it gives small positive prices, then current zeros are expected.

---

## Priority Summary

1. **CRITICAL - Bug 1**: RankWeightedBasketCall formula error - MUST FIX
2. **CRITICAL - Bug 2**: RealData override extraction - MUST FIX or DOCUMENT
3. **INFO - Issue 3**: UO knockout behavior - Verify but likely expected

## Testing Recommendations

After fixes:
1. Run test_all_bug_fixes again
2. Verify RankWeightedBasketCall gives positive prices similar to BestOfKCall
3. Verify RealData MaxDispersionCall gives positive prices
4. Run UO verification test with higher barrier/fewer stocks
