# Exercise Time Discrepancy Investigation

## Problem Statement

When running `video_testing2` config:
- **run_algo → write_excel**: mean exercise time = **1.0**, price = 11.31
- **create_video**: mean exercise time = **0.8050**, price = 11.69

The video shows ~50% of paths exercising before maturity, confirming the 0.8050 result.

## Why This Matters

For American CALL options with **no dividends**, the optimal strategy is:
- **NEVER exercise early**
- **Expected exercise time should be 1.0** (always at maturity)

If exercise time < 1.0, the learned policy is **suboptimal**!

## Hypothesis: Different Evaluation Methods

### Method 1: `get_exercise_time()` (used by run_algo)
```python
def get_exercise_time(self):
    """Return average exercise time from EVALUATION SET during training."""
    normalized_times = self._exercise_dates[self.split:] / nb_dates
    return float(np.mean(normalized_times))
```

**Process:**
1. During backward induction training, `_exercise_dates` is updated when paths exercise early
2. Uses the **evaluation set** (50% of training paths)
3. These are the SAME paths used during training
4. Reports when backward induction decided to exercise

### Method 2: `predict()` (used by create_video)
```python
def predict(self, stock_paths, var_paths=None):
    """Apply learned policy to NEW paths via forward simulation."""
    # For each time step:
    #   - Compute continuation_value using learned coefficients
    #   - Exercise if immediate_payoff > continuation_value
```

**Process:**
1. Takes **completely NEW paths** (not seen during training)
2. Applies learned policy in **forward simulation**
3. Uses stored regression coefficients from training
4. Reports when forward simulation decides to exercise

## Key Differences

| Aspect | Training (`get_exercise_time`) | Prediction (`predict`) |
|--------|-------------------------------|----------------------|
| Paths | Same paths used in backward induction | Completely new paths |
| Direction | Backward induction | Forward simulation |
| Evaluation | During training | After training |
| Sample size | 50% of training paths (eval set) | All visualization paths |

## Potential Causes

### 1. **Generalization Failure**
The learned policy doesn't generalize well to new paths:
- **Training**: Coefficients are fitted to specific path realizations
- **New paths**: Different distribution of stock prices
- **Result**: Continuation values are systematically underestimated for new paths
- **Consequence**: Algorithm exercises early when it shouldn't

### 2. **Regression Noise/Errors**
The least squares regression is imperfect:
- For OTM CALL options, continuation values should be small but positive
- Regression noise might make them **negative** or **very small**
- Comparison: `immediate_payoff > continuation_value` triggers early exercise
- This error compounds across time steps in forward simulation

### 3. **Forward vs Backward Dynamics**
Backward induction has different dynamics than forward simulation:
- **Backward**: Future exercise decisions inform current regression
- **Forward**: Using fixed coefficients without feedback
- The learned policy might only be optimal in the backward direction

### 4. **train_ITM_only=True Issue**
When `train_ITM_only=True`:
- Only in-the-money paths are used for training regression
- **OTM paths get continuation_value = 0** (our recent fix)
- For CALL options, many paths might be OTM at early times
- This could cause the learned policy to be incomplete

## Diagnostic Steps

### Step 1: Run debug_exercise_times.py
```bash
python debug_exercise_times.py
```

This script will:
1. Train LSM on 40,000 paths (like video_testing2)
2. Report exercise time from training (backward induction)
3. Apply `predict()` to the SAME paths used in training
4. Apply `predict()` to NEW paths (like create_video does)
5. Compare all three exercise times

**Expected outcomes:**
- If training ≈ 1.0, predict(same) ≈ 1.0, predict(new) ≈ 0.8:
  → **Generalization failure** (doesn't work on new paths)

- If training ≈ 1.0, predict(same) ≈ 0.8, predict(new) ≈ 0.8:
  → **Bug in predict() method** (doesn't match training decisions)

- If training ≈ 0.8, predict(same) ≈ 0.8, predict(new) ≈ 0.8:
  → **Bug in backward induction** (learning suboptimal policy)

### Step 2: Test with train_ITM_only=False
Modify video_testing2:
```python
train_ITM_only=(False,)  # Train on ALL paths, not just ITM
```

If exercise time improves → ITM-only training is too restrictive

### Step 3: Increase hidden_size
Modify video_testing2:
```python
hidden_size=(100,)  # More basis functions for better approximation
```

If exercise time improves → Regression needs more flexibility

### Step 4: Test with more paths
```python
nb_paths=(100000,)  # More paths for better coefficient estimation
```

If exercise time improves → Sample size too small

## Theoretical Considerations

### American CALL with no dividends

**Black-Scholes formula:**
```
dS = μ S dt + σ S dW
```

**Optimal exercise:** NEVER exercise early because:
1. Option has time value (can always be worth more if you wait)
2. No dividends means no cost to holding
3. Strike price is constant (not discounted)
4. European CALL = American CALL

**What LSM should learn:**
- Continuation value ≈ E[max(S_T - K, 0) | S_t] × exp(-r(T-t))
- For any reasonable stock price, continuation > immediate payoff
- Exercise decision: always WAIT

**If LSM exercises early:**
- Learned continuation values are WRONG (too low)
- This is a fundamental failure of the approximation

## Next Steps

1. **Run diagnostic script** to identify which phase is failing
2. **Test with train_ITM_only=False** to see if that's the issue
3. **Inspect learned coefficients** to see if they're sensible
4. **Compare with debug_lsm.py** (which might handle this correctly)
5. **Consider using different basis functions** (Laguerre polynomials might work better)

## Questions to Answer

1. Does `debug_lsm.py` (our reference implementation) show exercise time ≈ 1.0 for CALL options?
2. Does RLSM have the same issue as LSM?
3. Do FQI/RFQI (which use ex_dates.copy()) behave correctly?
4. Is this issue specific to CALL options or does it affect PUT options too?

## Suspected Root Cause

My current hypothesis is **#4: train_ITM_only issue**.

For CALL options with S=K=100:
- At early times (t=1, 2, 3), many paths have S_t ≈ 100 (around strike)
- These paths are **barely ITM or OTM**
- With `train_ITM_only=True`, we might be filtering out too many paths
- The regression is then trained only on paths where S_t is significantly above strike
- When we predict on new paths with S_t ≈ 100, the continuation value extrapolates poorly
- This causes spurious early exercises

**Test:** Run with `train_ITM_only=False` and see if the problem goes away.
