# Payoff System Validation Test Plan

This document describes the comprehensive validation tests for the 408-payoff system.

## Test Configurations Overview

All tests use:
- **d = 10 stocks** (large basket to stress-test multi-asset logic)
- **5000 paths** (sufficient for convergence)
- **10 time steps** (moderate maturity)
- **5 runs** (for statistical stability)

---

## Test 1: `validation_barrier_convergence`

### Purpose
Verify that barrier options converge to vanilla options when barriers are extreme.

### Payoffs Tested
- Vanilla: `BasketCall`, `BasketPut`, `Call`, `Put`
- Up-and-Out (UO): Same payoffs with `UO_` prefix
- Down-and-Out (DO): Same payoffs with `DO_` prefix

### Barrier Levels
- **10000** (extremely high): UO should ≈ vanilla, DO should ≈ 0
- **150** (moderately high): UO < vanilla, DO ≈ vanilla
- **50** (moderately low): UO ≈ vanilla, DO < vanilla

### Expected Results
| Barrier | UO vs Vanilla | DO vs Vanilla |
|---------|---------------|---------------|
| 10000   | UO ≈ Vanilla  | DO ≈ 0        |
| 150     | UO < Vanilla  | DO ≈ Vanilla  |
| 50      | UO ≈ Vanilla  | DO < Vanilla  |

### Validation Criteria
✅ **PASS**: UO(barrier=10000) within 5% of Vanilla
✅ **PASS**: DO(barrier=10000) < 1% of Vanilla
❌ **FAIL**: Any significant deviation from expected ordering

---

## Test 2: `validation_alpha_sensitivity`

### Purpose
Verify quantile options respond correctly to α parameter changes.

### Payoffs Tested
- `QuantileBasketCall`, `QuantileBasketPut`
- `QuantileCall`, `QuantilePut`

### Alpha Levels
- **0.5** (median): Conservative
- **0.75**: Moderate
- **0.95**: Aggressive
- **0.99**: Extreme

### Expected Results
- **Calls**: Price should **INCREASE** with α
  - Higher α → higher quantile → better payoff for calls
- **Puts**: Price should **DECREASE** with α
  - Higher α → higher quantile → worse payoff for puts

### Validation Criteria
✅ **PASS**: QuantileCall(α=0.99) > QuantileCall(α=0.95) > QuantileCall(α=0.75) > QuantileCall(α=0.5)
✅ **PASS**: QuantilePut(α=0.99) < QuantilePut(α=0.95) < QuantilePut(α=0.75) < QuantilePut(α=0.5)
❌ **FAIL**: Non-monotonic relationship

---

## Test 3: `validation_k_sensitivity`

### Purpose
Verify rank-based options respond correctly to k parameter (out of d=10 stocks).

### Payoffs Tested
- `BestOfKCall`, `WorstOfKPut`
- `RankWeightedBasketCall`, `RankWeightedBasketPut`

### K Values
- **k=2**: Best/worst of 2 out of 10
- **k=5**: Best/worst of 5 out of 10
- **k=8**: Best/worst of 8 out of 10

### Expected Results
- **BestOfKCall**: Price should **INCREASE** with k
  - Larger k → more chances to find high performers → higher value
- **WorstOfKPut**: Price should **INCREASE** with k
  - Larger k → worst of many is lower → better for puts
- **RankWeightedBasket**: Similar sensitivity to k (weights change with selection)

### Validation Criteria
✅ **PASS**: BestOfKCall(k=8) > BestOfKCall(k=5) > BestOfKCall(k=2)
✅ **PASS**: WorstOfKPut(k=8) > WorstOfKPut(k=5) > WorstOfKPut(k=2)
✅ **PASS**: RankWeighted options show consistent behavior with k
❌ **FAIL**: Non-monotonic relationship

---

## Test 4: `validation_step_barriers`

### Purpose
Verify step barriers respond correctly to drift parameters.

### Payoffs Tested
- `StepB_BasketCall`, `StepB_BasketPut`, `StepB_Call`, `StepB_Put`
- `DStepB_BasketCall`, `DStepB_Call`

### Drift Configurations
| step_param1 | step_param2 | Drift Direction | Effect on Barrier |
|-------------|-------------|-----------------|-------------------|
| -2          | 0           | Downward        | Barrier drifts DOWN |
| -1          | 1           | Symmetric       | No net drift |
| 0           | 2           | Upward          | Barrier drifts UP |

### Expected Results for Calls (Up-and-Out logic)
- **Upward drift [0,2]**: Barrier rises → easier to hit → **LOWER price**
- **Symmetric [-1,1]**: No drift bias → **MEDIUM price**
- **Downward drift [-2,0]**: Barrier falls → harder to hit → **HIGHER price**

### Validation Criteria
✅ **PASS**: StepB_Call(drift=down) > StepB_Call(drift=sym) > StepB_Call(drift=up)
❌ **FAIL**: Incorrect monotonicity or extreme deviations

---

## Test 5: `validation_large_basket`

### Purpose
Stress-test all payoff types with large basket (d=10) and varying volatility.

### Payoffs Tested
- Simple: `BasketCall`, `GeometricCall`, `MaxCall`, etc.
- Path-dependent: `AsianFixedStrikeCall`, `LookbackMaxCall`
- Quantile: `QuantileBasketCall`
- Rank: `BestOfKCall`, `WorstOfKPut`
- Range: `RangeCall`, `DispersionCall`
- Barriers: `UO_BasketCall`, `DO_BasketPut`, `UODO_BasketCall`

### Parameter Variations
- **Volatility**: [0.2, 0.4] (low vs high)
- **k**: [3, 7] (different rank sizes)
- **Barriers**: [10000, 130] (standard vs barrier)

### Expected Results
- **Higher volatility → higher call prices** (more upside potential)
- **Higher volatility → higher put prices** (more downside potential)
- All payoffs should produce positive prices
- No NaN or infinite values

### Validation Criteria
✅ **PASS**: All prices > 0
✅ **PASS**: Price(vol=0.4) > Price(vol=0.2) for most payoffs
❌ **FAIL**: Negative prices, NaN, or inf values

---

## Test 6: `validation_in_barriers`

### Purpose
Verify "knock-in" barriers converge to vanilla when barrier is always hit.

### Payoffs Tested
- Vanilla: `BasketCall`, `BasketPut`
- Up-and-In (UI): `UI_BasketCall`, `UI_BasketPut`
- Down-and-In (DI): `DI_BasketCall`, `DI_BasketPut`

### Barrier Levels
- **80** (low): UI should ≈ vanilla (always hit), DI should ≈ 0 (never hit)
- **120** (high): DI should ≈ vanilla (always hit), UI should ≈ 0 (never hit)
- **150** (very high): UI should ≈ 0
- **50** (very low): DI should ≈ 0

### Expected Results
| Barrier | UI vs Vanilla | DI vs Vanilla |
|---------|---------------|---------------|
| 80      | UI ≈ Vanilla  | DI ≈ 0        |
| 120     | UI ≈ 0        | DI ≈ Vanilla  |
| 150     | UI ≈ 0        | DI ≈ Vanilla  |
| 50      | UI ≈ Vanilla  | DI ≈ 0        |

### Validation Criteria
✅ **PASS**: UI(barrier=80) ≈ Vanilla
✅ **PASS**: DI(barrier=120) ≈ Vanilla
❌ **FAIL**: Incorrect convergence behavior

---

## Test 7: `validation_payoff_ordering`

### Purpose
Verify fundamental option pricing inequality: **Lookback > Vanilla > Asian**

### Payoffs Tested
- Vanilla: `Call`, `BasketCall`
- Asian: `AsianFixedStrikeCall`
- Lookback: `LookbackFixedCall`
- Floating: `AsianFloatingStrikeCall`, `LookbackFloatCall`

### Expected Ordering (for Calls)
1. **Lookback** (most valuable): Uses maximum over all paths → max(S) ≥ S(T) always
2. **Vanilla** (moderate): Uses terminal value S(T) → full volatility exposure
3. **Asian** (least valuable): Uses average over all paths → reduced effective volatility

**Mathematical Justification**:
- Lookback: Payoff uses max(S) which is always ≥ S(T), so Lookback ≥ Vanilla
- Vanilla vs Asian: Averaging reduces effective volatility. Since option value increases with volatility (vega > 0), Asian < Vanilla
- Therefore: **Lookback > Vanilla > Asian**

### Validation Criteria
✅ **PASS**: LookbackFixedCall > Call > AsianFixedStrikeCall (correct ordering)
✅ **PASS**: LookbackFloatCall > AsianFloatingStrikeCall (floating strike variants)
✅ **PASS**: For BasketCall, verify ordering with Asian variants
❌ **FAIL**: Any violation of the ordering

---

## Running the Tests

To run all validation tests:

```bash
# Run each validation config
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_barrier_convergence
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_alpha_sensitivity
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_k_sensitivity
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_step_barriers
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_large_basket
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_in_barriers
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:validation_payoff_ordering

# Generate Excel reports
python -m optimal_stopping.run.write_excel --configs=validation_barrier_convergence
python -m optimal_stopping.run.write_excel --configs=validation_alpha_sensitivity
python -m optimal_stopping.run.write_excel --configs=validation_k_sensitivity
python -m optimal_stopping.run.write_excel --configs=validation_step_barriers
python -m optimal_stopping.run.write_excel --configs=validation_large_basket
python -m optimal_stopping.run.write_excel --configs=validation_in_barriers
python -m optimal_stopping.run.write_excel --configs=validation_payoff_ordering
```

---

## Summary of Expected Patterns

### ✅ HEALTHY PATTERNS
- Monotonic relationships (alpha, k parameters)
- Convergence to vanilla at extreme barriers
- Correct ordering: Lookback > Asian > Vanilla
- Positive prices for all payoffs
- Higher volatility → higher prices (generally)

### ❌ RED FLAGS
- Non-monotonic parameter sensitivity
- Negative prices
- NaN or infinite values
- Barrier options that don't converge
- Violation of no-arbitrage bounds
- Inconsistent behavior across RFQI/RLSM/SRFQI/SRLSM

---

## Notes

- All tests use **10 stocks** to ensure multi-asset logic is properly tested
- **5000 paths** and **5 runs** provide statistical reliability
- Tests focus on **relative comparisons** (ordering, monotonicity) rather than absolute values
- Some variance is expected due to Monte Carlo noise, but patterns should be clear
