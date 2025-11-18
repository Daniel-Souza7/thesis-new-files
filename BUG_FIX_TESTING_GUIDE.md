# Bug Fix Testing Guide

This guide provides instructions for testing all 9 bug fixes from the comprehensive debugging session.

## Quick Reference: Test Configs

| Config Name | Bugs Tested | Runtime | Primary Focus |
|-------------|-------------|---------|---------------|
| `test_bug1_barrier_path_dependent` | Bug 1 | ~5 min | Barrier-wrapped path-dependent payoffs (220 payoffs) |
| `test_bug2_default_barriers` | Bug 2, 5 | ~3 min | Default barriers compatibility |
| `test_bug3_fractional_bs` | Bug 3 | ~3 min | FractionalBlackScholes methods |
| `test_bug4_real_data_init` | Bug 4 | ~4 min | RealDataModel initialization |
| `test_bug6_create_video` | Bug 6 | ~1 min | create_video parameters |
| `test_bug7_k_validation` | Bug 7 | ~3 min | K parameter validation |
| `test_bug8_weights_validation` | Bug 8 | ~3 min | Weights parameter validation |
| `test_bug9_step_barriers` | Bug 9 | ~4 min | Step barrier formula |
| `test_user_data_model` | New Feature | ~3 min | UserData model with CSV |
| `test_all_bug_fixes` | All | ~15 min | Comprehensive integration test |

**Total runtime for all tests**: ~45 minutes

---

## Test 1: Barrier-Wrapped Path-Dependent Payoffs (Bug 1)

**Bug Description**: 220 barrier-wrapped path-dependent payoffs crashed with `AxisError: axis 2 is out of bounds for array of dimension 2` because barrier wrapper passed 2D arrays to payoffs expecting 3D.

**Fix Location**: `optimal_stopping/payoffs/barrier_wrapper.py:69`

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug1_barrier_path_dependent
```

**What This Tests**:
- Multi-asset path-dependent base payoffs (MaxDispersion, Dispersion, Asian, Lookback)
- Barrier-wrapped versions (UO_, DO_, UI_, DI_, UODO_, UIDI_, StepB_, DStepB_)
- Both 5 and 10 stocks
- Algorithms: SRLSM, SRFQI

**Success Criteria**:
✅ All runs complete without `AxisError`
✅ CSV output contains prices for all barrier-wrapped payoffs
✅ Prices are non-negative and reasonable (0 < price < strike for most)
✅ UO/DO prices < base payoff prices (barriers reduce value)

**Expected Output Location**:
`output/metrics_draft/metrics_<timestamp>.csv`

**How to Verify**:
```bash
# Check that file was created and has results
tail -20 output/metrics_draft/metrics_*.csv | grep "UO_MaxDispersionCall\|DO_MaxDispersionPut"

# Check for errors
grep -i "error\|axis" output/metrics_draft/metrics_*.csv
```

**Common Issues**:
- If you see `AxisError`: Bug 1 fix failed, check barrier_wrapper.py:69
- If prices are zero: Check barrier levels (barriers=[120] should be reasonable)

---

## Test 2: Default Barriers (Bugs 2 & 5)

**Bug Description**: Default `barriers=(1,)` filtered out standard payoffs (barrier=100000) in write_excel, causing `AssertionError: No data after filtering`.

**Fix Location**: `optimal_stopping/run/configs.py:64`

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug2_default_barriers
```

**What This Tests**:
- Standard payoffs without explicit barriers parameter
- Default barriers value (should be 100000)
- write_excel compatibility

**Success Criteria**:
✅ All runs complete without errors
✅ CSV shows `barrier=100000` for all payoffs
✅ write_excel can process the results without filtering errors

**How to Verify**:
```bash
# Check barrier values in CSV
grep "BasketCall\|BasketPut" output/metrics_draft/metrics_*.csv | head -5

# Verify all have barrier=100000
awk -F',' '/BasketCall/{print $0}' output/metrics_draft/metrics_*.csv | cut -d',' -f20 | sort -u

# Try generating Excel (should not crash)
python -m optimal_stopping.run.write_excel --configs=test_bug2_default_barriers
```

**Expected Behavior**:
- All barrier columns show `100000`
- Excel generation succeeds with table and charts
- No "No data after filtering" errors

---

## Test 3: FractionalBlackScholes Methods (Bug 3)

**Bug Description**: FractionalBlackScholes raised `NotImplementedError: Subclasses must implement diffusion_fct()` because `drift_fct()` and `diffusion_fct()` were never implemented.

**Fix Location**: `optimal_stopping/data/stock_model.py:179-187`

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug3_fractional_bs
```

**What This Tests**:
- FractionalBlackScholes path generation
- Both standard (RLSM/RFQI) and path-dependent (SRLSM/SRFQI) algorithms
- Multiple Hurst parameters (0.6, 0.7, 0.8)
- Various payoff types

**Success Criteria**:
✅ No `NotImplementedError` exceptions
✅ All runs complete successfully
✅ Prices vary with Hurst parameter (H=0.8 > H=0.6 for calls due to long memory)
✅ Output CSV shows model="FractionalBlackScholes"

**How to Verify**:
```bash
# Check for NotImplementedError
grep -i "notimplementederror\|must implement" output/metrics_draft/metrics_*.csv

# Verify model name
grep "FractionalBlackScholes" output/metrics_draft/metrics_*.csv | head -3

# Check that different Hurst values produced different prices
awk -F',' '/BasketCall.*FractionalBlackScholes/{print $8, $9}' output/metrics_draft/metrics_*.csv | sort -u
```

**Expected Behavior**:
- All Hurst values (0.6, 0.7, 0.8) complete successfully
- Prices increase slightly with Hurst (more persistence = more upside)
- No errors about missing methods

---

## Test 4: RealDataModel Initialization (Bug 4)

**Bug Description**: RealDataModel had two issues:
1. Missing `name='RealData'` parameter causing TypeError
2. Failed tuple handling: `(None,) - dividend` raised TypeError

**Fix Location**: `optimal_stopping/data/real_data.py:115-128`

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug4_real_data_init
```

**What This Tests**:
- RealDataModel initialization with `drift=(None,)` and `volatilities=(None,)`
- Empirical parameter extraction from historical data
- yfinance data download (first run may take 1-2 min)

**Success Criteria**:
✅ Model initializes without TypeError
✅ Prints "📊 Loading real market data..." and "✅ Loaded X stocks..."
✅ Shows empirical drift and volatility percentages
✅ All runs complete successfully

**How to Verify**:
```bash
# Look for initialization messages in stdout (not CSV)
# You'll see these during the run:
# "📊 Loading real market data from yfinance..."
# "✅ Loaded 5 stocks: AAPL, MSFT, GOOGL, AMZN, NVDA"
# "   1258 days of returns"
# "   Empirical return: 25.67%, volatility: 31.42%"

# Check CSV for model name
grep "RealData" output/metrics_draft/metrics_*.csv | head -5

# Verify no TypeError
grep -i "typeerror\|missing.*argument" output/metrics_draft/metrics_*.csv
```

**Expected Behavior**:
- First run downloads data from yfinance (~1-2 min)
- Subsequent runs use cached data (fast)
- Prints empirical statistics (drift ~10-30%, vol ~20-40% for tech stocks)
- All algorithms complete successfully

**Note**: Requires internet connection for first run. Data is cached locally.

---

## Test 5: create_video Parameters (Bug 6)

**Bug Description**: `create_video.py` was hardcoding `dividend=0` and not extracting/passing `risk_free_rate` to BlackScholes models.

**Fix Location**: `optimal_stopping/run/create_video.py:118-122, 162, 201-207`

**Test Commands**:
```bash
# First run the pricing algorithm
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug6_create_video

# Then create video visualization
python -m optimal_stopping.run.create_video optimal_stopping.run.configs:test_bug6_create_video
```

**What This Tests**:
- Parameter extraction from config
- Passing `risk_free_rate` and `dividend` to both model instantiations
- Video generation with correct model parameters

**Success Criteria**:
✅ Algorithm runs successfully
✅ Video generation completes without AttributeError
✅ Video shows correct parameters in plots (drift=0.08, r=0.04, div=0.02)

**How to Verify**:
```bash
# Check that pricing completed
ls output/metrics_draft/metrics_*.csv | tail -1

# Run create_video and check for errors
python -m optimal_stopping.run.create_video optimal_stopping.run.configs:test_bug6_create_video 2>&1 | grep -i "error\|missing"

# If successful, video file should be created
ls results/video_*.mp4 | tail -1
```

**Expected Output**:
- Video file in `results/` directory
- No "missing argument" or "unexpected keyword" errors
- Video shows evolving price paths with exercise decisions

**Note**: Video generation requires `matplotlib` and `ffmpeg`. If ffmpeg is missing, you'll get a clear error message (this is expected on some systems).

---

## Test 6: K Parameter Validation (Bug 7)

**Bug Description**: Rank-based payoffs (BestOfK, WorstOfK, RankWeighted) didn't validate k parameter, allowing invalid values like k < 1, k > nb_stocks, or non-integer k.

**Fix Location**: `optimal_stopping/payoffs/basket_rank.py` (all 4 eval methods)

**Test Command**:
```bash
# Test valid k values (should succeed)
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug7_k_validation
```

**What This Tests**:
- Valid k values: k=2, 5, 8 (all < nb_stocks=10)
- All rank-based payoffs: BestOfKCall, WorstOfKPut, RankWeightedBasketCall/Put

**Success Criteria**:
✅ All runs complete successfully
✅ Prices increase with k for BestOfK (more options = higher value)
✅ Prices decrease with k for WorstOfK (more stocks to avoid = lower value)

**How to Verify**:
```bash
# Check that all k values worked
awk -F',' '/BestOfKCall/{print $0}' output/metrics_draft/metrics_*.csv | grep -o "k=[0-9]*" | sort -u

# Verify prices increase with k for BestOfK
awk -F',' '/BestOfKCall/{print $0}' output/metrics_draft/metrics_*.csv | awk -F',' '{print $10, $11}' | sort -n

# Check for validation errors (should be none)
grep -i "k must be\|valueerror" output/metrics_draft/metrics_*.csv
```

**Manual Validation Test** (optional):
```bash
# Test invalid k (should raise ValueError)
python -c "
from optimal_stopping.payoffs import get_payoff_class
import numpy as np
BestOfKCall = get_payoff_class('BestOfKCall')
payoff = BestOfKCall(strike=100, k=15)  # k > nb_stocks
X = np.random.randn(100, 10) * 100  # 10 stocks
try:
    payoff.eval(X)
    print('❌ FAILED: Should have raised ValueError for k > nb_stocks')
except ValueError as e:
    print(f'✅ PASSED: Caught error: {e}')
"
```

**Expected Behavior**:
- k=2, 5, 8 all succeed
- Manual test with k=15 raises: "k (15) cannot exceed number of stocks (10)"

---

## Test 7: Weights Parameter Validation (Bug 8)

**Bug Description**: RankWeightedBasket payoffs didn't validate weights parameter, allowing wrong length, negative weights, or wrong types.

**Fix Location**: `optimal_stopping/payoffs/basket_rank.py` (RankWeighted classes)

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug8_weights_validation
```

**What This Tests**:
- Auto-generated weights (1/k for each)
- k=3 and k=5
- RankWeightedBasketCall and RankWeightedBasketPut

**Success Criteria**:
✅ All runs complete successfully
✅ Weights are auto-generated correctly (should see equal weights)
✅ Prices are reasonable (between simple basket and BestOfK)

**How to Verify**:
```bash
# Check that runs completed
grep "RankWeightedBasketCall" output/metrics_draft/metrics_*.csv | wc -l

# Verify no validation errors
grep -i "length of weights\|weights must be" output/metrics_draft/metrics_*.csv
```

**Manual Validation Test** (optional):
```bash
# Test invalid weights (should raise ValueError)
python -c "
from optimal_stopping.payoffs import get_payoff_class
import numpy as np
RankWeighted = get_payoff_class('RankWeightedBasketCall')
payoff = RankWeighted(strike=100, k=3, weights=[0.5, 0.3, 0.2, 0.1])  # len=4, k=3
X = np.random.randn(100, 10) * 100
try:
    payoff.eval(X)
    print('❌ FAILED: Should have raised ValueError for len(weights) != k')
except ValueError as e:
    print(f'✅ PASSED: Caught error: {e}')
"
```

**Expected Behavior**:
- Auto-generated weights work correctly
- Manual test with wrong length raises: "Length of weights (4) must equal k (3)"

---

## Test 8: Step Barrier Formula (Bug 9)

**Bug Description**: Step barriers used incorrect time discretization: `exp(r * T * t / nb_dates)` instead of `exp(r * T * t / (nb_dates-1))`. This caused barriers to not reach target value at maturity.

**Fix Location**: `optimal_stopping/payoffs/barrier_wrapper.py:170, 211, 226`

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug9_step_barriers
```

**What This Tests**:
- StepB and DStepB barrier payoffs
- Different discretizations: 10, 20, 50 time steps
- Risk-free rate growth (r=0.04)
- Multiple base payoffs

**Success Criteria**:
✅ All runs complete successfully
✅ Prices converge as nb_dates increases (discretization error decreases)
✅ Barriers reach correct value at maturity: B(T) = B(0) * exp(r * T)
✅ No off-by-one errors in barrier evaluation

**How to Verify**:
```bash
# Check that all nb_dates values worked
grep "StepB_BasketCall" output/metrics_draft/metrics_*.csv | grep -o "nb_dates=[0-9]*" | sort -u

# Compare prices across discretizations (should be similar)
awk -F',' '/StepB_BasketCall/{print $6, $11}' output/metrics_draft/metrics_*.csv | sort -n

# Check for formula errors
grep -i "index.*bound\|out of bound" output/metrics_draft/metrics_*.csv
```

**Mathematical Verification**:
For r=0.04, T=1.0, B(0)=110:
- Expected B(T) = 110 * exp(0.04 * 1.0) = 110 * 1.0408 = 114.49
- With bug: B(T) ≈ 114.05 (at nb_dates=10) - slightly low
- After fix: B(T) = 114.49 (exact) - correct

**Expected Behavior**:
- Prices decrease slightly as nb_dates increases (finer discretization)
- All discretizations complete successfully
- No indexing errors

---

## Test 9: UserData Model (New Feature)

**Bug Description**: Not a bug - this is the new UserData model feature for loading user-provided CSV files with stationary block bootstrap.

**Related Files**:
- `optimal_stopping/data/user_data_model.py` (new)
- `optimal_stopping/data/user_data/README.md` (new)
- `optimal_stopping/data/user_data/example_data.csv` (new)

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_user_data_model
```

**What This Tests**:
- Loading CSV data from user_data/ folder
- Stationary block bootstrap path generation
- Automatic block length detection
- Empirical drift/volatility extraction
- Standard, path-dependent, and barrier payoffs

**Success Criteria**:
✅ Prints "📊 Loading user data from example_data.csv..."
✅ Shows "✅ Loaded 3 stocks: AAPL, GOOGL, MSFT"
✅ Shows empirical statistics and block length
✅ All algorithms complete successfully
✅ Prices are reasonable given the small sample data

**How to Verify**:
```bash
# Check CSV exists
ls optimal_stopping/data/user_data/example_data.csv

# Look for UserData initialization messages during run
# You should see:
# "📊 Loading user data from example_data.csv..."
# "✅ Loaded 3 stocks: AAPL, GOOGL, MSFT"
# "   4 days of returns"
# "   Empirical return: X.XX%, volatility: X.XX%"
# "   Block length: X days"

# Check results
grep "UserData" output/metrics_draft/metrics_*.csv | head -5
```

**Expected Behavior**:
- Loads 3 stocks from CSV
- Shows 4 days of data (5 rows - 1 for returns)
- Block length should be small (5 days, since data is tiny)
- Generates paths using block bootstrap
- All payoffs work with UserData model

**Note**: The example data is tiny (5 days) and is only for testing. Real usage requires at least 100+ days of data for meaningful block bootstrap.

---

## Test 10: Comprehensive Integration Test

**Description**: Tests all bug fixes together in one config with multiple models, payoffs, and parameters.

**Test Command**:
```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_all_bug_fixes
```

**What This Tests**:
- All stock models: BlackScholes, FractionalBlackScholes, RealData
- All payoff categories: standard, rank-based, path-dependent, barrier-wrapped, step barriers
- All bug fixes working together
- Both standard and path-dependent algorithms

**Runtime**: ~15 minutes (longest test)

**Success Criteria**:
✅ All model × payoff × parameter combinations complete
✅ No errors from any of the 9 bugs
✅ Results are internally consistent (barriers reduce prices, etc.)
✅ Can generate Excel report without errors

**How to Verify**:
```bash
# Count total runs (should be large number)
wc -l output/metrics_draft/metrics_*.csv

# Check that all models ran
awk -F',' '{print $2}' output/metrics_draft/metrics_*.csv | sort -u

# Check that all payoff types ran
awk -F',' '{print $3}' output/metrics_draft/metrics_*.csv | sort -u

# Generate Excel report to verify write_excel works
python -m optimal_stopping.run.write_excel --configs=test_all_bug_fixes
```

**Expected Output**:
- Hundreds of result rows (many combinations)
- All 3 models present: BlackScholes, FractionalBlackScholes, RealData
- All payoff types present: BasketCall, BestOfKCall, MaxDispersionCall, UO_MaxDispersionCall, StepB_BasketCall
- Excel file generated successfully

---

## Quick Sanity Check (5 min)

If you want to quickly verify that everything is working, run this minimal test:

```bash
# Quick test of critical fixes
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_bug1_barrier_path_dependent --nb_runs=1 --nb_paths=1000

# Check output
tail -20 output/metrics_draft/metrics_*.csv
```

Should complete in ~2 minutes and show prices for all barrier-wrapped payoffs.

---

## Troubleshooting

### Common Error Messages and Solutions

1. **"Data file not found: example_data.csv"**
   - **Solution**: Check that `optimal_stopping/data/user_data/example_data.csv` exists
   - If missing, it should have been created during bug fixing session

2. **"AxisError: axis 2 is out of bounds for array of dimension 2"**
   - **Solution**: Bug 1 fix failed - check `barrier_wrapper.py:69` for `is_path_dependent` check

3. **"NotImplementedError: Subclasses must implement diffusion_fct()"**
   - **Solution**: Bug 3 fix failed - check `stock_model.py` for FractionalBlackScholes methods

4. **"TypeError: Model.__init__() missing required positional argument: 'name'"**
   - **Solution**: Bug 4 fix failed - check `real_data.py` for `kwargs['name'] = 'RealData'`

5. **"ValueError: k (11) cannot exceed number of stocks (10)"**
   - **Solution**: This is expected! Bug 7 fix is working correctly (validation active)

6. **"AssertionError: No data after filtering"**
   - **Solution**: Bug 2 fix failed - check `configs.py:64` for `barriers=(100000,)`

### Performance Notes

- **RealDataModel**: First run downloads data from yfinance (~1-2 min), subsequent runs are fast (cached)
- **FractionalBlackScholes**: Slower than BlackScholes due to fBm generation
- **Large nb_paths**: If tests are too slow, reduce `nb_paths` in config (e.g., 3000 → 1000)
- **Parallel execution**: Tests can be run in parallel if you have multiple CPU cores

### Verification Checklist

After running all tests, verify:

- [ ] All test configs completed without errors
- [ ] CSV files created in `output/metrics_draft/`
- [ ] All prices are non-negative
- [ ] Barrier prices < base payoff prices
- [ ] k validation prevents k > nb_stocks
- [ ] Step barriers work with all discretizations
- [ ] Excel generation succeeds: `python -m optimal_stopping.run.write_excel --configs=test_all_bug_fixes`

---

## Next Steps

After confirming all tests pass:

1. **Run validation tests** from `VALIDATION_TEST_PLAN.md`
2. **Generate Excel reports** for analysis
3. **Create visualizations** with write_figures
4. **Test create_video** on your configs
5. **Try UserData model** with your own CSV files

For production experiments, use larger values:
- `nb_paths=[10000]` or higher
- `nb_runs=5` or higher
- `nb_dates=[50, 100]` or higher
- More payoff combinations

---

## Contact

If you encounter issues not covered here:
1. Check agent debugging reports (see `AGENT_REPORTS_REVIEW_GUIDE.md`)
2. Review `CLAUDE.md` for general project documentation
3. Check individual payoff files in `optimal_stopping/payoffs/`
4. Review bug fix commits in git history
