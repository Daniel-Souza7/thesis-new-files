# Factor Influence Testing Scripts

These scripts empirically test how different `factors` parameters influence the accuracy of SRLSM and SRFQI algorithms on path-dependent options.

## Scripts Overview

### 1. `test_factors_quick.py` ⚡ (Recommended for initial testing)
**Quick test with reduced parameter grid**

- **Runtime:** ~5-10 minutes
- **Grid:** 3 activation slopes × 3 input scales = 9 combinations per algorithm
- **Problems:**
  - Simple: Down-and-out barrier put
  - Complex: Lookback put with Heston volatility
- **Paths:** 5,000
- **Dates:** 30

**Usage:**
```bash
cd /home/user/thesis-new-files
python test_factors_quick.py
```

**Output:**
- Console output with real-time results
- `factor_quick_test_results.csv` - Full results table

---

### 2. `test_factors_influence.py` 🔬 (Comprehensive testing)
**Full grid search with extensive analysis**

- **Runtime:** ~30-60 minutes
- **Grid:** 6 activation slopes × 5 input scales = 30 combinations per algorithm
- **Problems:**
  - Simple: Down-and-out barrier put
  - Complex: Lookback put with Heston volatility
- **Paths:** 10,000
- **Dates:** 50
- **Includes:** Heatmap visualizations (requires matplotlib/seaborn)

**Usage:**
```bash
cd /home/user/thesis-new-files
python test_factors_influence.py
```

**Output:**
- Console output with detailed statistics
- `factor_influence_results.csv` - Full results table
- `factor_influence_heatmaps.png` - Visual heatmaps (if matplotlib installed)

---

## Understanding the Results

### Key Metrics

1. **Lower Bound**: Option price estimate (suboptimal policy)
2. **Upper Bound**: Dual upper bound (martingale method)
3. **Gap**: Upper - Lower (absolute gap)
4. **Relative Gap %**: (Gap / Lower) × 100
   - **Lower is better** (tighter bounds = better approximation)
   - Typical: 5-15% for good configurations
   - Poor: >20%

### Factor Parameters

**`factors = (activation_slope, input_scale)`**

- **activation_slope**: Controls LeakyReLU slope (= activation_slope / 2)
  - Higher → more linear features
  - Lower → more nonlinear features
  - Test range: 0.6 to 1.6

- **input_scale**: Multiplies input before passing to network
  - Higher → amplifies input variations
  - Lower → compresses input variations
  - Test range: 0.5 to 1.5

---

## Interpreting Results

### Example Output:
```
SRLSM on Simple Problem:
  Best factors: (activation=1.2, input_scale=1.0)
  Relative Gap: 8.34%

SRFQI on Complex Problem:
  Best factors: (activation=0.8, input_scale=1.3)
  Relative Gap: 12.67%
```

### What to Look For:

1. **Best Configurations**: Which factors minimize the relative gap?

2. **Sensitivity**: How much does gap vary with factors?
   - Low variance → factors not critical
   - High variance → careful tuning needed

3. **Problem-Specific Patterns**:
   - Simple problems: Often work well with default (1.0, 1.0)
   - Complex problems: May benefit from tuning

4. **Algorithm Differences**:
   - SRLSM: Typically less sensitive to factors
   - SRFQI: May show more variation

---

## Expected Findings

Based on reservoir computing theory:

### Simple Problems (Barrier Put)
- **Expected best:** `factors ≈ (1.0, 1.0)` to `(1.2, 1.0)`
- **Gap range:** 5-12%
- **Sensitivity:** Low (±2%)

### Complex Problems (Lookback + Heston)
- **Expected best:** `factors ≈ (0.8, 1.0)` to `(1.0, 1.3)`
- **Gap range:** 10-18%
- **Sensitivity:** Moderate (±5%)

### General Patterns:
- **High activation slope** (1.2-1.4) → Better for smooth problems
- **Moderate activation slope** (0.8-1.0) → Better for complex dynamics
- **Input scaling** usually stays near 1.0 for normalized problems

---

## Customization

### Modify Test Grid

Edit the factor ranges in either script:

```python
# In test_factors_quick.py or test_factors_influence.py
activation_slopes = [0.6, 0.8, 1.0, 1.2, 1.4]  # Your values
input_scales = [0.5, 0.7, 1.0, 1.3, 1.5]       # Your values
```

### Add New Problems

Add your own problems to the `problems` list:

```python
# Example: Add up-and-out call
model_new = black_scholes.BlackScholesModel(...)
payoff_new = barrier_options.UpAndOutCall(strike=100.0, barrier=120.0)
problems.append((model_new, payoff_new, "UpAndOut_Call"))
```

### Change Algorithm Parameters

Modify `nb_epochs`, `hidden_size`, `nb_paths`, etc. in the script:

```python
algo = srlsm.SRLSM(
    model=model,
    payoff=payoff,
    hidden_size=50,      # Increase for more features
    nb_epochs=30,        # More epochs for RFQI/SRFQI
    factors=factors,
    train_ITM_only=True
)
```

---

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the thesis-new-files directory
cd /home/user/thesis-new-files

# Check Python path
python -c "import sys; print(sys.path)"
```

### Memory Issues
Reduce `nb_paths` or `nb_dates` in the problem setup:
```python
model = black_scholes.BlackScholesModel(
    nb_paths=2000,  # Reduce from 10000
    nb_dates=20,    # Reduce from 50
    ...
)
```

### Slow Runtime
Use the quick test version or reduce the grid:
```python
activation_slopes = [0.8, 1.0, 1.2]  # Fewer values
input_scales = [1.0]                  # Test only default
```

---

## Advanced Usage

### Test Extended Factors (8 parameters)

Modify the script to test weight initialization:

```python
factors = (
    1.0,    # activation slope
    1.0,    # input scale
    0,      # unused
    0.0,    # weight mean
    0.5,    # weight std (vary this!)
    0.0,    # bias mean
    0.5,    # bias std
    0       # distribution (0=normal)
)
```

### Statistical Analysis

Load results in Python for further analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('factor_influence_results.csv')

# Plot gap vs activation slope
df_srlsm = df[df['Algorithm'] == 'SRLSM']
plt.scatter(df_srlsm['Activation_Slope'], df_srlsm['Relative_Gap_%'])
plt.xlabel('Activation Slope')
plt.ylabel('Relative Gap (%)')
plt.show()
```

---

## References

**Reservoir Computing:**
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.

**Randomized Networks for Options:**
- Herrera, C., et al. (2021). Optimal stopping via randomized neural networks.

---

## Quick Start

```bash
# Run quick test (5-10 minutes)
python test_factors_quick.py

# View results
cat factor_quick_test_results.csv

# Or open in spreadsheet
libreoffice factor_quick_test_results.csv
```
