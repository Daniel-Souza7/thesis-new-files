# User Data Folder

This folder is for user-provided CSV files to use with the `UserDataModel` for block bootstrap simulations.

## Quick Start

1. **Place your CSV file** in this folder (e.g., `my_stocks.csv`)
2. **Configure your experiment** to use `UserData` model:
   ```python
   my_config = _DefaultConfig(
       stock_models=['UserData'],
       user_data_file='my_stocks.csv',  # Your CSV filename
       drift=(None,),  # Use empirical drift from your data
       volatilities=(None,),  # Use empirical volatility from your data
       nb_stocks=3,
       ...
   )
   ```
3. **Run your experiment** as usual

## CSV Format Requirements

### Option 1: Price Data (Recommended)

```csv
date,ticker,price
2020-01-02,AAPL,300.35
2020-01-02,MSFT,160.62
2020-01-02,GOOGL,1368.68
2020-01-03,AAPL,297.43
2020-01-03,MSFT,158.62
2020-01-03,GOOGL,1360.66
...
```

**Requirements:**
- **Columns:** `date`, `ticker`, `price` (names are configurable)
- **Date format:** Any pandas-compatible format (YYYY-MM-DD recommended)
- **All tickers must have the same dates** (no missing values)
- **Sorted by date** (ascending)

### Option 2: Return Data

```csv
date,ticker,return
2020-01-02,AAPL,0.015
2020-01-02,MSFT,-0.012
2020-01-02,GOOGL,0.008
...
```

When using returns, specify `value_type='return'` in config.

## Configuration Options

### Basic Usage

```python
# Use empirical drift and volatility from your data
config = _DefaultConfig(
    stock_models=['UserData'],
    user_data_file='my_stocks.csv',
    drift=(None,),  # Empirical
    volatilities=(None,),  # Empirical
    nb_stocks=3,  # Must match number of tickers in CSV
)
```

### Override Drift/Volatility

```python
# Use your data's correlations but override drift/vol
config = _DefaultConfig(
    stock_models=['UserData'],
    user_data_file='my_stocks.csv',
    drift=(0.05,),  # Force 5% annual drift
    volatilities=(0.2,),  # Force 20% annual volatility
    nb_stocks=3,
)
```

### Custom Column Names

If your CSV has different column names, pass them explicitly:

```python
from optimal_stopping.data.user_data_model import UserDataModel

model = UserDataModel(
    data_file='my_data.csv',
    date_column='Date',  # Your date column name
    ticker_column='Symbol',  # Your ticker column name
    value_column='Close',  # Your price column name
    drift_override=None,  # Use empirical
    volatility_override=None,
    ...
)
```

### Return Data Instead of Prices

```python
from optimal_stopping.data.user_data_model import UserDataModel

model = UserDataModel(
    data_file='returns.csv',
    value_type='return',  # Interpret values as returns, not prices
    value_column='return',
    ...
)
```

## How It Works

The `UserDataModel` uses **stationary block bootstrap** (Politis & Romano, 1994) to generate realistic price paths:

1. **Loads your CSV data** (prices or returns)
2. **Calculates empirical statistics** (drift, volatility, correlations)
3. **Determines optimal block length** from autocorrelation structure (default)
4. **Generates paths** by resampling blocks of historical data
   - Preserves autocorrelation (momentum/mean reversion)
   - Preserves volatility clustering
   - Preserves fat tails and jumps
   - Preserves cross-stock correlations

### Block Length

The block length controls how much temporal structure is preserved:
- **Auto-detected** by default (recommended)
- **Short blocks** (~5-10 days): Less autocorrelation preserved
- **Long blocks** (~30-50 days): More autocorrelation preserved

The optimal length is automatically calculated as the lag where autocorrelation becomes insignificant.

## Example Workflow

### 1. Export Data from Your Source

```python
# Example: Export from pandas DataFrame
import pandas as pd

# Your data: DataFrame with date index and stock columns
df = pd.DataFrame({
    'AAPL': [300.35, 297.43, 299.80, ...],
    'MSFT': [160.62, 158.62, 159.03, ...],
    'GOOGL': [1368.68, 1360.66, 1397.81, ...]
}, index=pd.date_range('2020-01-02', periods=100))

# Convert to required format
long_df = df.reset_index().melt(
    id_vars='index',
    var_name='ticker',
    value_name='price'
).rename(columns={'index': 'date'})

# Save to CSV
long_df.to_csv('optimal_stopping/data/user_data/my_stocks.csv', index=False)
```

### 2. Create Config

```python
# In configs.py
test_user_data = _DefaultConfig(
    algos=['RLSM', 'RFQI'],
    stock_models=['UserData'],
    user_data_file='my_stocks.csv',
    payoffs=['BasketCall', 'MaxCall'],
    drift=(None,),  # Use empirical
    volatilities=(None,),  # Use empirical
    nb_stocks=3,
    nb_paths=[10000],
    nb_dates=[10],
    strikes=[100],
    spots=[100],
)
```

### 3. Run Experiment

```bash
python -m optimal_stopping.run.run_algo optimal_stopping.run.configs:test_user_data
```

## Comparison with RealDataModel

| Feature | UserDataModel | RealDataModel |
|---------|---------------|---------------|
| Data source | User CSV files | yfinance downloads |
| Tickers | Any (your choice) | S&P 500 + 200 extras |
| Date range | Your data | 2010-2024 (configurable) |
| Data format | CSV required | Automatic download |
| Use case | Custom data, backtesting | Quick experiments |
| Bootstrap | ✓ Same (stationary block) | ✓ Same |
| Block length | ✓ Auto-detected | ✓ Auto-detected |

## Troubleshooting

### "Data file not found"

**Solution:** Ensure CSV is in `/home/user/thesis-new-files/optimal_stopping/data/user_data/`

### "CSV missing required columns"

**Solution:** Check your CSV has `date`, `ticker`, `price` columns (case-sensitive). Or specify custom column names.

### "No complete data rows found"

**Solution:** Ensure all tickers have the same dates. Fill or remove missing values:
```python
df = df.dropna()  # Remove rows with NaN
# OR
df = df.fillna(method='ffill')  # Forward-fill missing values
```

### "nb_stocks doesn't match"

**Solution:** Set `nb_stocks` in config to match the number of unique tickers in your CSV.

## Example: Convert Yahoo Finance Data

```python
import yfinance as yf
import pandas as pd

# Download data
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')

# Extract adjusted close prices
prices = data['Adj Close']

# Convert to long format
long_df = prices.reset_index().melt(
    id_vars='Date',
    var_name='ticker',
    value_name='price'
).rename(columns={'Date': 'date'})

# Save
long_df.to_csv('optimal_stopping/data/user_data/yf_data.csv', index=False)
```

## Advanced: Manually Specify Block Length

```python
from optimal_stopping.data.user_data_model import UserDataModel

model = UserDataModel(
    data_file='my_data.csv',
    avg_block_length=20,  # Force 20-day blocks
    ...
)
```

Use this if you know the temporal structure of your data (e.g., weekly rebalancing → 5-day blocks).

## Questions?

See `CLAUDE.md` for general project documentation, or examine:
- `/optimal_stopping/data/user_data_model.py` - Implementation
- `/optimal_stopping/data/real_data.py` - Similar model using yfinance
- `/optimal_stopping/data/stock_model.py` - Base class
