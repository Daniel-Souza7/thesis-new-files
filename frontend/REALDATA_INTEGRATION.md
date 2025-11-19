# RealData Model Integration Guide

## Overview

The Python API has been enhanced to support the RealDataModel with stationary block bootstrap. This allows pricing options using real market data while preserving autocorrelation, volatility clustering, and fat tails from historical returns.

## Architecture

### Backend (Python)

#### 1. Pricing Engine (`/frontend/api/pricing_engine.py`)

**Purpose**: Main pricing engine that handles all models including RealData.

**Key Features**:
- Supports all stock models: BlackScholes, Heston, FractionalBlackScholes, RoughHeston, RealData
- Handles RealData-specific parameters: tickers, date ranges, drift/volatility overrides
- Caches RealData models to avoid re-downloading data
- Returns comprehensive pricing results including:
  - Option price
  - Computation time
  - Exercise time (for American options)
  - Sample paths for visualization
  - Model information (empirical drift/vol for RealData)

**Commands**:
```bash
# Price an option
python pricing_engine.py price '{"model_type": "RealData", "tickers": ["AAPL"], ...}'

# Get stock information
python pricing_engine.py stock_info '{"tickers": ["AAPL", "MSFT"]}'

# Get available tickers
python pricing_engine.py available_tickers
```

**Example Usage**:
```python
from pricing_engine import PricingEngine

engine = PricingEngine()

# Price a call option using RealData
result = engine.price_option({
    'model_type': 'RealData',
    'payoff_type': 'Call',
    'algorithm': 'RLSM',
    'tickers': ['AAPL', 'MSFT'],
    'start_date': '2010-01-01',
    'end_date': '2024-01-01',
    'drift_override': None,  # Use empirical drift
    'volatility_override': None,  # Use empirical volatility
    'spot': 100,
    'strike': 100,
    'rate': 0.03,
    'maturity': 1.0,
    'nb_stocks': 2,
    'nb_dates': 50,
    'nb_paths': 10000,
})
```

#### 2. Stock Data Manager (`/frontend/api/stock_data.py`)

**Purpose**: Utilities for managing stock ticker data.

**Key Features**:
- Pre-loaded list of 50+ common tickers (S&P 500)
- Validates ticker symbols
- Fetches empirical statistics (drift, volatility, correlation)
- Caches downloaded data

**Commands**:
```bash
# Get ticker information
python stock_data.py info '{"tickers": ["AAPL", "MSFT"]}'

# Validate tickers
python stock_data.py validate '{"tickers": ["AAPL", "INVALID"]}'

# Get pre-loaded tickers
python stock_data.py preloaded
```

### Frontend (Next.js API Routes)

#### 1. Stocks API (`/app/api/stocks/route.ts`)

**Endpoints**:

**GET /api/stocks**
- Returns pre-loaded ticker list
- Response:
  ```json
  {
    "success": true,
    "tickers": ["AAPL", "MSFT", ...],
    "count": 50,
    "default_date_range": {
      "start": "2010-01-01",
      "end": "2024-01-01"
    }
  }
  ```

**POST /api/stocks**
- Get detailed ticker information or validate tickers
- Request body:
  ```json
  {
    "action": "info",  // or "validate"
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2010-01-01",  // optional
    "end_date": "2024-01-01"  // optional
  }
  ```
- Response (info):
  ```json
  {
    "success": true,
    "tickers": ["AAPL", "MSFT"],
    "overall": {
      "drift_annual": 0.12,
      "volatility_annual": 0.25
    },
    "ticker_stats": [
      {
        "ticker": "AAPL",
        "drift_annual": 0.15,
        "volatility_annual": 0.28
      }
    ],
    "correlation_matrix": [[1.0, 0.7], [0.7, 1.0]],
    "data_days": 3500,
    "block_length": 20
  }
  ```

#### 2. Pricing API (`/app/api/price/route.ts`)

**Endpoints**:

**GET /api/price/models**
- Returns available models and their parameters
- Includes detailed RealData parameter documentation

**POST /api/price**
- Price an option using any model
- Request body:
  ```json
  {
    "model_type": "RealData",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "tickers": ["AAPL"],
    "start_date": "2010-01-01",
    "end_date": "2024-01-01",
    "drift_override": null,
    "volatility_override": null,
    "spot": 100,
    "strike": 100,
    "rate": 0.03,
    "maturity": 1.0,
    "nb_stocks": 1,
    "nb_dates": 50,
    "nb_paths": 10000
  }
  ```
- Response:
  ```json
  {
    "success": true,
    "price": 8.45,
    "computation_time": 2.34,
    "exercise_time": 0.75,
    "paths_sample": [...],
    "model_info": {
      "type": "RealData",
      "tickers": ["AAPL"],
      "empirical_drift": 0.12,
      "empirical_volatility": 0.25,
      "block_length": 20,
      "data_days": 3500
    }
  }
  ```

### Frontend (React Components)

#### Calculator Page (`/app/calculator/page.tsx`)

**Features**:
- Model selection dropdown (includes RealData)
- Algorithm selection
- Payoff type selection
- Common parameters: spot, strike, rate, maturity
- RealData-specific controls:
  - Ticker input (comma-separated)
  - Date range selection
  - Empirical statistics display
  - Drift override toggle + input
  - Volatility override toggle + input
  - Crisis period exclusion toggle
- Real-time pricing results display
- Loading indicator with progress feedback

**RealData Workflow**:
1. User selects "Real Data (Bootstrap)" model
2. UI shows RealData parameters
3. User enters tickers (e.g., "AAPL, MSFT")
4. API fetches empirical statistics and displays them
5. User can use empirical drift/vol OR override with custom values
6. User clicks "Price Option"
7. Results show price + empirical statistics used

## Pre-loaded Tickers

The system comes with 50+ pre-loaded tickers for quick access:

**Mega Caps** (Top 10):
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, LLY, V, UNH

**Large Caps** (11-30):
- JNJ, XOM, WMT, JPM, MA, PG, AVGO, CVX, HD, MRK
- ABBV, KO, PEP, COST, ADBE, MCD, CSCO, CRM, TMO, BAC

**Diversified** (31-50):
- ACN, ABT, NFLX, WFC, DHR, NKE, DIS, VZ, CMCSA, INTC
- TXN, NEE, PM, UNP, RTX, ORCL, AMD, COP, UPS, MS
- LOW, HON, QCOM, GS, IBM, BA, CAT, SPGI, AXP, AMGN

**Default Date Range**: 2010-01-01 to 2024-01-01 (14 years of data)

## RealData Model Parameters

### Required Parameters

- `tickers`: List of stock ticker symbols (e.g., `['AAPL', 'MSFT']`)
- `spot`: Initial stock price (e.g., `100`)
- `rate`: Risk-free interest rate (e.g., `0.03` for 3%)
- `maturity`: Option maturity in years (e.g., `1.0`)

### Optional Parameters

- `start_date`: Start date for historical data (default: `'2010-01-01'`)
- `end_date`: End date for historical data (default: `'2024-01-01'`)
- `drift_override`: Override empirical drift (default: `null` = use empirical)
- `volatility_override`: Override empirical volatility (default: `null` = use empirical)
- `exclude_crisis`: Exclude 2008 and 2020 crisis periods (default: `false`)
- `only_crisis`: Only use crisis periods (default: `false`)
- `nb_stocks`: Number of stocks (default: `1`)
- `nb_dates`: Number of time steps (default: `50`)
- `nb_paths`: Number of Monte Carlo paths (default: `10000`)

### Empirical vs Override Mode

**Empirical Mode** (drift_override=null, volatility_override=null):
- Uses historical drift and volatility computed from data
- Preserves real market characteristics
- Recommended for realistic pricing

**Override Mode** (drift_override=0.05, volatility_override=0.2):
- Uses specified drift and volatility
- Resamples historical returns but adjusts for target drift/vol
- Useful for stress testing or what-if scenarios

## Block Bootstrap Details

The RealData model uses **stationary block bootstrap** (Politis & Romano, 1994):

1. **Block Length Selection**: Automatically estimated from autocorrelation structure
   - Typical range: 5-50 days
   - Preserves short-term dependencies in returns

2. **Sampling Process**:
   - Randomly selects starting points in historical data
   - Samples consecutive blocks (wrapping around)
   - Block lengths follow geometric distribution

3. **Path Generation**:
   - Aggregates sampled daily returns into time steps
   - Preserves correlations between stocks
   - Maintains volatility clustering and fat tails

## Performance Considerations

### Long Computation Times

RealData bootstrap can be slower than parametric models:
- First call: Downloads data via yfinance (~5-10 seconds)
- Subsequent calls: Uses cached data (~1-3 seconds)
- More paths = longer computation (linear scaling)
- More stocks = longer download time

**Optimization Tips**:
1. Use caching (enabled by default)
2. Start with fewer paths (1000-5000) for testing
3. Pre-load common tickers
4. Use date ranges with good data coverage (2010-2024)

### Memory Usage

- Paths are generated on-demand (not stored)
- Model caching stores minimal metadata
- Safe for typical web applications

## Error Handling

### Common Errors

**Invalid Ticker**:
```json
{
  "success": false,
  "error": "Ticker INVALID not found or has insufficient data"
}
```

**Date Range Too Short**:
```json
{
  "success": false,
  "error": "No common dates found across tickers. Try a different date range."
}
```

**Download Failure**:
```json
{
  "success": false,
  "error": "Failed to download data: Connection timeout"
}
```

### Validation

The API validates:
- Ticker symbols (via yfinance)
- Date ranges (must have sufficient data)
- Parameter ranges (positive values, etc.)
- Model-payoff compatibility

## Testing

### Manual Testing

```bash
# Test pricing engine directly
python /home/user/thesis-new-files/frontend/api/pricing_engine.py price '{
  "model_type": "RealData",
  "payoff_type": "Call",
  "algorithm": "RLSM",
  "tickers": ["AAPL"],
  "spot": 100,
  "strike": 100,
  "rate": 0.03,
  "maturity": 1.0,
  "nb_dates": 50,
  "nb_paths": 5000
}'

# Test stock data manager
python /home/user/thesis-new-files/frontend/api/stock_data.py info '{
  "tickers": ["AAPL", "MSFT"]
}'
```

### API Testing

```bash
# Test stocks endpoint
curl http://localhost:3000/api/stocks

# Test pricing endpoint
curl -X POST http://localhost:3000/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "RealData",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "tickers": ["AAPL"],
    "spot": 100,
    "strike": 100,
    "rate": 0.03,
    "maturity": 1.0
  }'
```

## Future Enhancements

### Potential Improvements

1. **Streaming Progress**:
   - Server-Sent Events for long computations
   - Real-time progress updates during bootstrap

2. **Advanced Caching**:
   - Redis for persistent cache across sessions
   - Pre-compute common ticker combinations

3. **More Statistics**:
   - Skewness and kurtosis
   - Value-at-Risk (VaR)
   - Greeks (delta, gamma, vega, theta, rho)

4. **Visualization**:
   - Historical price charts
   - Return distribution plots
   - Correlation heatmaps
   - Sample path visualizations

5. **Data Sources**:
   - Support for custom data uploads
   - Alternative data providers
   - Cryptocurrency data

## File Structure

```
frontend/
├── api/
│   ├── pricing_engine.py       # Main pricing engine
│   └── stock_data.py            # Stock data utilities
├── app/
│   ├── api/
│   │   ├── stocks/
│   │   │   └── route.ts         # Stocks API endpoint
│   │   └── price/
│   │       └── route.ts         # Pricing API endpoint
│   ├── calculator/
│   │   └── page.tsx             # Calculator UI
└── REALDATA_INTEGRATION.md      # This file
```

## Summary

The RealData integration provides:

✅ **Full RealData Support**: All features from optimal_stopping package
✅ **50+ Pre-loaded Tickers**: Common S&P 500 stocks
✅ **Empirical Statistics**: Real drift/volatility from historical data
✅ **Override Controls**: Option to use custom drift/volatility
✅ **Caching**: Avoid re-downloading data
✅ **Error Handling**: Comprehensive validation and error messages
✅ **API Endpoints**: RESTful API for stocks and pricing
✅ **React UI**: User-friendly calculator with RealData controls

The system is production-ready for pricing American options using real market data with block bootstrap resampling!
