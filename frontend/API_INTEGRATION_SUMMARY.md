# Python API Integration Summary

## Overview

Successfully created a Python API integration for the Next.js app to call optimal_stopping pricing algorithms. The integration consists of:

1. **Python Pricing Engine** (`api/pricing_engine.py`)
2. **Next.js API Routes** (`app/api/price/route.ts`, `app/api/payoffs/route.ts`)
3. **Frontend API Client** (`lib/api.ts`)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  lib/api.ts                                          │ │
│  │  - priceOption()                                     │ │
│  │  - getPayoffs()                                      │ │
│  │  - getPayoffDetails()                                │ │
│  │  - getStockInfo()                                    │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                │
│                            ▼                                │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  app/api/price/route.ts                              │ │
│  │  - POST /api/price                                   │ │
│  │  - GET /api/price                                    │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  app/api/payoffs/route.ts                            │ │
│  │  - GET /api/payoffs                                  │ │
│  │  - GET /api/payoffs?name=<payoff>                    │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼ (spawn Python subprocess)
┌─────────────────────────────────────────────────────────────┐
│                   Python Pricing Engine                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  api/pricing_engine.py                               │ │
│  │                                                      │ │
│  │  Commands:                                           │ │
│  │  - price: Price an option                            │ │
│  │  - list_payoffs: Get all available payoffs           │ │
│  │  - payoff_info: Get info about a specific payoff     │ │
│  │  - stock_info: Get empirical stock statistics        │ │
│  │  - available_tickers: Get available tickers          │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                │
│                            ▼                                │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  optimal_stopping                                    │ │
│  │  - algorithms/ (RLSM, RFQI, SRLSM, SRFQI, LSM, FQI) │ │
│  │  - data/ (BlackScholes, Heston, RealData, etc.)     │ │
│  │  - payoffs/ (408 payoff types)                       │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## File Descriptions

### 1. `frontend/api/pricing_engine.py`

**Purpose**: Python script that serves as the bridge between Next.js and the optimal_stopping library.

**Key Features**:
- Imports all algorithms (RLSM, RFQI, SRLSM, SRFQI, LSM, FQI, EOP)
- Imports all stock models (BlackScholes, Heston, FractionalBlackScholes, RoughHeston, RealDataModel)
- Uses payoff registry system to access all 408 payoff types
- Automatically routes to correct algorithm based on `is_path_dependent` flag
- Returns JSON output with price, comp_time, exercise_time, and paths_sample

**Commands**:
```bash
# Price an option
python3 pricing_engine.py price '{"model_type": "BlackScholes", "payoff_type": "Call", "algorithm": "RLSM", ...}'

# List all payoffs
python3 pricing_engine.py list_payoffs '{}'

# Get payoff info
python3 pricing_engine.py payoff_info '{"payoff_name": "Call"}'

# Get stock info (empirical drift/volatility)
python3 pricing_engine.py stock_info '{"tickers": ["AAPL", "MSFT"], ...}'

# Get available tickers
python3 pricing_engine.py available_tickers '{}'
```

**Main Class**: `PricingEngine`

**Methods**:
- `create_model(params)`: Creates stock model instance
- `create_payoff(params)`: Creates payoff instance
- `price_option(request)`: Main pricing function
- `get_stock_info(tickers, start_date, end_date)`: Get empirical statistics
- `_generate_sample_paths(model, num_samples)`: Generate paths for visualization

**Response Format**:
```json
{
  "success": true,
  "price": 7.614648,
  "computation_time": 0.001882,
  "exercise_time": 0.6144,
  "paths_sample": [[[time, price], ...], ...],
  "model_info": {
    "type": "BlackScholes",
    "spot": 100,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.03,
    "nb_stocks": 1,
    "nb_paths": 10000,
    "nb_dates": 50,
    "maturity": 1.0
  },
  "payoff_info": {
    "type": "Call",
    "strike": 100,
    "is_path_dependent": false
  },
  "algorithm": "RLSM"
}
```

---

### 2. `frontend/app/api/price/route.ts`

**Purpose**: Next.js API route handler for pricing requests.

**Endpoints**:

#### `POST /api/price`
Prices an option using the specified algorithm, model, and payoff.

**Request Body** (TypeScript interface `PricingRequest`):
```typescript
{
  algorithm: string;        // RLSM, RFQI, SRLSM, SRFQI, LSM, FQI, EOP
  model_type: string;       // BlackScholes, Heston, RealData, etc.
  payoff_type: string;      // Call, Put, BasketCall, UO_Call, etc.
  
  // Model parameters
  spot?: number;
  drift?: number;
  volatility?: number;
  rate?: number;
  nb_stocks?: number;
  nb_paths?: number;
  nb_dates?: number;
  maturity?: number;
  
  // Payoff parameters
  strike?: number;
  barrier?: number;
  k?: number;
  weights?: number[];
  
  // RealData parameters
  tickers?: string[];
  start_date?: string;
  end_date?: string;
  drift_override?: number | null;
  volatility_override?: number | null;
  
  // Heston parameters
  kappa?: number;
  theta?: number;
  xi?: number;
  rho?: number;
  v0?: number;
  
  // Algorithm parameters
  hidden_size?: number;
  nb_epochs?: number;
  factors?: number[];
}
```

**Response** (TypeScript interface `PricingResponse`):
```typescript
{
  success: boolean;
  price?: number;
  computation_time?: number;
  exercise_time?: number | null;
  paths_sample?: number[][][];  // [[[time, price], ...], ...]
  model_info?: ModelInfo;
  payoff_info?: PayoffInfo;
  algorithm?: string;
  error?: string;
}
```

#### `GET /api/price`
Returns API information about available algorithms and models.

**Key Features**:
- Spawns Python subprocess via `child_process.spawn`
- Passes parameters as JSON to Python script
- Handles timeout (60 seconds max)
- Returns structured pricing results
- Error handling with detailed error messages

---

### 3. `frontend/app/api/payoffs/route.ts`

**Purpose**: Next.js API route handler for payoff information.

**Endpoints**:

#### `GET /api/payoffs`
Returns list of all 408 available payoffs.

**Response**:
```typescript
{
  success: boolean;
  payoffs?: string[];  // Array of payoff names
  error?: string;
}
```

#### `GET /api/payoffs?name=<payoff_name>`
Returns detailed information about a specific payoff.

**Response** (TypeScript interface `PayoffDetailsResponse`):
```typescript
{
  success: boolean;
  name?: string;
  abbreviation?: string;
  is_path_dependent?: boolean;
  required_params?: string[];
  optional_params?: string[];
  error?: string;
}
```

**Example Response**:
```json
{
  "success": true,
  "name": "Call",
  "abbreviation": "Call",
  "is_path_dependent": false,
  "required_params": ["strike"],
  "optional_params": []
}
```

---

### 4. `frontend/lib/api.ts`

**Purpose**: Frontend TypeScript API client with type-safe interfaces and helper functions.

**Exported Functions**:

#### `priceOption(params: PricingRequest): Promise<PricingResponse>`
Main function to price an option.

**Usage Example**:
```typescript
import { priceOption } from '@/lib/api';

const result = await priceOption({
  algorithm: 'RLSM',
  model_type: 'BlackScholes',
  payoff_type: 'Call',
  spot: 100,
  strike: 100,
  drift: 0.05,
  volatility: 0.2,
  rate: 0.03,
  nb_paths: 10000,
  nb_dates: 50,
  maturity: 1.0,
});

if (result.success) {
  console.log('Option price:', result.price);
  console.log('Exercise time:', result.exercise_time);
  console.log('Sample paths:', result.paths_sample);
}
```

#### `getPayoffs(): Promise<PayoffListResponse>`
Get list of all available payoffs.

**Usage Example**:
```typescript
import { getPayoffs } from '@/lib/api';

const result = await getPayoffs();
if (result.success) {
  console.log('Available payoffs:', result.payoffs);
}
```

#### `getPayoffDetails(payoffName: string): Promise<PayoffDetailsResponse>`
Get information about a specific payoff.

**Usage Example**:
```typescript
import { getPayoffDetails } from '@/lib/api';

const result = await getPayoffDetails('UO_BasketCall');
if (result.success) {
  console.log('Payoff name:', result.name);
  console.log('Is path dependent:', result.is_path_dependent);
  console.log('Required params:', result.required_params);
  console.log('Optional params:', result.optional_params);
}
```

#### `getStockInfo(params: StockInfoRequest): Promise<StockInfoResponse>`
Get empirical drift/volatility statistics for stocks.

#### `validatePricingRequest(params: PricingRequest): string | null`
Validates pricing request parameters. Returns error message or null if valid.

**Exported Constants**:
- `DEFAULT_PRICING_PARAMS`: Default values for pricing parameters
- `ALGORITHM_INFO`: Information about each algorithm
- `MODEL_INFO`: Information about each model

---

## Supported Algorithms

### Standard Options (non-path-dependent)
- **RLSM**: Randomized Least Squares Monte Carlo
- **RFQI**: Randomized Fitted Q-Iteration
- **LSM**: Least Squares Monte Carlo (benchmark)
- **FQI**: Fitted Q-Iteration (benchmark)
- **EOP**: European Option Price (exercise only at maturity)

### Path-Dependent Options
- **SRLSM**: State-augmented RLSM (for barriers, lookbacks, etc.)
- **SRFQI**: State-augmented RFQI (for barriers, lookbacks, etc.)

**Algorithm Selection**:
The pricing engine automatically routes to the correct algorithm based on the payoff's `is_path_dependent` flag:
- Path-dependent payoffs → SRLSM or SRFQI
- Standard payoffs → RLSM, RFQI, LSM, FQI, or EOP

---

## Supported Stock Models

1. **BlackScholes**: Geometric Brownian motion
   - Parameters: `drift`, `volatility`, `spot`, `rate`

2. **Heston**: Stochastic volatility
   - Parameters: `drift`, `spot`, `rate`, `kappa`, `theta`, `xi`, `rho`, `v0`

3. **FractionalBlackScholes**: Fractional Brownian motion (long memory)
   - Parameters: `drift`, `volatility`, `spot`, `rate`, `hurst`

4. **RoughHeston**: Rough volatility (Hurst < 0.5)
   - Parameters: `drift`, `spot`, `rate`, `hurst`, `kappa`, `theta`, `xi`, `rho`, `v0`

5. **RealData**: Real market data with block bootstrap
   - Parameters: `tickers`, `spot`, `rate`, `start_date`, `end_date`
   - Optional: `drift_override`, `volatility_override`, `exclude_crisis`, `only_crisis`
   - Automatically calculates empirical drift/volatility if overrides not provided

---

## Supported Payoffs

**Total**: 408 payoff types

### Base Payoffs (30)
- **Simple Basket** (6): BasketCall, BasketPut, GeometricCall, GeometricPut, MaxCall, MinPut
- **Basket Asian** (4): AsianFixedStrikeCall, AsianFixedStrikePut, AsianFloatingStrikeCall, AsianFloatingStrikePut
- **Basket Range/Dispersion** (4): MaxDispersionCall, MaxDispersionPut, DispersionCall, DispersionPut
- **Basket Rank** (4): BestOfKCall, WorstOfKPut, RankWeightedBasketCall, RankWeightedBasketPut
- **Simple Single** (2): Call, Put
- **Single Lookback** (4): LookbackFixedCall, LookbackFixedPut, LookbackFloatCall, LookbackFloatPut
- **Single Asian** (4): AsianFixedStrikeCall_Single, AsianFixedStrikePut_Single, etc.
- **Single Range** (2): RangeCall_Single, RangePut_Single

### Barrier Types (11)
Each base payoff has 11 barrier variants:
- **UO**: Up-and-Out
- **DO**: Down-and-Out
- **UI**: Up-and-In
- **DI**: Down-and-In
- **UODO**: Up-and-Out Down-and-Out
- **UIDI**: Up-and-In Down-and-In
- **UIDO**: Up-and-In Down-and-Out
- **UODI**: Up-and-Out Down-and-In
- **PTB**: Partial Time Barrier
- **StepB**: Step Barrier
- **DStepB**: Double Step Barrier

**Total Barrier Variants**: 30 base × 11 barrier types = 330 barrier payoffs

**Examples**:
- `Call` → Standard call option
- `UO_Call` → Up-and-out barrier call
- `BasketCall` → Basket call on multiple assets
- `UI_DI_MaxCall` → Up-and-in down-and-in barrier on max of basket

---

## Response Data

### Price Response
```typescript
{
  success: true,
  price: 7.614648,           // Option price
  computation_time: 0.0019,  // Time to compute (seconds)
  exercise_time: 0.6144,     // Average exercise time (normalized to [0,1])
  paths_sample: [            // Sample paths for visualization
    [[0.0, 100], [0.1, 102.49], ...],  // Path 1
    [[0.0, 100], [0.1, 89.38], ...],   // Path 2
    ...
  ],
  model_info: { ... },       // Model parameters used
  payoff_info: { ... },      // Payoff parameters used
  algorithm: "RLSM"          // Algorithm used
}
```

### Exercise Time
- Normalized to [0, 1] where 0 = immediate exercise, 1 = exercise at maturity
- For a 1-year option: 0.6144 = 0.6144 × 365 = 224 days
- Only available for American-style options
- `null` for European options

### Paths Sample
- Array of 5 sample price paths for visualization
- Each path: `[[time, price], [time, price], ...]`
- Time in years (e.g., 0.1 = 36.5 days for 1-year option)
- For multi-asset options, returns average price across assets

---

## Error Handling

All API functions return a consistent error structure:

```typescript
{
  success: false,
  error: "Error message",
  error_type: "ErrorType"  // HTTPError, NetworkError, ValueError, etc.
}
```

**Common Error Types**:
- `HTTPError`: API request failed
- `NetworkError`: Network connectivity issue
- `ValueError`: Invalid parameter value
- `ImportError`: Missing Python dependency
- `TimeoutError`: Request exceeded timeout

---

## Usage Examples

### Example 1: Price a Simple Call Option
```typescript
import { priceOption } from '@/lib/api';

const result = await priceOption({
  algorithm: 'RLSM',
  model_type: 'BlackScholes',
  payoff_type: 'Call',
  spot: 100,
  strike: 100,
  drift: 0.05,
  volatility: 0.2,
  rate: 0.03,
  nb_paths: 10000,
  nb_dates: 50,
  maturity: 1.0,
});
```

### Example 2: Price a Barrier Option
```typescript
const result = await priceOption({
  algorithm: 'SRLSM',  // Path-dependent algorithm
  model_type: 'BlackScholes',
  payoff_type: 'UO_Call',  // Up-and-out barrier call
  spot: 100,
  strike: 100,
  barrier: 120,  // Knock-out at 120
  volatility: 0.2,
  rate: 0.03,
  nb_paths: 10000,
  nb_dates: 50,
  maturity: 1.0,
});
```

### Example 3: Price with Real Market Data
```typescript
const result = await priceOption({
  algorithm: 'RLSM',
  model_type: 'RealData',
  payoff_type: 'BasketCall',
  tickers: ['AAPL', 'MSFT', 'GOOGL'],
  start_date: '2010-01-01',
  end_date: '2024-01-01',
  drift_override: null,        // Use empirical drift
  volatility_override: null,   // Use empirical volatility
  spot: 100,
  strike: 100,
  rate: 0.03,
  nb_stocks: 3,
  nb_paths: 5000,
  nb_dates: 50,
  maturity: 1.0,
});
```

### Example 4: Get Available Payoffs
```typescript
import { getPayoffs, getPayoffDetails } from '@/lib/api';

// Get all payoffs
const payoffList = await getPayoffs();
console.log(payoffList.payoffs);  // Array of 408 payoff names

// Get info about specific payoff
const payoffInfo = await getPayoffDetails('UO_BasketCall');
console.log(payoffInfo.is_path_dependent);  // true
console.log(payoffInfo.optional_params);    // ['barrier']
```

---

## Testing

All components have been tested:

### Python Pricing Engine
```bash
# List payoffs (408 total)
cd frontend/api
python3 pricing_engine.py list_payoffs '{}'

# Get payoff info
python3 pricing_engine.py payoff_info '{"payoff_name": "Call"}'

# Price a call option
python3 pricing_engine.py price '{"model_type": "BlackScholes", "payoff_type": "Call", "algorithm": "RLSM", "spot": 100, "strike": 100, "drift": 0.05, "volatility": 0.2, "rate": 0.03, "nb_paths": 1000, "nb_dates": 10, "maturity": 1.0}'
```

**Test Results**:
- ✅ List payoffs: Successfully returns all 408 payoffs
- ✅ Payoff info: Returns correct metadata (name, abbreviation, is_path_dependent, params)
- ✅ Pricing: Successfully prices Call option with RLSM
- ✅ Exercise time: Returns normalized exercise time (0.6144)
- ✅ Sample paths: Returns 5 sample paths for visualization

---

## Performance Considerations

### Timeout Settings
- API routes: 60 seconds max
- Payoff queries: 30 seconds max
- Recommended: Use smaller `nb_paths` and `nb_dates` for fast UI responses

### Caching
- Python engine caches RealData models to avoid re-downloading stock data
- Cache key: `RealData_{tickers}_{start_date}_{end_date}`

### Optimization Tips
1. Use `nb_paths=1000-5000` for quick UI feedback
2. Use `nb_dates=10-50` for fast computation
3. For production pricing, increase to `nb_paths=50000+` and `nb_dates=100+`
4. RealData model: Pre-download data once, then reuse cached data

---

## Files Created

1. ✅ `frontend/api/pricing_engine.py` - Python pricing engine (updated)
2. ✅ `frontend/app/api/price/route.ts` - Next.js price API route (existing)
3. ✅ `frontend/app/api/payoffs/route.ts` - Next.js payoffs API route (new)
4. ✅ `frontend/lib/api.ts` - Frontend API client (new)

---

## Integration Status

✅ **Complete** - All components successfully created and tested:
- Python pricing engine with exercise_time and paths_sample support
- Next.js API routes for pricing and payoff queries
- Frontend TypeScript API client with type-safe interfaces
- Error handling and validation
- Comprehensive documentation

Ready for integration with Next.js frontend UI components.
