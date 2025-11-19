# Integration Test Script Documentation

## Overview

The `test-integration.sh` script provides comprehensive testing of the thesis project's full stack:
- Python pricing engine with multiple models and algorithms
- Next.js API routes integration
- Frontend components and TypeScript compilation
- File structure validation
- Error handling verification

## Script Location

```
/home/user/thesis-new-files/frontend/test-integration.sh
```

## Usage

### Basic Execution
```bash
cd /home/user/thesis-new-files/frontend
./test-integration.sh
```

### Skip API Route Tests
```bash
./test-integration.sh --skip-api
```

### Test Against Custom Server URL
```bash
./test-integration.sh --server-url http://localhost:8080
```

## Test Sections

### Section 1: File Verification (8 tests)
Validates that all required files and directories exist:
- Python backend (`optimal_stopping/`)
- Pricing engine script (`api/pricing_engine.py`)
- API routes (`app/api/payoffs/route.ts`, `app/api/price/route.ts`)
- Frontend components (`components/`)
- Next.js configuration
- Payoff system files

**Status**: All 8 tests passing ✓

### Section 2: Python Pricing Engine (6 tests)

#### 2.1 List Payoffs
- **Command**: `python3 pricing_engine.py list_payoffs`
- **Expected**: JSON response with payoff registry
- **Status**: ✓ PASS

#### 2.2 Price Simple Call with RLSM
- **Algorithm**: Randomized Least Squares Monte Carlo
- **Payoff**: Vanilla European Call
- **Parameters**:
  - Spot: $100
  - Strike: $100
  - Drift: 5%
  - Volatility: 20%
  - Maturity: 1 year
  - Paths: 1,000
- **Result**: $3.67
- **Status**: ✓ PASS

#### 2.3 Price Barrier Option (UO_Call) with SRLSM
- **Algorithm**: State-augmented RLSM (for path-dependent options)
- **Payoff**: Up-and-Out Barrier Call
- **Barrier Level**: $120
- **Status**: ✗ FAIL - TypeError in barrier comparison logic

#### 2.4 Price with RealData Model
- **Model**: Real market data with stationary block bootstrap
- **Ticker**: AAPL
- **Status**: ✗ FAIL - yfinance not installed (expected, optional dependency)

#### 2.5 Get Payoff Information
- **Command**: `python3 pricing_engine.py payoff_info {"payoff_name": "Call"}`
- **Status**: ✓ PASS

#### 2.6 Price with LSM Algorithm
- **Algorithm**: Least Squares Monte Carlo (classic)
- **Status**: ✓ PASS

### Section 3: Frontend Build (5 tests)

#### 3.1 Node.js Installation
- **Version**: v22.21.1
- **Status**: ✓ PASS

#### 3.2 npm Installation
- **Version**: 10.9.4
- **Status**: ✓ PASS

#### 3.3 Dependencies Installation
- **Check**: `node_modules/` exists and populated
- **Status**: ✓ PASS

#### 3.4 TypeScript Availability
- **Check**: TypeScript compiler available
- **Status**: ✓ PASS

#### 3.5 Component Validation
- **Check**: Components have proper TypeScript exports
- **Status**: ✓ PASS

### Section 4: Payoff Registry (1 test)

Tests that the payoff auto-registration system works:
- Imports `Call` class ✓
- Imports `Put` class ✓
- Imports `UO_Call` barrier class ✓

**Status**: ✓ PASS

### Section 5: Data & Storage (2 tests)

#### 5.1 Stored Paths Directory
- Location: `optimal_stopping/data/stored_paths/`
- Purpose: Memory-mapped HDF5 storage for large experiments
- Status: ✓ PASS (1 file found)

#### 5.2 Results Output Directory
- Locations: `output/`, `results/`
- Purpose: Store experiment results and figures
- Status: ✓ PASS

### Section 6: Error Handling (3 tests)

#### 6.1 Invalid Payoff Type
```json
{
    "payoff_type": "InvalidPayoff"
}
```
- **Expected**: Error message returned
- **Status**: ✓ PASS

#### 6.2 Invalid Model Type
```json
{
    "model_type": "InvalidModel"
}
```
- **Expected**: Error message returned
- **Status**: ✓ PASS

#### 6.3 Invalid Algorithm
```json
{
    "algorithm": "InvalidAlgo"
}
```
- **Expected**: Error message returned
- **Status**: ✓ PASS

### Section 7: API Route Tests (Available when server running)

When a Next.js development server is running on port 3000 (or custom URL):

#### 7.1 GET /api/payoffs
- **Purpose**: List all available payoffs
- **Response**: JSON array with payoff information

#### 7.2 GET /api/payoffs?name=Call
- **Purpose**: Get specific payoff information
- **Response**: JSON object with Call payoff details

#### 7.3 POST /api/price
- **Purpose**: Price an option with specified model/algorithm
- **Request Body**: JSON with pricing parameters
- **Response**: Calculated option price and metadata

#### 7.4 POST /api/price (Barrier Option)
- **Purpose**: Price barrier options via API
- **Example Payoff**: UO_Call (Up-and-Out Call)

#### 7.5 GET /api/price
- **Purpose**: Get available models and algorithms
- **Response**: JSON with model and algorithm information

## Test Results

### Overall Statistics
- **Total Tests Run**: 27
- **Tests Passed**: 25 (92.6%)
- **Tests Failed**: 2 (7.4%)

### Detailed Results

| Test Category | Total | Passed | Failed |
|---|---|---|---|
| File Verification | 8 | 8 | 0 |
| Python Engine | 6 | 4 | 2 |
| Frontend Build | 5 | 5 | 0 |
| Payoff Registry | 1 | 1 | 0 |
| Data & Storage | 2 | 2 | 0 |
| Error Handling | 3 | 3 | 0 |
| API Routes | - | - | - |

### Known Issues

#### 1. Barrier Option Pricing (FAIL)
**Issue**: TypeError when pricing UO_Call barrier option
```
Error: '<' not supported between instances of 'float' and 'NoneType'
```
**Cause**: Possible missing parameter in barrier option setup
**Investigation Needed**: Check barrier parameter handling in SRLSM algorithm

#### 2. RealData Model (FAIL)
**Issue**: yfinance library not installed
```
Error: yfinance is required for RealDataModel. Install with: pip install yfinance
```
**Expected**: This is an optional dependency for real market data
**Resolution**: Install with `pip install yfinance` if real data pricing needed

## Output Files

### Results Log
```
/tmp/test-integration-results.txt
```
Contains timestamped test results with PASS/FAIL status for each test.

### Console Output
Colored output to terminal with:
- GREEN: Passing tests
- RED: Failing tests
- YELLOW: Test indicators
- BLUE: Section headers and info messages

## Python Pricing Engine Commands

The `pricing_engine.py` script supports these commands:

### 1. List Payoffs
```bash
python3 api/pricing_engine.py list_payoffs
```
**Returns**: JSON with all available payoff types and variants

### 2. Price Option
```bash
python3 api/pricing_engine.py price <json_params>
```
**Parameters**:
- `model_type`: BlackScholes, Heston, FractionalBlackScholes, RoughHeston, RealData
- `payoff_type`: Call, Put, UO_Call, etc.
- `algorithm`: RLSM, RFQI, LSM, FQI, SRLSM, SRFQI
- `spot`: Initial stock price
- `strike`: Strike price
- `maturity`: Time to maturity in years
- `drift`: Annual drift (float)
- `volatility`: Annual volatility (float)
- `rate`: Risk-free rate
- `nb_stocks`: Number of stocks
- `nb_paths`: Number of simulation paths
- `nb_dates`: Number of time steps

### 3. Get Payoff Info
```bash
python3 api/pricing_engine.py payoff_info <json_params>
```
**Parameters**: `{"payoff_name": "Call"}`

## Running with Next.js Server

### Start Development Server
```bash
cd /home/user/thesis-new-files/frontend
npm run dev
```
Server will be available at `http://localhost:3000`

### Run Full Integration Tests
```bash
./test-integration.sh
```
This will test both Python engine AND API routes

## Dependencies

### Python
- Python 3.11.14
- numpy, scipy
- torch (for neural network algorithms)
- scikit-learn
- pandas
- h5py (HDF5 support)
- yfinance (optional, for RealData model)

### Node.js
- Node.js v22.21.1
- npm 10.9.4
- Next.js dependencies (in node_modules/)

## Customization

### Add Custom Tests
Edit the script and add new test functions:
```bash
section_custom_tests() {
    print_section "SECTION X: Custom Tests"
    run_test "Test description"
    # Your test logic here
}
```

### Change Server URL
```bash
./test-integration.sh --server-url http://custom-server:3000
```

### Skip Sections
Modify the `main()` function to comment out unwanted sections:
```bash
# section_api_routes  # Comment to skip
```

## Troubleshooting

### Python Script Not Found
```
Error: pricing_engine.py not found
```
**Solution**: Verify `api/` directory structure

### Port Already in Use
```
Error: address already in use
```
**Solution**: Kill existing process or change port:
```bash
npm run dev -- -p 3001
```

### Python Dependencies Missing
```
Error: ModuleNotFoundError: No module named 'optimal_stopping'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

## Integration with CI/CD

The script can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Integration Tests
  run: |
    cd frontend
    ./test-integration.sh --skip-api
```

Exit codes:
- `0`: All tests passed
- `1`: One or more tests failed

## Future Improvements

- [ ] Add performance benchmarking
- [ ] Add memory usage monitoring
- [ ] Add network latency tests
- [ ] Add database connectivity tests
- [ ] Parallel test execution
- [ ] HTML report generation
- [ ] Slack/email notifications on failure
