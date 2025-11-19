# Test Script Quick Start Guide

## Run Tests Immediately

```bash
cd /home/user/thesis-new-files/frontend
./test-integration.sh --skip-api
```

## What Gets Tested

✓ **27 Total Tests** (92.6% passing)

```
1. File Structure (8 tests)
   ✓ Python backend exists
   ✓ Pricing engine installed
   ✓ API routes configured
   ✓ Components ready
   ✓ Next.js setup complete

2. Python Pricing Engine (6 tests)
   ✓ List available payoffs
   ✓ Price vanilla Call option
   ✗ Price barrier option (needs fix)
   ✗ RealData model (yfinance optional)
   ✓ Retrieve payoff info
   ✓ Price with LSM algorithm

3. Frontend Build (5 tests)
   ✓ Node.js v22.21.1 installed
   ✓ npm 10.9.4 installed
   ✓ Dependencies in node_modules
   ✓ TypeScript available
   ✓ Components are valid

4. Payoff Registry (1 test)
   ✓ Auto-registration system works

5. Data Storage (2 tests)
   ✓ HDF5 paths directory ready
   ✓ Results directory ready

6. Error Handling (3 tests)
   ✓ Invalid payoffs rejected
   ✓ Invalid models rejected
   ✓ Invalid algorithms rejected
```

## Test Output Examples

### Successful Pricing
```
[PASS] Successfully priced Call with RLSM
  Option price: 3.6707633893219946
```

### Payoff Listing
```
[PASS] Successfully listed payoffs
  Found payoff registry response
```

### Barrier Option (Currently Failing)
```
[FAIL] Failed to price barrier option
  Error: '<' not supported between instances of 'float' and 'NoneType'
```

## Common Commands

### Run All Tests
```bash
./test-integration.sh
```

### Skip API Server Tests
```bash
./test-integration.sh --skip-api
```

### Test Against Custom Server
```bash
./test-integration.sh --server-url http://localhost:8080
```

### View Results Log
```bash
cat /tmp/test-integration-results.txt
```

## API Route Testing (Optional)

To test Next.js API routes, start the dev server first:

```bash
# Terminal 1: Start Next.js server
npm run dev

# Terminal 2: Run full tests
./test-integration.sh
```

## Understanding Test Results

### Green (PASS)
- Test passed successfully
- Component works as expected

### Red (FAIL)
- Test failed
- Check error message for details
- See Known Issues section below

### Blue (INFO)
- Additional information
- Details about test behavior

## Known Issues

### Issue 1: Barrier Option Pricing Fails
**Problem**: UO_Call barrier option pricing returns TypeError
**Impact**: Barrier options cannot be priced
**Status**: Needs investigation in SRLSM algorithm
**Workaround**: Use vanilla options for now

### Issue 2: RealData Model Fails
**Problem**: yfinance library not installed
**Impact**: Cannot price with real market data
**Status**: Expected - optional dependency
**Solution**: `pip install yfinance` if needed

## Quick Test Scenarios

### Scenario 1: Quick Sanity Check
```bash
./test-integration.sh --skip-api
```
Expected result: 25/27 tests pass

### Scenario 2: Development Testing
```bash
# Start dev server
npm run dev &

# Run full tests
./test-integration.sh
```
Expected: API routes tested

### Scenario 3: CI/CD Pipeline
```bash
./test-integration.sh --skip-api || exit 1
```
Fails if any test fails (useful for CI)

## Test Execution Times

Typical execution times:
- File checks: < 1 second
- Python engine tests: 10-30 seconds
- Frontend build tests: 1-2 seconds
- Payoff registry: 2-3 seconds
- Error handling: 5-10 seconds
- **Total**: ~30-50 seconds (without API server tests)

## Test Requirements

### Installed on System
- ✓ Python 3.11.14
- ✓ Node.js v22.21.1
- ✓ npm 10.9.4
- ✓ TypeScript compiler
- ✓ Next.js (in node_modules)

### Optional
- ✗ Next.js dev server (for API tests)
- ✗ yfinance (for RealData model)

## Results Storage

Test results are automatically saved to:
```
/tmp/test-integration-results.txt
```

Each run appends results with timestamps.

## Integration Examples

### GitHub Actions Workflow
```yaml
- name: Run Integration Tests
  run: |
    cd frontend
    chmod +x test-integration.sh
    ./test-integration.sh --skip-api
```

### Local Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
cd frontend
./test-integration.sh --skip-api || exit 1
```

### Docker Build Integration
```dockerfile
RUN cd /app/frontend && \
    chmod +x test-integration.sh && \
    ./test-integration.sh --skip-api
```

## Next Steps

1. Review test results: `cat /tmp/test-integration-results.txt`
2. Check failing tests (if any)
3. Read full documentation: `TEST_SCRIPT_DOCUMENTATION.md`
4. Start dev server: `npm run dev`
5. Run API tests: `./test-integration.sh`

## Support

For detailed information, see:
- `TEST_SCRIPT_DOCUMENTATION.md` - Full documentation
- `README.md` - Project overview
- `DEPLOYMENT.md` - Deployment guide

## Performance Notes

- Python tests use small sample sizes (500-1000 paths)
- Frontend uses existing compiled Next.js build
- Tests are designed to complete in < 1 minute
- Can be run repeatedly without cleanup

---

**Last Updated**: 2025-11-19
**Script Version**: 1.0
**Tests Version**: 27 tests
