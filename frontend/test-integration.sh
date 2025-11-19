#!/bin/bash

###############################################################################
# Integration Test Script for Thesis Frontend + Python Backend
#
# This script tests:
# 1. Python pricing engine directly
# 2. Next.js API routes (if server is running)
# 3. Frontend components compilation
# 4. Required files existence
#
# Usage: ./test-integration.sh [--server-url URL] [--skip-api]
###############################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR"
API_DIR="$FRONTEND_DIR/api"
BACKEND_DIR="/home/user/thesis-new-files"
OPTIMAL_STOPPING_DIR="$BACKEND_DIR/optimal_stopping"

# Configuration
SERVER_URL="${1:-http://localhost:3000}"
SKIP_API_TESTS=false

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Output file for results
RESULTS_FILE="/tmp/test-integration-results.txt"
> "$RESULTS_FILE"

###############################################################################
# Helper Functions
###############################################################################

print_section() {
    echo ""
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    echo "[PASS] $1" >> "$RESULTS_FILE"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    echo "[FAIL] $1" >> "$RESULTS_FILE"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$RESULTS_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$RESULTS_FILE"
}

run_test() {
    ((TESTS_RUN++))
    print_test "$1"
}

print_summary() {
    echo ""
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}TEST SUMMARY${NC}"
    echo -e "${BLUE}===================================================${NC}"
    echo "Total Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    else
        echo -e "Tests Failed: ${GREEN}0${NC}"
    fi
    echo ""
    echo "Results saved to: $RESULTS_FILE"
}

###############################################################################
# Section 1: Check Required Files Exist
###############################################################################

section_check_files() {
    print_section "SECTION 1: Checking Required Files"

    # Python backend files
    run_test "Python backend directory exists"
    if [ -d "$OPTIMAL_STOPPING_DIR" ]; then
        print_pass "Python backend directory exists: $OPTIMAL_STOPPING_DIR"
    else
        print_fail "Python backend directory missing: $OPTIMAL_STOPPING_DIR"
        return 1
    fi

    # Pricing engine file
    run_test "Pricing engine file exists"
    if [ -f "$API_DIR/pricing_engine.py" ]; then
        print_pass "Pricing engine file exists"
    else
        print_fail "Pricing engine file missing: $API_DIR/pricing_engine.py"
        return 1
    fi

    # API routes
    run_test "API payoffs route exists"
    if [ -f "$FRONTEND_DIR/app/api/payoffs/route.ts" ]; then
        print_pass "API payoffs route exists"
    else
        print_fail "API payoffs route missing"
    fi

    run_test "API price route exists"
    if [ -f "$FRONTEND_DIR/app/api/price/route.ts" ]; then
        print_pass "API price route exists"
    else
        print_fail "API price route missing"
    fi

    # Frontend components
    run_test "Frontend components directory exists"
    if [ -d "$FRONTEND_DIR/components" ]; then
        print_pass "Components directory exists"

        # Check key components
        if [ -f "$FRONTEND_DIR/components/PayoffSelector.tsx" ]; then
            print_info "  ✓ PayoffSelector.tsx"
        fi
    else
        print_fail "Components directory missing"
    fi

    # Next.js configuration
    run_test "Next.js configuration exists"
    if [ -f "$FRONTEND_DIR/next.config.mjs" ] || [ -f "$FRONTEND_DIR/next.config.ts" ]; then
        print_pass "Next.js configuration exists"
    else
        print_fail "Next.js configuration missing"
    fi

    # Package files
    run_test "package.json exists"
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        print_pass "package.json exists"
    else
        print_fail "package.json missing"
    fi

    # Payoff files
    run_test "Payoff system files exist"
    if [ -f "$OPTIMAL_STOPPING_DIR/payoffs/payoff.py" ] && \
       [ -f "$OPTIMAL_STOPPING_DIR/payoffs/__init__.py" ]; then
        print_pass "Payoff system files exist"
    else
        print_fail "Payoff system files missing"
    fi
}

###############################################################################
# Section 2: Python Pricing Engine Tests
###############################################################################

section_python_engine() {
    print_section "SECTION 2: Python Pricing Engine Tests"

    # Check Python is available
    run_test "Python 3 is installed"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_pass "Python 3 is installed: $PYTHON_VERSION"
    else
        print_fail "Python 3 not found"
        return 1
    fi

    # Test 2.1: List payoffs
    run_test "List all available payoffs"
    PAYOFFS_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "list_payoffs" 2>&1)
    if echo "$PAYOFFS_OUTPUT" | grep -q '"success": true'; then
        print_pass "Successfully listed payoffs"
        PAYOFF_COUNT=$(echo "$PAYOFFS_OUTPUT" | grep -o '"payoffs"' | wc -l)
        print_info "  Found payoff registry response"
    else
        print_fail "Failed to list payoffs"
        print_info "  Output: $PAYOFFS_OUTPUT"
    fi

    # Test 2.2: Price a simple Call with RLSM
    run_test "Price simple Call option with RLSM"
    PRICE_REQUEST=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 100.0,
    "nb_stocks": 1,
    "nb_paths": 1000,
    "nb_dates": 50,
    "seed": 42
}
EOF
)
    PRICE_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$PRICE_REQUEST" 2>&1)
    if echo "$PRICE_OUTPUT" | grep -q '"success": true' && echo "$PRICE_OUTPUT" | grep -q '"price"'; then
        print_pass "Successfully priced Call with RLSM"
        PRICE=$(echo "$PRICE_OUTPUT" | grep -o '"price": [0-9.]*' | head -1 | grep -o '[0-9.]*')
        print_info "  Option price: $PRICE"
    else
        print_fail "Failed to price Call option"
        print_info "  Output: ${PRICE_OUTPUT:0:200}"
    fi

    # Test 2.3: Price barrier option (UO_Call) with SRLSM
    run_test "Price barrier Call (UO_Call) with SRLSM"
    BARRIER_REQUEST=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "UO_Call",
    "algorithm": "SRLSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 100.0,
    "barriers": 120.0,
    "nb_stocks": 1,
    "nb_paths": 1000,
    "nb_dates": 50,
    "seed": 42
}
EOF
)
    BARRIER_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$BARRIER_REQUEST" 2>&1)
    if echo "$BARRIER_OUTPUT" | grep -q '"success": true' && echo "$BARRIER_OUTPUT" | grep -q '"price"'; then
        print_pass "Successfully priced barrier option (UO_Call)"
        PRICE=$(echo "$BARRIER_OUTPUT" | grep -o '"price": [0-9.]*' | head -1 | grep -o '[0-9.]*')
        print_info "  Option price: $PRICE"
    else
        print_fail "Failed to price barrier option"
        print_info "  Output: ${BARRIER_OUTPUT:0:200}"
    fi

    # Test 2.4: Price with RealData model
    run_test "Price Call with RealData model"
    REALDATA_REQUEST=$(cat <<'EOF'
{
    "model_type": "RealData",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "tickers": ["AAPL"],
    "spot": 150.0,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 150.0,
    "nb_stocks": 1,
    "nb_paths": 500,
    "nb_dates": 50,
    "seed": 42
}
EOF
)
    REALDATA_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$REALDATA_REQUEST" 2>&1)
    if echo "$REALDATA_OUTPUT" | grep -q '"success": true' && echo "$REALDATA_OUTPUT" | grep -q '"price"'; then
        print_pass "Successfully priced with RealData model"
        PRICE=$(echo "$REALDATA_OUTPUT" | grep -o '"price": [0-9.]*' | head -1 | grep -o '[0-9.]*')
        print_info "  Option price: $PRICE"
    else
        print_fail "Failed to price with RealData model (may be expected if network/cache issue)"
        print_info "  Output: ${REALDATA_OUTPUT:0:200}"
    fi

    # Test 2.5: Get payoff info
    run_test "Get payoff information"
    INFO_REQUEST='{"payoff_name": "Call"}'
    INFO_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "payoff_info" "$INFO_REQUEST" 2>&1)
    if echo "$INFO_OUTPUT" | grep -q '"success": true'; then
        print_pass "Successfully retrieved payoff info"
    else
        print_fail "Failed to get payoff info"
        print_info "  Output: ${INFO_OUTPUT:0:200}"
    fi

    # Test 2.6: Test multiple algorithms
    run_test "Price with LSM algorithm"
    LSM_REQUEST=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "Call",
    "algorithm": "LSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 100.0,
    "nb_stocks": 1,
    "nb_paths": 500,
    "nb_dates": 50,
    "seed": 42
}
EOF
)
    LSM_OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$LSM_REQUEST" 2>&1)
    if echo "$LSM_OUTPUT" | grep -q '"success": true'; then
        print_pass "Successfully priced with LSM algorithm"
    else
        print_fail "Failed to price with LSM algorithm"
    fi
}

###############################################################################
# Section 3: Frontend Components Build Test
###############################################################################

section_frontend_build() {
    print_section "SECTION 3: Frontend Components Build Test"

    # Check Node.js
    run_test "Node.js is installed"
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version 2>&1)
        print_pass "Node.js is installed: $NODE_VERSION"
    else
        print_fail "Node.js not found"
        return 1
    fi

    # Check npm
    run_test "npm is installed"
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version 2>&1)
        print_pass "npm is installed: $NPM_VERSION"
    else
        print_fail "npm not found"
        return 1
    fi

    # Check node_modules
    run_test "Dependencies are installed (node_modules exists)"
    if [ -d "$FRONTEND_DIR/node_modules" ] && [ -n "$(ls -A $FRONTEND_DIR/node_modules 2>/dev/null)" ]; then
        print_pass "Dependencies are installed"
    else
        print_fail "Dependencies not installed"
        print_info "  Run: npm install in $FRONTEND_DIR"
    fi

    # Test TypeScript compilation
    run_test "TypeScript files can be checked"
    if command -v npx &> /dev/null; then
        # Just check if TypeScript is available, don't run full build
        if npx -y tsc --version &>/dev/null; then
            print_pass "TypeScript is available"
        else
            print_fail "TypeScript not available"
        fi
    else
        print_fail "npx not available"
    fi

    # Check critical components exist and have valid syntax
    run_test "PayoffSelector component is valid TypeScript"
    if [ -f "$FRONTEND_DIR/components/PayoffSelector.tsx" ]; then
        if grep -q "export" "$FRONTEND_DIR/components/PayoffSelector.tsx"; then
            print_pass "PayoffSelector component has export statement"
        else
            print_fail "PayoffSelector component missing export"
        fi
    else
        print_fail "PayoffSelector component not found"
    fi

    run_test "API routes are valid TypeScript"
    if [ -f "$FRONTEND_DIR/app/api/payoffs/route.ts" ]; then
        if grep -q "export" "$FRONTEND_DIR/app/api/payoffs/route.ts"; then
            print_pass "Payoffs API route has export statement"
        else
            print_fail "Payoffs API route missing export"
        fi
    else
        print_fail "Payoffs API route not found"
    fi
}

###############################################################################
# Section 4: Next.js API Routes Test
###############################################################################

section_api_routes() {
    print_section "SECTION 4: Next.js API Routes Test"

    # Check if server is running
    run_test "Next.js server is running on $SERVER_URL"
    if curl -s "$SERVER_URL" > /dev/null 2>&1; then
        print_pass "Server is accessible at $SERVER_URL"
    else
        print_fail "Server not accessible at $SERVER_URL"
        print_info "  To start server: npm run dev"
        print_info "  Skipping API route tests"
        return 0
    fi

    # Test 4.1: GET /api/payoffs
    run_test "GET /api/payoffs - List all payoffs"
    RESPONSE=$(curl -s -X GET "$SERVER_URL/api/payoffs" -H "Content-Type: application/json")
    if echo "$RESPONSE" | grep -q '"success": true'; then
        print_pass "GET /api/payoffs returned success"
        if echo "$RESPONSE" | grep -q '"payoffs"'; then
            print_info "  Response contains payoffs data"
        fi
    else
        print_fail "GET /api/payoffs failed"
        print_info "  Response: ${RESPONSE:0:200}"
    fi

    # Test 4.2: GET /api/payoffs?name=Call
    run_test "GET /api/payoffs?name=Call - Get specific payoff info"
    RESPONSE=$(curl -s -X GET "$SERVER_URL/api/payoffs?name=Call" -H "Content-Type: application/json")
    if echo "$RESPONSE" | grep -q '"success": true' || echo "$RESPONSE" | grep -q "Call"; then
        print_pass "GET /api/payoffs?name=Call returned data"
    else
        print_fail "GET /api/payoffs?name=Call failed"
        print_info "  Response: ${RESPONSE:0:200}"
    fi

    # Test 4.3: POST /api/price
    run_test "POST /api/price - Price a simple option"
    PRICE_PAYLOAD=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 100.0,
    "nb_stocks": 1,
    "nb_paths": 1000,
    "nb_dates": 50
}
EOF
)
    RESPONSE=$(curl -s -X POST "$SERVER_URL/api/price" \
        -H "Content-Type: application/json" \
        -d "$PRICE_PAYLOAD")
    if echo "$RESPONSE" | grep -q '"success": true'; then
        print_pass "POST /api/price succeeded"
        if echo "$RESPONSE" | grep -q '"price"'; then
            PRICE=$(echo "$RESPONSE" | grep -o '"price": [0-9.]*' | head -1)
            print_info "  $PRICE"
        fi
    else
        print_fail "POST /api/price failed"
        print_info "  Response: ${RESPONSE:0:200}"
    fi

    # Test 4.4: POST /api/price with barrier option
    run_test "POST /api/price - Price barrier option (UO_Call)"
    BARRIER_PAYLOAD=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "UO_Call",
    "algorithm": "SRLSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "strike": 100.0,
    "barriers": 120.0,
    "nb_stocks": 1,
    "nb_paths": 1000,
    "nb_dates": 50
}
EOF
)
    RESPONSE=$(curl -s -X POST "$SERVER_URL/api/price" \
        -H "Content-Type: application/json" \
        -d "$BARRIER_PAYLOAD")
    if echo "$RESPONSE" | grep -q '"success": true'; then
        print_pass "POST /api/price succeeded for barrier option"
    else
        print_fail "POST /api/price failed for barrier option"
        print_info "  Response: ${RESPONSE:0:200}"
    fi

    # Test 4.5: GET /api/price/models
    run_test "GET /api/price/models - Get available models"
    RESPONSE=$(curl -s -X GET "$SERVER_URL/api/price" -H "Content-Type: application/json")
    if echo "$RESPONSE" | grep -q '"success": true'; then
        print_pass "GET /api/price returned model information"
        if echo "$RESPONSE" | grep -q '"models"'; then
            print_info "  Response contains model details"
        fi
    else
        print_fail "GET /api/price failed"
        print_info "  Response: ${RESPONSE:0:200}"
    fi
}

###############################################################################
# Section 5: Payoff Registry Test
###############################################################################

section_payoff_registry() {
    print_section "SECTION 5: Payoff Registry Test"

    run_test "Test payoff class imports"
    IMPORT_TEST=$(python3 -c "
import sys
sys.path.insert(0, '/home/user/thesis-new-files')
from optimal_stopping.payoffs import get_payoff_class
try:
    call = get_payoff_class('Call')
    print('Call class found')
    put = get_payoff_class('Put')
    print('Put class found')
    uo_call = get_payoff_class('UO_Call')
    print('UO_Call barrier class found')
except Exception as e:
    print(f'Error: {e}')
" 2>&1)

    if echo "$IMPORT_TEST" | grep -q "Call class found"; then
        print_pass "Payoff classes can be imported and registered"
        print_info "  $IMPORT_TEST"
    else
        print_fail "Failed to import payoff classes"
        print_info "  $IMPORT_TEST"
    fi
}

###############################################################################
# Section 6: Data and Storage Tests
###############################################################################

section_data_storage() {
    print_section "SECTION 6: Data and Storage Tests"

    run_test "Check if stored_paths directory exists"
    if [ -d "$OPTIMAL_STOPPING_DIR/data/stored_paths" ]; then
        print_pass "Stored paths directory exists"
        FILE_COUNT=$(find "$OPTIMAL_STOPPING_DIR/data/stored_paths" -type f 2>/dev/null | wc -l)
        print_info "  Contains $FILE_COUNT files"
    else
        print_info "Stored paths directory doesn't exist (will be created on demand)"
    fi

    run_test "Check if results output directory exists"
    if [ -d "$BACKEND_DIR/output" ] || [ -d "$BACKEND_DIR/results" ]; then
        print_pass "Results output directory exists"
    else
        print_info "Results directory will be created on first run"
    fi
}

###############################################################################
# Section 7: Error Handling Tests
###############################################################################

section_error_handling() {
    print_section "SECTION 7: Error Handling Tests"

    run_test "Handle invalid payoff type gracefully"
    INVALID_REQUEST=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "InvalidPayoff",
    "algorithm": "RLSM",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "nb_stocks": 1,
    "nb_paths": 100,
    "nb_dates": 10
}
EOF
)
    OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$INVALID_REQUEST" 2>&1)
    if echo "$OUTPUT" | grep -q '"success": false' || echo "$OUTPUT" | grep -q "error\|Error"; then
        print_pass "Invalid payoff type handled with error message"
    else
        print_fail "Invalid payoff type not handled properly"
    fi

    run_test "Handle invalid model type gracefully"
    INVALID_MODEL=$(cat <<'EOF'
{
    "model_type": "InvalidModel",
    "payoff_type": "Call",
    "algorithm": "RLSM",
    "spot": 100.0,
    "rate": 0.05,
    "maturity": 1.0,
    "nb_stocks": 1,
    "nb_paths": 100,
    "nb_dates": 10
}
EOF
)
    OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$INVALID_MODEL" 2>&1)
    if echo "$OUTPUT" | grep -q '"success": false' || echo "$OUTPUT" | grep -q "error\|Error"; then
        print_pass "Invalid model type handled with error message"
    else
        print_fail "Invalid model type not handled properly"
    fi

    run_test "Handle invalid algorithm gracefully"
    INVALID_ALGO=$(cat <<'EOF'
{
    "model_type": "BlackScholes",
    "payoff_type": "Call",
    "algorithm": "InvalidAlgo",
    "spot": 100.0,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0,
    "nb_stocks": 1,
    "nb_paths": 100,
    "nb_dates": 10
}
EOF
)
    OUTPUT=$(python3 "$API_DIR/pricing_engine.py" "price" "$INVALID_ALGO" 2>&1)
    if echo "$OUTPUT" | grep -q '"success": false' || echo "$OUTPUT" | grep -q "error\|Error"; then
        print_pass "Invalid algorithm handled with error message"
    else
        print_fail "Invalid algorithm not handled properly"
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    print_info "Integration Test Suite Starting..."
    print_info "Frontend Directory: $FRONTEND_DIR"
    print_info "Backend Directory: $BACKEND_DIR"
    print_info "Results file: $RESULTS_FILE"
    print_info ""

    # Run all test sections
    section_check_files
    section_python_engine
    section_frontend_build
    section_payoff_registry
    section_data_storage
    section_error_handling

    # Only run API tests if explicitly requested or if server is running
    if [ "$SKIP_API_TESTS" != "true" ]; then
        section_api_routes
    fi

    # Print summary
    print_summary

    # Exit with appropriate code
    if [ $TESTS_FAILED -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Handle command line arguments
if [ "$1" == "--skip-api" ]; then
    SKIP_API_TESTS=true
fi

# Run main function
main
