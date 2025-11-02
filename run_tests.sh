#!/bin/bash

# Test runner script with proper environment setup
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "🧪 Running Employee Similarity Search Test Suite"
echo "================================================"

# Show current working directory and PYTHONPATH
echo "📍 Working directory: $(pwd)"
echo "🔧 PYTHONPATH: $PYTHONPATH"

# Check if API module exists
if [ -f "api/main.py" ]; then
    echo "✅ API module found"
else
    echo "❌ API module not found - some tests may be skipped"
fi

echo ""

# Function to run tests with proper error handling
run_test() {
    local test_name=$1
    local test_path=$2
    
    echo "🎯 Running $test_name tests..."
    if PYTHONPATH="$(pwd)" pytest "$test_path" -v; then
        echo "✅ $test_name tests passed!"
    else
        echo "❌ $test_name tests failed!"
        return 1
    fi
    echo ""
}

# Run tests in order of complexity
echo "Starting test execution..."
echo ""

# 1. Basic unit tests (no API dependencies)
run_test "Basic Unit" "tests/unit/test_basic.py"

# 2. Setup verification
run_test "Setup Verification" "tests/test_setup.py"

# 3. Full unit tests (if API available)
if [ -f "api/main.py" ]; then
    run_test "Full Unit" "tests/unit/"
    run_test "API" "tests/api/"
    run_test "Integration" "tests/integration/"
    run_test "E2E" "tests/e2e/"
else
    echo "⚠️  Skipping API-dependent tests - API module not found"
fi

echo "🏁 Test execution completed!"