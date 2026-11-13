#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ge 1 ]]; then
    ENABLE_HTML_VIEW="$1"
    PORT_INC_MAX=10
else
    ENABLE_HTML_VIEW=0
fi

# 1. Build unittest
echo "============================================"
echo "Step 1: Cleaning old build artifacts and generating new unittest"
echo "============================================"
rm -rf build 
bash scripts/build.sh --tests catlass_unittest -DENABLE_COVERAGE=ON

# 2. Running test
echo ""
echo "============================================"
echo "Step 2: Running unittest..."
echo "============================================"
./build/tests/unittest/catlass_unittest_2201 2>&1 | tee unittest_2201.log
./build/tests/unittest/catlass_unittest_3510 2>&1 | tee unittest_3510.log

# 3. Coverage report
echo ""
echo "============================================"
echo "Step 3: Generating coverage report..."
echo "============================================"
cd build && make coverage_collect

# 4. View report (use inner server)
echo ""
echo "============================================"
echo "Step 4: Viewing coverage report..."
echo "============================================"
if [[ $ENABLE_HTML_VIEW -eq 1 ]]; then
    cd coverage/html
    PORT=8080
    PORT_INC=0
    while lsof -i :$PORT &>/dev/null && ((PORT_INC < PORT_INC_MAX)); do
        echo "Port $PORT is already in use, trying next port..."
        PORT=$((PORT + 1))
        PORT_INC=$((PORT_INC + 1))
    done
    echo "Starting server on port $PORT..."
    npx serve -l $PORT
fi

