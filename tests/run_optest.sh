#!/bin/bash

# Usage: run_optest.sh [CASE_NAME ...]
#   CASE_NAME 为受影响用例名（如 04_padding_matmul），按数字前缀匹配 tests/test_NN_*.py；
#   不传参数时执行 tests/ 下全部用例。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(which python3)}"
echo "Using python: $($PYTHON --version 2>&1) at $PYTHON"

clear_caches() {
    # JIT cache priority (kernels/jit/jit_compiler.cpp):
    #   1. $CATLASS_JIT_CACHE_DIR  (env override)
    #   2. $HOME/.cache/catlass/jit_cache  (default, with <version> subdir)
    #   3. /tmp/catlass_jit  (fallback when HOME is unset)
    if [ -n "${CATLASS_JIT_CACHE_DIR:-}" ] && [ -d "$CATLASS_JIT_CACHE_DIR" ]; then
        echo "    \$CATLASS_JIT_CACHE_DIR=$CATLASS_JIT_CACHE_DIR"
        rm -rf "$CATLASS_JIT_CACHE_DIR"
    fi
    local home_jit="$HOME/.cache/catlass/jit_cache"
    if [ -d "$home_jit" ]; then
        echo "    $home_jit"
        rm -rf "$home_jit"
    fi
    if [ -d /tmp/catlass_jit ]; then
        echo "    /tmp/catlass_jit (fallback)"
        rm -rf /tmp/catlass_jit
    fi
    echo "    $1/.pytest_cache"
    rm -rf "$1/.pytest_cache"
    echo "    $1/tests/__pycache__"
    rm -rf "$1/tests/__pycache__"
    echo "    $1/tests/test.log"
    rm -rf "$1/tests/test.log"
    echo "    $1/.ruff_cache"
    rm -rf "$1/.ruff_cache"
}

# 1. 清理旧的编译产物
echo "============================================"
echo "Step 1: Cleaning old build artifacts..."
echo "============================================"
rm -rf optest/build optest/dist optest/*.egg-info optest/_skbuild

echo ""
echo "Cleaning runtime caches before build..."
clear_caches "optest"

# 2. 编译 optest
echo ""
echo "============================================"
echo "Step 2: Building optest..."
echo "============================================"

if ! CC=$(command -v gcc); then
    echo "Error: gcc not found in PATH" >&2
    exit 1
fi
if ! CXX=$(command -v g++); then
    echo "Error: g++ not found in PATH" >&2
    exit 1
fi
export CC CXX

echo "Using CC=$CC, CXX=$CXX"
pip install scikit-build-core
cd "$SCRIPT_DIR/optest"
bash build.sh

# 3. 安装 optest（从 dist）
echo ""
echo "============================================"
echo "Step 3: Installing optest from dist..."
echo "============================================"
cd "$SCRIPT_DIR/optest"
WHEEL_FILE=$(ls -1 dist/*.whl | head -1)
echo "Installing: $WHEEL_FILE"
pip install "$WHEEL_FILE" --force-reinstall --no-deps

# 4. 运行测试
echo ""
echo "============================================"
echo "Step 4: Cleaning runtime caches before run..."
echo "============================================"
clear_caches "optest"

echo ""
echo "============================================"
echo "Step 5: Running tests..."
echo "============================================"
# JIT 日志全开：0=None, 1=Info, 2=Debug（详细 compile/cache/mem hit 等）
export CATLASS_JIT_LOG_LEVEL=2
cd "$SCRIPT_DIR/optest"

# 用例名在 examples/ 与 tests/ 之间可能不一致（如 examples/55_ascend950_mx_grouped_matmul_slice_m
# 对应 tests/test_55_mx_grouped_matmul_slice_m.py），因此只按数字前缀匹配 tests/test_NN_*.py；
# 没有对应 optest 的用例（如 35/36/61、102~106）自动忽略。
if [ "$#" -gt 0 ]; then
    case_files=()
    for case_name in "$@"; do
        for f in "tests/test_${case_name%%_*}_"*.py; do
            if [ -e "$f" ]; then
                case_files+=("$f")
            fi
        done
    done

    if [ "${#case_files[@]}" -eq 0 ]; then
        echo "WARN: no optest case for: $*"
    else
        echo "Running optest cases: ${case_files[*]}"
        python3 -m pytest "${case_files[@]}" -v
    fi
else
    echo "Running all optest cases"
    python3 -m pytest tests/ -v
fi

# 6. 卸载 optest
echo ""
echo "============================================"
echo "Step 6: Uninstalling optest..."
echo "============================================"
pip uninstall torch-catlass -y

# 7. 收尾清理
echo ""
echo "============================================"
echo "Step 7: Cleaning runtime caches after run..."
echo "============================================"
clear_caches "optest"

echo ""
echo "============================================"
echo "All steps completed successfully!"
echo "============================================"