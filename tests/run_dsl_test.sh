#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# End-to-end validation for python/tla_dsl/examples/end_to_end/basic_mmad (basic_matmul*.py, basic_mmad_ptr.py,
# basic_matmul_l0c2l1.py),
# python/tla_dsl/examples/end_to_end/basic_vadd (basic_vadd.py),
# python/tla_dsl/examples/end_to_end/extern_op (extern_vecadd.py, extern_dual_core.py),
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed.py), python/tla_dsl/examples/end_to_end/basic_mixed_mutex (basic_mixed_mutex.py) and
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed_ub2l1.py, basic_mixed_store_zN.py,
# basic_mixed_store_zNUnAlign.py, basic_mixed_fixpipe_nz2dn.py).
# python/tla_dsl/examples/end_to_end/vector_ops (binary_op.py, masked_binary.py,
# bitwise_ops.py, reduction_ops.py, compare_mask.py, unary_ops.py, arange_op.py,
# interleave_op.py, mask_interleave_op.py, load_dintlv_op.py, load_store_mask.py, squeeze_op.py,
# register_control_flow.py, load_and_store_scalar_after_reduction.py, load_us_b8_op.py,
# cast_multi.py, gather_op.py, shift_op.py).
# python/tla_dsl/examples/end_to_end/tensor_index (scalar_index_control_flow.py,
# scalar_kernel_arg.py).
# python/tla_dsl/examples/end_to_end/debug_print (debug_print.py, debug_print_mixed.py,
# debug_print_format.py).
# python/tla_dsl/examples/end_to_end/scalar_arg_alignment (scalar_arg_alignment.py).
# python/tla_dsl/examples/end_to_end/dataclass_arg (dataclass_arg.py: stdlib
# @dataclass instance unpacked into scalar kernel args).
# python/tla_dsl/examples/end_to_end/structured_arguments (recursive arguments,
# Tensor interfaces, scalar subclasses and mixed memref ABI).
# python/tla_dsl/examples/end_to_end/print_tensor (print_tensor.py: GM/UB/L1/L0C
# tensor dumps plus multi-block and multi-call cases).
# python/tla_dsl/examples/end_to_end/basic_mmad_epilogue (matmul_add.py, matmul_add_ub.py,
# matmul_bias.py, matmul_leaky_relu.py, matmul_sigmoid.py, matmul_silu.py, matmul_tanh.py).
# python/tla_dsl/examples/end_to_end/flash_attention_infer (flash_attention_infer.py).
# python/tla_dsl/examples/end_to_end/lazy_conditions (lazy_conditions.py).
# python/tla_dsl/examples/end_to_end/constexpr_callable (constexpr_callable_epilogue.py).
# python/tla_dsl/examples/end_to_end/jit_callable (jit_callable_epilogue.py).
# python/tla_dsl/examples/end_to_end/simt (basic_vadd_simt.py).
# python/tla_dsl/examples/end_to_end/multi_core_splitk_matmul (multi_core_splitk_matmul.py,
# tail_multi_core_splitk_matmul.py).
# python/tla_dsl/examples/end_to_end/basic_mmad_streamk (basic_mmad_streamk.py).
# python/tla_dsl/examples/end_to_end/batched_matmul (batched_matmul.py).
# python/tla_dsl/examples/end_to_end/grouped_matmul_slice_m (grouped_matmul_slice_m.py).
#
# Toolchain paths:
#   CANN:             ASCEND_HOME_PATH (source set_env.sh if not already in env)
#                     → WORKSPACE_ROOT/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh
#   AscendNPU-IR:     CATLASS_DSL_PREBUILT_ASCENDNPU_IR
#                     with repository submodule fallback
#   TLA DSL:          CATLASS_DSL_DIR → CATDSL_ROOT/python/tla_dsl
#
# CANN 9.1+ ships hivmc-a5 in toolkit; no separate HIVMC sibling is required.
#
# Usage:
#   bash tests/run_dsl_test.sh
#   bash tests/run_dsl_test.sh --device 0
#   CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR bash tests/run_dsl_test.sh --device 0

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CATDSL_ROOT="$(realpath "${SCRIPT_PATH}/..")"
WORKSPACE_ROOT="${ASCEND_CATLASS_DSL_ROOT:-$(dirname "${CATDSL_ROOT}")}"

CATLASS_DSL_DIR="${CATLASS_DSL_DIR:-${CATDSL_ROOT}/python/tla_dsl}"

_resolve_cann_set_env_sh() {
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        printf '%s\n' "${ASCEND_HOME_PATH}/set_env.sh"
        return 0
    fi
    local fallback="${WORKSPACE_ROOT}/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh"
    if [[ -f "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    return 1
}

DEVICE_ID="${DEVICE_ID:-1}"
export CATLASS_DSL_FORCE_RECOMPILE="${CATLASS_DSL_FORCE_RECOMPILE:-1}"
tla_dsl_wheel_path=""
tla_dsl_wheel_site=""

BASIC_MMAD_REL="examples/end_to_end/basic_mmad/basic_matmul.py"
BASIC_MMAD_AUTO_SYNC_REL="examples/end_to_end/basic_mmad/basic_matmul_auto_sync.py"
BASIC_MMAD_PTR_REL="examples/end_to_end/basic_mmad/basic_mmad_ptr.py"
BASIC_MMAD_L0C2L1_REL="examples/end_to_end/basic_mmad/basic_matmul_l0c2l1.py"
BASIC_VADD_REL="examples/end_to_end/basic_vadd/basic_vadd.py"
EXTERN_VECADD_REL="examples/end_to_end/extern_op/extern_vecadd.py"
EXTERN_DUAL_CORE_REL="examples/end_to_end/extern_op/extern_dual_core.py"
BASIC_MIXED_REL="examples/end_to_end/basic_mixed/basic_mixed.py"
BASIC_MIXED_MUTEX_REL="examples/end_to_end/basic_mixed/basic_mixed_mutex.py"
BASIC_MIXED_UB2L1_REL="examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py"
BASIC_MIXED_STORE_ZN_REL="examples/end_to_end/basic_mixed/basic_mixed_store_zN.py"
BASIC_MIXED_STORE_ZNUNALIGN_REL="examples/end_to_end/basic_mixed/basic_mixed_store_zNUnAlign.py"
BASIC_MIXED_FIXPIPE_NZ2DN_REL="examples/end_to_end/basic_mixed/basic_mixed_fixpipe_nz2dn.py"
MASKED_BINARY_REL="examples/end_to_end/vector_ops/masked_binary.py"
BITWISE_OPS_REL="examples/end_to_end/vector_ops/bitwise_ops.py"
BINARY_OP_REL="examples/end_to_end/vector_ops/binary_op.py"
REDUCTION_OPS_REL="examples/end_to_end/vector_ops/reduction_ops.py"
LOAD_STORE_SCALAR_AFTER_REDUCTION_REL="examples/end_to_end/vector_ops/load_and_store_scalar_after_reduction.py"
COMPARE_MASK_REL="examples/end_to_end/vector_ops/compare_mask.py"
COMPARE_MASK_OPS=(
    vector_vector_lt vector_vector_le vector_vector_gt vector_vector_ge vector_vector_eq vector_vector_ne
    vector_scalar_gt vector_scalar_ge
    masked_vector_vector_lt cmp_masked_fused static_dynamic_lt
)
UNARY_OPS_REL="examples/end_to_end/vector_ops/unary_ops.py"
ARANGE_OP_REL="examples/end_to_end/vector_ops/arange_op.py"
INTERLEAVE_OP_REL="examples/end_to_end/vector_ops/interleave_op.py"
MASK_INTERLEAVE_OP_REL="examples/end_to_end/vector_ops/mask_interleave_op.py"
LOAD_DINTLV_OP_REL="examples/end_to_end/vector_ops/load_dintlv_op.py"
LOAD_US_B8_OP_REL="examples/end_to_end/vector_ops/load_us_b8_op.py"
LOAD_DIST_OPS_REL="examples/end_to_end/vector_ops/load_dist_ops.py"
LOAD_STORE_MASK_REL="examples/end_to_end/vector_ops/load_store_mask.py"
STORE_PACK_REL="examples/end_to_end/vector_ops/store_pack.py"
STORE_FIRST_ELEMENT_OP_REL="examples/end_to_end/vector_ops/store_first_element_op.py"
SQUEEZE_OP_REL="examples/end_to_end/vector_ops/squeeze_op.py"
REGISTER_CONTROL_FLOW_REL="examples/end_to_end/vector_ops/register_control_flow.py"
SCALAR_INDEX_CONTROL_FLOW_REL="examples/end_to_end/tensor_index/scalar_index_control_flow.py"
SCALAR_KERNEL_ARG_REL="examples/end_to_end/tensor_index/scalar_kernel_arg.py"
DEBUG_PRINT_REL="examples/end_to_end/debug_print/debug_print.py"
DEBUG_PRINT_MIXED_REL="examples/end_to_end/debug_print/debug_print_mixed.py"
DEBUG_PRINT_FORMAT_REL="examples/end_to_end/debug_print/debug_print_format.py"
SCALAR_ARG_ALIGNMENT_REL="examples/end_to_end/scalar_arg_alignment/scalar_arg_alignment.py"
PRINT_TENSOR_REL="examples/end_to_end/print_tensor/print_tensor.py"
STREAMK_OPS_REL="examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py"
GROUPED_MATMUL_SLICEM_REL="examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py"
BATCHED_MATMUL_REL="examples/end_to_end/batched_matmul/batched_matmul.py"
FLASH_ATTENTION_INFER_REL="examples/end_to_end/flash_attention_infer/flash_attention_infer.py"
LAZY_CONDITIONS_REL="examples/end_to_end/lazy_conditions/lazy_conditions.py"
CONSTEXPR_CALLABLE_REL="examples/end_to_end/constexpr_callable/constexpr_callable_epilogue.py"
JIT_CALLABLE_REL="examples/end_to_end/jit_callable/jit_callable_epilogue.py"
SIMT_VADD_REL="examples/end_to_end/simt/basic_vadd_simt.py"
MULTI_CORE_SPLITK_REL="examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py"
TAIL_MULTI_CORE_SPLITK_REL="examples/end_to_end/multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py"
BASIC_MMAD_EPILOGUE_ADD_REL="examples/end_to_end/basic_mmad_epilogue/matmul_add.py"
BASIC_MMAD_EPILOGUE_ADD_UB_REL="examples/end_to_end/basic_mmad_epilogue/matmul_add_ub.py"
BASIC_MMAD_EPILOGUE_BIAS_REL="examples/end_to_end/basic_mmad_epilogue/matmul_bias.py"
BASIC_MMAD_EPILOGUE_LEAKY_RELU_REL="examples/end_to_end/basic_mmad_epilogue/matmul_leaky_relu.py"
BASIC_MMAD_EPILOGUE_SIGMOID_REL="examples/end_to_end/basic_mmad_epilogue/matmul_sigmoid.py"
BASIC_MMAD_EPILOGUE_SILU_REL="examples/end_to_end/basic_mmad_epilogue/matmul_silu.py"
BASIC_MMAD_EPILOGUE_TANH_REL="examples/end_to_end/basic_mmad_epilogue/matmul_tanh.py"
CAST_MULTI_REL="examples/end_to_end/vector_ops/cast_multi.py"
GATHER_OP_REL="examples/end_to_end/vector_ops/gather_op.py"
SHIFT_OP_REL="examples/end_to_end/vector_ops/shift_op.py"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Run end-to-end validation for:
  - basic_mmad (full flag-sync mnk/layout/dtype matrix; representative manual and automatic mutex cases;
    atomic-add coverage for each supported input dtype)
  - basic_mmad_ptr (basic_mmad_ptr.py)
  - basic_mmad_l0c2l1 (basic_matmul_l0c2l1.py: cube-only E=(A@B)@D with L0C->L1 staging)
  - basic_vadd (basic_vadd.py with per-dtype CLI invocations, plus mutex variants)
  - extern_op (extern_vecadd.py and extern_dual_core.py)
  - basic_mixed (basic_mixed.py with dynamic GM mnk list, including --use-mutex; basic_mixed_ub2l1.py,
    basic_mixed_store_zN.py, basic_mixed_store_zNUnAlign.py for m=64/m=50)
  - binary_op (binary_op.py <op> --all-dtypes for add/sub/mul/div/max/min/add_unalign/add_brc_b32)
  - masked_binary (masked_binary.py masked_binary --all-dtypes)
  - bitwise_ops (bitwise_ops.py bitwise_ops --all-dtypes)
  - reduction_ops (reduction_ops.py <op> --run for add/max/min)
  - load_and_store_scalar_after_reduction (UB scalar load/store in tla.vector and outlined tla.vec.func)
  - compare_mask (compare_mask.py <op> --all-dtypes for each compare-mask op)
  - unary_ops (unary_ops.py <op> --all-dtypes for exp/log/sqrt/abs/neg/masked_unary/masked_abs/masked_neg)
  - arange_op (arange_op.py [increase/decrease] --all-dtypes)
  - interleave_op (interleave_op.py interleave/deinterleave --all-dtypes)
  - mask_interleave_op (mask_interleave_op.py mask_interleave/mask_deinterleave
    --all-dtypes --shape 512; i8/f16/f32, both outputs checked against CPU golden)
  - load_dintlv_op (load_dintlv_op.py dintlv_b32 --all-dtypes; f32 only)
  - load_us_b8_op (load_us_b8_op.py us_b8 --sweep --shapes 512; i8 only:
    DIST_US_B8 2x up-sample load of b8 elements)
  - load_dist_ops (load_dist_ops.py <op> --sweep --shapes 512; one script for
    the issue-519 load distribution modes, op in brc_b16/us_b16/unpack_b16/
    e2b_b16/e2b_b32/blk:
    brc_b16 broadcasts one b16 element to a VL register (i16 only),
    us_b16 2x up-samples b16 elements (i16 only),
    unpack_b16 unpacks b16 elements with a zero interleaved after each
    (i16 only),
    e2b_b16/e2b_b32 broadcast one element per 32-byte DataBlock (i16/i32
    only),
    blk replicates one 32-byte DataBlock across all 8 DBs (i16 smoke only;
    any element size works))
  - load_store_mask (load_store_mask.py load_store_mask --all-dtypes:
    MaskSSA load/store round-trip via MaskLoadParams/MaskStoreParams for
    b8/b16/b32 UB carriers; companion vector fixed to f32)
  - store_pack (store_pack.py store_pack --all-dtypes; i32/i16 only)
  - store_first_element (store_first_element_op.py store_first_element
    --all-dtypes; i32/i16/i8/f32/f16 -> DIST_FIRST_ELEMENT_B32/B16/B8 lane-0
    store, one element per VL tile written to the 32B-aligned tile base)
  - squeeze_op (squeeze_op.py squeeze --all-dtypes)
  - register_control_flow (register_control_flow.py register_carriers:
    mixed VectorSSA/MaskSSA scf.for carriers and masked store)
  - scalar_index_control_flow (scalar_index_control_flow.py: GM scalar read/write,
    loop/dynamic-if/constexpr-if, vec.func, AST Numeric / index-vs-Int32 compare)
  - scalar_kernel_arg (scalar_kernel_arg.py: host Numeric kernel args used in
    same-type scalar arithmetic)
  - debug_print (all direct scalar dtypes, supported computed values, and two-block prints on AIV and AIC)
  - debug_print_mixed (all scalar dtypes in cube-only, vector-only, and combined regions)
  - debug_print_format (formatted multicall and multiblock prints on AIV and AIC)
  - scalar_arg_alignment (scalar_arg_alignment.py: tensor-i16-tensor host ABI)
  - dataclass_arg (dataclass_arg.py: stdlib @dataclass instance unpacked into
    per-field scalar kernel args; constexpr + tensor + scalar fields)
  - print_tensor (print_tensor.py: supported GM/UB/L1/L0C dumps with AIV/AIC
    multi-block and multi-call coverage)
  - basic_mmad_epilogue (matmul_add.py, ...: CV fused examples)
  - flash_attention_infer (flash_attention_infer.py)
  - lazy_conditions (lazy_conditions.py)
  - constexpr_callable (constexpr_callable_epilogue.py: Constexpr Callable epilogue)
  - jit_callable (jit_callable_epilogue.py: @tla.jit helper as Constexpr epilogue)
  - simt (basic_vadd_simt.py: SIMT thread-block add)
  - multi_core_splitk_matmul (multi_core_splitk_matmul.py: using split-k strategy for workload balancing)
  - basic_mmad_streamk (basic_mmad_streamk.py: streamK workload balancing)
  - batched_matmul (batched_matmul.py)
  - grouped_matmul_slice_m (grouped_matmul_slice_m.py)
  - cast_multi (cast_multi.py)
  - gather_op (gather_op.py)
  - shift_op (shift_op.py <op> --all-dtypes for shift_left/shift_right/shift_lefts/shift_rights)
Runs the basic_mmad flag-sync matrix with irregular CLI shapes (333×444×555
and 1×2×3); representative mutex and atomic-add cases use 333×444×555.
Example defaults remain regular (256×512×1024).
Uses the current Python environment, sources CANN set_env.sh, builds the DSL,
then runs the test battery.

Options:
  -h, --help              Show this help
  --device ID             NPU device id (default: ${DEVICE_ID})

Paths (auto from script location):
  WORKSPACE_ROOT=${WORKSPACE_ROOT}   (override: ASCEND_CATLASS_DSL_ROOT)
  CATDSL_ROOT=${CATDSL_ROOT}
  CATLASS_DSL_DIR=${CATLASS_DSL_DIR}
  Python: $(command -v python 2>/dev/null || echo '<not found>')

Toolchain:
  ASCEND_HOME_PATH               current: ${ASCEND_HOME_PATH:-<unset>}
    resolve: ASCEND_HOME_PATH/set_env.sh
             → WORKSPACE_ROOT/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh
    note: sourcing CANN set_env.sh sets ASCEND_HOME_PATH automatically
  CATLASS_DSL_PREBUILT_ASCENDNPU_IR  current: ${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-<unset>}

Example:
  bash ${SCRIPT_PATH}/run_dsl_test.sh
  bash ${SCRIPT_PATH}/run_dsl_test.sh --device 0
  CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR \\
    bash ${SCRIPT_PATH}/run_dsl_test.sh --device 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --device)
            shift
            DEVICE_ID="${1:?--device requires an argument}"
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done


_export_cann_build_env() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        local toolkit_dir
        toolkit_dir="$(dirname "${_cann_set_env_sh}")"
        if [[ -d "${toolkit_dir}/latest" ]]; then
            ASCEND_HOME_PATH="$(realpath "${toolkit_dir}/latest")"
        else
            ASCEND_HOME_PATH="$(realpath "${toolkit_dir}")"
        fi
        export ASCEND_HOME_PATH
    fi
    export BISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH:-${ASCEND_HOME_PATH}/bin}"
    echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
    echo "    BISHENG_COMPILER_PATH=${BISHENG_COMPILER_PATH}"
}

_export_toolchain_env() {
    echo "==> Exporting toolchain env"
    echo "    WORKSPACE_ROOT=${WORKSPACE_ROOT}"

}

_prepare_tla_dsl() {
    echo "==> Building TLA DSL"
    if [[ -f "${CATDSL_ROOT}/.gitmodules" ]]; then
        (
            cd "${CATDSL_ROOT}"
            git submodule update --init --depth 1 3rdparty/googletest 2>/dev/null || true
        )
    fi

    echo "==> ./build.sh --release (under ${CATLASS_DSL_DIR})"
    (
        cd "${CATLASS_DSL_DIR}"
        ./build.sh --release
    )

    # Locate the wheel produced by the Release build.
    local wheel
    wheel="$(find "${CATLASS_DSL_DIR}/dist" -maxdepth 1 -type f \
        -name 'ascend_catlass_dsl-*.whl' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "${wheel}" || ! -f "${wheel}" ]]; then
        echo "error: Release wheel was not produced under ${CATLASS_DSL_DIR}/dist" >&2
        exit 1
    fi

    tla_dsl_wheel_path="${wheel}"
    tla_dsl_wheel_site="$(mktemp -d "${TMPDIR:-/tmp}/tla-dsl-wheel.XXXXXX")"
    echo "==> Installing Release wheel into ${tla_dsl_wheel_site}"
    python -m pip install --no-deps --upgrade --target "${tla_dsl_wheel_site}" "${tla_dsl_wheel_path}"

    local package_path
    package_path="$(PYTHONPATH="${tla_dsl_wheel_site}:${PYTHONPATH:-}" python -c 'import catlass; print(catlass.__file__)')"
    if [[ "${package_path}" != "${tla_dsl_wheel_site}"/* ]]; then
        echo "error: wheel package was not selected (resolved ${package_path})" >&2
        exit 1
    fi
    echo "==> Testing wheel: ${tla_dsl_wheel_path}"
    echo "    imported catlass from ${package_path}"
}

# --- main ---

if [[ ! -d "${CATLASS_DSL_DIR}" ]]; then
    echo "error: CATLASS_DSL_DIR does not exist: ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/build.sh" ]]; then
    echo "error: missing build.sh under CATLASS_DSL_DIR=${CATLASS_DSL_DIR}" >&2
    exit 1
fi

if ! _cann_set_env_sh="$(_resolve_cann_set_env_sh)"; then
    echo "error: CANN set_env.sh not found." >&2
    echo "       Set ASCEND_HOME_PATH to your CANN toolkit root (with set_env.sh)," >&2
    echo "       or source CANN set_env.sh before running this script," >&2
    echo "       or place CANN at ${WORKSPACE_ROOT}/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh" >&2
    exit 1
fi
echo "==> Sourcing CANN: ${_cann_set_env_sh}"
# shellcheck disable=SC1090
source "${_cann_set_env_sh}"

echo "==> Using Python: $(command -v python) ($(python -V 2>&1))"

echo "==> Exporting CANN build env"
_export_cann_build_env

_export_toolchain_env

echo "==> Using CATLASS_DSL_DIR=${CATLASS_DSL_DIR}"

trap 'if [[ -n "${tla_dsl_wheel_site}" && -d "${tla_dsl_wheel_site}" ]]; then rm -rf "${tla_dsl_wheel_site}"; fi; rm -rf "${HOME}/.cache/catlass/bc"' EXIT
_prepare_tla_dsl

if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_AUTO_SYNC_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_AUTO_SYNC_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_PTR_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_PTR_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_L0C2L1_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_L0C2L1_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_VADD_REL}" ]]; then
    echo "error: missing ${BASIC_VADD_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${EXTERN_VECADD_REL}" ]]; then
    echo "error: missing ${EXTERN_VECADD_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${EXTERN_DUAL_CORE_REL}" ]]; then
    echo "error: missing ${EXTERN_DUAL_CORE_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_MUTEX_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_MUTEX_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_UB2L1_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_UB2L1_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_STORE_ZN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_STORE_ZN_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_STORE_ZNUNALIGN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_STORE_ZNUNALIGN_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MIXED_FIXPIPE_NZ2DN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_FIXPIPE_NZ2DN_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${MASKED_BINARY_REL}" ]]; then
    echo "error: missing ${MASKED_BINARY_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BITWISE_OPS_REL}" ]]; then
    echo "error: missing ${BITWISE_OPS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BINARY_OP_REL}" ]]; then
    echo "error: missing ${BINARY_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${REDUCTION_OPS_REL}" ]]; then
    echo "error: missing ${REDUCTION_OPS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${LOAD_STORE_SCALAR_AFTER_REDUCTION_REL}" ]]; then
    echo "error: missing ${LOAD_STORE_SCALAR_AFTER_REDUCTION_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${COMPARE_MASK_REL}" ]]; then
    echo "error: missing ${COMPARE_MASK_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${MASK_INTERLEAVE_OP_REL}" ]]; then
    echo "error: missing ${MASK_INTERLEAVE_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${LOAD_DIST_OPS_REL}" ]]; then
    echo "error: missing ${LOAD_DIST_OPS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${STORE_FIRST_ELEMENT_OP_REL}" ]]; then
    echo "error: missing ${STORE_FIRST_ELEMENT_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${UNARY_OPS_REL}" ]]; then
    echo "error: missing ${UNARY_OPS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${ARANGE_OP_REL}" ]]; then
    echo "error: missing ${ARANGE_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${SQUEEZE_OP_REL}" ]]; then
    echo "error: missing ${SQUEEZE_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${REGISTER_CONTROL_FLOW_REL}" ]]; then
    echo "error: missing ${REGISTER_CONTROL_FLOW_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${SCALAR_INDEX_CONTROL_FLOW_REL}" ]]; then
    echo "error: missing ${SCALAR_INDEX_CONTROL_FLOW_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${DEBUG_PRINT_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${DEBUG_PRINT_MIXED_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_MIXED_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${DEBUG_PRINT_FORMAT_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_FORMAT_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${SCALAR_ARG_ALIGNMENT_REL}" ]]; then
    echo "error: missing ${SCALAR_ARG_ALIGNMENT_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${PRINT_TENSOR_REL}" ]]; then
    echo "error: missing ${PRINT_TENSOR_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${FLASH_ATTENTION_INFER_REL}" ]]; then
    echo "error: missing ${FLASH_ATTENTION_INFER_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${LAZY_CONDITIONS_REL}" ]]; then
    echo "error: missing ${LAZY_CONDITIONS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${CONSTEXPR_CALLABLE_REL}" ]]; then
    echo "error: missing ${CONSTEXPR_CALLABLE_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${JIT_CALLABLE_REL}" ]]; then
    echo "error: missing ${JIT_CALLABLE_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${SIMT_VADD_REL}" ]]; then
    echo "error: missing ${SIMT_VADD_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${MULTI_CORE_SPLITK_REL}" ]]; then
    echo "error: missing ${MULTI_CORE_SPLITK_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${TAIL_MULTI_CORE_SPLITK_REL}" ]]; then
    echo "error: missing ${TAIL_MULTI_CORE_SPLITK_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${STREAMK_OPS_REL}" ]]; then
    echo "error: missing ${STREAMK_OPS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BATCHED_MATMUL_REL}" ]]; then
    echo "error: missing ${BATCHED_MATMUL_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${GROUPED_MATMUL_SLICEM_REL}" ]]; then
    echo "error: missing ${GROUPED_MATMUL_SLICEM_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_ADD_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_ADD_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_ADD_UB_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_ADD_UB_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_BIAS_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_BIAS_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_LEAKY_RELU_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_LEAKY_RELU_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_SIGMOID_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_SIGMOID_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_SILU_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_SILU_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${BASIC_MMAD_EPILOGUE_TANH_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_EPILOGUE_TANH_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${CAST_MULTI_REL}" ]]; then
    echo "error: missing ${CAST_MULTI_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${GATHER_OP_REL}" ]]; then
    echo "error: missing ${GATHER_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CATLASS_DSL_DIR}/${SHIFT_OP_REL}" ]]; then
    echo "error: missing ${SHIFT_OP_REL} under ${CATLASS_DSL_DIR}" >&2
    exit 1
fi

# --- run the battery -----------------------------------------------------------
#
# Every case runs in ONE process. Spawning a `python examples/...` per case makes
# the battery pay the torch_npu import and the device bring-up ~112 times over,
# which dwarfs the cases themselves: on an Ascend950PR box a basic_mmad case
# costs ~23 s as its own process but ~1 s as an extra case in a process that is
# already up.
#
# The case list lives in tests/dsl_battery/test_battery.py as pytest parameters;
# that file is the authoritative list of what the gate covers.
#
# Do NOT add -n/xdist here: that puts each case back in its own process and
# throws the saving away. Several cases are order-dependent too.

echo "==> Pre-compiling BC stubs"
PYTHONPATH="${tla_dsl_wheel_site}:${PYTHONPATH:-}" \
    python -m catlass.bc_compile

echo "==> Running the DSL battery (tests/dsl_battery)"
(
    cd "${CATDSL_ROOT}"
    TLA_DSL_RUN_BATTERY=1 PYTHONPATH="${tla_dsl_wheel_site}:${PYTHONPATH:-}" \
    python -m pytest tests/dsl_battery \
        --run-battery --device "${DEVICE_ID}" \
        -p no:randomly -v
)

echo "==> run_dsl_test.sh finished successfully"
