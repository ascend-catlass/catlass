"""The end-to-end DSL validation battery -- the RDV gate, as pytest cases.

Every case replays an example's own main() with a given argv, inside the one
process, so the device bring-up and the torch_npu import are paid once. The
examples are not modified in any way; a case is just an argv.

Adding coverage means adding to the tables below. Keep them as data: the point of
this file is that the whole gate is readable in one place.
"""

from __future__ import annotations

from typing import Iterator

import pytest

pytestmark = pytest.mark.npu

MMAD_SHAPES = (("333", "444", "555"), ("1", "2", "3"))
BIG_SHAPES = (("3200", "4096", "257"), ("8192", "2906", "1027"))
BATCH_SHAPES = (("1", "333", "444", "555"), ("1", "1", "2", "3"), ("8", "320", "333", "444"))
MMAD_LAYOUTS = (("row", "row"), ("row", "col"), ("col", "row"), ("col", "col"))
#: (dtype-a/b, dtype-c)
MMAD_TRIPLES = (
    ("f16", "f32"),
    ("f16", "f16"),
    ("bf16", "f32"),
    ("bf16", "bf16"),
    ("f32", "f32"),
)
MMAD_MNK = ("--m", "333", "--n", "444", "--k", "555")
MMAD_ROW_ROW = ("--layout-a", "row", "--layout-b", "row")

VADD_DTYPES = ("i8", "i16", "i32", "f16", "f32")
VADD_MODES = ("", "--use-mutex", "--use-mutex-with", "--use-atomic-add")

COMPARE_MASK_OPS = (
    "vector_vector_lt", "vector_vector_le", "vector_vector_gt", "vector_vector_ge",
    "vector_vector_eq", "vector_vector_ne", "vector_scalar_gt", "vector_scalar_ge",
    "masked_vector_vector_lt", "cmp_masked_fused", "static_dynamic_lt",
)

#: (block-num, calls, dtype-args)
PRINT_TENSOR_VARIANTS = (
    ("1", "1", ("--all-dtypes",)),
    ("2", "1", ("--dtype", "f32")),
    ("1", "2", ("--dtype", "f32")),
    ("2", "2", ("--all-dtypes",)),
)
#: (case, block-num, calls, dtype-args)
PRINT_TENSOR_UB_VARIANTS = (
    ("base", "1", "1", ("--all-dtypes",)),
    ("base", "2", "1", ("--dtype", "f32")),
    ("base", "1", "2", ("--dtype", "f32")),
    ("base", "2", "2", ("--all-dtypes",)),
    ("aligned-offset", "1", "1", ("--all-dtypes",)),
)
LOCAL_PRINT_CASE_ID = "print-tensor-local-memory"

EVG_OPS = (
    "add",
    "add_ub",
    "bias",
    "leaky_relu",
    "sigmoid",
    "silu",
    "tanh",
)

def _mmad_cases(
    op_name: str,
    op_script: str,
    device: int,
    **kwargs,
) -> Iterator[tuple[str, list[list[str]]]]:
    """Matmul-like cases generator
    
    Iterate over all shapes, layouts, dtype triples and relu_enable
    """
    for m, n, k in kwargs.get("shapes", MMAD_SHAPES):
        for la, lb in kwargs.get("layouts", MMAD_LAYOUTS):
            for dab, dc in kwargs.get("triples", MMAD_TRIPLES):
                for relu_enable in kwargs.get("relu_enable", (0, 1)):
                    yield (
                        f"{op_name}-{m}x{n}x{k}-{la}{lb}-{dab}-{dc}-relu{relu_enable}",
                        [[
                            op_script,
                            "--m", m, "--n", n, "--k", k,
                            "--layout-a", la, "--layout-b", lb,
                            "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                            "--relu-enable", str(relu_enable),
                            "--device", str(device),
                        ]]
                    )

def _cases(device: int) -> Iterator[tuple[str, list[list[str]]]]:
    """Yield (test id, [argv, ...]) for every case in the battery.

    A case is usually one argv. It is a *sequence* only where a later invocation
    depends on an earlier one -- see the basic_mixed compile-then-cache-reuse
    case, which must stay a single test so the two halves can never be split up
    or reordered.
    """

    dev = ["--device", str(device)]

    # --- basic_mmad: the flag-sync layout x dtype matrix ---
    yield from _mmad_cases(
        "mmad",
        "basic_mmad/basic_matmul.py",
        device,
    )

    # --- mx_mmad: the microscaling matmul coverage matrix ---
    #
    # Two axes, and both are real. The operand formats are the four MXFP8
    # pairings plus the two MXFP4 encodings -- the cube has a mad_mx for every
    # one, including mixed. The layout axis moves each operand and its scale
    # block together: row-major A and column-major B reach L1 through a
    # transposing DN2NZ, the other orientations through a plain ND2NZ, so
    # the two settings exercise genuinely different copies.
    #
    # The operands themselves have no layout axis: the GM -> L1 routes exist only
    # for a row-major A into zN and a column-major B into nZ.
    for case in (
        "f8e4m3fn,f8e4m3fn",
        "f8e5m2,f8e5m2",
        "f8e4m3fn,f8e5m2",
        "f8e5m2,f8e4m3fn",
        "f4e2m1",
        "f4e1m2",
    ):
        # fp8 takes all four orientations; packed fp4 needs K contiguous, which
        # only the row/col pairing gives it.
        layouts = (
            (("row", "col"),)
            if case.startswith("f4")
            else (("row", "row"), ("row", "col"), ("col", "row"), ("col", "col"))
        )
        for la, lb in layouts:
            yield (
                f"mx-mmad-{case.replace(',', '-')}-{la}{lb}",
                [[
                    "mx_mmad/mx_matmul.py", "--case", case,
                    "--layout-a", la, "--layout-b", lb, *dev,
                ]],
            )

    for script, label in (
        ("basic_matmul_mutex.py", "mutex"),
        ("basic_matmul_mutex_with.py", "mutex-with"),
        ("basic_matmul_auto_sync.py", "auto-sync"),
    ):
        yield (
            f"mmad-{label}",
            [[
                f"basic_mmad/{script}", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", "f16", "--dtype-b", "f16", "--dtype-c", "f32", *dev,
            ]],
        )

    for dtype in ("f16", "bf16", "f32"):
        yield (
            f"mmad-atomic-add-{dtype}",
            [[
                "basic_mmad/basic_matmul_atomic_add.py", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", dtype, "--dtype-b", dtype, "--dtype-c", "f32", *dev,
            ]],
        )

    yield ("mmad-ptr", [["basic_mmad/basic_mmad_ptr.py", *dev]])

    # basic_mmad_l0c2l1: cube-only E=(A@B)@D, L0C->L1 (fixpipe) staging reused
    # as the second mmad's LHS; single static tile (M=50/N=60/K=64).
    yield ("mmad-l0c2l1", [["basic_mmad/basic_matmul_l0c2l1.py", *dev]])

    # --- basic_vadd: every dtype under each sync mode ---
    for mode in VADD_MODES:
        for dtype in VADD_DTYPES:
            label = mode.lstrip("-") or "flag-sync"
            yield (
                f"vadd-{label}-{dtype}",
                [[
                    "basic_vadd/basic_vadd.py", "--dtype", dtype, *dev,
                    *([mode] if mode else []),
                ]],
            )

    # --- extern_op: user-provided Ascend C functions called from TLA kernels ---
    yield (
        "extern-vecadd",
        [["extern_op/extern_vecadd.py", *dev]],
    )
    yield (
        "extern-dual-core",
        [["extern_op/extern_dual_core.py", *dev]],
    )
    yield (
        "extern-multi-ops",
        [["extern_op/extern_multi_ops.py", *dev]],
    )
    yield (
        "extern-custom-include",
        [["extern_op/extern_custom_include.py", *dev]],
    )

    # --- basic_mixed ---
    # One test on purpose: the second invocation exercises cache reuse and is
    # only meaningful straight after the one that populates the cache.
    yield (
        "mixed-compile-then-cache-reuse",
        [
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
        ],
    )
    yield (
        "mixed-mutex",
        [["basic_mixed/basic_mixed_mutex.py", *dev, "--block-num", "1"]],
    )
    yield ("mixed-ub2l1", [["basic_mixed/basic_mixed_ub2l1.py", *dev]])
    yield ("mixed-store-zN", [["basic_mixed/basic_mixed_store_zN.py", *dev]])
    # m=64 is fractal-aligned (multiple of 16); m=50 exercises the zNUnAlign M
    # axis, where the dest leaf[0] is the runtime row count and the stride varies.
    for m in ("64", "50"):
        yield (
            f"mixed-store-zNUnAlign-m{m}",
            [["basic_mixed/basic_mixed_store_zNUnAlign.py", *dev, "--m", m]],
        )
    yield ("mixed-fixpipe-nz2dn", [["basic_mixed/basic_mixed_fixpipe_nz2dn.py", *dev]])

    # --- mixed-core handshake and standalone control-flow probes ---
    yield (
        "flash-attention-infer",
        [["flash_attention_infer/flash_attention_infer.py", *dev]],
    )
    yield (
        "lazy-conditions",
        [["lazy_conditions/lazy_conditions.py", *dev]],
    )
    # Constexpr Callable / @tla.jit helper as Constexpr epilogue (Phase-1).
    yield (
        "constexpr-callable",
        [["constexpr_callable/constexpr_callable_epilogue.py", *dev]],
    )
    yield (
        "jit-callable",
        [["jit_callable/jit_callable_epilogue.py", *dev]],
    )
    yield (
        "simt-basic-vadd",
        [["simt/basic_vadd_simt.py", "--block-num", "1", *dev]],
    )
    # The only battery case that puts a SIMT region inside a mixed AIC+AIV
    # kernel, and the only one that gives SIMT any UB at all: basic_vadd_simt
    # is pure AIV and reads straight from global memory. It also covers the
    # launch-sized UB region, whose buffer the SIMD and SIMT stages share.
    yield (
        "simt-mixed",
        [["simt/simt_mixed.py", *dev]],
    )
    # The op coverage for SIMT: arithmetic, abs/exp/log/sqrt, max/min/where,
    # over six dtypes -- including the bf16 round-trip through f32 and the
    # integer divide. simt-basic-vadd exercises a single addition.
    yield (
        "simt-multi-ops",
        [["simt/multi_ops_simt.py", *dev]],
    )

    # --- multi_core_splitk_matmul and tail_multi_core_splitk_matmul ---
    for script in ("multi_core_splitk_matmul.py", "tail_multi_core_splitk_matmul.py"):
        for m, n, k in MMAD_SHAPES + BIG_SHAPES:
            yield (
                f"{script}-{m}x{n}x{k}",
                [[f"multi_core_splitk_matmul/{script}", "--m", m, "--n", n, "--k", k, *dev]],
            )

    # --- basic_mmad_streamk: a streamK example for workload balance ---
    for m, n, k in MMAD_SHAPES + BIG_SHAPES:
        yield (
            f"mmad-streamk-{m}x{n}x{k}",
            [["basic_mmad_streamk/basic_mmad_streamk.py", "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- batched_matmul ---
    for b, m, n, k in BATCH_SHAPES:
        yield (
            f"batched-matmul-{b}x{m}x{n}x{k}",
            [["batched_matmul/batched_matmul.py", "--batch", b, "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- grouped_matmul_slice_m: grouped matmul example ---
    yield (
        "grouped-matmul-slice-m",
        [["grouped_matmul_slice_m/grouped_matmul_slice_m.py", *dev],
         ["grouped_matmul_slice_m/grouped_matmul_slice_m.py", 
            "--groups", "3", "--m", "768", "--n", "333", "--k", "333", *dev]]
    ) 

    # --- basic_mmad_epilogue: multiple epilogue examples ---
    for op in EVG_OPS:
        if op in ("add_ub", "tanh"):
            # f32 (dtype-c) only examples
            triples = (
                ("f16", "f32"),
                ("f32", "f32"),
            )
        else:
            triples = (
                ("f16", "f32"),
                ("f32", "f32"),
                ("f16", "f16"),
                ("bf16", "f16"),
            )
        for m, n, k in MMAD_SHAPES:
            for dab, dc in triples:
                yield (
                    f"mmad-epilogue-{op.replace('_', '-')}-{m}x{n}x{k}-{dab}-{dc}",
                    [[
                        f"basic_mmad_epilogue/matmul_{op}.py",
                        "--m", m, "--n", n, "--k", k,
                        "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                        *dev
                    ]],
                )

    # --- vector_ops: each of these sweeps or batches many kernels internally ---
    yield (
        "vector-masked-binary",
        [["vector_ops/masked_binary.py", "masked_binary", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-bitwise-ops",
        [["vector_ops/bitwise_ops.py", "bitwise_ops", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-binary-op",
        [[
            "vector_ops/binary_op.py", "--batch-run",
            "add", "sub", "mul", "div", "max", "min", "add_unalign", "add_brc_b32",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield ("vector-reduction-ops", [["vector_ops/reduction_ops.py", "--batch-run", *dev]])
    yield (
        "vector-load-store-scalar-after-reduction",
        [["vector_ops/load_and_store_scalar_after_reduction.py", *dev]],
    )
    yield (
        "vector-compare-mask",
        [[
            "vector_ops/compare_mask.py", "--batch-run", *COMPARE_MASK_OPS,
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-unary-ops",
        [[
            "vector_ops/unary_ops.py", "--batch-run",
            "exp", "log", "sqrt", "abs", "neg", "masked_unary", "masked_abs", "masked_neg",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-arange-op",
        [[
            "vector_ops/arange_op.py", "--batch-run", "increase", "decrease",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-interleave-op",
        [[
            "vector_ops/interleave_op.py", "--batch-run", "interleave", "deinterleave",
            "--shape", "512", "--batch-size", "4", *dev,
        ]],
    )
    # Predicate rearrangement: both outputs checked for b8/b16/b32.
    for op in ("mask_interleave", "mask_deinterleave"):
        yield (
            f"vector-{op.replace('_', '-')}-op",
            [[
                "vector_ops/mask_interleave_op.py", op,
                "--all-dtypes", "--shape", "512", *dev,
            ]],
        )
    yield (
        "vector-load-dintlv-op",
        [["vector_ops/load_dintlv_op.py", "dintlv_b32", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-us-b8-op",
        [["vector_ops/load_us_b8_op.py", "us_b8", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-dist-ops",
        [
            [
                "vector_ops/load_dist_ops.py",
                op,
                "--sweep",
                "--shapes",
                "512",
                *dev,
            ]
            for op in ("brc_b16", "us_b16", "e2b_b16", "e2b_b32", "blk", "unpack_b16")
        ],
    )
    yield (
        "vector-load-store-mask",
        [["vector_ops/load_store_mask.py", "load_store_mask", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-pack",
        [["vector_ops/store_pack.py", "store_pack", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-first-element",
        [["vector_ops/store_first_element_op.py", "store_first_element", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride",
        [["vector_ops/load_stride_op.py", "stride", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride-repeat",
        [["vector_ops/load_stride_op.py", "repeat", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride-post",
        [["vector_ops/load_stride_op.py", "post", "--all-dtypes", *dev]],
    )
    yield (
        "vector-squeeze-op",
        [["vector_ops/squeeze_op.py", "squeeze", "--sweep", "--shapes", "64", *dev]],
    )
    yield (
        "vector-register-control-flow",
        [["vector_ops/register_control_flow.py", "register_carriers", *dev]],
    )
    yield (
        "vector-cast-multi",
        [["vector_ops/cast_multi.py", "cast_multi", "--shape", "256", *dev]],
    )
    yield (
        "vector-gather",
        [["vector_ops/gather_op.py", "--run", *dev]],
    )
    yield (
        "vector-shift-op",
        [
            ["vector_ops/shift_op.py", shift_op, "--all-dtypes", *dev]
            for shift_op in ("shift_left", "shift_right", "shift_lefts", "shift_rights")
        ],
    )

    # --- tensor_index ---
    yield (
        "tensor-index-scalar-control-flow",
        [["tensor_index/scalar_index_control_flow.py", *dev]],
    )
    yield (
        "tensor-index-scalar-kernel-arg",
        [["tensor_index/scalar_kernel_arg.py", *dev]],
    )

    # --- print_tensor, GM storage ---
    for blocks, calls, dtype_args in PRINT_TENSOR_VARIANTS:
        yield (
            f"print-tensor-gm-b{blocks}-c{calls}",
            [[
                "print_tensor/print_tensor.py", "--run", *dev,
                "--block-num", blocks, "--calls", calls, *dtype_args,
            ]],
        )

    # --- debug_print ---
    yield (
        "debug-print-matrix",
        [[
            "debug_print/debug_print.py", "--run", *dev,
            "--all-dtypes", "--block-num", "2", "--expect-count", "2",
        ]],
    )
    for value in ("-0.0", "nan", "inf", "-inf"):
        yield (
            f"debug-print-f16-{value}",
            [[
                "debug_print/debug_print.py", "--run", *dev,
                "--dtype", "f16", f"--value={value}",
            ]],
        )
    yield (
        "debug-print-expression",
        [["debug_print/debug_print.py", "--run", *dev, "--all-dtypes", "--expression"]],
    )
    for region in ("cube", "vector", "both"):
        yield (
            f"debug-print-mixed-{region}",
            [[
                "debug_print/debug_print_mixed.py", "--run", *dev,
                "--all-dtypes", "--print-region", region,
            ]],
        )
    yield (
        "debug-print-format",
        [["debug_print/debug_print_format.py", "--run", *dev, "--block-num", "2"]],
    )

    # --- scalar_arg_alignment ---
    yield (
        "scalar-arg-alignment",
        [["scalar_arg_alignment/scalar_arg_alignment.py", *dev]],
    )

    # --- mixed Dynamic-GM and static memref arguments ---
    yield (
        "mixed-memref-arguments",
        [["mixed_memref_arguments/mixed_memref_arguments.py", *dev]],
    )

    # --- dataclass_arg (stdlib @dataclass unpacked into scalar kernel args) ---
    yield (
        "dataclass-arg",
        [["dataclass_arg/dataclass_arg.py", *dev]],
    )

    # --- print_tensor, UB storage ---
    for case_name, blocks, calls, dtype_args in PRINT_TENSOR_UB_VARIANTS:
        yield (
            f"print-tensor-ub-{case_name}-b{blocks}-c{calls}",
            [[
                "print_tensor/print_tensor.py", "--run", *dev, "--storage", "ub",
                "--case", case_name, "--block-num", blocks, "--calls", calls,
                *dtype_args,
            ]],
        )
    # --- structured arguments ---
    for name in (
        "mixed_memref_arguments",
        "scalar_subclass_arguments",
        "static_tensor_interfaces",
    ):
        yield (
            f"structured-args-{name}",
            [[f"structured_arguments/{name}.py", *dev]],
        )

    # --- print_tensor, local AIC storage ---
    # Base cases sweep dtypes; repeated entries verify in-process cache reuse.
    local_common = [
        "print_tensor/print_tensor.py", "--run", *dev,
        "--arch-scope", "aic.c310",
    ]
    l1_zn_base = [*local_common, "--storage", "l1", "--layout", "zn", "--all-dtypes"]
    l1_nz_base = [*local_common, "--storage", "l1", "--layout", "nz", "--all-dtypes"]
    l0c_base = [*local_common, "--storage", "l0c", "--all-dtypes"]
    yield (
        LOCAL_PRINT_CASE_ID,
        [
            l1_zn_base,
            list(l1_zn_base),
            l1_nz_base,
            l0c_base,
            list(l0c_base),
            [*local_common, "--storage", "l1", "--layout", "zn", "--case", "aligned-offset", "--dtype", "i8"],
            [*local_common, "--storage", "l1", "--layout", "nz", "--case", "aligned-offset", "--dtype", "f16"],
            [*local_common, "--storage", "l1", "--layout", "zn", "--calls", "2"],
            [
                *local_common,
                "--storage", "l1", "--layout", "zn",
                "--case", "block-dependent", "--dtype", "f32",
                "--block-num", "2",
            ],
            [
                *local_common,
                "--storage", "l0c",
                "--case", "larger-origin",
                "--all-dtypes",
            ],
        ],
    )


def battery_cases(device: int = 0) -> list[tuple[str, list[list[str]]]]:
    """The whole battery as data. Used for the tests, and to audit coverage."""

    return list(_cases(device))


def _ids() -> list[str]:
    return [case_id for case_id, _ in battery_cases()]


@pytest.mark.parametrize("case_id", _ids())
def test_case(case_id: str, device: int) -> None:
    import example_runner

    if case_id == LOCAL_PRINT_CASE_ID:
        from catlass.base_dsl.runtime import ascend_debug_fifo

        if reason := ascend_debug_fifo._debug_bus_support_error():
            pytest.skip(reason)

    argvs = dict(battery_cases(device))[case_id]
    for argv in argvs:
        script = example_runner.EXAMPLES_DIR / argv[0]
        assert script.is_file(), f"no such example: {argv[0]}"
        rc = example_runner.run_case([str(script), *argv[1:]])
        assert rc == 0, f"{argv[0]} {' '.join(argv[1:])} exited {rc}"
