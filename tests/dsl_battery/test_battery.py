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

# per-tensor and per-channel shares the same dtype map
FIXPIPE_QUANT_SRC_DST = (
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("f32", "i8"),
    ("f32", "u8"),
    ("i32", "f16"),
    ("i32", "bf16"),
    ("i32", "i8"),
    ("i32", "u8"),
)

#: The basic_mixed per-tensor example ships float destinations only.
MIXED_FIXPIPE_QUANT_SRC_DST = (
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("i32", "f16"),
    ("i32", "bf16"),
)

#: Shapes the mixed per-tensor example adds over basic_mmad: it walks the grid
#: of tiles from one kernel, so the tail tile is ragged on either axis and K can
#: be ragged too. MIXED_TILING is 64x64 at every level.
#: (label, m, n, k, extra-args)
MIXED_FIXPIPE_SHAPES = (
    ("tiles-row", "150", "100", "64", ("--layout-c", "row", "--block-num", "1")),
    ("tiles-col", "150", "100", "64", ("--layout-c", "col", "--block-num", "1")),
    ("ragged-col", "33", "65", "17", ("--layout-c", "col")),
    ("ragged-row-i32", "65", "33", "17", ("--layout-c", "row", "--dtype-ab", "i8")),
)

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

    # --- SYRK: aligned tiles, M/K tails, K boundary, single/multi-core stores ---
    for dtype in ("f16", "bf16"):
        for m, k in (("1", "1"), ("1024", "1024"), ("769", "129"), ("257", "4095")):
            for blocks in ("1", "-1"):
                yield (
                    f"syrk-{m}x{k}-{dtype}-b{blocks}",
                    [[
                        "tla/kernel/matmul/basic_syrk/basic_syrk.py", "--m", m, "--k", k,
                        "--dtype", dtype, "--block-num", blocks, *dev,
                    ]],
                )

    # --- SYRK: the UB-transpose mirror, the polynomial epilogue and split-K ---
    #
    # M=145 is a tail for every tile size and spans several tile rows, so the
    # mirror lands on off-diagonal tiles; the script's symmetry check is what
    # catches a mirror written to the wrong triangle. The polynomial cases carry
    # a symmetric residual so that check still holds, and the split-K cases pick
    # a K long enough for the planner to split it.
    for dtype in ("f16", "bf16"):
        syrk = ["tla/kernel/matmul/basic_syrk/basic_syrk.py", "--dtype", dtype, *dev]
        shape = ["--batch", "2", "--m", "145", "--k", "1025"]
        poly = ["--alpha", "2.0", "--diagonal", "3.0"]
        # Reusing L1 buffers after a one-tile prologue, then an uneven K split.
        yield (
            f"syrk-prefetch-single-k-{dtype}",
            [[*syrk, "--batch", "2", "--m", "145", "--k", "513", "--tile", "64", "--splitk", "2", "--block-num", "1"]],
        )
        yield (
            f"syrk-prefetch-split-tail-{dtype}",
            [[*syrk, "--batch", "2", "--m", "145", "--k", "1025", *poly,
              "--beta", "-1.0", "--tile", "64", "--splitk", "2", "--block-num", "1"]],
        )
        yield (f"syrk-batched-mirror-{dtype}", [[*syrk, *shape]])
        for tile in ("64", "128"):
            yield (
                f"syrk-batched-mirror-{dtype}-t{tile}",
                [[*syrk, *shape, "--tile", tile]],
            )
        yield (
            f"syrk-poly-splitk-tail-{dtype}",
            [[*syrk, *shape, *poly, "--beta", "-1.0", "--tile", "64", "--splitk", "2"]],
        )
        yield (f"syrk-poly-residual-{dtype}", [[*syrk, *shape, *poly, "--beta", "-1.0"]])
        yield (f"syrk-poly-no-residual-{dtype}", [[*syrk, *shape, *poly]])
        yield (
            f"syrk-splitk-{dtype}",
            [[*syrk, "--m", "256", "--k", "4096", "--splitk", "2"]],
        )
        yield (
            f"syrk-splitk-auto-{dtype}",
            [[*syrk, "--m", "64", "--k", "4096", "--tile", "0", "--splitk", "0"]],
        )

    # --- basic_mmad: the flag-sync layout x dtype matrix ---
    yield from _mmad_cases(
        "mmad",
        "tla/kernel/matmul/basic_mmad/basic_matmul.py",
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
        # A third axis: the shape. The regular one divides evenly and never
        # reaches a boundary tile, so every pairing runs again unaligned --
        # 333x444x535, where 535 is not a multiple of 32 and the last scale
        # group along k is short. Spelled out rather than left to the script's
        # defaults, which an edit there could change without touching this.
        for shape_label, shape_args in (
            ("regular", []),
            ("unaligned", ["--m", "333", "--n", "444", "--k", "535"]),
        ):
            for la, lb in layouts:
                yield (
                    f"mx-mmad-{case.replace(',', '-')}-{la}{lb}-{shape_label}",
                    [[
                        "tla/kernel/matmul/mx_mmad/mx_matmul.py", "--case", case,
                        "--layout-a", la, "--layout-b", lb,
                        *shape_args, *dev,
                    ]],
                )

    # --- mx_mmad with bias: the same kernel plus a 1 x N fp32 row that travels
    # GM -> L1 -> biasbuf once per block and is consumed by the first mmad_mx of
    # that block. The bias tile is VectorLayout, so these cases also exercise the flat
    # GM -> L1 / L1 -> biasbuf copy routes end to end.
    #
    # Deliberately fewer axes than the plain mx-mmad block above: the bias
    # pipeline does not depend on the operand format or on the scale-block
    # orientation, so one fp8 pair across all four layouts plus both fp4
    # encodings pins everything that can differ -- and still runs the unaligned
    # shape, where the last block's bias slice is shorter than the tile.
    for case, bias_layouts in (
        ("f8e4m3fn,f8e5m2", (("row", "row"), ("row", "col"), ("col", "row"), ("col", "col"))),
        ("f4e2m1", (("row", "col"),)),
        ("f4e1m2", (("row", "col"),)),
    ):
        for shape_label, shape_args in (
            ("regular", []),
            ("unaligned", ["--m", "333", "--n", "444", "--k", "535"]),
        ):
            for la, lb in bias_layouts:
                yield (
                    f"mx-mmad-bias-{case.replace(',', '-')}-{la}{lb}-{shape_label}",
                    [[
                        "tla/kernel/matmul/mx_mmad/mx_matmul_bias.py", "--case", case,
                        "--layout-a", la, "--layout-b", lb,
                        *shape_args, *dev,
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
                f"tla/kernel/matmul/basic_mmad/{script}", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", "f16", "--dtype-b", "f16", "--dtype-c", "f32", *dev,
            ]],
        )

    for dtype in ("f16", "bf16", "f32"):
        yield (
            f"mmad-atomic-add-{dtype}",
            [[
                "tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", dtype, "--dtype-b", dtype, "--dtype-c", "f32", *dev,
            ]],
        )

    # --- basic_matmul_bias: a required 1 x N bias operand, loaded GM -> L1 ->
    # biasbuf through the flat VectorLayout routes and consumed by the first MMAD of
    # every block. Operand dtypes follow basic_matmul.py, and the output follows
    # from them, so there is no --dtype-c to vary: i8 x i8 writes an i32
    # accumulator, every other route keeps fp32.
    yield ("mmad-bias-default", [["tla/kernel/matmul/basic_mmad/basic_matmul_bias.py", *dev]])
    for dtype in ("f16", "bf16", "f32", "i8"):
        yield (
            f"mmad-bias-{dtype}",
            [[
                "tla/kernel/matmul/basic_mmad/basic_matmul_bias.py",
                *MMAD_MNK,
                "--dtype-a", dtype, "--dtype-b", dtype, *dev,
            ]],
        )
    # fp8 is the one family whose two formats mix freely, so each pairing is a
    # distinct route through the cube and gets its own case.
    for dtype_a, dtype_b in (
        ("f8e4m3fn", "f8e4m3fn"),
        ("f8e4m3fn", "f8e5m2"),
        ("f8e5m2", "f8e4m3fn"),
        ("f8e5m2", "f8e5m2"),
    ):
        yield (
            f"mmad-bias-{dtype_a}-{dtype_b}",
            [[
                "tla/kernel/matmul/basic_mmad/basic_matmul_bias.py",
                *MMAD_MNK,
                "--dtype-a", dtype_a, "--dtype-b", dtype_b, *dev,
            ]],
        )

    yield ("mmad-ptr", [["tla/kernel/matmul/basic_mmad/basic_mmad_ptr.py", *dev]])

    # basic_mmad_l0c2l1: cube-only E=(A@B)@D, L0C->L1 (fixpipe) staging reused
    # as the second mmad's LHS; single static tile (M=50/N=60/K=64).
    yield ("mmad-l0c2l1", [["tla/kernel/matmul/basic_mmad/basic_matmul_l0c2l1.py", *dev]])

    # --- basic_mmad fixpipe per-tensor quant, with and without relu ---
    # `--dtype-ab` picks the accumulator (f32 -> f32, i8 -> i32) and with it the
    # legal `--dtype-c` set. `--layout-c col` stores the result ColumnMajor: the
    # fixpipe transposes the tile (nz2dn) on its way out of L0C and applies the
    # per-tensor quant -- and the relu, when enabled -- to that transposed store.
    # RowMajor and ColumnMajor pick different runtime wrappers, so both layouts
    # run for every (accumulator, destination) pair.
    for src, dst in FIXPIPE_QUANT_SRC_DST:
        # The example names the accumulator by its operands: f32 MMAD is
        # `--dtype-ab f32`, the i32 MMAD is `--dtype-ab i8` (i8 x i8 -> i32).
        dtype_ab = "f32" if src == "f32" else "i8"
        for layout_c in ("row", "col"):
            for relu in (0, 1):
                yield (
                    f"mmad-fixpipe-per-tensor-{layout_c}-{src}-{dst}-relu{relu}",
                    [[
                        "tla/kernel/matmul/basic_mmad/basic_matmul_fixpipe_per_tensor.py",
                        *MMAD_MNK,
                        "--dtype-ab", dtype_ab, "--dtype-c", dst,
                        "--relu-enable", str(relu),
                        "--layout-c", layout_c,
                        *dev,
                    ]],
                )

    # --- basic_mmad fixpipe per-channel quant, with and without relu ---
    # Same accumulator / dtype-c pairing as per-tensor. What differs is the
    # scale: one value per output channel instead of one scalar, staged GM -> L1
    # as a flat VectorLayout tile and handed to the fixpipe as its dequant
    # tensor. `--layout-c col` takes the same transposing (nz2dn) store as
    # per-tensor, with the per-channel dequant applied to the transposed tile.
    # The kernel drives the fixpipe with the MMAD unit flag unconditionally, as
    # per-tensor does, so there is no separate flag-sync case to cover.
    for src, dst in FIXPIPE_QUANT_SRC_DST:
        dtype_ab = "f32" if src == "f32" else "i8"
        for layout_c in ("row", "col"):
            for relu in (0, 1):
                yield (
                    f"mmad-fixpipe-per-channel-{layout_c}-{src}-{dst}-relu{relu}",
                    [[
                        "tla/kernel/matmul/basic_mmad/basic_matmul_fixpipe_per_channel.py",
                        *MMAD_MNK,
                        "--dtype-ab", dtype_ab, "--dtype-c", dst,
                        "--relu-enable", str(relu),
                        "--layout-c", layout_c,
                        *dev,
                    ]],
                )

    # --- basic_vadd: every dtype under each sync mode ---
    for mode in VADD_MODES:
        for dtype in VADD_DTYPES:
            label = mode.lstrip("-") or "flag-sync"
            yield (
                f"vadd-{label}-{dtype}",
                [[
                    "tla/kernel/vec/basic_vadd/basic_vadd.py", "--dtype", dtype, *dev,
                    *([mode] if mode else []),
                ]],
            )

    # --- extern_op: user-provided Ascend C functions called from TLA kernels ---
    yield (
        "extern-vecadd",
        [["dsl_tutorials/extern_op/extern_vecadd.py", *dev]],
    )
    yield (
        "extern-dual-core",
        [["dsl_tutorials/extern_op/extern_dual_core.py", *dev]],
    )
    yield (
        "extern-multi-ops",
        [["dsl_tutorials/extern_op/extern_multi_ops.py", *dev]],
    )
    yield (
        "extern-custom-include",
        [["dsl_tutorials/extern_op/extern_custom_include.py", *dev]],
    )

    # --- basic_mixed ---
    # One test on purpose: the second invocation exercises cache reuse and is
    # only meaningful straight after the one that populates the cache.
    yield (
        "mixed-compile-then-cache-reuse",
        [
            ["tla/kernel/matmul/basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
            ["tla/kernel/matmul/basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
        ],
    )
    yield (
        "mixed-mutex",
        [["tla/kernel/matmul/basic_mixed/basic_mixed_mutex.py", *dev, "--block-num", "1"]],
    )
    yield ("mixed-ub2l1", [["tla/kernel/matmul/basic_mixed/basic_mixed_ub2l1.py", *dev]])
    yield ("mixed-store-zN", [["tla/kernel/matmul/basic_mixed/basic_mixed_store_zN.py", *dev]])
    # m=64 is fractal-aligned (multiple of 16); m=50 exercises the zNUnAlign M
    # axis, where the dest leaf[0] is the runtime row count and the stride varies.
    for m in ("64", "50"):
        yield (
            f"mixed-store-zNUnAlign-m{m}",
            [["tla/kernel/matmul/basic_mixed/basic_mixed_store_zNUnAlign.py", *dev, "--m", m]],
        )
    yield ("mixed-fixpipe-nz2dn", [["tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_nz2dn.py", *dev]])
    # CV 1:1 (core_ratio=C1V1): single-AIV epilogue, L0C→GM, mode-4 aiv_id=0.
    # Script under python/tla_dsl/tests/end_to_end/; battery must keep this case
    # as the C1V1 device guard (do not drop when it fails — fix the launch path).
    yield (
        "mixed-core-ratio-11",
        [["end_to_end/basic_mixed_core_ratio_11.py", *dev, "--block-num", "1"]],
    )
    # Host make_ptr/make_layout/make_tensor Torch bind (no DLPack), vector add.
    yield (
        "host-make-tensor-torch-vadd",
        [["end_to_end/host_make_tensor_torch_vadd.py", *dev, "--block-num", "1"]],
    )

    # basic_mixed fixpipe per-tensor quant: the cube writes L0C into the UB tile
    # of the AIV that owns the block and the vector side stores it out, so the
    # two cores hand UB back and forth per block. The dtype map is the float
    # half of the basic_mmad one -- 8-bit destinations are not part of this
    # example -- and, as there, `--layout-c col` takes the transposing (nz2dn)
    # fixpipe and applies the quant to the transposed store.
    for src, dst in MIXED_FIXPIPE_QUANT_SRC_DST:
        # The example names the accumulator by its operands: f32 MMAD is
        # `--dtype-ab f32`, the i32 MMAD is `--dtype-ab i8` (i8 x i8 -> i32).
        dtype_ab = "f32" if src == "f32" else "i8"
        for layout_c in ("row", "col"):
            for relu in (0, 1):
                yield (
                    f"mixed-fixpipe-per-tensor-{layout_c}-{src}-{dst}-relu{relu}",
                    [[
                        "tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_tensor.py",
                        "--dtype-ab", dtype_ab, "--dtype-c", dst,
                        "--relu-enable", str(relu),
                        "--layout-c", layout_c,
                        *dev,
                    ]],
                )
    # What this example adds over basic_mmad is the grid walk: one kernel covers
    # every tile, so the hand-off repeats per block and the tail tile comes out
    # smaller than a full one on either axis.
    for label, m, n, k, extra in MIXED_FIXPIPE_SHAPES:
        yield (
            f"mixed-fixpipe-per-tensor-{label}",
            [[
                "tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_tensor.py",
                "--m", m, "--n", n, "--k", k, *extra, *dev,
            ]],
        )

    # The per-channel twin. Instead of one scalar, the scale is a tensor with one
    # entry per output column: packed into a uint64 word per column and staged
    # GM -> L1 once per task block, because each block covers a different run of
    # columns. That per-block staging is the part basic_mmad cannot cover -- its
    # example runs a single tile -- so the shape cases below are what exercise
    # the rewrite and the tail-block crop.
    for src, dst in MIXED_FIXPIPE_QUANT_SRC_DST:
        dtype_ab = "f32" if src == "f32" else "i8"
        for layout_c in ("row", "col"):
            for relu in (0, 1):
                yield (
                    f"mixed-fixpipe-per-channel-{layout_c}-{src}-{dst}-relu{relu}",
                    [[
                        "tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_channel.py",
                        "--dtype-ab", dtype_ab, "--dtype-c", dst,
                        "--relu-enable", str(relu),
                        "--layout-c", layout_c,
                        *dev,
                    ]],
                )
    for label, m, n, k, extra in MIXED_FIXPIPE_SHAPES:
        yield (
            f"mixed-fixpipe-per-channel-{label}",
            [[
                "tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_channel.py",
                "--m", m, "--n", n, "--k", k, *extra, *dev,
            ]],
        )

    # --- mixed-core handshake and standalone control-flow probes ---
    # flash_attention_infer routes --mask to two independently compiled
    # kernels: causal -> flash_attention_mask_kernel (TileMask + fine mask),
    # none -> flash_attention_nomask_kernel. Both binaries must run.
    for mask_mode in ("causal", "none"):
        yield (
            f"flash-attention-infer-{mask_mode}",
            [["tla/kernel/attention/flash_attention_infer/flash_attention_infer.py", "--mask", mask_mode, *dev]],
        )
    yield (
        "flash-attention-infer-causal-no-paged",
        [[
            "tla/kernel/attention/flash_attention_infer/flash_attention_infer.py",
            "--mask", "causal", "--no-paged", "--kvseqlen", "160", *dev,
        ]],
    )
    yield (
        "flash-attention-infer-nomask-no-paged",
        [[
            "tla/kernel/attention/flash_attention_infer/flash_attention_infer.py",
            "--mask", "none", "--no-paged", "--kvseqlen", "224", *dev,
        ]],
    )
    yield (
        "lazy-conditions",
        [["dsl_tutorials/lazy_conditions/lazy_conditions.py", *dev]],
    )
    # Constexpr Callable / @tla.jit helper as Constexpr epilogue (Phase-1).
    yield (
        "constexpr-callable",
        [["dsl_tutorials/constexpr_callable/constexpr_callable_epilogue.py", *dev]],
    )
    yield (
        "jit-callable",
        [["dsl_tutorials/jit_callable/jit_callable_epilogue.py", *dev]],
    )
    yield (
        "simt-basic-vadd",
        [["dsl_tutorials/vector_simt_ops/basic_vadd_simt/basic_vadd_simt.py", *dev]],
    )
    # The only battery case that puts a SIMT region inside a mixed AIC+AIV
    # kernel, and the only one that gives SIMT any UB at all: basic_vadd_simt
    # is pure AIV and reads straight from global memory. It also covers the
    # launch-sized UB region, whose buffer the SIMD and SIMT stages share.
    yield (
        "simt-mixed",
        [["dsl_tutorials/vector_simt_ops/simt_mixed/simt_mixed.py", *dev]],
    )
    # The op coverage for SIMT: arithmetic, abs/exp/log/sqrt, max/min/where,
    # over six dtypes -- including the bf16 round-trip through f32 and the
    # integer divide. simt-basic-vadd exercises a single addition.
    yield (
        "simt-multi-ops",
        [["dsl_tutorials/vector_simt_ops/multi_ops_simt/multi_ops_simt.py", *dev]],
    )

    # --- multi_core_splitk_matmul and tail_multi_core_splitk_matmul ---
    for script in ("multi_core_splitk_matmul.py", "tail_multi_core_splitk_matmul.py"):
        for m, n, k in MMAD_SHAPES + BIG_SHAPES:
            yield (
                f"{script}-{m}x{n}x{k}",
                [[f"tla/kernel/matmul/multi_core_splitk_matmul/{script}", "--m", m, "--n", n, "--k", k, *dev]],
            )

    # --- basic_mmad_streamk: a streamK example for workload balance ---
    for m, n, k in MMAD_SHAPES + BIG_SHAPES:
        yield (
            f"mmad-streamk-{m}x{n}x{k}",
            [["tla/kernel/matmul/basic_mmad_streamk/basic_mmad_streamk.py", "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- batched_matmul ---
    for b, m, n, k in BATCH_SHAPES:
        yield (
            f"batched-matmul-{b}x{m}x{n}x{k}",
            [["tla/kernel/matmul/batched_matmul/batched_matmul.py", "--batch", b, "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- grouped_matmul_slice_m: grouped matmul example ---
    yield (
        "grouped-matmul-slice-m",
        [["tla/kernel/gmm/grouped_matmul_slice_m/grouped_matmul_slice_m.py", *dev],
         ["tla/kernel/gmm/grouped_matmul_slice_m/grouped_matmul_slice_m.py",
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
                        f"tla/kernel/matmul/basic_mmad_epilogue/matmul_{op}.py",
                        "--m", m, "--n", n, "--k", k,
                        "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                        *dev
                    ]],
                )

    # --- vector_ops: each of these sweeps or batches many kernels internally ---
    yield (
        "vector-masked-binary",
        [["dsl_tutorials/vector_simd_ops/masked_binary.py", "masked_binary", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-bitwise-ops",
        [["dsl_tutorials/vector_simd_ops/bitwise_ops.py", "bitwise_ops", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-binary-op",
        [[
            "dsl_tutorials/vector_simd_ops/binary_op.py", "--batch-run",
            "add", "sub", "mul", "div", "max", "min", "add_unalign", "add_brc_b32",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield ("vector-reduction-ops", [["dsl_tutorials/vector_simd_ops/reduction_ops.py", "--batch-run", *dev]])
    yield (
        "vector-load-store-scalar-after-reduction",
        [["dsl_tutorials/vector_simd_ops/load_and_store_scalar_after_reduction.py", *dev]],
    )
    yield (
        "vector-compare-mask",
        [[
            "dsl_tutorials/vector_simd_ops/compare_mask.py", "--batch-run", *COMPARE_MASK_OPS,
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-unary-ops",
        [[
            "dsl_tutorials/vector_simd_ops/unary_ops.py", "--batch-run",
            "exp", "log", "sqrt", "abs", "neg", "masked_unary", "masked_abs", "masked_neg",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-arange-op",
        [[
            "dsl_tutorials/vector_simd_ops/arange_op.py", "--batch-run", "increase", "decrease",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-interleave-op",
        [[
            "dsl_tutorials/vector_simd_ops/interleave_op.py", "--batch-run", "interleave", "deinterleave",
            "--shape", "512", "--batch-size", "4", *dev,
        ]],
    )
    # Predicate rearrangement: both outputs checked for b8/b16/b32.
    for op in ("mask_interleave", "mask_deinterleave"):
        yield (
            f"vector-{op.replace('_', '-')}-op",
            [[
                "dsl_tutorials/vector_simd_ops/mask_interleave_op.py", op,
                "--all-dtypes", "--shape", "512", *dev,
            ]],
        )
    yield (
        "mask-pack-op",
        [
            [
                "dsl_tutorials/vector_simd_ops/pack_op.py",
                "--mask-width", mask_width,
                "--part", part,
                *dev,
            ]
            for mask_width in ("64", "128")
            for part in ("low", "high")
        ],
    )
    # Vector Pack/UnPack: 16/32-bit truncation and sign/zero-extension, both
    # register halves, both the signed-unpack and bitcast-through-unsigned
    # routes.
    yield (
        "vector-pack-op",
        [
            [
                "dsl_tutorials/vector_simd_ops/pack_op.py",
                "--kind", "vector",
                "--wide-bits", wide_bits,
                "--part", part,
                *(["--signed-unpack"] if signed_unpack else []),
                *dev,
            ]
            for wide_bits in ("16", "32")
            for part in ("low", "high")
            for signed_unpack in (False, True)
        ],
    )
    yield (
        "vector-mask-select-op",
        [[
            "dsl_tutorials/vector_simd_ops/mask_select_op.py", "mask_select",
            "--all-dtypes", "--shape", "512", *dev,
        ]],
    )
    yield (
        "vector-load-dintlv-op",
        [["dsl_tutorials/vector_simd_ops/load_dintlv_op.py", "dintlv_b32", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-us-b8-op",
        [["dsl_tutorials/vector_simd_ops/load_us_b8_op.py", "us_b8", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-dist-ops",
        [
            [
                "dsl_tutorials/vector_simd_ops/load_dist_ops.py",
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
        [["dsl_tutorials/vector_simd_ops/load_store_mask.py", "load_store_mask", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-pack",
        [["dsl_tutorials/vector_simd_ops/store_pack.py", "store_pack", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-first-element",
        [["dsl_tutorials/vector_simd_ops/store_first_element_op.py", "store_first_element", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride",
        [["dsl_tutorials/vector_simd_ops/load_stride_op.py", "stride", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride-repeat",
        [["dsl_tutorials/vector_simd_ops/load_stride_op.py", "repeat", "--all-dtypes", *dev]],
    )
    yield (
        "vector-load-stride-post",
        [["dsl_tutorials/vector_simd_ops/load_stride_op.py", "post", "--all-dtypes", *dev]],
    )
    yield (
        "vector-squeeze-op",
        [["dsl_tutorials/vector_simd_ops/squeeze_op.py", "squeeze", "--sweep", "--shapes", "64", *dev]],
    )
    yield (
        "vector-register-control-flow",
        [["dsl_tutorials/vector_simd_ops/register_control_flow.py", "register_carriers", *dev]],
    )
    yield (
        "vector-cast-multi",
        [["dsl_tutorials/vector_simd_ops/cast_multi.py", "cast_multi", "--shape", "256", *dev]],
    )
    # CAST_NONE / CAST_RINT / CAST_ODD: lossy NONE equals RINT, lossless NONE
    # widen drops rnd, f32->f16 ODD. Script under python/tla_dsl/tests/end_to_end/.
    yield (
        "vector-cast-round-modes",
        [
            [
                "end_to_end/cast_round_modes_op.py",
                "cast_round_modes",
                "--shape",
                "256",
                *dev,
            ]
        ],
    )
    # PACK4/UNPACK4 are UB distributions; keep their device guards outside tutorials.
    for fp8_case in ("layout", "e4m3", "e5m2"):
        yield (
            f"vector-fp8-pack4-{fp8_case}",
            [["end_to_end/fp8_pack4_op.py", "--case", fp8_case, *dev]],
        )

    # Same-width bit reinterpretation: the f32 round trips through i32 and u32
    # are subtracted to zero, so a rewritten register shows as a mismatch
    # rather than hiding behind the exponent term.
    yield (
        "vector-bitcast",
        [["dsl_tutorials/vector_simd_ops/bitcast_op.py", "bitcast", "--shape", "256", *dev]],
    )
    yield (
        "vector-gather",
        [["dsl_tutorials/vector_simd_ops/gather_op.py", "--run", *dev]],
    )
    yield (
        "vector-shift-op",
        [
            ["dsl_tutorials/vector_simd_ops/shift_op.py", shift_op, "--all-dtypes", *dev]
            for shift_op in ("shift_left", "shift_right", "shift_lefts", "shift_rights")
        ],
    )
    # AscendC LoadAlign<T, MaskDist> address typing (copy-capable: i8/i16/i32).
    for mask_dt in ("i8", "i16", "i32"):
        yield (
            f"vector-load-mask-dist-ds-{mask_dt}",
            [
                [
                    "dsl_tutorials/vector_simd_ops/load_mask_dist.py",
                    "ds",
                    "--run",
                    "--mask-dtype",
                    mask_dt,
                    *dev,
                ]
            ],
        )
        yield (
            f"vector-load-mask-dist-us-{mask_dt}",
            [
                [
                    "dsl_tutorials/vector_simd_ops/load_mask_dist.py",
                    "us",
                    "--run",
                    "--mask-dtype",
                    mask_dt,
                    *dev,
                ]
            ],
        )
    yield (
        "vector-load-store-update",
        [["dsl_tutorials/vector_simd_ops/load_store_update.py", "add", "--shape", "400", *dev]],
    )
    yield (
        "vector-load-store-block-update",
        [["dsl_tutorials/vector_simd_ops/load_store_block_update.py", "add", "--shape", "400", *dev]],
    )
    yield (
        "vector-load-store-unalign",
        [[
            "dsl_tutorials/vector_simd_ops/load_store_unalign.py",
            "add",
            "--dtype",
            "f32",
            "--offset",
            "1",
            "--mode",
            "update",
            *dev,
        ]],
    )
    yield (
        "vector-load-store-unalign-once",
        [[
            "dsl_tutorials/vector_simd_ops/load_store_unalign.py",
            "add",
            "--dtype",
            "f16",
            "--offset",
            "1",
            "--mode",
            "once",
            *dev,
        ]],
    )
    # Align UPDATE: stride≠VL (overlap + hole); UnAlign is dense-only now.
    yield (
        "vector-load-store-update-stride-geoms",
        [["dsl_tutorials/vector_simd_ops/load_store_update.py", "--stride-geoms", *dev]],
    )

    # --- tensor_index ---
    yield (
        "tensor-index-scalar-control-flow",
        [["dsl_tutorials/tensor_index/scalar_index_control_flow.py", *dev]],
    )
    yield (
        "tensor-index-scalar-kernel-arg",
        [["dsl_tutorials/tensor_index/scalar_kernel_arg.py", *dev]],
    )

    # --- print_tensor, GM storage ---
    for blocks, calls, dtype_args in PRINT_TENSOR_VARIANTS:
        yield (
            f"print-tensor-gm-b{blocks}-c{calls}",
            [[
                "dsl_tutorials/print_tensor/print_tensor.py", "--run", *dev,
                "--block-num", blocks, "--calls", calls, *dtype_args,
            ]],
        )

    # --- debug_print ---
    yield (
        "debug-print-matrix",
        [[
            "dsl_tutorials/debug_print/debug_print.py", "--run", *dev,
            "--all-dtypes", "--block-num", "2", "--expect-count", "2",
        ]],
    )
    for value in ("-0.0", "nan", "inf", "-inf"):
        yield (
            f"debug-print-f16-{value}",
            [[
                "dsl_tutorials/debug_print/debug_print.py", "--run", *dev,
                "--dtype", "f16", f"--value={value}",
            ]],
        )
    yield (
        "debug-print-expression",
        [["dsl_tutorials/debug_print/debug_print.py", "--run", *dev, "--all-dtypes", "--expression"]],
    )
    for region in ("cube", "vector", "both"):
        yield (
            f"debug-print-mixed-{region}",
            [[
                "dsl_tutorials/debug_print/debug_print_mixed.py", "--run", *dev,
                "--all-dtypes", "--print-region", region,
            ]],
        )
    yield (
        "debug-print-format",
        [["dsl_tutorials/debug_print/debug_print_format.py", "--run", *dev, "--block-num", "2"]],
    )

    # --- scalar_arg_alignment ---
    yield (
        "scalar-arg-alignment",
        [["dsl_tutorials/scalar_arg_alignment/scalar_arg_alignment.py", *dev]],
    )

    # --- mixed Dynamic-GM and static memref arguments ---
    yield (
        "mixed-memref-arguments",
        [["dsl_tutorials/mixed_memref_arguments/mixed_memref_arguments.py", *dev]],
    )

    # --- dataclass_arg (stdlib @dataclass unpacked into scalar kernel args) ---
    yield (
        "dataclass-arg",
        [["dsl_tutorials/dataclass_arg/dataclass_arg.py", *dev]],
    )

    # --- print_tensor, UB storage ---
    for case_name, blocks, calls, dtype_args in PRINT_TENSOR_UB_VARIANTS:
        yield (
            f"print-tensor-ub-{case_name}-b{blocks}-c{calls}",
            [[
                "dsl_tutorials/print_tensor/print_tensor.py", "--run", *dev, "--storage", "ub",
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
            [[f"dsl_tutorials/structured_arguments/{name}.py", *dev]],
        )

    # --- print_tensor, local AIC storage ---
    # Base cases sweep dtypes; repeated entries verify in-process cache reuse.
    local_common = [
        "dsl_tutorials/print_tensor/print_tensor.py", "--run", *dev,
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
        script = example_runner.resolve_script(argv[0])
        assert script.is_file(), f"no such example: {argv[0]}"
        try:
            rc = example_runner.run_case([str(script), *argv[1:]])
        except Exception as exc:
            # Driver may advertise RES_DBG_ADDR but fail at open (error 62).
            if case_id == LOCAL_PRINT_CASE_ID:
                from catlass.base_dsl.runtime.ascend_debug_fifo import (
                    AscDebugFifoError,
                )

                if isinstance(exc, AscDebugFifoError):
                    pytest.skip(str(exc))
            raise
        assert rc == 0, f"{argv[0]} {' '.join(argv[1:])} exited {rc}"


def test_syrk_api_rejects_invalid_block_num(device: int) -> None:
    import example_runner
    import torch

    example = example_runner.load_example(
        example_runner.EXAMPLES_DIR / "tla/kernel/matmul/basic_syrk/basic_syrk.py"
    )
    x = torch.ones((16, 16), dtype=torch.float16, device=f"npu:{device}")
    for block_num in (0, -2):
        with pytest.raises(ValueError, match="block_num must be -1 .* or positive"):
            example.syrk(x, block_num=block_num)
