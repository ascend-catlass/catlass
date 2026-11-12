# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Block mean attention scoring: BF16 mean attention probabilities for 16-token blocks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

_REPO_PATH = str(Path(_DSL_EXAMPLE_PATH).parents[2])
if _REPO_PATH not in sys.path:
    sys.path.append(_REPO_PATH)

import catlass.tla as tla
import torch
import torch_npu

from helpers import get_block_num
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.constants import (
    BLOCK_TOKENS,
)
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.golden import (
    assert_bf16_output_matches_fp32,
    block_mean_attention_score_golden_cpu,
)
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.kernel import (
    block_mean_attention_score_kernel,
)
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.stage_qk import (
    TILING as _TILING,
)

_PATTERNS = ("full", "partial", "mixed", "empty", "first_tile_empty")
_CASES = (
    (1, BLOCK_TOKENS, 1, "full"),
    (2, 48, 63, "mixed"),
    (1, 80, 65, "partial"),
    (2, 240, 127, "mixed"),
    (2, 256, 128, "full"),
    (2, 272, 129, "partial"),
    (1, 304, 256, "full"),
    (2, 496, 257, "mixed"),
    (2, 528, 513, "partial"),
    (2, 272, 65, "first_tile_empty"),
    (2, 512, 128, "empty"),
    (2, 2048, 128, "partial"),
)


@dataclass
class _BlockMeanAttentionScoreInputs:
    """CPU originals for golden and flattened NPU buffers for the kernel."""

    query: torch.Tensor  # CPU BF16 [H, S, D]
    key: torch.Tensor  # CPU BF16 [H, S, D]
    counts: torch.Tensor  # CPU INT32 [S // 16], shared by Query and Key
    query_2d: torch.Tensor  # NPU BF16 [H * S, D]
    key_2d: torch.Tensor  # NPU BF16 [H * S, D]
    counts_device: torch.Tensor  # NPU INT32 [S // 16]


@dataclass
class RunResult:
    name: str
    passed: bool | None
    max_abs_error: float | None
    launch_blocks: int
    workspace_bytes: int
    ub_bytes_per_aiv: int
    input_sha256: str


def _validate(q, k, counts, block_num):
    if not all(isinstance(x, torch.Tensor) for x in (q, k, counts)):
        raise TypeError("q, k and variable_block_size_16 must be torch.Tensor")
    if q.ndim != 3 or q.shape != k.shape:
        raise ValueError("q and k must have the same [H, S, D] shape")
    heads, slots, dim = q.shape
    if heads < 1 or dim < 1 or slots < BLOCK_TOKENS or slots % BLOCK_TOKENS:
        raise ValueError(
            f"H and D must be positive; S must be a positive multiple of {BLOCK_TOKENS}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must be bfloat16")
    if counts.dtype != torch.int32:
        raise TypeError("variable_block_size_16 must be int32")
    if counts.shape != (slots // BLOCK_TOKENS,):
        raise ValueError(
            f"variable_block_size_16 must have shape [S // {BLOCK_TOKENS}]"
        )
    if q.device.type != "npu" or k.device != q.device or counts.device != q.device:
        raise ValueError("q, k and variable_block_size_16 must be on the same NPU")
    if not all(x.is_contiguous() for x in (q, k, counts)):
        raise ValueError("all inputs must be contiguous")
    if isinstance(block_num, bool) or not isinstance(block_num, int):
        raise TypeError("block_num must be an integer")
    if block_num != -1 and block_num <= 0:
        raise ValueError("block_num must be -1 (automatic) or positive")
    host_counts = counts.detach().cpu()
    if bool(((host_counts < 0) | (host_counts > BLOCK_TOKENS)).any()):
        raise ValueError(
            f"variable_block_size_16 values must be in [0, {BLOCK_TOKENS}]"
        )
    return heads, slots, dim


def _workspace_floats(slots, launch_blocks):
    key_tiles = (slots + _TILING.l1_tm - 1) // _TILING.l1_tm
    return launch_blocks * (slots // BLOCK_TOKENS + key_tiles) * _TILING.l1_tn


@lru_cache(maxsize=128)
def _compiled(heads, dim, device):
    # Sample extents do not enter the ABI/cache key. Only D and its compact
    # stride stay static; S comes from the runtime counts extent.
    q = tla.make_fake_tensor(
        tla.BFloat16, (heads * BLOCK_TOKENS, dim), (dim, 1)
    ).mark_compact_shape_dynamic(0)
    counts = tla.make_fake_tensor(tla.Int32, (1,), (1,)).mark_compact_shape_dynamic(0)
    out = tla.make_fake_tensor(tla.BFloat16, (1,), (1,)).mark_compact_shape_dynamic(0)
    ws = tla.make_fake_tensor(tla.Float32, (1,), (1,)).mark_compact_shape_dynamic(0)
    with torch.npu.device(device):
        return tla.compile(
            block_mean_attention_score_kernel,
            q,
            q,
            counts,
            out,
            ws,
            heads,
            dim,
            _TILING,
            options="--npu-arch 3510",
        )


@dataclass(frozen=True)
class _Runner:
    """One compiled H/D specialization; each call owns new output and workspace."""

    heads: int
    dim: int
    device: torch.device
    artifact: object

    @torch.no_grad()
    def _bind(self, q, k, counts, block_num=-1):
        """Bind one invocation before timing; no input or buffer is kept by self."""
        heads, slots, dim = _validate(q, k, counts, block_num)
        if (heads, dim) != (self.heads, self.dim):
            raise ValueError(f"runner requires H={self.heads} and D={self.dim}")
        if q.device != self.device:
            raise ValueError(f"runner requires inputs on {self.device}")
        blocks = slots // BLOCK_TOKENS
        core_count = get_block_num(-1, self.device.index, kind="mix")
        if block_num > core_count:
            raise ValueError(f"block_num exceeds the device's {core_count} Cube Cores")
        query_tiles = (slots + _TILING.l1_tn - 1) // _TILING.l1_tn
        launch_blocks = (
            min(core_count, heads * query_tiles) if block_num == -1 else block_num
        )
        output = torch.empty(
            (heads, blocks, blocks), dtype=torch.bfloat16, device=self.device
        )
        workspace = torch.empty(
            _workspace_floats(slots, launch_blocks),
            dtype=torch.float32,
            device=self.device,
        )
        tensors = tuple(
            tla.from_dlpack(x, layout_tag=tla.arch.RowMajor).mark_compact_shape_dynamic(
                0
            )
            for x in (
                q.view(heads * slots, dim),
                k.view(heads * slots, dim),
                counts,
                output.view(-1),
                workspace,
            )
        )
        artifact = self.artifact

        def launch():
            # The executor is loaded lazily on the first call's current device.
            with torch.npu.device(self.device):
                artifact(
                    *tensors,
                    block_num=launch_blocks,
                    stream=torch.npu.current_stream(self.device),
                )

        launch.artifact = artifact
        launch.kernel_name = "block_mean_attention_score_kernel"
        launch.ub_bytes = artifact.get_kernel_ub_size()
        launch.workspace = workspace
        launch.workspace_bytes = workspace.numel() * workspace.element_size()
        launch.launch_blocks = launch_blocks
        return output, launch

    @torch.no_grad()
    def __call__(self, q, k, counts, *, block_num=-1):
        output, launch = self._bind(q, k, counts, block_num)
        launch()
        return output


def prepare(H: int, D: int, device) -> _Runner:
    """Compile once for H/D/device and return runner(q, k, counts, *, block_num=-1).

    S and all tensors belong to each invocation. Each call validates inputs,
    allocates independent output/workspace and launches on the current stream.
    """
    for name, value in (("H", H), ("D", D)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(device, bool):
        raise TypeError("device must be an NPU device or a nonnegative device index")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("device index must be nonnegative")
        device = torch.device("npu", device)
    else:
        device = torch.device(device)
    if device.type != "npu":
        raise ValueError("device must be an NPU")
    index = torch.npu.current_device() if device.index is None else device.index
    if index >= torch.npu.device_count():
        raise ValueError("device index exceeds the visible NPU count")
    device = torch.device("npu", index)
    return _Runner(H, D, device, _compiled(H, D, index))


@torch.no_grad()
def block_mean_attention_score(q, k, variable_block_size_16, *, block_num=-1):
    """Return BF16 [H, S//16, S//16] block means; empty-block pairs are NaN.

    Inputs are contiguous BF16 [H,S,D] Q/K and INT32 [S//16] counts on one
    NPU. Counts validation synchronizes with the host. Execution then uses
    the current stream, with a separate workspace for each call.
    """
    if not isinstance(q, torch.Tensor) or q.device.type != "npu":
        raise ValueError("q must be an NPU tensor")
    if q.ndim != 3:
        raise ValueError("q and k must have the same [H, S, D] shape")
    return prepare(q.shape[0], q.shape[2], q.device)(
        q, k, variable_block_size_16, block_num=block_num
    )


def _build_inputs(args):
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.headnum, args.seqlen, args.head_dim)
    query = (torch.randn(shape, generator=rng) * args.amplitude).bfloat16()
    key = (torch.randn(shape, generator=rng) * args.amplitude).bfloat16()
    blocks = args.seqlen // BLOCK_TOKENS
    if args.pattern == "full":
        counts = torch.full((blocks,), BLOCK_TOKENS, dtype=torch.int32)
    elif args.pattern == "partial":
        counts_rng = torch.Generator(device="cpu").manual_seed(args.seed)
        # Before rounding/clamping, 4.0 is the standard deviation and 8.0 is
        # the mean valid-token count (half of a 16-token block).
        counts = (torch.randn(blocks, generator=counts_rng) * 4.0 + 8.0).round()
        counts = counts.clamp(0, BLOCK_TOKENS).to(torch.int32)
    elif args.pattern == "mixed":
        counts = (torch.arange(blocks, dtype=torch.int32) * 7 + 3) % (BLOCK_TOKENS + 1)
        counts[::3] = 0
        counts[-1] = 1
    elif args.pattern == "first_tile_empty":
        counts = torch.full((blocks,), 7, dtype=torch.int32)
        # Empty every routing block in the first Query tile.
        counts[: _TILING.l1_tn // BLOCK_TOKENS] = 0
    else:
        counts = torch.zeros(blocks, dtype=torch.int32)
    if args.poison_padding:
        valid = (torch.arange(BLOCK_TOKENS)[None, :] < counts[:, None]).flatten()
        query[:, ~valid] = float("nan")
        key[:, ~valid] = float("inf")
    device = f"npu:{args.device}"
    return _BlockMeanAttentionScoreInputs(
        query,
        key,
        counts,
        query.reshape(-1, args.head_dim).to(device),
        key.reshape(-1, args.head_dim).to(device),
        counts.to(device),
    )


def _compile_launch(args, inputs):
    runner = prepare(
        inputs.query.shape[0], inputs.query.shape[2], inputs.query_2d.device
    )
    return runner._bind(
        inputs.query_2d.view(inputs.query.shape),
        inputs.key_2d.view(inputs.key.shape),
        inputs.counts_device,
        args.block_num,
    )


@torch.no_grad()
def run(args):
    inputs = _build_inputs(args)
    output, launch = _compile_launch(args, inputs)
    output.fill_(args.sentinel)
    digest = hashlib.sha256()
    for tensor in (inputs.query, inputs.key, inputs.counts):
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if args.perf_only:
        import mstx

        # Load the binary without launching a warmup kernel.
        with torch.npu.device(inputs.query_2d.device):
            launch.artifact._get_executor()
    torch.npu.synchronize(args.device)
    if args.perf_only:
        range_id = mstx.range_start("block_mean_attention_score")
        launch()
        mstx.range_end(range_id)
    else:
        launch()
    torch.npu.synchronize(args.device)
    error = None
    if not args.perf_only:
        expected = block_mean_attention_score_golden_cpu(
            inputs.query,
            inputs.key,
            inputs.counts,
            query_chunk_size=args.query_chunk_size,
        )
        error = assert_bf16_output_matches_fp32(output, expected)
    return RunResult(
        name=f"H{args.headnum}_S{args.seqlen}_D{args.head_dim}_{args.pattern}",
        passed=None if args.perf_only else True,
        max_abs_error=error,
        launch_blocks=launch.launch_blocks,
        workspace_bytes=launch.workspace_bytes,
        ub_bytes_per_aiv=launch.ub_bytes,
        input_sha256=digest.hexdigest(),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--headnum", "--h", type=int, default=2)
    parser.add_argument("--seqlen", "--s", type=int, default=512)
    parser.add_argument("--head-dim", "--d", type=int, default=128)
    parser.add_argument("--pattern", choices=_PATTERNS, default="mixed")
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--sentinel", type=float, default=-7.0)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--poison-padding",
        action="store_true",
        help="fill invalid Q/K slots with NaN/Inf to check masking",
    )
    parser.add_argument(
        "--cases", action="store_true", help="validate the boundary-case suite"
    )
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="one complete launch in an MSTX range, without golden or warmup",
    )
    args = parser.parse_args(argv)
    if (
        args.device < 0
        or args.headnum < 1
        or args.head_dim < 1
        or args.seqlen < BLOCK_TOKENS
        or args.seqlen % BLOCK_TOKENS
    ):
        parser.error(
            f"device must be nonnegative; H/D positive; S a positive multiple of {BLOCK_TOKENS}"
        )
    if args.block_num != -1 and args.block_num <= 0:
        parser.error("--block-num must be -1 or positive")
    if args.query_chunk_size < BLOCK_TOKENS or args.query_chunk_size % BLOCK_TOKENS:
        parser.error(
            f"--query-chunk-size must be a positive multiple of {BLOCK_TOKENS}"
        )
    if not math.isfinite(args.amplitude) or args.amplitude < 0:
        parser.error("--amplitude must be finite and nonnegative")
    if not math.isfinite(args.sentinel) or args.sentinel >= 0:
        parser.error("--sentinel must be finite and negative")
    if args.cases and args.perf_only:
        parser.error("--perf-only profiles exactly one case; omit --cases")
    if (
        not args.cases
        and args.pattern == "first_tile_empty"
        and args.seqlen <= _TILING.l1_tn
    ):
        parser.error(f"first_tile_empty requires S > {_TILING.l1_tn}")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    cases = (
        _CASES
        if args.cases
        else ((args.headnum, args.seqlen, args.head_dim, args.pattern),)
    )
    for heads, slots, dim, pattern in cases:
        case_args = argparse.Namespace(**vars(args))
        case_args.headnum, case_args.seqlen, case_args.head_dim, case_args.pattern = (
            heads,
            slots,
            dim,
            pattern,
        )
        print(json.dumps(asdict(run(case_args))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
