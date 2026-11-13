# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend 950 native CATLASS DSL Flash KDA forward host interface.

Input validation, ABI4 integer scheduling and forward launch share this module.
Adapted from cann/cannbot-dsl fc438d3ef03129929eb8faae3906365b3162f163;
see NOTICE and README.md. Device arithmetic is implemented in flash_kda_fwd.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from threading import Lock
from types import SimpleNamespace

import torch
from torch import Tensor
from catlass import tla

from .flash_kda_fwd import StagedMetadataKernel, PersistentMetadataKernel

__all__ = ["flash_kda_fwd"]


# -----------------------------------------------------------------------------
# Host input validation.
# -----------------------------------------------------------------------------

D = 128


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(name, value, dtype, shape=None, rank=None):
    _require(isinstance(value, Tensor), f"{name} must be a Tensor")
    _require(value.device.type == "npu", f"{name} must be on NPU")
    _require(value.dtype in dtype, f"{name}: unsupported dtype {value.dtype}")
    _require(value.is_contiguous(), f"{name} must be contiguous")
    if rank is not None:
        _require(value.ndim == rank, f"{name}: expected rank {rank}")
    if shape is not None:
        _require(tuple(value.shape) == tuple(shape), f"{name}: expected shape {shape}")


def _validate(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    *,
    scale,
    layout,
    flags,
    lower_bound,
):
    _require(layout in ("BNSD", "BSND", "TND"), "unsupported layout")
    _require(metadata is None, "metadata must be None in this interface version")
    for name, flag in flags.items():
        _require(isinstance(flag, bool), f"{name} must be bool")
    _require(
        scale is None
        or (
            isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(scale)
        ),
        "scale must be finite",
    )
    rank = 3 if layout == "TND" else 4
    for name, x in (("q", q), ("k", k), ("v", v)):
        _tensor(name, x, (torch.bfloat16,), rank=rank)
    _tensor("g", g, (torch.bfloat16, torch.float32), v.shape)
    _tensor("beta", beta, (torch.bfloat16, torch.float32), v.shape[:-1])
    _require(k.shape == q.shape, "q/k shapes must match")
    _require(q.shape[-1] == D and v.shape[-1] == D, "D must be 128")
    if layout == "TND":
        _tensor("cu_seqlens", cu_seqlens, (torch.int32,), rank=1)
        batch, length, nq, nv = (
            cu_seqlens.numel() - 1,
            q.shape[0],
            q.shape[1],
            v.shape[1],
        )
        expected_v = (length, nv, D)
    else:
        batch = q.shape[0]
        length = q.shape[2 if layout == "BNSD" else 1]
        nq, nv = (
            q.shape[1 if layout == "BNSD" else 2],
            v.shape[1 if layout == "BNSD" else 2],
        )
        expected_v = (
            (batch, nv, length, D) if layout == "BNSD" else (batch, length, nv, D)
        )
    _require(batch > 0 and length > 0, "batch and storage length must be positive")
    _require(1 <= nq <= nv <= 96 and nv % nq == 0, "invalid GQA head counts")
    _require(tuple(v.shape) == expected_v, "v shape does not match q")
    lengths = None
    if cu_seqlens is not None:
        _tensor("cu_seqlens", cu_seqlens, (torch.int32,), (batch + 1,))
        prefix = cu_seqlens.cpu().tolist()
        _require(prefix[0] == 0, "cu_seqlens must start at 0")
        lengths = [b - a for a, b in zip(prefix, prefix[1:])]
        _require(all(0 < x <= length for x in lengths), "invalid sequence lengths")
        if layout == "TND":
            _require(prefix[-1] == length, "cu_seqlens endpoint must equal T")
    if seqused is not None:
        _tensor("seqused", seqused, (torch.int32,), (batch,))
        used = seqused.cpu().tolist()
        _require(all(0 < x <= length for x in used), "invalid seqused")
        _require(lengths is None or lengths == used, "inconsistent length tensors")
        lengths = used
    lengths = lengths if lengths is not None else [length] * batch
    _require(sum(lengths) <= 2**31 - 1, "length prefix exceeds INT32")
    if initial_state is not None:
        _tensor("initial_state", initial_state, (torch.float32,), (batch, nv, D, D))
    if flags["use_gate_in_kernel"]:
        _tensor("A_log", A_log, (torch.float32,), (nv,))
        if dt_bias is not None:
            _tensor("dt_bias", dt_bias, (torch.float32,), (nv, D))
        _require(
            lower_bound is None
            or (
                isinstance(lower_bound, (int, float))
                and not isinstance(lower_bound, bool)
                and math.isfinite(lower_bound)
                and -5 <= lower_bound < 0
            ),
            "invalid lower_bound",
        )
    _require(
        not flags["allow_neg_eigval"] or flags["use_beta_sigmoid_in_kernel"],
        "allow_neg_eigval requires beta sigmoid",
    )
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
        ("A_log", A_log if flags["use_gate_in_kernel"] else None),
        ("dt_bias", dt_bias if flags["use_gate_in_kernel"] else None),
        ("cu_seqlens", cu_seqlens),
        ("seqused", seqused),
    ):
        if tensor is not None:
            _require(
                isinstance(tensor, Tensor) and tensor.device == q.device,
                f"{name} must be on the same NPU as q",
            )
    return batch, nq, nv, length, lengths


# -----------------------------------------------------------------------------
# Host ABI4 scheduling, translated from fixed cannbot-dsl metadata.py.
# -----------------------------------------------------------------------------

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
MAX_VALUE_HEADS = 96
WORKSPACE_SLOTS = 1024
MAX_CHUNK_GROUP = 512
HEADER_WORDS = 11
INT32_MAX = 2**31 - 1
STATUS_OK = 0
I64 = int


def zeros(dtype, count):
    return [0] * count


MAGIC = 0x4B444132

ABI_VERSION = 4

MAX_AIC_CORES = 32

ROUND_FIXED_WORDS = 133

ROUND_WORDS_PER_BATCH = 9

HEADER_MAGIC = 0

HEADER_ABI_VERSION = 1

HEADER_STATUS = 2

HEADER_CORE_NUM = 3

HEADER_STAGE12_ROUND_NUM = 4

HEADER_BATCH = 5

HEADER_VALUE_HEADS = 6

HEADER_HAS_CU = 7

HEADER_WORKSPACE_SLOTS = 8

HEADER_TEMPLATE = 9

HEADER_CHUNK_TASK_NUM = 10

TEMPLATE_AUTO = 0

TEMPLATE_STAGED = 1

TEMPLATE_PERSISTENT = 2

CHUNK_TASK_WORDS = 6

CHUNK_TASK_BATCH = 0

CHUNK_TASK_STORAGE_BATCH = 1

CHUNK_TASK_TOKEN_START = 2

CHUNK_TASK_LENGTH = 3

CHUNK_TASK_VALUE_HEAD = 4

CHUNK_TASK_NEXT = 5

STATUS_BAD_CORE_NUM = 1

STATUS_BAD_WORKSPACE_SLOTS = 2

STATUS_BAD_FLAGS = 3

STATUS_BAD_SHAPE = 4

STATUS_BAD_CU_START = 5

STATUS_BAD_LENGTH = 6

STATUS_BAD_PADDED_LENGTH = 7

STATUS_BAD_PACKED_END = 8

STATUS_BAD_METADATA_CAPACITY = 9

SHORT_TRAIN_CHUNK_GROUP = 13

SHORT_TRAIN_MIN_LENGTH = 8192

SHORT_TRAIN_HEADS = 32

INT64_MAX = 2**63 - 1

ROUTE_COST_SCALE = 64

MAX_SCALED_QUANTITY = INT64_MAX // ROUTE_COST_SCALE

ROUTE_PARAMETER_WORDS = 16

ROUTE_SCHEMA_INDEX = 13

ROUTE_MARGIN_INDEX = 14

PARTS_PER_MILLION = 1_000_000

PICOSECONDS_PER_NANOSECOND = 1000

FULL_CHUNK_COST_NS = 4930

CALIBRATED_MIN_CHUNKS = 64

CALIBRATED_MAX_CHUNKS = 256

CALIBRATED_GROUP_LIMIT = 16

MAX_ACTIVE_BATCHES = WORKSPACE_SLOTS

MAX_STAGE1_TASKS = WORKSPACE_SLOTS

MAX_STAGE2_TASKS = WORKSPACE_SLOTS * 8


def _interpolate_stage1_task_cost(
    valid_row_num,
    left_row_num,
    left_cost,
    right_row_num,
    right_cost,
):
    # 拟合系数单位为皮秒，调度成本单位为纳秒。
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (
        right_cost - left_cost
    )
    denominator = row_span * PICOSECONDS_PER_NANOSECOND
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    cost = 0
    if valid_row_num > 0:
        if valid_row_num <= 16:
            cost = _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
        elif valid_row_num <= 32:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 16, 4920438, 32, 4970718
            )
        elif valid_row_num <= 48:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 32, 4970718, 48, 4992347
            )
        elif valid_row_num <= 63:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 48, 4992347, 63, 4952972
            )
        else:
            cost = FULL_CHUNK_COST_NS
    return cost


def _stage2_task_cost(candidate_idx, chunk_num):
    item_init = 352
    chunk_cost = 1839
    if candidate_idx == 1:
        item_init = 133
        chunk_cost = 1186
    elif candidate_idx == 2:
        item_init = 19
        chunk_cost = 884
    elif candidate_idx == 3:
        item_init = 65
        chunk_cost = 729
    return item_init + chunk_num * chunk_cost


def round_record_words(active_batch_num: int) -> int:
    """返回一条变长记录占用的字数。"""

    active_batch_num = int(active_batch_num)
    if active_batch_num <= 0:
        raise ValueError("active_batch_num must be positive")
    return ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH * active_batch_num


def metadata_capacity_upper_bound(
    batch: int,
    storage_length: int,
    *,
    is_packed: bool,
    value_heads: int = MAX_VALUE_HEADS,
) -> int:
    """返回覆盖所有合法轮次划分的安全容量上界。"""

    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    storage_chunk_num = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if is_packed:
        chunk_entries = storage_chunk_num + batch - 1
    else:
        chunk_entries = batch * storage_chunk_num
    if not 1 <= int(value_heads) <= MAX_VALUE_HEADS:
        raise ValueError("value_heads must be in [1, 96]")
    # 每个可能的 chunk 条目预留一个目录字和一条单批次记录。
    group_capacity = (
        HEADER_WORDS
        + 1
        + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries
    )
    # chunk 调度器在任务记录后使用 B 个临时批次索引。
    chunk_capacity = (
        HEADER_WORDS
        + MAX_AIC_CORES
        + CHUNK_TASK_WORDS * batch * int(value_heads)
        + batch
    )
    return max(group_capacity, chunk_capacity)


def _select_group_num(
    remaining_chunk_num,
    active_batch_num,
    value_heads,
    workspace_slots,
    max_chunk_group,
):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > max_chunk_group:
        upper_group_num = max_chunk_group
    workspace_group_limit = workspace_slots // value_heads
    if upper_group_num > workspace_group_limit:
        upper_group_num = workspace_group_limit
    # 槽位用量随组大小单调增长。选取最大可行分组，减少长序列跨越全局阶段屏障的次数。
    while lower_group_num <= upper_group_num:
        candidate_group_num = (lower_group_num + upper_group_num) // 2
        slot_num = 0
        for active_idx in range(0, active_batch_num):
            current_chunk_num = remaining_chunk_num[active_idx]
            if current_chunk_num > candidate_group_num:
                current_chunk_num = candidate_group_num
            slot_num += value_heads * current_chunk_num
        if slot_num <= workspace_slots:
            selected_group_num = candidate_group_num
            lower_group_num = candidate_group_num + 1
        else:
            upper_group_num = candidate_group_num - 1
    return selected_group_num


def _partition_costs(
    task_costs,
    task_num,
    core_num,
    core_ranges,
    core_costs,
):
    for core_idx in range(0, MAX_AIC_CORES):
        core_ranges[core_idx * 2] = 0
        core_ranges[core_idx * 2 + 1] = 0
        core_costs[core_idx] = 0

    if task_num > 0:
        active_core_num = task_num
        if active_core_num > core_num:
            active_core_num = core_num
        remaining_cost = 0
        for task_idx in range(0, task_num):
            remaining_cost += task_costs[task_idx]

        start = 0
        for core_idx in range(0, active_core_num):
            remaining_core_num = active_core_num - core_idx
            if remaining_core_num == 1:
                end = task_num
            else:
                max_end = task_num - (remaining_core_num - 1)
                end = start + 1
                segment_cost = task_costs[start]
                while end < max_end:
                    next_cost = segment_cost + task_costs[end]
                    next_error = abs(next_cost * remaining_core_num - remaining_cost)
                    current_error = abs(
                        segment_cost * remaining_core_num - remaining_cost
                    )
                    if next_error <= current_error:
                        segment_cost = next_cost
                        end += 1
                    else:
                        break

            segment_cost = 0
            for task_idx in range(start, end):
                segment_cost += task_costs[task_idx]
            core_ranges[core_idx * 2] = start
            core_ranges[core_idx * 2 + 1] = end
            core_costs[core_idx] = segment_cost
            remaining_cost -= segment_cost
            start = end


def _chunk_batch_precedes(cu_seqlens, left_batch, right_batch):
    left_length = cu_seqlens[left_batch + 1] - cu_seqlens[left_batch]
    right_length = cu_seqlens[right_batch + 1] - cu_seqlens[right_batch]
    left_chunks = (left_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    right_chunks = (right_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    precedes = 0
    if left_chunks > right_chunks:
        precedes = 1
    elif left_chunks == right_chunks and left_batch < right_batch:
        precedes = 1
    return precedes


def _chunk_heap_sift_down(cu_seqlens, metadata, heap_base, heap_num, root_idx):
    # 最大堆先按 chunk 数排序，再按原始批次索引排序。
    while root_idx * 2 + 1 < heap_num:
        child_idx = root_idx * 2 + 1
        if child_idx + 1 < heap_num:
            if (
                _chunk_batch_precedes(
                    cu_seqlens,
                    metadata[heap_base + child_idx + 1],
                    metadata[heap_base + child_idx],
                )
                == 1
            ):
                child_idx += 1
        root_batch = metadata[heap_base + root_idx]
        child_batch = metadata[heap_base + child_idx]
        if _chunk_batch_precedes(cu_seqlens, child_batch, root_batch) == 0:
            break
        metadata[heap_base + root_idx] = child_batch
        metadata[heap_base + child_idx] = root_batch
        root_idx = child_idx


def _training_group_limit(schedule_args, max_chunks):
    requested = schedule_args.max_chunk_group
    if schedule_args.calibration_enabled == 0:
        return requested
    if (
        schedule_args.layout_kind == 1
        and schedule_args.core_num == MAX_AIC_CORES
        and schedule_args.batch == 1
        and schedule_args.value_heads > 32
        and schedule_args.value_heads <= MAX_VALUE_HEADS
        and max_chunks >= CALIBRATED_MIN_CHUNKS
        and max_chunks <= CALIBRATED_MAX_CHUNKS
    ):
        return min(requested, CALIBRATED_GROUP_LIMIT)
    return requested


def _route_add_cost(time, cost, quantity):
    if time < 0 or cost < 0 or quantity < 0:
        return -1
    if quantity > 0 and cost > (INT64_MAX - time) // quantity:
        return -1
    return time + cost * quantity


def _route_rounds(schedule_args, max_group, remaining_chunks):
    rounds = 0
    batch_start = 0
    batches_per_set = schedule_args.workspace_slots // schedule_args.value_heads
    while batch_start < schedule_args.batch:
        batch_count = min(batches_per_set, schedule_args.batch - batch_start)
        longest = 0
        for local_batch in range(batch_count):
            batch = batch_start + local_batch
            length = schedule_args.storage_length
            if schedule_args.has_cu == 1:
                length = (
                    schedule_args.cu_seqlens[batch + 1]
                    - schedule_args.cu_seqlens[batch]
                )
            chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
            remaining_chunks[local_batch] = chunks
            longest = max(longest, chunks)
        while longest > 0:
            low = 1
            high = min(max_group, min(longest, batches_per_set))
            while low < high:
                middle = (low + high + 1) // 2
                chunk_count = 0
                for local_batch in range(batch_count):
                    chunk_count += min(remaining_chunks[local_batch], middle)
                if (
                    chunk_count
                    <= schedule_args.workspace_slots // schedule_args.value_heads
                ):
                    low = middle
                else:
                    high = middle - 1
            longest = 0
            for local_batch in range(batch_count):
                chunks = max(0, remaining_chunks[local_batch] - low)
                remaining_chunks[local_batch] = chunks
                longest = max(longest, chunks)
            rounds += 1
        batch_start += batch_count
    return rounds


def _route_persistent_loads(schedule_args, uniform_chunks, core_chunks, core_rows):
    # 使用与执行调度相同的堆及按 chunk 数排序的 LPT 顺序。
    task_count = schedule_args.batch * schedule_args.value_heads
    heap_base = HEADER_WORDS + MAX_AIC_CORES + CHUNK_TASK_WORDS * task_count
    if heap_base + schedule_args.batch > schedule_args.metadata_word_capacity:
        return -1
    heap_count = schedule_args.batch
    if uniform_chunks == 0:
        for batch in range(schedule_args.batch):
            schedule_args.metadata[heap_base + batch] = batch
        root = heap_count // 2
        while root > 0:
            root -= 1
            _chunk_heap_sift_down(
                schedule_args.cu_seqlens,
                schedule_args.metadata,
                heap_base,
                heap_count,
                root,
            )
    peak = 0
    for scheduled_batch in range(schedule_args.batch):
        batch = scheduled_batch
        if uniform_chunks == 0:
            batch = schedule_args.metadata[heap_base]
            heap_count -= 1
            if heap_count > 0:
                schedule_args.metadata[heap_base] = schedule_args.metadata[
                    heap_base + heap_count
                ]
                _chunk_heap_sift_down(
                    schedule_args.cu_seqlens,
                    schedule_args.metadata,
                    heap_base,
                    heap_count,
                    0,
                )
        length = schedule_args.storage_length
        if schedule_args.has_cu == 1:
            length = (
                schedule_args.cu_seqlens[batch + 1] - schedule_args.cu_seqlens[batch]
            )
        chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        for head in range(schedule_args.value_heads):
            selected_core = 0
            for core in range(1, schedule_args.core_num):
                if core_chunks[core] < core_chunks[selected_core]:
                    selected_core = core
            core_chunks[selected_core] += chunks
            core_rows[selected_core] += length
            peak = max(peak, core_chunks[selected_core])
    return peak


def _select_route_from_model(
    schedule_args,
    staged_peak,
    total_chunks,
    total_rows,
    query_rows,
    persistent_peak,
    core_rows,
    max_group,
    remaining_chunks,
):
    # 非负成本单位为飞秒，两种时间均乘以 64 后比较。
    if (
        staged_peak < 1
        or total_chunks < 1
        or total_rows < 1
        or query_rows < 1
        or persistent_peak < 1
    ):
        return TEMPLATE_STAGED
    parallel_chunks = (total_chunks - 1) // schedule_args.core_num + 1
    head_count = schedule_args.batch * schedule_args.value_heads
    rounds = _route_rounds(schedule_args, max_group, remaining_chunks)
    if (
        parallel_chunks > MAX_SCALED_QUANTITY
        or staged_peak > MAX_SCALED_QUANTITY
        or persistent_peak > MAX_SCALED_QUANTITY
        or rounds > MAX_SCALED_QUANTITY
        or head_count > MAX_SCALED_QUANTITY
    ):
        return TEMPLATE_STAGED
    base = schedule_args.layout_kind * ROUTE_PARAMETER_WORDS
    schema = schedule_args.route_coefficients[base + ROUTE_SCHEMA_INDEX]
    if schema != 6 and schema != 7:
        return TEMPLATE_STAGED
    for index in range(ROUTE_SCHEMA_INDEX):
        if schedule_args.route_coefficients[base + index] < 0:
            return TEMPLATE_STAGED
    staged = 0
    for index in range(5):
        quantity = ROUTE_COST_SCALE
        if index == 1:
            quantity = parallel_chunks * ROUTE_COST_SCALE
        elif index == 2:
            quantity = staged_peak * ROUTE_COST_SCALE
        elif index == 3:
            quantity = rounds * ROUTE_COST_SCALE
        elif index == 4:
            quantity = query_rows
        staged = _route_add_cost(
            staged, schedule_args.route_coefficients[base + index], quantity
        )
        if staged < 0:
            return TEMPLATE_STAGED
    compute = _route_add_cost(
        0,
        schedule_args.route_coefficients[base + 5],
        parallel_chunks * ROUTE_COST_SCALE,
    )
    memory = _route_add_cost(0, schedule_args.route_coefficients[base + 6], total_rows)
    if compute < 0 or memory < 0:
        return TEMPLATE_STAGED
    staged = _route_add_cost(staged, max(compute, memory), 1)
    if staged < 0:
        return TEMPLATE_STAGED
    persistent = 0
    for index in range(7, 11):
        quantity = ROUTE_COST_SCALE
        if index == 8:
            quantity = persistent_peak * ROUTE_COST_SCALE
        elif index == 9:
            quantity = head_count * ROUTE_COST_SCALE
        elif index == 10:
            quantity = query_rows
        persistent = _route_add_cost(
            persistent, schedule_args.route_coefficients[base + index], quantity
        )
        if persistent < 0:
            return TEMPLATE_STAGED
    previous_rows = 0
    for phase in range(schedule_args.core_num):
        next_rows = 0
        active_cores = 0
        for core in range(schedule_args.core_num):
            rows = core_rows[core]
            if rows > previous_rows:
                active_cores += 1
                if next_rows == 0 or rows < next_rows:
                    next_rows = rows
        if next_rows == 0:
            break
        memory = _route_add_cost(
            0, schedule_args.route_coefficients[base + 12], active_cores
        )
        if memory < 0:
            return TEMPLATE_STAGED
        cost = max(schedule_args.route_coefficients[base + 11], memory)
        persistent = _route_add_cost(persistent, cost, next_rows - previous_rows)
        if persistent < 0:
            return TEMPLATE_STAGED
        previous_rows = next_rows
    margin = schedule_args.route_coefficients[base + ROUTE_MARGIN_INDEX]
    if margin < 0 or margin >= PARTS_PER_MILLION:
        return TEMPLATE_STAGED
    required_gain = (staged // PARTS_PER_MILLION) * margin + (
        staged % PARTS_PER_MILLION
    ) * margin // PARTS_PER_MILLION
    if persistent < staged - required_gain:
        return TEMPLATE_PERSISTENT
    return TEMPLATE_STAGED


def flash_kda_fwd_metadata_kernel(a: FlashKDAFwdMetadataArgs):  # noqa: F821
    # Source annotation is postponed and never evaluated; callers pass SimpleNamespace.
    valid_seq_len_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    token_start_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    tail_row_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    active_local_batch_idx = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    remaining_chunk_num = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_per_group = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_task_costs = zeros(I64, MAX_STAGE1_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    stage2_task_costs = zeros(I64, MAX_STAGE2_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    selected_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    route_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    route_lpt_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    route_valid_rows = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_tails = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。

    if a.metadata_word_capacity < HEADER_WORDS:
        # AICPU 将非零内核返回值视为任务异常（ACL 507005）。
        # 诊断码写入 metadata 头部，不作为内核返回值。
        return STATUS_OK

    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_CORE_NUM] = a.core_num
    a.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_HAS_CU] = a.has_cu
    a.metadata[HEADER_WORKSPACE_SLOTS] = a.workspace_slots
    a.metadata[HEADER_TEMPLATE] = TEMPLATE_AUTO
    a.metadata[HEADER_CHUNK_TASK_NUM] = 0
    if a.metadata_word_capacity > HEADER_WORDS:
        a.metadata[HEADER_WORDS] = HEADER_WORDS + 1

    status = STATUS_OK
    if a.core_num < 1 or a.core_num > MAX_AIC_CORES:
        status = STATUS_BAD_CORE_NUM
    elif (
        a.template_kind != TEMPLATE_AUTO
        and a.template_kind != TEMPLATE_STAGED
        and a.template_kind != TEMPLATE_PERSISTENT
    ):
        status = STATUS_BAD_FLAGS
    elif a.calibration_enabled != 0 and a.calibration_enabled != 1:
        status = STATUS_BAD_FLAGS
    elif a.layout_kind != 0 and a.layout_kind != 1:
        status = STATUS_BAD_FLAGS
    elif a.workspace_slots != WORKSPACE_SLOTS:
        status = STATUS_BAD_WORKSPACE_SLOTS
    elif a.max_chunk_group < 1 or a.max_chunk_group > MAX_CHUNK_GROUP:
        status = STATUS_BAD_FLAGS
    elif not (a.has_cu == 0 or a.has_cu == 1):
        status = STATUS_BAD_FLAGS
    elif not (a.is_packed == 0 or a.is_packed == 1):
        status = STATUS_BAD_FLAGS
    elif a.is_packed == 1 and a.has_cu == 0:
        status = STATUS_BAD_FLAGS
    elif a.batch < 1 or a.storage_length < 1 or a.value_heads < 1:
        status = STATUS_BAD_SHAPE
    elif a.value_heads > MAX_VALUE_HEADS:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 1 and a.physical_batches != 1:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 0 and a.physical_batches != a.batch:
        status = STATUS_BAD_SHAPE
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    if a.has_cu == 1:
        if a.cu_seqlens[0] != 0:
            a.metadata[HEADER_STATUS] = STATUS_BAD_CU_START
            return STATUS_OK

    # 发布选路结果或任务前，检查所有实际序列长度。
    uniform_chunk_num = 1
    first_chunk_num = 0
    route_max_chunk_num = 0
    route_total_chunk_num = 0
    route_peak_chunk_num = 0
    route_total_rows = 0
    route_query_rows = 0
    evaluate_route = 0
    if a.template_kind == TEMPLATE_AUTO and a.calibration_enabled == 1:
        evaluate_route = 1
    for batch_idx in range(0, a.batch):
        valid_seq_len = a.storage_length
        if a.has_cu == 1:
            valid_seq_len = a.cu_seqlens[batch_idx + 1] - a.cu_seqlens[batch_idx]
        if valid_seq_len <= 0:
            status = STATUS_BAD_LENGTH
        elif a.is_packed == 0 and valid_seq_len > a.storage_length:
            status = STATUS_BAD_PADDED_LENGTH
        else:
            current_chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            if batch_idx == 0:
                first_chunk_num = current_chunk_num
            elif current_chunk_num != first_chunk_num:
                uniform_chunk_num = 0
            if current_chunk_num > route_max_chunk_num:
                route_max_chunk_num = current_chunk_num
            if evaluate_route == 1:
                if current_chunk_num > INT64_MAX // a.value_heads:
                    evaluate_route = 0
                else:
                    batch_chunk_num = a.value_heads * current_chunk_num
                    if route_total_chunk_num > INT64_MAX - batch_chunk_num:
                        evaluate_route = 0
                    else:
                        route_total_chunk_num += batch_chunk_num
                        route_total_rows = _route_add_cost(
                            route_total_rows, a.value_heads, valid_seq_len
                        )
                        route_query_rows = _route_add_cost(
                            route_query_rows, a.query_heads, valid_seq_len
                        )
                        if route_total_rows < 0 or route_query_rows < 0:
                            evaluate_route = 0
                        # 每个核的成本不超过已检查的总成本。
                        for value_head_idx in range(0, a.value_heads):
                            cyclic_core = (
                                batch_idx * a.value_heads + value_head_idx
                            ) % a.core_num
                            route_core_costs[cyclic_core] += current_chunk_num
    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK
    selected_template = a.template_kind
    if selected_template == TEMPLATE_AUTO:
        selected_template = TEMPLATE_STAGED
        if evaluate_route == 1:
            for core_idx in range(0, a.core_num):
                if route_core_costs[core_idx] > route_peak_chunk_num:
                    route_peak_chunk_num = route_core_costs[core_idx]
            persistent_peak = _route_persistent_loads(
                a, uniform_chunk_num, route_lpt_costs, route_valid_rows
            )
            selected_template = _select_route_from_model(
                a,
                route_peak_chunk_num,
                route_total_chunk_num,
                route_total_rows,
                route_query_rows,
                persistent_peak,
                route_valid_rows,
                _training_group_limit(a, route_max_chunk_num),
                remaining_chunk_num,
            )
            # 短多序列训练保留直接 U 路径的缓存。
            if a.has_cu == 1 and a.batch > 1 and route_max_chunk_num <= 2:
                selected_template = TEMPLATE_PERSISTENT
    a.metadata[HEADER_TEMPLATE] = selected_template
    if selected_template == TEMPLATE_PERSISTENT:
        task_num = a.batch * a.value_heads
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        heap_base = used_words
        if used_words + a.batch > a.metadata_word_capacity:
            a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
            return STATUS_OK
        for core_idx in range(0, MAX_AIC_CORES):
            a.metadata[HEADER_WORDS + core_idx] = 0
        # 等长序列保持原始批次和头顺序；变长批次采用 O(B log B) 的 LPT 调度。
        # 使用 B 个 GM 临时字，避免固定本地数组和反复扫描全部批次。
        heap_num = a.batch
        if uniform_chunk_num == 0:
            for batch_idx in range(0, a.batch):
                a.metadata[heap_base + batch_idx] = batch_idx
            heap_root = a.batch // 2
            while heap_root > 0:
                heap_root -= 1
                _chunk_heap_sift_down(
                    a.cu_seqlens, a.metadata, heap_base, heap_num, heap_root
                )
        task_cursor = task_base
        for scheduled_batch_idx in range(0, a.batch):
            batch_idx = scheduled_batch_idx
            if uniform_chunk_num == 0:
                batch_idx = a.metadata[heap_base]
                heap_num -= 1
                if heap_num > 0:
                    a.metadata[heap_base] = a.metadata[heap_base + heap_num]
                    _chunk_heap_sift_down(
                        a.cu_seqlens, a.metadata, heap_base, heap_num, 0
                    )
            storage_batch_idx = batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                valid_seq_len = a.cu_seqlens[batch_idx + 1] - token_start
            if a.is_packed == 1:
                storage_batch_idx = 0
            else:
                token_start = 0
            chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            for value_head_idx in range(0, a.value_heads):
                selected_core = 0
                for core_idx in range(1, a.core_num):
                    if chunk_core_costs[core_idx] < chunk_core_costs[selected_core]:
                        selected_core = core_idx
                a.metadata[task_cursor + CHUNK_TASK_BATCH] = batch_idx
                a.metadata[task_cursor + CHUNK_TASK_STORAGE_BATCH] = storage_batch_idx
                a.metadata[task_cursor + CHUNK_TASK_TOKEN_START] = token_start
                a.metadata[task_cursor + CHUNK_TASK_LENGTH] = valid_seq_len
                a.metadata[task_cursor + CHUNK_TASK_VALUE_HEAD] = value_head_idx
                a.metadata[task_cursor + CHUNK_TASK_NEXT] = 0
                previous_task = chunk_core_tails[selected_core]
                if previous_task == 0:
                    a.metadata[HEADER_WORDS + selected_core] = task_cursor
                else:
                    a.metadata[previous_task + CHUNK_TASK_NEXT] = task_cursor
                chunk_core_tails[selected_core] = task_cursor
                chunk_core_costs[selected_core] += chunk_num
                task_cursor += CHUNK_TASK_WORDS
        a.metadata[HEADER_CHUNK_TASK_NUM] = task_num
        return STATUS_OK

    max_chunk_group = a.max_chunk_group
    if a.template_kind == TEMPLATE_AUTO:
        max_chunk_group = _training_group_limit(a, route_max_chunk_num)
    batch_num_per_set = a.workspace_slots // a.value_heads
    stage12_round_num = 0
    total_active_batch_entries = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            if valid_seq_len <= 0:
                status = STATUS_BAD_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            elif a.is_packed == 0 and valid_seq_len > a.storage_length:
                status = STATUS_BAD_PADDED_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            else:
                chunk_num_by_batch[local_batch_idx] = (
                    valid_seq_len + CHUNK_SIZE - 1
                ) // CHUNK_SIZE
                tail_row_num_by_batch[local_batch_idx] = (
                    (valid_seq_len - 1) % CHUNK_SIZE
                ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break
            group_num = _select_group_num(
                remaining_chunk_num,
                active_batch_num,
                a.value_heads,
                a.workspace_slots,
                max_chunk_group,
            )
            stage12_round_num += 1
            total_active_batch_entries += active_batch_num
            chunk_start += group_num
        batch_start += batch_num_in_set

    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    record_begin = HEADER_WORDS + stage12_round_num + 1
    used_words = (
        record_begin
        + ROUND_FIXED_WORDS * stage12_round_num
        + ROUND_WORDS_PER_BATCH * total_active_batch_entries
    )
    if used_words > a.metadata_word_capacity:
        a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_OK

    a.metadata[HEADER_STAGE12_ROUND_NUM] = stage12_round_num
    record_cursor = record_begin
    global_round_idx = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            chunk_num_by_batch[local_batch_idx] = (
                valid_seq_len + CHUNK_SIZE - 1
            ) // CHUNK_SIZE
            tail_row_num_by_batch[local_batch_idx] = (
                (valid_seq_len - 1) % CHUNK_SIZE
            ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break

            group_num = _select_group_num(
                remaining_chunk_num,
                active_batch_num,
                a.value_heads,
                a.workspace_slots,
                max_chunk_group,
            )
            for active_idx in range(0, active_batch_num):
                current_chunk_num = remaining_chunk_num[active_idx]
                if current_chunk_num > group_num:
                    current_chunk_num = group_num
                chunk_num_per_group[active_idx] = current_chunk_num

            a.metadata[HEADER_WORDS + global_round_idx] = record_cursor
            cursor = record_cursor
            a.metadata[cursor] = global_round_idx
            a.metadata[cursor + 1] = active_batch_num
            cursor += 2
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = (
                    batch_start + active_local_batch_idx[active_idx]
                )
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                logical_batch_idx = batch_start + active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = 0
                else:
                    a.metadata[cursor + active_idx] = logical_batch_idx
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = token_start_by_batch[
                        local_batch_idx
                    ]
                else:
                    a.metadata[cursor + active_idx] = 0
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                a.metadata[cursor + active_idx] = valid_seq_len_by_batch[
                    local_batch_idx
                ]
            cursor += active_batch_num
            a.metadata[cursor] = group_num
            cursor += 1
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_start
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_num_per_group[active_idx]
            cursor += active_batch_num

            stage1_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage1_task_num += a.value_heads * chunk_num_per_group[active_idx]
                a.metadata[cursor + active_idx + 1] = stage1_task_num
            cursor += active_batch_num + 1

            stage1_task_idx = 0
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                for _ in range(0, a.value_heads):
                    for local_chunk_idx in range(0, chunk_num_per_group[active_idx]):
                        global_chunk_idx = chunk_start + local_chunk_idx
                        stage1_cost = FULL_CHUNK_COST_NS
                        if (
                            global_chunk_idx + 1 == chunk_num_by_batch[local_batch_idx]
                            and tail_row_num_by_batch[local_batch_idx] < CHUNK_SIZE
                        ):
                            stage1_cost = _stage1_task_cost(
                                tail_row_num_by_batch[local_batch_idx]
                            )
                        stage1_task_costs[stage1_task_idx] = stage1_cost
                        stage1_task_idx += 1
            _partition_costs(
                stage1_task_costs,
                stage1_task_num,
                a.core_num,
                stage1_core_ranges,
                stage1_core_costs,
            )
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = stage1_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            selected_dv_splits_num = 1
            selected_max_core_cost = 0
            selected_total_cost = 0
            selected_task_num = 0
            selected_is_set = 0
            for candidate_idx in range(0, 4):
                candidate_dv_splits_num = 1 << candidate_idx
                candidate_task_num = 0
                candidate_total_cost = 0
                for active_idx in range(0, active_batch_num):
                    task_cost = (
                        _stage2_task_cost(
                            candidate_idx, chunk_num_per_group[active_idx]
                        )
                        + a.stage2_overhead_ns * chunk_num_per_group[active_idx]
                    )
                    batch_task_num = a.value_heads * candidate_dv_splits_num
                    for _ in range(0, batch_task_num):
                        stage2_task_costs[candidate_task_num] = task_cost
                        candidate_task_num += 1
                        candidate_total_cost += task_cost
                _partition_costs(
                    stage2_task_costs,
                    candidate_task_num,
                    a.core_num,
                    candidate_stage2_core_ranges,
                    candidate_stage2_core_costs,
                )
                candidate_max_core_cost = 0
                for core_idx in range(0, MAX_AIC_CORES):
                    if candidate_stage2_core_costs[core_idx] > candidate_max_core_cost:
                        candidate_max_core_cost = candidate_stage2_core_costs[core_idx]

                candidate_is_better = 0
                if selected_is_set == 0:
                    candidate_is_better = 1
                elif candidate_max_core_cost < selected_max_core_cost:
                    candidate_is_better = 1
                elif candidate_max_core_cost == selected_max_core_cost:
                    if candidate_total_cost < selected_total_cost:
                        candidate_is_better = 1
                    elif candidate_total_cost == selected_total_cost:
                        if candidate_task_num < selected_task_num:
                            candidate_is_better = 1
                        elif candidate_task_num == selected_task_num:
                            if candidate_dv_splits_num < selected_dv_splits_num:
                                candidate_is_better = 1
                if candidate_is_better == 1:
                    selected_is_set = 1
                    selected_dv_splits_num = candidate_dv_splits_num
                    selected_max_core_cost = candidate_max_core_cost
                    selected_total_cost = candidate_total_cost
                    selected_task_num = candidate_task_num
                    for range_idx in range(0, MAX_AIC_CORES * 2):
                        selected_stage2_core_ranges[range_idx] = (
                            candidate_stage2_core_ranges[range_idx]
                        )

            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = selected_dv_splits_num
            cursor += active_batch_num
            stage2_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage2_task_num += a.value_heads * selected_dv_splits_num
                a.metadata[cursor + active_idx + 1] = stage2_task_num
            cursor += active_batch_num + 1
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = selected_stage2_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            record_cursor = cursor
            global_round_idx += 1
            chunk_start += group_num
        batch_start += batch_num_in_set

    a.metadata[HEADER_WORDS + stage12_round_num] = record_cursor
    return STATUS_OK


def _take(words: list[int], cursor: int, count: int) -> tuple[list[int], int]:
    end = cursor + count
    if cursor < 0 or count < 0 or end > len(words):
        raise ValueError("metadata record extends past available words")
    return words[cursor:end], end


def _decode_ranges(flat: list[int]) -> list[tuple[int, int]]:
    if len(flat) != MAX_AIC_CORES * 2:
        raise ValueError("core range block must contain 64 words")
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)]


def _validate_prefix(prefix: list[int], name: str) -> None:
    if not prefix or prefix[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(end < start for start, end in zip(prefix, prefix[1:])):
        raise ValueError(f"{name} must be nondecreasing")


def _validate_ranges(ranges: list[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    saw_empty = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            saw_empty = True
            continue
        if saw_empty:
            raise ValueError(f"{name} has work after an unused core")
        if start != cursor or end <= start or end > total:
            raise ValueError(f"{name} is not a contiguous partition")
        cursor = end
    if cursor != total:
        raise ValueError(f"{name} does not cover its task total")


def _decode_round(
    words: list[int],
    start: int,
    end: int,
    *,
    batch: int,
    value_heads: int,
) -> dict[str, object]:
    if start < 0 or end > len(words) or start + 2 > end:
        raise ValueError("metadata round record is truncated")
    # 将后续所有 _take 和标量访问限制在当前记录范围内。
    words = words[start:end]
    start, end = 0, end - start
    cursor = 0
    stage12_round_idx, active_batch_num = words[cursor : cursor + 2]
    cursor += 2
    if not 1 <= active_batch_num <= min(batch, WORKSPACE_SLOTS):
        raise ValueError("metadata active_batch_num outside batch/workspace bounds")
    if start + round_record_words(active_batch_num) != end:
        raise ValueError("metadata round record length does not match active batches")

    batch_idx, cursor = _take(words, cursor, active_batch_num)
    storage_batch_idx, cursor = _take(words, cursor, active_batch_num)
    token_start, cursor = _take(words, cursor, active_batch_num)
    valid_seq_len, cursor = _take(words, cursor, active_batch_num)
    group_num = words[cursor]
    cursor += 1
    chunk_start_per_group, cursor = _take(words, cursor, active_batch_num)
    chunk_num_per_group, cursor = _take(words, cursor, active_batch_num)
    stage1_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage1_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    dv_splits_num, cursor = _take(words, cursor, active_batch_num)
    stage2_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage2_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    if cursor != end:
        raise ValueError("metadata round decoder did not consume the full record")

    if group_num <= 0:
        raise ValueError("group_num must be positive")
    if any(index < 0 or index >= batch for index in batch_idx):
        raise ValueError("batch_idx is outside logical batch")
    if any(length <= 0 for length in valid_seq_len):
        raise ValueError("valid_seq_len must be positive")
    if any(count <= 0 for count in chunk_num_per_group):
        raise ValueError("chunk_num_per_group must be positive")
    if any(value not in (1, 2, 4, 8) for value in dv_splits_num):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")

    _validate_prefix(stage1_task_prefix, "stage1_task_prefix")
    _validate_prefix(stage2_task_prefix, "stage2_task_prefix")
    for index, chunk_num in enumerate(chunk_num_per_group):
        if stage1_task_prefix[index + 1] - stage1_task_prefix[index] != (
            value_heads * chunk_num
        ):
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_task_prefix[index + 1] - stage2_task_prefix[index] != (
            value_heads * dv_splits_num[index]
        ):
            raise ValueError("stage2_task_prefix does not match DV tasks")
    if stage1_task_prefix[-1] > WORKSPACE_SLOTS:
        raise ValueError("stage1 task total exceeds workspace slots")
    if stage2_task_prefix[-1] > WORKSPACE_SLOTS * 8:
        raise ValueError("stage2 task total exceeds workspace slots")

    stage1_core_ranges = _decode_ranges(stage1_ranges_flat)
    stage2_core_ranges = _decode_ranges(stage2_ranges_flat)
    _validate_ranges(stage1_core_ranges, stage1_task_prefix[-1], "stage1_core_ranges")
    _validate_ranges(stage2_core_ranges, stage2_task_prefix[-1], "stage2_core_ranges")
    return {
        "stage12_round_idx": stage12_round_idx,
        "active_batch_num": active_batch_num,
        "batch_idx": batch_idx,
        "storage_batch_idx": storage_batch_idx,
        "token_start": token_start,
        "valid_seq_len": valid_seq_len,
        "group_num": group_num,
        "chunk_start_per_group": chunk_start_per_group,
        "chunk_num_per_group": chunk_num_per_group,
        "stage1_task_prefix": stage1_task_prefix,
        "stage1_core_ranges": stage1_core_ranges,
        "dv_splits_num": dv_splits_num,
        "stage2_task_prefix": stage2_task_prefix,
        "stage2_core_ranges": stage2_core_ranges,
    }


def decode_metadata(raw: Sequence[int]) -> dict[str, object]:
    """解码并检查 metadata，供测试和诊断使用。"""

    words = [int(value) for value in raw]
    if len(words) < HEADER_WORDS:
        raise ValueError("metadata is shorter than its header")
    if words[HEADER_MAGIC] != MAGIC:
        raise ValueError("metadata magic does not match")
    if words[HEADER_ABI_VERSION] != ABI_VERSION:
        raise ValueError("metadata ABI version does not match")

    status = words[HEADER_STATUS]
    header = {
        "magic": words[HEADER_MAGIC],
        "abi_version": words[HEADER_ABI_VERSION],
        "status": status,
        "core_num": words[HEADER_CORE_NUM],
        "stage12_round_num": words[HEADER_STAGE12_ROUND_NUM],
        "batch": words[HEADER_BATCH],
        "value_heads": words[HEADER_VALUE_HEADS],
        "has_cu": words[HEADER_HAS_CU],
        "workspace_slots": words[HEADER_WORKSPACE_SLOTS],
        "template": {TEMPLATE_STAGED: "staged", TEMPLATE_PERSISTENT: "persistent"}.get(
            words[HEADER_TEMPLATE], "unknown"
        ),
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError("metadata core_num must be in [1, 32]")
    if header["workspace_slots"] != WORKSPACE_SLOTS:
        raise ValueError(f"metadata workspace_slots must be {WORKSPACE_SLOTS}")
    if header["batch"] <= 0:
        raise ValueError("successful metadata requires positive batch")
    if not 1 <= header["value_heads"] <= MAX_VALUE_HEADS:
        raise ValueError("metadata value_heads must be in [1, 96]")

    if header["has_cu"] not in (0, 1):
        raise ValueError("metadata has_cu must be 0 or 1")
    if header["template"] == "unknown":
        raise ValueError("metadata template is invalid")
    if header["template"] == "persistent":
        task_num = words[HEADER_CHUNK_TASK_NUM]
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        if (
            header["stage12_round_num"] != 0
            or task_num != header["batch"] * header["value_heads"]
        ):
            raise ValueError("chunk metadata task count does not match its shape")
        if used_words > len(words):
            raise ValueError("chunk metadata tasks are truncated")
        seen_offsets = set()
        seen_jobs = set()
        tasks_by_core = []
        for core_idx in range(MAX_AIC_CORES):
            cursor = words[HEADER_WORDS + core_idx]
            if core_idx >= header["core_num"] and cursor:
                raise ValueError("inactive core has chunk tasks")
            tasks = []
            while cursor:
                if cursor in seen_offsets:
                    raise ValueError("chunk task chain contains a cycle or duplicate")
                if (
                    cursor < task_base
                    or cursor + CHUNK_TASK_WORDS > used_words
                    or (cursor - task_base) % CHUNK_TASK_WORDS
                ):
                    raise ValueError("chunk task pointer is out of bounds or unaligned")
                seen_offsets.add(cursor)
                batch_idx = words[cursor + CHUNK_TASK_BATCH]
                head_idx = words[cursor + CHUNK_TASK_VALUE_HEAD]
                if (
                    not 0 <= batch_idx < header["batch"]
                    or not 0 <= head_idx < header["value_heads"]
                ):
                    raise ValueError("chunk task batch/head index is out of bounds")
                if (batch_idx, head_idx) in seen_jobs:
                    raise ValueError("chunk sequence/head task is duplicated")
                seen_jobs.add((batch_idx, head_idx))
                length = words[cursor + CHUNK_TASK_LENGTH]
                storage_batch = words[cursor + CHUNK_TASK_STORAGE_BATCH]
                token_start = words[cursor + CHUNK_TASK_TOKEN_START]
                if (
                    length <= 0
                    or token_start < 0
                    or storage_batch not in (0, batch_idx)
                ):
                    raise ValueError("chunk task has invalid sequence coordinates")
                tasks.append(
                    dict(
                        batch_idx=batch_idx,
                        storage_batch_idx=storage_batch,
                        token_start=token_start,
                        sequence_length=length,
                        value_head_idx=head_idx,
                    ),
                )
                cursor = words[cursor + CHUNK_TASK_NEXT]
            tasks_by_core.append(tasks)
        if len(seen_jobs) != task_num:
            raise ValueError("chunk metadata does not cover every task")
        header.update(
            used_words=used_words,
            stage12_rounds=[],
            chunk_task_num=task_num,
            chunk_tasks_by_core=tasks_by_core,
        )
        return header

    if words[HEADER_CHUNK_TASK_NUM] != 0:
        raise ValueError("staged metadata must not contain persistent tasks")
    stage12_round_num = int(header["stage12_round_num"])
    if stage12_round_num < 1:
        raise ValueError("staged metadata round count must be positive")
    offset_begin = HEADER_WORDS
    offset_end = offset_begin + stage12_round_num + 1
    if offset_end > len(words):
        raise ValueError("metadata offset table is truncated")
    min_record_words = round_record_words(1)
    if offset_end + stage12_round_num * min_record_words > len(words):
        raise ValueError("metadata capacity is insufficient for round records")
    offsets = words[offset_begin:offset_end]
    if any(offset < 0 for offset in offsets):
        raise ValueError("metadata offset must be nonnegative")
    if offsets[0] < offset_end:
        raise ValueError("metadata first record offset overlaps the offset table")
    if offsets[-1] > len(words):
        raise ValueError("metadata final offset exceeds capacity")
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("metadata offsets must be strictly increasing")

    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = [
        _decode_round(
            words,
            offsets[index],
            offsets[index + 1],
            batch=int(header["batch"]),
            value_heads=int(header["value_heads"]),
        )
        for index in range(stage12_round_num)
    ]
    return header


_ROUTE_PARAMETERS = (
    0,
    2711857704,
    699286307,
    16555661144,
    28359460,
    94263834,
    94263834,
    12013349865,
    1045656019,
    84358156,
    12842477,
    2527878117,
    181898838,
    6,
    8441,
    0,
    0,
    692059912,
    851406583,
    21510815509,
    73490570,
    81791118,
    166178779,
    12029971961,
    2552246171,
    459139222,
    45086212,
    2303638937,
    143977434,
    6,
    18992,
    0,
)

_DT_ROUTE_PARAMETERS = (
    4135271144,
    193981546,
    366448654,
    12881511041,
    29417161,
    4351059785,
    158980371,
    10652650460,
    3206978745,
    332321804,
    0,
    419242242,
    29945874,
    7,
    0,
    0,
    20317858430,
    1160598799,
    1018161115,
    1015361233,
    14389959,
    933231823,
    115749683,
    11499084207,
    4373896639,
    413928806,
    41555011,
    0,
    0,
    7,
    0,
    0,
)


def build_schedule(
    *,
    batch,
    nheads_qk,
    nheads_v,
    storage_length,
    layout,
    lengths=None,
    enable_grad=False,
    core_num=28,
    template="auto",
    device_kind="PR",
    stage2_overhead_ns=0,
):
    """ABI4 scheduling with the native workspace capacity.

    The original 861-slot capacity is checked separately against fixed-source
    ABI words; production uses WORKSPACE_SLOTS consistently for allocations.

    Lengths are already validated host metadata; None means unpadded uniform input.
    The source has no empty-sequence route. Do not silently invent one here.
    A nonzero native per-chunk Stage2 overhead changes only its DV cost ranking;
    all Stage1 groups, numerical formulas and sequence ownership remain intact.
    """
    if layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("invalid layout")
    if not 0 < nheads_qk <= nheads_v <= 96 or nheads_v % nheads_qk:
        raise ValueError("invalid GQA heads")
    if template not in ("auto", "staged", "persistent"):
        raise ValueError("invalid template")
    has_cu = lengths is not None
    if layout == "TND" and not has_cu:
        raise ValueError("TND requires lengths")
    lengths = [storage_length] * batch if lengths is None else list(lengths)
    if len(lengths) != batch or any(x <= 0 for x in lengths):
        raise ValueError("source schedule requires positive sequence lengths")
    if layout == "TND" and sum(lengths) != storage_length:
        raise ValueError("packed lengths must cover storage")
    prefix = [0]
    for size in lengths:
        prefix.append(prefix[-1] + size)
    capacity = metadata_capacity_upper_bound(
        batch, storage_length, is_packed=layout == "TND", value_heads=nheads_v
    )
    short = (
        enable_grad
        and layout == "BNSD"
        and batch == 1
        and nheads_qk == nheads_v == SHORT_TRAIN_HEADS
        and storage_length >= SHORT_TRAIN_MIN_LENGTH
        and storage_length % CHUNK_SIZE == 0
        and not has_cu
        and core_num == MAX_AIC_CORES
    )
    words = [0] * capacity
    args = SimpleNamespace(
        cu_seqlens=prefix,
        route_coefficients=_DT_ROUTE_PARAMETERS
        if device_kind == "DT"
        else _ROUTE_PARAMETERS,
        calibration_enabled=1,
        layout_kind=0 if layout == "BNSD" else 1,
        metadata=words,
        metadata_word_capacity=capacity,
        batch=batch,
        physical_batches=1 if layout == "TND" else batch,
        storage_length=storage_length,
        value_heads=nheads_v,
        query_heads=nheads_qk,
        has_cu=int(has_cu),
        is_packed=int(layout == "TND"),
        core_num=core_num,
        workspace_slots=WORKSPACE_SLOTS,
        template_kind={"auto": 0, "staged": 1, "persistent": 2}[template],
        max_chunk_group=SHORT_TRAIN_CHUNK_GROUP if short else MAX_CHUNK_GROUP,
        stage2_overhead_ns=stage2_overhead_ns,
    )
    status = flash_kda_fwd_metadata_kernel(args)
    if status != STATUS_OK:
        raise RuntimeError(f"metadata kernel returned status {status}")
    decoded = decode_metadata(words)
    if decoded["status"]:
        raise ValueError(f"source schedule status {decoded['status']}")
    return words[: decoded["used_words"]], decoded


# -----------------------------------------------------------------------------
# Runtime caches and public forward entry.
# -----------------------------------------------------------------------------

_COMPILED = {}
_LOCK = Lock()
_IDENTITIES = {}


def _identity16(device):
    """Cache only the immutable constant; publish it safely across streams."""
    stream = torch.npu.current_stream(device)
    with _LOCK:
        entry = _IDENTITIES.get(device.index)
        if entry is None:
            value = torch.eye(16, dtype=torch.float16, device=device)
            ready = stream.record_event()
            _IDENTITIES[device.index] = (value, ready, stream)
        else:
            value, ready, owner = entry
            if stream != owner:
                stream.wait_event(ready)
    return value


def flash_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    metadata=None,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    *,
    scale=128**-0.5,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    lower_bound=None,
    layout="TND",
):
    flags = dict(
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
    )
    batch, nq, nv, length, lengths = _validate(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        scale=scale,
        layout=layout,
        flags=flags,
        lower_bound=lower_bound,
    )
    training = not torch.is_inference_mode_enabled()
    save = training and disable_recompute
    norm = training and use_qk_l2norm_in_kernel
    save_beta = training and use_beta_sigmoid_in_kernel
    scale = 128**-0.5 if scale is None else float(scale)
    mode = 0 if not use_gate_in_kernel else (2 if lower_bound is None else 1)
    lower = -0.1 if mode != 1 else float(lower_bound)
    packed = layout == "TND"
    chunks = (length + 63) // 64
    chunk_shape = (chunks + batch,) if packed else (batch, chunks)
    norm_shape = (length, nq) if packed else (batch, nq, length)
    beta_shape = (length, nv) if packed else (batch, length, nv)
    with torch.npu.device(q.device):
        properties = torch.npu.get_device_properties(q.device)
        cores = properties.cube_core_num
        native_template = "auto"
        if (
            not training
            and use_qk_l2norm_in_kernel
            and mode == 1
            and lower == -5
            and abs(scale) <= 0.125
            and all(n % 64 == 0 for n in lengths)
        ):
            native_template = "persistent"
        try:
            schedule_result = build_schedule(
                batch=batch,
                nheads_qk=nq,
                nheads_v=nv,
                storage_length=length,
                layout=layout,
                lengths=lengths
                if cu_seqlens is not None or seqused is not None
                else None,
                enable_grad=training,
                core_num=cores,
                device_kind="DT" if "DT" in properties.name else "PR",
                stage2_overhead_ns=1000,
                template=native_template,
            )
        except ValueError as error:
            raise ValueError(
                f"Flash KDA schedule validation failed: {error}"
            ) from error
        except RuntimeError as error:
            raise RuntimeError(
                f"Flash KDA schedule generation failed: {error}"
            ) from error
        if not isinstance(schedule_result, tuple) or len(schedule_result) != 2:
            raise RuntimeError("Flash KDA schedule returned an invalid result")
        words, decoded = schedule_result
        if decoded["status"] != STATUS_OK:
            raise RuntimeError(
                f"Flash KDA schedule returned status {decoded['status']}"
            )

        def allocate(shape, dtype=torch.bfloat16, zero=False):
            element_size = 2 if dtype == torch.bfloat16 or dtype == torch.float16 else 4
            _require(
                math.prod(shape) <= (2**63 - 1) // element_size,
                "tensor byte size exceeds signed 64-bit addressing",
            )
            factory = torch.zeros if zero else torch.empty
            return factory(shape, dtype=dtype, device=q.device)

        initial = (
            initial_state
            if initial_state is not None
            else allocate((batch, nv, 128, 128), torch.float32, True)
        )
        alpha = A_log if use_gate_in_kernel else None
        bias = dt_bias if use_gate_in_kernel else None
        if alpha is None:
            alpha = allocate((nv,), torch.float32, True)
        if bias is None:
            bias = allocate((nv, 128), torch.float32, True)
        exchange = allocate((cores, 256), torch.float32)
        identity = _identity16(q.device)
        schedule = torch.tensor(words, dtype=torch.int64, device=q.device)
        staged = decoded["template"] == "staged"
        aq = allocate(chunk_shape + (nv, 64, 64) if training or staged else (1,))
        ak = allocate(chunk_shape + (nv, 64, 64) if training else (1,))
        cache_shape = tuple(v.shape) if save or staged else (1,)
        w, u, kg = [allocate(cache_shape) for _ in range(3)]
        qg, vnew = [allocate(tuple(v.shape) if save else (1,)) for _ in range(2)]
        out = allocate(
            tuple(v.shape), zero=not packed and any(n < length for n in lengths)
        )
        qs = allocate((WORKSPACE_SLOTS, 64, 128) if staged else (1,))
        gamma = allocate((WORKSPACE_SLOTS, 128) if staged else (1,), torch.float32)
        qn, kn = [allocate(norm_shape + (128,) if norm else (1,)) for _ in range(2)]
        rstd = allocate((2,) + norm_shape if norm else (1,), torch.float32)
        # Packed capacity includes unused chunk slots; padded seqused may leave
        # whole unused chunks even when each valid length is a multiple of 64.
        gc = allocate(
            chunk_shape + (nv, 64, 128) if save else (1,),
            torch.float32,
            save and (packed or any(n % 64 or n < length for n in lengths)),
        )
        bact = allocate(beta_shape if save_beta else (1,), torch.float32)
        history = allocate(
            chunk_shape + (nv, 128, 128)
            if return_intermediate_states or staged
            else (1,),
            torch.float32,
        )
        final = allocate((batch, nv, 128, 128), torch.float32)
        tensors = [
            q,
            k,
            v,
            g,
            beta,
            alpha,
            bias,
            exchange,
            identity,
            initial,
            schedule,
            aq,
            ak,
            w,
            u,
            qg,
            kg,
            qs,
            gamma,
            qn,
            kn,
            rstd,
            gc,
            bact,
            out,
            vnew,
            history,
            final,
        ]
        bound = []
        for tensor in tensors:
            bound.append(
                tla.from_dlpack(tensor.flatten(), layout_tag=tla.arch.RowMajor)
            )
        config = (
            batch,
            length,
            nq,
            nv,
            layout,
            mode,
            allow_neg_eigval,
            lower,
            scale,
            use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel,
        )
        options = dict(
            save_matrices=training,
            save_cache=save,
            save_history=return_intermediate_states,
            save_norm=norm,
            save_beta=save_beta,
        )
        if staged:
            options["pipeline"] = True
        else:
            options.update(prefetch=True, bulk_operands=True, pipeline=True)
        key = (
            decoded["template"],
            config,
            tuple(sorted(options.items())),
            tuple((x.numel(), x.dtype) for x in tensors),
        )
        with _LOCK:
            binary = _COMPILED.get(key)
            if binary is None:
                cls = (
                    PersistentMetadataKernel
                    if decoded["template"] == "persistent"
                    else StagedMetadataKernel
                )
                binary = tla.compile(
                    cls(*config, **options), *bound, options="--npu-arch 3510"
                )
                if len(_COMPILED) >= 128:
                    _COMPILED.pop(next(iter(_COMPILED)))
                _COMPILED[key] = binary
        stream = torch.npu.current_stream(q.device)
        for tensor in tensors:
            tensor.record_stream(stream)
        binary(*bound, block_num=cores, stream=stream.npu_stream)
    return (
        out,
        final if output_final_state else None,
        history if return_intermediate_states else None,
        aq if training else None,
        ak if training else None,
        w if save else None,
        u if save else None,
        qg if save else None,
        kg if save else None,
        vnew if save else None,
        gc if save else None,
        qn if norm else None,
        rstd[0] if norm else None,
        kn if norm else None,
        rstd[1] if norm else None,
        bact if training and use_beta_sigmoid_in_kernel else None,
    )
