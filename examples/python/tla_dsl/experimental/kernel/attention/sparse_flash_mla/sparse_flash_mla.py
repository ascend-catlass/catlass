# ---------------------------------------------------------------------------
# sparse_flash_mla.py — SparseFlashMLA 样例的 host 薄入口
#
# 实现对齐 arch35 sparse_flash_mla 的算法与缓冲划分，按 workload 形态分派到
# 5 个分支 kernel：
#   swa  : 稠密 ori 段（dense / causal / 滑动窗口），无 compressed KV
#   hca  : ori / cmp 都是连续段（compressed KV，无稀疏索引）
#   csa  : ori 连续 + cmp 稀疏索引
#   os   : ori 稀疏索引（ORI_SPARSE），可选写回 lse
#   ocs  : ori / cmp 双稀疏索引
#
# 本文件只做参数校验、tiling / workspace / 物理地址表准备、from_dlpack 绑定与
# executor 缓存，device 侧代码全部在被导入的 8 个模块里。
# 公开 run(...) 签名与返回值（out, lse）与旧版样例保持一致。
# ---------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

from sparse_flash_mla_csa_kernel import sparse_flash_mla_csa_kernel
from sparse_flash_mla_hca_kernel import sparse_flash_mla_hca_kernel
from sparse_flash_mla_ori_cmp_sparse_kernel import (
    sparse_flash_mla_ori_cmp_sparse_kernel,
)
from sparse_flash_mla_ori_sparse_kernel import sparse_flash_mla_ori_sparse_kernel
from sparse_flash_mla_swa_kernel import sparse_flash_mla_swa_kernel

HEAD_DIM = 512
KV_BLOCK = 128
NPU_ARCH = "--npu-arch 3510"
# UB 默认预算是 253952 B（256 KiB 内编译器预留 8 KiB）。个别形态（奇数 G 的
# hca preload）静态占用 257288 B，需要释放编译器预留才能放满 262144 B；释放
# 开关会改变编译配置与产物，所以只在默认预算编译失败时按需重编一次，其余形态
# 的编译配置与不带开关时完全一致。
UB_RELEASE_FLAGS = (
    " --cce-disable-asc-reserved-ubuf --cce-disable-vf-stack-reserved-ubuf"
)
# 稀疏路径的 GM 暂存槽：按 AIC 核分片，每核 KV_SLOTS(3) x 128 x 512 元素。
WS_BLOCKS = 32
KV_STAGE_BYTES_EL = 3 * 128 * 512

_LAYOUT_CONST = {"BSND": 0, "TND": 1, "PA_BBND": 2}

# ------------------------------------------------------------------ 设备侧缓存

# 输出 / lse 缓冲池：kernel 会写满全部输出元素，无需零初始化；而在有在飞
# kernel 时新分配几十 MB 设备内存约需 0.25~0.35 ms（acl 侧同步查询）。
_OUT_POOL = {}
_LSE_POOL = {}
_ZERO_LEN = {}
_OUT_POOL_SLOTS = 4

# 执行器缓存：按 (device, dtype, 张量签名, 全部 Constexpr) 复用编译结果。
_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_LIMIT = 16

# from_dlpack 绑定缓存：按对象身份复用，持有源张量避免 id 复用。
_BINDING_CACHE = {}
_BINDING_CACHE_LIMIT = 48

_DUMMY_CACHE = {}
_DEV_PROP = {}


def _pooled_output(q):
    key = (str(q.device), tuple(q.shape), q.dtype)
    state = _OUT_POOL.get(key)
    if state is None:
        state = {"bufs": [], "i": 0}
        _OUT_POOL[key] = state
    while len(state["bufs"]) < _OUT_POOL_SLOTS:
        state["bufs"].append(torch.empty_like(q))
    out = state["bufs"][state["i"] % _OUT_POOL_SLOTS]
    state["i"] += 1
    return out


def _pooled_lse(device, shape):
    key = (str(device), tuple(shape))
    state = _LSE_POOL.get(key)
    if state is None:
        state = {"bufs": [], "i": 0}
        _LSE_POOL[key] = state
    while len(state["bufs"]) < _OUT_POOL_SLOTS:
        state["bufs"].append(
            torch.empty(tuple(shape), dtype=torch.float32, device=device)
        )
    out = state["bufs"][state["i"] % _OUT_POOL_SLOTS]
    state["i"] += 1
    return out


def _zero_len(device):
    key = str(device)
    cached = _ZERO_LEN.get(key)
    if cached is None:
        cached = torch.empty((0,), dtype=torch.float32, device=device)
        _ZERO_LEN[key] = cached
    return cached


def _cube_cores(device):
    key = str(device)
    cores = _DEV_PROP.get(key)
    if cores is None:
        cores = int(torch_npu.npu.get_device_properties(device).cube_core_num)
        if cores < 1:
            raise RuntimeError("NPU device reports no available Cube cores")
        _DEV_PROP[key] = cores
    return cores


def _tla_tensor(value, layout_tag=None):
    key = id(value)
    hit = _BINDING_CACHE.get(key)
    if hit is not None and hit[0] is value:
        return hit[1]
    contiguous = value.contiguous()
    bound = from_dlpack(contiguous, layout_tag=layout_tag or tla.arch.RowMajor)
    # 让绑定对象持有源存储，避免 DLPack capsule 生命周期问题。
    bound._torch_storage = contiguous
    if len(_BINDING_CACHE) >= _BINDING_CACHE_LIMIT:
        _BINDING_CACHE.clear()
    _BINDING_CACHE[key] = (value, bound)
    return bound


def _dummy(device, dtype):
    key = (str(device), str(dtype))
    cached = _DUMMY_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(1, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = cached
    return cached


def _align128(n):
    return (int(n) + 127) // 128 * 128


def _check_tensor(name, value, dtype, ndim=None):
    if value is None:
        raise ValueError(f"{name} is required")
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D")


# ------------------------------------------------------------------ host 计算


def _build_cu(cu, seqused, batch, per_batch, device):
    """返回 (int32[B+1] 设备张量, Python list)。

    cu_seqlens 优先；否则由 seqused 求前缀和；两者都缺时按每 batch 定长重建。
    """
    i32 = torch.int32
    if cu is not None:
        _check_tensor("cu_seqlens", cu, i32, 1)
        t = cu.to(i32).contiguous()
        return t, [int(v) for v in t.reshape(-1).tolist()]
    if seqused is not None:
        _check_tensor("seqused", seqused, i32, 1)
        vals = [int(v) for v in seqused.reshape(-1).tolist()]
        acc = [0]
        for v in vals:
            acc.append(acc[-1] + v)
        return torch.tensor(acc, dtype=i32, device=device), acc
    if per_batch is not None:
        n, step = int(per_batch[0]), int(per_batch[1])
        acc = [i * step for i in range(n + 1)]
        return torch.tensor(acc, dtype=i32, device=device), acc
    if batch is None:
        raise ValueError("cannot derive cu_seqlens: pass cu_seqlens_* or seqused_*")
    acc = [0] * (int(batch) + 1)
    return torch.tensor(acc, dtype=i32, device=device), acc


def _block_table(bt, kv, device):
    """返回 (展平 int32 设备张量, block_size, pa_stride, max_blocks, pages)。"""
    i32 = torch.int32
    if bt is None:
        return torch.zeros(1, dtype=i32, device=device), 0, 0, 0, 1
    _check_tensor("block_table", bt, i32, 2)
    b = bt.to(i32).contiguous()
    block_size = int(kv.shape[1])
    pa_stride = int(kv.numel() // kv.shape[0])
    max_blocks = int(b.shape[1])
    pages = 1 + (KV_BLOCK - 1 + max(block_size, 1) - 1) // max(block_size, 1)
    return b.reshape(-1), block_size, pa_stride, max_blocks, pages


def _vectorize_flag(route, layout_kv, ori_kv, cmp_kv, ori_bt, cmp_bt, ori_w, cmp_w):
    """OT CalcVectorizeKvPhyAddrWorkspaceSize（sparse_flash_mla_tiling.cpp）。

    只有 csa / os / ocs 的稀疏物理地址路径会置位；要求 PA 页大小为 2 的幂
    （os 只看 ori，csa/ocs 要求 ori 与 cmp 同时满足），且向量化 UB 需求
    ≤ 184 KiB。
    """
    if route not in ("csa", "os", "ocs"):
        return 0
    if layout_kv == "PA_BBND":
        ori_bs = int(ori_kv.shape[1])
        ori_ok = int(ori_bs & (ori_bs - 1) == 0)
        if route == "os":
            blocksize_flag = ori_ok
            pa = int(ori_bt.shape[1]) * 4 if ori_bt is not None else 0
        else:
            cmp_bs = int(cmp_kv.shape[1])
            cmp_ok = int(cmp_bs & (cmp_bs - 1) == 0)
            blocksize_flag = int(ori_ok and cmp_ok)
            max_blocks = 0
            if ori_bt is not None:
                max_blocks = max(max_blocks, int(ori_bt.shape[1]))
            if cmp_bt is not None:
                max_blocks = max(max_blocks, int(cmp_bt.shape[1]))
            pa = max_blocks * 4
    else:
        blocksize_flag = 1
        pa = 0
    vec_ub = max(pa + _align128(cmp_w) * 12, pa + _align128(ori_w) * 12)
    if blocksize_flag == 0 or vec_ub > 184 * 1024:
        return 0
    return 1


def _fa_s2_max_num(
    route,
    cu_q,
    cu_kv,
    cu_cmp,
    ori_topk,
    cmp_topk,
    ori_w,
    cmp_w,
    ori_mask_mode,
    win_l,
    win_r,
    cmp_mask_mode,
    cmp_ratio,
    resid,
    block_num,
    g_split,
):
    """OT FA_S2_MAX_NUM：各 AIC 上 s2 loop 数的最大值。

    kernel 用它作 IS_SPLIT_G 的 MTE2 屏障配额（tiling[1]）。预算只决定同步
    密度、不影响正确性，但按 OT 公式逐 token 复算可保持与参考实现同形。
    """
    nseq = len(cu_q) - 1
    kv_block = KV_BLOCK

    def _ceil_div(a, b):
        return (a + b - 1) // b if a > 0 else 0

    tiles = []
    for tok in range(len(cu_q) - 1):
        seq = 0
        rowq = tok
        for bb in range(1, nseq):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        causal = kv_len - q_len
        next_tok = kv_len
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal
        elif ori_mask_mode == 4:
            pre_tok = q_len if win_l == -1 else win_l - causal
            next_tok = kv_len if win_r == -1 else causal + win_r
            if pre_tok > q_len:
                pre_tok = q_len
        end = max(0, min(rowq + next_tok + 1, kv_len))
        start = max(0, min(rowq - pre_tok, kv_len))
        range_len = max(0, end - start)

        n = 0
        cmp_lines = 0
        if route in ("csa", "ocs"):
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]
            next_cmp = cmp_len * cmp_ratio
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cmp_lines = min(max(0, (rowq + 1 + next_cmp) // cmp_ratio), cmp_len)
            if cmp_topk is not None:
                cmp_lines = min(cmp_lines, max(0, min(cmp_topk[tok], cmp_w)))
        if route == "csa":
            n += _ceil_div(range_len, kv_block)
            n += _ceil_div(cmp_lines, kv_block)
        elif route == "os":
            lines = min(range_len, kv_len)
            if ori_topk is not None:
                lines = min(lines, max(0, min(ori_topk[tok], ori_w)))
            n += _ceil_div(lines, kv_block)
        else:  # ocs
            ori_lines = min(range_len, kv_len)
            if ori_topk is not None:
                ori_lines = min(ori_lines, max(0, min(ori_topk[tok], ori_w)))
            n += _ceil_div(ori_lines, kv_block)
            n += _ceil_div(cmp_lines, kv_block)
        tiles.append(n)

    per_core = [0] * max(block_num, 1)
    for t in range(len(tiles) * g_split):
        per_core[t % max(block_num, 1)] += tiles[t // g_split]
    return max(per_core) if per_core else 0


# ------------------------------------------------------------------ 公开入口


def run(
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    sinks,
    metadata,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
    return_softmax_lse,
    *,
    compile_only=False,
):
    # ------------------------------------------------ can_implement
    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if int(q.shape[-1]) != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    heads = int(q.shape[-2])
    if heads < 1:
        raise ValueError("q must have at least one head")
    if layout_q not in ("TND", "BSND"):
        raise ValueError("layout_q must be TND or BSND")
    if layout_kv not in ("TND", "BSND", "PA_BBND"):
        raise ValueError("layout_kv must be TND, BSND or PA_BBND")
    # 分页 KV 与 query 布局独立：BSND query 按 batch 行、TND query 按展平行寻址，
    # 页表只影响 KV 侧，两种 query 布局都支持。非分页时两侧布局必须一致。
    if layout_kv != "PA_BBND" and layout_q != layout_kv:
        raise ValueError("non-paged Q and KV layouts must match")

    _check_tensor("ori_kv", ori_kv, dtype)
    if int(ori_kv.shape[-2]) != 1:
        raise ValueError("only a single KV head (num_kv_heads == 1) is supported")

    ori_mask_mode = int(ori_mask_mode)
    cmp_mask_mode = int(cmp_mask_mode)
    cmp_ratio = int(cmp_ratio)
    if ori_mask_mode not in (0, 3, 4):
        raise ValueError("ori_mask_mode must be 0, 3 or 4")
    if cmp_mask_mode not in (0, 3):
        raise ValueError("cmp_mask_mode must be 0 or 3")
    if cmp_ratio < 1:
        raise ValueError("cmp_ratio must be positive")
    if cmp_kv is None and cmp_mask_mode != 0:
        raise ValueError("cmp_mask_mode must be 0 without compressed KV")
    if cmp_kv is None and cmp_sparse_indices is not None:
        raise ValueError("compressed sparse indices require compressed KV")
    # metadata 是 host 侧生成的预留元数据（tiling / 调度提示），本实现不使用；
    # 只校验它是 1-D int32，长度随上游 op 版本（1024 / 2048 等）而定。
    if metadata is None or metadata.dtype != torch.int32:
        raise ValueError("metadata must be int32")
    if metadata.ndim != 1:
        raise ValueError("metadata must be 1-D")
    scale_value = float(softmax_scale)
    want_lse = bool(return_softmax_lse)

    has_cmp = cmp_kv is not None
    has_ori_idx = ori_sparse_indices is not None
    has_cmp_idx = cmp_sparse_indices is not None

    # ------------------------------------------------ 形状 / batch
    paged = layout_kv == "PA_BBND"
    if layout_q == "TND":
        if q.ndim != 3:
            raise ValueError("TND query layout requires q [T1, N1, 512]")
        if cu_seqlens_q is None and seqused_q is None:
            raise ValueError("TND query layout requires int32 cu_seqlens_q")
        batch = (
            int(cu_seqlens_q.numel()) - 1
            if cu_seqlens_q is not None
            else int(seqused_q.numel())
        )
        total_rows = int(q.shape[0])
        s1_len = 0
    else:
        if q.ndim != 4:
            raise ValueError("BSND query layout requires q [B, S1, N1, 512]")
        batch = int(q.shape[0])
        s1_len = int(q.shape[1])
        total_rows = batch * s1_len

    if paged:
        if ori_kv.ndim != 4:
            raise ValueError("PA_BBND KV requires [blocks, page, 1, 512]")
        kv_bs = 0
        if seqused_ori_kv is None:
            raise ValueError("paged KV requires seqused_ori_kv")
        if ori_block_table is None or ori_block_table.dtype != torch.int32:
            raise ValueError("paged KV requires int32 ori_block_table")
        _check_tensor("ori_block_table", ori_block_table, torch.int32, 2)
        ori_bts = int(ori_block_table.shape[1])
        if int(ori_block_table.shape[0]) != batch:
            raise ValueError("ori_block_table batch mismatch")
    elif layout_kv == "TND":
        if ori_kv.ndim != 3:
            raise ValueError("TND KV requires [T2, 1, 512]")
        kv_bs = 0
        ori_bts = 0
        if cu_seqlens_ori_kv is None and seqused_ori_kv is None:
            raise ValueError("TND KV requires int32 cu_seqlens_ori_kv")
    else:
        if ori_kv.ndim != 4:
            raise ValueError("BSND KV requires [B, S2, 1, 512]")
        kv_bs = int(ori_kv.shape[1])
        ori_bts = 0

    cmp_bs = 0
    cmp_bts = 0
    if has_cmp:
        _check_tensor("cmp_kv", cmp_kv, dtype)
        if paged:
            # 压缩 KV 有自己的页大小（与 ori KV 页大小无关），页表与页尺寸
            # 都按各自的张量取。
            if cmp_kv.ndim != 4:
                raise ValueError("paged compressed KV requires [blocks, page, 1, 512]")
            if seqused_cmp_kv is None:
                raise ValueError("paged compressed KV requires seqused_cmp_kv")
            if cmp_block_table is None or cmp_block_table.dtype != torch.int32:
                raise ValueError("paged compressed KV requires int32 cmp_block_table")
            cmp_bts = int(cmp_block_table.shape[1])
        elif layout_kv == "BSND":
            if cmp_kv.ndim != 4:
                raise ValueError("BSND compressed KV requires [B, S2c, 1, 512]")
            cmp_bs = int(cmp_kv.shape[1])
        else:
            if cmp_kv.ndim != 3:
                raise ValueError("TND compressed KV requires [T2c, 1, 512]")
            if cu_seqlens_cmp_kv is None and seqused_cmp_kv is None:
                raise ValueError("TND compressed KV requires int32 cu_seqlens_cmp_kv")

    for name, value in (
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_residual_kv", cmp_residual_kv),
    ):
        if value is not None:
            _check_tensor(name, value, torch.int32)
    if sinks is not None:
        if sinks.dtype != torch.float32 or not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous float32")
        if int(sinks.numel()) != heads:
            raise ValueError("sinks must have N1 elements")

    ori_w = int(ori_sparse_indices.shape[-1]) if has_ori_idx else 0
    cmp_w = int(cmp_sparse_indices.shape[-1]) if has_cmp_idx else 0
    if has_ori_idx:
        if int(ori_sparse_indices.numel()) != total_rows * ori_w:
            raise ValueError("ori_sparse_indices shape does not match the query rows")
    if has_cmp_idx:
        if int(cmp_sparse_indices.numel()) != total_rows * cmp_w:
            raise ValueError("cmp_sparse_indices shape does not match the query rows")

    # ------------------------------------------------ 选路
    # 5 个分支 kernel 对应 OT 的 5 个 perfMode。公开接口没有显式 mode 参数，
    # 因此按「是否有 compressed KV / 是否有稀疏索引」确定，与 workload 的
    # SWA / HCA / CSA / ORI_SPARSE / ORI_CMP_SPARSE 一一对应。
    if has_cmp_idx and has_ori_idx:
        route = "ocs"
    elif has_cmp_idx:
        route = "csa"
    elif has_ori_idx:
        if has_cmp:
            raise ValueError(
                "ori sparse indices with compressed KV require "
                "cmp_sparse_indices (ORI_CMP_SPARSE route)"
            )
        route = "os"
    elif has_cmp:
        route = "hca"
    else:
        route = "swa"

    if route == "csa" and cmp_topk_length is None:
        raise ValueError("CSA route requires cmp_topk_length")
    if route == "os" and ori_topk_length is None:
        raise ValueError("ORI_SPARSE route requires ori_topk_length")
    if route == "ocs" and (ori_topk_length is None or cmp_topk_length is None):
        raise ValueError("ORI_CMP_SPARSE route requires both *_topk_length")

    device = q.device
    i32 = torch.int32
    f32 = torch.float32
    zero_len = _zero_len(device)

    if total_rows == 0:
        output = _pooled_output(q)
        lse = (
            _pooled_lse(
                device,
                (1, total_rows, heads)
                if layout_q == "TND"
                else (batch, 1, s1_len, heads),
            )
            if want_lse
            else zero_len
        )
        return output, lse

    # ------------------------------------------------ cu_seqlens
    per_q = (batch, s1_len) if layout_q == "BSND" else None
    per_kv = (batch, kv_bs) if layout_kv == "BSND" else None
    cu_q_t, cu_q = _build_cu(cu_seqlens_q, seqused_q, batch, per_q, device)
    cu_kv_t, cu_kv = _build_cu(cu_seqlens_ori_kv, seqused_ori_kv, batch, per_kv, device)
    num_seqs = len(cu_q) - 1

    # ------------------------------------------------ BSND ragged 行重排
    # kernel 的 q / out 行空间是「有效行紧凑排列」：行偏移由 cu_q（seqused_q 的
    # 前缀和）给出。BSND 的张量是 [B, S, ...] 物理连续，seqused_* < S 时两者
    # 不一致——直接下传会让 b>=1 的行读到相邻 batch 的数据，padding 行则根本
    # 不被写（输出池里的旧内容会被当成结果）。这里在宿主侧把有效行 gather 成
    # 紧凑张量，跑完再 scatter 回物理位置、padding 行写 0。非 ragged 形态完全
    # 不走这条路径（q_rows_idx 为 None）。
    def _pack_index(lengths, step):
        parts = [
            torch.arange(b * step, b * step + v, device=device)
            for b, v in enumerate(lengths)
            if v > 0
        ]
        return torch.cat(parts).to(torch.long) if parts else None

    q_rows_idx = kv_rows_idx = cmp_rows_idx = None
    if layout_q == "BSND" and seqused_q is not None:
        q_lens = [int(v) for v in seqused_q.reshape(-1).tolist()]
        if len(q_lens) == batch and any(v != s1_len for v in q_lens):
            q_rows_idx = _pack_index(q_lens, s1_len)
    if not paged and layout_kv == "BSND" and seqused_ori_kv is not None:
        kv_lens = [int(v) for v in seqused_ori_kv.reshape(-1).tolist()]
        if len(kv_lens) == batch and any(v != kv_bs for v in kv_lens):
            kv_rows_idx = _pack_index(kv_lens, kv_bs)
    if not paged and has_cmp and layout_kv == "BSND" and seqused_cmp_kv is not None:
        cmp_lens = [int(v) for v in seqused_cmp_kv.reshape(-1).tolist()]
        if len(cmp_lens) == batch and any(v != cmp_bs for v in cmp_lens):
            cmp_rows_idx = _pack_index(cmp_lens, cmp_bs)
    run_rows = int(q_rows_idx.numel()) if q_rows_idx is not None else total_rows

    if has_cmp:
        cmp_per_batch = (batch, cmp_bs) if layout_kv == "BSND" else None
        cu_cmp_t, cu_cmp = _build_cu(
            cu_seqlens_cmp_kv, seqused_cmp_kv, batch, cmp_per_batch, device
        )
    else:
        cu_cmp_t, cu_cmp = None, [0, 0]

    # ------------------------------------------------ 输出 / workspace
    output = _pooled_output(q)
    if want_lse:
        lse = _pooled_lse(
            device,
            (1, total_rows, heads) if layout_q == "TND" else (batch, 1, s1_len, heads),
        )
        lse_flat = (
            torch.empty(run_rows * heads, dtype=f32, device=device)
            if q_rows_idx is not None
            else lse.reshape(-1)
        )
    else:
        # kernel 把 lse 当真正的输出缓冲：形状参与编译期 UB 划分，核对范围内
        # 也仍会被写。未请求 lse 时按同样的形状分配一块临时缓冲，只在返回时
        # 换成空张量（此前的 1 元素占位会让 UB 划分与正常路径不同，实测
        # ORI_SPARSE 的 3 例因此出现 1%~2% 的元素偏差）。
        lse = zero_len
        lse_flat = torch.empty(run_rows * heads, dtype=f32, device=device)

    q_rows_flat = q.reshape(total_rows, heads * HEAD_DIM)
    if q_rows_idx is not None:
        q_rows_flat = q_rows_flat.index_select(0, q_rows_idx)
    q2 = q_rows_flat.reshape(-1, HEAD_DIM).contiguous()
    kv_rows_flat = ori_kv.reshape(-1, HEAD_DIM)
    if kv_rows_idx is not None:
        kv_rows_flat = kv_rows_flat.index_select(0, kv_rows_idx)
    kv2 = kv_rows_flat.contiguous()
    if q_rows_idx is not None:
        out_pack = torch.empty(
            run_rows, heads * HEAD_DIM, dtype=output.dtype, device=device
        )
        out2 = out_pack.reshape(-1, HEAD_DIM)
    else:
        out_pack = None
        out2 = output.reshape(total_rows * heads, HEAD_DIM)
    if has_cmp:
        cmp_rows_flat = cmp_kv.reshape(-1, HEAD_DIM)
        if cmp_rows_idx is not None:
            cmp_rows_flat = cmp_rows_flat.index_select(0, cmp_rows_idx)
        cmp2 = cmp_rows_flat.contiguous()
    else:
        cmp2 = None

    sinks_use = (
        sinks if sinks is not None else torch.zeros(heads, dtype=f32, device=device)
    )
    has_sinks = 0 if sinks is None else 1

    resid = (
        cmp_residual_kv.reshape(-1).to(i32).contiguous()
        if (cmp_residual_kv is not None and cmp_ratio > 1 and cmp_mask_mode != 0)
        else torch.zeros(num_seqs, dtype=i32, device=device)
    )

    bt_o_t, pa_o_bs, pa_o_stride, pa_o_max, pa_o_pages = _block_table(
        ori_block_table, ori_kv, device
    )
    if has_cmp:
        bt_c_t, pa_c_bs, pa_c_stride, pa_c_max, pa_c_pages = _block_table(
            cmp_block_table, cmp_kv, device
        )
    else:
        bt_c_t = _dummy(device, i32)
        pa_c_bs = pa_c_stride = pa_c_max = pa_c_pages = 0

    g_split = 2 if heads > 64 else 1  # OT: splitG = gSize > 64
    kv_layout = _LAYOUT_CONST[layout_kv]
    is_vec = _vectorize_flag(
        route,
        layout_kv,
        ori_kv,
        cmp_kv if has_cmp else ori_kv,
        ori_block_table,
        cmp_block_table if has_cmp else None,
        ori_w,
        cmp_w,
    )

    block_num = min(_cube_cores(device), WS_BLOCKS)
    tiling = torch.zeros(4, dtype=i32, device=device)
    if route in ("csa", "os", "ocs") and g_split == 2:
        ori_topk = (
            [int(v) for v in ori_topk_length.reshape(-1).tolist()]
            if ori_topk_length is not None
            else None
        )
        cmp_topk = (
            [int(v) for v in cmp_topk_length.reshape(-1).tolist()]
            if cmp_topk_length is not None
            else None
        )
        resid_list = [int(v) for v in resid.reshape(-1).tolist()]
        tiling[1] = _fa_s2_max_num(
            route,
            cu_q,
            cu_kv,
            cu_cmp,
            ori_topk,
            cmp_topk,
            ori_w,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            resid_list,
            block_num,
            g_split,
        )

    scale_t = tla.Float32(scale_value)

    # 未使用的可选输入用 1 元素占位，保持 kernel 张量面固定。
    if route in ("csa", "os", "ocs"):
        ws = torch.zeros(WS_BLOCKS * KV_STAGE_BYTES_EL, dtype=dtype, device=device)
    else:
        ws = _dummy(device, dtype)

    # ------------------------------------------------ 分派
    if route == "swa":
        t_kv = _tla_tensor(kv2)
        m_rows = heads // g_split - (heads // g_split) // 2
        rows8 = (m_rows + 7) // 8 * 8
        bc_stride = m_rows * HEAD_DIM + rows8 * 2
        bc_ws = torch.zeros(block_num * 2 * bc_stride, dtype=f32, device=device)
        tensor_args = (
            _tla_tensor(q2),
            t_kv,
            _tla_tensor(bt_o_t),
            _tla_tensor(sinks_use),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(bc_ws),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                bt_o_t,
                sinks_use,
                cu_q_t,
                cu_kv_t,
                out2,
                lse_flat,
                tiling,
                bc_ws,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_o_pages,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            0,
        )
        kernel_fn = sparse_flash_mla_swa_kernel
    elif route == "hca":
        q_hca = q2
        if g_split == 2 and heads % 2 == 1:
            half_g = (heads + 1) // 2
            m_cube = half_g + (half_g % 2)
            pad_rows = max(0, m_cube - (heads - half_g))
            if pad_rows:
                q_pad = torch.zeros(
                    (q2.shape[0] + pad_rows, HEAD_DIM),
                    dtype=q2.dtype,
                    device=device,
                )
                q_pad[: q2.shape[0]] = q2
                q_hca = q_pad
        sinks_hca = sinks_use
        if g_split == 2 and heads % 2 == 1:
            sink_pad = ((heads + 7) // 8) * 8 - heads
            if sink_pad:
                s_pad = torch.zeros(heads + sink_pad, dtype=f32, device=device)
                s_pad[:heads] = sinks_use
                sinks_hca = s_pad
        tensor_args = (
            _tla_tensor(q_hca),
            _tla_tensor(kv2),
            _tla_tensor(cmp2),
            _tla_tensor(bt_o_t),
            _tla_tensor(bt_c_t),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(cu_cmp_t),
            _tla_tensor(resid),
            _tla_tensor(sinks_hca),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q_hca,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                resid,
                sinks_hca,
                out2,
                lse_flat,
                tiling,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            cmp_ratio,
            cmp_mask_mode,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_o_pages,
            pa_c_bs,
            pa_c_stride,
            pa_c_max,
            pa_c_pages,
            g_split,
            has_sinks,
            1 if want_lse else 0,
        )
        kernel_fn = sparse_flash_mla_hca_kernel
    elif route == "csa":
        idx = cmp_sparse_indices.reshape(total_rows, cmp_w)
        topk = cmp_topk_length.reshape(-1)
        if q_rows_idx is not None:
            idx = idx.index_select(0, q_rows_idx)
            topk = topk.index_select(0, q_rows_idx)
        idx = idx.reshape(-1).to(i32).contiguous()
        topk = topk.to(i32).contiguous()
        phy_w = _align128(cmp_w) if is_vec == 1 else 1
        cmp_phy = torch.zeros(run_rows * phy_w, dtype=torch.int64, device=device)
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(cmp2),
            _tla_tensor(bt_o_t),
            _tla_tensor(bt_c_t),
            _tla_tensor(idx),
            _tla_tensor(topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(cu_cmp_t),
            _tla_tensor(resid),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(cmp_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                idx,
                topk,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                resid,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                cmp_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_o_pages,
            pa_c_bs,
            pa_c_stride,
            pa_c_max,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            1,
            kv_layout,
            is_vec,
            phy_w,
        )
        kernel_fn = sparse_flash_mla_csa_kernel
    elif route == "os":
        idx = ori_sparse_indices.reshape(total_rows, ori_w)
        topk = ori_topk_length.reshape(-1)
        if q_rows_idx is not None:
            idx = idx.index_select(0, q_rows_idx)
            topk = topk.index_select(0, q_rows_idx)
        idx = idx.reshape(-1).to(i32).contiguous()
        topk = topk.to(i32).contiguous()
        phy_w = _align128(ori_w) if is_vec == 1 else 1
        ori_phy = torch.zeros(run_rows * phy_w, dtype=torch.int64, device=device)
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(bt_o_t),
            _tla_tensor(idx),
            _tla_tensor(topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(ori_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                bt_o_t,
                idx,
                topk,
                cu_q_t,
                cu_kv_t,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                ori_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            1 if want_lse else 0,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            g_split,
            has_sinks,
            1,
            kv_layout,
            is_vec,
            phy_w,
        )
        kernel_fn = sparse_flash_mla_ori_sparse_kernel
    else:  # ocs
        ori_idx = ori_sparse_indices.reshape(total_rows, ori_w)
        ori_topk = ori_topk_length.reshape(-1)
        cmp_idx = cmp_sparse_indices.reshape(total_rows, cmp_w)
        cmp_topk = cmp_topk_length.reshape(-1)
        if q_rows_idx is not None:
            ori_idx = ori_idx.index_select(0, q_rows_idx)
            ori_topk = ori_topk.index_select(0, q_rows_idx)
            cmp_idx = cmp_idx.index_select(0, q_rows_idx)
            cmp_topk = cmp_topk.index_select(0, q_rows_idx)
        ori_idx = ori_idx.reshape(-1).to(i32).contiguous()
        ori_topk = ori_topk.to(i32).contiguous()
        cmp_idx = cmp_idx.reshape(-1).to(i32).contiguous()
        cmp_topk = cmp_topk.to(i32).contiguous()
        ori_phy_w = _align128(ori_w) if is_vec == 1 else 1
        cmp_phy_w = _align128(cmp_w) if is_vec == 1 else 1
        ori_phy = torch.zeros(run_rows * ori_phy_w, dtype=torch.int64, device=device)
        cmp_phy = torch.zeros(run_rows * cmp_phy_w, dtype=torch.int64, device=device)
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(cmp2),
            _tla_tensor(bt_o_t),
            _tla_tensor(bt_c_t),
            _tla_tensor(ori_idx),
            _tla_tensor(ori_topk),
            _tla_tensor(cmp_idx),
            _tla_tensor(cmp_topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(cu_cmp_t),
            _tla_tensor(resid),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(ori_phy),
            _tla_tensor(cmp_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                ori_idx,
                ori_topk,
                cmp_idx,
                cmp_topk,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                resid,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                ori_phy,
                cmp_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_w,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_c_bs,
            pa_c_stride,
            pa_c_max,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            1,
            kv_layout,
            is_vec,
            ori_phy_w,
            cmp_phy_w,
        )
        kernel_fn = sparse_flash_mla_ori_cmp_sparse_kernel

    # ------------------------------------------------ 编译 / 复用
    key = (
        route,
        str(device),
        str(dtype),
        tuple(tensor_shapes),
        tuple(int(v) for v in const_args),
    )
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        compile_args = tensor_args + (scale_t,) + const_args
        try:
            executor = tla.compile(
                kernel_fn,
                *compile_args,
                options=NPU_ARCH,
            )
        except Exception as exc:  # noqa: BLE001
            if "253952" not in str(exc):
                raise
            executor = tla.compile(
                kernel_fn,
                *compile_args,
                options=NPU_ARCH + UB_RELEASE_FLAGS,
            )
        if len(_EXECUTOR_CACHE) >= _EXECUTOR_CACHE_LIMIT:
            _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE[key] = executor

    if compile_only:
        return output, lse
    executor(*tensor_args, scale_t, block_num=block_num)
    if out_pack is not None:
        # ragged：紧凑结果写回物理行，padding 行（该 batch 的 seqused_q 之后）
        # 不在 kernel 的任务空间里，必须显式清零。
        output.zero_()
        output.reshape(total_rows, heads * HEAD_DIM).index_copy_(
            0, q_rows_idx, out_pack
        )
        if want_lse:
            lse.zero_()
            lse.reshape(total_rows, heads).index_copy_(
                0, q_rows_idx, lse_flat.reshape(run_rows, heads)
            )
    return output, lse
