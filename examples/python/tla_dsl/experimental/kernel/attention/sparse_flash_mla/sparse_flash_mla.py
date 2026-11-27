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

import json
import os

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

# F7：整调用 launch 计划缓存。键 = 「输入签名」（每个输入张量的存储地址 +
# shape + dtype + 版本计数）＋ 全部标量参数 ＋ 输出 / lse 池槽身份。命中时
# 跳过 host 侧的 cu 重建、规划、表物化、from_dlpack 绑定与 tla 运行时分发，
# 只做 workspace 零初始化、流解析与一次 aclrtLaunchKernelWithHostArgs。
# 语义前提：同一签名的输入逐位不变；`_version` 覆盖 torch 的原地写。
# 这是纯 host 侧缓存，不修改 DSL 包；计划条目持有张量/执行器引用，防止地址复用。
_FAST_PLANS = {}
_FAST_PLAN_LIMIT = 32
_FAST_ON = os.environ.get("SMLA_FAST_PLAN", "1") != "0"
_ZERO_TRACE = None


def _sig_tensor(x):
    if x is None:
        return None
    try:
        return (int(x.data_ptr()), tuple(x.shape), x.dtype, int(x._version))
    except Exception:
        return (id(x),)


def _fast_key(route, device, dtype, scal, tensors, out, lse):
    return (
        route,
        str(device),
        str(dtype),
        scal,
        tuple(_sig_tensor(t) for t in tensors),
        _sig_tensor(out),
        _sig_tensor(lse),
    )


def _register_fast_plan(key, executor, tensor_args, extra, block_num):
    """Capture a launch plan for `key` from a just-completed regular launch."""
    global _ZERO_TRACE
    zero = list(dict.fromkeys(_ZERO_TRACE or ()))
    _ZERO_TRACE = None
    try:
        import functools
        import catlass.execution as _execution
        from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes
        from catlass.base_dsl.runtime.ascend_stream_adapter import current_stream

        module = executor.jit_module
        jit_exec = executor._get_executor()
        if (
            module.uses_scalar_print
            or module.uses_tensor_print
            or module.print_argument_layout
            or module.simt_print_formats is not None
        ):
            return
        payload = module.execution_args.generate_launch_payload(
            tuple(tensor_args) + tuple(extra)
        )
        kwargs = dict(
            function_handle=int(jit_exec.function_handle),
            block_num=int(block_num),
            payload=payload,
            dyn_ubuf_bytes=_resolved_dyn_ubuf_bytes(module, 0),
            uses_scalar_print=False,
            uses_tensor_print=False,
            is_mixed=bool(module.is_mixed),
            print_tensor_position=module.print_tensor_position,
            device_id=int(jit_exec.device),
        )
        if len(_FAST_PLANS) >= _FAST_PLAN_LIMIT:
            _FAST_PLANS.clear()
        _FAST_PLANS[key] = {
            "zero": zero,
            "launch": functools.partial(_execution.launch_kernel, **kwargs),
            "stream": current_stream,
            "device": int(jit_exec.device),
            # 持有引用：绑定对象（含源存储）、执行器与整型表，避免地址被回收复用。
            "keep": (tuple(tensor_args), tuple(extra), executor, jit_exec),
        }
    except Exception:
        if os.environ.get("SMLA_FAST_DEBUG") == "1":
            import traceback

            traceback.print_exc()


_BINDING_CACHE_LIMIT = 48

_DUMMY_CACHE = {}
_DEV_PROP = {}

# E33：host 侧 FD 规划缓存。规划函数都是纯函数（task_tiles + 核数 + 枚举参数），
# 输入不变时逐位复用，跳过 Python 枚举（实测 #022 每次 9.7 ms、#028 1.1 ms）。
_FD_PLAN_CACHE = {}
_FD_PLAN_CACHE_LIMIT = 32
_FD_TILE_CACHE = {}
_FD_TILE_CACHE_LIMIT = 64
_FD_TAB_CACHE = {}
_FD_TAB_CACHE_LIMIT = 32


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


# F19：常零 GM 源（KV tail 清零用；形状 [KV_BLOCK, HEAD_DIM]，只读、不随调用改写，
# 因此走独立缓存而不进 `_workspace` 池——池里的 zero=True 缓冲每次调用都要重清零）。
_KV_ZERO_POOL = {}


def _kv_zero(device, dtype):
    key = (str(device), str(dtype))
    t = _KV_ZERO_POOL.get(key)
    if t is None:
        t = torch.zeros(KV_BLOCK * HEAD_DIM, dtype=dtype, device=device)
        _KV_ZERO_POOL[key] = t
    return t


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
    # E33：缓存键从 `id(obj)` 改为「存储地址 + shape + dtype + layout」。
    # 每次调用都会重建 reshape/contiguous 视图（q2/kv2/out2 等），对象身份
    # 每次都变但底层存储相同；按存储身份命中可把 from_dlpack（实测 ~20 µs/个）
    # 从「每次全量重绑」降到「每存储一次」。条目持有源张量，保证地址不被复用。
    tag = layout_tag or tla.arch.RowMajor
    if value.is_contiguous():
        key = (value.data_ptr(), tuple(value.shape), value.dtype, tag)
        hit = _BINDING_CACHE.get(key)
        if hit is not None:
            return hit[1]
    else:
        key = None
    contiguous = value.contiguous()
    bound = from_dlpack(contiguous, layout_tag=tag)
    # 让绑定对象持有源存储，避免 DLPack capsule 生命周期问题。
    bound._torch_storage = contiguous
    if key is not None:
        if len(_BINDING_CACHE) >= _BINDING_CACHE_LIMIT:
            _BINDING_CACHE.clear()
        _BINDING_CACHE[key] = (contiguous, bound)
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


def _d2h_lists(tensors):
    """把若干 int32 设备张量一次拷回 host，切成与输入等长的 Python int 列表。

    E33：每次 `.tolist()` 都是一次独立 D2H + 同步（实测 ~20 µs）；一次 run 里
    shape / rag 检查 / tiling / 规划最多要读 9 组小张量，合并成一次 cat +
    一次同步。
    """
    flats = [t.reshape(-1) for t in tensors]
    if len(flats) == 1:
        vals = flats[0].cpu().tolist()
    else:
        vals = torch.cat(flats).cpu().tolist()
    out = []
    off = 0
    for f in flats:
        n = int(f.numel())
        out.append(vals[off : off + n])
        off += n
    return out


_I32_TENSOR_CACHE = {}
_I32_TENSOR_CACHE_LIMIT = 64
_PARAM_TAB_CACHE = {}
_PARAM_TAB_CACHE_LIMIT = 32
_WORK_POOL = {}


def _i32_cached(values, device):
    """按内容缓存只读 int32 设备表（tiling / tp_tab / fd_tab）。

    kernel 对这些表只读；steady-state 下内容逐次相同，复用同一张量既省
    H2D（1024 元素 ~60 µs），也让 _tla_tensor 走存储级绑定缓存。
    """
    key = (str(device), tuple(values))
    hit = _I32_TENSOR_CACHE.get(key)
    if hit is not None:
        return hit
    t = torch.tensor(values, dtype=torch.int32, device=device)
    if len(_I32_TENSOR_CACHE) >= _I32_TENSOR_CACHE_LIMIT:
        _I32_TENSOR_CACHE.clear()
    _I32_TENSOR_CACHE[key] = t
    return t


def _drop_bindings(data_ptr):
    """剔除绑定缓存中指向该 data_ptr 的条目（缓冲扩容前调用）。"""
    for k in [k for k in _BINDING_CACHE if k[0] == data_ptr]:
        _BINDING_CACHE.pop(k, None)


def _workspace(role, device, dtype, numel, zero=False):
    """按「用途 + 设备 + dtype」复用设备缓冲，按需扩容。

    E33：此前每次 run 都 torch.zeros/torch.empty 新分配（#043 的 int64 物理
    地址表、FD 部分量槽、稀疏暂存槽等共 ~10 个）。复用后地址稳定、绑定命中，
    也避免「有在飞 kernel 时新分配几十 MB 需 0.25~0.35 ms」的同步查询。
    original 为零初始化的用途仍保持 zero_()（同一 stream，次序不变）。

    每个 role 只保留一块容量 >= 当前需求的缓冲：命中时返回 buf[:numel]，
    data_ptr 不变，_tla_tensor 的存储级绑定缓存持续命中；容量不足时先释放
    旧缓冲并按旧 data_ptr 剔除绑定条目，避免旧缓冲被绑定缓存继续持有。
    """
    key = (role, str(device), str(dtype))
    numel = int(numel)
    buf = _WORK_POOL.get(key)
    if buf is None or buf.numel() < numel:
        old = _WORK_POOL.pop(key, None)
        if old is not None:
            old_ptr = old.data_ptr()
            del old
            _drop_bindings(old_ptr)
        buf = (
            torch.zeros(numel, dtype=dtype, device=device)
            if zero
            else torch.empty(numel, dtype=dtype, device=device)
        )
        _WORK_POOL[key] = buf
        out = buf
    else:
        out = buf[:numel]
        if zero:
            out.zero_()
    if zero and _ZERO_TRACE is not None:
        _ZERO_TRACE.append(out)
    return out


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


def _build_cu(cu, seqused, batch, per_batch, device, vals=None):
    """返回 (int32[B+1] 设备张量, Python list)。

    cu_seqlens 优先；否则由 seqused 求前缀和；两者都缺时按每 batch 定长重建。
    vals：调用方已批量 D2H 取回的 host 值（E33），避免重复同步。
    """
    i32 = torch.int32
    if cu is not None:
        _check_tensor("cu_seqlens", cu, i32, 1)
        t = cu.to(i32).contiguous()
        return t, (
            list(vals) if vals is not None else [int(v) for v in t.reshape(-1).tolist()]
        )
    if seqused is not None:
        _check_tensor("seqused", seqused, i32, 1)
        vals = (
            list(vals)
            if vals is not None
            else [int(v) for v in seqused.reshape(-1).tolist()]
        )
        acc = [0]
        for v in vals:
            acc.append(acc[-1] + v)
        return _i32_cached(acc, device), acc
    if per_batch is not None:
        n, step = int(per_batch[0]), int(per_batch[1])
        acc = [i * step for i in range(n + 1)]
        return _i32_cached(acc, device), acc
    if batch is None:
        raise ValueError("cannot derive cu_seqlens: pass cu_seqlens_* or seqused_*")
    acc = [0] * (int(batch) + 1)
    return _i32_cached(acc, device), acc


def _block_table(bt, kv, device):
    """返回 (展平 int32 设备张量, block_size, pa_stride, max_blocks, pages)。"""
    i32 = torch.int32
    if bt is None:
        # 常驻单元素 int32：非分页路由每次 run 都会取这张 dummy 表，
        # 若每次新建会被绑定缓存按 data_ptr 当作新键持有。
        return _i32_cached([0], device), 0, 0, 0, 1
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


# ---------------------------------------------------------------------------
# F3（长序列）：SWA/HCA 的「本核 tile 前缀 / 前后真实任务」宿主机标量表。
#
# 背景（实测）：kernel 旧实现每个 task 都从 task0 步进 q_bn 游走到 task 自身，
# 逐任务重算 ct 的可见区间与 tile 数，复杂度 O((N/q_bn)^2)；N=32766 时单核
# 约 1.37M 次循环，AIC/AIV 各跑一遍，ext100 长序列 8~20x 劣化的墙
# 是 aic_scalar/aiv_scalar（0456 60.5ms、0313 24.8ms、0020 5.2ms），
# 而非 MAC（仅占 3.6%~7.7%）。这里在 host 侧用与 kernel 同式的整数运算
# 一次性算出每个（core, local）任务所需的标量，kernel 只做 13 次查表。
#
# 表项（stride PTAB_STRIDE，索引 = local*q_bn + core）：
#   0 g_k        本核在 task 之前的真实 tile 总数（gtile 前缀）
#   1 p_task     本核中 task 之前最近的非空任务 id（无则 core 自身）
#   2 p_nt       该任务的 tile 数
#   3 p_vis      该任务的可见列数
#   4 p_bbn      该任务的 baseBlockNum（SWA BC）
#   5 p_goff     该任务的 goIdx（g_split 分半偏移）
#   6..10        次近非空任务的同五项（pp_*）
#   11 nx_task   本核中 task 之后最近的非空任务 id（无则 core 自身）
#   12 any_after 之后是否还有非空任务（= nx_done）
# HCA 只用 g_k，其余字段填写但不会被读。
# ---------------------------------------------------------------------------
PTAB_STRIDE = 13
# 触发阈值：N<4096 时旧前缀游走总迭代 <0.6M（≈0.3ms 以内），保持原指令流
# （官方 50 的 SWA/HCA 全部 N<=1024，因此整条旧路径逐字节不变）。
PTAB_MIN_TASKS = 4096
_PTAB_CACHE = {}
_PTAB_CACHE_LIMIT = 32


def _ptab_build(
    route,
    cu_q,
    cu_kv,
    cu_cmp,
    resid_host,
    ori_mask_mode,
    ori_win_left,
    ori_win_right,
    cmp_mask_mode,
    cmp_ratio,
    g_split,
    q_bn,
    head_num,
):
    """返回 CPU int32 flat 表（长度 Lmax*q_bn*PTAB_STRIDE）。纯 torch 向量化。"""
    b = len(cu_q) - 1
    q_lens = [int(cu_q[i + 1] - cu_q[i]) for i in range(b)]
    kv_lens = [int(cu_kv[i + 1] - cu_kv[i]) for i in range(b)]
    cmp_lens = (
        [int(cu_cmp[i + 1] - cu_cmp[i]) for i in range(b)]
        if cu_cmp is not None and len(cu_cmp) == b + 1
        else [0] * b
    )
    total = int(cu_q[-1])
    if total <= 0:
        return None
    t = torch.arange(total, dtype=torch.long)
    ql_b = torch.tensor(q_lens, dtype=torch.long)
    seq = torch.repeat_interleave(torch.arange(b, dtype=torch.long), ql_b)
    cu_q_t = torch.tensor([int(v) for v in cu_q], dtype=torch.long)
    rowq = t - cu_q_t[seq]
    ql = ql_b[seq]
    kvl = torch.tensor(kv_lens, dtype=torch.long)[seq]
    cl = torch.tensor(cmp_lens, dtype=torch.long)[seq]
    if resid_host is not None and len(resid_host) == b:
        rsv = torch.tensor([int(v) for v in resid_host], dtype=torch.long)[seq]
    else:
        rsv = torch.zeros(total, dtype=torch.long)
    coff = kvl - ql
    if ori_mask_mode == 3:
        pre, nxt = ql, coff
    elif ori_mask_mode == 4:
        pre = (
            torch.full_like(ql, int(ori_win_left)) - coff
            if int(ori_win_left) != -1
            else ql.clone()
        )
        nxt = coff + int(ori_win_right) if int(ori_win_right) != -1 else kvl.clone()
        pre = torch.minimum(pre, ql)
    else:
        pre, nxt = ql, kvl
    start = torch.clamp(rowq - pre, min=0)
    start = torch.minimum(start, kvl)
    end = torch.clamp(rowq + nxt + 1, min=0)
    end = torch.minimum(end, kvl)
    vis = torch.clamp(end - start, min=0)
    n_ori = (vis + 127) // 128
    if route == "hca":
        next_cmp = cl * int(cmp_ratio)
        if int(cmp_mask_mode) != 0:
            next_cmp = cl * int(cmp_ratio) + rsv - ql
        num = rowq + 1 + next_cmp
        cend = torch.div(num, int(cmp_ratio), rounding_mode="trunc")
        cend = torch.clamp(cend, min=0)
        cend = torch.minimum(cend, cl)
        n_cmp = (cend + 127) // 128
        nt_tok = n_ori + n_cmp
    else:
        nt_tok = n_ori
    # SWA BC 的 baseBlockNum（kernel 同式）
    raw_rb = (vis + 31) // 32
    rb = ((raw_rb + 127) // 128) * 128
    bbn_tok = rb // 128
    bbn_tok = torch.where(rb == 0, torch.ones_like(bbn_tok), bbn_tok)
    m = head_num // g_split
    if g_split > 1:
        nt_t = nt_tok.repeat_interleave(g_split)
        vis_t = vis.repeat_interleave(g_split)
        bbn_t = bbn_tok.repeat_interleave(g_split)
    else:
        nt_t, vis_t, bbn_t = nt_tok, vis, bbn_tok
    n_tasks = total * g_split
    half = torch.arange(n_tasks, dtype=torch.long) % g_split
    goff_t = half * m
    lmax = (n_tasks + q_bn - 1) // q_bn
    pad_n = lmax * q_bn - n_tasks

    def _pad2d(x):
        if pad_n:
            x = torch.cat([x, torch.zeros(pad_n, dtype=x.dtype)])
        return x.reshape(lmax, q_bn)

    nt2 = _pad2d(nt_t)
    vis2 = _pad2d(vis_t)
    bbn2 = _pad2d(bbn_t)
    goff2 = _pad2d(goff_t)
    li = torch.arange(lmax, dtype=torch.long).reshape(lmax, 1)
    li = li.expand(lmax, q_bn)
    nonempty = nt2 > 0
    gk = torch.cumsum(nt2, dim=0) - nt2
    last_inc = torch.where(nonempty, li, torch.full_like(li, -1)).cummax(dim=0).values
    zero_row = torch.full((1, q_bn), -1, dtype=torch.long)
    p_loc = torch.cat([zero_row, last_inc[:-1]], dim=0) if lmax > 1 else zero_row
    p_idx = p_loc.clamp(min=0)
    p_task = p_loc * q_bn + torch.arange(q_bn, dtype=torch.long).reshape(1, q_bn)
    p_nt = torch.gather(nt2, 0, p_idx)
    p_vis = torch.gather(vis2, 0, p_idx)
    p_bbn = torch.gather(bbn2, 0, p_idx)
    p_goff = torch.gather(goff2, 0, p_idx)
    p_task = torch.where(p_loc >= 0, p_task, torch.arange(q_bn).reshape(1, q_bn))
    p_nt = torch.where(p_loc >= 0, p_nt, torch.ones_like(p_nt))
    p_vis = torch.where(p_loc >= 0, p_vis, torch.full_like(p_vis, 128))
    p_bbn = torch.where(p_loc >= 0, p_bbn, torch.ones_like(p_bbn))
    p_goff = torch.where(p_loc >= 0, p_goff, torch.zeros_like(p_goff))
    pp_loc = torch.gather(last_inc, 0, (p_loc - 1).clamp(min=0))
    pp_loc = torch.where(p_loc > 0, pp_loc, torch.full_like(pp_loc, -1))
    pp_idx = pp_loc.clamp(min=0)
    pp_task = pp_loc * q_bn + torch.arange(q_bn, dtype=torch.long).reshape(1, q_bn)
    pp_nt = torch.gather(nt2, 0, pp_idx)
    pp_vis = torch.gather(vis2, 0, pp_idx)
    pp_bbn = torch.gather(bbn2, 0, pp_idx)
    pp_goff = torch.gather(goff2, 0, pp_idx)
    pp_task = torch.where(pp_loc >= 0, pp_task, torch.arange(q_bn).reshape(1, q_bn))
    pp_nt = torch.where(pp_loc >= 0, pp_nt, torch.ones_like(pp_nt))
    pp_vis = torch.where(pp_loc >= 0, pp_vis, torch.full_like(pp_vis, 128))
    pp_bbn = torch.where(pp_loc >= 0, pp_bbn, torch.ones_like(pp_bbn))
    pp_goff = torch.where(pp_loc >= 0, pp_goff, torch.zeros_like(pp_goff))
    nxt_inc = (
        torch.where(nonempty, li, torch.full_like(li, lmax))
        .flip(0)
        .cummin(dim=0)
        .values.flip(0)
    )
    big_row = torch.full((1, q_bn), lmax, dtype=torch.long)
    nx_loc = torch.cat([nxt_inc[1:], big_row], dim=0) if lmax > 1 else big_row
    any_after = (nx_loc < lmax).to(torch.long)
    nx_task = nx_loc * q_bn + torch.arange(q_bn, dtype=torch.long).reshape(1, q_bn)
    nx_task = torch.where(nx_loc < lmax, nx_task, torch.arange(q_bn).reshape(1, q_bn))
    tab = torch.stack(
        [
            gk,
            p_task,
            p_nt,
            p_vis,
            p_bbn,
            p_goff,
            pp_task,
            pp_nt,
            pp_vis,
            pp_bbn,
            pp_goff,
            nx_task,
            any_after,
        ],
        dim=-1,
    )
    return tab.reshape(-1).to(torch.int32)


def _swa_hca_task_table(
    route,
    cu_q,
    cu_kv,
    cu_cmp,
    resid_host,
    ori_mask_mode,
    ori_win_left,
    ori_win_right,
    cmp_mask_mode,
    cmp_ratio,
    g_split,
    q_bn,
    head_num,
    device,
):
    """按内容缓存设备侧 ptab；N<PTAB_MIN_TASKS 返回 None（走旧前缀游走）。"""
    n_tasks = int(cu_q[-1]) * g_split
    if n_tasks < PTAB_MIN_TASKS:
        return None
    key = (
        route,
        str(device),
        int(q_bn),
        int(g_split),
        int(head_num),
        int(ori_mask_mode),
        int(ori_win_left),
        int(ori_win_right),
        int(cmp_mask_mode),
        int(cmp_ratio),
        tuple(int(v) for v in cu_q),
        tuple(int(v) for v in cu_kv),
        tuple(int(v) for v in cu_cmp),
        tuple(int(v) for v in (resid_host or ())),
    )
    hit = _PTAB_CACHE.get(key)
    if hit is not None:
        return hit
    cpu = _ptab_build(
        route,
        cu_q,
        cu_kv,
        cu_cmp,
        resid_host,
        ori_mask_mode,
        ori_win_left,
        ori_win_right,
        cmp_mask_mode,
        cmp_ratio,
        g_split,
        q_bn,
        head_num,
    )
    if cpu is None:
        return None
    t = cpu.to(device=device, non_blocking=False)
    if len(_PTAB_CACHE) >= _PTAB_CACHE_LIMIT:
        _PTAB_CACHE.clear()
    _PTAB_CACHE[key] = t
    return t


def _tdiv_i32(a, b):
    """C/i32 语义的整数除法（trunc toward zero），与 kernel 的 divsi 一致。"""
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _os_token_plan(
    cu_q,
    cu_kv,
    cu_cmp,
    ori_topk,
    cmp_topk,
    ori_w,
    cmp_w,
    ori_mask_mode,
    ori_win_left,
    ori_win_right,
    cmp_mask_mode,
    cmp_ratio,
    resid,
    num_seqs,
    is_ocs,
):
    """D2/E13：逐 token 复算 OS/OCS kernel 的任务量（与 kernel 同式）。

    返回每个 token 一条：
      (seq, n_tiles, ori_lines, n_ori_tiles, cmp_lines, n_cmp_tiles,
       kv_row0, cmp_row0)
    OS（is_ocs=False）的 cmp 段为 0。供 needInit 判据与 task 参数表共用。
    """
    plan = []
    for tok in range(int(cu_q[-1])):
        seq = 0
        rowq = tok
        for bb in range(1, num_seqs):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        # ---- ori 可见区间：与 kernel ComputeParamBatch 三分支同式 ----
        causal = kv_len - q_len
        next_tok = kv_len
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal
            pre_tok = q_len
        elif ori_mask_mode == 4:
            pre_tok = q_len if ori_win_left == -1 else ori_win_left - causal
            next_tok = kv_len if ori_win_right == -1 else causal + ori_win_right
            if pre_tok > q_len:
                pre_tok = q_len
        end = rowq + next_tok + 1
        if end < 0:
            end = 0
        if end > kv_len:
            end = kv_len
        start = rowq - pre_tok
        if start < 0:
            start = 0
        if start > kv_len:
            start = kv_len
        rng = end - start
        if rng < 0:
            rng = 0
        oc = ori_topk[tok]
        if oc > ori_w:
            oc = ori_w
        if oc < 0:
            oc = 0
        olines = rng
        if olines > oc:
            olines = oc
        if olines > kv_len:
            olines = kv_len
        if olines < 0:
            olines = 0
        n_ori = (olines + 127) // 128
        n_cmp = 0
        clines = 0
        if is_ocs:
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]
            cc = cmp_topk[tok]
            if cc > cmp_w:
                cc = cmp_w
            if cc < 0:
                cc = 0
            next_cmp = cmp_len * cmp_ratio
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cend = _tdiv_i32(rowq + 1 + next_cmp, cmp_ratio)
            if cend < 0:
                cend = 0
            if cend > cmp_len:
                cend = cmp_len
            clines = cend if cend <= cc else cc
            if clines < 0:
                clines = 0
            n_cmp = (clines + 127) // 128
        plan.append(
            (
                seq,
                n_ori + n_cmp,
                olines,
                n_ori,
                clines,
                n_cmp,
                cu_kv[seq],
                cu_cmp[seq] if is_ocs else 0,
            )
        )
    return plan


def _os_task_need_init(
    cu_q,
    cu_kv,
    cu_cmp,
    ori_topk,
    cmp_topk,
    ori_w,
    cmp_w,
    ori_mask_mode,
    ori_win_left,
    ori_win_right,
    cmp_mask_mode,
    cmp_ratio,
    resid,
    num_seqs,
    is_ocs,
):
    """D2/E13：needInit = 0 当且仅当每个 token 的 tile 数 >= 1。"""
    plan = _os_token_plan(
        cu_q,
        cu_kv,
        cu_cmp,
        ori_topk,
        cmp_topk,
        ori_w,
        cmp_w,
        ori_mask_mode,
        ori_win_left,
        ori_win_right,
        cmp_mask_mode,
        cmp_ratio,
        resid,
        num_seqs,
        is_ocs,
    )
    return 0 if all(p[1] > 0 for p in plan) else 1


def _os_task_param_table(plan, g_split, tasks):
    """D2/E13：把逐 task 的标量打包成 int32 表（TP_STRIDE = 8）。

    字段：
      0 = seq | n_tiles<<16
      1 = ori_lines | n_ori_tiles<<16
      2 = cmp_lines | n_cmp_tiles<<16   （OS: 0）
      3 = kv_row0
      4 = cmp_row0
      5/6/7 = 0
    打包界（否则 ok=0，kernel 走原路径）：各 u16 字段 < 65536。
    """
    _key = (tuple(plan), int(g_split), int(tasks))
    _hit = _PARAM_TAB_CACHE.get(_key)
    if _hit is not None:
        return _hit
    tab = [0] * (tasks * 8)
    ok = True
    for t in range(tasks):
        seq, nt, ol, not_, cl, nct, kv0, cmp0 = plan[t // g_split]
        for v in (seq, nt, ol, not_, cl, nct):
            if not (0 <= v < 65536):
                ok = False
        if not (0 <= kv0 < 2**31 and 0 <= cmp0 < 2**31):
            ok = False
        b = t * 8
        tab[b] = int(seq) | (int(nt) << 16)
        tab[b + 1] = int(ol) | (int(not_) << 16)
        tab[b + 2] = int(cl) | (int(nct) << 16)
        tab[b + 3] = int(kv0)
        tab[b + 4] = int(cmp0)
    if len(_PARAM_TAB_CACHE) >= _PARAM_TAB_CACHE_LIMIT:
        _PARAM_TAB_CACHE.clear()
    _PARAM_TAB_CACHE[_key] = (tab, ok)
    return tab, ok


def _csa_task_param_table(
    cu_q,
    cu_kv,
    cu_cmp,
    cmp_topk,
    cmp_w,
    cmp_mask_mode,
    cmp_ratio,
    resid,
    tok2,
    g_split,
    tasks,
    ori_mask_mode,
    ori_win_left,
    ori_win_right,
):
    """E13：逐 task 复算 kernel 的映射量并打包（TP_STRIDE = 8）。

    字段（int32）：
      0 = vis | vis_b<<16
      1 = cmp_lines | cmp_lines_b<<16
      2 = n_ori_tiles | seq<<16
      3 = start_idx | kv_len<<16
      4 = kv_row0
      5 = cmp_row0
      6/7 = 0
    打包要求（否则返回 ok=0，kernel 走原路径）：vis/vis_b/cmp_lines/
    cmp_lines_b/n_ori_tiles/start_idx < 65536，seq/kv_len < 32768。
    """
    _key = (
        "csa",
        tuple(cu_q),
        tuple(cu_kv),
        tuple(cu_cmp),
        tuple(cmp_topk),
        int(cmp_w),
        int(cmp_mask_mode),
        int(cmp_ratio),
        tuple(resid),
        int(tok2),
        int(g_split),
        int(tasks),
        int(ori_mask_mode),
        int(ori_win_left),
        int(ori_win_right),
    )
    _hit = _PARAM_TAB_CACHE.get(_key)
    if _hit is not None:
        return _hit
    nseq = len(cu_q) - 1
    tab = [0] * (tasks * 8)
    vals = []
    for t in range(tasks):
        tok = t // g_split
        if tok2:
            tok *= 2
        rowq = tok
        seq = 0
        for bb in range(1, nseq):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]
        causal = kv_len - q_len
        next_tok = kv_len
        pre_tok = q_len
        # ori mask 模式：与 kernel 相同的三分支（mode 0/3/4）
        if ori_mask_mode == 3:
            next_tok = causal
            pre_tok = q_len
        elif ori_mask_mode == 4:
            pre_tok = q_len if ori_win_left == -1 else ori_win_left - causal
            next_tok = kv_len if ori_win_right == -1 else causal + ori_win_right
            if pre_tok > q_len:
                pre_tok = q_len
        end = rowq + next_tok + 1
        if end < 0:
            end = 0
        if end > kv_len:
            end = kv_len
        start = rowq - pre_tok
        if start < 0:
            start = 0
        if start > kv_len:
            start = kv_len
        vis = end - start
        if vis < 0:
            vis = 0
        n_ori = (vis + 127) // 128
        vis_b = 0
        if tok2:
            end_b = rowq + 1 + next_tok + 1
            if end_b < 0:
                end_b = 0
            if end_b > kv_len:
                end_b = kv_len
            vis_b = end_b - start
            if vis_b < 0:
                vis_b = 0
            if vis_b > vis:
                n_ori = (vis_b + 127) // 128
        sc = cmp_topk[tok]
        if sc > cmp_w:
            sc = cmp_w
        if sc < 0:
            sc = 0
        next_cmp = cmp_len * cmp_ratio
        if cmp_mask_mode != 0:
            next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
        cmp_end = _tdiv_i32(rowq + 1 + next_cmp, cmp_ratio)
        if cmp_end < 0:
            cmp_end = 0
        if cmp_end > cmp_len:
            cmp_end = cmp_len
        cmp_lines = cmp_end if cmp_end <= sc else sc
        cmp_lines_b = 0
        if tok2:
            sc_b = cmp_topk[tok + 1]
            if sc_b > cmp_w:
                sc_b = cmp_w
            if sc_b < 0:
                sc_b = 0
            cmp_end_b = _tdiv_i32(rowq + 2 + next_cmp, cmp_ratio)
            if cmp_end_b < 0:
                cmp_end_b = 0
            if cmp_end_b > cmp_len:
                cmp_end_b = cmp_len
            cmp_lines_b = cmp_end_b if cmp_end_b <= sc_b else sc_b
        kv_row0 = cu_kv[seq] + start
        cmp_row0 = cu_cmp[seq]
        vals.append(
            (
                vis,
                vis_b,
                cmp_lines,
                cmp_lines_b,
                n_ori,
                start,
                kv_len,
                seq,
                kv_row0,
                cmp_row0,
            )
        )
        base = t * 8
        tab[base] = vis | (vis_b << 16)
        tab[base + 1] = cmp_lines | (cmp_lines_b << 16)
        tab[base + 2] = n_ori | (seq << 16)
        tab[base + 3] = start | (kv_len << 16)
        tab[base + 4] = kv_row0
        tab[base + 5] = cmp_row0
    ok = True
    for vis, vis_b, cl, clb, n_ori, start, kvl, seq, _r0, _r1 in vals:
        if not (
            0 <= vis < 65536
            and 0 <= vis_b < 65536
            and 0 <= cl < 65536
            and 0 <= clb < 65536
            and 0 <= n_ori < 65536
            and 0 <= start < 65536
            and 0 <= kvl < 32768
            and 0 <= seq < 32768
        ):
            ok = False
    if len(_PARAM_TAB_CACHE) >= _PARAM_TAB_CACHE_LIMIT:
        _PARAM_TAB_CACHE.clear()
    _PARAM_TAB_CACHE[_key] = (tab, ok)
    return tab, ok


# ------------------------------------------------------------------
# E7：FD unit 规划（按 tile 区间的 unit + LPT 均衡）
#
# 口径与 kernel 完全一致（mask3 / TND / 非 PA 的 CSA 路径）：
#   ori visible = [clip(row-pre,0,kv), clip(row+next+1,0,kv))，mask3 下
#   pre=q_len、next=kv_len-q_len；ori tile = ceil(vis/128)；
#   cmp 行数 = min(clip((rowq+1+next_cmp)//cmp_ratio,0,cmp_len), min(topk,idx_w))，
#   其中 next_cmp = cmp_len*cmp_ratio + resid - q_len（cmp_mask_mode!=0）；
#   cmp tile = ceil(cmp_lines/128)。TOK2 时一个 task 覆盖相邻两 token，
#   ori tile 取两行 vis 的较大者，cmp tile 为两行之和。
# ------------------------------------------------------------------
FD_TAB_STRIDE = 8  # 与 sparse_flash_mla_csa_kernel.FD_TAB_STRIDE 一致
FD_MAX_SPLITS = 8  # 每 task 的最大 unit 数（归并展开宽度）
FD_MIN_TILES_PER_UNIT = 8  # 单元碎片化下限（小规模时按 total/cores 收窄）


def _csa_token_vis_cmp(
    cu_q, cu_kv, cu_cmp, tok, cmp_topk, cmp_w, cmp_mask_mode, cmp_ratio, resid
):
    """单个 token 的 (vis, cmp_lines)（kernel mask3 公式）。"""
    nseq = len(cu_q) - 1
    seq = 0
    rowq = tok
    for bb in range(1, nseq):
        if tok >= cu_q[bb]:
            seq = bb
            rowq = tok - cu_q[bb]
    q_len = cu_q[seq + 1] - cu_q[seq]
    kv_len = cu_kv[seq + 1] - cu_kv[seq]
    causal = kv_len - q_len
    end = rowq + causal + 1
    if end < 0:
        end = 0
    if end > kv_len:
        end = kv_len
    start = rowq - q_len
    if start < 0:
        start = 0
    if start > kv_len:
        start = kv_len
    vis = end - start
    if vis < 0:
        vis = 0
    cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]
    next_cmp = cmp_len * cmp_ratio
    if cmp_mask_mode != 0:
        next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
    cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
    if cmp_end < 0:
        cmp_end = 0
    if cmp_end > cmp_len:
        cmp_end = cmp_len
    sc = cmp_topk[tok]
    if sc > cmp_w:
        sc = cmp_w
    if sc < 0:
        sc = 0
    return vis, min(cmp_end, sc)


def _csa_swiz_on_host(
    cu_q,
    cu_kv,
    cu_cmp,
    cmp_w,
    ori_mask_mode,
    ori_win_left,
    cmp_mask_mode,
    cmp_ratio,
    cores,
):
    """Choose contiguous (1) vs round-robin (0) task scheduling on host (CPU-only, zero NPU sync).

    Contiguous scheduling maximizes L2 KV locality when tokens have uniform tile counts
    (e.g., decode/prefill with long KV >> Q), whereas causal-triangle workloads with
    long Q (e.g., S_q == S_kv >= 2048) cause severe core load imbalance under contiguous
    partitioning (1.2x-1.7x) and benefit from round-robin scheduling.
    """
    total_q = int(cu_q[-1])
    if total_q <= cores or cores <= 1:
        return 1
    num_seqs = len(cu_q) - 1
    cq = [int(v) for v in cu_q]
    ckv = [int(v) for v in cu_kv]
    ccmp = [int(v) for v in cu_cmp]
    cr = max(1, int(cmp_ratio))
    cw = int(cmp_w)
    om = int(ori_mask_mode)
    owl = int(ori_win_left)
    cm = int(cmp_mask_mode)

    def _tiles_at(t):
        lo, hi = 0, num_seqs - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cq[mid + 1] <= t:
                lo = mid + 1
            else:
                hi = mid
        s = lo
        ql = cq[s + 1] - cq[s]
        kvl = ckv[s + 1] - ckv[s]
        cmpl = ccmp[s + 1] - ccmp[s]
        row = t - cq[s]
        if om in (3, 4):
            q_pos = (kvl - ql) + row
            if q_pos < 0:
                return 0
            ori_end = kvl if q_pos + 1 > kvl else q_pos + 1
            ori_start = (q_pos - owl) if (om == 4 and owl >= 0 and q_pos > owl) else 0
            ori_len = ori_end - ori_start if ori_end > ori_start else 0
        else:
            ori_len = kvl
        n_ori = (ori_len + 127) // 128
        if cm == 3:
            q_pos = (kvl - ql) + row
            cmp_end = (q_pos - owl) if (om == 4 and owl >= 0) else (q_pos + 1)
            if cmp_end < 0:
                cmp_end = 0
            cmp_avail = cmp_end // cr
            if cmp_avail > cmpl:
                cmp_avail = cmpl
            cmp_len = cw if cmp_avail > cw else cmp_avail
        else:
            cmp_len = cw if cmpl > cw else cmpl
        n_cmp = (cmp_len + 127) // 128
        return n_ori + n_cmp

    q_base, q_rem = divmod(total_q, cores)
    if total_q <= 1024:
        swiz_loads = []
        pos = 0
        for c in range(cores):
            cnt = q_base + (1 if c < q_rem else 0)
            swiz_loads.append(sum(_tiles_at(t) for t in range(pos, pos + cnt)))
            pos += cnt
        rr_loads = []
        for c in range(cores):
            rr_loads.append(sum(_tiles_at(t) for t in range(c, total_q, cores)))
    else:
        S = 16
        swiz_loads = []
        pos = 0
        for c in range(cores):
            cnt = q_base + (1 if c < q_rem else 0)
            if cnt == 0:
                swiz_loads.append(0.0)
            else:
                swiz_loads.append(
                    (sum(_tiles_at(pos + (k * cnt) // S) for k in range(S)) / S) * cnt
                )
            pos += cnt
        avg_all = sum(swiz_loads) / cores
        rr_loads = [avg_all]
    max_s = max(swiz_loads) if swiz_loads else 0
    max_r = max(rr_loads) if rr_loads else 0
    if max_r > 0 and max_s > max_r * 1.08:
        return 0
    return 1


def _csa_task_tiles(
    cu_q, cu_kv, cu_cmp, cmp_topk, cmp_w, cmp_mask_mode, cmp_ratio, resid, tok2
):
    """每个 FD task 的 tile 总数（ori + cmp），与 kernel 的 n_tiles 同口径。"""
    key = (
        tuple(cu_q),
        tuple(cu_kv),
        tuple(cu_cmp),
        tuple(cmp_topk),
        int(cmp_w),
        int(cmp_mask_mode),
        int(cmp_ratio),
        tuple(resid),
        int(tok2),
    )
    hit = _FD_TILE_CACHE.get(key)
    if hit is not None:
        return list(hit)
    total_tok = cu_q[-1]
    tiles = []
    if not tok2:
        for tok in range(total_tok):
            vis, cl = _csa_token_vis_cmp(
                cu_q,
                cu_kv,
                cu_cmp,
                tok,
                cmp_topk,
                cmp_w,
                cmp_mask_mode,
                cmp_ratio,
                resid,
            )
            tiles.append(
                (vis + KV_BLOCK - 1) // KV_BLOCK + (cl + KV_BLOCK - 1) // KV_BLOCK
            )
    else:
        for t in range(total_tok // 2):
            vis_a, cl_a = _csa_token_vis_cmp(
                cu_q,
                cu_kv,
                cu_cmp,
                2 * t,
                cmp_topk,
                cmp_w,
                cmp_mask_mode,
                cmp_ratio,
                resid,
            )
            vis_b, cl_b = _csa_token_vis_cmp(
                cu_q,
                cu_kv,
                cu_cmp,
                2 * t + 1,
                cmp_topk,
                cmp_w,
                cmp_mask_mode,
                cmp_ratio,
                resid,
            )
            vis = vis_a if vis_a > vis_b else vis_b
            tiles.append(
                (vis + KV_BLOCK - 1) // KV_BLOCK
                + (cl_a + KV_BLOCK - 1) // KV_BLOCK
                + (cl_b + KV_BLOCK - 1) // KV_BLOCK
            )
    if len(_FD_TILE_CACHE) >= _FD_TILE_CACHE_LIMIT:
        _FD_TILE_CACHE.clear()
    _FD_TILE_CACHE[key] = tuple(tiles)
    return tiles


def _fd_lpt_assign(segments, cores):
    """LPT：区间按 tile 数降序，贪心给当前负载最小的核。"""
    loads = [0] * cores
    assign = [[] for _ in range(cores)]
    order = sorted(
        range(len(segments)),
        key=lambda i: (-segments[i][3], segments[i][0], segments[i][1]),
    )
    for si in order:
        c = min(range(cores), key=lambda c: (loads[c], c))
        loads[c] += segments[si][3]
        assign[c].append(si)
    return loads, assign


def _fd_emit_order(assign, cores):
    """把 LPT 分配展开成 unit 流：position % cores 恰为该 unit 的核。"""
    max_round = max((len(v) for v in assign), default=0)
    order = []
    for r in range(max_round):
        for c in range(cores):
            if r < len(assign[c]):
                order.append(assign[c][r])
    return order


def _fd_round_robin_crit(task_tiles, splits, cores):
    """现状（(task,split) 等分 + 取模映射）的临界 tile 数。"""
    loads = [0] * cores
    for t, T in enumerate(task_tiles):
        for s in range(splits):
            lo = (s * T) // splits
            hi = ((s + 1) * T) // splits
            loads[(t * splits + s) % cores] += hi - lo
    return max(loads) if loads else 0


def _fd_plan_units(
    task_tiles,
    cores,
    max_splits=FD_MAX_SPLITS,
    min_tiles=FD_MIN_TILES_PER_UNIT,
    force_k=None,
    piece=None,
):
    """枚举每-unit tile 目标 p，返回 LPT 临界负载最小的规划。

    返回 dict：segments=[(task, idx, lo, hi, k_size)]（按 task 分组累计的
    lo/hi 由 idx 重算）、units=发射序、k_max=每 task 最大 unit 数、crit。
    """
    total = sum(task_tiles)
    if total <= 0 or not task_tiles:
        return None
    _cache_key = (
        "u",
        tuple(task_tiles),
        int(cores),
        int(max_splits),
        int(min_tiles),
        force_k,
        piece,
    )
    _hit = _FD_PLAN_CACHE.get(_cache_key)
    if _hit is not None:
        return _hit
    avg = (total + cores - 1) // cores
    min_piece = max(1, min(min_tiles, avg))
    max_piece = max(task_tiles)
    best = None
    p_range = [piece] if piece is not None else list(range(min_piece, max_piece + 1))
    for p in p_range:
        segments = []
        k_max = 1
        for t, T in enumerate(task_tiles):
            if T <= 0:
                segments.append((t, 0, 0, 1))
                continue
            if force_k is not None:
                k = min(force_k, max_splits, max(1, T // min_piece))
            else:
                k = (T + p - 1) // p
                k = min(k, max_splits)
                k = min(k, max(1, T // min_piece))
            k = max(1, k)
            k_max = max(k_max, k)
            for i in range(k):
                lo = (i * T) // k
                hi = ((i + 1) * T) // k
                segments.append((t, i, lo, hi - lo))
        loads, assign = _fd_lpt_assign(segments, cores)
        crit = max(loads)
        # 目标：临界 tile 数为主，unit 数按 0.25 tile/unit 折价（E7 实测 #022：
        # K=3/96 unit 优于 K=4/128 unit 与 K=7/224 unit——归并落盘/读取和
        # 单元末 MTE3 闸门的固定成本不可忽略）。整数化避免浮点比较。
        score = crit * 4 + len(segments)
        key = (score, crit, len(segments))
        if best is None or key < best[0]:
            best = (key, segments, assign, k_max)
    key, segments, assign, k_max = best
    crit = key[1]
    _res = {
        "segments": segments,
        "units": _fd_emit_order(assign, cores),
        "k_max": k_max,
        "crit": crit,
        "n_units": len(segments),
        "min_piece": min_piece,
    }
    if len(_FD_PLAN_CACHE) >= _FD_PLAN_CACHE_LIMIT:
        _FD_PLAN_CACHE.clear()
    _FD_PLAN_CACHE[_cache_key] = _res
    return _res


def _fd_plan_anchors(task_tiles, cores, peel=None, kcap=2, unit_penalty=0.5):
    """E9：锚点切分规划（FD 部分量的「大段留 UB」变体）。

    每个 task 切成 [anchor, small×（k-1）]：anchor 段最大、放在该核最后发射，
    它自己就是归并单元 —— 未归一化 O 留在 UB 不落盘，归并只读其余 small 段的
    fp32 部分量。约束：
      - 每核至多 1 个 anchor（anchor 的第一条 O 必须活到屏障之后）；
      - 每核内 anchor 排在最后（否则后续 unit 的 vec2 会覆盖 ub_o）。

    枚举 small 段大小 p 与每 task 段数 k（k<=kcap），LPT 分配后按
    「折算临界 = load + unit_penalty*(unit 数-1)」选优（同 tile 数下
    单 unit 核优于多 unit 核）。
    """
    total = sum(task_tiles)
    if total <= 0 or not task_tiles:
        return None
    # E33：每个 task 恒产生 1 个 anchor，约束「每核至多 1 个 anchor」等价于
    # task 数 <= 核数；不满足时逐 p 枚举后必然全部 `len(anchors) > cores`
    # 跳过并返回 None，这里直接早退（实测 #022 每次省 0.52 ms 纯枚举）。
    if len(task_tiles) > cores:
        return None
    _cache_key = (
        "a",
        tuple(task_tiles),
        int(cores),
        peel,
        int(kcap),
        float(unit_penalty),
    )
    _hit = _FD_PLAN_CACHE.get(_cache_key)
    if _hit is not None:
        return _hit
    pmax = max(task_tiles)
    p_range = [peel] if peel is not None else list(range(1, pmax))
    best = None
    for p in p_range:
        if p <= 0:
            continue
        for kfix in range(2, min(kcap, FD_MAX_SPLITS) + 1):
            segments = []
            anchors = set()
            for t, T in enumerate(task_tiles):
                if T <= 0:
                    anchors.add(len(segments))
                    segments.append((t, 0, 0, 1))
                    continue
                k = min(kfix, max(1, T // p))
                anchor_sz = T - (k - 1) * p
                anchors.add(len(segments))
                segments.append((t, 0, 0, anchor_sz))
                for i in range(1, k):
                    lo = anchor_sz + (i - 1) * p
                    segments.append((t, i, lo, p))
            if len(anchors) > cores:
                continue
            loads = [0] * cores
            nunits = [0] * cores
            assign = [[] for _ in range(cores)]
            free = list(range(cores))
            for si in sorted(anchors, key=lambda si: (-segments[si][3], si)):
                c = min(free, key=lambda c: (loads[c], c))
                free.remove(c)
                loads[c] += segments[si][3]
                nunits[c] += 1
                assign[c].append(si)
            for si in sorted(
                (si for si in range(len(segments)) if si not in anchors),
                key=lambda si: (-segments[si][3], si),
            ):
                c = min(range(cores), key=lambda c: (loads[c], c))
                loads[c] += segments[si][3]
                nunits[c] += 1
                assign[c].append(si)
            # 每核内 anchor 必须最后发射（WAR：后续 unit 会重写 ub_o）。
            for c in range(cores):
                assign[c].sort(key=lambda si: (1 if si in anchors else 0))
            crit = max(loads) if loads else 0
            crit_eff = max(
                (loads[c] + unit_penalty * (nunits[c] - 1) for c in range(cores)),
                default=0.0,
            )
            k_max = max(
                (sum(1 for s in segments if s[0] == t) for t in range(len(task_tiles))),
                default=1,
            )
            key = (round(crit_eff * 4), crit, len(segments))
            if best is None or key < best[0]:
                best = (key, segments, assign, k_max, crit, anchors)
    if best is None:
        return None
    key, segments, assign, k_max, crit, anchors = best
    # ---- 位置映射修正（E9 根因）------------------------------------------
    # kernel 的 unit 流是「unit 号 u 由核 u % cores 执行、步长 = cores」，即
    # 位置 p 恒属核 p%cores。任何一轮不满 cores 时，简单按轮展平会让后续
    # unit 整体错位到别的核（E7 的 LPT 折价账因此经常名不副实：预测 crit 12
    # 实际跑成 17）。这里按位置 p = c + r*cores 显式落位，空槽填 n_tiles=0
    # 的空 unit（idx=-1，不参与 ws 计数、不落盘、不归并）。
    #
    # 空 unit 一律放在该核的**最前面**：若排在真实 unit 之后，真实 unit 末
    # 2~3 个 tile 的流水要等 sentinel 步才排空，而空 unit 的 Q 信用链等的正是
    # 那次预取 ⇒ 挂死（实测 60s 超时）。放最前面时 q_chain==0，链按既有条件
    # 跳过，且没有需要消费的预取。
    max_round = max((len(v) for v in assign), default=0)
    nslots = max_round * cores
    order = [None] * nslots
    for c in range(cores):
        lst = assign[c]
        n_empty = max_round - len(lst)
        t_first = segments[lst[0]][0] if lst else 0
        units_c = []
        for _ in range(n_empty):
            segments.append((t_first, -1, 0, 0))
            units_c.append(len(segments) - 1)
        units_c.extend(lst)
        for r, si in enumerate(units_c):
            order[c + r * cores] = si
    _res = {
        "segments": segments,
        "units": order,
        "k_max": k_max,
        "crit": crit,
        "crit_eff": key[0] / 4.0,
        "n_units": nslots,
        "n_real": sum(1 for s in segments if s[1] >= 0),
        "anchors": anchors,
        "min_piece": 1,
    }
    if len(_FD_PLAN_CACHE) >= _FD_PLAN_CACHE_LIMIT:
        _FD_PLAN_CACHE.clear()
    _FD_PLAN_CACHE[_cache_key] = _res
    return _res


def _fd_plan_table(task_tiles, plan):
    """把规划展开为 GM int32 表（发射序，FD_TAB_STRIDE 项/unit）。

    字段：[task, tile_start, tile_end, ws_slot, merge_flag, k, idx, task*K]。
    ws_slot = task*K + idx（归并侧 base0 = task*K*stride，与 E6 布局一致）。
    E9 锚点模式：merge_flag = 1 标在 **anchor 单元**（idx==0，O 留在 UB、由
    该单元自己归并）；非 anchor 段按 idx>0 落盘。非锚点模式仍标在末段。
    """
    segs = plan["segments"]
    _cache_key = id(plan)
    _hit = _FD_TAB_CACHE.get(_cache_key)
    if _hit is not None and _hit[0] is plan:
        # 返回副本：E10 项表臂会在返回的表后追加项。
        return list(_hit[1])
    k_max = plan["k_max"]
    anchors = plan.get("anchors")
    counts = {}
    for t, i, _lo, _sz in segs:
        if i >= 0:
            counts[t] = counts.get(t, 0) + 1
    tab = [0] * (len(segs) * FD_TAB_STRIDE)
    for pos, si in enumerate(plan["units"]):
        t, idx, lo, sz = segs[si]
        if idx < 0:
            # E9：空 unit（位置映射占位）——task 仅用于 Q 预取，区间为空。
            tab[pos * FD_TAB_STRIDE + 0] = t
            tab[pos * FD_TAB_STRIDE + 5] = 1
            continue
        k = counts[t]
        base = pos * FD_TAB_STRIDE
        tab[base + 0] = t
        tab[base + 1] = lo
        tab[base + 2] = lo + sz
        tab[base + 3] = t * k_max + idx
        tab[base + 4] = (
            (1 if si in anchors else 0)
            if anchors is not None
            else (1 if idx == k - 1 else 0)
        )
        tab[base + 5] = k
        tab[base + 6] = idx
        tab[base + 7] = t * k_max
    if len(_FD_TAB_CACHE) >= _FD_TAB_CACHE_LIMIT:
        _FD_TAB_CACHE.clear()
    _FD_TAB_CACHE[_cache_key] = (plan, tab)
    return list(tab)


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
    global _ZERO_TRACE
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

    # ------------------------------------------------ F7 launch plan fast path
    # 输出 / lse 池槽先取，快慢两路共用（池索引每次调用只前进一次）。
    _fast_output = _pooled_output(q)
    _fast_lse = (
        _pooled_lse(
            device,
            (1, total_rows, heads) if layout_q == "TND" else (batch, 1, s1_len, heads),
        )
        if want_lse
        else zero_len
    )
    _fkey = None
    if _FAST_ON and not compile_only:
        _fkey = _fast_key(
            route,
            device,
            dtype,
            (
                heads,
                int(ori_mask_mode),
                int(cmp_mask_mode),
                int(cmp_ratio),
                int(ori_win_left),
                int(ori_win_right),
                layout_q,
                layout_kv,
                want_lse,
                scale_value,
            ),
            (
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
            ),
            _fast_output,
            _fast_lse,
        )
        _fplan = _FAST_PLANS.get(_fkey)
        if _fplan is not None and _fplan["device"] != int(
            torch_npu.npu.current_device()
        ):
            # 当前设备与计划绑定的设备不一致：退回常规路径（与 tla 执行器的
            # 设备一致性检查同语义），不在错误的设备上发射。
            _fplan = None
        if _fplan is not None:
            _fstream = _fplan["stream"](_fplan["device"])
            for _fbuf in _fplan["zero"]:
                _fbuf.zero_()
            _fplan["launch"](stream=int(_fstream))
            return _fast_output, _fast_lse
    _ZERO_TRACE = [] if _FAST_ON and not compile_only else None

    # ------------------------------------------------ cu_seqlens
    per_q = (batch, s1_len) if layout_q == "BSND" else None
    per_kv = (batch, kv_bs) if layout_kv == "BSND" else None
    # E33：本调用 host 需要的全部小 int32 张量一次 D2H，供 cu_seqlens 重建、
    # BSND ragged 检查复用，避免逐项 .tolist() 各自同步。
    _pre = (
        ("cq", cu_seqlens_q),
        ("sq", seqused_q),
        ("ckv", cu_seqlens_ori_kv),
        ("skv", seqused_ori_kv),
        ("cc", cu_seqlens_cmp_kv if has_cmp else None),
        ("sc", seqused_cmp_kv if has_cmp else None),
    )
    # 只在全部是 1-D int32（_build_cu 的合法输入）时批量取；否则退回
    # 逐项取，保持 _build_cu 的校验报错与原实现一致。
    _pre_ok = all(_t is None or (_t.dtype == i32 and _t.ndim == 1) for _tag, _t in _pre)
    _pre_t = []
    _pre_tag = []
    if _pre_ok:
        for _tag, _t in _pre:
            if _t is not None:
                _pre_t.append(_t)
                _pre_tag.append(_tag)
    _host_vals = dict(zip(_pre_tag, _d2h_lists(_pre_t))) if _pre_t else {}
    cu_q_t, cu_q = _build_cu(
        cu_seqlens_q,
        seqused_q,
        batch,
        per_q,
        device,
        _host_vals.get("cq") if cu_seqlens_q is not None else _host_vals.get("sq"),
    )
    cu_kv_t, cu_kv = _build_cu(
        cu_seqlens_ori_kv,
        seqused_ori_kv,
        batch,
        per_kv,
        device,
        _host_vals.get("ckv")
        if cu_seqlens_ori_kv is not None
        else _host_vals.get("skv"),
    )
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
        q_lens = _host_vals.get("sq")
        if q_lens is None:
            q_lens = [int(v) for v in seqused_q.reshape(-1).tolist()]
        if len(q_lens) == batch and any(v != s1_len for v in q_lens):
            q_rows_idx = _pack_index(q_lens, s1_len)
    if not paged and layout_kv == "BSND" and seqused_ori_kv is not None:
        kv_lens = _host_vals.get("skv")
        if kv_lens is None:
            kv_lens = [int(v) for v in seqused_ori_kv.reshape(-1).tolist()]
        if len(kv_lens) == batch and any(v != kv_bs for v in kv_lens):
            kv_rows_idx = _pack_index(kv_lens, kv_bs)
    if not paged and has_cmp and layout_kv == "BSND" and seqused_cmp_kv is not None:
        cmp_lens = _host_vals.get("sc")
        if cmp_lens is None:
            cmp_lens = [int(v) for v in seqused_cmp_kv.reshape(-1).tolist()]
        if len(cmp_lens) == batch and any(v != cmp_bs for v in cmp_lens):
            cmp_rows_idx = _pack_index(cmp_lens, cmp_bs)
    run_rows = int(q_rows_idx.numel()) if q_rows_idx is not None else total_rows

    if has_cmp:
        cmp_per_batch = (batch, cmp_bs) if layout_kv == "BSND" else None
        cu_cmp_t, cu_cmp = _build_cu(
            cu_seqlens_cmp_kv,
            seqused_cmp_kv,
            batch,
            cmp_per_batch,
            device,
            _host_vals.get("cc")
            if cu_seqlens_cmp_kv is not None
            else _host_vals.get("sc"),
        )
    else:
        cu_cmp_t, cu_cmp = None, [0, 0]

    # ------------------------------------------------ 输出 / workspace
    output = _fast_output
    if want_lse:
        lse = _fast_lse
        lse_flat = (
            _workspace("lse_pack", device, f32, run_rows * heads)
            if q_rows_idx is not None
            else lse.reshape(-1)
        )
    else:
        # kernel 把 lse 当真正的输出缓冲：形状参与编译期 UB 划分，核对范围内
        # 也仍会被写。未请求 lse 时按同样的形状分配一块临时缓冲，只在返回时
        # 换成空张量（此前的 1 元素占位会让 UB 划分与正常路径不同，实测
        # ORI_SPARSE 的 3 例因此出现 1%~2% 的元素偏差）。
        lse = _fast_lse
        lse_flat = _workspace("lse_flat", device, f32, run_rows * heads)

    q_rows_flat = q.reshape(total_rows, heads * HEAD_DIM)
    if q_rows_idx is not None:
        # 复用 workspace：ragged 打包张量若每次新分配，会被 _tla_tensor 的
        # (data_ptr, ...) 绑定缓存当作新键持有，缓存随调用增长。
        q_pack = _workspace(
            "q_pack", device, q.dtype, int(q_rows_idx.numel()) * heads * HEAD_DIM
        ).reshape(run_rows, heads * HEAD_DIM)
        torch.index_select(q_rows_flat, 0, q_rows_idx, out=q_pack)
        q2 = q_pack.reshape(-1, HEAD_DIM)
    else:
        q2 = q_rows_flat.reshape(-1, HEAD_DIM).contiguous()
    kv_rows_flat = ori_kv.reshape(-1, HEAD_DIM)
    if kv_rows_idx is not None:
        kv_pack = _workspace(
            "kv_pack", device, ori_kv.dtype, int(kv_rows_idx.numel()) * HEAD_DIM
        ).reshape(-1, HEAD_DIM)
        torch.index_select(kv_rows_flat, 0, kv_rows_idx, out=kv_pack)
        kv2 = kv_pack
    else:
        kv2 = kv_rows_flat.contiguous()
    if q_rows_idx is not None:
        out_pack = _workspace(
            "out_pack", device, output.dtype, run_rows * heads * HEAD_DIM
        ).reshape(run_rows, heads * HEAD_DIM)
        out2 = out_pack.reshape(-1, HEAD_DIM)
    else:
        out_pack = None
        out2 = output.reshape(total_rows * heads, HEAD_DIM)
    if has_cmp:
        cmp_rows_flat = cmp_kv.reshape(-1, HEAD_DIM)
        if cmp_rows_idx is not None:
            cmp_pack = _workspace(
                "cmp_pack",
                device,
                cmp_kv.dtype,
                int(cmp_rows_idx.numel()) * HEAD_DIM,
            ).reshape(-1, HEAD_DIM)
            torch.index_select(cmp_rows_flat, 0, cmp_rows_idx, out=cmp_pack)
            cmp2 = cmp_pack
        else:
            cmp2 = cmp_rows_flat.contiguous()
    else:
        cmp2 = None

    sinks_use = (
        sinks
        if sinks is not None
        else _workspace("sinks", device, f32, heads, zero=True)
    )
    has_sinks = 0 if sinks is None else 1

    resid = (
        cmp_residual_kv.reshape(-1).to(i32).contiguous()
        if (cmp_residual_kv is not None and cmp_ratio > 1 and cmp_mask_mode != 0)
        else _workspace("resid0", device, i32, num_seqs, zero=True)
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
    # E13：实际发射块数（默认 = 规划核数；CSA+FD 时按 unit/归并项数收窄）。
    block_num_launch = block_num
    # F3：SWA/HCA 长序列的逐任务标量表（ptab）。N>=PTAB_MIN_TASKS 时启用，
    # 取代 kernel 内 O((N/q_bn)^2) 的 gtile 前缀游走；小规模用例保持旧路径。
    _ptab_t = None
    _use_ptab = 0
    if route in ("swa", "hca") and run_rows * g_split >= PTAB_MIN_TASKS:
        _resid_host = _d2h_lists([resid])[0] if route == "hca" else [0] * num_seqs
        _ptab_t = _swa_hca_task_table(
            route,
            cu_q,
            cu_kv,
            cu_cmp,
            _resid_host,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            g_split,
            block_num_launch,
            heads,
            device,
        )
        if _ptab_t is not None:
            _use_ptab = 1
    # E33：tiling 用 Python 列表累积，各分支入口一次物化成（按内容缓存的）
    # 设备张量；避免 2~3 次逐元素 H2D 赋值（每次 ~20 µs）。
    tiling = [0, 0, 0, 0]
    # E33：稀疏路由后面各分支都要用的 topk / resid 主机列表，一次 D2H 取齐
    # （替换 tiling 段与分支内重复的逐项 .tolist()）。
    _sparse_lists = {}
    if route in ("csa", "os", "ocs"):
        _want_t = []
        _want_n = []
        if ori_topk_length is not None:
            _want_t.append(ori_topk_length)
            _want_n.append("ori")
        if cmp_topk_length is not None:
            _want_t.append(cmp_topk_length)
            _want_n.append("cmp")
        if route in ("csa", "ocs"):
            _want_t.append(resid)
            _want_n.append("resid")
        for _n, _v in zip(_want_n, _d2h_lists(_want_t)):
            _sparse_lists[_n] = _v
    if route in ("csa", "os", "ocs") and g_split == 2:
        ori_topk = _sparse_lists.get("ori")
        cmp_topk = _sparse_lists.get("cmp")
        resid_list = _sparse_lists.get("resid") or [0] * num_seqs
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
        ws = _workspace("ws", device, dtype, WS_BLOCKS * KV_STAGE_BYTES_EL, zero=True)
    else:
        ws = _dummy(device, dtype)

    # ------------------------------------------------ 分派
    if route == "swa":
        t_kv = _tla_tensor(kv2)
        m_rows = heads // g_split - (heads // g_split) // 2
        rows8 = (m_rows + 7) // 8 * 8
        bc_stride = m_rows * HEAD_DIM + rows8 * 2
        bc_ws = _workspace("bc_ws", device, f32, block_num * 2 * bc_stride, zero=True)
        tiling = _i32_cached(tiling, device)
        ptab = _ptab_t if _use_ptab else _dummy(device, i32)
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
            _tla_tensor(ptab),
            _tla_tensor(_kv_zero(device, dtype)),
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
                ptab,
                _kv_zero(device, dtype),
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
            _use_ptab,
        )
        kernel_fn = sparse_flash_mla_swa_kernel
    elif route == "hca":
        q_hca = q2
        if g_split == 2 and heads % 2 == 1:
            half_g = (heads + 1) // 2
            m_cube = half_g + (half_g % 2)
            pad_rows = max(0, m_cube - (heads - half_g))
            if pad_rows:
                # 复用 workspace：heads 为奇数时的 padding 张量若每次新建，会被
                # _tla_tensor 的 (data_ptr, ...) 绑定缓存当作新键持有。
                q_pad = _workspace(
                    "q_hca_pad",
                    device,
                    q2.dtype,
                    (q2.shape[0] + pad_rows) * HEAD_DIM,
                    zero=True,
                ).reshape(q2.shape[0] + pad_rows, HEAD_DIM)
                q_pad[: q2.shape[0]] = q2
                q_hca = q_pad
        sinks_hca = sinks_use
        if g_split == 2 and heads % 2 == 1:
            sink_pad = ((heads + 7) // 8) * 8 - heads
            if sink_pad:
                s_pad = _workspace(
                    "sink_hca_pad",
                    device,
                    f32,
                    heads + sink_pad,
                    zero=True,
                )
                s_pad[:heads] = sinks_use
                sinks_hca = s_pad
        tiling = _i32_cached(tiling, device)
        ptab = _ptab_t if _use_ptab else _dummy(device, i32)
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
            _tla_tensor(ptab),
            _tla_tensor(_kv_zero(device, dtype)),
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
                ptab,
                _kv_zero(device, dtype),
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
            _use_ptab,
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
        cmp_phy = _workspace(
            "cmp_phy", device, torch.int64, run_rows * phy_w, zero=True
        )
        # E3：H=32 的 TND/mask3 非 PA CSA 把同 seq 相邻两个 token 合成一个 task
        # （kernel m=64，ori tile 数减半）。前置条件全部成立才置 1；其余任何
        # 路由/用例一律 TOK2=0，指令流与旧映射逐条一致。
        tok2 = 0
        if (
            kv_layout == 1
            and g_split == 1
            and heads == 32
            and ori_mask_mode == 3
            and pa_o_bs == 0
        ):
            tok2 = 1
            for _i in range(num_seqs):
                _len = int(cu_q[_i + 1]) - int(cu_q[_i])
                if _len <= 0 or _len % 2 != 0:
                    tok2 = 0
        # E4：FlashDecoding（KV 维分片）。只对 CSA / TND 或 PA_BBND / mask3 /
        # g_split==1 的族启用（E11：PA_BBND 也纳入；FD 只按逻辑 tile 区间切分，
        # 与 K 的寻址方式无关，kernel 的 load_k_pa 已支持任意逻辑起点）。
        # E6：正确性侧已证明「多 unit/核」可安全启用（根因是 fd_store_partial
        # 的 MTE3 落盘与下一单元 vec 写的 WAR + 空 unit 的 Q 信用链，均已修，
        # 见 RUNLOG §3/§4）。但**性能**上的自动档仍要按临界负载选：
        #   - tasks < cores：每 task 独占一核，只有把同一 task 的 tile 切到
        #     空闲核才有收益，且必须 units <= cores（每 unit 一核）——
        #     fd_splits = min(4, cores // tasks)。#021（5 task/28 核）=4 → −0.8%/持平；
        #     #026（20 task）=1、#028（15 task）=1 → 不切（实测 S=2 分别 +25%、+0.4%：
        #     40/30 unit 落在 28 核上时有两个核要背 2 个 unit，临界负载没降，只多
        #     了归并/屏障开销）。
        #   - tasks >= cores：天然多 task/核，S=2 把 task 粒度减半，临界核从
        #     ceil(tasks/cores)=2 个 task 降到 1.5 个 task —— #022（32 task）
        #     195.54→159.5 µs（−18.4%）。
        # 缺省 1 ⇒ kernel 走原路径（新增指令全部折叠）。
        fd_splits = 1
        _tasks = 0
        _fd_ok_layout = (
            (kv_layout == 1 and pa_o_bs == 0)  # TND 非 PA（原有路径）
            or (kv_layout == 2 and pa_o_bs > 0)  # E11：PA_BBND
        )
        _fd_ok = (
            _fd_ok_layout and g_split == 1 and ori_mask_mode == 3 and cmp_mask_mode == 3
        )
        if _fd_ok:
            _tokens = int(cu_q[num_seqs])
            _tasks = _tokens // 2 if tok2 == 1 else _tokens
            _cores = int(block_num)
            _auto = 1
            if _tasks > 0:
                if _tasks < _cores:
                    # E10：归并统计段向量化后，小 task 数的临界从「更多分片」
                    # 转回「更少分片」：S=3 与 S=4 的临界 tile 数同为 3
                    # （#021：round-robin crit 3），但 S=3 少 5 个 unit、归并
                    # 每个 task 少读 1 份 128KB 部分量。实测 #021 S=3 中位
                    # 16.1µs vs S=4 16.4~17.4（同窗口配对）。
                    _auto = max(1, min(3, _cores // _tasks))
                else:
                    _auto = 2
            _env = os.environ.get("SMLA_FD_SPLITS")
            fd_splits = int(_env) if _env is not None else _auto
            fd_splits = max(1, min(4, fd_splits))
        # E7：tile 区间 unit 规划 + LPT 均衡。默认只有预计临界 tile 数比现状
        # （等分映射）少 >=5% 且每 task 切出 >1 段时才启用；FD=0/FD_SPLITS=1
        # 的旧路径完全不变。env：SMLA_FD_PLAN=0/1 强制、SMLA_FD_MAXSPLITS、
        # SMLA_FD_MIN_TILES、SMLA_FD_K（每 task 固定段数）、SMLA_FD_PIECE。
        fd_plan = 0
        fd_anchor = 0
        fd_item = 0
        fd_tab = _i32_cached([0], device)
        if _fd_ok and _tasks > 0:
            # E33：与 tiling 段/参数表共用同一份 host 列表。BSND ragged 时
            # topk 已被 index_select，必须以重排后的张量为准重新取。
            if q_rows_idx is None and "cmp" in _sparse_lists:
                _cmp_topk_list = _sparse_lists["cmp"]
            else:
                _cmp_topk_list = [int(v) for v in topk.reshape(-1).tolist()]
            _resid_list = (
                _sparse_lists["resid"]
                if "resid" in _sparse_lists
                else [int(v) for v in resid.reshape(-1).tolist()]
            )
            _plan = None
            try:
                _task_tiles = _csa_task_tiles(
                    cu_q,
                    cu_kv,
                    cu_cmp,
                    _cmp_topk_list,
                    cmp_w,
                    cmp_mask_mode,
                    cmp_ratio,
                    _resid_list,
                    tok2,
                )
                if len(_task_tiles) == _tasks:
                    _env_k = os.environ.get("SMLA_FD_K")
                    _env_p = os.environ.get("SMLA_FD_PIECE")
                    _plan = _fd_plan_units(
                        _task_tiles,
                        _cores,
                        max_splits=int(
                            os.environ.get("SMLA_FD_MAXSPLITS", FD_MAX_SPLITS)
                        ),
                        min_tiles=int(
                            os.environ.get("SMLA_FD_MIN_TILES", FD_MIN_TILES_PER_UNIT)
                        ),
                        force_k=int(_env_k) if _env_k is not None else None,
                        piece=int(_env_p) if _env_p is not None else None,
                    )
                # E9：锚点切分（anchor 段的 O 留 UB，归并只读其余段的部分量）。
                # 选择规则：SMLA_FD_ANCHOR=1 强制；=0 关闭；未设时仅在锚点
                # 计划的临界 tile 数 **严格优于** E7 规划时采用（#026：12<17）。
                # 这样 #021（锚点 5 > E7 3）、#028（5 == 5）、#022（task 数
                # 32 > cores，无锚点计划）都保持 E7 原路径不变。
                # E11：PA_BBND 下允许「同 crit 也选锚点」——#025 的 E7 与锚点
                # 同为 crit 9，但锚点的 O 留 UB、8 个 split task 各少读 1 份
                # 部分量，实测 26.30 vs 27.49 µs；非 PA 仍保持严格 <，不受影响。
                _pa_layout = kv_layout == 2 and pa_o_bs > 0
                _use_anchor = 0
                _anchor_env = os.environ.get("SMLA_FD_ANCHOR")
                if _plan is not None and _anchor_env != "0":
                    _a_peel = os.environ.get("SMLA_FD_APEEL")
                    _a_k = os.environ.get("SMLA_FD_AK")
                    _ap = _fd_plan_anchors(
                        _task_tiles,
                        _cores,
                        peel=int(_a_peel) if _a_peel is not None else None,
                        kcap=int(_a_k) if _a_k is not None else 2,
                    )
                    if _ap is not None and (
                        _anchor_env == "1"
                        or _ap["crit"] < _plan["crit"]
                        or (_pa_layout and _ap["crit"] <= _plan["crit"])
                    ):
                        _plan = _ap
                        _use_anchor = 1
                _dbg = os.environ.get("SMLA_FD_DEBUG") == "1"
                if _plan is not None:
                    _cur_crit = _fd_round_robin_crit(
                        _task_tiles, fd_splits if fd_splits > 1 else 1, _cores
                    )
                    _enable = (
                        _plan["k_max"] > 1 and _plan["crit"] * 100 <= _cur_crit * 95
                    )
                    _force = os.environ.get("SMLA_FD_PLAN")
                    if _force is not None:
                        _enable = int(_force) != 0 and _plan["k_max"] > 1
                    if _dbg:
                        print(
                            json.dumps(
                                {
                                    "fd": "anchor"
                                    if (_use_anchor and _enable)
                                    else ("plan" if _enable else "legacy"),
                                    "tasks": _tasks,
                                    "cores": _cores,
                                    "task_tiles": _task_tiles,
                                    "cur_crit": _cur_crit,
                                    "plan_crit": _plan["crit"],
                                    "crit_eff": _plan.get("crit_eff"),
                                    "units": _plan["n_units"],
                                    "k_max": _plan["k_max"],
                                    "min_piece": _plan["min_piece"],
                                    "legacy_splits": fd_splits,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if _enable:
                        fd_splits = _plan["k_max"]
                        fd_plan = 1
                        fd_anchor = _use_anchor
                        fd_tab = _i32_cached(_fd_plan_table(_task_tiles, _plan), device)
                        tiling[2] = _plan["n_units"]
                        tiling[3] = _tasks
                        # E10：归并项表（仅非锚点、每 task split 数 <= 2、
                        # 项数 <= cores）。每项 [task, part, mode]：
                        # mode0 = 单 split task 整宽（mrg=1, splits=1）；
                        # mode1 = 双 split task D 半分（mrg=2, splits=2）。
                        # 每核至多 1 项 ⇒ 临界归并读量减半且不读空槽。
                        if _use_anchor == 0:
                            _counts = {}
                            for _t, _i, _lo, _sz in _plan["segments"]:
                                if _i >= 0:
                                    _counts[_t] = _counts.get(_t, 0) + 1
                            _maxk = max(_counts.values()) if _counts else 1
                            _n_multi = 0
                            for _t in range(_tasks):
                                if _counts.get(_t, 0) >= 2:
                                    _n_multi += 1
                            # E10 结论：#028 为 AIC 临界，项表只削 1~2.6%
                            # （<3% 门槛），默认关闭以还原 base 路径；
                            # SMLA_FD_ITEM=1 可复现实验臂。
                            _env_it = os.environ.get("SMLA_FD_ITEM")
                            _want_it = 0 if _env_it is None else int(_env_it)
                            if (
                                len(_counts) == _tasks
                                and _maxk <= 2
                                and _tasks + _n_multi <= _cores
                                and _want_it
                            ):
                                _items = []
                                for _t in range(_tasks):
                                    if _counts.get(_t, 0) == 1:
                                        _items.extend((_t, 0, 0, 0))
                                    elif _counts.get(_t, 0) == 2:
                                        _items.extend((_t, 0, 1, 0, _t, 1, 1, 0))
                                _tab = _fd_plan_table(_task_tiles, _plan)
                                _tab.extend(_items)
                                fd_tab = _i32_cached(_tab, device)
                                tiling[0] = len(_items) // 4
                                fd_item = 1
                            if _dbg:
                                print(
                                    json.dumps(
                                        {
                                            "fd_item": fd_item,
                                            "item_count": tiling[0],
                                            "per_task_k": [
                                                _counts.get(_t, 0)
                                                for _t in range(_tasks)
                                            ],
                                        }
                                    ),
                                    flush=True,
                                )
                elif _dbg:
                    print(
                        json.dumps(
                            {"fd": "legacy", "tasks": _tasks, "reason": "no-plan"}
                        ),
                        flush=True,
                    )
            except Exception as _exc:  # noqa: BLE001 — 规划失败退回旧路径
                fd_plan = 0
                fd_tab = _i32_cached([0], device)
                if os.environ.get("SMLA_FD_DEBUG") == "1":
                    print(json.dumps({"fd": "legacy", "error": str(_exc)}), flush=True)
        if fd_splits > 1:
            # 归并 D 维分摊：小 task 数时一个 task 的 O 累加拆到多个空闲核
            # （512/FD_MRG 列/项，FD_MRG ∈ {1,2,4}）。
            fd_mrg = 1
            if _tasks <= _cores:
                fd_mrg = min(4, max(1, _cores // _tasks))
            _envm = os.environ.get("SMLA_FD_MRG")
            if _envm is not None:
                fd_mrg = int(_envm)
            fd_mrg = 1 if fd_mrg <= 1 else (2 if fd_mrg <= 2 else 4)
            if fd_splits == 1:
                fd_mrg = 1
            if fd_anchor == 1:
                # E9 锚点臂：归并由 anchor 单元自己做（其 O 在 UB 整宽），
                # D 维不再分摊（分摊会与「O 留 UB」冲突，且 20 个 anchor
                # 已占满 20 核）。
                fd_mrg = 1
            _m_host = (2 * heads) if tok2 == 1 else heads
            # 统计块宽 = 64（kernel 的 FD_STAT_SLOT），unit 槽尾带 4 块统计。
            _fd_stride = _m_host * HEAD_DIM + 4 * 64
            ws_fd = _workspace(
                "ws_fd", device, f32, _tasks * fd_splits * _fd_stride, zero=True
            )
            # E10：归并统计段跨行向量化。只在归并 D 维已分摊（fd_mrg > 1，
            # 即 #021 这类 tasks=5/cores=28 的小 task 例）且 rows <= VL 时启用；
            # #022/#024/#025/#026/#028 全部保持 0，旧指令流逐条不变。
            # 依据：E10 实测 #028 是 AIC 临界（aic 16.65 > aiv 15.08），归并
            # 尾被 AIC 等待掩盖，向量化收益 <3%；按验收规则还原 base 路径。
            _fd_vec = 1 if (fd_mrg > 1 and fd_anchor == 0) else 0
            _envv = os.environ.get("SMLA_FD_VEC")
            if _envv is not None:
                _fd_vec = int(_envv)
        else:
            fd_mrg = 1
            _fd_vec = 0
            ws_fd = _workspace("ws_fd_dummy", device, f32, 1, zero=True)
        if fd_splits > 1:
            # E13：把发射块数收窄到「unit 数 / 归并项数」的上界（<= 规划核数）。
            # 位置 p < q_bn 时 pos % q_bn == pos，故 plan 表与 legacy 的
            # unit/归并位置映射逐位不变，只减少空核的派发与屏障参与。
            _u_need = _plan["n_units"] if fd_plan else _tasks * fd_splits
            if fd_anchor == 1:
                _m_need = _u_need
            elif fd_item == 1:
                _m_need = int(tiling[0])
            else:
                _m_need = _tasks * fd_mrg
            _need = max(_u_need, _m_need)
            if 0 < _need < block_num_launch:
                block_num_launch = _need
        # ---- E13：needInit（Constexpr）与 task 参数表 ----
        need_init_host = 0
        if ori_mask_mode == 0:
            need_init_host = 1
        else:
            for _b in range(num_seqs):
                if (int(cu_q[_b + 1]) - int(cu_q[_b])) > (
                    int(cu_kv[_b + 1]) - int(cu_kv[_b])
                ):
                    need_init_host = 1
        tp_on = 0
        tp_tab = _i32_cached([0] * 8, device)
        if _fd_ok and _tasks > 0:
            _tp_vals, _tp_ok = _csa_task_param_table(
                [int(v) for v in cu_q],
                [int(v) for v in cu_kv],
                [int(v) for v in cu_cmp],
                _cmp_topk_list,
                cmp_w,
                cmp_mask_mode,
                cmp_ratio,
                _resid_list,
                tok2,
                g_split,
                _tasks,
                int(ori_mask_mode),
                int(ori_win_left),
                int(ori_win_right),
            )
            if _tp_ok:
                tp_on = 1
                tp_tab = _i32_cached(_tp_vals, device)
        tiling = _i32_cached(tiling, device)
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
            _tla_tensor(ws_fd),
            _tla_tensor(cmp_phy),
            _tla_tensor(fd_tab),
            _tla_tensor(tp_tab),
            _tla_tensor(_kv_zero(device, dtype)),
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
                ws_fd,
                cmp_phy,
                fd_tab,
                tp_tab,
                _kv_zero(device, dtype),
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
            tok2,
            fd_splits,
            fd_mrg,
            fd_plan,
            fd_anchor,
            # E9（实验插桩，默认全 1 = 与 base 等价）：成本分解臂开关。
            int(os.environ.get("SMLA_FD_STORE", "1")),
            int(os.environ.get("SMLA_FD_MERGE", "1")),
            int(os.environ.get("SMLA_FD_GUARD", "1")),
            _fd_vec,
            fd_item,
            need_init_host,
            tp_on,
            # 2 = 大工作集多token（Q>=16MiB 且 Q+active_cmp>72MiB）：cm4 + 分槽预取(pre3+early_h0)；
            # 1 = 超长上下文小T（active_cmp>=256MiB）：仅 cm4；0 = L2 驻留工作集（cm1）。
            2
            if (
                int(q2.numel()) * q2.element_size() >= 16 * 1024 * 1024
                and (
                    int(q2.numel()) * q2.element_size()
                    + int(cu_cmp[num_seqs]) * HEAD_DIM * cmp2.element_size()
                )
                > 72 * 1024 * 1024
            )
            else (
                1
                if (
                    int(cu_cmp[num_seqs]) * HEAD_DIM * cmp2.element_size()
                    >= 256 * 1024 * 1024
                )
                else 0
            ),
            # SWIZ_ON: contiguous (1) for uniform tile loads, round-robin (0) for
            # causal-triangle load imbalance (unused when fd_splits>1/g_split!=1/tok2==1).
            _csa_swiz_on_host(
                cu_q,
                cu_kv,
                cu_cmp,
                cmp_w,
                ori_mask_mode,
                ori_win_left,
                cmp_mask_mode,
                cmp_ratio,
                block_num_launch,
            )
            if (fd_splits == 1 and g_split == 1 and tok2 == 0)
            else 1,
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
        # D2/E13：needInit + task 参数表（host 逐 token 精确复算）。
        _ori_topk_list = (
            _sparse_lists["ori"]
            if (q_rows_idx is None and "ori" in _sparse_lists)
            else [int(v) for v in topk.reshape(-1).tolist()]
        )
        _os_plan = _os_token_plan(
            [int(v) for v in cu_q],
            [int(v) for v in cu_kv],
            None,
            _ori_topk_list,
            None,
            ori_w,
            0,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            0,
            1,
            [0] * num_seqs,
            num_seqs,
            False,
        )
        need_init_host = 0 if all(p[1] > 0 for p in _os_plan) else 1
        _os_tasks = run_rows * g_split
        _tp_vals, _tp_ok = _os_task_param_table(_os_plan, g_split, _os_tasks)
        tp_on = 1 if _tp_ok else 0
        tp_tab = _i32_cached(_tp_vals if tp_on else [0] * 8, device)
        # D2/E13：发射块数收窄到 task 数（OS/OCS 无归并；g_split==2 时核与
        # 配对核共享暂存、块号参与配对映射，且其任务数恒 > 核数，不收窄）。
        if g_split == 1 and 0 < _os_tasks < block_num_launch:
            block_num_launch = _os_tasks
        ori_phy = _workspace(
            "ori_phy", device, torch.int64, run_rows * phy_w, zero=True
        )
        tiling = _i32_cached(tiling, device)
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
            _tla_tensor(tp_tab),
            _tla_tensor(_kv_zero(device, dtype)),
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
                tp_tab,
                _kv_zero(device, dtype),
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
            need_init_host,
            tp_on,
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
        # D2/E13：needInit + task 参数表（host 逐 token 精确复算）。
        _ori_topk_list = (
            _sparse_lists["ori"]
            if (q_rows_idx is None and "ori" in _sparse_lists)
            else [int(v) for v in ori_topk.reshape(-1).tolist()]
        )
        _cmp_topk_list = (
            _sparse_lists["cmp"]
            if (q_rows_idx is None and "cmp" in _sparse_lists)
            else [int(v) for v in cmp_topk.reshape(-1).tolist()]
        )
        _resid_list = (
            _sparse_lists["resid"]
            if "resid" in _sparse_lists
            else [int(v) for v in resid.reshape(-1).tolist()]
        )
        _os_plan = _os_token_plan(
            [int(v) for v in cu_q],
            [int(v) for v in cu_kv],
            [int(v) for v in cu_cmp],
            _ori_topk_list,
            _cmp_topk_list,
            ori_w,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            _resid_list,
            num_seqs,
            True,
        )
        need_init_host = 0 if all(p[1] > 0 for p in _os_plan) else 1
        _os_tasks = run_rows * g_split
        _tp_vals, _tp_ok = _os_task_param_table(_os_plan, g_split, _os_tasks)
        tp_on = 1 if _tp_ok else 0
        tp_tab = _i32_cached(_tp_vals if tp_on else [0] * 8, device)
        # D2/E13：发射块数收窄到 task 数（OS/OCS 无归并；g_split==2 时核与
        # 配对核共享暂存、块号参与配对映射，且其任务数恒 > 核数，不收窄）。
        if g_split == 1 and 0 < _os_tasks < block_num_launch:
            block_num_launch = _os_tasks
        ori_phy = _workspace(
            "ori_phy", device, torch.int64, run_rows * ori_phy_w, zero=True
        )
        cmp_phy = _workspace(
            "cmp_phy", device, torch.int64, run_rows * cmp_phy_w, zero=True
        )
        tiling = _i32_cached(tiling, device)
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
            _tla_tensor(tp_tab),
            _tla_tensor(_kv_zero(device, dtype)),
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
                tp_tab,
                _kv_zero(device, dtype),
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
            need_init_host,
            tp_on,
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
        _ZERO_TRACE = None
        return output, lse
    executor(*tensor_args, scale_t, block_num=block_num_launch)
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
    if _fkey is not None and out_pack is None:
        # ragged 回写路径不做计划缓存（输出物理布局需要逐次 scatter）。
        _register_fast_plan(_fkey, executor, tensor_args, (scale_t,), block_num_launch)
    else:
        _ZERO_TRACE = None
    return output, lse
