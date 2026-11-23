"""KDA chunked backward: host entry point, device kernels and the whole pipeline.

Layout of this file, top to bottom:

* host layer (bottom of the file) — ``kda_bwd`` entry point, ``validate_inputs`` and
  the chunk-boundary padding/slicing.
* device code — two Cube kernels for the work that cannot be expressed as batched
  torch ops:

  ``inv_chain_kernel``
      per chunk ``A = (I + L)^-1`` by Neumann doubling, then ``c = A kh``,
      ``u = A v`` and ``A`` itself in one launch (12 dependent mmads).
  ``bistates_kernel``
      the two linear scans of the backward in one launch, one block per walk:
      forward ``S' = M S + B`` and reverse ``dS' = Q dS + P``.

* pipeline — everything else is chunk-parallel, so it is issued as *batched* torch ops
  (one dispatch for all chunks) to keep the host cost at ~30 dispatches instead of
  ``NC`` times that.

The math is the chunked (WY) form of the KDA backward; the sample README carries the
interface, the constraints and the accuracy evidence.
"""

import math
import numbers

import torch
import torch_npu  # noqa: F401
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

C = 64  # chunk length, fixed by the WY structure
_LG_FLOOR = 1e-30  # fp32 floor for exp(cumsum(g)); see the note in chunked_backward
ROUNDS = 5  # Neumann doubling rounds for C = 64 (L is nilpotent at index 64)

_ARTIFACT_CACHE = {}
_WS_CACHE = {}
_WS_CACHE_LIMIT = 8
_MASK_CACHE = {}


# -----------------------------------------------------------------------------------------------------------
# shared helpers
# -----------------------------------------------------------------------------------------------------------
def _core_count(dev, kind="vector"):
    """Number of cores to cap the launch grid at, queried from the running device.

    ``block_num`` is the number of AI Vector (AIV) blocks and ``tla.arch.block_num()`` returns it,
    so the cap must come from the vector core count.  Measured on Ascend 950 (28 cube / 56 vector):
    capping with the cube count still gives correct results but runs 11~22% slower
    (0.917 -> 0.742 ms, 1.051 -> 0.936 ms, 1.735 -> 1.345 ms on three dev shapes), so do not
    "simplify" this to the cube count.
    """
    idx = getattr(dev, "index", None)
    if idx is None:
        idx = torch.npu.current_device()
    props = torch.npu.get_device_properties(int(idx))
    attr = "vector_core_num" if kind == "vector" else "cube_core_num"
    num = int(getattr(props, attr, 0) or 0)
    return max(1, num)


def _wrap(t):
    """Flat, layout-dynamic TLA view of a GM tensor (kernels index by element offset)."""
    return from_dlpack(
        t.reshape(-1), layout_tag=tla.arch.RowMajor, origin_shape=(t.numel(),)
    ).mark_layout_dynamic()


class _Stage:
    """Persistent buffers + cached TLA views for one kernel, keyed by (slot, shape, dtype).

    A buffer belongs to exactly one slot and one direction (input or output), so
    a later call can never alias data a still-running kernel is reading.
    """

    def __init__(self):
        self.bufs = {}
        self.views = {}
        self.inited = set()

    def put(self, slot, shape, src, dtype):
        key = (slot, tuple(shape), dtype)
        buf = self.bufs.get(key)
        if buf is None:
            buf = torch.empty(tuple(shape), dtype=dtype, device=src.device)
            self.bufs[key] = buf
            self.views[key] = _wrap(buf)
        buf.view(tuple(shape)).copy_(src)
        return self.views[key]

    def get(self, slot, shape, dtype, device, rotate=False, ring=2):
        key = (slot, tuple(shape), dtype)
        if rotate:
            idx = self.bufs.get(("rr",) + key, 0)
            self.bufs[("rr",) + key] = (idx + 1) % ring
            key = (f"{slot}~{idx}",) + key[1:]
        buf = self.bufs.get(key)
        if buf is None:
            buf = torch.empty(tuple(shape), dtype=dtype, device=device)
            self.bufs[key] = buf
            self.views[key] = _wrap(buf)
        return buf, self.views[key]

    def once(self, slot, shape, dtype, device, init):
        key = (slot, tuple(shape), dtype)
        if key not in self.inited:
            buf = self.bufs.get(key)
            if buf is None:
                buf = torch.empty(tuple(shape), dtype=dtype, device=device)
                self.bufs[key] = buf
                self.views[key] = _wrap(buf)
            buf.view(tuple(shape)).copy_(init())
            self.inited.add(key)
        return self.views[key]


def _stage(device, batch, hv, nc):
    key = (str(device), batch, hv, nc)
    stage = _WS_CACHE.get(key)
    if stage is None:
        stage = _Stage()
        if len(_WS_CACHE) >= _WS_CACHE_LIMIT:
            _WS_CACHE.clear()
        _WS_CACHE[key] = stage
    return stage


def _masks(dev):
    key = str(dev)
    cached = _MASK_CACHE.get(key)
    if cached is None:
        idx = torch.arange(C, device=dev)
        cached = (
            (idx[None, :] < idx[:, None]).to(torch.float32),  # strict lower
            (idx[None, :] <= idx[:, None]).to(torch.float32),  # lower inclusive
        )
        _MASK_CACHE[key] = cached
    return cached


def _chunk_major(x, batch, length, hv):
    """``(B,T,HV,W)`` -> ``(B,HV,T/C,C,W)`` f32 chunk-major."""
    nc = length // C
    return (
        x.permute(0, 2, 1, 3).reshape(batch, hv, nc, C, x.shape[3]).float().contiguous()
    )


def _unchunk_into(out, x5, batch, length, hv, group=1):
    """``(B,HV,T/C,C,W)`` -> ``out`` in ``(B,T,HV,W)`` layout, GQA-reduced when asked."""
    nc = length // C
    h = hv // group
    view = x5.permute(0, 2, 3, 1, 4).reshape(batch, nc * C, hv, -1)
    if group == 1:
        out.copy_(view)
    else:
        out.copy_(view.reshape(batch, length, h, group, -1).sum(3))


# -----------------------------------------------------------------------------------------------------------
# kernel 1: per-chunk inverse A = (I+L)^-1 plus c = A kh, u = A v
# -----------------------------------------------------------------------------------------------------------
@tla.kernel
def inv_chain_kernel(
    gm_l: tla.Tensor,  # (NT, C, C) bf16  strictly-lower L
    gm_y0: tla.Tensor,  # (NT, C, C) bf16  I - L
    gm_i: tla.Tensor,  # (C, C)     bf16  identity
    gm_kh: tla.Tensor,  # (NT, C, K) bf16
    gm_v: tla.Tensor,  # (NT, C, V) bf16
    gm_c: tla.Tensor,  # (NT, C, K) f32
    gm_u: tla.Tensor,  # (NT, C, V) f32
    gm_a: tla.Tensor,  # (NT, C, C) bf16  A
    NT: tla.Int32,
) -> None:
    # One mutex per physical buffer; ownership alternates writer -> reader(s) -> writer.
    m_l1_l = tla.mutex("l1_l", id=0)
    m_l1_y = tla.mutex("l1_y", id=1)
    m_l1_p = tla.mutex("l1_p", id=2)
    m_l1_i = tla.mutex("l1_i", id=3)
    m_l1_kh = tla.mutex("l1_kh", id=4)
    m_l1_v = tla.mutex("l1_v", id=5)
    m_l0a = tla.mutex("l0a", id=6)
    m_l0b = tla.mutex("l0b", id=7)
    m_l0c_y = tla.mutex("l0c_y", id=8)
    m_l0c_p = tla.mutex("l0c_p", id=9)
    m_l0c_c = tla.mutex("l0c_c", id=10)
    m_l0c_u = tla.mutex("l0c_u", id=11)

    l1_l = tla.allocate(C * C, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_y = tla.allocate(C * C, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_p = tla.allocate(C * C, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_i = tla.allocate(C * C, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_kh = tla.allocate(C * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_v = tla.allocate(C * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(C * 128, tla.BFloat16, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(C * 128, tla.BFloat16, tla.AddressSpace.l0b, 512)
    l0_yy_ptr = tla.allocate(C * C, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0_pp_ptr = tla.allocate(C * C, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0_cu_ptr = tla.allocate(C * 128, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0_cu2_ptr = tla.allocate(C * 128, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), NT, tla.arch.block_num()):

            def gm_view(mem, rows, cols, off):
                return tla.make_tensor(
                    mem.ptr + off,
                    tla.make_layout(
                        tla.make_shape(rows, cols), tla.make_stride(cols, 1)
                    ),
                )

            l_gm = gm_view(gm_l, C, C, task * (C * C))
            y_gm = gm_view(gm_y0, C, C, task * (C * C))
            i_gm = gm_view(gm_i, C, C, 0)
            kh_gm = gm_view(gm_kh, C, 128, task * (C * 128))
            v_gm = gm_view(gm_v, C, 128, task * (C * 128))
            c_gm = gm_view(gm_c, C, 128, task * (C * 128))
            u_gm = gm_view(gm_u, C, 128, task * (C * 128))
            a_gm = gm_view(gm_a, C, C, task * (C * C))

            l_l1 = tla.make_tensor_like(l1_l, l_gm)
            y_l1 = tla.make_tensor_like(l1_y, y_gm)
            p_l1 = tla.make_tensor_like(l1_p, l_gm)
            i_l1 = tla.make_tensor_like(l1_i, i_gm)
            kh_l1 = tla.make_tensor_like(l1_kh, kh_gm)
            v_l1 = tla.make_tensor_like(l1_v, v_gm)

            yy_l0c = tla.make_tensor_like(l0_yy_ptr, y_gm, tla.arch.L0Clayout)
            pp_l0c = tla.make_tensor_like(l0_pp_ptr, l_gm, tla.arch.L0Clayout)
            c_l0c = tla.make_tensor_like(l0_cu_ptr, c_gm, tla.arch.L0Clayout)
            u_l0c = tla.make_tensor_like(l0_cu2_ptr, u_gm, tla.arch.L0Clayout)

            # ---- GM -> L1: MTE2 owns each tile until its first reader ----
            m_l1_l.lock(pipe=tla.arch.MTE2)
            tla.copy(l_l1, l_gm)
            m_l1_l.unlock(pipe=tla.arch.MTE2)
            m_l1_y.lock(pipe=tla.arch.MTE2)
            tla.copy(y_l1, y_gm)
            m_l1_y.unlock(pipe=tla.arch.MTE2)
            m_l1_i.lock(pipe=tla.arch.MTE2)
            tla.copy(i_l1, i_gm)
            m_l1_i.unlock(pipe=tla.arch.MTE2)
            m_l1_kh.lock(pipe=tla.arch.MTE2)
            tla.copy(kh_l1, kh_gm)
            m_l1_kh.unlock(pipe=tla.arch.MTE2)
            m_l1_v.lock(pipe=tla.arch.MTE2)
            tla.copy(v_l1, v_gm)
            m_l1_v.unlock(pipe=tla.arch.MTE2)

            # ---- P0 = L L ----
            la = tla.make_tensor_like(l0a, l_l1, tla.arch.zN)
            lb = tla.make_tensor_like(l0b, l_l1, tla.arch.nZ)
            m_l0a.lock(pipe=tla.arch.MTE1)
            m_l1_l.lock(pipe=tla.arch.MTE1)
            tla.copy(la, l_l1)
            m_l1_l.unlock(pipe=tla.arch.MTE1)
            m_l0a.unlock(pipe=tla.arch.MTE1)
            m_l0b.lock(pipe=tla.arch.MTE1)
            m_l1_l.lock(pipe=tla.arch.MTE1)
            tla.copy(lb, l_l1)
            m_l1_l.unlock(pipe=tla.arch.MTE1)
            m_l0b.unlock(pipe=tla.arch.MTE1)
            m_l0a.lock(pipe=tla.arch.CUBE)
            m_l0b.lock(pipe=tla.arch.CUBE)
            m_l0c_p.lock(pipe=tla.arch.CUBE)
            tla.mmad(pp_l0c, la, lb, init_c=True, unit_flag=0b11)
            m_l0c_p.unlock(pipe=tla.arch.CUBE)
            m_l0b.unlock(pipe=tla.arch.CUBE)
            m_l0a.unlock(pipe=tla.arch.CUBE)
            m_l0c_p.lock(pipe=tla.arch.FIX)
            m_l1_p.lock(pipe=tla.arch.FIX)
            tla.copy(p_l1, pp_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
            m_l1_p.unlock(pipe=tla.arch.FIX)
            m_l0c_p.unlock(pipe=tla.arch.FIX)

            # ---- Y <- Y (I + P), P <- P P, five doubling rounds ----
            for _round in tla.range(ROUNDS):
                la = tla.make_tensor_like(l0a, y_l1, tla.arch.zN)
                lb = tla.make_tensor_like(l0b, i_l1, tla.arch.nZ)
                m_l0a.lock(pipe=tla.arch.MTE1)
                m_l1_y.lock(pipe=tla.arch.MTE1)
                tla.copy(la, y_l1)
                m_l1_y.unlock(pipe=tla.arch.MTE1)
                m_l0a.unlock(pipe=tla.arch.MTE1)
                m_l0b.lock(pipe=tla.arch.MTE1)
                m_l1_i.lock(pipe=tla.arch.MTE1)
                tla.copy(lb, i_l1)
                m_l1_i.unlock(pipe=tla.arch.MTE1)
                m_l0b.unlock(pipe=tla.arch.MTE1)
                m_l0a.lock(pipe=tla.arch.CUBE)
                m_l0b.lock(pipe=tla.arch.CUBE)
                m_l0c_y.lock(pipe=tla.arch.CUBE)
                tla.mmad(yy_l0c, la, lb, init_c=True, unit_flag=0b10)
                m_l0b.unlock(pipe=tla.arch.CUBE)
                # accumulate += Y @ P; m_l0a stays held so la cannot be refilled mid-pair
                lb2 = tla.make_tensor_like(l0b, p_l1, tla.arch.nZ)
                m_l0b.lock(pipe=tla.arch.MTE1)
                m_l1_p.lock(pipe=tla.arch.MTE1)
                tla.copy(lb2, p_l1)
                m_l1_p.unlock(pipe=tla.arch.MTE1)
                m_l0b.unlock(pipe=tla.arch.MTE1)
                m_l0b.lock(pipe=tla.arch.CUBE)
                tla.mmad(yy_l0c, la, lb2, init_c=False, unit_flag=0b11)
                m_l0b.unlock(pipe=tla.arch.CUBE)
                m_l0c_y.unlock(pipe=tla.arch.CUBE)
                m_l0a.unlock(pipe=tla.arch.CUBE)

                # ---- P <- P P ----
                la = tla.make_tensor_like(l0a, p_l1, tla.arch.zN)
                lb = tla.make_tensor_like(l0b, p_l1, tla.arch.nZ)
                m_l0a.lock(pipe=tla.arch.MTE1)
                m_l1_p.lock(pipe=tla.arch.MTE1)
                tla.copy(la, p_l1)
                m_l1_p.unlock(pipe=tla.arch.MTE1)
                m_l0a.unlock(pipe=tla.arch.MTE1)
                m_l0b.lock(pipe=tla.arch.MTE1)
                m_l1_p.lock(pipe=tla.arch.MTE1)
                tla.copy(lb, p_l1)
                m_l1_p.unlock(pipe=tla.arch.MTE1)
                m_l0b.unlock(pipe=tla.arch.MTE1)
                m_l0a.lock(pipe=tla.arch.CUBE)
                m_l0b.lock(pipe=tla.arch.CUBE)
                m_l0c_p.lock(pipe=tla.arch.CUBE)
                tla.mmad(pp_l0c, la, lb, init_c=True, unit_flag=0b11)
                m_l0c_p.unlock(pipe=tla.arch.CUBE)
                m_l0b.unlock(pipe=tla.arch.CUBE)
                m_l0a.unlock(pipe=tla.arch.CUBE)

                # FIX publishes Y and P back into L1 for the next round.
                m_l0c_y.lock(pipe=tla.arch.FIX)
                m_l1_y.lock(pipe=tla.arch.FIX)
                tla.copy(y_l1, yy_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
                m_l1_y.unlock(pipe=tla.arch.FIX)
                m_l0c_y.unlock(pipe=tla.arch.FIX)
                m_l0c_p.lock(pipe=tla.arch.FIX)
                m_l1_p.lock(pipe=tla.arch.FIX)
                tla.copy(p_l1, pp_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
                m_l1_p.unlock(pipe=tla.arch.FIX)
                m_l0c_p.unlock(pipe=tla.arch.FIX)

            # ---- c = Y kh , u = Y v ----
            la = tla.make_tensor_like(l0a, y_l1, tla.arch.zN)
            lb = tla.make_tensor_like(l0b, kh_l1, tla.arch.nZ)
            m_l0a.lock(pipe=tla.arch.MTE1)
            m_l1_y.lock(pipe=tla.arch.MTE1)
            tla.copy(la, y_l1)
            m_l1_y.unlock(pipe=tla.arch.MTE1)
            m_l0a.unlock(pipe=tla.arch.MTE1)
            m_l0b.lock(pipe=tla.arch.MTE1)
            m_l1_kh.lock(pipe=tla.arch.MTE1)
            tla.copy(lb, kh_l1)
            m_l1_kh.unlock(pipe=tla.arch.MTE1)
            m_l0b.unlock(pipe=tla.arch.MTE1)
            m_l0a.lock(pipe=tla.arch.CUBE)
            m_l0b.lock(pipe=tla.arch.CUBE)
            m_l0c_c.lock(pipe=tla.arch.CUBE)
            tla.mmad(c_l0c, la, lb, init_c=True, unit_flag=0b11)
            m_l0c_c.unlock(pipe=tla.arch.CUBE)
            m_l0b.unlock(pipe=tla.arch.CUBE)
            m_l0a.unlock(pipe=tla.arch.CUBE)
            m_l0c_c.lock(pipe=tla.arch.FIX)
            tla.copy(c_gm, c_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
            m_l0c_c.unlock(pipe=tla.arch.FIX)

            la = tla.make_tensor_like(l0a, y_l1, tla.arch.zN)
            lb = tla.make_tensor_like(l0b, v_l1, tla.arch.nZ)
            m_l0a.lock(pipe=tla.arch.MTE1)
            m_l1_y.lock(pipe=tla.arch.MTE1)
            tla.copy(la, y_l1)
            m_l1_y.unlock(pipe=tla.arch.MTE1)
            m_l0a.unlock(pipe=tla.arch.MTE1)
            m_l0b.lock(pipe=tla.arch.MTE1)
            m_l1_v.lock(pipe=tla.arch.MTE1)
            tla.copy(lb, v_l1)
            m_l1_v.unlock(pipe=tla.arch.MTE1)
            m_l0b.unlock(pipe=tla.arch.MTE1)
            m_l0a.lock(pipe=tla.arch.CUBE)
            m_l0b.lock(pipe=tla.arch.CUBE)
            m_l0c_u.lock(pipe=tla.arch.CUBE)
            tla.mmad(u_l0c, la, lb, init_c=True, unit_flag=0b11)
            m_l0c_u.unlock(pipe=tla.arch.CUBE)
            m_l0b.unlock(pipe=tla.arch.CUBE)
            m_l0a.unlock(pipe=tla.arch.CUBE)
            m_l0c_u.lock(pipe=tla.arch.FIX)
            tla.copy(u_gm, u_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
            m_l0c_u.unlock(pipe=tla.arch.FIX)

            # A itself for the backward's `A^T du`; L1->GM is not a copy route and a second
            # fixpipe off yy_l0c hangs the core, so round it through the (dead) P tile.
            la = tla.make_tensor_like(l0a, y_l1, tla.arch.zN)
            lb = tla.make_tensor_like(l0b, i_l1, tla.arch.nZ)
            m_l0a.lock(pipe=tla.arch.MTE1)
            m_l1_y.lock(pipe=tla.arch.MTE1)
            tla.copy(la, y_l1)
            m_l1_y.unlock(pipe=tla.arch.MTE1)
            m_l0a.unlock(pipe=tla.arch.MTE1)
            m_l0b.lock(pipe=tla.arch.MTE1)
            m_l1_i.lock(pipe=tla.arch.MTE1)
            tla.copy(lb, i_l1)
            m_l1_i.unlock(pipe=tla.arch.MTE1)
            m_l0b.unlock(pipe=tla.arch.MTE1)
            m_l0a.lock(pipe=tla.arch.CUBE)
            m_l0b.lock(pipe=tla.arch.CUBE)
            m_l0c_p.lock(pipe=tla.arch.CUBE)
            tla.mmad(pp_l0c, la, lb, init_c=True, unit_flag=0b11)
            m_l0c_p.unlock(pipe=tla.arch.CUBE)
            m_l0b.unlock(pipe=tla.arch.CUBE)
            m_l0a.unlock(pipe=tla.arch.CUBE)
            m_l0c_p.lock(pipe=tla.arch.FIX)
            tla.copy(a_gm, pp_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
            m_l0c_p.unlock(pipe=tla.arch.FIX)

        tla.pipe_barrier(tla.pipes.ALL)


def _inv_chain(stage, l_m, kh, v, batch, hv, nc):
    num = batch * hv * nc
    v_l = stage.put("l", (num, C, C), l_m.reshape(num, C, C), torch.bfloat16)
    eye = torch.eye(C, dtype=torch.float32, device=l_m.device)
    v_y0 = stage.put("y0", (num, C, C), eye - l_m.reshape(num, C, C), torch.bfloat16)
    v_i = stage.once("i64", (C, C), torch.bfloat16, l_m.device, eye.bfloat16)
    v_kh = stage.put("kh", (num, C, 128), kh.reshape(num, C, 128), torch.bfloat16)
    v_v = stage.put("v", (num, C, 128), v.reshape(num, C, 128), torch.bfloat16)
    c, v_c = stage.get("c", (num, C, 128), torch.float32, l_m.device, rotate=True)
    u, v_u = stage.get("u", (num, C, 128), torch.float32, l_m.device, rotate=True)
    a, v_a = stage.get("a", (num, C, C), torch.bfloat16, l_m.device, rotate=True)
    args = (tla.Int32(num),)
    w = [v_l, v_y0, v_i, v_kh, v_v, v_c, v_u, v_a]
    artifact = _ARTIFACT_CACHE.get(("inv", str(l_m.device), num))
    if artifact is None:
        artifact = tla.compile(inv_chain_kernel, *w, *args, options="--npu-arch 3510")
        _ARTIFACT_CACHE[("inv", str(l_m.device), num)] = artifact
    artifact(*w, *args, block_num=max(1, min(num, _core_count(l_m.device))))
    return (
        c.view(batch, hv, nc, C, 128),
        u.view(batch, hv, nc, C, 128),
        a.view(batch, hv, nc, C, C),
    )


# -----------------------------------------------------------------------------------------------------------
# kernel 2: both scans (forward S and reverse dS) in one launch, one block per walk
# -----------------------------------------------------------------------------------------------------------
@tla.kernel
def bistates_kernel(
    gm_m: tla.Tensor,  # (2*NB*NC, K, K)  [forward M | reverse Q]  bf16
    gm_b: tla.Tensor,  # (2*NB*NC, K, V)  [forward B | reverse P]  bf16
    gm_i: tla.Tensor,  # (V, V)           identity                bf16
    gm_s: tla.Tensor,  # (2*NB*(NC+1), K, V)  [forward slots | reverse slots]
    NC: tla.Int32,
    NB: tla.Int32,
) -> None:
    # One mutex per physical buffer; ownership alternates writer -> reader(s) -> writer.
    # m_state guards the GM state slot: FIX writes it, MTE2 reads it on the next step.
    m_l1_i = tla.mutex("l1_i_st", id=0)
    m_l1_m = tla.mutex("l1_m_st", id=1)
    m_l1_b = tla.mutex("l1_b_st", id=2)
    m_l1_s = tla.mutex("l1_s_st", id=3)
    m_l0a = tla.mutex("l0a_st", id=4)
    m_l0b_i = tla.mutex("l0b_i_st", id=5)
    m_l0b_s = tla.mutex("l0b_s_st", id=6)
    m_l0c_s = tla.mutex("l0c_s_st", id=7)
    m_state = tla.mutex("state_slot", id=8)

    l1_m = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_b = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_s = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l1_i = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l0a, 512)
    l0b_s = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l0b, 512)
    l0b_i = tla.allocate(128 * 128, tla.BFloat16, tla.AddressSpace.l0b, 512)
    l0c_s = tla.allocate(128 * 128, tla.Float32, tla.AddressSpace.l0c, 1024)
    s_ref = tla.make_tensor(
        gm_s.ptr, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1))
    )

    with tla.cube():
        # the first state of each walk is the zeroed slot NC-1: mark it as published
        m_state.lock(pipe=tla.arch.FIX)
        m_state.unlock(pipe=tla.arch.FIX)
        for wb in tla.range(tla.arch.block_idx(), NB + NB, tla.arch.block_num()):
            wid = wb - (wb // 2) * 2  # 0 = forward, 1 = reverse
            bh = wb // 2
            dirn = 1 - wid - wid  # +1 / -1
            i_gm = tla.make_tensor(
                gm_i.ptr,
                tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
            )
            i_l1 = tla.make_tensor_like(l1_i, i_gm)

            # identity tile for the "+ B" mmad: resident in L0B for the whole walk
            m_l1_i.lock(pipe=tla.arch.MTE2)
            tla.copy(i_l1, i_gm)
            m_l1_i.unlock(pipe=tla.arch.MTE2)
            m_l1_i.lock(pipe=tla.arch.MTE1)
            i_l0 = tla.make_tensor_like(l0b_i, i_l1, tla.arch.nZ)
            m_l0b_i.lock(pipe=tla.arch.MTE1)
            tla.copy(i_l0, i_l1)
            m_l0b_i.unlock(pipe=tla.arch.MTE1)
            m_l1_i.unlock(pipe=tla.arch.MTE1)

            # both walks need only NC-1 steps: the forward's last step would write
            # the state *after* the last chunk (nobody reads it) and the reverse
            # starts from the zeroed slot NC-1.
            for step in tla.range(NC - 1):
                ci = wid * (NC - 1) + (1 - wid - wid) * step
                task = wid * (NB * NC) + bh * NC + ci
                slot = wid * (NB * (NC + 1)) + bh * (NC + 1) + ci
                s_gm = tla.make_tensor(
                    gm_s.ptr + slot * (128 * 128),
                    tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                )
                sa_gm = tla.make_tensor(
                    gm_s.ptr + (slot + dirn) * (128 * 128),
                    tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                )
                m_gm = tla.make_tensor(
                    gm_m.ptr + task * (128 * 128),
                    tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                )
                b_gm = tla.make_tensor(
                    gm_b.ptr + task * (128 * 128),
                    tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                )
                m_l1 = tla.make_tensor_like(l1_m, m_gm)
                b_l1 = tla.make_tensor_like(l1_b, b_gm)
                s_l1 = tla.make_tensor_like(l1_s, s_gm)
                ma = tla.make_tensor_like(l0a, m_l1, tla.arch.zN)
                ba = tla.make_tensor_like(l0a, b_l1, tla.arch.zN)
                s_l0 = tla.make_tensor_like(l0b_s, s_l1, tla.arch.nZ)
                s_l0c = tla.make_tensor_like(l0c_s, s_ref, tla.arch.L0Clayout)

                # ---- MTE2 loads this step's tiles; m_state orders the state slot ----
                m_state.lock(pipe=tla.arch.MTE2)
                m_l1_s.lock(pipe=tla.arch.MTE2)
                tla.copy(s_l1, s_gm)
                m_l1_s.unlock(pipe=tla.arch.MTE2)
                m_l1_m.lock(pipe=tla.arch.MTE2)
                tla.copy(m_l1, m_gm)
                m_l1_m.unlock(pipe=tla.arch.MTE2)
                m_l1_b.lock(pipe=tla.arch.MTE2)
                tla.copy(b_l1, b_gm)
                m_l1_b.unlock(pipe=tla.arch.MTE2)
                m_state.unlock(pipe=tla.arch.MTE2)

                # ---- S' = M S ----
                m_l0a.lock(pipe=tla.arch.MTE1)
                m_l1_m.lock(pipe=tla.arch.MTE1)
                tla.copy(ma, m_l1)
                m_l1_m.unlock(pipe=tla.arch.MTE1)
                m_l0a.unlock(pipe=tla.arch.MTE1)
                m_l0b_s.lock(pipe=tla.arch.MTE1)
                m_l1_s.lock(pipe=tla.arch.MTE1)
                tla.copy(s_l0, s_l1)
                m_l1_s.unlock(pipe=tla.arch.MTE1)
                m_l0b_s.unlock(pipe=tla.arch.MTE1)
                m_l0a.lock(pipe=tla.arch.CUBE)
                m_l0b_s.lock(pipe=tla.arch.CUBE)
                m_l0c_s.lock(pipe=tla.arch.CUBE)
                tla.mmad(s_l0c, ma, s_l0, init_c=True, unit_flag=0b10)
                m_l0b_s.unlock(pipe=tla.arch.CUBE)
                m_l0a.unlock(pipe=tla.arch.CUBE)

                # ---- S' += B @ I  (m_l0c_s stays held across the accumulate pair) ----
                m_l0a.lock(pipe=tla.arch.MTE1)
                m_l1_b.lock(pipe=tla.arch.MTE1)
                tla.copy(ba, b_l1)
                m_l1_b.unlock(pipe=tla.arch.MTE1)
                m_l0a.unlock(pipe=tla.arch.MTE1)
                m_l0a.lock(pipe=tla.arch.CUBE)
                m_l0b_i.lock(pipe=tla.arch.CUBE)
                tla.mmad(s_l0c, ba, i_l0, init_c=False, unit_flag=0b11)
                m_l0b_i.unlock(pipe=tla.arch.CUBE)
                m_l0a.unlock(pipe=tla.arch.CUBE)
                m_l0c_s.unlock(pipe=tla.arch.CUBE)

                # ---- FIX writes the next state and hands the slot to MTE2 ----
                m_l0c_s.lock(pipe=tla.arch.FIX)
                m_state.lock(pipe=tla.arch.FIX)
                tla.copy(sa_gm, s_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
                m_state.unlock(pipe=tla.arch.FIX)
                m_l0c_s.unlock(pipe=tla.arch.FIX)

        tla.pipe_barrier(tla.pipes.ALL)


def _bistates(stage, m_f, b_f, q_m, p_k, batch, hv, nc):
    nb = batch * hv
    dev = m_f.device
    # staged as [forward | reverse] so one kernel body can serve both walks
    mm, v_m = stage.get("mm", (2 * nb * nc, 128 * 128), torch.bfloat16, dev)
    mm[: nb * nc].copy_(m_f.reshape(nb * nc, 128 * 128))
    mm[nb * nc :].copy_(q_m.reshape(nb * nc, 128 * 128))
    bb, v_b = stage.get("bb", (2 * nb * nc, 128 * 128), torch.bfloat16, dev)
    bb[: nb * nc].copy_(b_f.reshape(nb * nc, 128 * 128))
    bb[nb * nc :].copy_(p_k.reshape(nb * nc, 128 * 128))
    v_i = stage.once(
        "i128",
        (128, 128),
        torch.bfloat16,
        dev,
        lambda: torch.eye(128, dtype=torch.bfloat16, device=dev),
    )
    ss, v_s = stage.get("ss", (2 * nb * (nc + 1), 128 * 128), torch.bfloat16, dev)
    ss3 = ss.view(2, nb, nc + 1, 128 * 128)
    # forward starts from slot 0 = 0; reverse from slot NC-1 = 0 (dS_out(last))
    ss3[0, :, 0].zero_()
    ss3[1, :, nc - 1].zero_()
    w = [v_m, v_b, v_i, v_s]
    args = (tla.Int32(nc), tla.Int32(nb))
    artifact = _ARTIFACT_CACHE.get(("st", str(dev), nb, nc))
    if artifact is None:
        artifact = tla.compile(bistates_kernel, *w, *args, options="--npu-arch 3510")
        _ARTIFACT_CACHE[("st", str(dev), nb, nc)] = artifact
    artifact(*w, *args, block_num=max(1, min(2 * nb, _core_count(dev))))
    return ss3[0, :, :nc], ss3[1, :, :nc]


# -----------------------------------------------------------------------------------------------------------
# the pipeline (chunked WY backward)
# -----------------------------------------------------------------------------------------------------------
def chunked_backward(q, k, v, g, beta, do, scale, outs):
    """Run the whole backward and write ``dq, dk, dv, dg, dbeta`` into ``outs``."""
    batch, length, heads, dim = q.shape
    hv = g.shape[2]
    group = hv // heads
    nc = length // C
    dev = q.device
    sl, tril = _masks(dev)
    eye = torch.eye(dim, dtype=torch.float32, device=dev)
    stage = _stage(dev, batch, hv, nc)

    # ---- chunk-parallel elementwise pieces (no kernel needed, see README) ----
    # ``exp(cumsum(g))`` underflows to 0 once a chunk's accumulated decay passes
    # about -104 in fp32; ``k / lg`` would then be inf and ``Eb = lg[-1]`` times
    # ``kt`` would be 0 * inf = NaN.  A floor keeps every factor inside the
    # fp32/bf16 range and is a no-op for ordinary gates (``lg5`` stays ~1).
    lg5 = (
        g.float()
        .reshape(batch, nc, C, hv, dim)
        .cumsum(2)
        .exp()
        .clamp_min(_LG_FLOOR)
        .permute(0, 3, 1, 2, 4)
    )
    beta5 = beta.float().reshape(batch, nc, C, hv).permute(0, 3, 1, 2)
    Eb5 = lg5[:, :, :, -1, :]
    q5 = _chunk_major(
        q if group == 1 else q.repeat_interleave(group, 2), batch, length, hv
    )
    k5 = _chunk_major(
        k if group == 1 else k.repeat_interleave(group, 2), batch, length, hv
    )
    do5 = _chunk_major(do, batch, length, hv)
    qg5 = q5 * lg5
    kh5 = k5 * lg5
    kt5 = beta5[:, :, :, :, None] * (k5 / lg5)
    w5 = Eb5[:, :, :, None, :] * kt5
    qgs5 = qg5 * scale
    intra5 = torch.einsum("bhntk,bhnsk->bhnts", qgs5, kt5) * tril

    # ---- pieces: L, then the per-chunk inverse and c/u/A ----
    l_m = torch.einsum("bhntk,bhnsk->bhnts", kh5, kt5) * sl
    vc = _chunk_major(v, batch, length, hv)
    c5, u5, a5 = _inv_chain(stage, l_m, kh5, vc, batch, hv, nc)
    a5f = a5.float()

    # ---- both scans: S' = M S + B (forward), dS' = Q dS + P (reverse) ----
    wc = torch.einsum("bhntk,bhntj->bhnkj", w5, c5)
    m_f = Eb5[:, :, :, None, :] * eye - wc
    # Q = Eb*I - c^T w is the transpose of M's second term
    q_m = Eb5[:, :, :, None, :] * eye - wc.transpose(-1, -2)
    b_f = torch.einsum("bhntk,bhntv->bhnkv", w5, u5)
    itd = torch.einsum("bhnts,bhntv->bhnsv", intra5, do5)
    p_k = torch.einsum("bhntk,bhntv->bhnkv", qgs5, do5) - torch.einsum(
        "bhntk,bhntv->bhnkv", c5, itd
    )
    s5, ds5 = _bistates(stage, m_f, b_f, q_m, p_k, batch, hv, nc)
    s5 = s5.reshape(batch, hv, nc, dim, dim)
    ds5 = ds5.reshape(batch, hv, nc, dim, dim)
    s5f = s5.float()
    ds5f = ds5.float()

    # ---- parallel part B ----
    x5 = u5 - torch.einsum("bhntk,bhnkv->bhntv", c5, s5f)
    dx = itd + torch.einsum("bhntk,bhnkv->bhntv", w5, ds5f)
    dv5 = torch.einsum("bhnts,bhntv->bhnsv", a5f, dx)  # A^T du
    dkh = -torch.einsum("bhntv,bhnkv->bhntk", dv5, s5f)  # -(A^T du) S^T
    d_l = -torch.einsum("bhntv,bhnsv->bhnts", dv5, x5) * sl  # -dv x^T
    dkh = dkh + torch.einsum("bhnts,bhnsk->bhntk", d_l, kt5)
    dkg = torch.einsum("bhnts,bhntk->bhnsk", d_l, kh5)
    dintra = torch.einsum("bhntv,bhnsv->bhnts", do5, x5) * tril
    dqg = torch.einsum(
        "bhntw,bhnwk->bhntk",
        torch.cat([do5, dintra], -1),
        torch.cat([s5f.transpose(-1, -2), kt5], -2),
    )
    dkg = dkg + torch.einsum("bhnts,bhntk->bhnsk", dintra, qgs5)
    dw = torch.einsum("bhnkv,bhntv->bhntk", ds5f, x5)
    dkg = dkg + Eb5[:, :, :, None, :] * dw
    de_b = (ds5f * s5f).sum(-1)
    de_b.add_((dw * kt5).sum(3))

    # ---- elementwise tail: dq / dk / dbeta / dg ----
    m = dqg * qgs5
    m.addcmul_(dkh, kh5)
    m.addcmul_(dkg, kt5, value=-1.0)
    m[:, :, :, -1, :].add_(de_b * Eb5)
    dg5 = m.flip(3).cumsum(3).flip(3)
    dq5 = dqg.mul(lg5).mul_(scale)
    dk5 = dkh.mul(lg5).addcmul_(dkg, beta5[:, :, :, :, None] / lg5)
    # dbeta = sum_k (k / lg) dkg.  `kt5 / beta5` would be the same value for
    # beta != 0 but is 0 / 0 = NaN for a zero beta (legal input, and what the tail
    # padding produces internally), so use the chain factor directly.
    dbeta5 = (k5 / lg5).mul(dkg).sum(-1)

    dq, dk, dv, dg, dbeta = outs
    _unchunk_into(dq, dq5, batch, length, hv, group)
    _unchunk_into(dk, dk5, batch, length, hv, group)
    _unchunk_into(dv, dv5, batch, length, hv)
    _unchunk_into(dg, dg5, batch, length, hv)
    dbeta.copy_(dbeta5.permute(0, 2, 3, 1).reshape(batch, length, hv))


# -----------------------------------------------------------------------------------------------------------
# host layer: input checks, chunk-boundary padding, and the public entry point
# -----------------------------------------------------------------------------------------------------------
def _check_tensor(name, value, dtypes):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in dtypes:
        raise TypeError(
            f"{name} must be one of {[str(d).replace('torch.', '') for d in dtypes]}"
        )


def validate_inputs(q, k, v, g, beta, do, scale):
    for name, value, dtypes in (
        ("q", q, (torch.bfloat16,)),
        ("k", k, (torch.bfloat16,)),
        ("v", v, (torch.bfloat16,)),
        ("do", do, (torch.bfloat16,)),
        ("g", g, (torch.bfloat16, torch.float32)),
        ("beta", beta, (torch.bfloat16, torch.float32)),
    ):
        _check_tensor(name, value, dtypes)
    if any(x.ndim != 4 for x in (q, k, v, g, do)):
        raise ValueError("q, k, v, g, do must be rank 4 in BSND layout")
    if beta.ndim != 3:
        raise ValueError("beta must be rank 3 in BSND layout")
    batch, length, heads, dim = q.shape
    hv = g.shape[2]
    if dim != 128 or v.shape[3] != 128:
        raise ValueError("K = V = 128 is required")
    if k.shape != q.shape:
        raise ValueError("q and k must have the same shape")
    if v.shape[:3] != (batch, length, hv) or do.shape != v.shape:
        raise ValueError("v/do must be [B, T, HV, V] with the same B, T, HV as g")
    if g.shape != (batch, length, hv, 128):
        raise ValueError("g must be [B, T, HV, K]")
    if beta.shape != (batch, length, hv):
        raise ValueError("beta must be [B, T, HV]")
    if batch < 1 or length < 1 or heads < 1 or hv < 1:
        raise ValueError("B, T, H and HV must be positive")
    if hv % heads or hv // heads not in (1, 2, 4):
        raise ValueError("HV / H must be 1, 2 or 4")
    if any(x.device.type != "npu" for x in (q, k, v, g, beta, do)):
        raise ValueError("all tensors must be on an NPU device")
    if any(x.device != q.device for x in (k, v, g, beta, do)):
        raise ValueError("all tensors must be on the same NPU device")
    if any(not x.is_contiguous() for x in (q, k, v, g, beta, do)):
        raise ValueError("all tensors must be contiguous")
    if scale is None:
        scale = 128**-0.5
    if (
        isinstance(scale, bool)
        or not isinstance(scale, numbers.Real)
        or not math.isfinite(scale)
    ):
        raise ValueError("scale must be a finite real scalar or None")
    return float(scale)


def _pad_tail(x, length):
    """Pad the token axis up to the next chunk boundary with zeros.

    ``g = 0`` makes ``lg = 1`` and ``beta = 0`` makes ``kg = 0``, so the padded
    chunk is the identity map of the state recursion (``S_out = Eb S_in + wᵀx``
    with ``Eb = 1``, ``w = 0``) and, because the recurrence is causal, it cannot
    change any real token's gradient.  Only the padded rows of the five outputs
    are meaningless and they are sliced away.
    """
    padded = ((length + C - 1) // C) * C
    if padded == length:
        return x
    out = torch.zeros(
        (x.shape[0], padded) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device
    )
    out[:, :length].copy_(x)
    return out


def kda_bwd(q, k, v, g, beta, do, scale=None):
    """KDA chunked backward in BSND layout.

    Args:
        q: ``[B, T, H, 128]`` BF16 queries.
        k: ``[B, T, H, 128]`` BF16 keys.
        v: ``[B, T, HV, 128]`` BF16 values.
        do: ``[B, T, HV, 128]`` BF16 output gradients.
        g: ``[B, T, HV, 128]`` already-activated per-K-channel log decay
            (BF16 or FP32).
        beta: ``[B, T, HV]`` BF16 or FP32.
        scale: query scale, ``None`` means ``128 ** -0.5``.

    Returns:
        ``(dq, dk, dv, dg, dbeta)`` in FP32, with ``dq``/``dk`` reduced over the
        GQA group (``[B, T, H, 128]``) and ``dv``/``dg``/``dbeta`` per value head.

    """
    scale = validate_inputs(q, k, v, g, beta, do, scale)
    batch, length, heads, _ = q.shape
    hv = g.shape[2]
    # Explicit 64-bit element offsets are used in the device code; reject sizes
    # whose byte count would not be addressable.
    for x in (q, k, v, g, beta, do):
        if x.numel() > ((1 << 63) - 1) // max(x.element_size(), 1):
            raise OverflowError("input byte count exceeds signed 64-bit addressing")

    padded = ((length + C - 1) // C) * C
    tail = padded != length
    inputs = (
        [_pad_tail(x, length) for x in (q, k, v, g, beta, do)]
        if tail
        else [q, k, v, g, beta, do]
    )
    tlen = padded if tail else length
    shapes = ((heads, 128), (heads, 128), (hv, 128), (hv, 128), (hv,))
    outs = tuple(
        torch.empty((batch, tlen) + shape, dtype=torch.float32, device=q.device)
        for shape in shapes
    )
    chunked_backward(*inputs, scale, outs)
    if not tail:
        return outs
    return tuple(x[:, :length].contiguous() for x in outs)


__all__ = ["kda_bwd"]
