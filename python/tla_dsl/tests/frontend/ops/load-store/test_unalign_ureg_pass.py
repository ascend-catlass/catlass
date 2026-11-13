"""IR checks for ``tla-unalign-ureg`` (Pre / ureg / Post + storageRoot conflicts).

Coverage is grouped by concern; alias / peel variants share factories instead of
one near-duplicate ``@tla.kernel`` per frontend spelling.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from typing import Any

import pytest

import catlass.tla as tla
from catlass.params import PostMode, UnalignLoadParams, UnalignStoreParams
from catlass.tla.runtime import make_fake_tensor

_S = UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE)
_L = UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found; build csrc/mlir first")
    return tla_compile


def _ir_after_unalign_ureg(kernel: Any, *, type_args: tuple[Any, ...], tmp_path: pathlib.Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "input.mlir"
    path.write_text(kernel.dump_mlir(type_args=type_args))
    result = subprocess.run(
        [
            str(_require_tla_compile()),
            str(path),
            "-o",
            str(tmp_path / "out.mlir"),
            "--mlir-print-ir-after=tla-unalign-ureg",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    # Need the dump after this pass. Later pipeline stages may fail on kernels
    # that only exist to exercise ureg conflicts (e.g. scalar_* on UPDATE next
    # windows); that must not hide a successful ureg rewrite.
    assert "IR Dump After" in output and "tla-unalign-ureg" in output, output
    assert "tla-unalign-ureg:" not in output, output
    return output


def _gm_tensor(extent: int = 256) -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (extent,),
        (1,),
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _gm_ub(mem: tla.Tensor, extent: int = 256):
    gm = tla.tile_view(mem, tla.make_shape(extent), tla.make_coord(0))
    ub = tla.make_tensor_like(
        tla.allocate(extent, tla.Float32, tla.AddressSpace.ub, extent),
        gm,
        tla.arch.RowMajor,
    )
    return gm, ub


def _layout64():
    return tla.make_layout(tla.make_shape(64), tla.make_stride(1), layoutTag=tla.arch.RowMajor)


def _innermost_for_span(ir: str) -> tuple[int, int]:
    last = None
    for m in re.finditer(r"\bscf\.for\b", ir):
        last = m
    assert last is not None, ir
    region = ir[last.start() :]
    yield_m = re.search(r"\bscf\.yield\b", region)
    assert yield_m, ir
    close = ir.find("}", last.start() + yield_m.end())
    assert close != -1, ir
    return last.start(), close + 1


def _after(ir: str) -> str:
    part = ir.split("IR Dump After", 1)[-1]
    # Drop later-pass failure text that may re-quote op names and inflate counts.
    for stop in ("Failed to run", ": error:", "PLEASE submit", "Stack dump:"):
        cut = part.find(stop)
        if cut != -1:
            part = part[:cut]
    return part


def _assert_elim_outside_for(after: str) -> None:
    for_start, for_end = _innermost_for_span(after)
    pre = re.search(r"\btla\.load_unalign_pre\b", after)
    empty = re.search(r"\btla\.create_ureg\b", after)
    load_u = re.search(r"\btla\.load_unalign_update\b", after)
    store_u = re.search(r"\btla\.store_unalign_update\b", after)
    post = re.search(r"\btla\.store_unalign_post\b", after)
    assert pre and empty and load_u and store_u and post, after
    assert pre.start() < for_start < load_u.start() < post.start(), after
    assert empty.start() < for_start < store_u.start() < post.start(), after
    assert post.start() >= for_end, after
    body = after[for_start:for_end]
    assert "load_unalign_pre" not in body, body
    assert "create_ureg" not in body, body
    assert "store_unalign_post" not in body, body


def _assert_dual_in_for_before_load(ir: str) -> None:
    after = _after(ir)
    for_start, for_end = _innermost_for_span(after)
    body = after[for_start:for_end]
    assert body.count("create_ureg") >= 1, body
    assert "store_unalign_post" in body, body
    assert "store_unalign_post" not in after[for_end:], after
    post = re.search(r"\btla\.store_unalign_post\b", body)
    load = re.search(r"\btla\.load(?:_unalign_update)?\b", body)
    assert post and load and post.start() < load.start(), body


# ---------------------------------------------------------------------------
# Placement / elim shapes
# ---------------------------------------------------------------------------


@tla.kernel
def _align_then_unalign(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            dst.store(src.load())
            x_u, src = src.load(_L)
            dst = dst.store(x_u, _S)
        tla.copy(z_gm, z_ub)


def test_ureg_pre_before_update_not_at_tile_create(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _align_then_unalign,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    align_load = re.search(r"\btla\.load\b(?!_)", after)
    align_store = re.search(r"\btla\.store\b(?!_)", after)
    pre = re.search(r"\btla\.load_unalign_pre\b", after)
    load_u = re.search(r"\btla\.load_unalign_update\b", after)
    empty = re.search(r"\btla\.create_ureg\b", after)
    store_u = re.search(r"\btla\.store_unalign_update\b", after)
    post = re.search(r"\btla\.store_unalign_post\b", after)
    assert align_load and align_store and pre and load_u and empty and store_u and post
    assert align_load.start() < align_store.start() < pre.start() < load_u.start()
    assert align_store.start() < empty.start() < store_u.start() < post.start()


@tla.kernel
def _natural_for(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                x, src = src.load(_L)
                dst = dst.store(x, _S)
        tla.copy(z_gm, z_ub)


@tla.kernel
def _if_then_for(mem_x: tla.Tensor, mem_z: tla.Tensor, flag: tla.Int32) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            if flag != 0:
                for _ in tla.range(3):
                    x, src = src.load(_L)
                    dst = dst.store(x, _S)
        tla.copy(z_gm, z_ub)


@tla.kernel
def _nested_for(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Outer for only for nesting; inner carries UnAlign windows (matches prior test)."""
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            for _outer in tla.range(2):
                src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
                dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
                for _ in tla.range(3):
                    x, src = src.load(_L)
                    dst = dst.store(x, _S)
        tla.copy(z_gm, z_ub)


@pytest.mark.parametrize(
    ("kernel", "type_args"),
    [
        (_natural_for, (_gm_tensor(), _gm_tensor())),
        (_if_then_for, (_gm_tensor(), _gm_tensor(), tla.Int32(1))),
        (_nested_for, (_gm_tensor(), _gm_tensor())),
    ],
    ids=["natural_for", "if_then_for", "nested_for"],
)
def test_ureg_elim_clean_for_shapes(kernel: Any, type_args: tuple[Any, ...], tmp_path: pathlib.Path) -> None:
    """Clean for (plain / if⊃for / for⊃for): Pre/create before for, Post after."""
    _assert_elim_outside_for(_after(_ir_after_unalign_ureg(kernel, type_args=type_args, tmp_path=tmp_path)))


@tla.kernel
def _for_then_if(mem_x: tla.Tensor, mem_z: tla.Tensor, flag: tla.Int32) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                if flag != 0:
                    x, src = src.load(_L)
                    dst = dst.store(x, _S)
        tla.copy(z_gm, z_ub)


def test_ureg_dual_for_then_if(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _for_then_if,
            type_args=(_gm_tensor(), _gm_tensor(), tla.Int32(1)),
            tmp_path=tmp_path,
        )
    )
    for_start, for_end = _innermost_for_span(after)
    body = after[for_start:for_end]
    assert "scf.if" in body and "load_unalign_pre" in body and "create_ureg" in body
    assert "store_unalign_post" in body
    assert "load_unalign_pre" not in after[:for_start]


@tla.kernel
def _multi_update_chain(mem_x: tla.Tensor, mem_a: tla.Tensor, mem_b: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    a_gm, a_ub = _gm_ub(mem_a)
    b_gm, b_ub = _gm_ub(mem_b)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out_a = tla.tile_view(a_ub, tla.make_shape(64), tla.make_coord(0))
            out_b = tla.tile_view(b_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                a, src = src.load(_L)
                out_a = out_a.store(a, _S)
                b, src = src.load(_L)
                out_b = out_b.store(b, _S)
        tla.copy(a_gm, a_ub)
        tla.copy(b_gm, b_ub)


def test_ureg_elim_multi_update_chain(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _multi_update_chain,
            type_args=(_gm_tensor(), _gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    for_start, for_end = _innermost_for_span(after)
    assert after[:for_start].count("load_unalign_pre") == 1
    assert "load_unalign_pre" not in after[for_start:for_end]
    assert after[for_start:for_end].count("load_unalign_update") == 2


@tla.kernel
def _two_independent_tiles(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_a: tla.Tensor, mem_b: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    y_gm, y_ub = _gm_ub(mem_y)
    a_gm, a_ub = _gm_ub(mem_a)
    b_gm, b_ub = _gm_ub(mem_b)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        with tla.vec.func(mode="simd"):
            s1 = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            s2 = tla.tile_view(y_ub, tla.make_shape(64), tla.make_coord(0))
            oa = tla.tile_view(a_ub, tla.make_shape(64), tla.make_coord(0))
            ob = tla.tile_view(b_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                a, s1 = s1.load(_L)
                b, s2 = s2.load(_L)
                oa = oa.store(a, _S)
                ob = ob.store(b, _S)
        tla.copy(a_gm, a_ub)
        tla.copy(b_gm, b_ub)


def test_ureg_elim_two_tiles_and_post_order(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _two_independent_tiles,
            type_args=(_gm_tensor(), _gm_tensor(), _gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    for_start, for_end = _innermost_for_span(after)
    pre, body, post = after[:for_start], after[for_start:for_end], after[for_end:]
    assert pre.count("load_unalign_pre") == 2 and pre.count("create_ureg") == 2
    assert "create_ureg" not in body and "store_unalign_post" not in body
    posts = [m.start() for m in re.finditer(r"\btla\.store_unalign_post\b", post)]
    assert len(posts) == 2 and posts[0] < posts[1]


@tla.kernel
def _inplace_load_store(mem_buf: tla.Tensor) -> None:
    buf_gm, buf_ub = _gm_ub(mem_buf)
    with tla.vector():
        tla.copy(buf_ub, buf_gm)
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(buf_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                cur = tile
                x, _ = cur.load(_L)
                tile = cur.store(x, _S)
        tla.copy(buf_gm, buf_ub)


def test_ureg_inplace_load_store_blocks_elim(tmp_path: pathlib.Path) -> None:
    after = _after(_ir_after_unalign_ureg(_inplace_load_store, type_args=(_gm_tensor(),), tmp_path=tmp_path))
    for_start, for_end = _innermost_for_span(after)
    body = after[for_start:for_end]
    assert "load_unalign_pre" in body and "create_ureg" in body
    assert "store_unalign_post" in body
    assert "create_ureg" not in after[:for_start]
    assert "store_unalign_post" not in after[for_end:]


# ---------------------------------------------------------------------------
# Same-root conflict (RAW / WAR / WAW) + peel variants
# ---------------------------------------------------------------------------


def _alias_from_kind(kind: str, *, z_ub: Any, cur: Any) -> Any:
    if kind == "tile_view":
        return tla.tile_view(cur, tla.make_shape(1), tla.make_coord(0))
    if kind == "make_tensor":
        return tla.make_tensor(z_ub.ptr, _layout64(), tla.make_coord(0))
    if kind == "make_tensor_ptr0":
        return tla.make_tensor(z_ub.ptr + 0, _layout64(), tla.make_coord(0))
    raise AssertionError(kind)


def _make_store_alias_load_kernel(kind: str) -> Any:
    @tla.kernel
    def kernel(mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
        z_gm, z_ub = _gm_ub(mem_z)
        out_gm, out_ub = _gm_ub(mem_out)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
                out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
                for _ in tla.range(3):
                    cur = dst
                    dst = cur.store(tla.full(11.0, tla.Float32), _S)
                    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                    alias = _alias_from_kind(kind, z_ub=z_ub, cur=cur)
                    out.store(alias.load(UnalignLoadParams()))
                    dst = dst.store(tla.full(13.0, tla.Float32), _S)
            tla.copy(z_gm, z_ub)
            tla.copy(out_gm, out_ub)

    kernel.__name__ = f"store_alias_{kind}"
    kernel.__qualname__ = kernel.__name__
    return kernel


@pytest.mark.parametrize(
    "kind",
    ["tile_view", "make_tensor", "make_tensor_ptr0"],
)
def test_ureg_same_root_alias_load_breaks_store_elim(kind: str, tmp_path: pathlib.Path) -> None:
    """Same-root observing load (view / make_tensor / ptr+0) ⇒ dual; Post before load."""
    _assert_dual_in_for_before_load(
        _ir_after_unalign_ureg(
            _make_store_alias_load_kernel(kind),
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )


def _make_store_peel_plain_load_kernel(kind: str) -> Any:
    """Store via peeled ptr spelling; plain tile_view load observes same root."""

    if kind == "ptr0":

        @tla.kernel
        def kernel(mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
            z_gm, z_ub = _gm_ub(mem_z)
            out_gm, out_ub = _gm_ub(mem_out)
            with tla.vector():
                with tla.vec.func(mode="simd"):
                    dst = tla.make_tensor(z_ub.ptr + 0, _layout64(), tla.make_coord(0))
                    plain = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
                    out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
                    for _ in tla.range(3):
                        dst = dst.store(tla.full(11.0, tla.Float32), _S)
                        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                        out.store(plain.load(UnalignLoadParams()))
                tla.copy(z_gm, z_ub)
                tla.copy(out_gm, out_ub)

    else:

        @tla.kernel
        def kernel(mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
            z_gm, z_ub = _gm_ub(mem_z)
            out_gm, out_ub = _gm_ub(mem_out)
            with tla.vector():
                with tla.vec.func(mode="simd"):
                    dst = tla.make_tensor(
                        tla.recast_ptr(z_ub.ptr, dtype=tla.Float32),
                        _layout64(),
                        tla.make_coord(0),
                    )
                    plain = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
                    out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
                    for _ in tla.range(3):
                        dst = dst.store(tla.full(11.0, tla.Float32), _S)
                        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                        out.store(plain.load(UnalignLoadParams()))
                tla.copy(z_gm, z_ub)
                tla.copy(out_gm, out_ub)

    kernel.__name__ = f"store_peel_{kind}"
    kernel.__qualname__ = kernel.__name__
    return kernel


@pytest.mark.parametrize("kind", ["ptr0", "recast_ptr"])
def test_ureg_store_peel_same_root_plain_load_dual(kind: str, tmp_path: pathlib.Path) -> None:
    _assert_dual_in_for_before_load(
        _ir_after_unalign_ureg(
            _make_store_peel_plain_load_kernel(kind),
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )


@tla.kernel
def _store_scalar_load_observe_for(mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
    """store UPDATE → ``dst[0]`` scalar_load → store: same-root RAW must dual."""
    z_gm, z_ub = _gm_ub(mem_z)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                dst = dst.store(tla.full(11.0, tla.Float32), _S)
                tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                v = dst[0]
                out[0] = v
                dst = dst.store(tla.full(13.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)
        tla.copy(out_gm, out_ub)


def test_ureg_scalar_load_breaks_store_elim(tmp_path: pathlib.Path) -> None:
    """``scalar_load`` is a same-root observer ⇒ Post before it; no for-elim."""
    after = _after(
        _ir_after_unalign_ureg(
            _store_scalar_load_observe_for,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    for_start, for_end = _innermost_for_span(after)
    body = after[for_start:for_end]
    assert "create_ureg" in body, body
    assert "store_unalign_post" in body, body
    assert "store_unalign_post" not in after[for_end:], after
    post = re.search(r"\btla\.store_unalign_post\b", body)
    scalar = re.search(r"\btla\.scalar_load\b", body)
    assert post and scalar and post.start() < scalar.start(), body


@tla.kernel
def _load_scalar_store_war(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """load UPDATE → ``nxt[0]=…`` scalar_store → load: same-root WAR cuts ureg."""
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            v, nxt = src.load(_L)
            out.store(v)
            nxt[0] = 11.0
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            seen, _ = nxt.load(_L)
            out.store(seen)
        tla.copy(z_gm, z_ub)


def test_ureg_scalar_store_breaks_load_reuse(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _load_scalar_store_war,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    assert after.count("load_unalign_pre") == 2, after
    assert after.count("load_unalign_update") == 2, after


@tla.kernel
def _nested_if_observe(mem_z: tla.Tensor, mem_out: tla.Tensor, flag: tla.Int32) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            dst = dst.store(tla.full(1.0, tla.Float32), _S)
            if flag != 0:
                if flag != 0:
                    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                    out.store(dst.load(UnalignLoadParams()))
            dst = dst.store(tla.full(2.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)
        tla.copy(out_gm, out_ub)


def test_ureg_nested_if_load_cuts_store_reuse(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _nested_if_observe,
            type_args=(_gm_tensor(), _gm_tensor(), tla.Int32(1)),
            tmp_path=tmp_path,
        )
    )
    assert after.count("create_ureg") == 2
    assert after.count("store_unalign_post") == 2


@tla.kernel
def _load_war_ptr0(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            _a, nxt = src.load(_L)
            alias = tla.make_tensor(nxt.ptr + 0, _layout64(), tla.make_coord(0))
            _ = alias.store(tla.full(11.0, tla.Float32), _S)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            seen, _ = nxt.load(_L)
            out.store(seen)
        tla.copy(z_gm, z_ub)


def test_ureg_load_war_ptr0_cuts_reuse(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _load_war_ptr0,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    assert after.count("load_unalign_pre") == 2
    assert after.count("load_unalign_update") == 2


@tla.kernel
def _store_then_load_dirs(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            dst = dst.store(dst.load(), _S)
            seen, dst = dst.load(_L)
            out = out.store(seen, _S)
        tla.copy(z_gm, z_ub)


def test_ureg_store_then_load_separate_maps(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _store_then_load_dirs,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    post = re.search(r"\btla\.store_unalign_post\b", after)
    pre = re.search(r"\btla\.load_unalign_pre\b", after)
    assert post and pre and post.start() < pre.start(), after


@tla.kernel
def _load_store_load(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            v, nxt = src.load(_L)
            out.store(v)
            _ = nxt.store(tla.full(9.0, tla.Float32), _S)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            seen, _ = nxt.load(_L)
            out.store(seen)
        tla.copy(z_gm, z_ub)


def test_ureg_intervening_store_invalidates_load(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _load_store_load,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    assert after.count("load_unalign_pre") == 2


@tla.kernel
def _straight_waw(mem_z: tla.Tensor) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            a = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            b = tla.make_tensor(z_ub.ptr, _layout64(), tla.make_coord(0))
            a = a.store(tla.full(1.0, tla.Float32), _S)
            b = b.store(tla.full(9.0, tla.Float32), _S)
            a = a.store(tla.full(2.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)


@tla.kernel
def _for_multi_waw(mem_z: tla.Tensor) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            a = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            b = tla.make_tensor(z_ub.ptr, _layout64(), tla.make_coord(0))
            for _ in tla.range(3):
                a = a.store(tla.full(1.0, tla.Float32), _S)
                b = b.store(tla.full(2.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)


def test_ureg_same_root_waw_dual(tmp_path: pathlib.Path) -> None:
    """Straight reuse cut + for multi-stream dual (both same-root WAW)."""
    straight = _after(_ir_after_unalign_ureg(_straight_waw, type_args=(_gm_tensor(),), tmp_path=tmp_path / "s"))
    assert straight.count("create_ureg") == 3
    assert straight.count("store_unalign_post") == 3

    for_ir = _after(_ir_after_unalign_ureg(_for_multi_waw, type_args=(_gm_tensor(),), tmp_path=tmp_path / "f"))
    for_start, for_end = _innermost_for_span(for_ir)
    body = for_ir[for_start:for_end]
    assert body.count("create_ureg") == 2 and body.count("store_unalign_post") == 2
    assert "store_unalign_post" not in for_ir[for_end:]


@tla.kernel
def _load_aba(mem_x: tla.Tensor, mem_out: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            a = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            b = tla.make_tensor(x_ub.ptr, _layout64(), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            v, a = a.load(_L)
            out.store(v)
            v, b = b.load(_L)
            out.store(v)
            v, a = a.load(_L)
            out.store(v)
        tla.copy(out_gm, out_ub)


def test_ureg_same_root_intervening_load_cuts_reuse(tmp_path: pathlib.Path) -> None:
    after = _after(_ir_after_unalign_ureg(_load_aba, type_args=(_gm_tensor(), _gm_tensor()), tmp_path=tmp_path))
    assert after.count("load_unalign_pre") == 3


@tla.kernel
def _other_root_still_elims(mem_z: tla.Tensor, mem_x: tla.Tensor, mem_out: tla.Tensor) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    x_gm, x_ub = _gm_ub(mem_x)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                dst = dst.store(tla.full(1.0, tla.Float32), _S)
                other = tla.make_tensor(x_ub.ptr + 0, _layout64(), tla.make_coord(0))
                out.store(other.load(UnalignLoadParams()))
        tla.copy(z_gm, z_ub)
        tla.copy(out_gm, out_ub)


def test_ureg_other_root_ptr0_does_not_block_elim(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _other_root_still_elims,
            type_args=(_gm_tensor(), _gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    for_start, for_end = _innermost_for_span(after)
    assert after[:for_start].count("create_ureg") == 1
    assert "create_ureg" not in after[for_start:for_end]
    assert after[for_end:].count("store_unalign_post") == 1


# ---------------------------------------------------------------------------
# No cross-for 续流
# ---------------------------------------------------------------------------


@tla.kernel
def _prologue_for_epilogue(mem_z: tla.Tensor) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            dst = dst.store(tla.full(1.0, tla.Float32), _S)
            for _ in tla.range(3):
                dst = dst.store(tla.full(2.0, tla.Float32), _S)
            dst = dst.store(tla.full(3.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)


@tla.kernel
def _for_then_observe_then_store(mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
    z_gm, z_ub = _gm_ub(mem_z)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                dst = dst.store(tla.full(11.0, tla.Float32), _S)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            out.store(dst.load(UnalignLoadParams()))
            dst = dst.store(tla.full(13.0, tla.Float32), _S)
        tla.copy(z_gm, z_ub)
        tla.copy(out_gm, out_ub)


@tla.kernel
def _load_prologue_for(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            v, src = src.load(_L)
            out.store(v)
            for _ in tla.range(3):
                v, src = src.load(_L)
                out.store(v)
        tla.copy(z_gm, z_ub)


def test_ureg_no_cross_for_continue(tmp_path: pathlib.Path) -> None:
    """Prologue/for/epilogue are separate streams; for result not reused past observer."""
    pe = _after(
        _ir_after_unalign_ureg(
            _prologue_for_epilogue,
            type_args=(_gm_tensor(),),
            tmp_path=tmp_path / "pe",
        )
    )
    fs, fe = _innermost_for_span(pe)
    assert pe.count("create_ureg") == 3
    assert "create_ureg" not in pe[fs:fe]
    assert pe[:fs].count("create_ureg") == 2
    assert pe[fe:].count("create_ureg") == 1
    assert pe[fe:].count("store_unalign_post") == 2

    obs = _after(
        _ir_after_unalign_ureg(
            _for_then_observe_then_store,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path / "obs",
        )
    )
    fs, fe = _innermost_for_span(obs)
    post_after = obs[fe:]
    assert post_after.count("create_ureg") == 1
    assert post_after.count("store_unalign_post") == 2
    post = re.search(r"\btla\.store_unalign_post\b", post_after)
    load_u = re.search(r"\btla\.load\b", post_after)
    assert post and load_u and post.start() < load_u.start()

    ld = _after(
        _ir_after_unalign_ureg(
            _load_prologue_for,
            type_args=(_gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path / "ld",
        )
    )
    fs, fe = _innermost_for_span(ld)
    assert ld[:fs].count("load_unalign_pre") == 2
    assert "load_unalign_pre" not in ld[fs:fe]


@tla.kernel
def _fallback_head(mem_x: tla.Tensor, mem_z: tla.Tensor, mem_out: tla.Tensor) -> None:
    x_gm, x_ub = _gm_ub(mem_x)
    z_gm, z_ub = _gm_ub(mem_z)
    out_gm, out_ub = _gm_ub(mem_out)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(out_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(3):
                cur = z_tile
                # Discard next window on purpose: non-chain head forces dual-instr.
                z_reg, z_next_discarded = cur.load(_L)
                assert z_next_discarded is not None
                x_reg, x_tile = x_tile.load(_L)
                z_tile = cur.store(tla.add(x_reg, z_reg), _S)
                _ = out.store(x_reg)
        tla.copy(z_gm, z_ub)
        tla.copy(out_gm, out_ub)


def test_ureg_fallback_head_rewrites_remaining(tmp_path: pathlib.Path) -> None:
    after = _after(
        _ir_after_unalign_ureg(
            _fallback_head,
            type_args=(_gm_tensor(), _gm_tensor(), _gm_tensor()),
            tmp_path=tmp_path,
        )
    )
    fs, fe = _innermost_for_span(after)
    assert "load_unalign_pre" in after[fs:fe]
    assert "load_unalign_pre" in after[:fs]
    assert after[fs:fe].count("load_unalign_update") == 2
