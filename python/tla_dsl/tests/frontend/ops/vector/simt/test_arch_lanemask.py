"""Frontend scope, mode, and unsigned semantics of SIMT lane masks."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.base_dsl import BaseDSL
from catlass.tla.runtime import make_fake_tensor

MODES = ("eq", "lt", "le", "gt", "ge")


def _query(mode):
    value = getattr(tla.arch, "lanemask_" + mode)()
    assert isinstance(value, tla.UInt32)
    return value


@tla.kernel
def _mask_query(out: tla.Tensor, mode: tla.Constexpr[str]) -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=(4, 4, 3)):
            out[0] = _query(mode)
            out[1] = (out[0].to(tla.UInt64) >> tla.UInt64(1)).to(tla.UInt32)


@tla.kernel
def _mask_in_vector(mode: tla.Constexpr[str]) -> None:
    with tla.vector():
        _ = _query(mode)


@tla.kernel
def _mask_in_simd(mode: tla.Constexpr[str]) -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simd"):
            _ = _query(mode)


@pytest.mark.parametrize("mode", MODES)
def test_lanemask_mode_and_unsigned_widening(mode) -> None:
    out = make_fake_tensor(tla.UInt32, (2,), (1,))
    lowered = BaseDSL()._lower(
        _mask_query.fn,
        kind=_mask_query.kind,
        options=dict(_mask_query.options),
        type_args=(out, mode),
        location=_mask_query.decorator_location,
    )
    assert lowered.module.operation.verify()
    mlir = lowered.asm()
    assert f'tla.arch.lanemask "{mode}" : i32' in mlir
    assert 'tla.simt_cast "extui"' in mlir
    assert '"extsi"' not in mlir


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "kernel",
    [_mask_in_vector, _mask_in_simd],
)
def test_lanemask_requires_simt_region(kernel, mode) -> None:
    with pytest.raises(Exception, match=rf"arch\.lanemask_{mode}.*mode='simt'"):
        kernel.dump_mlir(type_args=(mode,))


@pytest.mark.parametrize("mode", MODES)
def test_lanemask_rejects_host_calls(mode) -> None:
    with pytest.raises(Exception, match=rf"arch\.lanemask_{mode}"):
        getattr(tla.arch, "lanemask_" + mode)()
