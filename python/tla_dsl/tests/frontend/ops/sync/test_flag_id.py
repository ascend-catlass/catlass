from __future__ import annotations

import pytest

import catlass.tla as tla


@tla.kernel
def default_flag_kernel() -> None:
    with tla.vector():
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        tla.set_flag(ready)
        tla.wait_flag(ready)


def test_flag_defaults_to_automatic_id(compiler_tlair) -> None:
    text = compiler_tlair(default_flag_kernel)
    assert '"tla.flag"' in text
    assert "id = -1 : i64" in text


@pytest.mark.parametrize("value", [-1, 0, 3, 7])
def test_flag_accepts_explicit_id(value: int, compiler_tlair) -> None:
    @tla.kernel
    def kernel() -> None:
        with tla.vector():
            ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR, id=value)
            tla.set_flag(ready)
            tla.wait_flag(ready)

    assert f"id = {value} : i64" in compiler_tlair(kernel)


def test_flag_accepts_positional_id(compiler_tlair) -> None:
    @tla.kernel
    def kernel() -> None:
        with tla.vector():
            ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR, 5)
            tla.set_flag(ready)
            tla.wait_flag(ready)

    assert "id = 5 : i64" in compiler_tlair(kernel)


@pytest.mark.parametrize("value", [-2, 8])
def test_flag_rejects_out_of_range_id(value: int) -> None:
    @tla.kernel
    def kernel() -> None:
        tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR, id=value)

    with pytest.raises(tla.TlaCoreAPIError, match="id must be -1 or in range 0..7"):
        kernel.dump_mlir()


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_flag_rejects_non_integer_id(value: object) -> None:
    @tla.kernel
    def kernel() -> None:
        tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR, id=value)

    with pytest.raises(tla.TlaCoreAPIError, match="expected int"):
        kernel.dump_mlir()
