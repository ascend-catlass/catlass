"""Tests for PR #114: location handling refactor — direct mlir_ir.Location creation
in dsl_user_op, removal of _CapturedLocation intermediate class."""

from __future__ import annotations

import pytest
from catlass._mlir import ir as mlir_ir

import catlass.tla as tla
import catlass.runtime as runtime_mod

# ---------------------------------------------------------------------------
# _CapturedLocation / _to_mlir_location removal
# ---------------------------------------------------------------------------


def test_captured_location_class_is_removed() -> None:
    """_CapturedLocation dataclass no longer exists in runtime module."""
    assert not hasattr(runtime_mod, "_CapturedLocation")


def test_to_mlir_location_function_is_removed() -> None:
    """_to_mlir_location conversion function no longer exists in runtime module."""
    assert not hasattr(runtime_mod, "_to_mlir_location")


# ---------------------------------------------------------------------------
# _capture_caller_location now returns mlir_ir.Location directly
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# dsl_user_op loc type validation
# ---------------------------------------------------------------------------


def test_dsl_op_rejects_non_mlir_location_loc() -> None:
    """Passing a non-mlir_ir.Location, non-None loc raises TypeError."""
    with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
        tla.make_shape(1, 2, loc="bad_location")


def test_dsl_op_rejects_int_as_loc() -> None:
    with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
        tla.make_shape(1, 2, loc=42)





# ---------------------------------------------------------------------------
# Auto-capture location
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Backward compatibility: representative ops work without explicit loc
# ---------------------------------------------------------------------------


def test_representative_ops_work_without_explicit_loc() -> None:
    """Representative DSL ops across categories work with auto-captured location."""
    # Region-requiring verbs (set_flag/wait_flag/pipe_barrier/cross_core_*/
    # mutex_lock/unlock) can't be emitted here: they must be nested in a
    # tla.cube/tla.vector region, and region ops are not enterable under
    # _eager_capture. The auto-location path is identical for every dsl_user_op,
    # so representative region-free ops across categories still exercise it.
    with runtime_mod._eager_capture() as state:
        sh = tla.make_shape(1, 2)
        tla.make_coord(0, 0)
        st = tla.make_stride(1, 100)
        tla.make_layout(sh, st)
        tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        tla.cross_flag("x")
        tla.mutex(resource="l0a_ping", id=-1)

    mlir = state.module.operation.get_asm(
        print_generic_op_form=True, assume_verified=False
    )
    for op_name in (
        "tla.make_shape",
        "tla.make_layout",
        "tla.flag",
        "tla.cross_flag",
        "tla.mutex",
    ):
        assert op_name in mlir
