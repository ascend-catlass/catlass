"""Tensor / type-descriptor tests for host runtime tensors and MLIR descriptors."""

from __future__ import annotations

from pathlib import Path

import pytest

import catlass.tla as tla
from catlass import _tla_type_bridge
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import from_dlpack
from catlass._mlir import ir as mlir_ir


def test_type_bridge_prefers_nested_worktree_build_over_packaged_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "python" / "tla_dsl" / "catlass"
    package_root.mkdir(parents=True)
    module_path = package_root / "_tla_type_bridge.py"
    module_path.touch()
    packaged = package_root / "_tla_type_bridge_native.packaged.so"
    packaged.touch()
    nested = (
        tmp_path
        / "python"
        / "tla_dsl"
        / "csrc"
        / "mlir"
        / "build"
        / "python"
        / "catlass"
        / "_tla_type_bridge_native.nested.so"
    )
    nested.parent.mkdir(parents=True)
    nested.touch()

    monkeypatch.setattr(_tla_type_bridge, "__file__", str(module_path))

    assert _tla_type_bridge._resolve_bridge_extension_path() == nested


def test_type_bridge_uses_packaged_artifact_without_worktree_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "python" / "tla_dsl" / "catlass"
    package_root.mkdir(parents=True)
    module_path = package_root / "_tla_type_bridge.py"
    module_path.touch()
    packaged = package_root / "_tla_type_bridge_native.packaged.so"
    packaged.touch()

    monkeypatch.setattr(_tla_type_bridge, "__file__", str(module_path))

    assert _tla_type_bridge._resolve_bridge_extension_path() == packaged


def test_tla_type_descriptors_construct_native_mlir_types() -> None:
    with mlir_ir.Context() as ctx:
        shape = tla.types.TlaIndexTreeType("shape", ((None, 16), 8))
        stride = tla.types.TlaIndexTreeType("stride", ((16, 1), 128))
        coord = tla.types.TlaIndexTreeType("coord", (0, 0))
        origin = tla.types.TlaIndexTreeType("shape", ((32, 16), 8))
        layout = tla.types.TlaLayoutDescriptor(shape, stride, origin)
        tensor = tla.types.TlaTensorTypeDescriptor(
            layout=tla.types.TlaLayoutDescriptor(
                shape, stride, origin, layout_tag="zN"
            ),
            coord=(0, 0, 0),
            element_type="f16",
            addrspace="gm",
            ptr_alignment=2,
        )

        assert str(shape.to_mlir_type(ctx)) == "!tla.shape<(?,16),8>"
        assert str(stride.to_mlir_type(ctx)) == "!tla.stride<(16,1),128>"
        assert str(coord.to_mlir_type(ctx)) == "!tla.coord<0,0>"
        assert str(layout.to_mlir_type(ctx)) == (
            "!tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, RowMajor>"
        )
        assert str(tensor.to_mlir_type(ctx)) == (
            "!tla.tensor<!tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, zN>, !tla.coord<0,0,0>, !tla.ptr<f16, gm, 2>>"
        )


def test_vector_ssa_type_roundtrip_and_bridge_accessors() -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        f32 = tla.Float32.mlir_type(ctx)
        static_type = _tla_type_bridge.vector_ssa_type_get(ctx, 64, f32)
        dynamic_type = _tla_type_bridge.vector_ssa_type_get(ctx, None, f32)

        assert str(static_type) == "!tla.vector<64xf32>"
        assert str(dynamic_type) == "!tla.vector<?xf32>"
        assert _tla_type_bridge.type_is_vector_ssa(static_type)
        assert _tla_type_bridge.tla_type_category(static_type) == "vector_ssa"
        assert _tla_type_bridge.vector_ssa_valid_lanes_get(static_type) == 64
        assert _tla_type_bridge.vector_ssa_valid_lanes_get(dynamic_type) is None
        assert str(_tla_type_bridge.vector_ssa_element_type_get(static_type)) == "f32"


def test_mask_ssa_type_roundtrip_and_bridge_accessors() -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        for physical_lanes in (32, 64, 128, 256):
            mask_type = _tla_type_bridge.mask_ssa_type_get(ctx, physical_lanes)

            assert str(mask_type) == f"!tla.mask<{physical_lanes}>"
            assert _tla_type_bridge.type_is_mask_ssa(mask_type)
            assert _tla_type_bridge.tla_type_category(mask_type) == "mask_ssa"
            assert _tla_type_bridge.mask_ssa_physical_lanes_get(mask_type) == physical_lanes
            assert (
                str(tla.types.TlaMaskSSATypeDescriptor(physical_lanes).to_mlir_type(ctx))
                == f"!tla.mask<{physical_lanes}>"
            )


@pytest.mark.parametrize("physical_lanes", (0, 1, 31, 33, 63, 65, 257))
def test_mask_ssa_type_rejects_invalid_physical_lane_counts(physical_lanes: int) -> None:
    with pytest.raises(ValueError, match="physical_lanes must be one of"):
        tla.types.TlaMaskSSATypeDescriptor(physical_lanes)

    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        with pytest.raises(mlir_ir.MLIRError, match="lane count must be one of"):
            mlir_ir.Type.parse(f"!tla.mask<{physical_lanes}>")


def test_legacy_unparameterized_mask_type_is_rejected() -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        with pytest.raises(mlir_ir.MLIRError):
            mlir_ir.Type.parse("!tla.mask")


def test_register_ssa_wrappers_are_publicly_exported() -> None:
    assert tla.VectorSSA is core_api_mod.VectorSSA
    assert tla.MaskSSA is core_api_mod.MaskSSA


@pytest.mark.parametrize(
    ("type_text", "valid"),
    (
        ("!tla.vector<64xf32>", True),
        ("!tla.vector<65xf32>", False),
        ("!tla.vector<128xf16>", True),
        ("!tla.vector<129xf16>", False),
        ("!tla.vector<?xf32>", True),
    ),
)
def test_vector_ssa_type_enforces_register_capacity(
    type_text: str, valid: bool
) -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        if valid:
            assert str(mlir_ir.Type.parse(type_text)) == type_text
        else:
            with pytest.raises(mlir_ir.MLIRError, match="valid lane count"):
                mlir_ir.Type.parse(type_text)


def test_vector_ssa_type_rejects_i1_elements() -> None:
    with pytest.raises(ValueError, match="unsupported VectorSSA element type"):
        tla.types.TlaVectorSSATypeDescriptor(1, "i1")

    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        with pytest.raises(mlir_ir.MLIRError, match="byte-aligned width"):
            mlir_ir.Type.parse("!tla.vector<1xi1>")


def test_legacy_tla_value_type_and_python_marker_are_removed() -> None:
    assert not hasattr(tla, "TlaValue")
    assert not hasattr(_tla_type_bridge, "value_type_get")
    with mlir_ir.Context() as ctx, pytest.raises(mlir_ir.MLIRError):
        _tla_type_bridge.load_tla_dialect(ctx)
        mlir_ir.Type.parse("!tla.value<f32>")


def test_ptr_type_uses_bridge_accessors_for_nested_pointee() -> None:
    with mlir_ir.Context() as ctx:
        with mlir_ir.Location.unknown(ctx):
            pointee = mlir_ir.MemRefType.get((4,), tla.Float16.mlir_type(ctx))
        ptr_type = tla.types.PtrType.get(pointee, "l1", 32, context=ctx)

        assert tla.types.PtrType.isinstance(ptr_type)
        assert ptr_type.pointee == pointee
        assert ptr_type.addrspace == "l1"
        assert ptr_type.alignment == 32


def test_tla_type_descriptors_require_native_bridge(monkeypatch) -> None:
    monkeypatch.setattr(tla.types._tla_type_bridge, "_EXTENSION", None)
    monkeypatch.setattr(
        tla.types._tla_type_bridge, "_resolve_bridge_extension_path", lambda: None
    )

    with pytest.raises(
        tla.types._tla_type_bridge.TlaTypeBridgeUnavailableError,
        match="Tla type bridge extension not found",
    ):
        tla.types.TlaIndexTreeType("shape", (1, 2)).to_mlir_type()


def test_deep_nested_shape_groups_are_rejected() -> None:
    @tla.kernel
    def _bad() -> None:
        _ = tla.make_shape(((1, 2), (3, 4)))

    with pytest.raises(tla.TlaCoreAPIError, match="one-level leaf groups"):
        _ = _bad.dump_mlir()


def test_deep_nested_tensor_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-level leaf groups"):
        tla.types.TlaIndexTreeType("shape", (((1, 2), (3, 4)),))


def test_tensor_string_metadata_is_rejected() -> None:
    tensor_type: object = (
        "!tla.tensor<!tla.shape<1,2>, !tla.stride<2,1>, "
        "!tla.coord<0,0>, !tla.shape<1,2>, f16, gm, RowMajor>"
    )
    with pytest.raises(TlaLoweringError, match="TlaTensorTypeDescriptor"):
        core_api_mod._tla_tensor_descriptor_from_type_or_value(tensor_type)


def test_make_fake_tensor_is_unbound() -> None:
    from catlass.tla.runtime import make_fake_tensor

    fake = make_fake_tensor(
               tla.Float32,
               (4, 8),
               (8, 1),
               addrspace=tla.AddressSpace.ub,
               origin_shape=(4, 8),
               layout_tag=tla.arch.RowMajor,
           )
    assert fake.data_ptr == 0
    assert fake._external_binding is False
    assert fake._flat_shape == (4, 8)
    assert fake.stride == (8, 1)
    assert isinstance(fake, tla.Tensor)


def test_from_dlpack_requires_layout_tag() -> None:
    class _Buf:
        def __dlpack__(self, stream: int | None = None):
            del stream
            return object()

    with pytest.raises(TypeError, match=r"required keyword-only argument: 'layout_tag'"):
        from_dlpack(_Buf())
