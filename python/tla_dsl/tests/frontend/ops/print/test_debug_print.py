from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import inspect
import re

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from dsl_tutorials.debug_print import debug_print as debug_print_example
from dsl_tutorials.debug_print import debug_print_format


@tla.kernel
def _scalar_kernel(i: object, j: object, f: object) -> None:
    with tla.cube():
        tla.print(i)
        tla.print(f)
    with tla.vector():
        tla.print(f)
        tla.print(j)


@tla.kernel
def _computed_i32_kernel(x: object, y: object) -> None:
    with tla.vector():
        tla.print(x + y)


@tla.kernel
def _computed_f32_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.print(x + y)


@tla.kernel
def _computed_f16_kernel(x: object, y: object) -> None:
    with tla.vector():
        tla.print(x + y)


@tla.kernel
def _computed_narrow_integer_kernel(x: object, y: object) -> None:
    with tla.vector():
        tla.print(x + y)


@tla.kernel
def _literal_kernel() -> None:
    with tla.vector():
        tla.print(-(2**31))
        tla.print(2**31 - 1)
        tla.print(1.25)


@tla.kernel
def _formatted_literals_kernel(i: object, f: object) -> None:
    with tla.vector():
        tla.print("")
        tla.print("hello")
        tla.print("x={}", i)
        tla.print("v={}", f)
        tla.print("x={} y={}", i, f)
        tla.print("{} {}", i, i)
        tla.print("{{}} % {}", i)


@tla.kernel
def _formatted_cube_kernel(i: object) -> None:
    with tla.cube():
        tla.print("cube {}", i)


@tla.kernel
def _formatted_python_literals_kernel() -> None:
    with tla.vector():
        tla.print("x={} v={}", 7, 1.25)


@tla.kernel
def _formatted_unsigned_literals_kernel() -> None:
    with tla.vector():
        tla.print("u={} {} {}", tla.UInt8(255), tla.UInt16(65535), tla.UInt32(7))


@tla.kernel
def _f32_literal_location_kernel() -> None:
    with tla.vector():
        tla.print(1.25)


@tla.kernel
def _regionless_kernel() -> None:
    tla.print(tla.Int32(1))


@tla.kernel
def _regionless_formatted_kernel() -> None:
    tla.print("x={}", tla.Int32(1))


@tla.kernel
def _typed_scalar_kernel(value: object) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def _formatted_typed_scalar_kernel(value: object) -> None:
    with tla.vector():
        tla.print("x={}", value)


@tla.kernel
def _pointer_kernel() -> None:
    ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 32)
    with tla.vector():
        tla.print(ptr)


@tla.kernel
def _tensor_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def _formatted_tensor_arg_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print("tensor={}", value)


@tla.kernel
def _vector_value_kernel(value: tla.Tensor) -> None:
    tile = tla.tile_view(value, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.print(tile.load())


@tla.kernel
def _formatted_vector_value_kernel(value: tla.Tensor) -> None:
    tile = tla.tile_view(value, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.print("{}", tile.load())


def _host_vector_tensor() -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


def test_print_signature_is_variadic() -> None:
    signature = inspect.signature(tla.print)
    assert str(signature) == "(value, *args, /)"
    assert list(signature.parameters) == ["value", "args"]
    assert signature.parameters["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert signature.parameters["args"].kind is inspect.Parameter.VAR_POSITIONAL


def test_debug_print_materializes_api_local_i32_and_f32_literals() -> None:
    mlir = _literal_kernel.dump_mlir()

    assert mlir.count("tla.debug_print") == 3
    assert f"{-(2**31)} : i32" in mlir
    assert f"{2**31 - 1} : i32" in mlir
    assert "1.250000e+00 : f32" in mlir or "1.25" in mlir


def test_debug_print_emits_formatted_string_and_variadic_scalars() -> None:
    mlir = _formatted_literals_kernel.dump_mlir(
        type_args=(tla.Int32(0), tla.Float32(0.0))
    )

    assert mlir.count("tla.debug_print") == 7
    assert 'tla.debug_print format ""' in mlir
    assert 'tla.debug_print format "hello"' in mlir
    assert 'tla.debug_print format "x={}"' in mlir
    assert 'tla.debug_print format "v={}"' in mlir
    assert 'tla.debug_print format "x={} y={}"' in mlir
    assert 'tla.debug_print format "{} {}"' in mlir
    assert 'tla.debug_print format "{{}} % {}"' in mlir
    assert "debug_printf" not in mlir


def test_debug_print_materializes_unsigned_literals_through_signless_constants() -> (
    None
):
    mlir = _formatted_unsigned_literals_kernel.dump_mlir()

    assert mlir.count("builtin.unrealized_conversion_cast") == 3
    for width in (8, 16, 32):
        assert re.search(rf"arith\.constant.*(?:->|:) i{width}", mlir)
        assert re.search(rf"(?:\(i{width}\) -> ui{width}|i{width} to ui{width})", mlir)


@pytest.mark.parametrize(
    ("scalar", "expected_type"),
    (
        pytest.param(tla.Int8(-37), "i8", id="i8"),
        pytest.param(tla.Int16(-30000), "i16", id="i16"),
        pytest.param(tla.Int32(-7), "i32", id="i32"),
        pytest.param(tla.UInt8(255), "ui8", id="u8"),
        pytest.param(tla.UInt16(65535), "ui16", id="u16"),
        pytest.param(tla.UInt32(4294967295), "ui32", id="u32"),
        pytest.param(tla.Float16(1.25), "f16", id="f16"),
        pytest.param(tla.Float32(1.25), "f32", id="f32"),
    ),
)
def test_debug_print_emits_formatted_typed_scalar(
    scalar: object, expected_type: str
) -> None:
    mlir = _formatted_typed_scalar_kernel.dump_mlir(type_args=(scalar,))

    debug_line = next(line for line in mlir.splitlines() if "tla.debug_print" in line)
    assert 'format "x={}"' in debug_line or 'format = "x={}"' in debug_line
    assert re.search(rf"(?:\({expected_type}\)|: {expected_type}(?:\s|$))", debug_line)


def test_debug_print_emits_formatted_cube_region() -> None:
    mlir = _formatted_cube_kernel.dump_mlir(type_args=(tla.Int32(0),))

    assert "tla.cube" in mlir
    assert 'tla.debug_print format "cube {}"' in mlir


def test_debug_print_materializes_formatted_python_literals() -> None:
    mlir = _formatted_python_literals_kernel.dump_mlir()

    assert 'tla.debug_print format "x={} v={}"' in mlir
    assert "7 : i32" in mlir
    assert "1.250000e+00 : f32" in mlir or "1.25" in mlir


def test_debug_print_formatted_generic_mlir_uses_same_op() -> None:
    lowered = BaseDSL()._lower(
        _formatted_literals_kernel.fn,
        kind=_formatted_literals_kernel.kind,
        options=dict(_formatted_literals_kernel.options),
        type_args=(tla.Int32(0), tla.Float32(0.0)),
        location=_formatted_literals_kernel.decorator_location,
    )
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            print_generic_op_form=True,
            assume_verified=False,
        )

    assert '"tla.debug_print"' in mlir
    assert 'format = "x={} y={}"' in mlir
    assert 'format = "{{}} % {}"' in mlir
    assert "debug_printf" not in mlir


def test_debug_print_format_output_accepts_cross_core_reordering() -> None:
    output = "\n".join(
        (
            "TLA printf: core=3 block=1 repeated",
            "TLA printf: core=8 block=0 single",
            "TLA printf: core=2 block=0 repeated",
            "TLA printf: core=9 block=1 single",
            "TLA printf: core=4 block=1 repeated",
            "TLA printf: core=1 block=0 repeated",
        )
    )

    debug_print_format._verify_case_output(
        output, payloads=("single", "repeated", "repeated"), block=2
    )


@pytest.mark.parametrize(
    "output",
    (
        "TLA printf: core=1 malformed scalar printf TLV",
        "TLA printf: core=1 block=x value",
        "TLA printf: core=1 block=2 value",
        "TLA printf: core=1 block=0 value\nTLA printf: core=2 block=0 value",
        "TLA printf: core=1 block=0 unexpected",
        "",
    ),
)
def test_debug_print_format_output_rejects_invalid_records(output: str) -> None:
    with pytest.raises(RuntimeError):
        debug_print_format._verify_case_output(output, payloads=("value",), block=1)


def test_debug_print_emits_direct_scalars_in_both_regions() -> None:
    mlir = _scalar_kernel.dump_mlir(
        type_args=(tla.Int32(0), tla.Int32(0), tla.Float32(0.0))
    )

    cube_start = mlir.index("tla.cube")
    vector_start = mlir.index("tla.vector")
    assert cube_start < vector_start
    for section in (mlir[cube_start:vector_start], mlir[vector_start:]):
        assert section.count("tla.debug_print") == 2
        assert "i32" in section and "f32" in section
    debug_lines = [line for line in mlir.splitlines() if "tla.debug_print" in line]
    assert any("%arg0" in line for line in debug_lines)
    assert any("%arg2" in line for line in debug_lines)


@pytest.mark.parametrize(
    ("kernel", "type_args"),
    (
        (_computed_i32_kernel, (tla.Int32(0), tla.Int32(0))),
        (
            _computed_f32_kernel,
            (tla.Float32(0.0), tla.Float32(0.0)),
        ),
    ),
)
def test_debug_print_emits_computed_runtime_scalar(
    kernel: object, type_args: tuple[object, object]
) -> None:
    mlir = kernel.dump_mlir(type_args=type_args)

    assert mlir.count("tla.debug_print") == 1
    assert "arith.add" in mlir


def test_debug_print_preserves_computed_f16_type() -> None:
    mlir = _computed_f16_kernel.dump_mlir(
        type_args=(tla.Float16(1.25), tla.Float16(0.75))
    )

    debug_line = next(line for line in mlir.splitlines() if "tla.debug_print" in line)
    assert "arith.addf" in mlir
    assert re.search(r"(?:\(f16\)|: f16(?:\s|$))", debug_line)
    assert "arith.extf" not in mlir


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("1.25", "1.250000"),
        ("1.1", "1.099609"),
        ("-0.0", "-0.000000"),
        ("nan", "nan"),
        ("inf", "inf"),
        ("-inf", "-inf"),
    ),
)
def test_debug_print_f16_example_expected_spelling(text: str, expected: str) -> None:
    value = debug_print_example._f16(text)

    assert f"{value:.6f}" == expected


@pytest.mark.parametrize(
    ("dtype", "type_args"),
    (
        pytest.param("i8", (tla.Int8(-37), tla.Int8(5)), id="i8"),
        pytest.param("i16", (tla.Int16(-30000), tla.Int16(123)), id="i16"),
    ),
)
def test_debug_print_preserves_computed_narrow_signed_type(
    dtype: str, type_args: tuple[object, object]
) -> None:
    mlir = _computed_narrow_integer_kernel.dump_mlir(type_args=type_args)

    debug_line = next(line for line in mlir.splitlines() if "tla.debug_print" in line)
    assert "arith.addi" in mlir
    assert re.search(rf"(?:\({dtype}\)|: {dtype}(?:\s|$))", debug_line)


def test_debug_print_f32_literal_preserves_source_location() -> None:
    source_lines, first_lineno = inspect.getsourcelines(_f32_literal_location_kernel.fn)
    line = next(
        first_lineno + offset
        for offset, source in enumerate(source_lines)
        if "tla.print(1.25)" in source
    )
    lowered = BaseDSL()._lower(
        _f32_literal_location_kernel.fn,
        kind=_f32_literal_location_kernel.kind,
        options=dict(_f32_literal_location_kernel.options),
        type_args=(),
        location=_f32_literal_location_kernel.decorator_location,
    )
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            print_generic_op_form=True,
            assume_verified=False,
            enable_debug_info=True,
        )

    constant_line = next(
        source
        for source in mlir.splitlines()
        if "arith.constant" in source and "f32" in source
    )
    location_alias = re.search(r"loc\((#loc\d+)\)", constant_line)
    assert location_alias is not None
    location_id = location_alias.group(1)
    for _ in range(8):
        location_line = next(
            source
            for source in mlir.splitlines()
            if source.startswith(f"{location_id} =")
        )
        if f'"{__file__}":{line}:' in location_line:
            break
        location_alias = re.search(r"\((#loc\d+)\)", location_line)
        assert location_alias is not None
        location_id = location_alias.group(1)
    else:
        pytest.fail("f32 debug-print constant did not retain its source location")


@tla.kernel
def _debug_print_cube_static(value: object) -> None:
    with tla.cube():
        tla.print(value)


@tla.kernel
def _debug_print_cube_guarded_static(value: object) -> None:
    with tla.cube():
        if tla.arch.block_idx() == 0:
            tla.print(value)


@tla.kernel
def _debug_print_vector_static(value: object) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def _debug_print_vector_guarded_static(value: object) -> None:
    with tla.vector():
        if tla.arch.block_idx() == 0:
            tla.print(value)


_DEBUG_PRINT_MATRIX_KERNELS = {
    ("cube", False): _debug_print_cube_static,
    ("cube", True): _debug_print_cube_guarded_static,
    ("vector", False): _debug_print_vector_static,
    ("vector", True): _debug_print_vector_guarded_static,
}


@pytest.mark.parametrize("region", ("cube", "vector"))
@pytest.mark.parametrize(
    "dtype", ("i8", "i16", "i32", "u8", "u16", "u32", "f16", "f32")
)
@pytest.mark.parametrize("guarded", (False, True))
def test_debug_print_backend_matrix(region: str, dtype: str, guarded: bool) -> None:
    """Preserve canonical static debug-print region shapes."""

    scalar = {
        "i8": tla.Int8(-7),
        "i16": tla.Int16(-300),
        "i32": tla.Int32(7),
        "u8": tla.UInt8(255),
        "u16": tla.UInt16(65535),
        "u32": tla.UInt32(4294967295),
        "f16": tla.Float16(1.25),
        "f32": tla.Float32(1.25),
    }[dtype]
    kernel = _DEBUG_PRINT_MATRIX_KERNELS[(region, guarded)]
    mlir = kernel.dump_mlir(type_args=(scalar,))
    assert mlir.count("tla.debug_print") == 1
    expected_type = f"ui{dtype[1:]}" if dtype.startswith("u") else dtype
    assert expected_type in mlir


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((), {}, "expects at least one positional argument; got 0"),
        (
            (tla.Int32(1), tla.Int32(2), tla.Int32(3)),
            {},
            "format string must be a host Python str",
        ),
        ((), {"value": tla.Int32(1)}, "does not accept keyword arguments"),
        (("{}",), {"args": (tla.Int32(1),)}, "does not accept keyword arguments"),
        ((tla.Int32(1), 1), {}, "length is only valid"),
        ((2**31,), {}, "outside signless i32 range"),
        ((-(2**31) - 1,), {}, "outside signless i32 range"),
    ],
)
def test_debug_print_rejects_invalid_public_calls(
    args: tuple[object, ...], kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        tla.print(*args, **kwargs)


_SCALAR_ERROR = "expected one of f16, f32, i8, i16, i32, u8, u16, u32 scalar"
_FORMATTED_SCALAR_ERROR = "expected one of f16, f32, i8, i16, i32, u8, u16, u32 scalar"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("{",), "malformed format string"),
        (("}",), "malformed format string"),
        (("{:d}", tla.Int32(1)), "unsupported format field"),
        (("{:.2f}", tla.Float32(1.0)), "unsupported format field"),
        (("{0}", tla.Int32(1)), "unsupported format field"),
        (("{name}", tla.Int32(1)), "unsupported format field"),
        (("{!r}", tla.Int32(1)), "unsupported format field"),
        (("{:{}}", tla.Int32(1), tla.Int32(2)), "unsupported format field"),
        (("{} {}", tla.Int32(1)), "format argument count mismatch"),
        (("{}",), "format argument count mismatch"),
        (("{}", tla.Int32(1), tla.Int32(2)), "format argument count mismatch"),
        (("{} {} {} {} {} {} {} {} {}",) + (tla.Int32(1),) * 9, "at most 8"),
        (("x" * (1024 * 1024 - 24),), "debug FIFO limit"),
        (("bad\x00",), "embedded NUL"),
        (("snowman \u2603",), "ASCII"),
        (("{}", True), "unsupported value type bool"),
    ],
)
def test_debug_print_rejects_invalid_format_calls(
    args: tuple[object, ...], match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        tla.print(*args)


@pytest.mark.parametrize(
    ("kernel", "type_args", "match"),
    [
        *[
            (_typed_scalar_kernel, (value,), _SCALAR_ERROR)
            for value in (True, tla.Int64(1), tla.UInt64(1))
        ],
        *[
            (_formatted_typed_scalar_kernel, (value,), _FORMATTED_SCALAR_ERROR)
            for value in (tla.Int64(1), tla.UInt64(1))
        ],
        (_pointer_kernel, (), _SCALAR_ERROR),
        (
            _formatted_vector_value_kernel,
            (_host_vector_tensor(),),
            _FORMATTED_SCALAR_ERROR,
        ),
        (
            _formatted_tensor_arg_kernel,
            (_host_vector_tensor(),),
            "tensor arguments are unsupported in formatted print calls",
        ),
        (
            _regionless_kernel,
            (),
            r"must be nested inside tla\.cube\(\) or tla\.vector\(\)",
        ),
        (
            _regionless_formatted_kernel,
            (),
            r"must be nested inside tla\.cube\(\) or tla\.vector\(\)",
        ),
    ],
)
def test_debug_print_rejects_invalid_values_and_placement(
    kernel: object, type_args: tuple[object, ...], match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        kernel.dump_mlir(type_args=type_args)


def test_tensor_print_defers_effective_alignment_to_runtime() -> None:
    mlir = _tensor_kernel.dump_mlir(type_args=(_host_vector_tensor(),))

    assert "tla.print_tensor" in mlir


def test_debug_print_accepts_register_prefix() -> None:
    mlir = _vector_value_kernel.dump_mlir(type_args=(_host_vector_tensor(),))
    assert "tla.print_vector" in mlir
