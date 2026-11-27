import pytest

from catlass import tla


@tla.kernel
def _get(lo: tla.Constexpr, hi: tla.Constexpr):
    with tla.vector():
        _ = tla.arch.get_ctrl(start_bit=lo, end_bit=hi)


@tla.kernel
def _raw(value: tla.Constexpr, lo: tla.Constexpr, hi: tla.Constexpr):
    with tla.vector():
        tla.arch.set_ctrl(value, start_bit=lo, end_bit=hi)


@tla.kernel
def _patch(value: tla.Constexpr):
    with tla.vector():
        tla.arch.set_ctrl(tla.params.CTRLValue(integer_saturation=value))


@pytest.mark.parametrize("lo,hi", [(0, 63), (6, 8), (9, 10), (45, 45), (60, 60)])
def test_field_read(lo, hi):
    assert "tla.arch.get_ctrl" in _get.dump_mlir(type_args=(lo, hi))


@pytest.mark.parametrize("lo,hi", [(0, 0), (6, 7), (56, 56), (63, 0), (False, 63)])
def test_invalid_field(lo, hi):
    with pytest.raises(Exception, match="arch.get_ctrl"):
        _get.dump_mlir(type_args=(lo, hi))


@pytest.mark.parametrize(
    "value,lo,hi", [(7, 6, 8), (3, 9, 10), (2, 48, 48), (0, 0, 63), (1, 56, 56)]
)
def test_invalid_write(value, lo, hi):
    with pytest.raises(Exception, match="arch.set_ctrl"):
        _raw.dump_mlir(type_args=(value, lo, hi))


@pytest.mark.parametrize("value", [True, False])
def test_explicit_patch_value(value):
    assert "tla.arch.set_ctrl" in _patch.dump_mlir(type_args=(value,))


@tla.kernel
def _runtime_raw(lo: tla.Constexpr, hi: tla.Constexpr):
    with tla.vector():
        value = tla.arch.get_ctrl(start_bit=lo, end_bit=hi)
        tla.arch.set_ctrl(value, start_bit=lo, end_bit=hi)


def test_runtime_raw_fields():
    for lo, hi in ((6, 8), (9, 10)):
        with pytest.raises(Exception, match="raw atomic fields require a compile-time integer"):
            _runtime_raw.dump_mlir(type_args=(lo, hi))
    assert "tla.arch.set_ctrl" in _runtime_raw.dump_mlir(type_args=(53, 53))


def test_empty_patch():
    with pytest.raises(Exception, match="at least one field"):
        _patch.dump_mlir(type_args=(None,))


@tla.kernel
def _semantic(patch: tla.Constexpr):
    with tla.cube():
        tla.arch.set_ctrl(patch)


@pytest.mark.parametrize(
    "field,value,lo,hi,encoded",
    [
        ("atomic_format", tla.params.AtomicFormat.BF16, 6, 8, 6),
        ("atomic_operation", tla.params.AtomicOperation.ADD, 9, 10, 0),
        ("fp8_as_hifloat8", True, 45, 45, 1),
        ("float_saturation", False, 48, 48, 1),
        ("ub_dcache", True, 49, 49, 1),
        ("fp8_nan_saturation", True, 50, 50, 0),
        ("integer_saturation", False, 53, 53, 0),
        ("conversion_mode", tla.params.ConversionMode.TRUNCATE, 59, 59, 1),
        ("saturation_strategy", tla.params.OverrideStrategy.USE_API, 60, 60, 0),
    ],
)
def test_semantic_encoding(field, value, lo, hi, encoded):
    patch = tla.params.CTRLValue(**{field: value})
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    for kernel, args in ((_semantic, (patch,)), (_raw, (encoded, lo, hi))):
        module = kernel.dump_mlir(type_args=args)
        assert module.count("tla.arch.set_ctrl") == 1
        assert f"mask = {mask} : i64" in module
        assert f"arith.constant {encoded << lo} : i64" in module


@tla.kernel
def _ordered(count: tla.Int32):
    with tla.vector():
        _ = tla.arch.get_ctrl()
        for _i in tla.range(count):
            tla.arch.set_ctrl(0, start_bit=53, end_bit=53)
            _ = tla.arch.get_ctrl()
        _ = tla.arch.get_ctrl()


def test_ordered_ctrl_survives_optimization():
    from catlass._mlir import passmanager
    from catlass.base_dsl import BaseDSL

    module = (
        BaseDSL()
        ._lower(
            _ordered.fn,
            kind=_ordered.kind,
            options=dict(_ordered.options),
            type_args=(tla.Int32(3),),
            location=_ordered.decorator_location,
        )
        .module
    )
    with module.context:
        passmanager.PassManager.parse(
            "builtin.module(canonicalize,cse,loop-invariant-code-motion)"
        ).run(module.operation)

    def ops_in(block):
        return [child.operation for child in block.operations]

    (kernel,) = (op for op in ops_in(module.body) if op.name == "tla.func")
    vector = next(
        op for op in ops_in(kernel.regions[0].blocks[0]) if op.name == "tla.vector"
    )
    siblings = ops_in(vector.regions[0].blocks[0])
    before, after = (op for op in siblings if op.name == "tla.arch.get_ctrl")
    (loop,) = (op for op in siblings if op.name == "scf.for")
    assert siblings.index(before) < siblings.index(loop) < siblings.index(after)
    body = ops_in(loop.regions[0].blocks[0])
    (inside,) = (op for op in body if op.name == "tla.arch.get_ctrl")
    (write,) = (op for op in body if op.name == "tla.arch.set_ctrl")
    assert body.index(write) < body.index(inside)


@pytest.mark.parametrize("in_vf", [False, True], ids=["bare", "vec_func"])
@pytest.mark.parametrize(
    "name,statement",
    [
        ("get_ctrl", "%v = tla.arch.get_ctrl {lo = 0 : i32, hi = 63 : i32}"),
        ("set_ctrl", "tla.arch.set_ctrl %bits {mask = 9007199254740992 : i64}"),
    ],
)
def test_ctrl_ir_requires_scalar_region(in_vf, name, statement):
    from catlass._mlir import ir

    if in_vf:
        statement = (
            '"tla.vector"() ({ "tla.vec.func"() ({ '
            + statement
            + ' }) {mode = "simd"} : () -> () }) : () -> ()'
        )
    with (
        ir.Context(),
        pytest.raises(ir.MLIRError, match=rf"tla\.arch\.{name}.*requires a scalar"),
    ):
        ir.Module.parse(
            "module { func.func @invalid() { %bits = arith.constant 0 : i64 "
            + statement
            + " return } }"
        )


@pytest.mark.parametrize(
    "statement,diagnostic",
    [
        (
            "%v = tla.arch.get_ctrl {lo = 56 : i32, hi = 56 : i32}",
            "requires a documented CTRL field or the full register",
        ),
        *[
            (
                f"tla.arch.set_ctrl %bits {{mask = {mask} : i64}}",
                "mask must select only complete documented CTRL fields",
            )
            for mask in (1 << 56, (1 << 56) | (1 << 60))
        ],
    ],
)
def test_unsupported_ctrl_field_ir(statement, diagnostic):
    from catlass._mlir import ir

    with ir.Context(), pytest.raises(ir.MLIRError, match=diagnostic):
        ir.Module.parse(
            'module { func.func @invalid() { "tla.vector"() ({ '
            '%bits = arith.constant 0 : i64 '
            + statement
            + ' }) : () -> () return } }'
        )
