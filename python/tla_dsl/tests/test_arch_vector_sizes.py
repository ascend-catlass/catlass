import catlass.tla as tla
from catlass import _tla_type_bridge
from catlass._mlir import ir


def test_vector_sizes_on_host() -> None:
    assert tla.arch.get_vec_len_in_bytes() == 256
    assert tla.arch.get_datablock_size_in_bytes() == 32


@tla.kernel
def vector_blocks_range_kernel() -> None:
    blocks = tla.arch.get_vec_len_in_bytes() // tla.arch.get_datablock_size_in_bytes()
    for i in tla.range(blocks):
        tla.make_coord(i, 0)


def test_vector_sizes_fold_to_constant_in_kernel() -> None:
    mlir = vector_blocks_range_kernel.dump_mlir()
    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(mlir)
        loops = []

        def find_loops(op):
            if op.operation.name == "scf.for":
                loops.append(op.operation)
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        find_loops(child)

        find_loops(module.operation)
        assert len(loops) == 1
        # scf.for operands start with the lower bound, upper bound, and step.
        upper_bound = loops[0].operands[1]
        constant = upper_bound.owner
        assert isinstance(constant, ir.Operation)
        assert constant.name == "arith.constant"
        assert ir.IntegerAttr(constant.attributes["value"]).value == 8
