from catlass import tla
from catlass.runtime import TlaIRNotExecutableError
import pytest


@tla.kernel
def _query_kernel(out: tla.Tensor):
    out[0] = tla.arch.sub_block_num()


def test_query_ir(compiler_tlair):
    out = tla.make_fake_tensor(tla.Int64, (1,), (1,), layout_tag=tla.arch.RowMajor)
    ir = compiler_tlair(_query_kernel, type_args=(out,))
    assert ir.count('"tla.arch.sub_block_num"() : () -> i64') == 1


def test_query_requires_device_context():
    with pytest.raises(TlaIRNotExecutableError, match="sub_block_num"):
        tla.arch.sub_block_num()


@pytest.mark.parametrize("mode", ["simd", "simt"])
def test_query_rejected_in_vec_func(mode):
    dims = (32, 1, 1) if mode == "simt" else None

    @tla.kernel
    def invalid():
        with tla.vector():
            with tla.vec.func(mode=mode, thread_block_dim=dims):
                tla.arch.sub_block_num()

    with pytest.raises(RuntimeError, match=r"sub_block_num.*outside tla.vec.func"):
        invalid.dump_mlir(type_args=())
