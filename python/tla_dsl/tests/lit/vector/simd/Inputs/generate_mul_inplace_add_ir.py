"""Emit the same masked mul_inplace_add for each supported MLIR element type."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "dtype", choices=("f16", "bf16", "f32", "i16", "ui16", "i32", "ui32", "i64", "ui64")
)
dtype = parser.parse_args().dtype
bits = 16 if dtype == "bf16" else int(dtype.lstrip("fiu"))
lanes = 2048 // bits

print(
    f"""!ub = memref<{lanes}x{dtype}, #hivm.address_space<ub>>
!tile = !tla.tensor<!tla.layout<!tla.shape<{lanes}>, !tla.stride<1>, !tla.shape<{lanes}>, RowMajor>, !tla.coord<0>, !tla.ptr<{dtype}, ub, 32>>
!vec = !tla.vector<{lanes}x{dtype}>
!mask = !tla.mask<{lanes}>

func.func @mul_inplace_add(%am: !ub, %bm: !ub, %cm: !ub, %om: !ub) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {lanes} : index
  %acc = tla.tensor_desc %am shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : !ub -> !tile
  %lhs = tla.tensor_desc %bm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : !ub -> !tile
  %rhs = tla.tensor_desc %cm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : !ub -> !tile
  %out = tla.tensor_desc %om shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : !ub -> !tile
  "tla.vec.func"() ({{
    %a = tla.load %acc : !tile -> !vec
    %b = tla.load %lhs : !tile -> !vec
    %c = tla.load %rhs : !tile -> !vec
    %m = "tla.create_mask"() {{pattern = "H", dtype = {dtype}}} : () -> !mask
    %result = tla.mul_inplace_add %a, %b, %c mask %m : !vec, !vec, !vec mask !mask -> !vec
    tla.store %out, %result : !tile, !vec
  }}) : () -> ()
  return
}}"""
)
