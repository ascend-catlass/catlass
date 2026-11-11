import sys

import catlass.tla as tla


@tla.kernel
def mixed_markers(selected: tla.Constexpr[int]) -> None:
    with tla.cube():
        if tla.const_expr(selected == 0):
            with tla.prof_scope():
                tla.pipe_barrier(tla.arch.CUBE)
        else:
            tla.pipe_barrier(tla.arch.CUBE)
    with tla.vector():
        p = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        t = tla.make_tensor(p, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
        if tla.const_expr(selected == 1):
            tla.prof_start()
        with tla.vec.func(mode="simd"):
            x = t.load()
            if tla.const_expr(selected == 2):
                tla.prof_start()
            y = tla.add(x, x)
            if tla.const_expr(selected == 2):
                tla.prof_end()
            t.store(y)
        if tla.const_expr(selected == 1):
            tla.prof_end()
        with tla.vec.func(mode="simd"):
            t.store(t.load())


print(mixed_markers.dump_mlir(type_args=(int(sys.argv[1]),)))
