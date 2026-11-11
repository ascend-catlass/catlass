import catlass.tla as tla


@tla.kernel
def outer_pair() -> None:
    with tla.prof_scope():
        with tla.cube():
            tla.pipe_barrier(tla.arch.CUBE)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)


print(outer_pair.dump_mlir())
