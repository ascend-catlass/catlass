from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass import execution
from catlass.base_dsl.runtime import ascend_stream_adapter
from catlass.tla.runtime import make_fake_tensor


def _capacity_kernel(kind, calls):
    @tla.kernel
    def kernel(source: tla.Tensor):
        with tla.vector():
            if kind == "mixed":
                tla.print("input")
            elif kind == "vf":
                with tla.vec.func(mode="simd"):
                    tla.print("inside VF")
            tla.print(source, 200000)
            if calls == 2:
                tla.print(source, 200000)

    return kernel


@pytest.mark.parametrize(
    "kind,calls,rejected",
    [("tensor", 1, False), ("tensor", 2, True), ("mixed", 2, False), ("vf", 2, False)],
)
def test_capacity_policy_reaches_real_executor_without_device(
    monkeypatch, kind, calls, rejected
):
    # Each record fits, but two records exceed the 1 MiB per-core FIFO.
    source = make_fake_tensor(
        tla.Float32,
        (200000,),
        (1,),
        origin_shape=(200000,),
        layout_tag=tla.arch.RowMajor,
    )
    compiled = _capacity_kernel(kind, calls).compile(
        type_args=(source,), options="--npu-arch 3510"
    )
    source._external_binding = True
    source.data_ptr = 0x1000

    class ReachedLaunch(Exception):
        pass

    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        raise ReachedLaunch

    monkeypatch.setattr(ascend_stream_adapter, "current_device", lambda: 0)
    monkeypatch.setattr(ascend_stream_adapter, "current_stream", lambda device: 0)
    monkeypatch.setattr(execution, "load_acl", lambda: None)
    monkeypatch.setattr(execution, "load_binary", lambda **kwargs: (1, 2))
    monkeypatch.setattr(execution, "launch_kernel", launch)
    if rejected:
        with pytest.raises(execution.TlaExecutionError, match="statically executed"):
            compiled(source, block_num=1)
        assert not launches
    else:
        # Stop at the launch boundary: fake addresses are never sent to a device.
        with pytest.raises(ReachedLaunch):
            compiled(source, block_num=1)
        assert len(launches) == 1
