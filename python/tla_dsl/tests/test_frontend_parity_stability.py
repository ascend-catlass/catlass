from __future__ import annotations

from pathlib import Path

import catlass.tla as tla
import catlass.execution as execution


@tla.kernel
def fake_kernel() -> None:
    x = tla.make_shape(100, 200)
    _ = x


def test_full_demo_style_kernel_lowering_emits_mlir(compiler_tlair) -> None:
    mlir = compiler_tlair(fake_kernel)
    assert 'sym_name = "fake_kernel"' in mlir
    assert '"tla.make_shape"' in mlir
    assert '"tla.return"' in mlir


def _test_compiled_function(*, cache_key: str) -> tla.JitCompiledFunction:
    return execution._new_jit_compiled_function(
        cache_key=cache_key,
        cache_dir=Path("/tmp/cache"),
        tlair_mlir="module {}",
        lowered_llvm="; llvm",
        entrypoint="fake_kernel",
        compiler_bridge_path=Path("/tmp/_tla_type_bridge_native.so"),
        hivmc_path=Path("/tmp/hivmc-a5"),
        kernel_binary_path=Path("/tmp/kernel.o"),
        kernel_abi=object(),
        abi_packer=object(),
        runtime_metadata=execution.KernelRuntimeMetadata("fake_kernel"),
        compile_option=execution.TlaCompileOption(),
        pass_ir_dump="",
    )


def test_full_demo_style_compile_routes_kernel(monkeypatch) -> None:
    compiled = _test_compiled_function(cache_key="parity")
    calls: list[tuple[str, str]] = []

    def fake_compile(
        fn, *, kind, options, type_args=None, decorator_location=None, **kwargs
    ):
        del options, type_args, decorator_location, kwargs
        calls.append((fn.__name__, kind))
        return compiled

    monkeypatch.setattr(execution, "_compile_kernel", fake_compile)
    result = tla.compile(fake_kernel)

    assert isinstance(result, tla.JitCompiledFunction)
    assert result is compiled
    assert result.artifacts.MLIR == "module {}"
    assert result.artifacts.LLVM == "; llvm"
    assert result.artifacts.BINARY == Path("/tmp/kernel.o")
    assert result.cache_key == "parity"
    assert not hasattr(result, "artifact")
    assert calls == [("fake_kernel", "kernel")]
