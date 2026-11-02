---
nav_order: 45
---

# Compile and Launch

This document describes the behavior observable on the Host side when using `tla.compile` and launching a kernel, including compile samples,
the compilation cache, repeated launches, and stream selection. For how to create a Host tensor, see
[Python Tensor Integration](tensor_binding.md); for the complete interface signatures, see the
[Program Lifecycle Reference](../../api/runtime/index.md).

For how to pass composite arguments such as tuple/list and dataclass, see [Structured Kernel Arguments](structured_arguments.md).

---

## Basic Flow

First compile using Host tensors as type samples, then launch with tensors bound to real NPU buffers:

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# vadd is a function decorated with @tla.kernel.
torch.npu.set_device(0)
a = torch.rand(64, dtype=torch.float32, device="npu")
b = torch.rand(64, dtype=torch.float32, device="npu")
out = torch.zeros(64, dtype=torch.float32, device="npu")

ta = from_dlpack(a, layout_tag=tla.arch.RowMajor)
tb = from_dlpack(b, layout_tag=tla.arch.RowMajor)
tout = from_dlpack(out, layout_tag=tla.arch.RowMajor)

compiled = tla.compile(vadd, ta, tb, tout, options="--npu-arch 3510")
compiled(ta, tb, tout, block_num=1)
torch.npu.synchronize()
```

The arguments to `tla.compile` determine the compiled type of the kernel. The compile samples can be
real tensors bound with `from_dlpack`, or `make_fake_tensor` without a device pointer; at launch you must pass
tensors bound to real NPU buffers.

The default layout is static, and information such as shape and stride participates in compilation. If the same compiled artifact
must handle different shapes, properly mark the dynamic dimensions on the Host tensors used for compilation and launch; see
[Static and Dynamic Layout](layout.md).

---

## Objects the User Needs to Know

The following objects are encountered during compilation and launch:

| Object | How to obtain it | How the user uses it |
|------|----------|--------------|
| `TlaJitFunction` | obtained after a Python function is decorated with `@tla.kernel` | as the first argument of `tla.compile`; usually does not need to be constructed manually |
| `JitCompiledFunction` | the return value of `tla.compile(...)` | call it directly to launch the kernel; the first call lazily creates the executor, and later calls reuse it |
| `JitExecutor` | created internally by `JitCompiledFunction` on its first launch | carries the loaded binary and launch state; the user does not need to construct or call it directly |
| `TlaExecutionResult` | returned after calling `compiled(...)` | inspect the run information of this launch; computation results are still written to the kernel's output tensors |

### `JitCompiledFunction`

`JitCompiledFunction` represents the result of one `tla.compile`. It holds device-independent compiled artifacts,
but when it is first returned it has not yet created an executor or loaded a binary.

Common user operations include:

- `compiled(*args, block_num=...)`: launch using the internal executor; the executor is created only on the first call,
  and later calls reuse it;
- `compiled.cache_key`: inspect the cache key for this compilation;
- `compiled.kernel_binary_path`: inspect the device binary path;
- `compiled.artifacts`: inspect compilation artifact information such as MLIR, backend IR, and cache directory during debugging.

Every call to `tla.compile(...)` returns a new `JitCompiledFunction`. Even if the compilation cache is hit, the new
object does not inherit the executor or launch state already created by the old object.

### `JitExecutor`

`JitExecutor` is the execution object used internally by `JitCompiledFunction`. On the first call to `compiled(...)`
it loads the binary; later calls reuse the same executor and loaded execution state.

```python
result = compiled(ta, tb, tout, block_num=1)
```

The user does not need to obtain or call `JitExecutor` directly. Repeated launches, argument passing, and stream selection are all done through
`JitCompiledFunction`.

Objects such as `JitModule` and `ExecutionArgs` carry the launch information prepared by the framework. Ordinary users do not need to directly
construct or modify them; use `tla.compile(...)` for compilation, and call the return value directly at launch.

---

## Compilation Cache

The compilation cache is enabled by default. Every call to `tla.compile` first generates the compilation
representation from the kernel and the compiled type samples, and computes a cache key from it. The main contents affecting the cache key include:

- the kernel code, and compile-time inputs such as static shape and `constexpr`;
- the kernel mode, target architecture, and compilation options;
- the compilation toolchain and runtime ABI version;
- external Ascend C source files used by the kernel.

Device pointers, input data, `block_num`, and stream are launch state and do not enter the cache key. Therefore, merely
switching input buffers does not trigger recompilation; changing the static shape, `constexpr`, target architecture, or kernel code
usually produces a new key.

The cache is looked up in the following order:

1. **In-process cache**: when the key is the same in the same process and the compiled artifacts still exist, reuse them directly.
2. **Disk cache**: when there is no in-process hit, check the binary and its descriptor information already persisted in the cache directory.
3. **Recompile**: when neither cache level hits, run the backend compilation and update the disk and in-process caches.

Missing files or an expired cache version are treated as a miss; a corrupt cache descriptor or an invalid launch ABI reports an error and does not silently
use an untrusted old artifact.

Whether or not the cache is hit, every explicit call to `tla.compile(...)` returns a new
`JitCompiledFunction`:

| Reused | Not reused |
|------|------|
| the compiled device binary and its launch descriptor | previously created executor |
| compilation diagnostics and artifact information in the cache | loaded execution state and stream |

In other words, a cache hit only saves the repeated backend compilation; it does not carry the executor or stream used by the previous launch
into the new `JitCompiledFunction`.

Common cache switches are listed below; for the complete list see [Environment Variables](env_vars.md):

| Environment variable | Effect |
|------|------|
| `CATLASS_DSL_CACHE=0` | disable the compilation cache |
| `CATLASS_DSL_CACHE_DIR=/path/to/cache` | specify the disk cache directory |
| `CATLASS_DSL_FORCE_RECOMPILE=1` | ignore the existing cache and force recompilation |

---

## First Launch and Repeated Launch

| Call pattern | Behavior |
|------|------|
| first call to `compiled(...)` | lazily create the executor and load the binary |
| another call to the same `compiled(...)` | reuse the executor and the loaded execution state |

For repeated launches, reuse `compiled` directly:

```python
compiled(ta, tb, tout, block_num=1)
compiled(ta2, tb2, tout2, block_num=1)
```

---

## Selecting a Stream

When `stream=` is not passed explicitly, the executor queries the current NPU stream on its device at **every launch**.
Therefore the same executor can follow changes in the stream context:

```python
with torch.npu.stream(stream_a):
    compiled(ta, tb, tout, block_num=1)

with torch.npu.stream(stream_b):
    compiled(ta2, tb2, tout2, block_num=1)
```

When `stream=` is passed explicitly, that argument takes precedence. A stream change does not trigger recompilation and does not create a new
executor.

---

## Usage Recommendations

- Repeated launches: reuse the same `compiled(...)` directly.
- Different streams: reuse the same `compiled` and switch the stream context before launching.
- Type samples only: compile with `make_fake_tensor` and switch to real bound tensors at launch.
- Reuse across shapes: properly mark the dynamic layout; do not rely on the cache ignoring static shape differences.
