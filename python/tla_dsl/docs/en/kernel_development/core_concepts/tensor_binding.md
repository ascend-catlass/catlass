---
nav_order: 40
---

# Tensor Integration

This document explains how to obtain a `tla.Tensor` on the Host side that can be used by `tla.compile` / launch: bind a real buffer with [DLPack](https://github.com/dmlc/dlpack)'s `from_dlpack` (`torch` / `torch_npu`), or build a type sample without a device pointer with `make_fake_tensor`. For compilation, caching, and repeated launches, see [Compile and Launch](compile_and_launch.md). For static / dynamic layout, see [Layout](layout.md). For the complete interface description, see the [Program Lifecycle Reference](../../api/runtime/index.md).

---

## Explicit Conversion with `from_dlpack`

The runtime provides an interface for converting a DLPack-compatible tensor into a `tla.Tensor`. Below is the path from torch, through `from_dlpack` binding, all the way to `compile` / launch:

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(src: tla.Tensor, dst: tla.Tensor) -> None:
    n = src.origin_shape[0]
    # ... complete the computation inside the vector/cube region ...

torch.npu.set_device(0)

x = torch.rand(1024, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = torch.zeros_like(x)

tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(y.contiguous(), layout_tag=tla.arch.RowMajor)
# call mark_*_dynamic only when the compiled artifact must be reused across shapes
tx = tx.mark_compact_shape_dynamic(0)
ty = ty.mark_compact_shape_dynamic(0)

compiled = tla.compile(foo, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
torch.npu.synchronize()
```

How the compiled artifact is cached, how the first launch differs from repeated launches, and how to reuse the same compiled artifact on multiple devices are described in [Compile and Launch](compile_and_launch.md).

Here `x` must implement `__dlpack__` (usually a `torch` tensor on the NPU, for example via `torch_npu`). The conversion is **zero-copy**: the returned `tla.Tensor` shares the same device memory as the source tensor; the source tensor's lifetime must cover the subsequent `tla.compile` / launch, otherwise the pointer becomes invalid.

By default you get a **static layout** (the concrete shape / stride / origin are written into the compiled type). To get a dynamic or mixed static/dynamic layout, call `mark_layout_dynamic` / `mark_compact_shape_dynamic` after the conversion (see [Layout](layout.md)).

The full signature is as follows:

```python
def from_dlpack(
    tensor_dlpack: object,
    *,
    layout_tag: Any,
    origin_shape: Any | None = None,
    assumed_align: int | None = None,
    stream: int | None = -1,
) -> Tensor:
```

| Parameter | Description |
|------|------|
| `tensor_dlpack` | an object implementing `__dlpack__`; must be a buffer on an **Ascend/NPU device**; CPU / NumPy are not supported |
| `layout_tag` | **Required**. A `tla.arch` layout tag: 1D `RowMajor`; 2D linear `RowMajor` / `ColumnMajor`; and `zN` / `nZ` / `L0Clayout` / `zNUnAlign` (logical 2D, with shape/stride as nested quadruples) |
| `origin_shape` | Optional. The logical origin, a plain `tuple` (such as `(M, K)`). When omitted, it is derived from the framework tensor's shape and `layout_tag` |
| `assumed_align` | **Reserved parameter**. The value passed currently does not affect actual behavior |
| `stream` | Passed to the framework's `__dlpack__(stream=...)`; it only affects whether the framework side performs stream synchronization during conversion; the default `-1` means no synchronization |

### Layout and Physical Storage

When `origin_shape` is not passed, 2D `RowMajor` / `ColumnMajor` must satisfy the following physical layout conventions:

| `layout_tag` | Host-side preparation |
|--------------|-------------|
| `RowMajor` | `tensor.contiguous()` (physically row-major) |
| `ColumnMajor` | `tensor.permute(1, 0).contiguous()` (logically column-major, physically still a row-major compact buffer) |

A `RuntimeTensorError` is raised when the physical layout is inconsistent with `layout_tag`.

If `origin_shape` is passed explicitly, that logical origin is used and the physical layout is no longer validated against the table above. Under nested layouts such as `zN`, the logical origin is 2D `(M, N)`, and the corresponding shape / stride are nested quadruples.

### Code Example

The following demonstrates how to convert a PyTorch tensor into a `tla.Tensor` with `from_dlpack` and inspect the result:

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

torch.npu.set_device(0)
x = torch.rand(30, 20, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)

print(y.shape)         # logical shape
print(y.stride)        # stride tree
print(y.origin_shape)  # logical origin
print(y.layout_tag)    # e.g. 'RowMajor'
print(y.dtype)         # e.g. 'f32'
print(y.addrspace)     # e.g. 'gm'
print(y.data_ptr)      # device address
print(y)               # compiled type string
```

After the conversion, the tensor information can be inspected through the following attributes:

- `tensor.shape`: logical shape
- `tensor.stride`: stride
- `tensor.origin_shape`: logical origin
- `tensor.layout_tag`: layout tag
- `tensor.dtype`: element type
- `tensor.addrspace`: address space
- `tensor.data_ptr`: device pointer

---

## Building a Fake Tensor Without a ptr with `make_fake_tensor`

Use `make_fake_tensor` when you only need to provide a type / layout sample to `tla.compile` without binding a real NPU buffer. It always returns an unbound Host tensor (`data_ptr == 0`); for a real buffer use `from_dlpack`.

Usage: `(dtype, shape, stride)` are required; `layout_tag` is optional and defaults to `RowMajor`. An explicitly passed shape/stride is used as is.

```python
import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

fake = make_fake_tensor(
    tla.Float32,
    (64, 128),
    (128, 1),
)
assert fake.data_ptr == 0
assert fake.layout_tag == "RowMajor"

# nested layouts such as zN: pass nested shape/stride yourself (the logical origin is still 2D)
zn = make_fake_tensor(
    tla.Float16,
    ((16, 2), (16, 4)),
    ((16, 256), (1, 512)),
    layout_tag=tla.arch.zN,
    origin_shape=(32, 64),
)
```
