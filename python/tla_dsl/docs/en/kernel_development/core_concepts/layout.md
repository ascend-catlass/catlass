---
nav_order: 30
---

# Layout

This document introduces the meaning of static and dynamic layout, how to make a Host tensor dynamic-layout, and how to program in the Kernel. For integration methods such as `from_dlpack`, see [Python Tensor Integration](tensor_binding.md); for interfaces such as `mark_*_dynamic`, see the [Program Lifecycle Reference](../../api/runtime/index.md).

---

## Overview

After a host-side framework tensor (such as PyTorch) is integrated, you get a `tla.Tensor` that can be used by `tla.compile` / launch. Besides the data pointer, the compiler also writes the tensor's **layout** — mainly the shape and stride of each dimension — into the compiled type of the kernel. Depending on whether these sizes are fixed at **compile time**, there are two categories:

- **Static layout**: shape / stride are concrete numbers at compile time (for example `(4, 8)`). The compiled artifact is specialized for this set of numbers; switching to another set of sizes usually requires recompilation.
- **Dynamic layout**: the selected dimensions are written as `?` in the type (meaning "not determined at compile time"), and the real lengths are filled in on each launch. The same compiled artifact can serve multiple concrete shapes.

By default `from_dlpack` writes only the concrete shape / stride of the **current** tensor, yielding a static layout; DLPack also does not describe "which dimensions are variable". To reuse a compiled artifact across shapes, you must explicitly mark dynamic dimensions after the conversion with `mark_layout_dynamic` / `mark_compact_shape_dynamic`: the corresponding dimensions become `?` in the compiled type, and the real sizes are filled in at launch.

The sections below first give a complete static-layout example, then explain how to mark and use a dynamic layout in the Kernel.

---

## Static Layout

By default `from_dlpack` writes the current concrete shape / stride / origin into the Host metadata; these numbers enter the compiled type and are known at compile time.

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(mem: tla.Tensor) -> None:
    # the concrete length is visible at compile time, for example shape<3>
    n = mem.origin_shape[0]
    # ...

torch.npu.set_device(0)

a = torch.arange(3, dtype=torch.float32, device="npu")
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
artifact = tla.compile(foo, ta, options="--npu-arch 3510")
artifact(ta, block_num=1)
```

In the example above, `from_dlpack` writes the length 3 into the static layout, and `tla.compile` also specializes for that size. If you later switch to a tensor of length 5 but still call this `artifact`, the compile-time type and the runtime size do not match, and the result is wrong or fails; normally you must **compile again** for the new size:

```python
b = torch.arange(5, dtype=torch.float32, device="npu")
tb = from_dlpack(b.contiguous(), layout_tag=tla.arch.RowMajor)
artifact_5 = tla.compile(foo, tb, options="--npu-arch 3510")
artifact_5(tb, block_num=1)
```

In other words, different static layouts (for example `(3):(1)` and `(5):(1)`) correspond to different compilation specializations. Static layout suits scenarios where the problem sizes are fixed and you want to make the most of constant information at compile time.

---

## Dynamic Layout

Calling `mark_layout_dynamic` / `mark_compact_shape_dynamic` after `from_dlpack` yields a dynamic layout. Both modify in place and return `self`, so they can be chained; they must be applied to the root Host tensor that covers the whole buffer with `coord` 0 in every dimension, and cannot be called on an already-sliced sub-view. The Kernel-side code is the same as for static: read sizes with `origin_shape[i]` etc.; the difference is that these values are filled in at runtime on dynamic dimensions.

| API | Effect |
|-----|------|
| `mark_layout_dynamic` | Whole layout: all shape/origin dynamic; stride dynamic except the leading dimension |
| `mark_compact_shape_dynamic` | Marks only one compact shape mode, and propagates the affected major strides |

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(mem: tla.Tensor) -> None:
    n = mem.origin_shape[0]
    # ...

torch.npu.set_device(0)

a = torch.rand(4, 8, dtype=torch.float32, device="npu")
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
# type illustration: shape<?,?>  stride<?,1>  origin<?,?>

artifact = tla.compile(foo, ta, options="--npu-arch 3510")
artifact(ta, block_num=1)

b = torch.rand(16, 32, dtype=torch.float32, device="npu")
tb = from_dlpack(b.contiguous(), layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
artifact(tb, block_num=1)  # the same compiled artifact can serve different concrete shapes
```

### `mark_layout_dynamic`

```python
Tensor.mark_layout_dynamic(leading_dim: int | None = None) -> Tensor
```

After the call:

- All **shape** elements and the corresponding **origin_shape** elements become dynamic;
- **stride**: the `leading_dim` dimension stays `1`, and the other dimensions become dynamic.

| Parameter | Meaning |
|------|------|
| `leading_dim` | The stride of this dimension is fixed to `1`. When `None`, it is inferred from `layout_tag`: `RowMajor` → innermost dimension; `ColumnMajor` → dimension 0; other tags derive the unit-stride dimension from the compact layout |

The concrete stride of the dimension corresponding to `leading_dim` must be `1`, otherwise an error is reported.

```python
t = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
t = t.mark_layout_dynamic()              # leading=1 automatically
t = t.mark_layout_dynamic(leading_dim=1) # or specify it explicitly

tb = from_dlpack(
    b.permute(1, 0).contiguous(), layout_tag=tla.arch.ColumnMajor
).mark_layout_dynamic()
# stride illustration: <1,?>
```

### `mark_compact_shape_dynamic`

```python
Tensor.mark_compact_shape_dynamic(
    mode: int,
    stride_order: tuple[int, ...] | None = None,
) -> Tensor
```

Marks **one** shape dimension (and the corresponding origin_shape element) as dynamic at a time; the strides for which this dimension is major (in a compact layout, the more outer dimensions whose stride product contains this dimension's length) are also marked dynamic.

| Parameter | Meaning |
|------|------|
| `mode` | the index of the dimension to make dynamic, in the range `[0, rank)` |
| `stride_order` | a permutation of dimension indices of length `rank`, **left to right = outer→inner** (the same kind as `torch.Tensor.dim_order()`). Dimensions to the left of `mode` are treated as outer, and marking `mode` dynamic also makes the strides of those outer dimensions dynamic. When `None`, it is derived automatically from the current concrete strides |

For a **given set of compact shapes/strides**, the correct outer→inner order is determined (by the stride sizes of the dimensions) and cannot be changed arbitrarily. Under common `layout_tag`s the 2D convention is fixed:

| layout | Typical stride | `stride_order` |
|--------|-------------|----------------|
| `RowMajor` | `(cols, 1)`, column dimension innermost | `(0, 1)` |
| `ColumnMajor` | `(1, rows)`, row dimension innermost | `(1, 0)` |

In general you should pass `None` and let the runtime derive it from the current strides. Pass it explicitly only when the derived result does not match expectations (for example multiple dimensions have stride 1 and the order is ambiguous), and it must match the real compact layout; passing a wrong order does not change the physical buffer, but marks the wrong stride dimension as dynamic. `stride_order` must be a permutation of `range(rank)`, otherwise an error is reported.

When multiple dimensions need to be dynamic: chain multiple calls, or switch to `mark_layout_dynamic()`.

```python
# row-major (4, 8):(8, 1) → derive stride_order=(0, 1)
t = from_dlpack(v.contiguous(), layout_tag=tla.arch.RowMajor)

t.mark_compact_shape_dynamic(mode=1)  # make the innermost dimension (column) dynamic → shape<4,?>  stride<?,1>
t.mark_compact_shape_dynamic(mode=0)  # make the outer dimension (row) dynamic → shape<?,8>  stride<8,1>

tv = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
tv = tv.mark_compact_shape_dynamic(0)  # 1D: only the length is dynamic

# the explicit form equivalent to the automatic result (row-major 2D)
t.mark_compact_shape_dynamic(mode=1, stride_order=(0, 1))
```

---

## Summary

In summary: static layout is compiled specialized to concrete numbers; dynamic layout uses `?` to describe variable dimensions, so one compilation can cover multiple concrete shapes.

| | Static layout | Dynamic layout |
|--|-------------|-------------|
| Host | `from_dlpack` | then `mark_*_dynamic` |
| Compiled type | concrete numbers | dynamic dimensions are `?` |
| Shape change | usually requires recompilation | the same artifact can be reused |
| Size source | hard-coded in the type at compile time | filled in with the current sizes at launch |
| Applies to | fixed problem scale | input shapes that change |

The choice between static and dynamic layout depends on whether the problem sizes are fixed and whether one compiled artifact needs to cover multiple shapes.
