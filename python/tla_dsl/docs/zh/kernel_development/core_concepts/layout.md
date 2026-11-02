---
nav_order: 30
---

# Layout

本文介绍静态与动态 layout 的含义、如何把 Host tensor 设成动态 layout，以及在 Kernel 中如何编程。`from_dlpack` 等接入方式见 [Tensor 接入](tensor_binding.md)；`mark_*_dynamic` 等接口见 [程序生命周期参考](../../api/runtime/index.md)。

---

## 概述

Host 侧的框架张量（如 PyTorch）接入后，会得到可供 `tla.compile` / launch 使用的 `tla.Tensor`。除了数据指针，编译器还会把该 tensor 的 **layout**——主要是各维的 shape、stride——写进 kernel 的编译类型。按这些尺寸是否在**编译期**就固定，分成两类：

- **静态 layout**：shape / stride 在编译期就是具体数字（例如 `(4, 8)`）。编译产物针对这一组数字特化；换一组尺寸通常需要重新编译。
- **动态 layout**：选定维在类型里写作 `?`（表示“编译期不确定”），真实长度在每次 launch 时再填入。同一份编译产物可以服务多种具体 shape。

`from_dlpack` 默认只写入**当前**张量的具体 shape / stride，得到静态 layout；DLPack 也不描述“哪些维可变”。若要跨 shape 复用编译产物，须在转换之后用 `mark_layout_dynamic` / `mark_compact_shape_dynamic` 显式标记动态维：对应维在编译类型中变为 `?`，真实尺寸在 launch 时填入。

下文先给静态 layout 的完整示例，再说明如何标记并在 Kernel 中使用动态 layout。

---

## 静态 Layout

`from_dlpack` 默认把当前具体 shape / stride / origin 写进 Host 元数据；这些数字进入编译类型，在编译期已知。

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(mem: tla.Tensor) -> None:
    # 编译期可见具体长度，例如 shape<3>
    n = mem.origin_shape[0]
    # ...

torch.npu.set_device(0)

a = torch.arange(3, dtype=torch.float32, device="npu")
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
artifact = tla.compile(foo, ta, options="--npu-arch 3510")
artifact(ta, block_num=1)
```

上例中，`from_dlpack` 把长度 3 写进了静态 layout，`tla.compile` 也按该尺寸特化。若之后换成长度 5 的张量却仍调用这份 `artifact`，编译期类型与运行时尺寸对不上，结果会错误或失败；通常需要针对新尺寸 **再编译一次**：

```python
b = torch.arange(5, dtype=torch.float32, device="npu")
tb = from_dlpack(b.contiguous(), layout_tag=tla.arch.RowMajor)
artifact_5 = tla.compile(foo, tb, options="--npu-arch 3510")
artifact_5(tb, block_num=1)
```

也就是说：不同静态 layout（例如 `(3):(1)` 与 `(5):(1)`）对应不同的编译特化。静态 layout 适合问题尺寸固定、希望编译期尽量利用常量信息的场景。

---

## 动态 Layout

在 `from_dlpack` 之后调用 `mark_layout_dynamic` / `mark_compact_shape_dynamic`，即可得到动态 layout。二者均原地修改并返回 `self`，可链式调用；须作用于覆盖整块缓冲、各维 `coord` 为 0 的根 Host tensor，不能对已切片的子视图调用。Kernel 侧写法与静态相同：用 `origin_shape[i]` 等读取尺寸；区别在于这些值在动态维上是运行时填入的。

| API | 作用 |
|-----|------|
| `mark_layout_dynamic` | 整 layout：全部 shape/origin 动态；stride 除 leading 维外动态 |
| `mark_compact_shape_dynamic` | 只标一个 compact shape mode，并传播受影响的 major stride |

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
# 类型示意：shape<?,?>  stride<?,1>  origin<?,?>

artifact = tla.compile(foo, ta, options="--npu-arch 3510")
artifact(ta, block_num=1)

b = torch.rand(16, 32, dtype=torch.float32, device="npu")
tb = from_dlpack(b.contiguous(), layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
artifact(tb, block_num=1)  # 同一份编译产物可服务不同具体 shape
```

### `mark_layout_dynamic`

```python
Tensor.mark_layout_dynamic(leading_dim: int | None = None) -> Tensor
```

调用后：

- 所有 **shape** 元素与对应 **origin_shape** 元素变为动态；
- **stride**：`leading_dim` 维保持 `1`，其余维变为动态。

| 参数 | 含义 |
|------|------|
| `leading_dim` | 该维 stride 固定为 `1`。`None` 时按 `layout_tag` 推断：`RowMajor` → 最内维；`ColumnMajor` → 维 0；其它标签则按紧凑布局推导 unit-stride 维 |

`leading_dim` 对应维的具体 stride 必须为 `1`，否则报错。

```python
t = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
t = t.mark_layout_dynamic()              # 自动 leading=1
t = t.mark_layout_dynamic(leading_dim=1) # 或显式指定

tb = from_dlpack(
    b.permute(1, 0).contiguous(), layout_tag=tla.arch.ColumnMajor
).mark_layout_dynamic()
# stride 示意：<1,?>
```

### `mark_compact_shape_dynamic`

```python
Tensor.mark_compact_shape_dynamic(
    mode: int,
    stride_order: tuple[int, ...] | None = None,
) -> Tensor
```

一次只把 **一个** shape 维（及对应 origin_shape 元素）标成动态；对该维为 major 的 stride（紧凑布局下，stride 积中包含该维长度的更外层维）一并标成动态。

| 参数 | 含义 |
|------|------|
| `mode` | 要动态的维下标，范围 `[0, rank)` |
| `stride_order` | 长度为 `rank` 的维下标排列，**从左到右 = 外→内**（与 `torch.Tensor.dim_order()` 同类）。排在 `mode` 左侧的维视为外层，标 `mode` 动态时这些外层维的 stride 一并变动态。`None` 时由当前具体 stride 自动推导 |

对**给定的一套紧凑 shape/stride**，正确的外→内顺序是确定的（由各维 stride 大小决定），不能随意改。常见 `layout_tag` 下二维约定是固定的：

| layout | 典型 stride | `stride_order` |
|--------|-------------|----------------|
| `RowMajor` | `(cols, 1)`，列维最内 | `(0, 1)` |
| `ColumnMajor` | `(1, rows)`，行维最内 | `(1, 0)` |

一般应传 `None` 让运行时按当前 stride 推导。仅当推导结果不符合预期（例如多维 stride 同为 1、顺序有歧义）时才显式传入，且必须与真实紧凑布局一致；传入错误顺序不会改物理缓冲，但会标错该动态的 stride 维。`stride_order` 必须是 `range(rank)` 的排列，否则报错。

多维需要动态时：链式多次调用，或改用 `mark_layout_dynamic()`。

```python
# 行主 (4, 8):(8, 1) → 推导得到 stride_order=(0, 1)
t = from_dlpack(v.contiguous(), layout_tag=tla.arch.RowMajor)

t.mark_compact_shape_dynamic(mode=1)  # 动最内维(列) → shape<4,?>  stride<?,1>
t.mark_compact_shape_dynamic(mode=0)  # 动外层维(行) → shape<?,8>  stride<8,1>

tv = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
tv = tv.mark_compact_shape_dynamic(0)  # 一维：只动长度

# 与自动结果相同的显式写法（行主二维）
t.mark_compact_shape_dynamic(mode=1, stride_order=(0, 1))
```

---

## 小结

综上：静态 layout 按具体数字特化编译；动态 layout 用 `?` 描述可变维，同一份编译可覆盖多种具体 shape。

| | 静态 layout | 动态 layout |
|--|-------------|-------------|
| Host | `from_dlpack` | 再 `mark_*_dynamic` |
| 编译类型 | 具体数字 | 动态维为 `?` |
| 换 shape | 通常需重新编译 | 同一 artifact 可复用 |
| 尺寸来源 | 编译期写死在类型里 | launch 时填入当次尺寸 |
| 适用 | 问题规模固定 | 输入 shape 会变化 |

静态 layout 与动态 layout 的取舍，取决于问题尺寸是否固定，以及是否需要一份编译产物覆盖多种 shape。
