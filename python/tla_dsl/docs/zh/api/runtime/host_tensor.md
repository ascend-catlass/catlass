---
nav_order: 3
---

<!--
手工维护的中文 程序生命周期参考。
英文源：docs/en/api/runtime/host_tensor.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# Host Tensor 接入

[程序生命周期](index.md) | [API 速查](../quick_reference.md)

构造 Host 侧 `tla.Tensor`，并可将静态 layout 尺寸标为动态，使同一份编译产物可在不同 shape 下运行。详见 [静态与动态 Layout](../../kernel_development/core_concepts/layout.md)。

## 创建与绑定

用 `from_dlpack` 绑定真实 NPU 缓冲，或用 `make_fake_tensor` 造仅含元数据的类型样本。

### `tla.from_dlpack`

**源码：** [`catlass.tla.runtime.from_dlpack`](../../../../catlass/tla/runtime.py)

功能说明：

将 DLPack NPU tensor **零拷贝**绑定为 TLA Host tensor。返回对象与 `tensor_dlpack` 共享同一块设备缓冲。

函数原型：

```python
tla.from_dlpack(tensor_dlpack: object, *, layout_tag: Any, origin_shape: Any | None = None, assumed_align: int | None = None, stream: int | None = -1, element_type: type | None = None) -> _Tensor
```

参数说明：

- *`tensor_dlpack`*（`object`）：实现了 `__dlpack__()` 的对象。Ascend/NPU
  缓冲（如 `torch_npu`）得到可下发的 tensor；Host（CPU）缓冲（如 CPU 上的
  torch 张量）也可传入，但只取其元数据，返回的 tensor `data_ptr == 0`，
  仅可用于编译，不可下发。必填。
- *`layout_tag`*（`tla.arch.*`）：布局标签，如 `tla.arch.RowMajor`、
  `tla.arch.ColumnMajor`、`tla.arch.zN`。必填。
- *`origin_shape`*（`tuple | int | None`）：逻辑 origin，Python int 树。
  可选；省略时由 DLPack 物理 shape 与 `layout_tag` 推导。不是 Kernel 的
  `tla.make_shape`。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。
- *`stream`*（`int | None`）：传给 `__dlpack__(stream=...)`。默认 `-1`（不做流同步）。
  `None` 表示省略 `stream` 参数。
- *`element_type`*（`type | None`）：可选。覆盖从 DLPack 推导出的元素类型；
  默认 `None` 表示沿用 DLPack。当 DLPack 无法表达真实类型时使用（例如 fp8），
  传入 `tla.Float8E4M3FN` / `Float8E5M2`。须与导出缓冲的每元素位宽一致。

约束说明：

- Host（CPU）缓冲只绑定元数据，可在无设备的环境下编译 kernel；用它下发会抛异常。
- 所有权遵循 DLPack consumer 约定：capsule 会被消费，返回的 Host tensor
  销毁时调用 deleter；同时会保留对 `tensor_dlpack` 的引用，因此
  `from_dlpack(x.contiguous().to(device), ...)` 这类临时源是安全的。
- capsule 仅能消费一次；再次传入已消费的 capsule 会抛 `RuntimeTensorError`。
  需要再次绑定时请重新调用 `from_dlpack`。
- 二维 `RowMajor` 须先 `tensor.contiguous()`；二维 `ColumnMajor` 须先
  `tensor.permute(1, 0).contiguous()`。物理布局不符时抛 `RuntimeTensorError`。
  显式传入 `origin_shape` 则跳过该检查。
- 默认得到静态 layout。跨 shape 复用编译产物时再调用
  `mark_layout_dynamic` / `mark_compact_shape_dynamic`。
- 若指定 `element_type`，其每元素位宽须与导出的 DLPack 缓冲一致。

调用示例：

```python
tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(
    y.permute(1, 0).contiguous(),
    layout_tag=tla.arch.ColumnMajor,
)
```

---

### `tla.make_fake_tensor`

**源码：** [`catlass.tla.runtime.make_fake_tensor`](../../../../catlass/tla/runtime.py)

功能说明：

构造仅含元数据、不绑定设备缓冲的 Host tensor（`data_ptr == 0`）。
用于无需 NPU 时给 `tla.compile` 提供类型样本。真实缓冲请用 `from_dlpack`。

函数原型：

```python
tla.make_fake_tensor(dtype: Any, shape: Any, stride: Any, *, layout_tag: Any | None = None, addrspace: Any = AddressSpace.gm, origin_shape: Iterable[Any] | None = None, coord: Iterable[Any] | None = None, assumed_align: int | None = None) -> _Tensor
```

参数说明：

- *`dtype`*：元素类型，如 `tla.Float16` / `tla.Float32`。必填。
- *`shape`*（`int | tuple`）：逻辑 shape 树（zN 等物理布局用嵌套 tuple）。必填。
- *`stride`*（`int | tuple`）：stride 树，结构须与 `shape` 一致。必填。
- *`layout_tag`*：`tla.arch` 标签。可选，默认 `tla.arch.RowMajor`。
- *`addrspace`*：地址空间。可选，默认 `AddressSpace.gm`。
- *`origin_shape`*（`int | tuple | None`）：逻辑 origin。可选，默认等于 `shape`。
- *`coord`*（`int | tuple | None`）：坐标树。可选；省略时由 layout 推导（通常为零）。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。

约束说明：

- `shape` / `stride` / `origin_shape` / `coord` 须为 Python int 树，
  不能是 Kernel 侧 `tla.make_shape` / `tla.make_stride` / `tla.make_coord`。
- 始终未绑定，不能直接 launch；真实缓冲须改用 `from_dlpack`。
- 显式传入的 `shape` / `stride` 按原样使用（不做 layout remap）。

调用示例：

```python
fa = make_fake_tensor(tla.Float16, (128, 64), (64, 1))
fzn = make_fake_tensor(
    tla.Float16,
    ((16, 2), (16, 4)),
    ((16, 256), (1, 512)),
    layout_tag=tla.arch.zN,
    origin_shape=(32, 64),
)
```

---

## 动态 Layout

将静态 layout 尺寸标为动态。详见 [静态与动态 Layout](../../kernel_development/core_concepts/layout.md)。

### `tla.Tensor.mark_layout_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_layout_dynamic`](../../../../catlass/tla/runtime.py)

功能说明：

将所有 shape 维标为动态，使一份 artifact 可接受不同 extents。
stride 除 leading 维（保持 `1`）外均变为动态；广播 stride `0` 保留。
对应的 `origin_shape` 叶节点也变为动态，编译类型不再依赖具体 DLPack 尺寸。

函数原型：

```python
tla.Tensor.mark_layout_dynamic(leading_dim: int | None = None) -> '_Tensor'
```

参数说明：

- *`leading_dim`*（`int | None`）：stride 为 `1` 的 leading 维索引。
  可选，默认 `None`（由 `layout_tag` 或紧凑 stride 顺序推断）。

约束说明：

- 原地修改并返回 `self`（可链式调用）。
- 各 `coord` 叶节点必须为 `0`；切片子视图会失败。
- `leading_dim` 对应维的 stride 必须为 `1`。
- NZFamily 布局下，每组两个物理 shape 叶节点对应一个逻辑 `origin_shape` 轴。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_layout_dynamic()
artifact = tla.compile(my_kernel, ta, options="--npu-arch 3510")
```

---

### `tla.Tensor.mark_compact_shape_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_compact_shape_dynamic`](../../../../catlass/tla/runtime.py)

功能说明：

将指定的一个紧凑 shape 维（`mode`）标为动态。以该维为因子的 major 维 stride
也会变为动态。对应的 `origin_shape` 叶节点同步标记，编译类型不再依赖具体尺寸。

函数原型：

```python
tla.Tensor.mark_compact_shape_dynamic(mode: int, stride_order: tuple[int, ...] | None = None) -> '_Tensor'
```

参数说明：

- *`mode`*（`int`）：要标记为动态的扁平 shape 叶节点索引（从 0 开始）。必填。
- *`stride_order`*（`tuple[int, ...] | None`）：紧凑 stride 顺序（外层 → 内层）。
  可选；省略时由当前 stride 推断。

约束说明：

- 原地修改并返回 `self`。
- 各 `coord` 叶节点必须为 `0`。
- `stride_order` 须为 `range(rank)` 的一个排列。
- NZFamily 布局下，物理维 0/1 对应逻辑 M，物理维 2/3 对应逻辑 N。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_compact_shape_dynamic(mode=0)
```
