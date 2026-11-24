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

用 `from_dlpack`，或 Host 双模式 `tla.make_ptr` / `tla.make_layout` / `tla.make_tensor` 绑定真实 NPU 缓冲；用 `make_fake_tensor` 造仅含元数据的类型样本。

### `tla.from_dlpack`

**源码：** [`catlass.tla.runtime.from_dlpack`](../../../../catlass/tla/runtime.py)

功能说明：

将 DLPack 缓冲区绑定为 TLA Host tensor。NPU 缓冲区采用零拷贝共享，CPU 缓冲区提供编译元数据。

函数原型：

```python
tla.from_dlpack(tensor_dlpack: object, *, layout_tag: Any, origin_shape: Any | None = None, assumed_align: int | None = None, stream: int | None = -1, element_type: type | None = None) -> _Tensor
```

参数说明：

- *`tensor_dlpack`*（`object`）：实现了 `__dlpack__()` 的对象。Ascend/NPU
  缓冲（如 `torch_npu`）得到可下发的 tensor；Host（CPU）缓冲（如 CPU 上的
  torch 张量）也可传入，但只取其元数据，返回的 tensor `data_ptr == 0`，
  仅可用于编译，不可下发。必填。
- *`layout_tag`*（`tla.arch.*`）：必填的布局标签，如 `tla.arch.RowMajor`、
  `tla.arch.ColumnMajor`、`tla.arch.zN` 或 `tla.arch.nZ`。四维预打包 NZ 视图的形状
  和步长必须与指定的布局一致。
- *`origin_shape`*（`tuple | int | None`）：逻辑 origin，Python int 树。
  线性布局省略时由 DLPack 物理 shape 与 `layout_tag` 推导。一维预打包 NZ 输入需要
  `(M, N)`。四维预打包 NZ 视图默认使用填充后的尺寸，可传入 `(M, N)` 保留填充前的
  逻辑尺寸。GM MxScale 省略时可从二维输入推导。CPU 编译样本可用 `None` 表示动态维，
  四维视图仍会校验导出的实际布局。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。
- *`stream`*（`int | None`）：传给 `__dlpack__(stream=...)`。默认 `-1`（不做流同步）。
  `None` 表示省略 `stream` 参数。
- *`element_type`*（`type | None`）：可选，覆盖从 DLPack 推导出的元素类型。默认 `None`，
  沿用 DLPack 类型。DLPack 无法表达实际类型（如 FP8）时，可传入 `tla.Float8E4M3FN` 或
  `Float8E5M2`。存储位宽须一致；`Float4E1M2` / `Float4E2M1` / `Int4` 按每个整型字节
  打包两个元素时除外。`Bool` 占一个字节。

约束说明：

- Host（CPU）缓冲只绑定元数据，可在无设备的环境下编译 kernel；用它下发会抛异常。
- 所有权遵循 DLPack consumer 约定：capsule 会被消费，返回的 Host tensor
  销毁时调用 deleter；同时会保留对 `tensor_dlpack` 的引用，因此
  `from_dlpack(x.contiguous().to(device), ...)` 这类临时源是安全的。
- capsule 仅能消费一次；再次传入已消费的 capsule 会抛 `RuntimeTensorError`。
  需要再次绑定时请重新调用 `from_dlpack`。
- DLPack 导出省略步长时，按 C 连续布局推导步长。
- 二维 `RowMajor` 须先 `tensor.contiguous()`；二维 `ColumnMajor` 须先
  `tensor.permute(1, 0).contiguous()`。物理布局不符时抛 `RuntimeTensorError`。
  显式传入 `origin_shape` 则跳过该检查。
- 预打包 NZ 布局接受一维连续缓冲区（例如 `host_nd2nz` 的输出）或按物理叶顺序
  `(m0, m1, n0, n1)` 导出的四维 DLPack 视图。一维容量须至少达到填充后的存储长度。
  四维视图的 shape 和 stride 必须匹配布局，长度为 1 的维度忽略 stride。
  四维视图要求导出元素和 TLA 元素的存储位宽一致。按字节导出的 FP4 使用一维形式。
- GM MxScale 接受覆盖描述符全部寻址范围的稠密 scale 缓冲区，容量包含 scale 元素
  成对填充所占空间。若 DLPack 维度无法表示逻辑 scale 矩阵，需传入 `origin_shape`。
- 默认得到静态布局。将预打包 NZ 张量作为 GM 参数传给 kernel 时，编译前须调用
  `mark_layout_dynamic()`。线性布局可以用 `mark_layout_dynamic()` 或
  `mark_compact_shape_dynamic()` 将维度设为动态。

调用示例：

```python
tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(
    y.permute(1, 0).contiguous(),
    layout_tag=tla.arch.ColumnMajor,
)
t_zn = from_dlpack(
    zn_buf,
    layout_tag=tla.arch.zN,
    origin_shape=(M, N),
)
# zn_f32 是 32*64 FP32 矩阵的一维预打包缓冲区。
zn_view = zn_f32.as_strided(
    size=(16, 2, 8, 8),
    stride=(8, 128, 1, 256),
)
t_zn_view = from_dlpack(zn_view, layout_tag=tla.arch.zN)
t_zn_view.mark_layout_dynamic()
t_nz = from_dlpack(
    nz_buf,
    layout_tag=tla.arch.nZ,
    origin_shape=(M, N),
)
```

---

### `tla.make_tensor`

**源码：** [`catlass.core_api.make_tensor`](../../../../catlass/core_api.py)

功能说明：

已知缓冲区的地址和布局时，可在 Host 侧调用 `tla.make_tensor(ptr, layout)` 直接绑定。
`ptr` 与 `layout` 分别通过 `tla.make_ptr` 和 `tla.make_layout` 构造。
在 kernel 内调用 `tla.make_tensor` 会生成 `!tla.tensor` IR。

函数原型：

```python
tla.make_tensor(ptr, layout, coord=None, *, loc=None) -> Tensor
```

参数说明：

- *`ptr`*：Host 上 ``tla.make_ptr(dtype, address, mem_space)`` 的返回值（同 ``Pointer`` 类型）。
- *`layout`*：Host 上 ``tla.make_layout(make_shape(...), make_stride(...), layoutTag=...)`` 的返回值（同 ``Layout`` 类型）。
- *`coord`*：可选；Host ``tla.make_coord`` 或 int 树；省略时由 layout 推导。
- *`loc`*：Kernel 路径用；Host 路径忽略。

约束说明：

- Host 路径拒绝带 MLIR SSA 的 Kernel ``Pointer`` / ``Layout``；须在 Host 调用 ``tla.make_ptr`` / ``tla.make_layout``。
- 嵌套的 NZFamily 2*2 树必须满足布局约束，并使用平坦的逻辑 `origin_shape=(M, N)`。
  支持满足布局约束的局部视图。调用方提供缓冲区及其元数据。需验证完整 remap 和
  DLPack 缓冲区时，使用 `from_dlpack`。
- 不接管 DLPack 生命周期；``data_ptr == 0`` 为未绑定样本。

调用示例：

```python
import catlass.tla as tla

ptr = tla.make_ptr(
    tla.Float16,
    int(buf.data_ptr()),
    tla.AddressSpace.gm,
)
layout = tla.make_layout(
    tla.make_shape(128, 64),
    tla.make_stride(64, 1),
    layoutTag=tla.arch.RowMajor,
)
host = tla.make_tensor(ptr, layout)
```

---

### `tla.make_fake_tensor`

**源码：** [`catlass.tla.runtime.make_fake_tensor`](../../../../catlass/tla/runtime.py)

功能说明：

构造仅含元数据、不绑定设备缓冲的 Host tensor（`data_ptr == 0`）。
用于无需 NPU 时给 `tla.compile` 提供类型样本。真实缓冲请用 `make_tensor`（直绑）或 `from_dlpack`。

函数原型：

```python
tla.make_fake_tensor(dtype: Any, shape: Any, stride: Any, *, layout_tag: Any | None = None, addrspace: Any = AddressSpace.gm, origin_shape: Iterable[Any] | None = None, coord: Iterable[Any] | None = None, assumed_align: int | None = None) -> _Tensor
```

参数说明：

- *`dtype`*：元素类型，如 `tla.Float16` / `tla.Float32`。必填。
- *`shape`*（`int | tuple`）：逻辑 shape 树（zN 等物理布局用嵌套 tuple）。必填。
- *`stride`*（`int | tuple | None`）：步长树，结构须与 `shape` 一致。传入 `None` 时按布局推导步长。
- *`layout_tag`*：`tla.arch` 标签。可选，默认 `tla.arch.RowMajor`。
- *`addrspace`*：地址空间。可选，默认 `AddressSpace.gm`。
- *`origin_shape`*（`int | tuple | None`）：逻辑形状。线性布局默认为 `shape`，嵌套 NZFamily 布局默认为填充后的 `(M, N)`。
- *`coord`*（`int | tuple | None`）：坐标树。可选；省略时由 layout 推导（通常为零）。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。

约束说明：

- `shape` / `stride` / `origin_shape` / `coord` 须为 Python int 树，
  不能是 Kernel 侧 `tla.make_shape` / `tla.make_stride` / `tla.make_coord`。
- 始终未绑定，不能直接 launch；真实缓冲须改用 `make_tensor` / `from_dlpack`。
- 保留显式传入的 `shape` 和 `stride`。与 Host `make_tensor` 一样，嵌套
  NZFamily 布局必须满足布局约束，使用平坦的逻辑 origin，并支持合法的局部视图。

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
