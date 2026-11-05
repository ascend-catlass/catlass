---
nav_order: 1
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/layout_tensor.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# Layout 与 Tensor

[语言参考](index.md) | [API 速查](../quick_reference.md)

Shape / Coord / Stride / Layout / Tensor 等前端结构化值的构造与视图，以及指针相关辅助接口。

## `tla.make_shape`

**源码：** [`catlass.core_api.make_shape`](../../../../catlass/core_api.py)

功能说明：

构造 shape；分量可为嵌套 tuple。

在 ``@tla.kernel`` 内：构造打包的 `!tla.shape`。在 kernel 外：同一
``_Shape`` 类型，承载 Host int 树，供 ``tla.make_layout`` /
Host ``tla.make_tensor`` 使用。

函数原型：

```python
tla.make_shape(*components: IndexTree) -> TlaShape
```

参数说明：

- *`components`*（`IndexTree`）：shape 各维分量。`zN` / `nZ` /
  `L0Clayout` / `zNUnAlign` 的物理布局用嵌套 tuple。必填。Host 路径：仅 Python int。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内。
- Host 路径：无 frontend；仅 int 树。
- 至少提供 1 个 shape 分量。
- RowMajor / ColumnMajor：用二维 shape `(M, N)`。
- `zN` / `nZ` / `L0Clayout` / `zNUnAlign`：在 `make_layout` /
  `make_tensor` 中要用嵌套物理 shape `((m0, m1), (n0, n1))`。
  只写二维 `(M, N)` **不合法**；请改成嵌套，或优先用
  `make_tensor_like(..., layoutTag=zN)`（由逻辑二维 `origin_shape` 自动换算）。

调用示例：

```python
# RowMajor / ColumnMajor（逻辑二维）：
shape = tla.make_shape(256, 128)

# zN 物理 shape（f16，逻辑 128x64）：
# m0=16，m1=8（=128/16）；n0=16，n1=4（=64/16）
zn_shape = tla.make_shape((16, 8), (16, 4))
```

---

## `tla.make_coord`

**源码：** [`catlass.core_api.make_coord`](../../../../catlass/core_api.py)

功能说明：

构造坐标树。

在 ``@tla.kernel`` 内：构造打包的 `!tla.coord`。在 kernel 外：同一
``_Coord`` 类型，承载 Host int 树。

函数原型：

```python
tla.make_coord(*components: IndexTree) -> TlaCoord
```

参数说明：

- *`components`*（`IndexTree`）：coord 各维分量。必填。Host 路径：仅 Python int。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内。
- Host 路径：无 frontend；仅 int 树。
- 至少提供 1 个 coord 分量。

调用示例：

```python
coord = tla.make_coord(block_row, 0)
```

---

## `tla.make_stride`

**源码：** [`catlass.core_api.make_stride`](../../../../catlass/core_api.py)

功能说明：

构造 stride（嵌套规则与 `make_shape` 相同）。

在 ``@tla.kernel`` 内：构造打包的 `!tla.stride`。在 kernel 外：同一
``_Stride`` 类型，承载 Host int 树。

函数原型：

```python
tla.make_stride(*components: IndexTree) -> TlaStride
```

参数说明：

- *`components`*（`IndexTree`）：stride 各维分量。必填。Host 路径：仅 Python int。

  常见写法：

  | 布局 | 典型 `make_stride(...)` | 含义 |
  | --- | --- | --- |
  | RowMajor 二维 `(M, N)` | `(N, 1)` | 行方向每次跨 `N` 个元素；列方向跨 1 |
  | ColumnMajor 二维 `(M, N)` | `(1, M)` | 行方向跨 1；列方向每次跨 `M` 个元素 |
  | `zN` / `nZ` / … | 嵌套 `((s00, s01), (s10, s11))` | 与物理 `shape` 同结构；数值须匹配对应 layout |

  对 `zN` / `nZ` / `L0Clayout` / `zNUnAlign`：一块 C0 是 32 字节，
  M 方向分块大小是 16。记
  `每块C0元素数 = 32 // sizeof(dtype)`（f16→16，f32→8），
  `每分块元素数 = 每块C0元素数 * 16`。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内。
- Host 路径：无 frontend；仅 int 树。
- 至少提供 1 个 stride 分量。
- 对 `zN` / `nZ` / `L0Clayout` / `zNUnAlign`，stride 数值须匹配
  layout（见示例）；`make_tensor` 会按 `shape` + `layoutTag` 检查。

调用示例：

```python
# RowMajor 二维（逻辑 256x128，紧凑存放）：
stride = tla.make_stride(128, 1)

# f16 zN，逻辑尺寸 (128, 64)：
# shape = ((16, 8), (16, 4))
# stride = ((每块C0元素数，每分块元素数),
#           (1, 向上取整到16的M * 每块C0元素数))
#        = ((16, 256), (1, 2048))
zn_stride = tla.make_stride((16, 256), (1, 2048))

# f16 nZ，同一逻辑 (128, 64)：
# shape = ((16, 8), (16, 4))
# stride = ((1, 向上取整到16的N * 每块C0元素数),
#           (每块C0元素数，每分块元素数))
#        = ((1, 1024), (16, 256))
nz_stride = tla.make_stride((1, 1024), (16, 256))
```

---

## `tla.make_layout`

**源码：** [`catlass.core_api.make_layout`](../../../../catlass/core_api.py)

功能说明：

由 shape / stride 合成 layout。

在 ``@tla.kernel`` 内：构造 `!tla.layout`（对应 `tla.make_layout`）；
``shape`` / ``stride`` 须来自 ``make_shape`` / ``make_stride``。

在 kernel 外：同一 ``_Layout`` 类型；``shape`` / ``stride`` 来自 Host
``make_shape`` / ``make_stride``（或等价 int 树），供
Host ``tla.make_tensor`` 使用。

函数原型：

```python
tla.make_layout(shape: Any, stride: Any, *, origin_shape: Any | None = None, layoutTag: _LayoutTag | None = None) -> TlaLayout
```

参数说明：

- `shape`（`_Shape` | int 树）：由 `tla.make_shape` 构造；Host 也可传 Python int 树。
  RowMajor / ColumnMajor：二维 `(M, N)`。
  `zN` / `nZ` / `L0Clayout` / `zNUnAlign`：嵌套 `((m0, m1), (n0, n1))`。
- `stride`（`_Stride` | int 树）：由 `tla.make_stride` 构造；Host 也可传与 `shape` 同嵌套的 int 树。
- `origin_shape`（`_Shape` | int 树 | None）：逻辑工作尺寸。可选，默认 `None`
  （推断；Host 上省略时等于 `shape`）。
- `layoutTag`（`_LayoutTag | None`）：布局标签，见 [布局标签](system.md#布局标签)。
  可选，默认 `None`（`RowMajor`）。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内；`shape` / `stride` 为 `make_shape` / `make_stride`（SSA）。
- Host 路径：无 frontend；Host `make_shape` / `make_stride` 或 int 树；结果供
  Host `tla.make_tensor`，不是 Kernel IR。
- RowMajor / ColumnMajor 省略 `origin_shape` 时推断为 `shape`；
  `zN` / `nZ` / `L0Clayout` / `zNUnAlign` 在 Kernel 上从
  `shape=((m0,m1),(n0,n1))` 推断为 `(m0*m1, n0*n1)`；Host 省略 origin 时默认等于 `shape`。
- **不要**给 `zN` / `nZ` / `L0Clayout` / `zNUnAlign` 配普通二维
  `shape`，检查会失败。请写嵌套物理 shape，或用
  `make_tensor_like(ptr, like, layoutTag=...)` 由 `like.origin_shape` 自动换算。

调用示例：

```python
# RowMajor 二维：
layout = tla.make_layout(
    tla.make_shape(256, 128),
    tla.make_stride(128, 1),
    layoutTag=tla.arch.RowMajor,
)

# 显式 f16 zN（逻辑 128x64 → 嵌套物理 + 二维 origin）：
# 前：逻辑 ND tile 为 (128, 64)。
zn = tla.make_layout(
    tla.make_shape((16, 8), (16, 4)),
    tla.make_stride((16, 256), (1, 2048)),
    origin_shape=tla.make_shape(128, 64),  # 拷贝/切块用的逻辑尺寸
    layoutTag=tla.arch.zN,
)
# 后：layout.shape 为 zN 打包；layout.origin_shape 仍为 (128, 64)。
```

---

## `tla.make_layout_with_tag`

**源码：** [`catlass.core_api.make_layout_with_tag`](../../../../catlass/core_api.py)

功能说明：

由平铺 2-D shape、元素类型与 layout tag 构造片上 `!tla.layout`。

函数原型：

```python
tla.make_layout_with_tag(shape: tuple, dtype: Any, layoutTag: _LayoutTag | None = None) -> TlaLayout
```

参数说明：

- `shape`（`tuple`）：扁平二维逻辑 extent `(rows, cols)`。每个叶子为
  `int` 或运行时 `Numeric`（动态叶子）。必填。
- `dtype`（`Numeric` 子类）：元素类型；决定重映射物理树的 C0/分块几何。必填。
- `layoutTag`（`_LayoutTag | None`）：布局标签，见 [布局标签](system.md#布局标签)。
  可选，默认 `None`（`RowMajor`）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅用于片上 tensor 布局：合成后的 layout 应通过
  `tla.make_tensor(ptr, layout)` 落到片上指针，而不是绑定 GM Host 内存。
- `shape` 必须是扁平二元组；嵌套树与 `None` 叶子会被拒绝——整个 extent
  由调用方拥有。
- 结果 layout 的 origin shape 等于 `shape`（对齐后的 extent）；配合
  `tla.make_tensor(ptr, layout)` 落到片上指针。
- 本接口仅支持：`RowMajor`、`ColumnMajor`、`zN`、`nZ`、`zZMxScale`、
  `nNMxScale`、`L0Clayout`、`zNUnAlign`。
- `zZMxScale` / `nNMxScale` 要求元素类型为 `tla.Float8E8M0`。GM 侧
  MxScale tag 会被拒绝。

调用示例：

```python
# fp8 zN tile：M 运行时裁剪，K 向上取整到 64：
lay = tla.make_layout_with_tag((m_valid, k_l0), dtype_a, tla.arch.zN)
t_l1a = tla.make_tensor(l1a, lay)
```

---

## `tla.tile_view`

**源码：** [`catlass.core_api.tile_view`](../../../../catlass/core_api.py)

功能说明：

在 `!tla.tensor` 源上按 tile 坐标粒度创建 tile 视图。

函数原型：

```python
tla.tile_view(source: Tensor, shape: _Shape, coord: _Coord) -> TlaTensor
```

参数说明：

- `source`（`Tensor`）：源 `!tla.tensor`。必填。
- `shape`（`_Shape`）：tile 形状，由 `tla.make_shape` 构造。必填。
- `coord`（`_Coord`）：tile 坐标，由 `tla.make_coord` 构造（tile 粒度）。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `coord` 以 tile 粒度传入，前端会按 `shape` 换算为元素偏移。

调用示例：

```python
tile = tla.tile_view(
    source, tla.make_shape(256, 128), tla.make_coord(block_row, 0)
)
```

---

## `tla.get_tile`

**源码：** [`catlass.core_api.get_tile`](../../../../catlass/core_api.py)

功能说明：

从 tensor 切出平铺 2-D 子 tile（共享源缓冲）。

函数原型：

```python
tla.get_tile(tensor: Tensor, coord: TlaCoord, shape: TlaShape) -> TlaTensor
```

参数说明：

- `tensor`（`Tensor`）：源 `!tla.tensor`。必填。
- `coord`（`TlaCoord`）：由 `tla.make_coord` 得到的起点，相对源的扁平
  二维元素偏移。必填。
- `shape`（`TlaShape`）：由 `tla.make_shape` 得到的请求 tile extent，
  扁平二维。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `coord` 偏移必须落在源的 `origin_shape` 内——裁剪只缩短 extent，
  不会钳制偏移。
- 支持的 layout tag 与 `tla.make_tensor` 一致（RowMajor、ColumnMajor、
  zN、nZ、L0Clayout、zNUnAlign）。

调用示例：

```python
# 将 L1 A tile 的 K-pad 尾部清零（见 Tensor.fill）：
pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                   tla.make_shape(m, k_l0 - k_valid))
pad.fill(0)
```

---

## `tla.make_tensor`

**源码：** [`catlass.core_api.make_tensor`](../../../../catlass/core_api.py)

功能说明：

由显式指针、layout 与可选 coord 构造 tensor。

在 ``@tla.kernel`` 内：构造打包的 `!tla.tensor` IR。
在 kernel 外：同一入口用 Host ``tla.make_ptr`` / ``tla.make_layout``
直绑 Host tensor（无 DLPack）。

函数原型：

```python
tla.make_tensor(ptr: Pointer, layout: TlaLayout, coord: CoordLike | None = None) -> TlaTensor
```

参数说明：

- `ptr`（`Pointer`）：底层数据指针（Kernel 为 `!tla.ptr`，或 Host ``make_ptr``）。必填。
- `layout`（`TlaLayout`）：tensor 布局，由 `tla.make_layout` 构造。必填。
- `coord`（`CoordLike | None`）：可选起始坐标；缺省视为零坐标。可选，默认 `None`。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内；指针、layout、coord 须匹配目标地址空间与 dtype。
- Host 路径：无 frontend；仅 Host ``make_ptr`` / ``make_layout``；
  信任 layout 元数据（需要校验时请用 ``from_dlpack``）。``loc`` 忽略。
- `coord` 缺省为零坐标，秩与 layout 一致（秩 2 → `make_coord(0, 0)`，秩 1 → `make_coord(0)`）。元素类型与地址空间来自 `ptr`；layout tag、shape、stride、origin 来自 layout（未给 `origin_shape` 时 origin 默认为 `shape`）。
- Lowering 支持 RowMajor、ColumnMajor、zN、nZ、L0Clayout、zNUnAlign。
  对 `zN` / `nZ` / `L0Clayout` / `zNUnAlign`，物理 `shape` / `stride`
  为嵌套 2×2，逻辑 coord / `origin_shape` 仍是二维 `(M, N)`。若 `make_layout`
  省略 `origin_shape`，则从物理 shape 推断逻辑尺寸（例如 `(m0*m1, n0*n1)`）。
- 完整 Kernel 编译要求 `ptr` 具备底层存储；可运行 kernel 中片上指针的推荐形式是 `allocate`（可选再经 `recast_ptr`）。

调用示例：

```python
# Kernel IR：
tensor = tla.make_tensor(ptr, layout, coord=tla.make_coord(0, 0))

# Host 直绑（无 DLPack）：
host = tla.make_tensor(
    tla.make_ptr(tla.Float16, int(buf.data_ptr()), tla.AddressSpace.gm),
    tla.make_layout(
        tla.make_shape(128, 64),
        tla.make_stride(64, 1),
        layoutTag=tla.arch.RowMajor,
    ),
)
```

---

## `tla.make_tensor_like`

**源码：** [`catlass.core_api.make_tensor_like`](../../../../catlass/core_api.py)

功能说明：

按参考 tile 的结构化元数据，在给定指针上构造同形态 tensor。

函数原型：

```python
tla.make_tensor_like(ptr: Pointer, like: Tensor, layoutTag: _LayoutTag | None = None) -> TlaTensor
```

参数说明：

- `ptr`（`Pointer`）：目标数据指针。必填。
- `like`（`Tensor`）：参考 tile，提供结构化 tensor 元数据。必填。
- `layoutTag`（`_LayoutTag | None`）：覆盖参考 tile 的布局标签。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 参考 tile 须提供可用的结构化 tensor 元数据。
- 元素类型来自 `ptr` 的 `!tla.ptr` 指向类型；仅接受片上目标指针。

调用示例：

```python
dst = tla.make_tensor_like(ptr, like=src_tile, layoutTag=tla.arch.RowMajor)
```

---

## `tla.make_ptr`

**源码：** [`catlass.core_api.make_ptr`](../../../../catlass/core_api.py)

功能说明：

由地址值构造指针。

在 ``@tla.kernel`` 内：构造 `!tla.ptr`。在 kernel 外：同一 ``Pointer`` 类型，
持有 Host 地址，供 Host ``tla.make_tensor`` 使用（无 MLIR）；
``value`` 须为 Python ``int`` 地址。

函数原型：

```python
tla.make_ptr(dtype: type[Numeric] | None, value: int | mlir_ir.Value | Numeric, mem_space: AddressSpace = AddressSpace.gm, *, assumed_align: int | None = None) -> Pointer
```

参数说明：

- `dtype`（`type[Numeric] | None`）：指针指向的元素类型；`None` 表示 `Int8`。可选，默认 `None`。
- `value`（`int | mlir_ir.Value | Numeric`）：地址值。必填。Host 路径要求 ``int``。
- `mem_space`（`AddressSpace`）：指针所在地址空间。可选，默认 `AddressSpace.gm`。
- `assumed_align`（`int | None`）：假定对齐字节数。可选，默认 `None`。

约束说明：

- Kernel 路径：须在 `@tla.kernel` 内；整型地址位宽须与目标 `mem_space` 匹配。
- Host 路径：无 frontend；结果仅供 Host `tla.make_tensor`。

调用示例：

```python
ptr = tla.make_ptr(tla.Float16, addr, mem_space=tla.AddressSpace.gm)
```

---

## `tla.recast_ptr`

**源码：** [`catlass.core_api.recast_ptr`](../../../../catlass/core_api.py)

功能说明：

仅改变 `!tla.ptr` 的逻辑元素类型（不做 swizzle）。

函数原型：

```python
tla.recast_ptr(ptr: Pointer, *, dtype: type[Numeric]) -> Pointer
```

参数说明：

- `ptr`（`Pointer`）：待重解释的指针。必填。
- `dtype`（`type[Numeric]`）：新的元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅改变逻辑元素类型，不改变地址与 swizzle。

调用示例：

```python
ptr_f32 = tla.recast_ptr(ptr_f16, dtype=tla.Float32)
```
