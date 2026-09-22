---
nav_order: 10
---

<!--
Manually maintained Chinese Kernel API reference.
Generated English source of truth: docs/en/api/kernel_api_reference.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# TLA DSL Kernel API 参考

本文档介绍 **TLA DSL 的 kernel 侧 Core API**（通常以 `import catlass.tla as tla` 导入）。
内容覆盖基本数据类型、计算与同步接口、片上资源管理以及调试打印。Host 侧编译 / 启动 / tensor 绑定见 [Host API 参考](host_api_reference.md)；DLPack 接入教程见 [Host Tensor 接入](../kernel_development/core_concepts/tensor_binding.md)。

接口说明与调用示例来自各 op 源码 docstring；所有接口均须在 `@tla.kernel` 装饰的
kernel 函数体内调用。

---

## 目录

- [基本数据类型与操作](#基本数据类型与操作)
- [数据搬运](#数据搬运)
- [矩阵运算](#矩阵运算)
- [Vector 运算](#vector-运算)
  - [Mask 计算](#mask-计算)
  - [基础算术](#基础算术)
  - [逻辑计算](#逻辑计算)
  - [比较与选择](#比较与选择)
  - [数据填充](#数据填充)
  - [离散与聚合](#离散与聚合)
  - [数据重排](#数据重排)
  - [数据压缩](#数据压缩)
  - [类型转换](#类型转换)
- [同步控制](#同步控制)
- [系统变量访问](#系统变量访问)
- [资源管理](#资源管理)
- [调试接口](#调试接口)
- [作用域和控制流](#作用域和控制流)

---

## 基本数据类型与操作

Shape / Coord / Stride / Layout / Tensor 等前端结构化值的构造与视图，以及指针相关辅助接口。

### `make_shape`

**源码：** [`catlass.core_api.make_shape`](../../../catlass/core_api.py#L3519)

功能说明：

构造打包的 `!tla.shape`，分量可为嵌套 tuple。

函数原型：

```python
tla.make_shape(*components: IndexTree) -> TlaShape
```

参数说明：

- *`components`*（`IndexTree`）：shape 各维分量。`zN` / `nZ` /
  `L0Clayout` / `zNUnAlign` 的物理布局用嵌套 tuple。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
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

### `make_coord`

**源码：** [`catlass.core_api.make_coord`](../../../catlass/core_api.py#L3560)

功能说明：

构造打包的 `!tla.coord`。

函数原型：

```python
tla.make_coord(*components: IndexTree) -> TlaCoord
```

参数说明：

- *`components`*（`IndexTree`）：coord 各维分量。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 至少提供 1 个 coord 分量。

调用示例：

```python
coord = tla.make_coord(block_row, 0)
```

---

### `make_stride`

**源码：** [`catlass.core_api.make_stride`](../../../catlass/core_api.py#L3589)

功能说明：

构造打包的 `!tla.stride`（嵌套规则与 `make_shape` 相同）。

函数原型：

```python
tla.make_stride(*components: IndexTree) -> TlaStride
```

参数说明：

- *`components`*（`IndexTree`）：stride 各维分量。必填。

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

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 至少提供 1 个 stride 分量。
- 对 `zN` / `nZ` / `L0Clayout` / `zNUnAlign`，stride 数值须匹配
  layout（见示例）；`make_tensor` 会按 `shape` + `layoutTag` 检查。

调用示例：

```python
# RowMajor 二维（逻辑 256x128，紧凑存放）：
stride = tla.make_stride(128, 1)

# f16 zN，逻辑尺寸 (128, 64)：
# shape = ((16, 8), (16, 4))
# stride = ((每块C0元素数, 每分块元素数),
#           (1, 向上取整到16的M * 每块C0元素数))
#        = ((16, 256), (1, 2048))
zn_stride = tla.make_stride((16, 256), (1, 2048))

# f16 nZ，同一逻辑 (128, 64)：
# shape = ((16, 8), (16, 4))
# stride = ((1, 向上取整到16的N * 每块C0元素数),
#           (每块C0元素数, 每分块元素数))
#        = ((1, 1024), (16, 256))
nz_stride = tla.make_stride((1, 1024), (16, 256))
```

---

### `make_layout`

**源码：** [`catlass.core_api.make_layout`](../../../catlass/core_api.py#L3649)

功能说明：

由 shape / stride 合成 `!tla.layout`（对应 `tla.make_layout`）。

函数原型：

```python
tla.make_layout(shape: _Shape, stride: _Stride, *, origin_shape: _Shape | None = None, layoutTag: _LayoutTag | None = None) -> TlaLayout
```

参数说明：

- `shape`（`_Shape`）：布局 shape，由 `tla.make_shape` 构造。必填。
  RowMajor / ColumnMajor：二维 `(M, N)`。
  `zN` / `nZ` / `L0Clayout` / `zNUnAlign`：嵌套 `((m0, m1), (n0, n1))`。
- `stride`（`_Stride`）：布局 stride，由 `tla.make_stride` 构造。必填。
  嵌套形态须与 `shape` 一致。
- `origin_shape`（`_Shape | None`）：逻辑工作尺寸（对齐前的真实数据大小）。
  可选，默认 `None`。拷贝 / 切块按该逻辑尺寸；物理存放方式在 `shape` / `stride`。
- `layoutTag`（`_LayoutTag | None`）：布局标签（如 `tla.arch.RowMajor`、
  `tla.arch.zN`）。可选，默认 `None`（RowMajor）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `shape` / `stride` 须为 `make_shape` / `make_stride` 的返回值。
- RowMajor / ColumnMajor 省略 `origin_shape` 时推断为 `shape`；
  `zN` / `nZ` / `L0Clayout` / `zNUnAlign` 则从
  `shape=((m0,m1),(n0,n1))` 推断为 `(m0*m1, n0*n1)`。
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

### `tile_view`

**源码：** [`catlass.core_api.tile_view`](../../../catlass/core_api.py#L3818)

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

### `make_tensor`

**源码：** [`catlass.core_api.make_tensor`](../../../catlass/core_api.py#L3865)

功能说明：

由显式指针、layout 与可选 coord 构造 `!tla.tensor`。

函数原型：

```python
tla.make_tensor(ptr: Pointer, layout: TlaLayout, coord: CoordLike | None = None) -> TlaTensor
```

参数说明：

- `ptr`（`Pointer`）：底层数据指针（`!tla.ptr`）。必填。
- `layout`（`TlaLayout`）：tensor 布局，由 `tla.make_layout` 构造。必填。
- `coord`（`CoordLike | None`）：可选起始坐标；缺省视为零坐标。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 指针、layout、coord 须匹配目标地址空间与 dtype。
- `coord` 缺省为零坐标，秩与 layout 一致（秩 2 → `make_coord(0, 0)`，秩 1 → `make_coord(0)`）。元素类型与地址空间来自 `ptr` 的 `!tla.ptr`；layout tag、shape、stride、origin 来自 `!tla.layout` 操作数（未给 `origin_shape` 时 origin 默认为 `shape`）。
- Lowering 支持 RowMajor、ColumnMajor、zN、nZ、L0Clayout、zNUnAlign。
  对 `zN` / `nZ` / `L0Clayout` / `zNUnAlign`，物理 `shape` / `stride`
  为嵌套 2×2，逻辑 coord / `origin_shape` 仍是二维 `(M, N)`。若 `make_layout`
  省略 `origin_shape`，则从物理 shape 推断逻辑尺寸（例如 `(m0*m1, n0*n1)`）。
- 完整编译要求 `ptr` 具备底层存储；可运行 kernel 中片上指针的推荐形式是 `allocate`（可选再经 `recast_ptr`）。

调用示例：

```python
tensor = tla.make_tensor(ptr, layout, coord=tla.make_coord(0, 0))
```

---

### `make_tensor_like`

**源码：** [`catlass.core_api.make_tensor_like`](../../../catlass/core_api.py#L4060)

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

### `make_ptr`

**源码：** [`catlass.core_api.make_ptr`](../../../catlass/core_api.py#L7241)

功能说明：

由整型位型经 `tla.inttoptr` 构造指针。

函数原型：

```python
tla.make_ptr(dtype: type[Numeric] | None, value: int | mlir_ir.Value | Numeric, mem_space: AddressSpace = AddressSpace.gm, *, assumed_align: int | None = None) -> Pointer
```

参数说明：

- `dtype`（`type[Numeric] | None`）：指针指向的元素类型；`None` 表示 `Int8`。可选，默认 `None`。
- `value`（`int | mlir_ir.Value | Numeric`）：地址值（整数、`mlir` Value 或 Numeric）。必填。
- `mem_space`（`AddressSpace`）：指针所在地址空间。可选，默认 `AddressSpace.gm`。
- `assumed_align`（`int | None`）：假定对齐字节数。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 整型地址位宽须与目标 `mem_space` 匹配。

调用示例：

```python
ptr = tla.make_ptr(tla.Float16, addr, mem_space=tla.AddressSpace.gm)
```

---

### `recast_ptr`

**源码：** [`catlass.core_api.recast_ptr`](../../../catlass/core_api.py#L7295)

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

---

## 数据搬运

片上与全局内存之间的 tensor 拷贝，以及 UB 寄存器 load/store。

### `copy`

**源码：** [`catlass.core_api.copy`](../../../catlass/core_api.py#L4255)

功能说明：

在 tile 之间拷贝数据。硬件通路由 `src`/`dst` 地址空间决定（vector：GM↔UB、UB→L1；
cube：GM→L1、L1→L0A/L0B、L0C→GM|UB|L1）。两侧 layout tag 选择格式转换（例如 ND→zN）。

vector 通路支持的元素类型：

| 通路 | 源 layout → 目的 layout | 元素类型 |
|---|---|---|
| GM → UB | `RowMajor` → `RowMajor` | `f32`、`f16`、`bf16`、`i32`、`i16`、`i8`、`f8e4m3fn`、`f8e5m2` |
| UB → GM | `RowMajor` → `RowMajor` | 同上 |
| UB → L1 | `RowMajor` → `zN` | `f32`、`f16`、`bf16`、`i8`、`f8e4m3fn`、`f8e5m2` |
| UB → L1 | `zN` → `zN` | 同上 |
| UB → L1 | `zNUnAlign` → `zN` | 同上 |

UB → L1 的目的 layout 只能是 `zN`；没有以 UB 为源、`nZ` 为目的的通路。`i16` /
`i32` 只有 GM↔UB，没有 UB→L1。

`zN` 源的 UB tile 需要先在 UB 里按 zN 摆好：用 `BlockStoreParams` 的带 stride
block store 把每行分块写入打了 `zN` tag 的 UB view（见
`examples/end_to_end/basic_mixed/basic_mixed_store_zN.py` 与
`basic_mixed_store_zNUnAlign.py`，两者都用 `--dtype` 跑遍该通路支持的全部元素
类型）。分块长度按元素类型算：一个 vector register 是 256 字节，`vsstb` 一次
搬满它的 8 个 DataBlock，所以一块是 `256 / sizeof(T)` 个元素。8 位类型（`i8` /
fp8）的 block store 与 GM↔UB、UB→L1 拷贝一样，在 bc 层按 `int8_t` 实例化：搬运
只用到宽度与 stride，不解释元素类型，而 AIV 后端对 fp8 没有标量语义。

拷贝 / 切块大小按各 tile 的逻辑 `origin_shape`（不是嵌套物理 `shape`）。
物理 `shape` / `stride` 描述这些逻辑元素如何存放（凑齐对齐长度、zN 打包等）。

函数原型：

```python
tla.copy(dst: Tensor, src: Tensor, params: CopyParams | None = None) -> None
```

参数说明：

- `dst`（`Tensor`）：目的 tile。必填。
- `src`（`Tensor`）：源 tile。必填。
- `params`（`CopyParams | None`）：可选通路参数
  （`CopyL0C2DstParams`、`CopyUbToGmParams` / atomic 等）。默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用（上述 cube 通路在 `cube()`，
  vector 通路在 `vector()`）。
- 整 tile DMA 使用 `tla.copy`。寄存器级 UB 非对齐访问请改用
  `tensor.load` / `tensor.store`，并传入 `UnalignLoadParams` /
  `UnalignStoreParams`。
- **尺寸未按 32 字节对齐时：** C0 为 32 字节。ND GM↔UB 时，优先让主导维的
  字节数是 32 的倍数（如 RowMajor f16 选 `N` 使 `N % 16 == 0`）。若真实数据
  大小做不到这一点，请保持 `origin_shape` 为真实逻辑尺寸，并把物理 layout
  放大到对齐后的尺寸，或改用上方非对齐寄存器 load/store。zN 在 `M` 不是 16
  的整数倍时，用 `tla.arch.zNUnAlign` 而非 `zN`。

调用示例：

```python
# --- 1) 对齐 GM ↔ UB（RowMajor，满足 32 字节对齐） ---
# 前：x_gm[i, j] 为 ND 数据；x_ub 为空。origin_shape==(M, N)。
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
    # 后：对 origin_shape 内所有逻辑 (i, j)，x_ub[i, j] == x_gm[i, j]。
    tla.copy(dst=y_gm, src=y_ub)

# --- 2) ND → zN：GM RowMajor → L1 zN（排布变化） ---
# 前（逻辑）：gm_a.origin_shape==(128, 64)，RowMajor；元素 (r, c)
#   的 ND 偏移为 r*64+c。
# 后（L1 物理）：l1_a 使用嵌套 zN shape/stride，同一逻辑 (r, c) 按 zN 存放；
#   l1_a.origin_shape 仍为 (128, 64)。
l1_a = tla.make_tensor_like(l1_ptr, gm_a, layoutTag=tla.arch.zN)
with tla.cube():
    tla.copy(dst=l1_a, src=gm_a)

# 不用 make_tensor_like 时的显式 zN（同一逻辑 128x64 f16）：
l1_a = tla.make_tensor(
    l1_ptr,
    tla.make_layout(
        tla.make_shape((16, 8), (16, 4)),
        tla.make_stride((16, 256), (1, 2048)),
        origin_shape=tla.make_shape(128, 64),
        layoutTag=tla.arch.zN,
    ),
)

# --- 3) M 不是 16 的倍数：zNUnAlign ---
# 前：rows 可能为运行期值，且不是 16 的倍数。
l1_unalign = tla.make_tensor_like(l1_ptr, gm_tile, layoutTag=tla.arch.zNUnAlign)
with tla.cube():
    tla.copy(dst=l1_unalign, src=gm_tile)

# 相关（寄存器路径，不是 tla.copy）：UB ↔ vector 寄存器非对齐访问
#   with tla.vec.func(mode="simd"):
#       x_reg = x_ub.load(tla.params.UnalignLoadParams())
#       y_ub.store(y_reg, tla.params.UnalignStoreParams())
```

---

### `Tensor.load`

**源码：** [`catlass.tla.tensor._Tensor.load`](../../../catlass/tla/tensor.py#L215)

功能说明：

将本 UB tensor tile 载入 vector 或 mask SSA（`tile.load`）。

函数原型：

```python
tile.load(params: LoadParams | None = None) -> MaskSSA | VectorSSA | tuple[VectorSSA, VectorSSA]
```

参数说明：

- `params`（`LoadParams | None`）：载入模式。`None` / `NormalLoadParams` /
  `UnalignLoadParams` / `BlockLoadParams` → `VectorSSA`（`DIST_DINTLV_B32`
  时可为二元组）；`MaskLoadParams` → `MaskSSA`。
  `NormalLoadParams.load_dist` 选择 AscendC 分发模式：`norm`（默认）、
  `brc_b32`（单个 b32 元素广播到全部 lane）、`dintlv_b32`（f32 解交织，
  双返回值）、`us_b8` / `us_b16`（2 倍上采样）、`brc_b16`（单个 b16 元素
  广播）、`unpack_b16`（b16 零扩展到双倍宽度）、`e2b_b16` / `e2b_b32`
  （元素到 DataBlock 广播）、`blk`（一个 32 字节 DataBlock 广播到全部 8 个）。
  `BlockLoadParams` 为 `vsldb` 跨步收集：一条指令收集 8 个 DataBlock，
  块头间隔 `block_stride` 个 DataBlock（32B 单位）；`block_stride == 0`
  时将首个 DataBlock 复制到全部 8 槽；`post_update_stride` 为编译期地址
  预偏移，单位 32B DataBlock。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；源 tile 须位于 UB。
- Mask 载入要求 UB 元素类型为 1/2/4 字节标量。
- 分发模式限定元素宽度：`us_b8` 要求 i8/u8；`brc_b16` / `us_b16` /
  `unpack_b16` / `e2b_b16` 要求 2 字节（f16/bf16/i16/u16）；`e2b_b32`
  要求 4 字节（f32/i32/u32）；`dintlv_b32` 要求 f32；`blk` 适用于任意
  元素宽度。
- `BlockLoadParams` 要求 2/4 字节元素类型（f32/f16/bf16/i32/u32/i16/u16）；
  `block_stride` 与 `post_update_stride` 须在 [0, 65535] 范围内。

调用示例：

```python
with tla.vec.func(mode="simd"):
    x_reg = x_ub.load()
    x_unalign = x_ub.load(tla.params.UnalignLoadParams())
    x_blk = x_ub.load(tla.params.NormalLoadParams(
        load_dist=tla.params.LoadDist.DIST_BLK))
    x_gather = x_ub.load(tla.params.BlockLoadParams(block_stride=4))
```

---

### `Tensor.store`

**源码：** [`catlass.tla.tensor._Tensor.store`](../../../catlass/tla/tensor.py#L529)

功能说明：

将 vector 或 mask SSA 写回本 UB tensor tile（`tile.store`）。

函数原型：

```python
tile.store(value: VectorSSA | MaskSSA, params: StoreParams | None = None, *, mask: MaskSSA | None = None) -> None
```

参数说明：

- `value`（`VectorSSA | MaskSSA`）：要写入的 `VectorSSA` 或 `MaskSSA`。必填。
- `params`（`StoreParams | None`）：写回模式。`None` / `NormalStoreParams` /
  `UnalignStoreParams` / `BlockStoreParams` → vector 写回；`MaskStoreParams` → mask 写回。
  `NormalStoreParams.store_dist` 另可选 first-element 模式：`first_element_b8` /
  `first_element_b16` / `first_element_b32`——仅将源寄存器 lane 0 写入 tile
  基址，后缀限定写出的元素字节宽度。可选，默认 `None`。
- `mask`（`MaskSSA | None`）：vector 写回的可选谓词；与 `MaskStoreParams` 及
  first-element 模式均不可同时使用。默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；目标 tile 须位于 UB。
- first-element 模式要求被写值与目标 tile 的元素宽度匹配后缀
  （b8 → 1 字节，b16 → 2 字节，b32 → 4 字节）；传入谓词 `mask` 会报错。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y_ub.store(y_reg)
    y_ub.store(y_reg, tla.params.UnalignStoreParams())
    y_ub.store(y_reg, tla.params.NormalStoreParams(
        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B16))
```

---

## 矩阵运算

Cube 侧矩阵乘加（`tla.mmad`）。

### `mmad`

**源码：** [`catlass.core_api.mmad`](../../../catlass/core_api.py#L5204)

功能说明：

在 TLA tile 上执行矩阵乘累加。

函数原型：

```python
tla.mmad(acc: Tensor, lhs: Tensor, rhs: Tensor, init_c: bool | Bool | None = None, unit_flag: IndexLike | None = None, compute_order: ComputeOrder = ComputeOrder.M_FIRST, hf32_mode: HF32Mode = HF32Mode.HF32_DISABLE, **extra_kwargs: object) -> None
```

参数说明：

- `acc`（`Tensor`）：累加器 / 输出 tile（通常在 L0C）。必填。
- `lhs`（`Tensor`）：左矩阵 tile（通常在 L0A）。必填。
- `rhs`（`Tensor`）：右矩阵 tile（通常在 L0B）。必填。
- `init_c`（`bool | Bool | None`）：是否先清零累加器；省略时默认为 `False`。可选，默认 `None`。
- `unit_flag`（`IndexLike | None`）：unit flag 控制位；省略时默认为 `0`。可选，默认 `None`。
- `compute_order`（`ComputeOrder`）：M/N 计算方向优先级；默认 `M_FIRST`。
- `hf32_mode`（`HF32Mode`）：FP32 操作数在 L0A/L0B 上、矩阵乘之前的 HF32 舍入模式。可选，默认 `HF32_DISABLE`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 内调用；`acc`/`lhs`/`rhs` 须为匹配的 L0 tile。
- 支持的元素类型通路包括 `f16`/`bf16`/`f32`，以及任意 `f8e4m3fn` / `f8e5m2` 操作数配对，累加到 L0C 上的 fp32。
- `init_c` 仅接受 Python `bool` 或 `i1` SSA 值。
- 不接受未知关键字参数；传入则报错。

调用示例：

```python
# 前：l0a / l0b 为当前 K 切片；l0c 为 L0C 上的累加器。
with tla.cube():
    tla.mmad(l0c, l0a, l0b, init_c=True, unit_flag=0b11)
    # 后：l0c 累加 lhs@rhs（init_c=True 时先清零）。
```

---

## Vector 运算

寄存器 Vector 路径上的计算与 mask 操作，通常须在 `tla.vec.func()` 内调用。

### Mask 计算

Mask 创建与尾块更新。

#### `create_mask`

**源码：** [`catlass.core_api.create_mask`](../../../catlass/core_api.py#L7553)

功能说明：

按固定 pattern（`tla.mask.*` token）创建 vector mask。

函数原型：

```python
tla.create_mask(*, pattern: _MaskPattern | str | None = None, dtype: DTypeLike = Float32) -> MaskSSA
```

参数说明：

- `pattern`（`_MaskPattern | str | None`）：掩码 pattern token 或其名字字符串。
  运行时必填（传 `None` 会报错）。token 挂在 `tla.mask` 下（如
  `tla.mask.ALL`、`tla.mask.VL8`）：

  | Pattern | 含义 |
  | --- | --- |
  | `ALL` | 全部元素有效 |
  | `ALLF` | 全部元素无效 |
  | `VL1` / `VL2` / `VL3` / `VL4` | 最低 1 / 2 / 3 / 4 个元素有效 |
  | `VL8` / `VL16` / `VL32` / `VL64` / `VL128` | 最低 8 / 16 / 32 / 64 / 128 个元素有效 |
  | `M3` | 下标为 3 的倍数的元素有效 |
  | `M4` | 下标为 4 的倍数的元素有效 |
  | `H` | 最低一半元素有效 |
  | `Q` | 最低四分之一元素有效 |

- `dtype`（`DTypeLike`）：与掩码关联的元素类型（也决定一条 vector 能放多少元素：
  256 字节 / 元素大小）。可选，默认 `Float32`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`pattern` 必填。
- 接受 `mask=` 的接口需要 `create_mask` / `update_mask` 得到的 `MaskSSA`，
  不能直接传 `tla.mask.*` token。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 用 pattern token 构建 mask：
    m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
    m_tail = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float16)
    # 前：x_reg / y_reg 为整 vector；只让最低 8 个元素做加法。
    z = tla.add(x_reg, y_reg, mask=m_tail)
    # 后：有效元素为 x+y；被 mask 掉的元素不参与。
```

---

#### `update_mask`

**源码：** [`catlass.core_api.update_mask`](../../../catlass/core_api.py#L7617)

功能说明：

创建尾部 mask，并返回剩余元素计数。

函数原型：

```python
tla.update_mask(true_shape: IndexLike, dtype: DTypeLike = Float32) -> tuple[MaskSSA, Numeric]
```

参数说明：

- `true_shape`（`IndexLike`）：当前有效（true）区域形状。必填。
- `dtype`（`DTypeLike`）：与掩码关联的元素类型。可选，默认 `Float32`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    tail_mask, remain = tla.update_mask(true_shape, dtype=tla.Float32)
```

---

### 基础算术

#### `exp`

**源码：** [`catlass.core_api.exp`](../../../catlass/core_api.py#L5786)

功能说明：

vector 逐元素指数（需 f16/f32）。

函数原型：

```python
tla.exp(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.exp(x_reg)
```

---

#### `log`

**源码：** [`catlass.core_api.log`](../../../catlass/core_api.py#L5808)

功能说明：

vector 逐元素对数（需 f16/f32）。

函数原型：

```python
tla.log(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.log(x_reg)
```

---

#### `sqrt`

**源码：** [`catlass.core_api.sqrt`](../../../catlass/core_api.py#L5830)

功能说明：

vector 逐元素平方根（需 f16/f32）。

函数原型：

```python
tla.sqrt(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.sqrt(x_reg)
```

---

#### `abs`

**源码：** [`catlass.core_api.abs`](../../../catlass/core_api.py#L5852)

功能说明：

vector 逐元素绝对值。

函数原型：

```python
tla.abs(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.abs(x_reg)
```

---

#### `neg`

**源码：** [`catlass.core_api.neg`](../../../catlass/core_api.py#L6016)

功能说明：

vector 逐元素取负。

函数原型：

```python
tla.neg(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.neg(x_reg)
```

---

#### `add`

**源码：** [`catlass.core_api.add`](../../../catlass/core_api.py#L6195)

功能说明：

vector 逐元素加法（支持 vector–vector 与 vector–scalar）。
`VectorSSA` 在不需要 `mask` 时也可通过 `+` / `__radd__` 调用本接口。

函数原型：

```python
tla.add(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；支持 vector–vector 与 vector–scalar。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg、y_reg 已从 UB load 到寄存器。
    z = x_reg + y_reg            # 等价于 tla.add(x_reg, y_reg)
    z = x_reg + 1.0              # vector–scalar，走 __add__/__radd__
    # 带 mask：先 create_mask 得到 MaskSSA，再传 mask=（不能用运算符重载）。
    m = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Float16)
    z = tla.add(x_reg, y_reg, mask=m)
    # 计算后：有效元素为和；其余元素不参与。
```

---

#### `sub`

**源码：** [`catlass.core_api.sub`](../../../catlass/core_api.py#L6241)

功能说明：

vector 逐元素减法。
`VectorSSA` 在不需要 `mask` 时也可通过 `-`（`__sub__`）调用本接口。

函数原型：

```python
tla.sub(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数（被减数）。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数（减数）。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg / y_reg 为源 vector。
    z = x_reg - y_reg                 # 等价于 tla.sub(x_reg, y_reg)
    z = tla.sub(x_reg, y_reg, mask=m) # 需要 mask 时用函数形式
    # 计算后：z 为逐元素差。
```

---

#### `mul`

**源码：** [`catlass.core_api.mul`](../../../catlass/core_api.py#L6278)

功能说明：

vector 逐元素乘法。
`VectorSSA` 在不需要 `mask` 时也可通过 `*` / `__rmul__` 调用本接口。

函数原型：

```python
tla.mul(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg 为激活；scale 可为 vector 或标量。
    z = x_reg * y_reg      # 等价于 tla.mul(x_reg, y_reg)
    z = x_reg * 2.0        # vector–scalar
    z = tla.mul(x_reg, y_reg, mask=m)
    # 计算后：z 为逐元素积。
```

---

#### `max`

**源码：** [`catlass.core_api.max`](../../../catlass/core_api.py#L6181)

功能说明：

vector 逐元素最大值。

函数原型：

```python
tla.max(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.max(x_reg, y_reg)
```

---

#### `min`

**源码：** [`catlass.core_api.min`](../../../catlass/core_api.py#L6221)

功能说明：

vector 逐元素最小值。

函数原型：

```python
tla.min(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.min(x_reg, y_reg)
```

---

#### `div`

**源码：** [`catlass.core_api.div`](../../../catlass/core_api.py#L6407)

功能说明：

vector 逐元素除法。
`VectorSSA` 在不需要 `mask` 时也可通过 `/`（`__truediv__`）调用本接口。

函数原型：

```python
tla.div(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数（被除数）。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数（除数）。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg 为被除数；y_reg / 标量为除数。
    z = x_reg / y_reg                 # 等价于 tla.div(x_reg, y_reg)
    z = tla.div(x_reg, y_reg, mask=m)
    # 计算后：z 为逐元素商。
```

---

### 逻辑计算

#### `bitwise_not`

**源码：** [`catlass.core_api.bitwise_not`](../../../catlass/core_api.py#L6161)

功能说明：

逐元素按位/逻辑非（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_not(operand: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `operand`（`VectorSSA | MaskSSA`）：按位取反的源操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m2 = tla.bitwise_not(m)
```

---

#### `bitwise_and`

**源码：** [`catlass.core_api.bitwise_and`](../../../catlass/core_api.py#L6837)

功能说明：

逐元素按位与（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_and(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位与左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位与右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_and(m0, m1)
```

---

#### `bitwise_or`

**源码：** [`catlass.core_api.bitwise_or`](../../../catlass/core_api.py#L6875)

功能说明：

逐元素按位或（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_or(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位或左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位或右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_or(m0, m1)
```

---

#### `bitwise_xor`

**源码：** [`catlass.core_api.bitwise_xor`](../../../catlass/core_api.py#L6913)

功能说明：

逐元素按位异或（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_xor(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位异或左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位异或右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_xor(m0, m1)
```

---

#### `shift_left`

**源码：** [`catlass.core_api.shift_left`](../../../catlass/core_api.py#L7489)

功能说明：

逐元素左移，移位量可按通道或标量指定。

函数原型：

```python
tla.shift_left(source: VectorSSA, shift: Any, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `source`（`VectorSSA`）：待移位的源向量。必填。
- `shift`（`VectorSSA | Numeric | int`）：每通道移位量。与 `source` 元素宽度相同的有符号 `VectorSSA`，或可转换为 i16 的整数标量。必填。
- `mask`（`MaskSSA | None`）：执行掩码。可选，默认 `None`（全通道使能）；被掩码的通道置零。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 源元素须为有符号 8/16/32 位整数（i8/i16/i32）；不支持 64 位和无符号移位。
- 移位量须为非负数。

调用示例：

```python
with tla.vec.func(mode="simd"):
    _tile = tla.shift_left(tile, bit_shift_reg, mask=mask)
```

---

#### `shift_right`

**源码：** [`catlass.core_api.shift_right`](../../../catlass/core_api.py#L7525)

功能说明：

逐元素右移，移位量可按通道或标量指定。带符号源向量使用算术右移。

函数原型：

```python
tla.shift_right(source: VectorSSA, shift: Any, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `source`（`VectorSSA`）：待移位的源向量。必填。
- `shift`（`VectorSSA | Numeric | int`）：每通道移位量。与 `source` 元素宽度相同的有符号 `VectorSSA`，或可转换为 i16 的整数标量。必填。
- `mask`（`MaskSSA | None`）：执行掩码。可选，默认 `None`（全通道使能）；被掩码的通道置零。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 源元素须为有符号 8/16/32 位整数（i8/i16/i32）；不支持 64 位和无符号移位。
- 移位量须为非负数。

调用示例：

```python
with tla.vec.func(mode="simd"):
    _tile = tla.shift_right(tile, bit_shift_reg, mask=mask)
```

---

### 比较与选择

#### `where`

**源码：** [`catlass.core_api.where`](../../../catlass/core_api.py#L6579)

功能说明：

按 mask 在两路 vector 间逐元素选择；在 SIMT 区域内也可在两路 per-thread 标量间选择。

函数原型：

```python
tla.where(mask: Any, x: Any, y: Any) -> Any
```

参数说明：

- `mask`（`MaskSSA | Bool`）：选择掩码；真取 `x`，假取 `y`。可为整 vector 的 `MaskSSA`，或 SIMT 区域内比较得到的 `Bool`。必填。
- `x`（`VectorSSA | Numeric`）：掩码为真时的取值。必填。
- `y`（`VectorSSA | Numeric`）：掩码为假时的取值。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`mask`/`x`/`y` 的有效元素布局须匹配。
- per-thread 形式要求 `mode="simt"`，并生成 `tla.simt_where`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.where(m, x_reg, y_reg)
```


#### `cmp`

**源码：** [`catlass.core_api.cmp`](../../../catlass/core_api.py#L6761)

功能说明：

vector 比较，返回 mask。

函数原型：

```python
tla.cmp(lhs: VectorSSA, rhs: VectorSSA | Numeric | bool | int | float, mode: str, *, mask: MaskSSA | None = None) -> MaskSSA
```

参数说明：

- `lhs`（`VectorSSA`）：比较左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：比较右操作数（可为标量或 vector）。必填。
- `mode`（`str`）：比较模式，如 `'eq'` / `'lt'` / `'gt'` 等。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`mode` 须为支持的比较模式名。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m = tla.cmp(x_reg, y_reg, mode="lt")
```

---

### 数据填充

#### `full`

**源码：** [`catlass.core_api.full`](../../../catlass/core_api.py#L5319)

功能说明：

用 Python 标量字面量填充一维 vector SSA。

函数原型：

```python
tla.full(value: bool | int | float | Numeric, dtype: type[Numeric]) -> VectorSSA
```

参数说明：

- `value`（`bool | int | float | Numeric`）：填充常量。必填。
- `dtype`（`type[Numeric]`）：vector 元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`value` 须为 Python 标量字面量。

调用示例：

```python
with tla.vec.func(mode="simd"):
    zeros = tla.full(0.0, dtype=tla.Float32)
```

---

#### `arange`

**源码：** [`catlass.core_api.arange`](../../../catlass/core_api.py#L5392)

功能说明：

创建单调递增或递减的一维 vector SSA（`base` + `order`）。

函数原型：

```python
tla.arange(base: bool | int | float | Numeric = 0, *, order: str = 'increase', dtype: type[Numeric]) -> VectorSSA
```

参数说明：

- `base`（`bool | int | float | Numeric`）：起始基数。可选，默认 `0`。
- `order`（`str`）：`'increase'` 递增或 `'decrease'` 递减。可选，默认 `'increase'`。
- `dtype`（`type[Numeric]`）：vector 元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`order` 仅支持 `increase` / `decrease`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    lane_idx = tla.arange(base=0, order="increase", dtype=tla.Int32)
```

---

### 离散与聚合

#### `gather`

**源码：** [`catlass.core_api.gather`](../../../catlass/core_api.py#L6951)

功能说明：

按 vector 下标从 UB tensor gather 元素。

函数原型：

```python
tla.gather(x: Tensor, y: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `x`（`Tensor`）：被 gather 的源 tile / 表。必填。
- `y`（`VectorSSA`）：索引 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；源 tensor 须位于 UB。

调用示例：

```python
with tla.vec.func(mode="simd"):
    vals = tla.gather(ub_tile, idx_reg)
```

---

### 数据重排

#### `interleave`

**源码：** [`catlass.core_api.interleave`](../../../catlass/core_api.py#L6054)

功能说明：

两路 vector 交插，返回高低两半。

函数原型：

```python
tla.interleave(src0: VectorSSA, src1: VectorSSA) -> tuple[VectorSSA, VectorSSA]
```

参数说明：

- `src0`（`VectorSSA`）：偶数路输入 vector 寄存器。必填。
- `src1`（`VectorSSA`）：奇数路输入 vector 寄存器。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；两路 vector 的元素类型与元素个数须匹配。

调用示例：

```python
with tla.vec.func(mode="simd"):
    lo, hi = tla.interleave(a, b)
```

---

#### `deinterleave`

**源码：** [`catlass.core_api.deinterleave`](../../../catlass/core_api.py#L6107)

功能说明：

两路 vector 解交插，返回高低两半。

函数原型：

```python
tla.deinterleave(src0: VectorSSA, src1: VectorSSA) -> tuple[VectorSSA, VectorSSA]
```

参数说明：

- `src0`（`VectorSSA`）：交错输入的前半 / 一路。必填。
- `src1`（`VectorSSA`）：交错输入的后半 / 另一路。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；两路 vector 的元素类型与元素个数须匹配。

调用示例：

```python
with tla.vec.func(mode="simd"):
    even, odd = tla.deinterleave(a, b)
```

---

### 数据压缩

#### `squeeze`

**源码：** [`catlass.core_api.squeeze`](../../../catlass/core_api.py#L6608)

功能说明：

按 mask 将选中元素压缩到结果低位。

函数原型：

```python
tla.squeeze(src: VectorSSA, mask: MaskSSA) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）：待压缩的源 vector。必填。
- `mask`（`MaskSSA`）：保留元素的掩码。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    packed = tla.squeeze(src, m)
```

### 类型转换

#### `VectorSSA.to`

**源码：** [`catlass.core_api.VectorSSA.to`](../../../catlass/core_api.py#L535)

功能说明：

将寄存器内的 vector 转换为另一种元素类型。

lowering 支持有符号整数 `tla.Int8` ~ `tla.Int64`、浮点 `tla.Float16` / `tla.BFloat16` / `tla.Float32`，以及两种 OCP fp8 格式 `tla.Float8E4M3FN` / `tla.Float8E5M2`。无符号整数、`tla.Bool`（i1）、`tla.Float64` 与 `tla.Float8E8M0` 会被拒绝。

fp8 操作数可与 `tla.Float32`、`tla.Float16`、`tla.BFloat16` 互相转换。其中只有 f32 这一对是单条指令：f16 与 bf16 两对由 lowering 展开为经过 f32 的两次 cast，扩宽一侧精确、收窄一侧只舍入一次；`f8e5m2 -> f16` 则展开为一次整数扩宽加一次移位，因为这两种格式共享同一指数字段。fp8 之间没有重编码路径——两者指数范围不同，属于重量化而非 cast——也没有 fp8 与整数之间的路径。

函数原型：

```python
VectorSSA.to(dst_type: Any, params: CastParams, mask: Any | None = None) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）：目标元素类型。必填。
- `params`（`CastParams`）：舍入模式、饱和模式与寄存器 slot。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示所有 lane 使能。选填，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- `reg_slot` 的 `TWO` 与 `THREE` 指向 pack 的四分之一，仅在位宽变化为 4 倍的整数 cast 上可达。

调用示例：

```python
trait = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
with tla.vec.func(mode="simd"):
    wide = narrow.to(tla.Float32, trait)
```

---

## 同步控制

核内 / 跨核 flag、pipe barrier、mutex 与本地内存屏障。

### `flag`

**源码：** [`catlass.core_api.flag`](../../../catlass/core_api.py#L4496)

功能说明：

在两条 pipe 之间创建管内同步 flag。

函数原型：

```python
tla.flag(name: str, src_pipe: PipeLike, dst_pipe: PipeLike) -> TlaFlag
```

参数说明：

- `name`（`str`）：管内 flag 名称。必填。
- `src_pipe`（`PipeLike`）：源 pipe 标识（例如 `tla.arch.MTE2`）。必填。
- `dst_pipe`（`PipeLike`）：目的 pipe 标识（例如 `tla.arch.VECTOR`）。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 创建的是 flag 句柄；`set_flag`/`wait_flag` 须在 cube/vector 区域内成对使用。

调用示例：

```python
# MTE2 完成 GM→UB 拷贝后，VECTOR 侧才可消费该 UB tile。
ub_loaded = tla.flag("ub_loaded", src_pipe=tla.arch.MTE2, dst_pipe=tla.arch.VECTOR)
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
    tla.set_flag(ub_loaded)   # 拷贝后：标记 UB 数据就绪
    tla.wait_flag(ub_loaded)  # 计算前：等待就绪
```

---

### `cross_flag`

**源码：** [`catlass.core_api.cross_flag`](../../../catlass/core_api.py#L4549)

功能说明：

创建命名的跨核同步 flag。源/目的 pipe 由对应 set/wait 指定；`mode=4` 为 1:1 AIC↔AIV，并按 AIV0/AIV1 独立寻址。

函数原型：

```python
tla.cross_flag(name: str, *, mode: int = 2) -> TlaCrossFlag
```

参数说明：

- `name`（`str`）：跨核 flag 名称。必填。
- `mode`（`int`）：跨核同步模式。可选，默认 `2`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `mode` 仅支持 0/1/2/4；源/目的 pipe 由对应 set/wait 指定。

调用示例：

```python
cf = tla.cross_flag("aic_aiv", mode=2)
```

---

### `cross_core_set_flag`

**源码：** [`catlass.core_api.cross_core_set_flag`](../../../catlass/core_api.py#L4625)

功能说明：

从指定 `pipe` 置位跨核 flag；`mode=4` 时需提供 `aiv_id`（0 或 1）。

函数原型：

```python
tla.cross_core_set_flag(cross_flag_value: CrossFlagLike, pipe: PipeLike, aiv_id: int | None = None) -> None
```

参数说明：

- `cross_flag_value`（`CrossFlagLike`）：由 `tla.cross_flag` 得到的跨核 flag。必填。
- `pipe`（`PipeLike`）：设置 flag 的 pipe。必填。
- `aiv_id`（`int | None`）：目标 AIV 编号；缺省表示广播/默认路由。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；`mode=4` 时 `aiv_id` 须为 0 或 1。

调用示例：

```python
with tla.cube():
    tla.cross_core_set_flag(cf, tla.arch.CUBE)
    # mode=4: tla.cross_core_set_flag(cf, tla.arch.CUBE, aiv_id=0)
```

---

### `cross_core_wait_flag`

**源码：** [`catlass.core_api.cross_core_wait_flag`](../../../catlass/core_api.py#L4669)

功能说明：

在指定 `pipe` 上等待跨核 flag；`mode=4` 时需提供 `aiv_id`（0 或 1）。

函数原型：

```python
tla.cross_core_wait_flag(cross_flag_value: CrossFlagLike, pipe: PipeLike, aiv_id: int | None = None) -> None
```

参数说明：

- `cross_flag_value`（`CrossFlagLike`）：由 `tla.cross_flag` 得到的跨核 flag。必填。
- `pipe`（`PipeLike`）：执行 wait 的 pipe。必填。
- `aiv_id`（`int | None`）：目标 AIV 编号；缺省表示广播/默认路由。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；`mode=4` 时 `aiv_id` 须为 0 或 1。

调用示例：

```python
with tla.vector():
    tla.cross_core_wait_flag(cf, tla.arch.VECTOR)
```

---

### `set_flag`

**源码：** [`catlass.core_api.set_flag`](../../../catlass/core_api.py#L4712)

功能说明：

置位同核同步 flag。

函数原型：

```python
tla.set_flag(flag_value: FlagLike) -> None
```

参数说明：

- `flag_value`（`FlagLike`）：由 `tla.flag` 得到的核内 pipe flag。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。

调用示例：

```python
with tla.vector():
    tla.set_flag(ub_loaded)
```

---

### `wait_flag`

**源码：** [`catlass.core_api.wait_flag`](../../../catlass/core_api.py#L4738)

功能说明：

等待同核同步 flag。

函数原型：

```python
tla.wait_flag(flag_value: FlagLike) -> None
```

参数说明：

- `flag_value`（`FlagLike`）：由 `tla.flag` 得到的核内 pipe flag。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。

调用示例：

```python
with tla.vector():
    tla.wait_flag(ub_loaded)
```

---

### `pipe_barrier`

**源码：** [`catlass.core_api.pipe_barrier`](../../../catlass/core_api.py#L4764)

功能说明：

插入指定硬件 pipe 的屏障。

函数原型：

```python
tla.pipe_barrier(pipe: PipeLike) -> None
```

参数说明：

- `pipe`（`PipeLike`）：要插入 barrier 的 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。

调用示例：

```python
with tla.vector():
    tla.pipe_barrier(tla.arch.MTE2)
```

---

### `mutex`

**源码：** [`catlass.core_api.mutex`](../../../catlass/core_api.py#L4807)

功能说明：

创建与语义资源关联的 mutex 句柄。

函数原型：

```python
tla.mutex(resource: str, id: int = -1) -> TlaMutex
```

参数说明：

- `resource`（`str`）：互斥资源名。必填。
- `id`（`int`）：互斥实例编号；`-1` 表示默认。可选，默认 `-1`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `resource` 非空；`id` 为 -1 或 0..31。

调用示例：

```python
mtx = tla.mutex("l1_buf", id=0)
```

---

### `mutex_guard`

**源码：** [`catlass.core_api.mutex_guard`](../../../catlass/core_api.py#L4855)

功能说明：

创建上下文管理器，对代码块自动推断 mutex 访问。

函数原型：

```python
tla.mutex_guard(*mutexes: MutexLike) -> _MutexGuard
```

参数说明：

- *`mutexes`*（`MutexLike`）：一个或多个 mutex 对象。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 块内须调用 `copy` 或 `mmad`；guard 内不可再显式 lock/unlock。

调用示例：

```python
with tla.mutex_guard(mtx):
    tla.copy(dst, src)
```

---

### `mutex_lock`

**源码：** [`catlass.core_api.mutex_lock`](../../../catlass/core_api.py#L4896)

功能说明：

从指定 pipe 获取 mutex。

函数原型：

```python
tla.mutex_lock(mutex_value: MutexLike, *, pipe: PipeLike) -> None
```

参数说明：

- `mutex_value`（`MutexLike`）：要加锁的 mutex。必填。
- `pipe`（`PipeLike`）：加锁所在 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；须指定 `pipe`。

调用示例：

```python
tla.mutex_lock(mtx, pipe=tla.arch.MTE2)
```

---

### `mutex_unlock`

**源码：** [`catlass.core_api.mutex_unlock`](../../../catlass/core_api.py#L4926)

功能说明：

从指定 pipe 释放 mutex。

函数原型：

```python
tla.mutex_unlock(mutex_value: MutexLike, *, pipe: PipeLike) -> None
```

参数说明：

- `mutex_value`（`MutexLike`）：要解锁的 mutex。必填。
- `pipe`（`PipeLike`）：解锁所在 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；须指定 `pipe`。

调用示例：

```python
tla.mutex_unlock(mtx, pipe=tla.arch.MTE2)
```

---

### `local_mem_bar`

**源码：** [`catlass.core_api.local_mem_bar`](../../../catlass/core_api.py#L4956)

功能说明：

在 `vec.func` 内插入本地内存屏障（按 `MemType` 对编码）。

函数原型：

```python
tla.local_mem_bar(src: MemType, dst: MemType)
```

参数说明：

- `src`（`MemType`）：源本地内存类型。必填。
- `dst`（`MemType`）：目的本地内存类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`(src, dst)` 须为支持的 `MemType` 对。

调用示例：

```python
with tla.vec.func(mode="simd"):
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
```

---

## 系统变量访问

挂在 `tla.arch` 上的架构属性（布局标签、pipe、block 辅助等）。

### `arch`

**源码：** [`catlass.core_api.arch`](../../../catlass/core_api.py#L7418)

功能说明：

`tla.arch` 下的架构属性组：layout tag、pipe 标识、片上 memory-scope token，
以及 block / SIMT 辅助接口。

函数原型：

```python
tla.arch
```

参数说明：

- 布局标签（`_LayoutTag`，供 `make_layout` / `make_tensor` /
  `make_tensor_like` 使用）：`RowMajor`、`ColumnMajor`、`zN`、`nZ`、
  `L0Clayout`、`zNUnAlign`。
- Pipe 标识（供 `flag` / `pipe_barrier` / `mutex_*` / 跨核同步使用）：
  `SCALAR`、`VECTOR`、`CUBE`、`MTE1`、`MTE2`、`MTE3`、`FIX`。
- Memory-scope token（供 `local_mem_bar` 等相关接口使用）：`L1`、`L0A`、`L0B`、`L0C`、`UB`。
- 可调用接口（返回 `Int32` 或如下注明的三元组）：
  - `block_idx()`：当前 AI 核在本次 launch 中的 block 索引。
  - `block_num()`：本次 launch 的 block（AI 核）数量。
  - `sub_block_idx()`：当前 block 内的 sub-block 索引。
  - `thread_idx()`：SIMT thread block 内的线程索引 `(x, y, z)`（仅可在
    `tla.vec.func(mode="simt")` 内使用）。
  - `thread_block_dim()`：SIMT thread-block 规模 `(x, y, z)`（仅可在
    `tla.vec.func(mode="simt")` 内使用）。
  - `sync_threads()`：对当前 SIMT `tla.vec.func` 内线程做 barrier（仅
    `mode="simt"`）。
  - `get_capacity_in_bytes(mem_scope)`：返回编译目标上某片上存储空间的字节容量。入参为 `tla.AddressSpace`（`tla.AddressSpace.l1` / `l0a` / `l0b` / `l0c` / `ub`）。返回普通 `int`；host 侧与 kernel 内均可使用（kernel 内会折叠为常量）。
  - `get_dyn_ub(dtype, byte_alignment=None)`：返回动态 UB 区域的指针——该区域
    的大小由 launch 决定，而非编译期决定。区域起始于所有 `tla.allocate` 之上，
    launch 时用 `ub=` 指定实际提供多少字节。每个 kernel 至多声明
    一个区域：需要多块缓冲时，用指针算术在该区域内切分。按 kernel 实际使用量
    申请即可——SIMT 场景下未被 kernel 占用的 UB 会成为 Data Cache，申请过多会
    让 Data Cache 变小从而拖慢访存。

约束说明：

- layout tag / pipe 标识 / memory-scope token 是 `tla.arch` 对象上的普通属性
  （Python 没有 C++ 式命名空间）；它们本身不会生成计算 op。
- `block_idx` / `block_num` / `sub_block_idx` / `thread_idx` /
  `thread_block_dim` / `sync_threads` 为可调用对象，须在 `@tla.kernel`
  装饰的 kernel 函数体内使用。
- `thread_idx` / `thread_block_dim` / `sync_threads` 还须嵌套在
  `tla.vec.func(mode="simt")` 内。

调用示例：

```python
# make_layout / make_tensor 用的 layout tag：
tag = tla.arch.RowMajor
# flag / barrier / mutex 用的 pipe 标识：
pipe = tla.arch.MTE2
# 运行时 block 辅助：
bid = tla.arch.block_idx()
nblocks = tla.arch.block_num()
# 片上存储容量（host 侧或 kernel 内均可）：
l1_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1)
ub_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
# 动态 UB：大小在 launch 时确定，基址为编译期常量。
base = tla.arch.get_dyn_ub(tla.Float32, byte_alignment=256)
lo = tla.make_tensor_like(base, gm_a, tla.arch.RowMajor)
hi = tla.make_tensor_like(base + HALF, gm_b, tla.arch.RowMajor)
# host 侧：artifact(..., ub=64 * 1024)
```

---

## 资源管理

片上缓冲分配。

### `allocate`

**源码：** [`catlass.core_api.allocate`](../../../catlass/core_api.py#L7181)

功能说明：

分配本地内存并返回类型化指针。

函数原型：

```python
tla.allocate(shape: ShapeLike, dtype: type[Numeric], mem_scope: AddressSpace, byte_alignment: int) -> Pointer
```

参数说明：

- `shape`（`ShapeLike`）：分配块的形状。必填。
- `dtype`（`type[Numeric]`）：元素数值类型。必填。
- `mem_scope`（`AddressSpace`）：地址空间（如 L1 / UB）。必填。
- `byte_alignment`（`int`）：字节对齐要求。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `mem_scope` 须为片上地址空间（l1/l0a/l0b/l0c/ub），不可为 gm/generic；`shape` 须全静态。

调用示例：

```python
ptr = tla.allocate(
    shape=(256, 128),
    dtype=tla.Float16,
    mem_scope=tla.AddressSpace.ub,
    byte_alignment=32,
)
```

---

## 调试接口

kernel 内标量 / tensor 调试打印。

### `print`

**源码：** [`catlass.core_api.print`](../../../catlass/core_api.py#L3359)

功能说明：

在 `cube` / `vector` 区域内打印标量、格式化标量字符串，或打印 GM、UB、L1、L0C tensor 的物理前缀。

函数原型：

```python
tla.print(*args: object, **kwargs: object) -> None
```

参数说明：

- *`args`*（`object`）：要打印的值（可变位置参数）。必填。
- **`kwargs`**（`object`）：不接受关键字参数；传入则报错。可选。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。
- GM/UB 支持 rank-1/rank-2 物理前缀；L1/L0C 仅支持 C310 AIC 和经验证的 CANN 9.1 DebugTunnel ABI。
- L1 须为 32 字节对齐的 zN/nZ 稠密前缀，C0 维度占 32 字节，最多打印 8 个元素，并要求驱动提供 DebugBus HAL 映射接口。
- L0C 须为 32 字节对齐的 L0Clayout 16x16 完整分形（256 个元素）；L0A/L0B 不支持。
- Tensor dtype 支持 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。

调用示例：

```python
with tla.vector():
    tla.print(x_scalar)
    tla.print(x_ub, 64)  # tensor + prefix length
```

---

## 作用域和控制流

Cube / Vector / `vec.func` 区域以及 kernel 侧循环范围。

从这些 region 内调用的外层 `def`、`tla.Constexpr` 函数入参与 `@tla.jit` helper，须遵守本节各 region 约束，以及所用算子条目的「约束说明」。

### `range`

**源码：** [`catlass.core_api.range`](../../../catlass/core_api.py#L5008)

功能说明：

创建前端动态循环范围（支持 Python `range` 形态）。

函数原型：

```python
tla.range(start: IndexLike, end: IndexLike | None = None, step: IndexLike | None = None) -> _ast_helpers.FrontendRange
```

参数说明：

- `start`（`IndexLike`）：循环起点（或当 `end` 缺省时的终点，此时起点为 0）。必填。
- `end`（`IndexLike | None`）：循环终点（不含）；缺省时 `start` 视为终点。可选，默认 `None`。
- `step`（`IndexLike | None`）：步长；缺省为 1。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 动态范围；循环体须符合前端动态 for 约束。

调用示例：

```python
for i in tla.range(0, n, 1):
    ...
```

---

### `range_constexpr`

**源码：** [`catlass.core_api.range_constexpr`](../../../catlass/core_api.py#L5058)

功能说明：

创建前端静态范围，用于可展开的 Python 循环。

函数原型：

```python
tla.range_constexpr(start: int, end: int | None = None, step: int | None = None) -> range
```

参数说明：

- `start`（`int`）：编译期循环起点（或当 `end` 缺省时的终点）。必填。
- `end`（`int | None`）：编译期循环终点（不含）。可选，默认 `None`。
- `step`（`int | None`）：编译期步长；缺省为 1。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 起止与步长须为编译期常量，用于可展开循环。
- 边界也可来自 `tla.as_numeric(...)` 等编译期 Numeric 值。
- 迭代次数达到 64 次及以上时发出 `DSLOptimizationWarning`，但继续展开；大循环应优先用 `tla.range(...)`。

调用示例：

```python
for k in tla.range_constexpr(0, 4):
    ...
```

---

### `cube`

**源码：** [`catlass.core_api.cube`](../../../catlass/core_api.py#L5109)

功能说明：

进入 cube 核区域（矩阵乘与相关搬运）。

函数原型：

```python
tla.cube() -> TlaRegion
```

参数说明：

无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 用于包裹 cube 侧矩阵乘与相关搬运。

调用示例：

```python
with tla.cube():
    tla.mmad(acc=l0c, lhs=l0a, rhs=l0b, init_c=True)
```

---

### `vector`

**源码：** [`catlass.core_api.vector`](../../../catlass/core_api.py#L5131)

功能说明：

进入 vector 核区域（vector 搬运与同步）。

函数原型：

```python
tla.vector() -> TlaRegion
```

参数说明：

无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 用于包裹 vector 侧搬运与同步；`tla.vec.func` 须嵌套在其内。

调用示例：

```python
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
```

---

### `vec.func`

**源码：** [`catlass.core_api._vec_func`](../../../catlass/core_api.py#L5165)

功能说明：

进入寄存器 vector / mask 计算用的 vector 函数区域（`tla.vec.func`）。

函数原型：

```python
tla.vec.func(*, mode: str = 'simd', thread_block_dim: int | tuple[int, int, int] | list[int] | None = None) -> TlaRegion
```

参数说明：

- `mode`（`str`）：执行模式；`simd`（默认）或 `simt`。可选，默认 `"simd"`。
- `thread_block_dim`（`int | tuple[int, int, int] | list[int] | None`）：SIMT thread-block 形状；仅在 `mode="simt"` 时有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须嵌套在 `tla.vector()` 内。
- 寄存器 vector / mask API 与 `local_mem_bar` 须在此区域内调用。

调用示例：

```python
with tla.vector():
    with tla.vec.func(mode="simd"):
        z = tla.add(x_reg, y_reg)
```

---
