---
nav_order: 3
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/data_movement.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 数据搬运

[语言参考](index.md) | [API 速查](../quick_reference.md)

片上与全局内存之间的 tensor 拷贝，以及 UB 寄存器 load/store。

## `tla.copy`

**源码：** [`catlass.core_api.copy`](../../../../catlass/core_api.py)

功能说明：

将 tile 从 `src` 拷到 `dst`。支持的通路与 dtype 见约束说明后的表格。

函数原型：

```python
tla.copy(dst: Tensor, src: Tensor, params: CopyParams | None = None, *, scale: Tensor | None = None, cache_policy: CachePolicy | None = None) -> None
```

参数说明：

- `dst`（`Tensor`）：目的 tile。必填。
- `src`（`Tensor`）：源 tile。必填。
- `params`（`CopyParams | None`）：通路参数：`CopyL0C2DstParams`、
  `CopyUbToGmParams` 或 `CopyPadParams`。默认 `None`。
  - `CopyPadParams`：开启 GM→UB DataCopyPad（`isPad=true`）。支持的通路、
    dtype 与填充规则见下方约束说明。
  - `left_padding` / `right_padding`：最内维每侧填充的**元素个数**（编译期
    int，或运行期整型/index SSA）。
  - `padding_value`：填充区写入值（编译期字面量/Numeric，或运行期整型 bit
    pattern）。默认 `0` 表示零填充。
- `scale`（`Tensor | None`）：L1 上的 OCP `e8m0` 共享指数 tile，用于
  L1→L0A/L0B 的 MX load。以 `i8`/`u8` 存储；目的为 L0A 时 tag
  `tla.arch.zZMxScale`，为 L0B 时 tag `tla.arch.nNMxScale`。默认 `None`。
- `cache_policy`（`CachePolicy | None`）：可选的 L2 缓存提示，作用于本次
  copy 的 GM 端点（`DISABLE` / `NORMAL`）。`GM→*` 打在 GM 源上，`*→GM` 打在
  GM 目的上；不存在 GM→GM 拷贝路由。默认 `None`（省略 hint，BC 使用
  NORMAL）。非 GM 路由上显式传入任意值（含 `NORMAL`）都会报错，应省略该参数。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。
- 拷贝大小按各 tile 的 `origin_shape`。
- UB 到 GM 和 L0C 到 GM 的搬运可通过 `params.atomic_mode`
  指定本次写回的运算。`AtomicMode.ADD`、`MAX`、`MIN` 分别将写入值与 GM 旧值相加、
  取最大值或取最小值。默认的 `NONE` 直接覆盖旧值。原子写回支持的 GM 元素类型为
  i8、i16、i32、f16、bf16 和 f32。L0C 搬出的值先经过所选的类型转换、量化或 ReLU，再参与原子运算。
  UB 到 GM 搬运可省略 `params`，或传入 `CopyUbToGmParams`。该参数对象的
  `atomic_mode` 必须是 `AtomicMode` 枚举成员，默认值为 `AtomicMode.NONE`。
- 当 `params` 为 `CopyPadParams`：仅 GM→UB，layout 为 RowMajor→RowMajor 或
  VectorLayout→VectorLayout；元素 dtype 为
  f32/f16/bf16/i32/u32/i16/u16/i8/u8/i1/i64/u64/f64（不含 fp8/hif8/fp4）。
  侧边计数不得为 bool/`tla.Bool`/i1；静态侧边须满足 pad 字节 ≤32B；两端
  extent 亦为静态时，目的最内维 ≥ src + left + right。
- `CopyPadParams.padding_value` 须与源元素 dtype 一致（i1：bool/`tla.Bool`；
  整型：裸 int 或同 dtype Numeric；浮点：裸 float 或同 dtype 浮点 Numeric；
  裸 int `0` 为 dataclass 默认零填充）。运行期非 i1 按 i64 bit pattern；
  运行期 i1 SSA 不支持。对 8B（`i64`/`u64`/`f64`）仅保证静态零填充；运行期
  不做数值拦截，BC 恒填 0。
- Ascend 950 上，两个逻辑秩为 3 的视图 `(B, M, N)` 启用批量 GM→L1 布局转换。
  FP8 支持 RowMajor→zN 和 ColumnMajor→nZ；E8M0 支持下表的四条 GM MX scale 通路。
  批次数取自 `origin_shape[0]`，源、目的矩阵间距分别取各自 layout 的首维 stride，
  以元素为单位；内部矩阵 stride 保持二维语义。两端逻辑 extent 必须一致，包括运行时值。
  批次数或矩阵 extent 为零时不发起 DMA。
  其他三维 layout/dtype 组合在编译期报错，不回退到二维拷贝。
- 批量拷贝要求 `params=None`、`scale=None`，批次数为 0～65535，目的批次 stride
  为正、按 32 字节对齐且不超过 65535 个 32 字节块。E8M0 源矩阵间距必须保持指数对。
  运行时几何参数须满足相同约束。调用方提供足够底层存储，并保证目的矩阵互不重叠；
  允许交错 NZ 布局。
- packed fp4 在搬运前转换描述符的通路（GM→UB、UB→GM 与 UB→L1）按字节
  搬运：两个 fp4 元素共用一个字节，因此各 tile 最内层的 `origin_shape`、
  最内层起始坐标，以及转换时被折半的 stride，三者均须为偶数，任一为奇数
  都会错开半个字节。被折半的 stride 为：`RowMajor` tile 的行 stride，
  `zN` tile 的 C0 列间距；因此 UB→L1 的 `zN` 目的 tile 同样受此约束。
  仅在运行期才确定的取值无法证明为偶数，因此一并拒绝。Cube 的 GM→L1 fp4
  通路保持 fp4 计数，不受此约束；`tla.experimental.mx_to` 仍接受奇数尺寸
  的 tile，该 tile 只是无法经由字节描述符搬运。
- `scale` 仅对 L1→L0A/L0B 有效；由此加载的 tile 随后须由 `tla.mmad_mx` 消费。
- MX scale 目的（`zZMxScale` / `nNMxScale`）要求源 tile 覆盖 L1 整行，
  并按行堆叠 scale block。仅在行 pitch 为编译期常量时可检查。
- UB→UB 按整 32 字节块计数，行向上取整。起始未 32B 对齐、或多行 pitch
  容不下取整行时拒绝。目的 tag 为 `ColumnMajor` 时为转置：tile 形状须为
  整块 16×C0，1 字节类型为 32×32。
- `GM`→`L1` / `L1`→`biasbuf` 连续平铺用 rank-1 `VectorLayout`；多行 bias
  tile 不可表示，请切出单行。
- PER_CHANNEL 反量化参数（`CopyL0C2DstParams.quant_tensor`）为位于 `L1`
  地址空间的 rank-1 `VectorLayout` tile，元素类型 `u64`、单位 stride。
  每个元素携带一个输出通道内全部元素共享的反量化参数，因此长度须与目的
  tile 通道数精确一致（`tile_view` 裁剪后的 `originShape[1]`）。两侧长度
  均为静态时不相等即编译报错。
- **预打包 `GM` tile：** 已按分形顺序存放的缓冲区可用 `zN` 或转置排布的
  `nZ` tag 描述，并拷贝到相同布局的 L1 tile，不再重排数据。GM 源 tag 仅使用
  对齐的 `zN` / `nZ`；M 轴紧凑存储的 `zNUnAlign` 用于 UB→L1，GM 端的
  `zNUnAlign` 则表示供 `tla.experimental.img2col` 使用的 NC1HWC0 特征图。
  两端容量均须覆盖 Padding：`zN` 为 `ceil(M/16) * 16` 行，`nZ` 为
  `ceil(M/C0) * C0` 行、`ceil(N/16) * 16` 列；搬运以完整分形为单位，
  最后一个行组也会完整搬运。
- 整 tile DMA 使用 `tla.copy`。寄存器级 UB 非对齐访问请改用
  `tensor.load` / `tensor.store`，并传入 `UnalignLoadParams` /
  `UnalignStoreParams`。
- **尺寸未按 32 字节对齐时：** C0 为 32 字节。ND GM↔UB 时，优先让主导维的
  字节数是 32 的倍数。否则保持 `origin_shape` 为真实尺寸并放大物理
  layout，或改用非对齐 load/store。

无 scale 的 Cube 拷贝通路，`tla.copy(dst, src, scale=None)`

|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|
|----------|------------|----------|------------|--------|
|`gm`|`RowMajor`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`i1`,`hif8`,`i16`,`i32`,`u8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`ColumnMajor`|`l1`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`i1`,`hif8`,`i16`,`i32`,`u8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`zN`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`i16`,`i32`,`u8`,`u16`,`u32`,`i1`,`hif8`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`nZ`|`l1`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`i16`,`i32`,`u8`,`u16`,`u32`,`i1`,`hif8`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`zNUnAlign`|`l1`|`zNUnAlign`|`f32`,`f16`,`bf16`|
|`gm`|`RowMajorMxScaleA`|`l1`|`zZMxScale`|`f8e8m0`|
|`gm`|`ColMajorMxScaleA`|`l1`|`zZMxScale`|`f8e8m0`|
|`gm`|`RowMajorMxScaleB`|`l1`|`nNMxScale`|`f8e8m0`|
|`gm`|`ColMajorMxScaleB`|`l1`|`nNMxScale`|`f8e8m0`|
|`gm`|`VectorLayout`|`l1`|`VectorLayout`|`f32`,`f16`,`bf16`,`i8`,`i1`,`i16`,`i32`,`i64`,`u8`,`u16`,`u32`,`u64`,`f64`,`f8e4m3fn`,`f8e5m2`,`hif8`|
|`l1`|`VectorLayout`|`biasbuf`|`VectorLayout`|`f32`,`i32`|
|`l1`|`zN`|`l0a`|`zN`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`,`hif8`|
|`l1`|`nZ`|`l0a`|`zN`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`,`hif8`|
|`l1`|`VectorLayout`|`l0a`|`VectorLayout`|`f32`,`f16`,`bf16`,`i8`,`f8e4m3fn`,`f8e5m2`,`hif8`|
|`l1`|`zN`|`l0b`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`,`hif8`|
|`l1`|`nZ`|`l0b`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`,`hif8`|

带 scale 的 Cube 拷贝通路（供 `tla.mmad_mx`），即
`tla.copy(dst=l0a|l0b, src=l1, scale=scale)`；scale 仅支持 `f8e8m0`，
且 scale 的 `l1->l0` 不支持转置。

|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`scale_addr`|`scale_layout`|`dtypes`|
|----------|------------|----------|------------|------------|--------------|--------|
|`l1`|`zN`|`l0a`|`zN`|`l1`|`zZMxScale`|`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`l1`|`nZ`|`l0a`|`zN`|`l1`|`zZMxScale`|`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`l1`|`zN`|`l0b`|`nZ`|`l1`|`nNMxScale`|`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`l1`|`nZ`|`l0b`|`nZ`|`l1`|`nNMxScale`|`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|

Cube fixpipe 通路：从 L0C 拷到 GM/L1/UB。
fixpipe 支持随路 quant 与 relu；quant 模式支持 `NO_QUANT`、`PER_TENSOR`、`PER_CHANNEL`。
规则：

- 链路为 (l0c acc) → quant（可选）→ relu（可选）→ 写出到 dst。
- `NO_QUANT` 含纯数据类型转换，如 `f32`→`f16`、`f32`→`bf16`。
- 目的为 UB 时额外有 `split_mode`：nosplit（`NO_SPLIT_VEC_0`、`NO_SPLIT_VEC_1`）与
  split（`SPLIT_M`、`SPLIT_N`）。
- 目的为 `(ub, ColumnMajor)` 时仅支持 `nosplit`。
- UB 的 `split` 模式下禁用任何 quant_mode/relu，也不做纯 cast，仅允许
  `f32`→`f32` 与 `i32`→`i32`。
- 目的为 L1 的量化拷贝仅支持 `zN` 布局。

|`dst`|`dst_layout`|`split_mode`|`quant_mode`|`relu`|`src_dtype`:`dst_dtype`|
|-----|------------|------------|------------|------|-----------------------|
|`gm`|`RowMajor`|`-`|`NO_QUANT`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`gm`|`ColumnMajor`|`-`|`NO_QUANT`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`l1`|`zN`|`-`|`NO_QUANT`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`ub`|`RowMajor`|`nosplit`|`NO_QUANT`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`ub`|`RowMajor`|`split`|`NO_QUANT`|`False`|`f32`: `f32`<br>`i32`: `i32`|
|`ub`|`ColumnMajor`|`nosplit`|`NO_QUANT`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`gm`|`RowMajor`|`-`|`NO_QUANT`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`gm`|`ColumnMajor`|`-`|`NO_QUANT`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`l1`|`zN`|`-`|`NO_QUANT`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`ub`|`RowMajor`|`nosplit`|`NO_QUANT`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`ub`|`ColumnMajor`|`nosplit`|`NO_QUANT`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `i32`|
|`gm`|`RowMajor`|`-`|`PER_TENSOR`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`RowMajor`|`-`|`PER_TENSOR`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_TENSOR`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_TENSOR`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`ub`|`RowMajor`|`nosplit`|`PER_TENSOR`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`RowMajor`|`nosplit`|`PER_TENSOR`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`ColumnMajor`|`nosplit`|`PER_TENSOR`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`ColumnMajor`|`nosplit`|`PER_TENSOR`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`gm`|`RowMajor`|`-`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`RowMajor`|`-`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`ub`|`RowMajor`|`nosplit`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`RowMajor`|`nosplit`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`ColumnMajor`|`nosplit`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`ub`|`ColumnMajor`|`nosplit`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`<br>`i32`: `f16`,`bf16`|
|`l1`|`zN`|`-`|`PER_TENSOR`|`False`|`f32`: `f16`,`bf16`,`i8`<br>`i32`: `f16`,`bf16`,`i8`|
|`l1`|`zN`|`-`|`PER_TENSOR`|`True`|`f32`: `f16`,`bf16`,`i8`<br>`i32`: `f16`,`bf16`,`i8`|
|`l1`|`zN`|`-`|`PER_CHANNEL`|`False`|`f32`: `f16`,`bf16`,`i8`<br>`i32`: `f16`,`bf16`,`i8`|
|`l1`|`zN`|`-`|`PER_CHANNEL`|`True`|`f32`: `f16`,`bf16`,`i8`<br>`i32`: `f16`,`bf16`,`i8`|

Vector 拷贝通路

|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|
|----------|------------|----------|------------|--------|
|`gm`|`RowMajor`|`ub`|`RowMajor`|`i64`,`u64`,`f64`,`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`i1`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`,`f8e8m0`,`f4e2m1`,`f4e1m2`,`hif8`,`f4e2m1x2`,`f4e1m2x2`|
|`gm`|`VectorLayout`|`ub`|`VectorLayout`|`i64`,`u64`,`f64`,`i1`,`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`u32`,`u16`,`u8`|
|`ub`|`RowMajor`|`gm`|`RowMajor`|`i64`,`u64`,`f64`,`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`i1`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`,`f8e8m0`,`f4e2m1`,`f4e1m2`,`hif8`,`f4e2m1x2`,`f4e1m2x2`|
|`ub`|`RowMajor`|`ub`|`RowMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`i1`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`,`f8e8m0`,`f4e2m1`,`f4e1m2`|
|`ub`|`RowMajor`|`ub`|`ColumnMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`i1`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`RowMajor`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`i1`,`i16`,`i32`,`u8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`ub`|`zN`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`i1`,`i16`,`i32`,`u8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`ub`|`zZMxScale`|`l1`|`zZMxScale`|`f8e8m0`|
|`ub`|`nNMxScale`|`l1`|`nNMxScale`|`f8e8m0`|
|`ub`|`zNUnAlign`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`f8e4m3fn`,`f8e5m2`|

调用示例：

```python
# --- 1) 对齐 GM ↔ UB（RowMajor，满足 32 字节对齐） ---
# 前：x_gm[i, j] 为 ND 数据；x_ub 为空。origin_shape==(M, N)。
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
    tla.copy(
        dst=x_ub,
        src=x_gm,
        cache_policy=tla.params.CachePolicy.DISABLE,
    )  # 可选：对 GM 源设置 L2 Cache 提示
    # 后：对 origin_shape 内所有逻辑 (i, j)，x_ub[i, j] == x_gm[i, j]。
    tla.copy(dst=y_gm, src=y_ub)

# --- 2) ND → zN：GM RowMajor → L1 zN（排布变化） ---
l1_a = tla.make_tensor_like(l1_ptr, gm_a, layoutTag=tla.arch.zN)
with tla.cube():
    tla.copy(dst=l1_a, src=gm_a)

# --- 3) M 不是 16 的倍数：zNUnAlign ---
l1_unalign = tla.make_tensor_like(l1_ptr, gm_tile, layoutTag=tla.arch.zNUnAlign)
with tla.cube():
    tla.copy(dst=l1_unalign, src=gm_tile)

# 批量 FP8：两个 tensor 的 layout 携带批次数和间距。
# src_step 和 dst_step 是独立的元素间距。
src3 = tla.make_tensor(src_ptr, tla.make_layout(
    tla.make_shape(batch_count, valid_rows, 128),
    tla.make_stride(src_step, 128, 1)))
dst3 = tla.make_tensor(dst_ptr, tla.make_layout(
    tla.make_shape(batch_count, (16, 8), (32, 4)),
    tla.make_stride(dst_step, (32, 512), (1, 4096)),
    origin_shape=tla.make_shape(batch_count, valid_rows, 128),
    layoutTag=tla.arch.zN))
with tla.cube():
    tla.copy(dst3, src3)

# --- 4) 预打包 GM zN：按块搬运，不改变排布 ---
# Host 缓冲区已按 zN 排布，并补齐到 ceil(M/16) * 16 行。
# 两端使用相同布局树，搬运不再重排数据。
gm_zn = tla.make_tensor(
    gm_ptr, tla.make_layout_with_tag((m_pad16, k), dtype_a, tla.arch.zN)
)
l1_zn = tla.make_tensor_like(l1_ptr, gm_zn, tla.arch.zN)
with tla.cube():
    tla.copy(dst=l1_zn, src=gm_zn)

# --- 5) 预打包 GM nZ：转置排布的对应搬运 ---
# nZ 的两轴分别补齐到 ceil(M/C0) * C0 行、ceil(N/16) * 16 列，
# 搬运按完整 C0 行组进行，两端缓冲区均须覆盖该容量。
gm_nz = tla.make_tensor(
    gm_ptr, tla.make_layout_with_tag((m_pad_c0, n_pad16), dtype_a, tla.arch.nZ)
)
l1_nz = tla.make_tensor_like(l1_ptr, gm_nz, tla.arch.nZ)
with tla.cube():
    tla.copy(dst=l1_nz, src=gm_nz)

# --- 6) 一维连续数据：VectorLayout ---
# 一行 bias 用一维连续元素表示，不使用 (1, N) tile。搬运两端使用
# 相同 tag：先从 GM 搬到 L1，
# 再从 L1 搬到 bias table 缓冲区。
gm_bias = tla.tile_view(gm_bias_full, tla.make_shape(n), tla.make_coord(0))
l1_bias = tla.make_tensor_like(l1_bias_ptr, gm_bias, tla.arch.VectorLayout)
l0_bias = tla.make_tensor_like(l0_bias_ptr, l1_bias, tla.arch.VectorLayout)
with tla.cube():
    tla.copy(dst=l1_bias, src=gm_bias)
    tla.copy(dst=l0_bias, src=l1_bias)
    tla.mmad(l0_c, l0_a, l0_b, init_c=True, unit_flag=0, bias=l0_bias)

# 相关（寄存器路径，不是 tla.copy）：UB ↔ vector 寄存器非对齐访问
#   with tla.vec.func(mode="simd"):
#       x_reg = x_ub.load(tla.params.UnalignLoadParams())
#       y_ub.store(y_reg, tla.params.UnalignStoreParams())
```

---

## `tla.Tensor.load`

**源码：** [`catlass.tla.tensor._Tensor.load`](../../../../catlass/tla/tensor.py)

功能说明：

将本 UB tensor tile 载入 vector 或 mask SSA（`tile.load`）。

函数原型：

```python
tla.Tensor.load(params: LoadParams | None = None) -> MaskSSA | VectorSSA | tuple[VectorSSA, VectorSSA] | tuple[VectorSSA, _Tensor]
```

参数说明：

- `params`（`LoadParams | None`）：载入模式。可选，默认 `None`。
  - `None` / `NormalLoadParams` / `UnalignLoadParams` /
    `BlockLoadParams` → `VectorSSA`（`DIST_DINTLV_B8` / `DIST_DINTLV_B16` / `DIST_DINTLV_B32` 时为两个向量的二元组）。
  - `MaskLoadParams` → `MaskSSA`。
  - `NormalLoadParams`：
    - `load_dist`（`str`）：`tla.params.LoadDist` token。默认 `DIST_NORM`。

      | 取值 | 含义 |
      | --- | --- |
      | `DIST_NORM` | 默认连续载入 |
      | `DIST_BRC_B8` | 单个 b8 元素广播到全部 256 个 lane |
      | `DIST_BRC_B32` | 单个 b32 元素广播到全部 lane |
      | `DIST_DINTLV_B8` | 读取 512 字节，偶数和奇数下标的 b8 元素分别填充两个 256-lane 向量 |
      | `DIST_DINTLV_B16` | 读取 512 字节；返回两个 128-lane b16 寄存器，依次保存偶数、奇数位置的元素 |
      | `DIST_DINTLV_B32` | b32 解交织（f32/i32/u32）；双返回值 |
      | `DIST_US_B8` / `DIST_US_B16` | 2 倍上采样 |
      | `DIST_DS_B8` / `DIST_DS_B16` | 读取 512 字节，每隔一个 b8/b16 元素保留一个，填充一个完整向量 |
      | `DIST_BRC_B16` | 单个 b16 元素广播 |
      | `DIST_UNPACK_B8` | 读取 128 个 b8 元素，每个元素后插入一个零字节，填充一个 256-lane 向量 |
      | `DIST_UNPACK_B16` | b16 零扩展到双倍宽度 |
      | `DIST_UNPACK_B32` | 读取 32 个 b32 元素，每个元素后插入一个全零 b32 元素，填充一个 64-lane 向量 |
      | `DIST_UNPACK4_B8` | 固定读取 64 字节到寄存器字节位置 0、4、…、252，其余寄存器字节置零 |
      | `DIST_E2B_B16` / `DIST_E2B_B32` | 元素到 DataBlock 广播 |
      | `DIST_BLK` | 一个 32 字节 DataBlock 广播到全部 8 个 |

    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：`post_mode=POST_MODE_UPDATE` 时的元素步长；
      `0` 表示按 `tile_extent`（`origin_shape`）推进。默认 `0`。
  - `UnalignLoadParams`：
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：`0`（默认）按 `origin_shape` 推进；
      有效步长须在 `(0, VL]`（元素个数），由调用方保证重叠/脏 lane。默认 `0`。
  - `BlockLoadParams`：
    - `block_stride`（`int`）：块头间隔，单位为 32B DataBlock；`0` 时将首块
      复制到全部 8 槽。默认 `0`。
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `repeat_stride`（`Any`）：编译期地址预偏移（32B 块单位）；
      `POST_MODE_UPDATE` 时为推进步长。默认 `0`。
  - `MaskLoadParams`：
    - `load_dist`（`str`）：`tla.params.MaskLoadDist` token。默认
      `DIST_NORM`。

      | 取值 | 含义 |
      | --- | --- |
      | `DIST_NORM` | bit-packed 掩码载入 |
      | `DIST_US` | bit 上采样 |
      | `DIST_DS` | bit 下采样 |

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用；源 tile 须位于 UB。
- 实际读取范围须位于可用 UB 地址空间内。
- Mask 载入 `DIST_NORM` 要求 UB 元素类型为 b8/b16/b32/b64 标量。
- 带位宽后缀的分布模式要求对应的元素存储位宽。b8 包含 Bool 和以单字节存储的编码。
  `DIST_BLK` 支持 b8/b16/b32。b64 支持 `DIST_NORM` 和 32 lane 的完整
  `UnalignLoadParams` 载入。
- `DIST_DINTLV_B8` 支持 i8/u8/f8E4M3FN/f8E5M2/HiFloat8/E8M0 和打包类型
  Float4E2M1x2/Float4E1M2x2。每个寄存器容纳
  256 个单字节元素，一次载入读取 `2 * 256 = 512` 字节，返回两个保持源元素类型的
  完整向量，依次保存偶数、奇数位置的元素。例如输入 `[a0, a1, a2, a3, ...]`
  对应输出 `[a0, a2, ...]` 和 `[a1, a3, ...]`。实际读取地址须按 32 字节
  对齐，全部 512 字节均须有效，读取范围不取决于 tile shape。对于 FP4x2，
  每个元素是包含两个 FP4 值的一个字节。载入按这些打包字节解交织，保持字节内两个
  半字节的顺序；每个结果包含 256 个打包元素，即 512 个 FP4 值。E8M0 按原始字节搬运，包括 0xFF NaN 编码；
  结果可通过普通对齐 store 写回，不支持算术和寄存器类型转换。不支持逻辑（未打包）
  FP4。Bool 按 b8 move-only 搬运（与 i8 相同）。
- `DIST_DINTLV_B16` 的实际读取地址须按 32 字节对齐。
- `DIST_UNPACK4_B8` 要求 1 字节元素，实际读取地址须按 32 字节对齐。
- `DIST_BRC_B8` / `DIST_DS_B8` / `DIST_UNPACK_B8` 支持
  Bool/i8/u8/f8E4M3FN/f8E5M2/hif8/f8e8m0/fp4x2；`DIST_DS_B16` 支持 f16/bf16/i16/u16；
  `DIST_UNPACK_B32` 支持 f32/i32/u32。这些模式保持源元素类型，返回完整的
  256 字节向量寄存器。
  `DIST_BRC_B8` 要求源地址按 1 字节对齐，其余四种模式要求按 32 字节对齐。
  `DIST_UNPACK_B8` / `DIST_UNPACK_B32` 读取 128 字节，下采样读取 512 字节。
  调用方须保证完整的源读取范围有效，此范围独立于 tile shape。
- `BlockLoadParams` 对齐 AscendC ``DATA_BLOCK_COPY``（b8/b16/b32；
  b8 含 i8/u8，Bool 按 b8）；`block_stride` 与 `repeat_stride` 须在
  [0, 65535] 范围内。
- `DIST_US` / `DIST_DS`：UB 元素类型可为任意 b8/b16/b32/b64 标量。
  元素类型仅用于 UB 寻址；掩码位布局与元素宽度无关。
  静态 tile shape 乘积属于 `{32,64,128,256}` 时作为 `!tla.mask<N>` 的 `N`；
  否则 `N=256`。
- 掩码载入字节偏移对齐（UB 字节地址 `base + offset×sizeof(T)`）：
  `DIST_NORM` → `VL/8`；`DIST_US` → `VL/16`；`DIST_DS` → `min(32, VL/4)`。
  VL=256 时为 32B / 16B / 32B。
- `POST_MODE_UPDATE` 要求 rank-1 连续 tile、`DIST_NORM`，且元素位宽为
  b8/b16/b32/b64（Bool、有符号和无符号整数、FP8、HiFloat8、E8M0、FP4x2、
  f16、bf16、f32 和 f64）。返回 `(vector, next_tile)`。
  在 `tla.vec.func` 内须用 `tla.range` 赋回 `next_tile`（`while` 尚不支持嵌在
  `vec.func` 内）。窗口须已是原点窗（`coord=0`）；用 `tile_view`（或等价方式）
  切出所需 tile，UPDATE 不会自动折非零坐标。`NormalLoadParams`：
  `post_update_stride=0`（默认）按 `tile_extent`（`origin_shape`）推进；显式值
  保留填充与 stride 解耦。`UnalignLoadParams`：默认 `0`→`tile_extent`；有效
  步长须在 `(0, VL]`。干净 `tla.range` 由 `tla-unalign-ureg` 将 Pre/Post 提到
  环外；`for` 内套动态 `if` 仍走每趟双指令。

调用示例：

```python
with tla.vec.func(mode="simd"):
    x_reg = x_ub.load()
    x_unalign = x_ub.load(tla.params.UnalignLoadParams())
    x_blk = x_ub.load(tla.params.NormalLoadParams(
        load_dist=tla.params.LoadDist.DIST_BLK))
    x_gather = x_ub.load(tla.params.BlockLoadParams(block_stride=4))
    preg = mask_ub.load(
        tla.params.MaskLoadParams(load_dist=tla.params.MaskLoadDist.DIST_DS)
    )
    y = tla.where(preg, x_reg, min_reg)
    # fp8_ub：64 个 Float8E4M3FN 元素，实际起始地址按 32 字节对齐。
    fp8_reg = fp8_ub.load(tla.params.NormalLoadParams(
        load_dist=tla.params.LoadDist.DIST_UNPACK4_B8))
    fp32_reg = fp8_reg.to(tla.Float32)
```

---

## `tla.Tensor.store`

**源码：** [`catlass.tla.tensor._Tensor.store`](../../../../catlass/tla/tensor.py)

功能说明：

将 vector 或 mask SSA 写回本 UB tensor tile（`tile.store`）。

函数原型：

```python
tla.Tensor.store(value: VectorSSA | MaskSSA | tuple[VectorSSA, VectorSSA], params: StoreParams | None = None, *, mask: MaskSSA | None = None) -> _Tensor | None
```

参数说明：

- `value`（`VectorSSA | MaskSSA | tuple[VectorSSA, VectorSSA]`）：要写入的
  `VectorSSA` 或 `MaskSSA`。`DIST_INTLV_B8` / `DIST_INTLV_B16` /
  `DIST_INTLV_B32` 时传入双向量二元组 `(even, odd)`（同类型 / 满 VL）。必填。
- `params`（`StoreParams | None`）：写回模式。可选，默认 `None`。
  - `None` / `NormalStoreParams` / `UnalignStoreParams` /
    `BlockStoreParams` → vector 写回。
  - `MaskStoreParams` → mask 写回。
  - `NormalStoreParams`：
    - `store_dist`（`str`）：`tla.params.StoreDist` token。默认 `DIST_NORM`。

      | 取值 | 含义 |
      | --- | --- |
      | `DIST_NORM` | 默认连续写回 |
      | `DIST_PACK_B16` | AscendC 同类型 PK_B16（b16→b16，或先验 cast 后的 b8→b8） |
      | `DIST_PACK_B32` | AscendC 同类型 PK_B32（b32→b32，或先验 cast 后的 b16→b16） |
      | `DIST_PACK_B64` | 写入 i64、u64 或 f64 lane 的低 32 位 |
      | `DIST_PACK4_B32` | 按 mask，将寄存器字节位置 0、4、…、252 写到 UB 字节位置 0、1、…、63 |
      | `DIST_INTLV_B8` | 将两个 b8 寄存器交织写入连续目标元素 |
      | `DIST_INTLV_B16` | 将两个 b16 寄存器交织 |
      | `DIST_INTLV_B32` | 将两个 b32 寄存器交织 |
      | `DIST_FIRST_ELEMENT_B8` | 仅写 lane 0（1 字节） |
      | `DIST_FIRST_ELEMENT_B16` | 仅写 lane 0（2 字节） |
      | `DIST_FIRST_ELEMENT_B32` | 仅写 lane 0（4 字节） |

    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：`post_mode=POST_MODE_UPDATE` 时的元素步长；
      `0` 表示按 `tile_extent`（`origin_shape`）推进。默认 `0`。
  - `UnalignStoreParams`：
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）、`POST_MODE_UPDATE` 或
      `POST_MODE_UPDATE_NBYTES`。
    - `post_update_stride`（`Any`）：使用 `POST_MODE_UPDATE` 时，`0` 表示按
      `origin_shape` 推进，有效步长须在 `(0, VL]`。使用 `POST_MODE_UPDATE_NBYTES`
      时，须为 `0`。默认 `0`。
  - `BlockStoreParams`：
    - `block_stride`（`int`）：块头间隔，单位 32B DataBlock。默认 `0`。
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `repeat_stride`（`Any`）：仅在 `POST_MODE_UPDATE` 时用作推进步长
      （32B 块单位）。默认 `0`。
  - `MaskStoreParams`：
    - `store_dist`（`str`）：`tla.params.MaskStoreDist` token。默认
      `DIST_NORM`。
- `mask`（`MaskSSA | None`）：vector 写回的可选谓词；`UnalignStoreParams`、
  `MaskStoreParams`、first-element 和交织模式均不接受此参数。默认 `None`。`DIST_PACK4_B32` 的 mask
  按源 lane 计数，1/2/4 字节源元素每个目标字节分别对应 4/2/1 个源 lane。
  省略 mask 时按目标有效长度推导；显式 mask 不再受该长度限制。
  被 mask 屏蔽的目标字节保持原值。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用；目标 tile 须位于 UB。

**`POST_MODE_NORMAL` 模式**

- `DIST_PACK4_B32` 支持源和目标均为相同的 4 字节元素类型，也支持 1、2 或 4 字节
  源元素搭配 1 字节目标元素。该模式向按 32 字节对齐的地址最多写入 64 字节。
  处理 FP32→FP8 结果时，写回前使用 `RegSlot.ZERO` 将转换结果放入寄存器字节位置
  0、4、…、252。
- `DIST_PACK_B16`、`DIST_PACK_B32` 和 `DIST_PACK_B64` 要求源和目标元素类型匹配。
  B16 支持 1 字节或 2 字节元素，B32 支持 2 字节或 4 字节元素，B64 支持 8 字节元素。
  每个目标元素可容纳两个有效源 lane。
- `DIST_INTLV_B8` / `DIST_INTLV_B16` / `DIST_INTLV_B32` 要求 `value` 为
  同类型满 VL 双向量二元组，且与目标元素类型匹配（b8：Bool/i8/u8/f8*/hif8/e8m0/fp4x2；
  b16：f16/bf16/i16/u16；b32：f32/i32/u32）。拒绝谓词 `mask`。
- first-element 模式要求被写值与目标 tile 的元素宽度匹配后缀
  （b8 → 1 字节，b16 → 2 字节，b32 → 4 字节）；传入谓词 `mask` 会报错。
- `UnalignStoreParams` 在 NORMAL 模式下按目标有效长度写回，最多写入一个向量的元素。
  在 UPDATE 模式下，有效步长同时决定写回元素数和地址推进量。

```python
with tla.vec.func(mode="simd"):
    y_ub.store(y_reg)
    y_ub.store(y_reg, tla.params.UnalignStoreParams())
    y_ub.store(y_reg, tla.params.NormalStoreParams(
        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B16))
    # fp32_reg：64 个 FP32 lane；fp8_ub：64 个 Float8E4M3FN 元素。
    # fp8_ub 的实际起始地址按 32 字节对齐。
    fp8_reg = fp32_reg.to(tla.Float8E4M3FN, tla.params.CastParams(
        round_mode=tla.params.RoundMode.CAST_RINT,
        sat_mode=tla.params.SatMode.SAT,
        reg_slot=tla.params.RegSlot.ZERO))
    fp8_ub.store(fp8_reg, tla.params.NormalStoreParams(
        store_dist=tla.params.StoreDist.DIST_PACK4_B32))
```

- `POST_MODE_UPDATE` 要求 rank-1 连续 tile、`DIST_NORM`，且源和目标元素类型匹配，
  位宽为 b8/b16/b32/b64。返回下一个窗口。在
  `tla.vec.func` 内须用 `tla.range` 赋回。与 UPDATE `load` 相同：调用前须为
  原点窗（`coord=0`）。`NormalStoreParams`：默认 `post_update_stride=0` 按
  `tile_extent`（`origin_shape`）推进；显式 stride 与填充解耦。
  `UnalignStoreParams`：默认 `0`→`tile_extent`；有效步长须在 `(0, VL]`；干净
  for 上 Pre/Post 在环外。

**`POST_MODE_UPDATE_NBYTES` 模式**

该模式按 `squeeze` 记录的字节数追加数据。
每次追加时，调用者先在同一 SIMD 区域调用 `squeeze(..., record_nbytes=True)`，
由硬件计算并记录字节数。在计数被覆盖前，使用
`UnalignStoreParams(post_mode=POST_MODE_UPDATE_NBYTES)` 调用 `store`。
`store` 从源向量开头取记录的字节数，写入地址为传入窗口的基址加当前写入偏移。
写入后推进偏移，并返回传入的窗口。

**使用限制**

- 目标须为 `coord=0` 的一维连续 UB 窗口，元素类型须与源向量一致。支持
  `Float16`、`Float32`、`Int8`、`UInt8`、`Int16`、`UInt16`、`Int32` 和 `UInt32`。
  `post_update_stride` 须为编译期常量 `0`，并省略 `store` 的 `mask`。
- 同一 SIMD 区域内的追加共享写入偏移和待写尾部。尾部写回使用最后一次以
  `POST_MODE_UPDATE_NBYTES` 模式执行 `store` 时的目标窗口基址。
- 为追加和尾部写回预留足够空间。从首次追加到退出 SIMD 区域并完成同步，其他读写须避开压缩输出范围。
- 从初始化到尾部写回，外部调用须保留写入偏移和待写尾部。`squeeze` 记录的字节数须保留到对应的
  `store` 使用时。

**`POST_MODE_UPDATE_NBYTES` 模式使用示例**

下面的循环把 `src_ub[0:3]` 中的 `[7, 9, 12]` 和 `src_ub[128:130]` 中的 `[20, 24]`
依次追加到 `dst_ub`。示例中 `dst_ub` 的起始地址按 32 字节对齐。

1. **初始化**。进入 SIMD 区域时，编译器安排硬件把写入偏移置为 0，并清空暂存尾部数据的空间。
2. **压缩并追加**。第一轮，`squeeze(..., record_nbytes=True)` 压紧 3 个 Int16 元素，
   硬件记录 6 字节。调用 `store` 后，硬件暂存这 6 字节，并将写入偏移推进到 6 字节。
   第二轮，硬件为保留的 2 个元素记录 4 字节，`store` 把这 4 字节接在前 6 字节后面。
   两轮合计 10 字节，凑不满一个 32 字节块，因此都留在硬件中，成为待写回的尾部。
   接口用法见 [`tla.squeeze` 说明](simd/compress.md#tlasqueeze)。
3. **写回尾部**。循环结束后，编译器安排硬件把暂存的 10 字节写入 `dst_ub[0:5]`。
   退出 SIMD 区域并完成同步后，`dst_ub` 的前 5 个元素为 `[7, 9, 12, 20, 24]`。

调用示例：

```python
# src_ub 和 dst_ub 是两块独立的一维连续 Int16 UB 窗口，coord 均为 0。
# src_ub 有 256 个元素，dst_ub 有 128 个元素。
with tla.vec.func(mode="simd"):
    # 每次追加都使用硬件记录的字节数。
    update = tla.params.UnalignStoreParams(
        post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES,
    )
    block = tla.make_shape(128)
    for i in tla.range(2):
        # 第一轮选取源下标 0～127，第二轮选取 128～255。
        src_tile = tla.tile_view(src_ub, block, tla.make_coord(i))
        # 第一轮保留当前窗口的前 3 个元素，第二轮保留前 2 个元素。
        keep, _ = tla.update_mask(3 - i, dtype=tla.Int16)
        # 压紧保留的元素。Int16 每个元素占 2 字节，硬件依次记录 6 字节和 4 字节。
        packed = tla.squeeze(src_tile.load(), keep, record_nbytes=True)
        # 每轮都传入 dst_ub，由硬件推进写入偏移。
        dst_ub.store(packed, update)
```

---

## `tla.Tensor.fill`

**源码：** [`catlass.tla.tensor._Tensor.fill`](../../../../catlass/tla/tensor.py)

功能说明：

用数值常量填充当前 L1 tile。

函数原型：

```python
tla.Tensor.fill(value: int | float) -> None
```

参数说明：

- `value`（`int | float`）：trace 期标量，按目标类型编码。整数目标要求取值范围内的
  `int`；f16/bf16/f32 目标接受 `int` 或 `float`，有限值转换时溢出会报错。
  FP4/FP8 要求目标格式能精确表示的有限值；超出有限值范围或无法精确表示时编译报错，不进行舍入。
  E8M0 要求能精确表示的正二次幂，范围为 ``[2**-127, 2**127]``；
  零、负数和非二次幂会报错。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用，并嵌套在 `tla.cube()` 内。
- 目的地须为 L1 tensor。操作数类型使用 ``zN`` 或 ``nZ``；
  E8M0 scale 使用 ``zZMxScale`` 或 ``nNMxScale``。
- 填充范围为 ``[coord, coord + origin_shape)``。操作数 C0 轴两端按 32 字节单元
  向上对齐。MX scale 的 M/N 轴按 16 对齐，scale-group 轴按 2 对齐；
  每个 ``16*2`` 分形占用 32 字节。起点处不完整的单元保持不动；
  最后一个单元可能包含逻辑范围外的物理 Padding，分配容量必须覆盖该 Padding。
  空区域不执行写入。
- 支持的元素类型：``i16/u16/f16/bf16/i32/u32/f32``、打包的 fp4/fp8 操作数
  （``f4e2m1``/``f4e1m2``/``f8e4m3fn``/``f8e5m2``）以及 ``f8e8m0``。
  不支持 8 位整数目的地。

调用示例：

```python
# 对 L1 A tile 的 K 轴 Padding 尾段填零：先取 tile，再填充。
pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                   tla.make_shape(m, k_l0 - k_valid))
pad.fill(0)
# 数值 1 的存储编码由目标 dtype 决定。
t_l1a.fill(1.0)
t_l1sa.fill(1.0)  # E8M0 scale 的数值 1，编码为 0x7f。
```

## `tla.Tensor.__getitem__`

**源码：** [`catlass.tla.tensor._Tensor.__getitem__`](../../../../catlass/tla/tensor.py)

功能说明：

按坐标加载单个 tensor 元素为标量（`tensor[i]` / `tensor[i, j]`）。

函数原型：

```python
tla.Tensor.__getitem__(crd: Any) -> Any
```

参数说明：

- `crd`（`int | Int32 | tuple`）：元素坐标；每个逻辑维一个下标。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- tensor 须位于 `gm` 或 `ub`；`ub` tensor 须在 `tla.vector()` 内访问。
- 布局须为 `RowMajor` 或 `ColumnMajor`。
- 逻辑秩须为 1 或 2，且下标个数与秩一致。
- 除 `Bool` 外不支持亚字节元素类型。
- 拒绝 `None` / `_` 坐标；取子 tensor 请用 `tile_view`。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    value = gm_in[tid]
```

---

## `tla.Tensor.__setitem__`

**源码：** [`catlass.tla.tensor._Tensor.__setitem__`](../../../../catlass/tla/tensor.py)

功能说明：

将标量写入单个 tensor 元素（`tensor[i] = value`）。

函数原型：

```python
tla.Tensor.__setitem__(crd: Any, data: Any) -> None
```

参数说明：

- `crd`（`int | Int32 | tuple`）：元素坐标；每个逻辑维一个下标。必填。
- `data`（`Numeric | int | float | mlir.ir.Value`）：写入值；Python 字面量
  会转到 tensor 元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- tensor 须位于 `gm` 或 `ub`；`ub` tensor 须在 `tla.vector()` 内访问。
- 布局须为 `RowMajor` 或 `ColumnMajor`。
- 逻辑秩须为 1 或 2，且下标个数与秩一致。
- 除 `Bool` 外不支持亚字节元素类型。
- `Numeric` 类型须与 tensor 元素类型完全一致，无隐式提升；先用 `.to(...)` 转换。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = gm_a[tid] + gm_b[tid]
```
