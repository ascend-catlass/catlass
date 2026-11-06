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
tla.copy(dst: Tensor, src: Tensor, params: CopyParams | None = None, *, scale: Tensor | None = None) -> None
```

参数说明：

- `dst`（`Tensor`）：目的 tile。必填。
- `src`（`Tensor`）：源 tile。必填。
- `params`（`CopyParams | None`）：通路参数：`CopyL0C2DstParams`、
  `CopyUbToGmParams`。默认 `None`。
- `scale`（`Tensor | None`）：L1 上的 OCP `e8m0` 共享指数 tile，用于
  L1→L0A/L0B 的 MX load。以 `i8`/`u8` 存储；目的为 L0A 时 tag
  `tla.arch.zZMxScale`，为 L0B 时 tag `tla.arch.nNMxScale`。默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。
- 拷贝大小按各 tile 的 `origin_shape`。
- `scale` 仅对 L1→L0A/L0B 有效；由此加载的 tile 随后须由 `tla.mmad_mx` 消费。
- MX scale 目的（`zZMxScale` / `nNMxScale`）要求源 tile 覆盖 L1 整行，
  并按行堆叠 scale block。仅在行 pitch 为编译期常量时可检查。
- UB→UB 按整 32 字节块计数，行向上取整。起始未 32B 对齐、或多行 pitch
  容不下取整行时拒绝。目的 tag 为 `ColumnMajor` 时为转置：tile 形状须为
  整块 16×C0，1 字节类型为 32×32。
- `GM`→`L1` / `L1`→`biasbuf` 连续平铺用 rank-1 `VectorLayout`；多行 bias
  tile 不可表示，请切出单行。
- 整 tile DMA 使用 `tla.copy`。寄存器级 UB 非对齐访问请改用
  `tensor.load` / `tensor.store`，并传入 `UnalignLoadParams` /
  `UnalignStoreParams`。
- **尺寸未按 32 字节对齐时：** C0 为 32 字节。ND GM↔UB 时，优先让主导维的
  字节数是 32 的倍数。否则保持 `origin_shape` 为真实尺寸并放大物理
  layout，或改用非对齐 load/store。zN 在 `M` 不是 16 的整数倍时，用
  `tla.arch.zNUnAlign`。

无 scale 的 Cube 拷贝通路，`tla.copy(dst, src, scale=None)`

|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|
|----------|------------|----------|------------|--------|
|`gm`|`RowMajor`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`ColumnMajor`|`l1`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`u16`,`u32`,`f8e4m3fn`,`f8e5m2`,`f4e2m1`,`f4e1m2`|
|`gm`|`zNUnAlign`|`l1`|`zNUnAlign`|`f32`,`f16`,`bf16`|
|`gm`|`RowMajorMxScaleA`|`l1`|`zZMxScale`|`f8e8m0`|
|`gm`|`ColMajorMxScaleA`|`l1`|`zZMxScale`|`f8e8m0`|
|`gm`|`RowMajorMxScaleB`|`l1`|`nNMxScale`|`f8e8m0`|
|`gm`|`ColMajorMxScaleB`|`l1`|`nNMxScale`|`f8e8m0`|
|`gm`|`VectorLayout`|`l1`|`VectorLayout`|`f32`,`i32`,`u64`|
|`l1`|`VectorLayout`|`biasbuf`|`VectorLayout`|`f32`,`i32`|
|`l1`|`zN`|`l0a`|`zN`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`|
|`l1`|`nZ`|`l0a`|`zN`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`|
|`l1`|`zN`|`l0b`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`|
|`l1`|`nZ`|`l0b`|`nZ`|`f32`,`f16`,`bf16`,`i8`,`u32`,`f8e4m3fn`,`f8e5m2`|

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
- PER_CHANNEL 的反量化参数（`quant_tensor`）为 rank-1 `VectorLayout`，位于 L1，
  `u64` 元素、stride 1。每通道一个元素，通道内全部元素共享该反量化参数；长度
  须与目标 tile 的通道数精确一致（`tile_view` 裁剪后的 `originShape[1]`），两侧
  长度均为编译期常量时不相等即编译报错。元素类型、地址空间与布局由前端校验。
- 目的为 UB 时额外有 `split_mode`：nosplit（`NO_SPLIT_VEC_0`、`NO_SPLIT_VEC_1`）与
  split（`SPLIT_M`、`SPLIT_N`）。
- 目的为 `(ub, ColumnMajor)` 时仅支持 `nosplit`。
- UB 的 `split` 模式下禁用任何 quant_mode/relu，也不做纯 cast，仅允许
  `f32`→`f32` 与 `i32`→`i32`。

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
|`gm`|`RowMajor`|`-`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`RowMajor`|`-`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_CHANNEL`|`False`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|
|`gm`|`ColumnMajor`|`-`|`PER_CHANNEL`|`True`|`f32`: `f32`,`f16`,`bf16`,`i8`,`u8`<br>`i32`: `f16`,`bf16`,`i8`,`u8`|

Vector 拷贝通路

|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|
|----------|------------|----------|------------|--------|
|`gm`|`RowMajor`|`ub`|`RowMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`RowMajor`|`gm`|`RowMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`u32`,`u16`,`u8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`RowMajor`|`ub`|`RowMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`RowMajor`|`ub`|`ColumnMajor`|`f32`,`f16`,`bf16`,`i32`,`i16`,`i8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`RowMajor`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`zN`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`f8e4m3fn`,`f8e5m2`|
|`ub`|`zNUnAlign`|`l1`|`zN`|`f32`,`f16`,`bf16`,`i8`,`f8e4m3fn`,`f8e5m2`|

调用示例：

```python
# --- 1) 对齐 GM ↔ UB（RowMajor，满足 32 字节对齐） ---
# 前：x_gm[i, j] 为 ND 数据；x_ub 为空。origin_shape==(M, N)。
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
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
    `BlockLoadParams` → `VectorSSA`（`DIST_DINTLV_B32` 时可为二元组）。
  - `MaskLoadParams` → `MaskSSA`。
  - `NormalLoadParams`：
    - `load_dist`（`str`）：`tla.params.LoadDist` token。默认 `DIST_NORM`。

      | 取值 | 含义 |
      | --- | --- |
      | `DIST_NORM` | 默认连续载入 |
      | `DIST_BRC_B32` | 单个 b32 元素广播到全部 lane |
      | `DIST_DINTLV_B32` | f32 解交织；双返回值 |
      | `DIST_US_B8` / `DIST_US_B16` | 2 倍上采样 |
      | `DIST_BRC_B16` | 单个 b16 元素广播 |
      | `DIST_UNPACK_B16` | b16 零扩展到双倍宽度 |
      | `DIST_UNPACK4_B8` | 固定读取 64 字节到寄存器字节位置 0、4、…、252，其余寄存器字节置零 |
      | `DIST_E2B_B16` / `DIST_E2B_B32` | 元素到 DataBlock 广播 |
      | `DIST_BLK` | 一个 32 字节 DataBlock 广播到全部 8 个 |

    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：`post_mode=POST_MODE_UPDATE` 时的元素步长；
      `0` 表示按 `tile_extent`（`origin_shape`）推进。默认 `0`。
  - `UnalignLoadParams`：
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：须保持为 `0`（仅按 `origin_shape`
      推进窗口）。默认 `0`。
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
- Mask 载入 `DIST_NORM` 要求 UB 元素类型为 1/2/4 字节标量。
- 分发模式限定元素宽度：`DIST_US_B8` 要求 i8/u8；`DIST_BRC_B16` /
  `DIST_US_B16` / `DIST_UNPACK_B16` / `DIST_E2B_B16` 要求 2 字节
  （f16/bf16/i16/u16）；`DIST_E2B_B32` 要求 4 字节（f32/i32/u32）；
  `DIST_DINTLV_B32` 要求 f32；`DIST_BLK` 适用于任意元素宽度。
- `DIST_UNPACK4_B8` 要求 1 字节元素，实际读取地址须按 32 字节对齐。
- `BlockLoadParams` 要求 2/4 字节元素类型（f32/f16/bf16/i32/u32/i16/u16）；
  `block_stride` 与 `repeat_stride` 须在 [0, 65535] 范围内。
- `DIST_US` / `DIST_DS`：UB 元素类型可为任意 b8/b16/b32/b64 标量。
  元素类型仅用于 UB 寻址；掩码位布局与元素宽度无关。
  静态 tile shape 乘积属于 `{32,64,128,256}` 时作为 `!tla.mask<N>` 的 `N`；
  否则 `N=256`。
- 掩码载入字节偏移对齐（UB 字节地址 `base + offset×sizeof(T)`）：
  `DIST_NORM` → `VL/8`；`DIST_US` → `VL/16`；`DIST_DS` → `min(32, VL/4)`。
  VL=256 时为 32B / 16B / 32B。
- `POST_MODE_UPDATE` 要求 rank-1 连续 tile、`DIST_NORM`，且元素类型为
  `f16` / `bf16` / `f32`。返回 `(vector, next_tile)`。在 `tla.vec.func` 内须用
  `tla.range` 赋回 `next_tile`（`while` 尚不支持嵌在 `vec.func` 内）。窗口须
  已是原点窗（`coord=0`）；用 `tile_view`（或等价方式）切出所需 tile，UPDATE
  不会自动折非零坐标。`NormalLoadParams`：`post_update_stride=0`（默认）按
  `tile_extent`（`origin_shape`）推进；显式值保留填充与 stride 解耦。
  `UnalignLoadParams` 须保持 `post_update_stride=0`（仅密集 `origin_shape`）。

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
tla.Tensor.store(value: VectorSSA | MaskSSA, params: StoreParams | None = None, *, mask: MaskSSA | None = None) -> _Tensor | None
```

参数说明：

- `value`（`VectorSSA | MaskSSA`）：要写入的 `VectorSSA` 或 `MaskSSA`。必填。
- `params`（`StoreParams | None`）：写回模式。可选，默认 `None`。
  - `None` / `NormalStoreParams` / `UnalignStoreParams` /
    `BlockStoreParams` → vector 写回。
  - `MaskStoreParams` → mask 写回。
  - `NormalStoreParams`：
    - `store_dist`（`str`）：`tla.params.StoreDist` token。默认 `DIST_NORM`。

      | 取值 | 含义 |
      | --- | --- |
      | `DIST_NORM` | 默认连续写回 |
      | `DIST_PACK4_B32` | 按 mask，将寄存器字节位置 0、4、…、252 写到 UB 字节位置 0、1、…、63 |
      | `DIST_FIRST_ELEMENT_B8` | 仅写 lane 0（1 字节） |
      | `DIST_FIRST_ELEMENT_B16` | 仅写 lane 0（2 字节） |
      | `DIST_FIRST_ELEMENT_B32` | 仅写 lane 0（4 字节） |

    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：`post_mode=POST_MODE_UPDATE` 时的元素步长；
      `0` 表示按 `tile_extent`（`origin_shape`）推进。默认 `0`。
  - `UnalignStoreParams`：
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `post_update_stride`（`Any`）：须保持为 `0`。默认 `0`。
  - `BlockStoreParams`：
    - `block_stride`（`int`）：块头间隔，单位 32B DataBlock。默认 `0`。
    - `post_mode`（`str`）：`POST_MODE_NORMAL`（默认）或 `POST_MODE_UPDATE`。
    - `repeat_stride`（`Any`）：仅在 `POST_MODE_UPDATE` 时用作推进步长
      （32B 块单位）。默认 `0`。
  - `MaskStoreParams`：
    - `store_dist`（`str`）：`tla.params.MaskStoreDist` token。默认
      `DIST_NORM`。
- `mask`（`MaskSSA | None`）：vector 写回的可选谓词；与 `MaskStoreParams` 及
  first-element 模式均不可同时使用。默认 `None`。`DIST_PACK4_B32` 的 mask
  按源 lane 计数，1/2/4 字节源元素每个目标字节分别对应 4/2/1 个源 lane。
  省略 mask 时按目标有效长度推导；显式 mask 不再受该长度限制。
  被 mask 屏蔽的目标字节保持原值。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用；目标 tile 须位于 UB。
- `DIST_PACK4_B32` 要求 1/2/4 字节源元素、1 字节目标元素，以及
  按 32 字节对齐的实际目标地址。按 mask 最多写入 64 字节，不做数值转换或
  FP4 半字节合并。写回 FP32→FP8 结果时，使用 `RegSlot.ZERO`，使值位于
  寄存器字节位置 0、4、…、252。
- first-element 模式要求被写值与目标 tile 的元素宽度匹配后缀
  （b8 → 1 字节，b16 → 2 字节，b32 → 4 字节）；传入谓词 `mask` 会报错。
- `POST_MODE_UPDATE` 要求 rank-1 连续 tile、`DIST_NORM`，且元素类型为
  `f16` / `bf16` / `f32`；返回下一扇窗口。在 `tla.vec.func` 内须用
  `tla.range` 赋回。与 UPDATE `load` 相同：调用前须为原点窗（`coord=0`）。
  `NormalStoreParams`：默认 `post_update_stride=0` 按 `tile_extent`
  （`origin_shape`）推进；显式 stride 与填充解耦。`UnalignStoreParams` 须
  保持 stride 为 `0`（仅密集 `origin_shape`）。

调用示例：

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

---

## `tla.Tensor.fill`

**源码：** [`catlass.tla.tensor._Tensor.fill`](../../../../catlass/tla/tensor.py)
功能说明：

用常量填充本 L1 zN/nZ tile（当前仅支持写零）。

函数原型：

```python
tla.Tensor.fill(value: int | float) -> None
```

参数说明：

- `value`（`int | float`）：trace 期标量；当前仅支持 ``0``
  （设备写入全零比特模式）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用，并嵌套在 `tla.cube()` 内。
- 目的地须为带 ``zN`` 或 ``nZ`` 标签的 L1 tensor。
- 填充范围为 ``[coord, coord + origin_shape)``；C0 轴起点向上对齐到 32 字节
  边界，前导的不完整 C0 单元不写。
- 支持的元素类型：打包的 fp4/fp8 操作数格式
  （``f4e2m1``/``f4e1m2``/``f8e4m3fn``/``f8e5m2``），且仅支持填零。
  更宽类型、8 位整数与 e8m0 scale tile 不支持。

调用示例：

```python
# 将 L1 A tile 的 K-pad 尾部清零：先切 tile，再 fill。
pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                   tla.make_shape(m, k_l0 - k_valid))
pad.fill(0)
```

---

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
