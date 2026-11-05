# TileCopyTla（L1 → L0A 偏特化）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp)（Atlas A2）
> [代码位置](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp)（Ascend 950）

[TOC]

## 功能说明

`TileCopyTla` 是 TLA 风格的通用 tile 搬运模板。其在 `copy_l1_to_l0a.hpp` 中定义的偏特化专门负责将 A 矩阵的 tile 块从 L1（A1 Buffer）搬运到 L0A（A2 Buffer）。

与 [非 TLA CopyL1ToL0A](./copy_l1_to_l0a.md) 不同，TLA 版本通过 `tla::Tensor` 封装操作数，由 TLA 运行时自动推导 Layout/Shape/Stride，并通过 SFINAE（`iszN`/`iszZ`/`isnZ` 等 trait）自动匹配正确的偏特化。

## 模板原型

`TileCopyTla` 定义于 [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)：

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

L1 → L0A 的偏特化通过 SFINAE 匹配：源 tensor 的 Position 为 `AscendC::TPosition::A1`，目标 tensor 的 Position 为 `AscendC::TPosition::A2`。

## 偏特化实现

### Atlas A2

| 源 Tensor      | 目标 Tensor     | SFINAE 条件                                          | 说明                                  |
| :------------- | :-------------- | :--------------------------------------------------- | :------------------------------------ |
| zN L1          | zZ L0A          | `iszN<LayoutSrc> && iszZ<LayoutDst>`                 | 基础 Nd 拷贝                          |
| zN L1 (float)  | zZ L0A (float)  | `iszN<float, LayoutSrc> && iszZ<float, LayoutDst>`   | float 专用 LoadData3D                 |
| nZ L1          | zZ L0A          | `isnZ<LayoutSrc> && iszZ<LayoutDst>`                 | 转置拷贝                              |
| nZ L1 (int8_t) | zZ L0A (int8_t) | `isnZ<int8_t, LayoutSrc> && iszZ<int8_t, LayoutDst>` | int8_t 转置（LoadDataWithTranspose）  |
| nZ L1 (float)  | zZ L0A (float)  | `isnZ<float, LayoutSrc> && iszZ<float, LayoutDst>`   | float 转置（LoadData3D + SetFmatrix） |

### Ascend 950

| 源 Tensor         | 目标 Tensor        | SFINAE 条件                                                 | 说明                                            |
| :---------------- | :----------------- | :---------------------------------------------------------- | :---------------------------------------------- |
| zN L1             | zN L0A             | `iszN<LayoutSrc> && iszN<LayoutDst>`                        | 基础 Nd 拷贝。支持 l0Batch 重载和 MX Scale 重载 |
| nZ L1（非 B8/B4） | zN L0A（非 B8/B4） | `!is_one_of_v<Element, int8_t, float8_...> && isnZ && iszN` | 转置拷贝。支持 l0Batch 重载                     |
| nZ L1（B8/B4）    | zN L0A（B8/B4）    | `is_one_of_v<Element, int8_t, float8_...> && isnZ && iszN`  | B8/B4 转置拷贝。支持 l0Batch 和 MX Scale 重载   |
| Vector L1         | L0A                | `isVector<LayoutSrc>`                                       | Vector layout 专用路径                          |

> **注意**：Ascend 950 的 TLA L1→L0A 目标 layout 为 zN（非 zZ），且支持 MX Scale 浮点量化场景和 l0Batch 批量搬运。

## 调用接口

### 基础接口（Atlas A2 / Ascend 950 通用）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`：L1 上的源 tensor（`tla::Tensor<LocalTensor, Layout, Coord, A1>`）
- `dstTensor`：L0A 上的目标 tensor（`tla::Tensor<LocalTensor, Layout, Coord, A2>`）

### l0Batch 重载（Ascend 950 专用）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch);
```

- `l0Batch`：批量搬运的 batch 数，用于多 batch 场景的连续搬运

### MX Scale 重载（Ascend 950 专用，zN→zN / B8/B4 nZ→zN）

```cpp
template <class TensorDst, class TensorSrc, class TensorMxScale>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorMxScale const &scaleTensor);
```

- `srcTensor`：L1 上的源数据 tensor，元素类型 `float8_e4m3_t` / `float8_e5m2_t` / `float4_*`，layout 为 zN（或 nZ 转置）
- `dstTensor`：L0A 上的目标 tensor，元素类型 `AscendC::mx_fp8_e4m3_t` / `AscendC::mx_fp8_e5m2_t` / `float4_*`，layout 为 zN
- `scaleTensor`：L1 上的 MX Scale tensor，元素类型 `float8_e8m0_t`，layout 为 zZ（满足 `isMxScaleForzZ` trait）

> **重要**：MX Scale 搬运是 Ascend 950 专有能力。在实际 kernel 组装中，scale tensor 由 `PackedMxTileCopyTla` 统一管理。GM 端 scale 数据使用 `tla::MakeMxScaleLayout<ElementMxScale, LayoutTag, isMxScaleB>(rows, cols)` 创建 layout，经 TileCopyTla 搬运到 L1 后自动转换为 zZ 排布，再通过本重载实现 L1→L0A 的带 scale 搬运。

## 调用示例

### 基础 zN → zZ 搬运（Atlas A2，TLA）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// 通过 tla::MakeLayout 创建 Layout
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

// 通过 tla::MakeTensor 构造 TLA Tensor
AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// 实例化并调用（SFINAE 根据 src/dst layout trait 自动匹配偏特化）
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### 转置搬运 nZ → zZ（Atlas A2，TLA）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// isnZ<LayoutSrc> && iszZ<LayoutDst> → 自动匹配转置偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### 基础 zN → zN 搬运（Ascend 950，TLA）

```cpp
// Ascend 950 目标 layout 为 zN
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Ascend950: zN L1 → zN L0A
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### l0Batch 批量搬运（Ascend 950，TLA）

```cpp
uint32_t l0Batch = 4;

// Ascend 950 支持 l0Batch 重载：多 batch 连续搬运
copyOp(dstTensor, srcTensor, l0Batch);
```

### MX Scale 搬运（Ascend 950，TLA）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t M = 256;
const uint32_t K = 256;

// MX Scale 的 K 方向维度：每 MX_SCALE_GROUP_NUM（32）个元素共享一个 scale 值
const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// 源数据 layout（L1 zN）
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(M, K);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// 目标数据 layout（L0A zN，元素类型为 mx_fp8）
auto layoutDst = tla::MakeLayout<ElementDst, layout::zN>(M, K);
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// MX Scale layout（L1 zZ，使用 MakeMxScaleLayout 构造）
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::zZ, false>(M, mxScaleK);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale 重载：L1 zN 源数据 + L1 zZ scale → L0A zN mx 数据
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```
