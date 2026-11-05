# TileCopyTla（L1 → L0B 偏特化）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp)（Atlas A2）
> [代码位置](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp)（Ascend 950）

[TOC]

## 功能说明

`TileCopyTla` 是 TLA 风格的通用 tile 搬运模板。其在 `copy_l1_to_l0b.hpp` 中定义的偏特化专门负责将 B 矩阵的 tile 块从 L1（B1 Buffer）搬运到 L0B（B2 Buffer）。

与 [非 TLA CopyL1ToL0B](./copy_l1_to_l0b.md) 不同，TLA 版本通过 `tla::Tensor` 封装操作数，由 TLA 运行时自动推导 Layout/Shape/Stride，并通过 SFINAE（`iszN`/`isnZ` 等 trait）自动匹配正确的偏特化。

## 模板原型

`TileCopyTla` 定义于 [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)：

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

L1 → L0B 的偏特化通过 SFINAE 匹配：源 tensor 的 Position 为 `AscendC::TPosition::A1`，目标 tensor 的 Position 为 `AscendC::TPosition::B2`。

## 偏特化实现

### Atlas A2

| 源 Tensor      | 目标 Tensor     | SFINAE 条件                                          | 说明                                  |
| :------------- | :-------------- | :--------------------------------------------------- | :------------------------------------ |
| zN L1          | nZ L0B          | `iszN<LayoutSrc> && isnZ<LayoutDst>`                 | 基础转置拷贝（Transpose B）           |
| zN L1 (int8_t) | nZ L0B (int8_t) | `iszN<int8_t, LayoutSrc> && isnZ<int8_t, LayoutDst>` | int8_t 转置（LoadDataWithTranspose）  |
| zN L1 (float)  | nZ L0B (float)  | `iszN<float, LayoutSrc> && isnZ<float, LayoutDst>`   | float 转置（LoadData3D + SetFmatrix） |
| nZ L1          | nZ L0B          | `isnZ<LayoutSrc> && isnZ<LayoutDst>`                 | 非转置拷贝（直传）                    |

### Ascend 950

| 源 Tensor         | 目标 Tensor        | SFINAE 条件                                                 | 说明                                                     |
| :---------------- | :----------------- | :---------------------------------------------------------- | :------------------------------------------------------- |
| zN L1（非 B8/B4） | nZ L0B（非 B8/B4） | `!is_one_of_v<Element, int8_t, float8_...> && iszN && isnZ` | 转置拷贝。支持 l0Batch 重载                              |
| zN L1（B8/B4）    | nZ L0B（B8/B4）    | `is_one_of_v<Element, int8_t, float8_...> && iszN && isnZ`  | B8/B4 转置拷贝。支持 l0Batch 和 MX Scale 重载            |
| nZ L1             | nZ L0B             | `isnZ<LayoutSrc> && isnZ<LayoutDst>`                        | 非转置拷贝（Transpose B）。支持 l0Batch 和 MX Scale 重载 |

> **注意**：Ascend 950 的 TLA L1→L0B 目标 layout 为 nZ，且支持 MX Scale 浮点量化场景和 l0Batch 批量搬运。

## 调用接口

### 基础接口（Atlas A2 / Ascend 950 通用）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`：L1 上的源 tensor（`tla::Tensor<LocalTensor, Layout, Coord, A1>`）
- `dstTensor`：L0B 上的目标 tensor（`tla::Tensor<LocalTensor, Layout, Coord, B2>`）

### l0Batch 重载（Ascend 950 专用）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch);
```

- `l0Batch`：批量搬运的 batch 数，用于多 batch 场景的连续搬运

### MX Scale 重载（Ascend 950 专用，B8/B4 zN→nZ / nZ→nZ）

```cpp
template <class TensorDst, class TensorSrc, class TensorMxScale>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorMxScale const &scaleTensor);
```

- `srcTensor`：L1 上的源数据 tensor，元素类型 `float8_e4m3_t` / `float8_e5m2_t` / `float4_*`，layout 为 zN（或 nZ 直传）
- `dstTensor`：L0B 上的目标 tensor，元素类型 `AscendC::mx_fp8_e4m3_t` / `AscendC::mx_fp8_e5m2_t` / `float4_*`，layout 为 nZ
- `scaleTensor`：L1 上的 MX Scale tensor，元素类型 `float8_e8m0_t`，layout 为 nN（满足 `isMxScaleFornN` trait）

> **重要**：B 侧 MX Scale layout 使用 nN 排布（对应 `isMxScaleFornN` trait）。GM 端 scale 数据使用 `tla::MakeMxScaleLayout<ElementMxScale, LayoutTag, true>(rows, cols)`（`isMxScaleB = true`）创建 layout，经 TileCopyTla 搬运到 L1 后自动转换为 nN 排布。`PackedMxTileCopyTla` 已内置 `LayoutTagL1MxScaleB = layout::nN` 管理 B 侧 scale tensor。

## 调用示例

### zN → nZ 转置搬运（Atlas A2，TLA）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// iszN<LayoutSrc> && isnZ<LayoutDst> → 自动匹配转置偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### nZ → nZ 直传搬运（Atlas A2，TLA，Transpose B）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// isnZ<LayoutSrc> && isnZ<LayoutDst> → 自动匹配直传偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### zN → nZ 转置搬运（Ascend 950，TLA）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend950: zN L1 → nZ L0B
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### nZ → nZ 直传搬运（Ascend 950，TLA，Transpose B）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend950: nZ L1 → nZ L0B（非转置直传）
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### l0Batch 批量搬运（Ascend 950，TLA）

```cpp
uint32_t l0Batch = 4;

// Ascend 950 支持 l0Batch 重载：多 batch 连续搬运
copyOp(dstTensor, srcTensor, l0Batch);
```

### MX Scale 搬运（Ascend 950，TLA，B 侧）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t K = 256;
const uint32_t N = 256;

const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// 源数据 layout（L1 zN）
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(K, N);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// 目标数据 layout（L0B nZ，元素类型为 mx_fp8）
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout（L1 nN，B 侧 isMxScaleB=true，使用 MakeMxScaleLayout 构造）
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale 重载：L1 zN 源数据 + L1 nN scale → L0B nZ mx 数据
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

### MX Scale 搬运（Ascend 950，TLA，B 侧，nZ→nZ 直传）

当 B 侧数据 layout 为 nZ（Transpose B 场景）时，MX Scale 同样支持：

```cpp
// 源/目标数据 layout 均为 nZ
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout（L1 nN，B 侧 isMxScaleB=true）
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```
