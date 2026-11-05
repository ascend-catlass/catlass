# TileCopyTla (L1 → L0A Partial Specialization)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp) (Atlas A2)
> [Code Location](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp) (Ascend 950)

[TOC]

## Function

`TileCopyTla` is a general-purpose tile-level data movement template in the Tensor Layout Abstraction (TLA) style. The partial specialization defined in `copy_l1_to_l0a.hpp` is specifically responsible for moving the tile blocks of matrix A from L1 (A1 Buffer) to L0A (A2 Buffer).

Unlike [non-TLA-CopyL1ToL0A](./copy_l1_to_l0a.md), this TLA version encapsulates operands via `tla::Tensor`, allowing the TLA runtime to automatically deduce Layout, Shape, and Stride. The appropriate partial specialization is then matched via SFINAE (using traits such as `iszN`, `iszZ`, or `isnZ`).

## Template Prototype

`TileCopyTla` is defined in [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp).

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

The partial specialization of L1 → L0A is matched through SFINAE: The position of the source tensor is `AscendC::TPosition::A1`, and the position of the destination tensor is `AscendC::TPosition::A2`.

## Partial Specialization Implementation

### Atlas A2

| Source Tensor| Destination Tensor| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| zN L1 | zZ L0A | `iszN<LayoutSrc> && iszZ<LayoutDst>` | Basic Nd copy|
| zN L1 (float) | zZ L0A (float) | `iszN<float, LayoutSrc> && iszZ<float, LayoutDst>` | float-specific LoadData3D|
| nZ L1 | zZ L0A | `isnZ<LayoutSrc> && iszZ<LayoutDst>` | Transposed copy|
| nZ L1 (int8_t) | zZ L0A (int8_t) | `isnZ<int8_t, LayoutSrc> && iszZ<int8_t, LayoutDst>` | int8_t transpose (LoadDataWithTranspose)|
| nZ L1 (float) | zZ L0A (float) | `isnZ<float, LayoutSrc> && iszZ<float, LayoutDst>` | float transpose (LoadData3D + SetFmatrix)|

### Ascend 950

| Source Tensor| Destination Tensor| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| zN L1 | zN L0A | `iszN<LayoutSrc> && iszN<LayoutDst>` | Basic Nd copy. l0Batch and MX Scale overloads are supported.|
| nZ L1 (non-B8/B4)| zN L0A (non-B8/B4)| `!is_one_of_v<Element, int8_t, float8_...> && isnZ && iszN` | Transposed copy. l0Batch overload is supported.|
| nZ L1 (B8/B4)| zN L0A (B8/B4)| `is_one_of_v<Element, int8_t, float8_...> && isnZ && iszN` | B8/B4 transposed copy. l0Batch and MX Scale overloads are supported.|
| Vector L1 | L0A | `isVector<LayoutSrc>` | Dedicated path for vector layout|

> **Note**: The destination layout of the Ascend 950 TLA L1→L0A is zN (not zZ), and MX Scale floating-point quantization and l0Batch batch movement are supported.

## APIs

### Basic APIs (Common for Atlas A2 and Ascend 950)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`: source tensor (`tla::Tensor<LocalTensor, Layout, Coord, A1>`) in L1
- `dstTensor`: destination tensor (`tla::Tensor<LocalTensor, Layout, Coord, A2>`) in L0A

### l0Batch Overload (Ascend 950-specific)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch);
```

- `l0Batch`: number of batches for batch data movement, which is used for consecutive movements across multiple batches in multi-batch scenarios.

### MX Scale Overload (Ascend 950-specific, zN→zN/B8/B4 nZ→zN)

```cpp
template <class TensorDst, class TensorSrc, class TensorMxScale>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorMxScale const &scaleTensor);
```

- `srcTensor`: source data tensor in L1. The element type is `float8_e4m3_t`, `float8_e5m2_t`, or `float4_*`, and the layout is zN (or transposed from nZ).
- `dstTensor`: destination tensor in L0A. The element type is `AscendC::mx_fp8_e4m3_t`, `AscendC::mx_fp8_e5m2_t`, or `float4_*`, and the layout is zN.
- `scaleTensor`: MX scale tensor in L1. The element type is `float8_e8m0_t`, and the layout is zZ (satisfying the `isMxScaleForzZ` trait).

> **Note**: MX scale movement is a dedicated capability of Ascend 950. In actual kernel assembly, the scale tensor is managed by the `PackedMxTileCopyTla` in a unified manner. The scale data in Global Memory uses `tla::MakeMxScaleLayout<ElementMxScale, LayoutTag, isMxScaleB>(rows, cols)` to create a layout, and then this layout is automatically converted into a zZ layout after being transferred to L1 through TileCopyTla. Then, this overload is used to implement scale movement from L1 to L0A.

## Examples

### Basic zN → zZ Movement (Atlas A2, TLA)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// Create a layout using tla::MakeLayout.
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

// Construct a TLA tensor using tla::MakeTensor.
AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Instantiation and call (SFINAE automatically matches the partial specialization based on the src/dst layout trait).
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### nZ → zZ Transposed Movement (Atlas A2, TLA)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// isnZ<LayoutSrc> && iszZ<LayoutDst> → Automatic matching for transposed partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### Basic zN → zN Movement (Ascend 950, TLA)

```cpp
// The destination layout of Ascend 950 is zN.
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Ascend 950: zN L1 → zN L0A
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### l0Batch batch movement (Ascend 950, TLA)

```cpp
uint32_t l0Batch = 4;

// Ascend 950 supports l0Batch overload for continuous multi-batch data movement.
copyOp(dstTensor, srcTensor, l0Batch);
```

### MX Scale Movement (Ascend 950, TLA)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t M = 256;
const uint32_t K = 256;

// MX Scale K-dimension: One scale value is shared by every MX_SCALE_GROUP_NUM (32) elements along the K dimension.
const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// Source data layout (L1 zN)
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(M, K);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// Destination data layout (L0A zN, element type: mx_fp8)
auto layoutDst = tla::MakeLayout<ElementDst, layout::zN>(M, K);
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// MX Scale layout (L1 zZ, constructed using MakeMxScaleLayout)
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::zZ, false>(M, mxScaleK);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale overload: L1 zN source data + L1 zZ scale → L0A zN mx data
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```
