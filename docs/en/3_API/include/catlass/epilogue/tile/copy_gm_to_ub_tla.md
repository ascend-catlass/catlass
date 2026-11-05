# CopyGm2UbTla

> [Code Location](../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub_tla.hpp)

[TOC]

## Function

`CopyGm2UbTla` implements TLA-style data movement from global memory to Unified Buffer in the epilogue stage. Operands are encapsulated using `tla::Tensor`, and SFINAE is used to automatically select the movement policy based on the source/destination layout.

- Applicability: Atlas A2 series products and Atlas A3 series products (RowMajor) and Ascend 950PR&950DT series products (VectorLayout/RowMajor)
- Style: TLA (`tla::Tensor`)
- Difference from [CopyGm2Ub](./copy_gm_to_ub/README.md): TLA style, with template parameters inferred using `decltype`

## Template Prototype

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyGm2UbTla;
```

## Partial Specialization Implementation

| Architecture| SFINAE Condition| Movement Method|
| :------ | :------ | :------ |
| Atlas A2| `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad` + `DataCopyPadExtParams` |
| Ascend 950| `isVector<Src> && isVector<Dst>` | `DataCopyPad`, single-block movement|
| Ascend 950| `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad` + `DataCopyPadExtParams` |

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination TLA tensor (Unified Buffer, VECCALC)|
| `srcTensor` | Source TLA tensor (global memory)|

## Examples

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"

using namespace Catlass::Epilogue::Tile;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);

AscendC::GlobalTensor<half> srcData;
AscendC::LocalTensor<half> dstData;

auto srcTensor = tla::MakeTensor(srcData, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, dstLayout, Arch::PositionUB{});

CopyGm2UbTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
