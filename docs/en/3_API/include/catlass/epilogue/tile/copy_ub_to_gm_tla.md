# CopyUb2GmTla

> [Code Location](../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm_tla.hpp)

[TOC]

## Function

`CopyUb2GmTla` implements TLA-style data movement from Unified Buffer to global memory in the epilogue stage. Operands are encapsulated using `tla::Tensor`, and SFINAE is used to automatically select the movement policy based on the source/destination layout.

- Applicability: Atlas A2 series products and Atlas A3 series products (RowMajor) and Ascend 950PR&950DT series products (RowMajor)
- Style: TLA (`tla::Tensor`)
- Difference from [CopyUb2Gm](./copy_ub_to_gm/README.md): TLA style, with template parameters inferred using `decltype`

## Template Prototype

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyUb2GmTla;
```

## Partial Specialization Implementation

| Architecture| SFINAE Condition| Movement Method|
| :------ | :------ | :------ |
| Atlas A2| `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad`, stride aligned to C0|
| Ascend 950| `isRowMajor<Src> && isRowMajor<Dst>` | <idp:inline displayname="code" id="code10603231903">DataCopyPad</idp:inline>, stride aligned to C0|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination TLA tensor (global memory)|
| `srcTensor` | Source TLA tensor (Unified Buffer, VECCALC)|

## Examples

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"

using namespace Catlass::Epilogue::Tile;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);

AscendC::LocalTensor<half> srcData;
AscendC::GlobalTensor<half> dstData;

auto srcTensor = tla::MakeTensor(srcData, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstData, dstLayout, Arch::PositionGM{});

CopyUb2GmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
