# TileCopy

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopy` is a basic template for epilogue data movement and aggregation. It combines and references basic movement templates such as `CopyGm2Ub` and `CopyUb2Gm` for use in block-level epilogue operations. Depending on the number of operands (C/D, C/X/D, C/X/Y/D), it automatically assembles the required GM → UB and UB → GM child components.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- It does not directly execute operators. Instead, it exposes its child components as type members, allowing them to be accessed.

## Template Prototype

```cpp
// Two operands: C + D
template <class ArchTag, class CType, class DType>
struct TileCopy;

// Three operands: C + X + D
template <class ArchTag, class CType, class XType, class DType>
struct TileCopy;

// Four operands: C + X + Y + D
template <class ArchTag, class CType, class XType, class YType, class DType>
struct TileCopy;
```

## Member Types

| Template| Member Type| Description|
| :------ | :------ | :------ |
| `TileCopy<Arch, C, D>` | `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>` |
| | `CopyUbToGmD` | `CopyUb2Gm<Arch, DType>` |
| `TileCopy<Arch, C, X, D>` | `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>` |
| | `CopyGmToUbX` | `CopyGm2Ub<Arch, XType>` |
| | `CopyUbToGmD` | `CopyUb2Gm<Arch, DType>` |
| `TileCopy<Arch, C, X, Y, D>` | `CopyGmToUbC` / `CopyGmToUbX` / `CopyGmToUbY` | `CopyGm2Ub<...>` |
| | `CopyUbToGmD` | `CopyUb2Gm<Arch, DType>` |

## Examples

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DType = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopy<Arch::AtlasA2, CType, DType>;
using CopyC = typename Copy::CopyGmToUbC;
using CopyD = typename Copy::CopyUbToGmD;

// At the block level, CopyC is used to move C to Unified Buffer, while CopyD is used to move the result back to global memory.
```
