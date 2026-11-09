# TileCastInt4ToInt8

> [Code Location](../../../../../../../include/catlass/gemm/tile/cast_int4_to_int8.hpp)

[TOC]

## Function

The `TileCastInt4ToInt8` template converts INT4-quantized data (stored as `int8_t`, with two INT4 values per byte) into INT8 (`int8_t`) and writes the result directly to global memory. It is often used to complete the prologue type conversion from INT4 to INT8 for matrix A (weights) before computation.

**Pipeline**: Global memory (INT4 packed in INT8) → Unified Buffer → Cast (INT4 → half) → Cast (half → INT8) → Unified Buffer → global memory (INT8). The design uses two-stage double buffering (STAGES=2) and event-driven control to enable MTE2/V/MTE3 three-level pipeline concurrency.

The conversion is completed by the vector engine using two `AscendC::Cast` operations: INT4 → half → INT8.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported. `COMPUTE_LEN` must not exceed 24*1024.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag. Only Arch::AtlasA2 is supported.
    class SrcType_,                   // Source type, Gemm::GemmType<int8_t, LayoutSrc>
    class DstType_,                   // Destination type, Gemm::GemmType<int8_t, LayoutDst>
    uint32_t COMPUTE_LEN_,            // Length of each computation
    uint32_t STAGES = 2               // Number of pipeline stages (default: 2)
>
struct TileCastInt4ToInt8 {
    using ElementSrc = typename SrcType_::Element;   // int8_t (INT4 packed)
    using ElementDst = typename DstType_::Element;   // int8_t (INT8)
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LEN_`: the length of a single Cast computation, with an upper limit of 24*1024
- `SrcType_::Element` and `DstType_::Element`: The `sizeof` values must be the same (both are `int8_t` and `sizeof == 1`).
- `LayoutSrc`/`LayoutDst`: ColumnMajor and RowMajor are supported.

## Construction and Destruction

### Constructors

```cpp
TileCastInt4ToInt8(Arch::Resource<ArchTag> const &resource, Params const &params);
```

Allocate double buffering from the Resource Unified Buffer.

- `ubInTensor[2]`*`COMPUTE_LEN / 2` bytes (INT4 compact storage input)
- `ubOutTensor[2]`*`COMPUTE_LEN` bytes (INT8 output)
- `ubWorkspace[2]`*`COMPUTE_LEN * sizeof(half)` bytes (half workspace for intermediate conversion)

Initialize the event flag to ensure pipeline security.

`Params` is an empty structure. No additional parameters are required.

### Destructors

```cpp
~TileCastInt4ToInt8();
```

Wait for all incomplete MTE2/V/MTE3 events to ensure secure pipeline exit.

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,   // Global memory destination (INT8)
    AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc    // Global memory source (INT4 packed)
);
```

49 sub-blocks are processed in parallel. Each sub-block processes `tilesPerAiv` rows, with up to 32 rows (`tilesPerLoop=32`) per round. If fewer than 32 rows remain, the final round adapts accordingly.

## Examples

```cpp
#include "catlass/gemm/tile/cast_int4_to_int8.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;   // INT4 packed in int8
using ElementDst = int8_t;   // INT8
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LEN = 16 * 1024;

const int M = 256;
const int K = 8192; // INT4 packed K, unpacked as 16,384
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K * 2);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastInt4ToInt8<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN>::Params params;

TileCastInt4ToInt8<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN> castOp(resource, params);
castOp(gmDst, layoutDst, gmSrc, layoutSrc);
```
