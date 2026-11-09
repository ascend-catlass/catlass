# TileCastInt8ToFp16Dequant

> [Code Location](../../../../../../../include/catlass/gemm/tile/cast_int8_to_fp16.hpp)

[TOC]

## Function

The `TileCastInt8ToFp16Dequant` template dequantizes the INT8-quantized data into FP16 (`half`) and writes the result directly to global memory. It is often used in the prologue to dequantize the quantized weights and complete precision conversion before computation.

**Pipeline**: Global memory (int8) → Unified Buffer → Cast (int8 → half) → Adds  (+zeroPoint) → Muls (*scale) → Unified Buffer → global memory (half). The design uses two-stage double buffering (STAGES=2) and event-driven control to enable pipeline concurrency.

The difference from [TileCastFp8ToFp16Dequant](./cast_fp8_to_fp16.md) is that the element type is `int8_t` (instead of FP8), and the dequantization operations are `Cast` + `Adds` + `Muls` (instead of FP8-specific table lookup/bitwise operations).

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported. `COMPUTE_LEN` must not exceed 32*1024.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag. Only Arch::AtlasA2 is supported.
    class SrcType_,                   // Source type, Gemm::GemmType<int8_t, LayoutSrc>
    class DstType_,                   // Destination type, Gemm::GemmType<half, LayoutDst>
    uint32_t COMPUTE_LEN_,            // Length of each computation
    uint32_t STAGES = 2               // Number of pipeline stages (default: 2)
>
struct TileCastInt8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;   // int8_t
    using ElementDst = typename DstType_::Element;   // half
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LEN_`: length of a single Cast computation, with an upper limit of 32*1024
- `LayoutSrc`/`LayoutDst`: Only RowMajor and ColumnMajor are supported. The two values must be the same.

## Construction and Destruction

### Constructors

```cpp
TileCastInt8ToFp16Dequant(Arch::Resource<ArchTag> const &resource, Params const &params_);
```

Allocate double buffering from the Resource Unified Buffer.

- `ubInTensor[2]`*`COMPUTE_LEN` bytes (INT8 input)
- `ubOutTensor[2]`*`COMPUTE_LEN * sizeof(half)` bytes (FP16 output)

Initialize the event flag.

### Params

```cpp
struct Params {
    half deqScalar;      // Dequantization scale
    half deqZeroPoint;   // Dequantization zero point

    Params() = default;
    Params(half deqScalar_, half deqZeroPoint_);
};
```

### Destructors

```cpp
~TileCastInt8ToFp16Dequant();
```

Wait for all incomplete pipeline events.

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,   // Global memory destination (FP16)
    AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc    // Global memory source (INT8)
);
```

49 sub-blocks are processed in parallel. A maximum of `COMPUTE_LEN / tileLenRoundInt8` rows are processed per round, and the number of rows is automatically matched.

## Examples

```cpp
#include "catlass/gemm/tile/cast_int8_to_fp16.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;
using ElementDst = half;
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LEN = 16 * 1024;

const int M = 256;
const int K = 4096;
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastInt8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN>::Params params(0.5, 0.0);

TileCastInt8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN> castOp(resource, params);
castOp(gmDst, layoutDst, gmSrc, layoutSrc);
```
