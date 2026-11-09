# TileCastFp8ToFp16Dequant

> [Code Location](../../../../../../../include/catlass/gemm/tile/cast_fp8_to_fp16.hpp)

[TOC]

## Function

The `TileCastFp8ToFp16Dequant` template dequantizes the FP8-quantized data (stored as `int8_t`), converts it into FP16 (`half`), and writes the result directly to global memory. It is often used in the prologue of matrix A (weights) to complete data dequantization before computation.

**Pipeline**: Global memory (FP8) → Unified Buffer → Dequant (FP8 → FP16) → Unified Buffer → global memory (FP16). Dual buffering (BUFFER_NUM=2) and four event IDs are used to implement MTE2/V/MTE3 three-level pipeline concurrency.

The difference from `TileCastInt8ToFp16Dequant` lies in the dequantization (FP8 uses table lookup/bitwise operations) and the support for ColumnMajor layout.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag. Only Arch::AtlasA2 is supported.
    class SrcType_,                   // Source type, Gemm::GemmType<ElementSrc, LayoutSrc>
    class DstType_,                   // Destination type, Gemm::GemmType<ElementDst, LayoutDst>
    uint32_t COMPUTE_LENGTH           // Length of each vector engine computation
>
struct TileCastFp8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;   // int8_t (FP8)
    using ElementDst = typename DstType_::Element;   // half (FP16)
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LENGTH`: the length of a single dequantization computation, which affects the size of the Unified Buffer
- `LayoutSrc`/`LayoutDst`: Only RowMajor and ColumnMajor are supported. The two values must be the same.

## Construction and Destruction

### Constructors

```cpp
TileCastFp8ToFp16Dequant(Arch::Resource<ArchTag> &resource, Params const &params_);
```

Allocate double buffering from the Unified Buffer of `Arch::Resource`.

- `inputBuffer[2]`*`COMPUTE_LENGTH`*1 byte (FP8 input)
- `outputBuffer[2]`*`COMPUTE_LENGTH`*2 bytes (FP16 output)
- `workspace[2]`*`COMPUTE_LENGTH`*2 bytes (Workspace computation)

### Params

```cpp
struct Params {
    half scalar;      // Dequantization scale
    half zeroPoint;   // Dequantization zero point

    Params() = default;
    Params(half scalar_, half zeroPoint_);
};
```

### Destructors

None (The Unified Buffer is managed by `Resource`.)

## APIs

### Main API (FP8 → FP16 Dequant)

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> gmDst, LayoutDst const &layoutDst,   // Global memory destination (FP16)
    AscendC::GlobalTensor<ElementSrc> gmSrc, LayoutSrc const &layoutSrc,   // Global memory source (FP8)
    uint32_t &bufferIndex                                                   // Double buffering index (in/out)
);
```

### Epilogue Auxiliary API (FP32 → FP16 Cast)

```cpp
void EpCastFp32ToFp16(
    AscendC::GlobalTensor<half> gmDst, LayoutRowMajor layoutDst,
    AscendC::GlobalTensor<float> gmSrc, LayoutRowMajor layoutSrc
);
```

Cast accumulated results from float to half in the epilogue stage. Only RowMajor is supported.

## Examples

```cpp
#include "catlass/gemm/tile/cast_fp8_to_fp16.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;
using ElementDst = half;
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LENGTH = 16 * 1024;

const int M = 256;
const int K = 4096;
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastFp8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LENGTH>::Params params(0.5, 0.0);

TileCastFp8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LENGTH> castOp(resource, params);

uint32_t bufferIndex = 0;
castOp(gmDst, layoutDst, gmSrc, layoutSrc, bufferIndex);
```
