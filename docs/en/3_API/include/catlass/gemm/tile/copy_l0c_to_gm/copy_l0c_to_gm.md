# CopyL0CToGm

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l0c_to_gm.hpp)

[TOC]

## Function

The `CopyL0CToGm` template is responsible for moving the matrix multiply-accumulate result from the L0C (Accumulator Buffer, referred to as `CO1`) to Global Memory (GM). It supports:

- **Pure type conversion** (Cast): such as float → half/bfloat16_t and int32_t → int32_t
- **Per-tensor quantization/dequantization**: applies a unified scalar scale during the movement.
- **Per-channel quantization/dequantization**: applies a per-channel scale vector during the movement.
- **ReLU activation**: directly applies ReLU non-linearity in FixPipe during the movement.

This template is used as member type `CopyL0CToGm` of [TileCopy](../tile_copy/README.md) and is usually automatically managed by blockMmad. Explicit declaration is required only during the assembly of the custom kernel template.

> **Dependencies**: This module depends on the `ScaleGranularity` enumeration and `CopyL0CToDstQuantMode` (Ascend 950)/`CopyL0CToGmQuantMode` (Atlas A2) quantization mode mapping table defined in [copy_l0c_to_dst](../copy_l0c_to_dst/README.md).

## Template Prototype

```cpp
template <
    class ArchTag,                                               // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class ElementAccumulator,                                    // Accumulator element type (usually float or int32_t)
    class GmType,                                                // GM data description: Gemm::GemmType<ElementDst, LayoutDst>
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // Quantization granularity
    bool ReluEnable = false                                      // Whether to enable ReLU.
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};
```

- `LayoutDst` of `GmType` determines the GM data layout. Different layouts trigger different partial specializations.

## Partial Specialization Implementation

### NO_QUANT (pure type conversion)

| Architecture| Destination Layout| Source Layout| Movement Method| Description|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| RowMajor | zN | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` | Supports casts such as float → half/bfloat16_t.|
| Atlas A2| zN | zN | `AscendC::Fixpipe` + `CFG_NZ` | Preserves zN layout; `channelSplit=true` when float → float|
| Atlas A2| NDC1HWC0 | zN | `AscendC::Fixpipe` + `CFG_NZ` | Conv 5D tensor |
| Ascend 950| RowMajor | zN | `AscendC::DataCopy` + `SetFixpipeNz2ndFlag` | NZ → RowMajor layout conversion|
| Ascend 950| zN | zN | `AscendC::DataCopy` | Preserves zN layout; `channelSplit=true` when float → float|
| Ascend 950| NDC1HWC0 | zN | `AscendC::Fixpipe` + `CFG_NZ` | Conv 5D tensor |

### PER_TENSOR (Per-tensor Quantization/Dequantization)

| Architecture| Destination Layout| Source Layout| Movement Method| Params |
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| RowMajor | zN | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `deqScalar` | `float scale = 1.0` |
| Ascend 950| RowMajor | zN | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `deqScalar` | `float scale = 1.0` |

### PER_CHANNEL (Per-channel Quantization/DequantiZation)

| Architecture| Destination Layout| Source Layout| Movement Method| scale Parameter|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| RowMajor | zN | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `SetFixPipeConfig` | `LocalTensor<uint64_t>` bypass|
| Ascend 950| RowMajor | zN | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` (3-argument)| `LocalTensor<uint64_t>` passed directly|

## APIs

### NO_QUANT / PER_TENSOR

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,   // GM destination tensor
    AscendC::LocalTensor<ElementSrc> const &src,    // L0C source tensor (CO1)
    LayoutDst const &dstLayout,                     // GM data layout
    LayoutSrc const &srcLayout,                     // L0C data layout (fixed to zN)
    uint8_t unitFlag = 0                            // Unit flag
);
```

### PER_CHANNEL (Three-Argument, Scale Vector)

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,        // GM destination tensor
    AscendC::LocalTensor<ElementSrc> const &src,         // L0C source tensor (CO1)
    AscendC::LocalTensor<uint64_t> const &scale,         // per-channel scale tensor
    LayoutDst const &dstLayout,                          // GM data layout
    LayoutSrc const &srcLayout,                          // L0C data layout (fixed to zN)
    uint8_t unitFlag = 0                                 // Unit flag
);
```

## Examples

### NO_QUANT: float → half (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"

using namespace Catlass::Gemm::Tile;

using ElementAccumulator = float;
using ElementDst = half;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;

const int M = 128;
const int N = 256;
auto dstLayout = layout::RowMajor::MakeLayout<ElementDst>(M, N);
auto srcLayout = layout::zN::MakeLayout<ElementAccumulator>(M, N);

AscendC::GlobalTensor<ElementDst> dstGmTensor;
AscendC::LocalTensor<ElementAccumulator> srcL0CTensor;

using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType>;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### PER_TENSOR: int32 → half Dequantization (Atlas A2)

```cpp
using ElementAccumulator = int32_t;
using ElementDst = half;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType, ScaleGranularity::PER_TENSOR>;

auto dstLayout = layout::RowMajor::MakeLayout<ElementDst>(M, N);
auto srcLayout = layout::zN::MakeLayout<ElementAccumulator>(M, N);

CopyOp::Params params(0.5f);
CopyOp copyOp(params);
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### PER_CHANNEL: int32 → int8 (Atlas A2)

```cpp
using ElementAccumulator = int32_t;
using ElementDst = int8_t;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType, ScaleGranularity::PER_CHANNEL>;

AscendC::LocalTensor<uint64_t> scaleTensor;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, scaleTensor, dstLayout, srcLayout);
```

### ReLU Activation Output

```cpp
using CopyOp = CopyL0CToGm<Arch::AtlasA2, float, GmType,
    ScaleGranularity::NO_QUANT, true>;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### Ascend 950 RowMajor

```cpp
using GmType = Gemm::GemmType<half, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::Ascend950, float, GmType>;

auto dstLayout = layout::RowMajor::MakeLayout<half>(M, N);
auto srcLayout = layout::zN::MakeLayout<float>(M, N);

CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### Ascend 950 zN

```cpp
using GmType = Gemm::GemmType<float, layout::zN>;
using CopyOp = CopyL0CToGm<Arch::Ascend950, float, GmType>;

auto dstLayout = layout::zN::MakeLayout<float>(M, N);
auto srcLayout = layout::zN::MakeLayout<float>(M, N);

CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```
