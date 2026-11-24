# CopyL0CToUBTla

> [Code Location](../../../../../../../include/catlass/gemm/tile/copy_l0c_to_ub.hpp)

[TOC]

## Function

`CopyL0CToUBTla` is a template that moves the matrix multiply-accumulate result from L0C (the Accumulator Buffer, also referred to as `CO1`) to the Unified Buffer (UB, also referred to as `VECCALC`). This movement is a prerequisite for Vector engine post-processing such as activation functions and custom operators.

This template differs from [CopyL0CToGm](./copy_l0c_to_gm/README.md) in its destination. CopyL0CToUBTla places the data in UB that the Vector engine can directly access, enabling flexible intermediate processing without incurring additional GM read-back overhead.

> **Restriction**: Only the TLA style is supported. Only the Ascend 950 architecture (`CATLASS_ARCH == 3510`) is supported. Atlas A2 does not have the L0C → UB channel.
>
> **Dependencies**: This module depends on `ScaleGranularity`, `CopyL0CToDstQuantMode`, and `CopyL0CToUBMode` defined in [copy_l0c_to_dst](./copy_l0c_to_dst/README.md).

## Template Prototype

```cpp
template <
    class ArchTag,                                               // Architecture tag: only Arch::Ascend950
    class TensorSrc,                                             // Source TLA tensor (L0C, CO1)
    class TensorDst,                                             // Destination TLA tensor (UB, VECCALC)
    CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,        // Data movement mode
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // Quantization granularity
    bool ReluEnable = false,                                     // Whether to enable ReLU
    class Enable = void                                          // SFINAE dispatch
>
struct CopyL0CToUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to ub.");
};
```

- `CopyMode`: movement mode. The options are `NO_SPLIT`, `SPLIT_M` (M dimension split in half), and `SPLIT_N` (N dimension split in half with 32‑alignment).
- `TensorDst` must be `VECCALC`. The layout supports only `RowMajor`.

## Partial Specialization Implementation

| CopyMode | M Processing| N Processing| dualDstCtl | Movement Instruction|
| :------ | :------ | :------ | :------ | :------ |
| `NO_SPLIT` | RowMajor: original M; ColumnMajor: `RoundUp(M, ELE_NUM_PER_BLK)` (32 for 1-byte)| RowMajor: `RoundUp(N, ELE_NUM_PER_BLK)` (32 for 1-byte); ColumnMajor: original N| — | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` / `CFG_COLUMN_MAJOR_UB` |
| `SPLIT_M` | `RoundUp(M, 2)` | Original N| `1` | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` |
| `SPLIT_N` | Original M| `RoundUp(N, 32)` | `2` | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` |

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor (Unified Buffer, VECCALC, RowMajor)
    TensorSrc const &srcTensor,    // Source tensor (L0C, CO1)
    uint8_t unitFlag = 0           // Unit flag
);
```

Static constraints:

- `TensorDst::Layout` is `RowMajor`.
- `TensorSrc::position == CO1`
- `TensorDst::position == VECCALC`

## Examples

### NO_SPLIT

```cpp
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<float, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstUBTensor, dstLayout, Arch::PositionUB{});

CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### SPLIT_M

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::SPLIT_M> copyOp;
copyOp(dstTensor, srcTensor);
```

### SPLIT_N

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::SPLIT_N> copyOp;
copyOp(dstTensor, srcTensor);
```

### RE_QUANT + ReLU

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, true> copyOp;
copyOp(dstTensor, srcTensor);
```
