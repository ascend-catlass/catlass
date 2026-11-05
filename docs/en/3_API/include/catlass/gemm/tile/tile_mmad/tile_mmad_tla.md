# TileMmadTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## Function

`TileMmadTla` is the TLA-style version of [TileMmad](./tile_mmad.md). It uses `AscendC::Mmad` to perform `C += A * B`. All operands are wrapped via `tla::Tensor`. Dimensions are automatically extracted through `l0ATensor.data()` and `.layout().originShape()`.

The following four calling modes are supported:

1. Standard matrix multiply-accumulate (without bias)
2. Matrix multiply-accumulate with bias
3. L0 batch MMAD (batch-style matrix multiply-accumulate across multiple batches)
4. Automatic dimension extraction (deriving M, N, and K from `tla::Tensor.layout().originShape()`)

### Architecture Differences

| Architecture| `kDirectionAlign` | `disableGemv` | Auto GEMV Bypass|
| :------ | :------ | :------ | :------ |
| Atlas A2 (2201)| Enabled for `float` with L1A `nZ` layout| — | Mode 4: M = 1 → M = 16|
| Ascend 950 (3510)| — | false when L1A is `VectorLayout`| — |

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag
    class ElementA,           // A matrix element type
    class LayoutTagL1A        // L1 layout tag of matrix A (used to determine architecture differences)
>
struct TileMmadTla;
```

## APIs

### Mode 1: standard matrix multiply-accumulate (without bias)

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,    // L0C tensor (zZ)
    TensorA const &l0ATensor,    // L0A tensor (zZ)
    TensorB const &l0BTensor,    // L0B tensor (nZ)
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

### Mode 2: matrix multiply-accumulate with bias

```cpp
template <class TensorC, class TensorA, class TensorB, class TensorBias>
void operator()(
    TensorC const &l0CTensor,
    TensorA const &l0ATensor,
    TensorB const &l0BTensor,
    TensorBias const &l0BiasTensor,    // BT Bias tensor
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

Difference from mode 1: `cmatrixInitVal = false` (bias already preset to L0C), `disableGemv = true` (Ascend 950)

### Mode 3: L0 batch MMAD (batch-style matrix multiply-accumulate across multiple batches)

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,
    TensorA const &l0ATensor,
    TensorB const &l0BTensor,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t l0Batch                // Number of batches.
);
```

The offset for each batch is derived from the product of `tla::get<x,y>(tensor.shape())`. The `cmatrixInitVal` flag is set to true, and accumulation is not performed across batches.

### Mode 4: Automatic dimension extraction

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,    // m, n extracted from originShape
    TensorA const &l0ATensor, // k extracted from originShape[1]
    TensorB const &l0BTensor,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

Dimension inference:

- `m = tla::get<0>(l0CTensor.layout().originShape())`
- `n = tla::get<1>(l0CTensor.layout().originShape())`
- `k = tla::get<1>(l0ATensor.layout().originShape())`

On Atlas A2, m is auto-promoted to 16 when `m=1` to avoid low-efficiency GEMV.

## Examples

### Mode 1: standard matrix multiply-accumulate (without bias)

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using ElementA = half;
using ElementB = half;
using ElementC = float;

auto l0cLayout = tla::MakeLayout<ElementC, layout::zZ>(64, 64);
auto l0aLayout = tla::MakeLayout<ElementA, layout::zZ>(64, 32);
auto l0bLayout = tla::MakeLayout<ElementB, layout::nZ>(32, 64);

auto l0cTensor = tla::MakeTensor(l0cData, l0cLayout, Arch::PositionL0C{});
auto l0aTensor = tla::MakeTensor(l0aData, l0aLayout, Arch::PositionL0A{});
auto l0bTensor = tla::MakeTensor(l0bData, l0bLayout, Arch::PositionL0B{});

Tile::TileMmadTla<Arch::AtlasA2, ElementA, layout::zN> mmadOp;
mmadOp(l0cTensor, l0aTensor, l0bTensor, 64, 64, 32);
```

### Mode 2: Matrix multiply-accumulate with bias

```cpp
using ElementBias = float;
auto btLayout = tla::MakeLayout<ElementBias, layout::VectorLayout>(64);
auto btTensor = tla::MakeTensor(btData, btLayout, Arch::PositionBT{});

mmadOp(l0cTensor, l0aTensor, l0bTensor, btTensor, 64, 64, 32);
```

### Mode 4: Automatic dimension extraction (recommended)

```cpp
// No need to manually pass m/n/k; they are derived from originShape.
mmadOp(l0cTensor, l0aTensor, l0bTensor);

// Subsequent mmad: atomic accumulation
mmadOp(l0cTensor, l0aTensor, l0bTensor, false);
```

### Mode 3: L0 batch MMAD (batch-style matrix multiply-accumulate across multiple batches)

```cpp
mmadOp(l0cTensor, l0aTensor, l0bTensor, 64, 64, 32, 4);  // 4 batches
```
