# TileMmad/TileMmadTla (Tile-Level MMAD Computation)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## Overview

At the tile layer, the MMAD module performs the multiply-accumulate operation `C += A * B` using data from L0A (layout zZ) and L0B (layout nZ), and writes the result to L0C (layout zN). This operation is executed by directly calling the `AscendC::Mmad` hardware instruction.

Two styles are provided: non-TLA (`TileMmad`) and TLA (`TileMmadTla`).

## API List

| API | Style| Bias Supported| L0 Batch | Auto Dimension| Architecture| Description|
| :------ | :------ | :------ | :------ | :------ | :------ | :------ |
| [TileMmad](./tile_mmad.md) | Non-TLA| √| — | — | Atlas A2 + Ascend 950| Directly operates on AscendC::LocalTensor.|
| [TileMmadTla](./tile_mmad_tla.md) | TLA | √| √| √| Atlas A2 + Ascend 950| tla::Tensor wrapper with auto dimension extraction|

## Feature Comparison

| Feature| TileMmad | TileMmadTla |
| :------ | :------ | :------ |
| Operand type| `AscendC::LocalTensor<T>` | `tla::Tensor<LocalTensor<T>, ...>` |
| MMAD without Bias| √| √|
| MMAD with Bias| √| √|
| L0 Batch mmad | — | √|
| Auto dimension extraction| — | √ (mode 4)|
| unitFlag parallel data movement| √| √|
| kDirectionAlign | Atlas A2 float + nZ L1A | Same as the left column|
| GEMV mode control| Ascend 950 `disableGemv`| Ascend 950 `disableGemv`|
| GEMV auto bypass| — | Atlas A2 M = 1 → M = 16 (mode 4)|

## Examples

### Non-TLA

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"

using MmadOp = Tile::TileMmad<Arch::AtlasA2,
    Gemm::GemmType<half, layout::zZ>,
    Gemm::GemmType<half, layout::nZ>,
    void>;

MmadOp mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32);
```

### TLA (recommended)

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

auto l0cTensor = tla::MakeTensor(l0c, l0cLayout, Arch::PositionL0C{});
auto l0aTensor = tla::MakeTensor(l0a, l0aLayout, Arch::PositionL0A{});
auto l0bTensor = tla::MakeTensor(l0b, l0bLayout, Arch::PositionL0B{});

Tile::TileMmadTla<Arch::AtlasA2, half, layout::zN> mmadOp;
mmadOp(l0cTensor, l0aTensor, l0bTensor);  // Automatically extract m/n/k.
```

## Template Selection Guide

| Scenario| Recommendation|
| :------ | :------ |
| Traditional blockMmad assembly| `TileMmad` |
| TLA-style kernel (used with PackedTileCopyTla)| `TileMmadTla` (mode 1 or 4)|
| Bias accumulation required| `TileMmadTla` (mode 2)|
| Flash Attention L0 Batch | `TileMmadTla` (mode 3)|
| Simple code| `TileMmadTla` (mode 4, automatic extraction)|
