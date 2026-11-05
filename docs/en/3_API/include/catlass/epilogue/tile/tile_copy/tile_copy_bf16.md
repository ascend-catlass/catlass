# TileCopyBf16

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyBf16` is a template for epilogue BF16 specialization movement and aggregation. It reuses the assembly logic of `TileCopy`, but enforces replacement of the `Element` type of X/Y/D with `bfloat16_t`.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products

## Template Prototype

```cpp
template <
    class ArchTag,
    class CType,
    class XType,        // Layout is extracted, and the Element type is forcibly set to bfloat16_t.
    class YType,        // Layout is extracted, and the Element type is forcibly set to bfloat16_t.
    class DType         // Layout is extracted, and the Element type is forcibly set to bfloat16_t.
>
struct TileCopyBf16;
```

## Member Types

| Member Type| Description|
| :------ | :------ |
| `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>` |
| `CopyGmToUbX` | `CopyGm2Ub<Arch, GemmType<bfloat16_t, XLayout>>` |
| `CopyGmToUbY` | `CopyGm2Ub<Arch, GemmType<bfloat16_t, YLayout>>` |
| `CopyUbToGmD` | `CopyUb2Gm<Arch, GemmType<bfloat16_t, DLayout>>` |
