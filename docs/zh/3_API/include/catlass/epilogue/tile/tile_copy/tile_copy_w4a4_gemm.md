# TileCopyW4A4Gemm

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyW4A4Gemm` 是 epilogue 阶段 W4A4 GEMM 反量化的搬运聚合模板。与 `TileCopyPerTokenDequant` 类似，但不包含 per-channel scale 搬运（W4A4 使用 per-token scale 和 group size）。

- 适用范围：Atlas A2

## 模板原型

```cpp
template <
    class ArchTag,
    class CType,                // int32_t 累加结果
    class PerTokenScaleType,    // per-token scale（ColumnMajor）
    class DType                 // 目标类型
>
struct TileCopyW4A4Gemm;
```

## 成员类型定义

| 成员类型                  | 说明                                 |
| :------------------------ | :----------------------------------- |
| `CopyGmToUbC`             | `CopyGm2Ub<Arch, CType>`             |
| `CopyGmToUbPerTokenScale` | `CopyGm2Ub<Arch, PerTokenScaleType>` |
| `CopyUbToGmD`             | `CopyUb2Gm<Arch, DType>`             |
