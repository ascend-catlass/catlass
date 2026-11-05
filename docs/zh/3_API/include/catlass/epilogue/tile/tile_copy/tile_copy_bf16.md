# TileCopyBf16

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyBf16` 是 epilogue BF16 特化搬运聚合模板。复用 `TileCopy` 的组装逻辑，但强制将 X/Y/D 的 Element 替换为 `bfloat16_t`。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品

## 模板原型

```cpp
template <
    class ArchTag,
    class CType,
    class XType,        // Layout 被提取，Element 强制为 bfloat16_t
    class YType,        // Layout 被提取，Element 强制为 bfloat16_t
    class DType         // Layout 被提取，Element 强制为 bfloat16_t
>
struct TileCopyBf16;
```

## 成员类型定义

| 成员类型      | 说明                                             |
| :------------ | :----------------------------------------------- |
| `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>`                         |
| `CopyGmToUbX` | `CopyGm2Ub<Arch, GemmType<bfloat16_t, XLayout>>` |
| `CopyGmToUbY` | `CopyGm2Ub<Arch, GemmType<bfloat16_t, YLayout>>` |
| `CopyUbToGmD` | `CopyUb2Gm<Arch, GemmType<bfloat16_t, DLayout>>` |
