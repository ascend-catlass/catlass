# CopyL1ToL0B

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## 功能说明

`CopyL1ToL0B` 实现 Conv 场景下将 Filter（卷积核）数据从 L1 搬运到 L0B（非 TLA 风格）。使用 `LoadData2D` 将 `CI1KHKWCOCI0` 布局转为 `nZ` 分形格式。

- 适用范围：Atlas A2
- 风格：非 TLA

## 模板原型

```cpp
template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B;
```

| 模板参数  | 说明                                    |
| :-------- | :-------------------------------------- |
| `ArchTag` | 架构标签                                |
| `L1Type`  | `Gemm::GemmType<Element, CI1KHKWCOCI0>` |
| `L0Type`  | L0B 类型，默认 `void`                   |

## 偏特化实现

| 偏特化 | L1Type                            | LayoutSrc→LayoutDst | 说明                      |
| :----- | :-------------------------------- | :------------------ | :------------------------ |
| A2     | `GemmType<Element, CI1KHKWCOCI0>` | CI1KHKWCOCI0→nZ     | LoadData 2D，逐 KhKw 搬运 |
