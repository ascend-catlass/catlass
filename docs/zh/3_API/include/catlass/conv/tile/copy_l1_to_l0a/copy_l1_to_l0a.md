# CopyL1ToL0A

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## 功能说明

`CopyL1ToL0A` 实现 Conv 场景下将 Fmap 数据从 L1 搬运到 L0A（非 TLA 风格），同时完成 im2col 操作。通过 `LoadData3D`/`LoadData3DParamsV2` 将 `NC1HWC0` 布局转为 `zZ` 分形格式。

- 适用范围：Atlas A2
- 风格：非 TLA

## 模板原型

```cpp
template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A;
```

| 模板参数  | 说明                               |
| :-------- | :--------------------------------- |
| `ArchTag` | 架构标签                           |
| `L1Type`  | `Gemm::GemmType<Element, NC1HWC0>` |
| `L0Type`  | L0A 类型，默认 `void`              |

## 偏特化实现

| 偏特化 | L1Type                       | LayoutSrc→LayoutDst | 说明                      |
| :----- | :--------------------------- | :------------------ | :------------------------ |
| A2     | `GemmType<Element, NC1HWC0>` | NC1HWC0→zZ          | LoadData 3D v2，含 im2col |

构造函数接收 `Conv2dFilterParams` 参数。

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // (ho, wo, cin1, Kh, Kw, C0) zZ 格式
    AscendC::LocalTensor<Element> srcTensor,    // (cin1, hi, wi, C0) NC1HWC0 格式
    LayoutDst const &layoutDst,                 // zZ 的 layout
    LayoutSrc const &layoutSrc,                 // NC1HWC0 的 layout
    uint8_t *blockPadList                       // {padLeft, padRight, padTop, padBottom}
)
```
