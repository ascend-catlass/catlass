# CopyL0CToDst（共享基础设施）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp)

[TOC]

## 概述

`copy_l0c_to_dst` 模块定义了 L0C 数据搬运所需的共享基础设施，包括量化模式映射、Scale 粒度枚举、UB 搬运模式枚举和 TLA 模板声明。它被 [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) 和 [copy_l0c_to_ub](../copy_l0c_to_ub.md) 模块 include 引用，本身不包含算子调用接口。

## API 清单

| API                                                       | 风格 | 适用硬件            | 说明            |
| :-------------------------------------------------------- | :--- | :------------------ | :-------------- |
| [CopyL0CToDstQuantMode](./copy_l0c_to_dst.md)             | —    | Ascend 950           | 量化模式映射    |
| [ScaleGranularity](./copy_l0c_to_dst.md#scalegranularity) | —    | Atlas A2 / Ascend 950 | Scale 粒度枚举  |
| [CopyL0CToUBMode](./copy_l0c_to_dst.md)                   | —    | Ascend 950           | UB 搬运模式枚举 |

## 依赖关系

```cpp
copy_l0c_to_dst  (基础设施)
    ├── copy_l0c_to_gm  (L0C→GM: CopyL0CToGm + CopyL0CToGmTla)
    └── copy_l0c_to_ub  (L0C→UB: CopyL0CToUBTla)
```

## 适用硬件型号说明

| 架构                                | 支持情况                                                                            |
| :---------------------------------- | :---------------------------------------------------------------------------------- |
| Atlas A2（`CATLASS_ARCH == 2201`）   | `ScaleGranularity` + 等效 `CopyL0CToGmQuantMode`（在 `atlasa2/copy_l0c_to_gm.hpp`） |
| Ascend 950（`CATLASS_ARCH == 3510`） | `CopyL0CToDstQuantMode` + `CopyL0CToUBMode` + 模板声明                              |
