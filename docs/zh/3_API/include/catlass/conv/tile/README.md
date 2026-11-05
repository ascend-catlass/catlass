# Conv/Tile 类模板概述

> [代码位置](../../../../../../../include/catlass/conv/tile/)

[TOC]

Conv 的 tile 层 API 作为 Conv block 层的模板参数，负责卷积场景下的数据搬运和 im2col 操作。包含 GM→L1 搬运、L1→L0A（含 im2col）、L1→L0B、L0C→GM（含类型转换和 ReLU）以及聚合模板等组件。

## API 清单

| 组件                                         | 风格         | 适用硬件           | 说明                                         |
| :------------------------------------------- | :----------- | :----------------- | :------------------------------------------- |
| [copy_gm_to_l1](./copy_gm_to_l1/README.md)   | 非 TLA + TLA | Atlas A2, Ascend 950 | GM→L1 搬运（NC1HWC0 / CI1KHKWCOCI0）         |
| [copy_l1_to_l0a](./copy_l1_to_l0a/README.md) | 非 TLA + TLA | Atlas A2, Ascend 950 | L1→L0A 搬运（NC1HWC0→zZ，含 im2col）         |
| [copy_l1_to_l0b](./copy_l1_to_l0b/README.md) | 非 TLA + TLA | Atlas A2, Ascend 950 | L1→L0B 搬运（CI1KHKWCOCI0→nZ）               |
| [copy_l0c_to_gm](./copy_l0c_to_gm/README.md) | 非 TLA + TLA | Atlas A2, Ascend 950 | L0C→GM 写回（zN→NC1HWC0，Fixpipe）           |
| [tile_copy](./tile_copy/README.md)           | 非 TLA + TLA | Atlas A2, Ascend 950 | 搬运聚合模板（TileCopy / PackedTileCopyTla） |
