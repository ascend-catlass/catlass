# TileCopy（Conv）

> [代码位置](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopy` 是 Conv 搬运聚合的非 TLA 版本（Atlas A2）。组合引用四个子搬运组件，以类型成员方式暴露供 block 层 Conv 使用。

- 适用范围：Atlas A2
- 风格：非 TLA

## 模板原型

```cpp
template <class ArchTag, class FmapType, class FilterType, class OutputType, class BiasType = void>
struct TileCopy;
```

| 模板参数     | 说明                   |
| :----------- | :--------------------- |
| `ArchTag`    | 架构标签               |
| `FmapType`   | Fmap 的 GemmType       |
| `FilterType` | Filter 的 GemmType     |
| `OutputType` | Output 的 GemmType     |
| `BiasType`   | Bias 类型，默认 `void` |

## 成员类型定义

| 成员类型      | 对应子组件                                 | 说明                        |
| :------------ | :----------------------------------------- | :-------------------------- |
| `CopyGmToL1A` | `Conv::Tile::CopyGmToL1<Arch, FmapType>`   | Fmap: GM→L1                 |
| `CopyGmToL1B` | `Conv::Tile::CopyGmToL1<Arch, FilterType>` | Filter: GM→L1               |
| `CopyL1ToL0A` | `Conv::Tile::CopyL1ToL0A<...>`             | Fmap: L1→L0A (im2col)       |
| `CopyL1ToL0B` | `Conv::Tile::CopyL1ToL0B<...>`             | Filter: L1→L0B              |
| `CopyL0CToGm` | `Conv::Tile::CopyL0CToGm<...>`             | L0C→GM (含 Fixpipe/F322F16) |
