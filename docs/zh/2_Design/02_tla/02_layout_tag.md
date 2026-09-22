# LayoutTag（旧版 Layout）

本文介绍 CATLASS 中的 `LayoutTag`，即旧版 Layout 体系。

## 概述

`LayoutTag` 是 CATLASS 早期版本的 Layout 实现，定义在 `Catlass::layout` 命名空间下。与新版 `tla::Layout` 不同，`LayoutTag` 将布局类型和参数**绑定为固定的结构体类型**，每种布局模式对应一个独立的 struct：

| LayoutTag            | 说明                       | RANK |
| -------------------- | -------------------------- | ---- |
| `RowMajor`           | 行优先矩阵布局             | 2    |
| `ColumnMajor`        | 列优先矩阵布局             | 2    |
| `VectorLayout`       | 一维向量布局               | 1    |
| `zN`                 | 分形内行优先、分形间列优先 | 4    |
| `nZ`                 | 分形内列优先、分形间行优先 | 4    |
| `zZ`                 | 分形内行优先、分形间行优先 | 4    |
| `nN`                 | 分形内列优先、分形间列优先 | 4    |
| `PaddingRowMajor`    | 带填充的行优先分块布局     | 4    |
| `PaddingColumnMajor` | 带填充的列优先分块布局     | 4    |
| `NDC1HWC0`           | 5 维卷积张量布局           | 5    |
| `KDC1KHKWN1N0C0`     | 4 维卷积权重布局           | 4    |

所有 LayoutTag 的代码位于 [`include/catlass/layout/matrix.hpp`](../../../../include/catlass/layout/matrix.hpp)、[`vector.hpp`](../../../../include/catlass/layout/vector.hpp) 和 [`tensor.hpp`](../../../../include/catlass/layout/tensor.hpp)，统一由 [`include/catlass/layout/layout.hpp`](../../../../include/catlass/layout/layout.hpp) 聚合引用。

## LayoutTag 与 tla::Layout 的关系

`LayoutTag` 是旧版设计，`tla::Layout` 是新版设计。两者通过以下机制桥接：

### `tla::MakeLayout<Element, LayoutTag>`

新版推荐用法，直接通过模板参数指定 LayoutTag 和元素类型来构造 `tla::Layout`：

```cpp
auto layout = tla::MakeLayout<ElementA, Catlass::layout::RowMajor>(m, k);
```

内部同样根据 LayoutTag 类型分发，构造对应的嵌套或非嵌套 `tla::Layout`。这是实际开发中最常用的方式。

### `tla::MakeLayoutFromTag`

定义在 [`include/tla/layout.hpp`](../../../../include/tla/layout.hpp)，将任意 LayoutTag 实例转换为 `tla::Layout`：

```cpp
template <class LayoutTag>
auto MakeLayoutFromTag(LayoutTag const& tag);
```

内部根据 LayoutTag 的具体类型，提取其 `shape()` 和 `stride()`，构造对应的 `tla::Layout`：

- `RowMajor` → `Layout<Shape<rows, cols>, Stride<ldm, _1>>`
- `ColumnMajor` → `Layout<Shape<rows, cols>, Stride<_1, ldm>>`
- `VectorLayout` → `Layout<Shape<len>, Stride<_1>>`
- `zN` / `nZ` → 嵌套 `Layout<Shape<Shape<...>, Shape<...>>, Stride<Stride<...>, Stride<...>>>`

### `Catlass::detail::TagToLayout`

定义在 [`include/catlass/detail/tag_to_layout.hpp`](../../../../include/catlass/detail/tag_to_layout.hpp)，提供**编译期类型映射**，将 LayoutTag 映射为对应的 `tla::Layout` 类型：

```cpp
template <class Element, class LayoutTag>
struct TagToLayout;

// 特化示例：
template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};
```

对于分形布局（`zN`、`nZ`、`zZ`），`TagToLayout` 会根据 `Element` 类型计算 `ELE_NUM_PER_C0` 和 `ELE_NUM_PER_FRACTAL`，生成精确的嵌套 Layout 类型。

### 关系总结

```cpp
LayoutTag (旧版)                     tla::Layout (新版)
─────────────                       ────────────────
RowMajor::MakeLayout(m, k)  ────►  MakeLayout<Element, RowMajor>(m, k)
       │                                    │
       └──── MakeLayoutFromTag() ──────────►┘
       └──── TagToLayout<>      ──────────►┘  (编译期类型映射)
```

简而言之：**LayoutTag 是旧版的具体布局类型，新版代码通过 `MakeLayout<Element, LayoutTag>()` 使用 LayoutTag 作为标签来构造 `tla::Layout`**。

## 通用接口

所有 LayoutTag 都提供以下核心接口：

| 接口                              | 返回类型               | 说明                                         |
| --------------------------------- | ---------------------- | -------------------------------------------- |
| `MakeLayout<Element>(rows, cols)` | 自身类型               | 静态工厂方法，根据元素类型和逻辑尺寸构造布局 |
| `GetOffset(coord)`                | `LongIndex`            | 计算逻辑坐标对应的线性偏移                   |
| `GetTileLayout(tileShape)`        | 自身类型               | 返回子 tile 的布局视图，stride 继承自父布局  |
| `shape()` / `shape(idx)`          | `Shape` / `Index`      | 获取布局的 shape（可能包含分块结构）         |
| `stride()` / `stride(idx)`        | `Stride` / `LongIndex` | 获取各维度的步长                             |
| `orgShape(idx)`                   | `Index`                | 获取逻辑原始尺寸（仅分形布局有此接口）       |
| `Capacity()`                      | `LongIndex`            | 返回布局占用的总元素数                       |

## 各 LayoutTag 详解

### RowMajor — 行优先矩阵

```cpp
struct RowMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
};
```

行优先布局：同一行内相邻列元素在内存中连续存放，`stride = (ldm, 1)`。

**构造方式：**

```cpp
// 默认构造：stride 自动推导为 (cols, 1)
RowMajor tag(rows, cols);

// 指定 leading dimension
RowMajor tag(rows, cols, ldm);

// 通过 MakeLayout 工厂
auto tag = RowMajor::MakeLayout<Element>(rows, cols);
```

**GetOffset 计算：**

```cpp
offset = row * stride[0] + col
```

即 `offset = row * ldm + col`。

**可视化示例：** 以 3×4 矩阵为例，展示 RowMajor 的内存排布：

```cpp
逻辑矩阵 (3 rows × 4 cols):
        col0  col1  col2  col3
row0:   a00   a01   a02   a03
row1:   a10   a11   a12   a13
row2:   a20   a21   a22   a23

内存排布 (stride = (4, 1))，同一行内元素连续存放：
地址:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
数据:  a00   a01   a02   a03   a10   a11   a12   a13   a20   a21   a22   a23
      └────── row0 ──────┘ └────── row1 ──────┘ └────── row2 ──────┘

offset(row, col) = row × 4 + col
例如：a12 位于 row=1, col=2 → offset = 1×4 + 2 = 6
```

### ColumnMajor — 列优先矩阵

```cpp
struct ColumnMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
};
```

列优先布局：同一列内相邻行元素在内存中连续存放，`stride = (1, ldm)`。

**构造方式：**

```cpp
ColumnMajor tag(rows, cols);          // stride = (1, rows)
ColumnMajor tag(rows, cols, ldm);     // stride = (1, ldm)
auto tag = ColumnMajor::MakeLayout<Element>(rows, cols);
```

**GetOffset 计算：**

```cpp
offset = row + col * stride[1]
```

**可视化示例：** 以 3×4 矩阵为例，展示 ColumnMajor 的内存排布：

```cpp
逻辑矩阵 (3 rows × 4 cols):
        col0  col1  col2  col3
row0:   a00   a01   a02   a03
row1:   a10   a11   a12   a13
row2:   a20   a21   a22   a23

内存排布 (stride = (1, 3))，同一列内元素连续存放：
地址:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
数据:  a00   a10   a20   a01   a11   a21   a02   a12   a22   a03   a13   a23
      └─── col0 ───┘ └─── col1 ───┘ └─── col2 ───┘ └─── col3 ───┘

offset(row, col) = row + col × 3
例如：a12 位于 row=1, col=2 → offset = 1 + 2×3 = 7
```

### VectorLayout — 一维向量

```cpp
struct VectorLayout {
    static constexpr int RANK = 1;
};
```

用于 GEMV、Scale、Bias 等一维场景。`stride` 固定为 `1`。

### zN — 分形内行优先、分形间列优先

```cpp
struct zN {
    static constexpr int RANK = 4;
    static constexpr int ORG_SHAPE_RANK = 2;
};
```

`zN` 是昇腾 CUBE 核最常用的内部数据格式之一。其排布规则为：

- **分形内部**：按行优先排布（小方块内 row-major）。
- **分形之间**：按列优先排布（分形块之间 col-major）。

Shape 结构为 `(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)`，即：

```cpp
shape = (C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL,
         ELE_NUM_PER_C0,      colsRound / ELE_NUM_PER_C0)
```

其中：

- `C0_NUM_PER_FRACTAL` = 16（每个分形块的行数）
- `ELE_NUM_PER_C0` = `BYTE_PER_C0 / sizeof(Element)`（每个 C0 单元的元素数）
- `ELE_NUM_PER_FRACTAL` = `BYTE_PER_FRACTAL / sizeof(Element)`（每个分形块的总元素数）

**GetOffset 计算：**

```cpp
offset = (row / rowsInFractal) * strideRowsByFractal
       + (col / colsInFractal) * strideColsByFractal
       + (row % rowsInFractal) * strideRowsInFractal
       + (col % colsInFractal) * strideColsInFractal
```

**可视化示例：** 为便于理解，以简化参数演示（C0_NUM_PER_FRACTAL=2, ELE_NUM_PER_C0=2），原始矩阵 4×6：

```cpp
原始逻辑矩阵 (4 rows × 6 cols):
        col0  col1  col2  col3  col4  col5
row0:   a00   a01   a02   a03   a04   a05
row1:   a10   a11   a12   a13   a14   a15
row2:   a20   a21   a22   a23   a24   a25
row3:   a30   a31   a32   a33   a34   a35

Step 1 — 划分为 2×2 的分形块（Fractal）:
         [col0,col1]  [col2,col3]  [col4,col5]
[r0,r1]  ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  F00    │  │  F01    │  │  F02    │
[r2,r3]  ├─────────┤  ├─────────┤  ├─────────┤
         │  F10    │  │  F11    │  │  F12    │
         └─────────┘  └─────────┘  └─────────┘

Step 2 — zN 排布规则：分形间列优先，分形内行优先
分形间列优先遍历顺序：F00 → F10 → F01 → F11 → F02 → F12
每个分形内部行优先：F00 = [a00, a01, a10, a11]

内存排布：
地址:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
数据:  a00   a01   a10   a11   a20   a21   a30   a31   a02   a03   a12   a13
      └──── F00 ───┘ └──── F10 ───┘ └──── F01 ───┘

      +12   +13   +14   +15   +16   +17   +18   +19   +20   +21   +22   +23
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      a22   a23   a32   a33   a04   a05   a14   a15   a24   a25   a34   a35
      └──── F11 ───┘ └──── F02 ───┘ └──── F12 ───┘

关键理解：同一分形块内的元素在内存中连续，分形块之间按列方向排列。
例如：F00(第0行块,第0列块) 后紧跟 F10(第1行块,第0列块)，而非 F01。
```

### nZ — 分形内列优先、分形间行优先

与 `zN` 互为转置关系：

- **分形内部**：按列优先排布。
- **分形之间**：按行优先排布。

Shape 结构为 `(ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL, colsRound / C0_NUM_PER_FRACTAL)`。

**可视化示例：** 使用与 zN 相同的原始矩阵和分形划分，对比展示 nZ 的排布差异：

```cpp
原始逻辑矩阵 (4 rows × 6 cols):
        col0  col1  col2  col3  col4  col5
row0:   a00   a01   a02   a03   a04   a05
row1:   a10   a11   a12   a13   a14   a15
row2:   a20   a21   a22   a23   a24   a25
row3:   a30   a31   a32   a33   a34   a35

Step 1 — 划分为 2×2 的分形块（与 zN 相同）:
         [col0,col1]  [col2,col3]  [col4,col5]
[r0,r1]  ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  F00    │  │  F01    │  │  F02    │
[r2,r3]  ├─────────┤  ├─────────┤  ├─────────┤
         │  F10    │  │  F11    │  │  F12    │
         └─────────┘  └─────────┘  └─────────┘

Step 2 — nZ 排布规则：分形间行优先，分形内列优先
分形间行优先遍历顺序：F00 → F01 → F02 → F10 → F11 → F12
每个分形内部列优先：F00 = [a00, a10, a01, a11]

内存排布：
地址:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
数据:  a00   a10   a01   a11   a02   a12   a03   a13   a04   a14   a05   a15
      └──── F00 ───┘ └──── F01 ───┘ └──── F02 ───┘

      +12   +13   +14   +15   +16   +17   +18   +19   +20   +21   +22   +23
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      a20   a30   a21   a31   a22   a32   a23   a33   a24   a34   a25   a35
      └──── F10 ───┘ └──── F11 ───┘ └──── F12 ───┘

关键理解：与 zN 相反，分形块按行方向排列，分形内部按列方向排列。
例如：F00(第0行块,第0列块) 后紧跟 F01(第0行块,第1列块)，而非 F10。
```

### zN 与 nZ 排布对比

将同一矩阵分别用 zN 和 nZ 排布，直观对比差异：

```cpp
原始矩阵坐标 → 内存地址映射：

元素    逻辑坐标      zN offset    nZ offset
a00    (row=0,col=0)      0            0
a01    (row=0,col=1)      1            2
a10    (row=1,col=0)      2            1
a11    (row=1,col=1)      3            3
a02    (row=0,col=2)      8            4
a12    (row=1,col=2)     10            5
a20    (row=2,col=0)      4           12
a30    (row=3,col=0)      6           13

zN:  分形间列优先 → 先遍历完第0列的所有分形块(F00,F10)，再进入第1列(F01,F11)
nZ:  分形间行优先 → 先遍历完第0行的所有分形块(F00,F01,F02)，再进入第1行(F10,F11)
```

### zZ — 分形内行优先、分形间行优先

- **分形内部**：按行优先排布。
- **分形之间**：按行优先排布。

适用于需要完全行优先访问的分形场景。

### nN — 分形内列优先、分形间列优先

- **分形内部**：按列优先排布。
- **分形之间**：按列优先排布。

与 `zZ` 互为转置关系。

### 四种分形布局对比

| 布局 | 分形内部  | 分形之间  | 典型用途           |
| ---- | --------- | --------- | ------------------ |
| `zN` | Row-major | Col-major | CUBE 核 A 矩阵输入 |
| `nZ` | Col-major | Row-major | CUBE 核 B 矩阵输入 |
| `zZ` | Row-major | Row-major | 特殊排布场景       |
| `nN` | Col-major | Col-major | 特殊排布场景       |

### PaddingRowMajor / PaddingColumnMajor

针对非 512Byte 对齐场景设计的带填充布局，通过分块方式提升访存效率：

- `PaddingRowMajor`：块内行优先，块间行优先。
- `PaddingColumnMajor`：块内列优先，块间列优先。

构造时需指定块大小 `blockRows` 和 `blockCols`。

### NDC1HWC0 / KDC1KHKWN1N0C0

卷积场景的专用高维布局，定义在 `matrix.hpp` 末尾：

- `NDC1HWC0`（RANK=6）：用于卷积输入特征图，维度为 `(N, D, C1, H, W, C0)`。
- `KDC1KHKWN1N0C0`（RANK=4）：用于卷积权重，维度为 `(Kd*C1*Kh*Kw, N1, N0, C0)`。

## 实际使用示例

以下示例来自 [`examples/53_ascend950_fp8_mx_matmul/fp8_mx_matmul.cpp`](../../../../examples/53_ascend950_fp8_mx_matmul/fp8_mx_matmul.cpp)，展示 LayoutTag 在 GEMM 算子中的典型用法：

### 步骤 1：定义 LayoutTag 类型别名

```cpp
using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::ColumnMajor;
using LayoutTagC = layout::RowMajor;
```

### 步骤 2：用 LayoutTag 构造实例（用于内存分配和标杆计算）

```cpp
LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

size_t lenA = tagA.Capacity();  // 获取所需元素数量
size_t lenB = tagB.Capacity();
size_t lenC = tagC.Capacity();
```

### 步骤 3：用 LayoutTag 构造 tla::Layout（用于 kernel 计算）

```cpp
auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
```

### 步骤 4：将 LayoutTag 传递给 Tile 和 Kernel

```cpp
using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ...>;
```

LayoutTag 作为模板参数传递给 TileCopy 等组件，使其在编译期确定数据搬运和计算的布局策略。

## 与 tla::Layout 的选择建议

| 场景                              | 推荐                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| 新编写的 kernel / tile 代码       | 直接使用 `tla::Layout`，通过 `MakeLayout(shape, stride)` 构造                                 |
| 需要兼容旧版 GEMM 框架            | 使用 LayoutTag 作为模板参数，内部通过 `MakeLayout<Element, LayoutTag>()` 桥接到 `tla::Layout` |
| 简单的矩阵乘法（行优先 / 列优先） | 两者均可，LayoutTag 更简洁                                                                    |
| 需要灵活自定义布局                | 使用 `tla::Layout`，LayoutTag 只支持固定的几种模式                                            |

核心原则：**LayoutTag 是固定布局模式的快捷方式，`tla::Layout` 是通用的布局表达。新版代码推荐以 `tla::Layout` 为主，LayoutTag 作为兼容层和便捷入口。**
