# LayoutTag (Legacy Layout)

This document introduces `LayoutTag` in CATLASS, that is, the legacy layout system.

## Overview

`LayoutTag` is the layout implementation from the early versions of CATLASS, defined under the `Catlass::layout` namespace. Unlike the new `tla::Layout`, `LayoutTag` binds the layout type and parameters into **fixed struct types**. Each layout pattern corresponds to an independent struct.

| LayoutTag            | Description                                                | RANK |
| -------------------- | ---------------------------------------------------------- | ---- |
| `RowMajor`           | Row-major matrix layout                                    | 2    |
| `ColumnMajor`        | Column-major matrix layout                                 | 2    |
| `VectorLayout`       | One-dimensional vector layout                              | 1    |
| `zN`                 | Row-major within fractal, column-major between fractals    | 4    |
| `nZ`                 | Column-major within fractal, row-major between fractals    | 4    |
| `zZ`                 | Row-major within fractal, row-major between fractals       | 4    |
| `nN`                 | Column-major within fractal, column-major between fractals | 4    |
| `PaddingRowMajor`    | Row-major tiled layout with padding                        | 4    |
| `PaddingColumnMajor` | Column-major tiled layout with padding                     | 4    |
| `NDC1HWC0`           | 5D convolution tensor layout                               | 5    |
| `KDC1KHKWN1N0C0`     | 4D convolution weight layout                               | 4    |

The code of all LayoutTags is located in [`include/catlass/layout/matrix.hpp`](../../../../include/catlass/layout/matrix.hpp), [`vector.hpp`](../../../../include/catlass/layout/vector.hpp), and [`tensor.hpp`](../../../../include/catlass/layout/tensor.hpp), and is collectively referenced by [`include/catlass/layout/layout.hpp`](../../../../include/catlass/layout/layout.hpp).

## Relationship Between LayoutTag and tla::Layout

`LayoutTag` is the legacy design, and `tla::Layout` is the new design. The two are bridged through the following mechanisms:

### `tla::MakeLayout<Element, LayoutTag>`

Recommended usage for the new design. Construct a `tla::Layout` directly by specifying the LayoutTag and element type as template parameters:

```cpp
auto layout = tla::MakeLayout<ElementA, Catlass::layout::RowMajor>(m, k);
```

Internally, it dispatches based on the LayoutTag type and constructs the corresponding nested or non-nested `tla::Layout`. This is the most common approach in real-world development.

### `tla::MakeLayoutFromTag`

Defined in [`include/tla/layout.hpp`](../../../../include/tla/layout.hpp), it converts any LayoutTag instance into a `tla::Layout`:

```cpp
template <class LayoutTag>
auto MakeLayoutFromTag(LayoutTag const& tag);
```

Internally, based on the specific type of LayoutTag, it extracts its `shape()` and `stride()` to construct the corresponding `tla::Layout`:

- `RowMajor` → `Layout<Shape<rows, cols>, Stride<ldm, _1>>`
- `ColumnMajor` → `Layout<Shape<rows, cols>, Stride<_1, ldm>>`
- `VectorLayout` → `Layout<Shape<len>, Stride<_1>>`
- `zN` / `nZ` → Nested `Layout<Shape<Shape<...>, Shape<...>>, Stride<Stride<...>, Stride<...>>>`

### `Catlass::detail::TagToLayout`

Defined in [`include/catlass/detail/tag_to_layout.hpp`](../../../../include/catlass/detail/tag_to_layout.hpp), it provides **compile-time type mapping** to map a LayoutTag to the corresponding `tla::Layout` type:

```cpp
template <class Element, class LayoutTag>
struct TagToLayout;

// Specialization example:
template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};
```

For fractal layouts (`zN`, `nZ` and `zZ`), `TagToLayout` calculates `ELE_NUM_PER_C0` and `ELE_NUM_PER_FRACTAL` based on the `Element` type to generate an accurate nested layout type.

### Relationship Summary

```cpp
    LayoutTag (Old)                                    tla::Layout (New)
─────────────                       ────────────────
RowMajor::MakeLayout(m, k)  ────►  MakeLayout<Element, RowMajor>(m, k)
       │                                    │
       └──── MakeLayoutFromTag() ──────────►┘
       └──── TagToLayout<>      ──────────►┘  (Compile-time type mapping)
```

In short, **LayoutTag is the specific layout type from the legacy system. New code uses `MakeLayout<Element, LayoutTag>()` to consume LayoutTag as a tag to construct `tla::Layout`**.

## Common Interfaces

All LayoutTags provide the following core interfaces:

| Interface                         | Return Type            | Description                                                                            |
| --------------------------------- | ---------------------- | -------------------------------------------------------------------------------------- |
| `MakeLayout<Element>(rows, cols)` | Its own type           | Static factory method that constructs a layout based on element type and logical size. |
| `GetOffset(coord)`                | `LongIndex`            | Computes the linear offset corresponding to a logical coordinate.                      |
| `GetTileLayout(tileShape)`        | Its own type           | Returns a layout view of a sub-tile. Stride is inherited from the parent layout.       |
| `shape()` / `shape(idx)`          | `Shape` / `Index`      | Obtains the shape of the layout (which may include a tile structure).                  |
| `stride()` / `stride(idx)`        | `Stride` / `LongIndex` | Gets the stride along each dimension.                                                  |
| `orgShape(idx)`                   | `Index`                | Gets the logical original size (only available for fractal layouts).                   |
| `Capacity()`                      | `LongIndex`            | Returns the total number of elements occupied by the layout.                           |

## Detailed Explanation of Each LayoutTag

### RowMajor

```cpp
struct RowMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
};
```

Row-major layout: Elements in adjacent columns in the same row are stored contiguously in memory with `stride = (ldm, 1)`.

**Construction:**

```cpp
// Default construction: stride is automatically deduced as (cols, 1).
RowMajor tag(rows, cols);

// Specify the leading dimension.
RowMajor tag(rows, cols, ldm);

// Use the MakeLayout factory.
auto tag = RowMajor::MakeLayout<Element>(rows, cols);
```

**GetOffset Calculation:**

```cpp
offset = row * stride[0] + col
```

That is, `offset = row * ldm + col`.

**Visualization**: The following uses a 3×4 matrix as an example to show the RowMajor memory layout.

```cpp
Logical matrix (3 rows × 4 cols):
        col0  col1  col2  col3
row0:   a00   a01   a02   a03
row1:   a10   a11   a12   a13
row2:   a20   a21   a22   a23

Memory layout (stride = (4, 1)), with elements in the same row stored contiguously:
Address:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
Data:  a00   a01   a02   a03   a10   a11   a12   a13   a20   a21   a22   a23
      └────── row0 ──────┘ └────── row1 ──────┘ └────── row2 ──────┘

offset(row, col) = row × 4 + col
For example, a12 is at row=1, col=2 → offset = 1 × 4 + 2 = 6.
```

### ColumnMajor

```cpp
struct ColumnMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
};
```

Column-major layout: Elements in adjacent rows in the same column are stored contiguously in memory with `stride = (1, ldm)`.

**Construction:**

```cpp
ColumnMajor tag(rows, cols);          // stride = (1, rows)
ColumnMajor tag(rows, cols, ldm);     // stride = (1, ldm)
auto tag = ColumnMajor::MakeLayout<Element>(rows, cols);
```

**GetOffset Calculation:**

```cpp
offset = row + col * stride[1]
```

**Visualization**: The following uses a 3×4 matrix as an example to show the ColumnMajor memory layout.

```cpp
Logical matrix (3 rows × 4 cols):
        col0  col1  col2  col3
row0:   a00   a01   a02   a03
row1:   a10   a11   a12   a13
row2:   a20   a21   a22   a23

Memory layout (stride = (1, 3)), with elements in the same column stored contiguously:
Address:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Data:  a00   a10   a20   a01   a11   a21   a02   a12   a22   a03   a13   a23
      └─── col0 ───┘ └─── col1 ───┘ └─── col2 ───┘ └─── col3 ───┘

offset(row, col) = row + col × 3
For example, a12 is at row=1, col=2 → offset = 1 + 2 × 3 = 7.
```

### VectorLayout — One-Dimensional Vector

```cpp
struct VectorLayout {
    static constexpr int RANK = 1;
};
```

Used for one-dimensional scenarios such as GEMV, Scale, and Bias. `stride` is fixed at `1`.

### zN — Row-Major Within Fractal, Column-Major Between Fractals

```cpp
struct zN {
    static constexpr int RANK = 4;
    static constexpr int ORG_SHAPE_RANK = 2;
};
```

`zN` is one of the most commonly used internal data formats of Ascend Cube cores. The layout rules are as follows:

- **Within fractal**: row-major
- **Between fractals**: column-major

The shape structure is `(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)`, that is:

```cpp
shape = (C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL,
         ELE_NUM_PER_C0,      colsRound / ELE_NUM_PER_C0)
```

In the preceding information:

- `C0_NUM_PER_FRACTAL` = 16 (number of rows in each fractal)
- `ELE_NUM_PER_C0` = `BYTE_PER_C0 / sizeof(Element)` (number of elements in each C0 unit)
- `ELE_NUM_PER_FRACTAL` = `BYTE_PER_FRACTAL / sizeof(Element)` (total number of elements in each fractal)

**GetOffset Calculation:**

```cpp
offset = (row / rowsInFractal) * strideRowsByFractal
       + (col / colsInFractal) * strideColsByFractal
       + (row % rowsInFractal) * strideRowsInFractal
       + (col % colsInFractal) * strideColsInFractal
```

**Visualization:** For ease of understanding, simplified parameters are used for demonstration (C0_NUM_PER_FRACTAL=2, ELE_NUM_PER_C0=2), with an original matrix of size 4 × 6:

```cpp
Original logical matrix (4 rows × 6 cols):
        col0  col1  col2  col3  col4  col5
row0:   a00   a01   a02   a03   a04   a05
row1:   a10   a11   a12   a13   a14   a15
row2:   a20   a21   a22   a23   a24   a25
row3:   a30   a31   a32   a33   a34   a35

Step 1 — Divide into 2 × 2 fractals.
         [col0,col1]  [col2,col3]  [col4,col5]
[r0,r1]  ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  F00    │  │  F01    │  │  F02    │
[r2,r3]  ├─────────┤  ├─────────┤  ├─────────┤
         │  F10    │  │  F11    │  │  F12    │
         └─────────┘  └─────────┘  └─────────┘

Step 2 — Follow the zN layout rule: column-major between fractals, row-major within fractal.
Column-major traversal order between fractals: F00 → F10 → F01 → F11 → F02 → F12
Row-major order within each fractal: F00 = [a00, a01, a10, a11]

Memory layout:
Address:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Data:  a00   a01   a10   a11   a20   a21   a30   a31   a02   a03   a12   a13
      └──── F00 ───┘ └──── F10 ───┘ └──── F01 ───┘

      +12   +13   +14   +15   +16   +17   +18   +19   +20   +21   +22   +23
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      a22   a23   a32   a33   a04   a05   a14   a15   a24   a25   a34   a35
      └──── F11 ───┘ └──── F02 ───┘ └──── F12 ───┘

Key understanding: Elements within the same fractal are contiguous in memory, and fractals are arranged along the column direction.
For example, F00 (fractal in row 0, column 0) is followed by F10 (fractal in row 1, column 0), not F01.
```

### nZ — Column-Major Within Fractal, Row-Major Between Fractals

nZ is the transpose counterpart of zN:

- **Within fractal**: column-major
- **Between fractals**: row-major

The shape structure is `(ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL, colsRound / C0_NUM_PER_FRACTAL)`.

**Visualization**: The following uses the same original matrix and fractal division as zN to compare the layout differences of nZ.

```cpp
Original logical matrix (4 rows × 6 cols):
        col0  col1  col2  col3  col4  col5
row0:   a00   a01   a02   a03   a04   a05
row1:   a10   a11   a12   a13   a14   a15
row2:   a20   a21   a22   a23   a24   a25
row3:   a30   a31   a32   a33   a34   a35

Step 1 — Divide into 2 × 2 fractals (same as zN):
         [col0,col1]  [col2,col3]  [col4,col5]
[r0,r1]  ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  F00    │  │  F01    │  │  F02    │
[r2,r3]  ├─────────┤  ├─────────┤  ├─────────┤
         │  F10    │  │  F11    │  │  F12    │
         └─────────┘  └─────────┘  └─────────┘

Step 2 — Follow the nZ layout rule: row-major between fractals, column-major within fractal.
Row-major traversal order between fractals: F00 → F01 → F02 → F10 → F11 → F12
Column-major order within each fractal: F00 = [a00, a10, a01, a11]

Memory layout:
Address:  +0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10   +11
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Data:  a00   a10   a01   a11   a02   a12   a03   a13   a04   a14   a05   a15
      └──── F00 ───┘ └──── F01 ───┘ └──── F02 ───┘

      +12   +13   +14   +15   +16   +17   +18   +19   +20   +21   +22   +23
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      a20   a30   a21   a31   a22   a32   a23   a33   a24   a34   a25   a35
      └──── F10 ───┘ └──── F11 ───┘ └──── F12 ───┘

Key understanding: Opposite to zN, fractals are arranged along the row direction, and elements within a fractal are arranged along the column direction.
For example, F00 (fractal in row 0, column 0) is followed by F01 (fractal in row 0, column 1), not F10.
```

### Comparison Between zN and nZ Layouts

Using the same matrix can intuitively exhibit the differences between zN and nZ layouts:

```cpp
Mapping from original matrix coordinates to memory addresses:

Element   Coordinate   zN Offset   nZ Offset
a00    (row=0,col=0)      0            0
a01    (row=0,col=1)      1            2
a10    (row=1,col=0)      2            1
a11    (row=1,col=1)      3            3
a02    (row=0,col=2)      8            4
a12    (row=1,col=2)     10            6
a20    (row=2,col=0)      4           12
a30    (row=3,col=0)      6           13

zN: Column-major between fractals → Traverse all fractals in column 0 first (F00,F10), then move to column 1 (F01,F11)
nZ: Row-major between fractals → Traverse all fractals in row 0 first (F00,F01,F02), then move to row 1 (F10,F11)
```

### zZ — Row-Major Within Fractal, Row-Major Between Fractals

- **Within fractal**: row-major
- **Between fractals**: row-major

Suitable for fractals that require fully row-major access.

### nN — Column-Major Within Fractal, Column-Major Between Fractals

- **Within fractal**: column-major
- **Between fractals**: column-major

nN is the transpose counterpart of `zZ`.

### Comparison of Four Fractal Layouts

| Layout | Within Fractal | Between Fractals | Typical Use Case             |
| ------ | -------------- | ---------------- | ---------------------------- |
| `zN`   | Row-major      | Col-major        | Matrix A input for CUBE core |
| `nZ`   | Col-major      | Row-major        | Matrix B input for CUBE core |
| `zZ`   | Row-major      | Row-major        | Special layout scenarios     |
| `nN`   | Col-major      | Col-major        | Special layout scenarios     |

### PaddingRowMajor / PaddingColumnMajor

These are padded layouts designed for non-512B aligned scenarios, improving memory access efficiency through tiling:

- `PaddingRowMajor`: row-major within block, row-major between blocks
- `PaddingColumnMajor`: column-major within block, column-major between blocks

The block size `blockRows` and `blockCols` must be specified during construction.

### NDC1HWC0 / KDC1KHKWN1N0C0

High-dimensional layouts specialized for convolution scenarios, defined at the end of `matrix.hpp`:

- `NDC1HWC0` (RANK=5): used for convolution input feature maps, with dimensions `(N, D, C1, H, W, C0)`.
- `KDC1KHKWN1N0C0` (RANK=4): used for convolution weights, with dimensions `(Kd*C1*Kh*Kw, N1, N0, C0)`.

## Example

The following example is from [`examples/53_ascend950_fp8_mx_matmul/fp8_mx_matmul.cpp`](../../../../examples/53_ascend950_fp8_mx_matmul/fp8_mx_matmul.cpp) and demonstrates the typical use of LayoutTag in GEMM operators:

### Step 1: Define LayoutTag Type Aliases

```cpp
using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::ColumnMajor;
using LayoutTagC = layout::RowMajor;
```

### Step 2: Construct LayoutTag Instances (for Memory Allocation and Benchmark Calculation)

```cpp
LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

size_t lenA = tagA.Capacity(); // Obtain the number of required elements.
size_t lenB = tagB.Capacity();
size_t lenC = tagC.Capacity();
```

### Step 3: Construct tla::Layout Using LayoutTag (for Kernel Computation)

```cpp
auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
```

### Step 4: Pass LayoutTag to Tile and Kernel Components

```cpp
using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ...>;
```

LayoutTag is passed as a template parameter to components such as TileCopy, allowing them to determine the data movement and computation layout strategy at compile time.

## Recommendations for Choosing Between LayoutTag and tla::Layout

| Scenario                                              | Recommendation                                                                                                              |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Newly written kernel/tile code                        | Directly use `tla::Layout` and construct it using `MakeLayout(shape, stride)`.                                              |
| Compatibility with the legacy GEMM framework          | Use LayoutTag as the template parameter and bridge it to `tla::Layout` internally using `MakeLayout<Element, LayoutTag>()`. |
| Simple matrix multiplication (row-major/column-major) | Both work, but LayoutTag is more concise.                                                                                   |
| Flexible layout customization                         | Use `tla::Layout`. LayoutTag supports only a few fixed patterns.                                                            |

Core principle: LayoutTag is a shortcut for fixed layout patterns. `tla::Layout` is the general-purpose layout representation. For new code, primarily use `tla::Layout`, with LayoutTag serving as a compatibility layer and convenient entry point.
