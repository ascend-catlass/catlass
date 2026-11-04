# OptimizedMatmul Example Readme

## Code Organization

```text
├── 06_optimized_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── optimized_matmul.cpp # Main file
```

## Function

This example demonstrates optimized matrix multiplication. Compared to the `00_basic_matmul` example , this implementation replaces the dispatch policy with `MmadAtlasA2Preload` and introduces padding preprocessing for the input matrices to improve data transfer performance.

## Constraints

Constraint: `m`, `n`, and `k` must all be greater than 0; otherwise the example exits during argument parsing.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 06_optimized_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./06_optimized_matmul 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```

## Remarks

In this example, the default padding action uses `PADDING_NZ`. You can switch this to `PADDING_BLOCK_ND` to evaluate alternative performance profiles.

- **PADDING_NZ**
  The code configuration is as follows:

```cpp
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
```

The `COMPUTE_LENGTH` allocated in the UB under the `PADDING_NZ` policy is 48KB:

```cpp
static const uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
static const uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);
```

- **PADDING_BLOCK_ND**
  The modifications required to enable `PADDING_BLOCK_ND` are shown below. When the input matrix is not in NZ format, this policy aligns and pads the matrix according to `L1TileShape`:

```diff
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
-                                          : PaddingTag::PADDING_NZ;
+                                          : PaddingTag::PADDING_BLOCK_ND;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
-                                          : PaddingTag::PADDING_NZ;
+                                          : PaddingTag::PADDING_BLOCK_ND;
```

The `COMPUTE_LENGTH` allocated in the UB scales up to 96KB under the `PADDING_BLOCK_ND` policy:

```diff
-static const uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
-static const uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);
+static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
+static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
```
