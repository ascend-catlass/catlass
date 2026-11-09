# BasicMatmulPreloadZN Example Readme

## Code Organization

```text
├── 21_basic_matmul_preload_zN
│   ├── CMakeLists.txt   # CMake build file
│   ├── README.md
│   └── basic_matmul_preload_zN.cpp # Main file
```

## Function

This example is based on `00_basic_matmul` and uses the `MmadAtlasA2Preload` dispatch policy. In the current configuration, A and C use `RowMajor` layouts, and B uses the `layout::zN` fractal layout. For transposed B, change the example's `LayoutB` from `layout::zN` to `layout::nZ`. Recompile after changing the example source.

## Parameters

| Parameter | Shape | Data type | Layout | Description |
| ---- | ---- | ---- | ---- | ---- |
| A | `(m, k)` | `float16` | `RowMajor` | Input matrix |
| B | `(k, n)` | `float16` | `zN` fractal layout | Input matrix; default layout |
| C | `(m, n)` | `float16` | `RowMajor` | Output matrix |

These parameters describe the current example configuration. Recompile and update this document if the data types or layouts in the source are changed.

## Usage Scope and Constraints

- The current B input uses the `zN` fractal layout. Its logical dimension `k` must be a multiple of 16, and `n` must be a multiple of `32 / sizeof(ElementB)` (16 in the current configuration); inputs that do not meet these conditions are not supported by the current example.
- `M ≥ 256, N ≥ 256, 256 < K ≤ 3072` is the recommended MNK range for performance.
- When both A and B are fp32, `K ≤ 6144` is recommended. These recommendations are not size boundaries for all valid inputs.

## Example

- Compile the operator executable after obtaining the source. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator:

```bash
# Compile a specified case
bash scripts/build.sh 21_basic_matmul_preload_zN
cd /output/bin
# Executable file | Matrix m-axis | n-axis | k-axis | Device ID
# Device ID is optional and defaults to 0.
./21_basic_matmul_preload_zN 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```

## Recommended Use Case

This example and `06_optimized_matmul` use the same `MmadAtlasA2Preload` dispatch policy and tile. The difference is that this example uses the `zN` fractal layout (NZ layout) for B, so no ND2NZ on-the-fly conversion is needed when reading B from GM to L1. The recommended MNK range is `M ≥ 256, N ≥ 256, 256 < K ≤ 3072`. When both A and B are fp32, `K ≤ 6144` is recommended. These ranges are performance recommendations, not hard size boundaries.
