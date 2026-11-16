# GroupedMatmulSliceMPerTokenDequant Example Readme

## Code Organization

```text
├── 10_grouped_matmul_slice_m_per_token_dequant
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── 10_grouped_matmul_slice_m_per_token_dequant.cpp # Main file
```

## Function

This operator supports the splitting of matrix A along the m-axis, performing matrix multiplication with matrix B by group, and then performing the per-token dequantization operation.
Matrices A and B are of the `int8` type, the scale is of the `bfloat16` type, and the output is of the `bfloat16` type.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 10_grouped_matmul_slice_m_per_token_dequant
cd output/bin
# Executable file name | Number of groups | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./10_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
