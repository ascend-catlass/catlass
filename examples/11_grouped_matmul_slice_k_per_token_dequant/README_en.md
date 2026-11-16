# Grouped Matmul Slice K Per Token Dequant Example Readme

## Code Organization

```text
├── 11_grouped_matmul_slice_k_per_token_dequant
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│   └── 11_grouped_matmul_slice_k_per_token_dequant.cpp # Main file
```

## Function

This operator supports splitting matrix A along the K axis and performing matrix multiplication on matrix B by group. Then, per token quantization is performed.
When the A/B matrix is of the int8 type and the scale is bf16, the output result is bf16.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 11_grouped_matmul_slice_k_per_token_dequant
cd output/bin
# Executable file name|Number of groups|Matrix M-axis|N-axis|K-axis|Device ID
# The device ID is optional. The default value is 0.
./11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
