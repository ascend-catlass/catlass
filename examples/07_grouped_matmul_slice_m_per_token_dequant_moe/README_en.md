# GroupedMatmulSliceMPerTokenDequantMoe Example Readme

## Code Organization

```text
├── 07_grouped_matmul_slice_m_per_token_dequant_moe
│   ├── CMakeLists.txt     #CMake build file
│   ├── README.md
│   └── grouped_matmul_slice_m_per_token_dequant_moe.cpp # Main file
```

## Function

This operator supports grouped matrix multiplication where Matrix A is sliced along the M-axis and Matrix B is partitioned by group, followed by a per-token dequantization operation.
Matrix A and Matrix B are of `int8` type, the scale factor is of `fp32` type, and the output results are of `fp16` type.

## Example

Due to the large number of configuration parameters for `GroupedMatmul`, this example directly generates the input parameter `groupList` in the host code.

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 07_grouped_matmul_slice_m_per_token_dequant_moe
cd output/bin
# Executable file name | Number of groups | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
