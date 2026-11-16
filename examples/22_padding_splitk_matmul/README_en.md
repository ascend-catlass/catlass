# PaddingSplitkMatmul Example Readme

## Code Organization

```text
├── 22_padding_splitk_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── padding_splitk_matmul.cpp # Main file
```

## Function

This operator supports additional split-K parallelization along the k-axis for multi-core allocation after the padding of matrices A and B is completed. This improves Cube core utilization when the `m` and `n` dimensions are small.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 22_padding_splitk_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./22_padding_splitk_matmul 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
