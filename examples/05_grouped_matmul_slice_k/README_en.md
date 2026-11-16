# Grouped Matmul Slice K Example Readme

## Code Organization

```text
├── 05_grouped_matmul_slice_k
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── grouped_matmul_slice_k.cpp # Main file
```

## Function

This operator supports the splitting of matrix A along the k-axis, performing matrix multiplication with matrix B by group,

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 05_grouped_matmul_slice_k
cd output/bin
# Executable file name Group quantity | M-axis | N-axis | K-axis | Device ID
./05_grouped_matmul_slice_k 128 512 1024 2048 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
