# W8A16 Matmul Example Readme

## Code Organization

```text
├── 30_w8a16_matmul
│   ├── CMakeLists.txt     #CMake build file
│   ├── README.md
│ └── w8a16_matmul.cpp # Main file
```

## Function

- Added dequantization functionality: convert input B matrix from `int8` to `fp16_t` (`half`), then perform dequantization: sum with `deqZeroPoint`, multiply by `deqScalar`, and then perform Matmul with matrix A.
- The current implementation only supports RowMajor and ColumnMajor data layouts.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 30_w8a16_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./30_w8a16_matmul 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
