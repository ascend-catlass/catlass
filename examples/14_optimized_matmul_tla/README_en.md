# Optimized Matmul TLA Example Readme

## Code Organization

```text
├── 14_optimized_matmul_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── optimized_matmul_tla.cpp # Main file
```

## Remarks

The overall design of this test case is the same as that of 06_optimized_matmul. The difference is that TLA-related abstractions are used. Therefore, related examples are provided.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 14_optimized_matmul_tla
cd output/bin
# Executable name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./14_optimized_matmul_tla 256 512 1024 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
