# Gemm Example Readme

## Code Organization

```text
├── 15_gemm
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── gemm.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 15_gemm
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./15_gemm 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
