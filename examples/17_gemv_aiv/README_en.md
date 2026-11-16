# Gemv Aiv Example Readme

## Code Organization

```text
├── 17_gemv_aiv
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── gemv_aiv.cpp # Main file
```

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 17_gemv_aiv
cd output/bin
# Executable file name | Matrix M-axis | N-axis | Device ID
# The device ID is optional. The default value is 0.
./17_gemv_aiv 256 512 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
