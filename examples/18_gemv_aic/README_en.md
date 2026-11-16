# GemvAic Example Readme

## Code Organization

```text
├── 18_gemv_aic
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── gemv_aic.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 18_gemv_aic
cd output/bin
# Executable file name | Matrix M-axis | N-axis | Device ID
# The device ID is optional. The default value is 0.
./18_gemv_aic 256 512 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
