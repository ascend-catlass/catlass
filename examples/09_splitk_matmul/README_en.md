# Split K Matmul Example Readme

## Code Organization

```text
├── 09_splitk_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── splitk_matmul.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 09_splitk_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./09_splitk_matmul 256 512 1024 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
