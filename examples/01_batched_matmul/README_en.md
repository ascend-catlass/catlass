# BatchedMatmul Example Readme

## Code Organization

```text
├── 01_batched_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── batched_matmul.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 01_batched_matmul
cd output/bin
# Executable file name batch axis | m-axis | n-axis | k-axis | Device ID
# The device ID is optional. The default value is 0.
./01_batched_matmul 5 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
