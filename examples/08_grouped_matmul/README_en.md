# GroupedMatmul Example Readme

## Code Organization

```text
├── 08_grouped_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── grouped_matmul.cpp # Main file
```

## Remarks

- `grouped_matmul` is a general-purpose kernel, and the example is a scenario where the matrix is tiled along the k axis.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 08_grouped_matmul
cd output/bin
# Executable file name Group quantity | M-axis | N-axis | K-axis | Device ID
./08_grouped_matmul 128 512 1024 2048 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
