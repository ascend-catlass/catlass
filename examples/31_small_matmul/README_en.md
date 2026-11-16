# SmallMatmul Example Readme

## Code Organization

```text
├── 31_small_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── small_matmul.cpp # Main file
```

## Function

- This operator reduces unnecessary scalar computation overheads based on basic_matmul in the small-shape scenarios.
- The number of basic blocks to be tiled cannot exceed the number of cube cores, that is, `ceilDiv(m, L1TileShape::M)*ceilDiv(n, L1TileShape::N) ≤ aicCoreNum`.
- The k axis cannot exceed `L1TileShape::K`.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 31_small_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./31_small_matmul 256 1024 256 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
