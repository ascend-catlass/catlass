# MatmulBias Example Readme

## Code Organization

```text
├── 20_matmul_bias
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── matmul_bias.cpp # Main file
```

## Constraints

Constraint: `m`, `n`, and `k` must all be greater than 0; otherwise the example exits during argument parsing.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- When matmul_bias is used to compute with `float32` input data, the following tiling is recommended to prevent exceeding the L1 cache capacity:

```cpp
using L1TileShape = GemmShape<112, 128, 256>;
using L0TileShape = GemmShape<112, 128, 64>;
```

- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 20_matmul_bias
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./20_matmul_bias 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
