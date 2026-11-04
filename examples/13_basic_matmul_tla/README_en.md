# BasicMatmulTla Example Readme

## Code Organization

```text
├── 13_basic_matmul_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── basic_matmul_tla.cpp # Main file
```

## Remarks

The overall design of this test case is the same as that of [00_basic_matmul](../00_basic_matmul/README.md). The difference is that TLA-related abstractions are used. Therefore, related examples are provided.

## Constraints

Constraint: `m`, `n`, and `k` must all be greater than 0; otherwise the example exits during argument parsing.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 13_basic_matmul_tla
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./13_basic_matmul_tla 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
