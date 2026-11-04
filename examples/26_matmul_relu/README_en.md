# MatmulRelu Example Readme

## Code Organization

```text
├── 26_matmul_relu
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── matmul_relu.cpp # Main file
```

## Function

Performs the following mathematical calculation:

$$
out = ReLU(a*b)
$$

## Constraints

Constraint: `m`, `n`, and `k` must all be greater than 0; otherwise the example exits during argument parsing.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified test case
bash scripts/build.sh 26_matmul_relu
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./26_matmul_relu 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
