# Matmul Add Example Readme

## Code Organization

```text
├── 03_matmul_add
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── matmul_add.cpp # Main file
```

## Constraints

Constraint: `m`, `n`, and `k` must all be greater than 0; otherwise the example exits during argument parsing.

## Example

- After obtaining the code, build the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 03_matmul_add
cd output/bin
# Executable name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./03_matmul_add 256 512 1024 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
