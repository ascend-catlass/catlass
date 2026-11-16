# MatmulSilu Example Readme

## Code Organization

```text
├── 28_matmul_silu
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── matmul_silu.cpp # Main file
```

## Function

SiLU:

$$
SiLU(x) = x \cdot Sigmoid(x)
$$

Sigmoid:

$$
Sigmoid(x)=\frac{1}{1+e^{-x}}
$$

Therefore, the calculation function is as follows:

$$
x = a \times b\\
out=\frac{x}{1+e^{-x}}
$$

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 28_matmul_silu
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./28_matmul_silu 256 512 1024 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
