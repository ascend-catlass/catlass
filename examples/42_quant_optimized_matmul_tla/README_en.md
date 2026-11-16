# Quant Optimized Matmul TLA Example Readme

## Code Organization

```text
├── 42_quant_optimized_matmul_tla
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│ └── quant_optimized_matmul_tla.cpp # Main file
```

## Remarks

The overall design of this test case is similar to that of 12_quant_matmul. The difference is that PaddingBlockND pre-processing and TLA-related abstractions are used. Therefore, related examples are provided.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 42_quant_optimized_matmul_tla
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./42_quant_optimized_matmul_tla 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
