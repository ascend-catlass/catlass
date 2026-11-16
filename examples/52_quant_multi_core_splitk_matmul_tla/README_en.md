# Quant Matmul Full LoadA Example Readme

## Code Organization

```text
├── 52_quant_multi_core_splitk_matmul_tla
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│ ├── 52_quant_multi_core_splitk_matmul_tla.md # Design document
│ └── quant_multi_core_splitk_matmul_tla.cpp # Main file
```

## Function

This template is used for quantization multi-core K splitting. By splitting K, more task blocks are divided to utilize more computing cores. TLA-related abstractions are also used. Therefore, examples are provided for illustration.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 52_quant_multi_core_splitk_matmul_tla
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./52_quant_multi_core_splitk_matmul_tla 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
