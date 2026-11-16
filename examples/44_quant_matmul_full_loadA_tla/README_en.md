# Quant Matmul Full LoadA Example Readme

## Code Organization

```text
├── 44_quant_matmul_full_loadA_tla
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│ ├── 44_quant_matmul_full_loadA_tla.md # Design document
│ └── quant_matmul_full_loadA_tla.cpp # Main file
```

## Function

Based on 12_quant_matmul, this operator supports full loading of matrix A. A single core can load the entire matrix A into the L1 cache and make it resident to reduce repeated movement of matrix A in some matrix computation scenarios and improve performance. Currently, the full-load template of matrix A does not support input containing bias.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 44_quant_matmul_full_loadA_tla
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./44_quant_matmul_full_loadA_tla 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
