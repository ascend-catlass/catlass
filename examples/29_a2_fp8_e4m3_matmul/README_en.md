# A2 FP8 E4M3 Matmul Example Readme

## Code Organization

```text
├── 29_a2_fp8_e4m3_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ ├── gen_data.py # Data generation script
│ └── fp8_matmul.cpp # Main file
```

## Function

This operator supports the input matrices A and B in the FP8 E4M3 format (software implementation), and then performs matrix multiplication to output matrix C (FP16).

## Implementation Details

1. Input processing: Receives two input matrices A and B in the FP8 E4M3 format.

2. Fake-quantization: Fake-quantizes the FP8 data into the FP16 format (per-tensor quantization mode).

3. Matrix multiplication: Performs matrix multiplication using FP16 data, and accumulates the intermediate results using FP32 precision.

4. Output conversion: The final result is converted into the FP16 format for output.

## Example

Using the example

- First step: Build.
- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

```bash
# Build a specified test case.
bash scripts/build.sh 29_a2_fp8_e4m3_matmul
```

- Second step: Run `gen_data.py` to generate test data. The test case specifications are entered from the command line.

```bash
cd examples/29_a2_fp8_e4m3_matmul && python gen_data.py 256 512 1024 0 0 && cd ../..
# The input parameters correspond to m, n, k, trans_a, and trans_b, respectively.
# trans_a indicates whether to transpose matrix A. The value 0 indicates no transpose, and the value 1 indicates transpose.
# trans_b indicates whether matrix B is transposed. The value 0 indicates that matrix B is not transposed, and the value 1 indicates that matrix B is transposed.
```

After the command is executed, the input and output directories are generated in the current path, containing the operator input data and the golden data used for accuracy verification.

```text
├── input
│   ├── a_8.bin
│   ├── b_8.bin
└── output
    └── expected_data.bin
```

- Step 3: Execute the operator. Ensure that the input shape provided to the operator is the same as the shape of the data generated in step 2.

```text
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./output/bin/29_a2_fp8_e4m3_matmul 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```cpp
Compare success.
```

## Description

1. Compared with FP16 Matmul, this sample has obvious GPU memory benefits for large-shape cases.

2. For small-shape scenarios, you can refer to [catlass_optimize_guidance](../../docs/en/1_Practice/11_matmul_optimization.md#tileshape-adjustment) to perform tiling optimization on the sample.

3. The input of `gen_data.py` supports trans_a and trans_b, but the 29_a2_fp8_e4m3_matmul executable file does not support them. It is only an example where both trans_a and trans_b are 0.

To handle transposed cases, modify the layout in the example, as the layout implicitly represents the transpose state: layout::RowMajor means no transpose, while layout::ColumnMajor means transpose.

The table below lists the kernels and applicable systems.

| trans_a | trans_b | LayoutA             | LayoutB             |
| ------- | ------- | ------------------- | ------------------- |
| 0       | 0       | layout::RowMajor    | layout::RowMajor    |
| 0       | 1       | layout::RowMajor    | layout::ColumnMajor |
| 1       | 0       | layout::ColumnMajor | layout::RowMajor    |
| 1       | 1       | layout::ColumnMajor | layout::ColumnMajor |
