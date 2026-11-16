# GroupedMatmulSliceM Example Readme

## Code Organization

```text
├── 02_grouped_matmul_slice_m
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│   └── grouped_matmul_slice_m.cpp #Main file
```

## Function

This operator supports splitting matrix A along the M axis and then performing matrix multiplication on matrix B by group.

## Example

Because GroupedMatmul has many parameters, the example directly carries the output parameter list `groupList` in the code and uses `golden::GenerateGroupList` to generate a random split sequence.
For details about the input configuration, see [grouped_matmul_slice_m.cpp](grouped_matmul_slice_m.cpp).
If the grouplist configuration is required (for example, the input is constructed in tensorList mode), see the corresponding implementation in `python_extension`.

Using the example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 02_grouped_matmul_slice_m
cd output/bin
# Executable file name|Number of groups|Matrix M-axis|N-axis|K-axis|Device ID
# The device ID is optional. The default value is 0.
./02_grouped_matmul_slice_m 128 512 1024 2048 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
