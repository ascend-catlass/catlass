# SingleSplitK_Matmul Example Readme

## Description

- Operator function: Optimized matrix multiplication computation. For details about the optimization strategy, see the [Single-Core Split-K Policy Description](./34_single_splitk_matmul.md).

## Parameters

The command-line arguments for this sample include $M$, $N$, $K$, and `deviceId`, adhering to the identical structural criteria used in [00_basic_matmul Parameters](../00_basic_matmul/README_en.md#parameters).
The underlying operator prototype is structured as follows:

| Name | Class     | Data Type        | Dimensions | Format | Description                           |
| ---- | --------- | ---------------- | ---------- | ------ | ------------------------------------- |
| matA | inTensor  | fp16\|bf16\|fp32 | [m, k]     | ND\|NZ | Left matrix; supports transposition.  |
| matB | inTensor  | fp16\|bf16\|fp32 | [k, n]     | ND\|NZ | Right matrix; supports transposition. |
| matC | outTensor | fp16\|bf16\|fp32 | [m, n]     | ND     | Output matrix.                        |

## Constraints

None.

## Accumulation and K Usage Range

This operator targets performance tuning and retains its implementation that computes K tiles and accumulates them with FP32 atomic additions. K is the input reduction length, `L1TileShape::K` is the number of elements along K per tile, and the tile count is `ceil(K / L1TileShape::K)`. Long FP32 addition chains accumulate rounding error, so the selected usage range for FP16, BF16, and FP32 inputs is `1 <= K <= 8192`.

In the current PyTorch configuration, FP32 uses `L1TileShape::K=256`, so K=8192 corresponds to `8192 / 256 = 32` tiles. FP16 and BF16 use `L1TileShape::K=512`, giving 16 tiles for the same K. K=262145 requires 1025 tiles in the FP32 configuration and is outside this usage range. The upper bound of 8192 is a choice that retains the original algorithm, not a hardware limit, a mathematical error guarantee, or a runtime rejection condition in the kernel. Rounding error still depends on input values, cancellation, and tile configuration; the usage range needs reevaluation when the configuration changes.

## Code Organization

The sample code is organized as follows:

```text
├── 34_single_splitk_matmul
│   ├── CMakeLists.txt           # CMake build file
│   ├── single_core_splitk.cpp   # Main file
│   └── README.md
```

## Example

Compile the sample code to generate the corresponding operator executable:

```bash
# Compile a specified test case.
bash scripts/build.sh 34_single_core_splitk_matmul
```

Go to the compilation directory `output/bin` of the executable file and run the operator sample program. Similar to the [00_basic_matmul](../00_basic_matmul/README.md) implementation, verification workloads are randomly initialized based on the dimensions provided in the command line:

```bash
cd output/bin
# Executable name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./34_single_core_splitk_matmul 2048 2048 4000 0
```

• 2048: matrix m-axis

• 2048: n-axis

• 4000: k-axis

• 0: Device ID (optional). Defaults to 0.

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
