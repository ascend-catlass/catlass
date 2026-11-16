# 950_grouped_matmul_slice_m_per_token_dequant Example Readme

## Code Organization

```text
├── 47_ascend950_grouped_matmul_slice_m_per_token_dequant
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── grouped_matmul_slice_m_per_token_dequant_tla.cpp # Main file
```

## Function

This CV fusion operator implements grouped matrix multiplication and dequantization operations optimized for Ascend 950. It provides an efficient execution path for grouped, sliced ($M$-axis) matrix multiplication fused with both per-token and per-channel dequantization.

## Solution Overview

1. Introduces the [GroupedMatmulSliceMPerTokenTla kernel template class](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_tla.hpp). By leveraging [BlockMmad](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp), [Epilogue](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp), and [Scheduler](../../include/catlass/gemm/block/block_scheduler_aswt.hpp), it supports dequantization workloads across a collection of matrices defined by `groupCount`.
2. Adds the [EpilogueAscend950PerTokenDequantTla](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp) epilogue template for Ascend 950, which manages loading quantization scale factors from Global Memory (GM) and executing dequantization operations locally within the same Unified Buffer (UB).
3. Adds the [TilePerTokenDequant](../../include/catlass/epilogue/tile/tile_pertoken_dequant.hpp) tile template to enable high-performance dequantization calculations on Ascend 950.

## Parameters

| Name      | Class     | Data Type      | Dimensions         | Format | Description                                           |
| --------- | --------- | -------------- | ------------------ | ------ | ----------------------------------------------------- |
| matA      | inTensor  | int8           | [m, k]             | ND     | Left matrix                                           |
| matB      | inTensor  | int8           | [groupCount, n, k] | ND     | Right matrix, supports transposition                  |
| groupList | inTensor  | int32          | [groupCount]       | ND     | Group size in the m-axis direction, accumulation list |
| scale     | inTensor  | bf16/fp16/fp32 | [groupCount, n]    | ND     | perChannel quantization scale                         |
| perToken  | inTensor  | bf16/fp16/fp32 | [m]                | ND     | perToken quantization scale                           |
| matD      | outTensor | bf16/fp16/fp32 | [m, n]             | ND     | Output matrix                                         |

## Example

- After obtaining the code, compile the corresponding operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#). This test case is an Ascend 950 operator. During compilation, you need to add `DCATLASS_ARCH=3510`.
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 47_ascend950_grouped_matmul_slice_m_per_token_dequant -DCATLASS_ARCH=3510
cd output/bin
# Executable file name | Number of groups | Matrix M-axis | N-axis | K-axis | Device ID
# The number of groups and the dimensions of the matrix m-axis, n-axis, and k-axis must be greater than 0.
# The device ID is optional. The default value is 0.
./47_ascend950_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
