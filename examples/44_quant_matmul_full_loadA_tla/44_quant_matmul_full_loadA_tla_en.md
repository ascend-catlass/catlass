# CATLASS Quant Matmul Full LoadA TLA Sample

## Prototype Design

| Name          | Type      | Data Type | Dimension | Format | Description                           |
| ------------- | --------- | --------- | --------- | ------ | ------------------------------------- |
| matA          | inTensor  | int8      | [m, k]    | ND     | Left matrix                           |
| matB          | inTensor  | int8      | [n, k]    | ND     | Right matrix. Transpose is supported. |
| scale         | inTensor  | float     | [n]       | ND     | Per-channel quantization scale        |
| perTokenScale | inTensor  | float     | [m]       | ND     | Per-token quantization scale          |
| matD          | outTensor | bf16      | [m, n]    | ND     | Output matrix                         |

## Sample Implementation

The CATLASS [44_quant_matmul_full_loadA_tla example](./README.md) operator is an Ascend-friendly Matmul operator implemented based on the CATLASS Gemm API. It is optimized for large matrix computation scenarios. The key operator components include the following parts:

- Example: [quant_matmul_full_loadA_tla.cpp](./quant_matmul_full_loadA_tla.cpp)
- Kernel implementation:
  - Main kernel file: [quant_matmul_full_loadA_tla.hpp](../../include/catlass/gemm/kernel/quant_matmul_full_loadA_tla.hpp)

- **Block components**, including:
  - Full-load dedicated mmad component [block_mmad_pingpong_full_loadA_tla.hpp](../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA_tla.hpp)
  - Dequantization post-processing component [block_epilogue_per_token_dequant_tla.hpp](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant_tla.hpp)
  - Multi-core full-load swizzle[GemmIdentityBlockSwizzleL1FullLoad](../../include/catlass/gemm/block/block_swizzle.hpp)

## Solution Design

- As shown in the following figure, the key parameter `L1TileShape<M, N, K>` exists in CATLASS matrix. Matrix C is used to divide basic blocks and cores according to the `L1TileShape::M` and `L1TileShape::N` parameters. Then, the common Matmul template loads the matrix blocks of size `L1TileShape::M * L1TileShape::K` in matrix A to L1, while the full-load template of matrix A directly loads a matrix block of size `L1TileShape::M * K` into L1. For matrix B, both the common template and the full-load template of matrix A load a matrix block of size `L1TileShape::K * L1TileShape::N` into L1.

![Full-load matrix A](https://raw.gitcode.com/user-images/assets/7631999/38ccafec-7fd8-4b9f-a4bb-dd57085ffb63/53e318e0feb94c578d888e67365e4c97.png_tplv-a9rns2rl98-image-qvalue.png "53e318e0feb94c578d888e67365e4c97.png~tplv-a9rns2rl98-image-qvalue.png")

- The following figure shows the pipeline changes of the full-load template of matrix A compared with the common template. With the full-load solution, at the beginning of the calculation, the data blocks of matrix A are completely moved to L1 through MTE2, and then ping-pong data movement is performed with matrix B for calculation.

![Full-load pipeline of matrix A](https://raw.gitcode.com/user-images/assets/7631999/1de46727-7c46-411e-936c-7a437d951a3a/3e9c799e1de0405d89f07a6bfd7d7c54.png_tplv-a9rns2rl98-image-qvalue.png "3e9c799e1de0405d89f07a6bfd7d7c54.png~tplv-a9rns2rl98-image-qvalue.png")

- When the full-load template of matrix A is used, half of the L1 space is required to store the `L1TileShape::M * problemShape.K` data. If the L1 space is insufficient for the full-load template of matrix A, an error is reported.
- When matrix A is fully loaded, a larger N-axis indicates that a single core can reuse matrix A in L1 multiple times without transferring matrix A from GM or L2 cache. In this case, the performance gain is greater.
- When matrix A is fully loaded and the N-axis is small, matrix A cannot be reused, and the performance gain may deteriorate compared with that of 00_basic_matmul.
- If `problemShape.M > L1TileShape::M`, the `GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>` policy can be used to connect the basic blocks to be processed by each core as much as possible, improving the inter-block reuse rate when matrix A is fully loaded by the core.
- If `problemShape.M <= L1TileShape::M`, the `GemmIdentityBlockSwizzle` policy can be used. For details about how to select common policy parameters, see [swizzle_explanation](../../docs/en/2_Design/01_kernel_design/02_swizzle.md).
- Taking 20 cube cores as an example, the core division sequence of basic blocks for the `GemmIdentityBlockSwizzle` policy is `0-1-2-...-18-19-0-1-2...-18-19-0-1-2...`, and the basic blocks to be processed by each core are distributed in a jump mode. The core division sequence of basic blocks for the `GemmIdentityBlockSwizzleL1FullLoad` policy is `0-0...-0-1-1...-1-2-2...-19`, and the basic blocks to be processed by each core are continuously distributed.

## Performance Benefits

When the same tileShape and swizzle parameters are used, the performance of the benchmark sample with the full-load feature of matrix A is improved by 5% to 15% on average compared with that of the benchmark sample 12_quant_matmul. This is because the full-load implementation reduces the transfer of matrix A during computing and improves the data reuse rate of the fully loaded matrix. For details, see the following table.

| [M,N,K]            | 12_quant_matmul | 44_quant_matmul_full_loadA_tla |
| ------------------ | --------------- | ------------------------------ |
| [512, 4096, 1024]  | 33.26us         | 31.73us                        |
| [128,16384, 1024]  | 38.68us         | 37.05us                        |
| [1024, 4096, 1024] | 42.66us         | 41.96us                        |
| [512,8192,1024]    | 52.36us         | 44.05us                        |
| [128,16384,2048]   | 53.77us         | 52.75us                        |

Note:

- The benchmark is the [QuantMatmul](../12_quant_matmul/README.md) operator.
- The total time consumed by the kernel function is obtained using the [msprof](https://www.hiascend.com/en/document/redirect/CannCommunitymsOpProf) tool.
- In the preceding test case, matrices A, B, and the output matrix are in `layout::RowMajor` format.
- Test environment description: The NPU model is 910B1, and the CANN package version is 9.0.0.
