# CHANGELOG

## CATLASS 2.X

### CATLASS 2.2.0

#### CATLASS Python

- Key Features
  - **SIMD**: Extended SIMD operations, data transfers, and matrix computation for Ascend 950: added the `PostModeUpdate` mode for Load/Store in three scenarios: aligned contiguous, aligned non-contiguous (DataBlock-strided), and unaligned contiguous transfers, automatically updating addresses by the stride after each transfer; supported Histogram in both `FREQUENCY` and `ACCUMULATE` modes; supported inline ReLU (`relu_enable`) for both L0C→GM and L0C→UB transfers; and added PER_TENSOR / PER_CHANNEL quantization with scale loading, Reg→Reg `tla.gather`, unsigned integer transfers/Compare/Cast, L1→BiasTable transfers, and ordinary Matmul with Bias.
  - **SIMT**: Officially introduced SIMT programming: `tla.vec.func(mode="simt", thread_block_dim=...)` describes per-thread scalar computation, with thread coordinates, scalar capture across Blocks, task partitioning, per-thread GM/UB access, and thread-level arithmetic, comparison, selection, type conversion, and floating-point math. SIMT can be combined with Cube/SIMD in the same Kernel.
  - **Framework Capabilities**: Kernels can be organized as class instances: classes can encapsulate `tla.kernel` and access member variables; `tla.compile` can compile class instances or their bound methods; and instance attributes participate in orchestration and caching as compile-time constants.
- More Examples (New Ascend 950 Python Examples)
  - [Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md): Extended the existing nomask path with paged KV (block-table addressing) and a separate causal mask kernel, covering FP16/BF16, BSND/TND, GQA, page boundaries, and tail blocks.
  - [MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/mx_mmad/README.md): Demonstrates MX FP8/FP4 data preparation, E8M0 scale blocks, GM→L1→L0 transfers with scales, `tla.mmad_mx`, and FP32 result writeback.
  - [SIMT Multi-Block Cooperation](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multiple_blocks_simt/README.md): Distributes tasks across Blocks using linearized 3D thread coordinates and a global stride, covering different Block counts, thread configurations, and lengths that are not evenly divisible.
  - [SIMT Multiple Math Operations](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/README.md): Provides floating-point and integer computation paths covering arithmetic, comparison, `where`, type conversion, and floating-point math functions.
- Bug Fixes & Optimizations
  - Overrode the HIVMC compilation default `cce-vf-auto-sync=global` with `fused`: in `global` mode, CCEC analyzes and inserts synchronization; in `fused` mode, developers ensure synchronization correctness. This override avoids redundant synchronization barriers inserted by `global` inside VF loops.

#### CATLASS C++

- More Examples (New Ascend 950 Example)
  - [Ascend 950 HSTU Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/83_ascend950_hstu_infer/README.md): Implements `S=QKᵀ`, `P=siluScale×S×sigmoid(S)`, and `O=PV`, with an optional causal multiplicative mask, NTD/TND layouts, variable-length sequences, and Paged KV Cache.
- Documentation
  - Aligned CanImplement, Host input checks, and README constraints: established input support matrices to keep data type, layout, shape, alignment, paging, and Mask constraints consistent with implementations; added explicit diagnostics for invalid inputs before Kernel launch; and added regression cases for valid boundaries and invalid inputs. Initial coverage includes Basic Matmul, Grouped Matmul, Optimized Matmul, Quant series, Ascend 950 examples, X-Attention, and Basic SYRK.
- Bug Fixes & Optimizations
  - Optimized golden data generation for [Broadcast Matmul PerBlock Quant](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/62_ascend950_broadcast_matmul_perblock_quant/README.md) and [A8W4 Grouped MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/README.md): replaced E4M3 LUT processing with chunked vectorization and binary nearest-neighbor search, and eliminated repeated FP4 index computation. Outputs remain byte-for-byte identical; a single Broadcast Matmul PerBlock Quant case now takes a few seconds instead of approximately 37 minutes.
  - Fixed accuracy issues in [MLA](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/19_mla/README.md) atomic rescaling using addition in place of multiplication at zero, subnormal, and exponent-underflow boundaries; added missing synchronization for UB reuse; and removed extreme-value zeroing that could incorrectly clear valid large values.
  - Fixed unsigned underflow in Conv3d/Conv2d output size calculations using `uint32`; invalid shapes, including `stride=0`, are rejected by CanImplement / Host entry points.
  - Fixed accuracy issues in [Ascend 950 Flash Attention Chunk Prefill](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/70_ascend950_flash_attention_chunk_prefill/README.md) when `kvHeads != 1`, and updated its constraints and README.
  - Fixed inconsistencies between optest and the [A8W4 MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/59_ascend950_a8w4_mx_matmul/README.md) example.
  - Fixed the missing Bias argument in the catlass_cppgen EVG template.

### CATLASS 2.1.0

#### CATLASS Python

- Key Features
  - Extended low-precision computation and register programming for Ascend 950:
    - Supported MXFP8/MXFP4 matrix multiplication, E8M0 scale blocks, and MX layouts, with new `copy(..., scale=...)` and `mmad_mx` APIs.
    - Extended register data transfers with contiguous/non-contiguous aligned loads, multiple LoadDist modes, and single-element and interleaved dual-register stores.
    - Added masked register vector/scalar shift operations, including signed right shifts.
  - Added the [`@tla.extern`](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md) declaration API to declare and call user-provided Ascend C functions from DSL Kernels and reuse existing low-level implementations.
  - Added [AutoSync](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/dsl_development/feature_development/auto_sync_design.md): `@tla.kernel(auto_sync="v0")` automatically analyzes on-chip memory dependencies and handles synchronization between pipelines within a single AIC or AIV.
  - Enhanced compile-time reuse: [`@tla.kernel` supports Constexpr Callable arguments](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md#constexpr-callable-%E5%85%A5%E5%8F%82), allowing post-processing, Mask, and activation logic to be reused as compile-time policies included in specialization and compilation caching.
  - Improved DSL distribution support with source builds and installation of the [`ascend-catlass-dsl` wheel package](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/quick_start.md).
- Documentation
  - Added the [Vector Op Execution Guide](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/vector_ops/README.md), covering compilation, execution, and result validation for binary, unary, comparison, transfer, reduction, and register control-flow operations.
  - Added the [Host API Reference](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md), along with documentation on [DSL control flow](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/kernel_development/core_concepts/control_flow.md) and [AutoSync design and usage constraints](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/dsl_development/feature_development/auto_sync_design.md).
- Bug Fixes & Optimizations
  - Optimized DSL JIT execution and dynamic Tensor argument wrapping. Repeated launches reuse loaded binaries and function handles, reducing steady-state launch overhead for dynamic Tensors from 28 μs to 15 μs.

#### CATLASS C++

- More Examples (One New Ascend 950 Example and One Atlas A2 and Atlas A3 Example)
  - [Ascend 950 Basic SYRK](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/82_ascend950_basic_syrk/README.md): Supports the symmetric rank update `Y = X·Xᵀ`, computing only lower-triangular tiles and writing both symmetric positions from a single computation.
  - [Atlas A2 X-Attention](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/78_x_attention/README.md): Provides data generation, Tiling, Kernel, Host, and PyTorch reference implementations; fuses shared/unshared KV attention and supports two cache combinations.
- Bug Fixes & Optimizations
  - Fixed out-of-bounds access and accuracy issues in Basic Conv2D with large Padding and unaligned output channel counts.
  - Fixed incorrect MLA Tiling selection and missing tail-task writeback for certain core counts and shapes.
  - Fixed division by zero at BlockMmad Shuffle-K boundaries and the risk of stride overflow in large Batched Matmul cases.

### CATLASS 2.0.0

#### CATLASS Python

- Officially introduced **CATLASS Python**, supporting Pythonic Kernels and native Tile programming. The TLA Dialect is built on AscendNPU-IR, lowering TLA operations described in Python to MLIR and compiling them into executable kernels with the AscendNPU-IR toolchain.
  - Covers matrix multiplication, vector computation, and dynamic control flow. `tla.print` supports formatted scalar output on `GM/UB` and tensor/scalar printing within dynamic control flow.
  - Provides unified Host-side tensor construction through `from_dlpack` / `make_fake_tensor`, and supports `@dataclass` structures as kernel arguments.
- Examples
  - [Basic Mmad](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad/README.md): Basic matrix multiplication examples: `basic_matmul`, `basic_matmul_atomic_add`, `basic_matmul_mutex`, `basic_matmul_mutex_with`, and `basic_mmad_ptr` (mutex / atomic_add / ptr paths).
  - [Basic Mixed](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mixed/README.md): Basic Cube/Vector operator examples: `basic_mixed`, `basic_mixed_mutex`, `basic_mixed_fixpipe_nz2dn`, `basic_mixed_store_zN`, `basic_mixed_store_zNUnAlign`, and `basic_mixed_ub2l1` (mixed copy+mmad scenarios).
  - [Matmul Epilogue](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad_epilogue/README.md): Cube/Vector fusion examples with post-processing: `matmul_add`, `matmul_add_ub`, `matmul_bias`, `matmul_leaky_relu`, `matmul_sigmoid`, `matmul_silu`, and `matmul_tanh`.
  - [StreamK Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad_streamk/README.md): Matrix multiplication with StreamK scheduling: full-K computation for regular tiles and multi-core split-K reduction for tiles in the final round.
  - [Multi Core SplitK Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/multi_core_splitk_matmul/README.md): Multi-core split-K matrix multiplication: `multi_core_splitk_matmul` and `tail_multi_core_splitk_matmul`.
  - [Batched Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/batched_matmul/README.md): Batched matrix multiplication.
  - [Grouped Matmul Slice M](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/grouped_matmul_slice_m/README.md): Grouped matrix multiplication partitioned along the M axis.
  - [Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md): Flash Attention inference example supporting non-masked, non-paged scenarios with aligned and unaligned inputs.
- Tool Support
  - Build system: builds on the AscendNPU-IR submodule, provides a Docker development image, and supports CANN >= 9.1.0.
- Documentation
  - Provides the [CATLASS Python documentation](https://gitcode.com/cann/catlass/tree/v2.0.0/python/tla_dsl/docs/), including the [Kernel API Reference](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/kernel_api_reference.md), [DSL Python Syntax Guide](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/dsl_python_syntax_guide.md), [BC Backend Integration](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/bc-backend-integration.md), [Environment Variables](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/environment_variables.md), and [Environment Setup Guide](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/dev_guide/00_environment_setup.md).

#### CATLASS C++

- Key Features
  - Added the [official CATLASS documentation site](https://catlass.readthedocs.io/zh-cn/latest/) as a single entry point for learning and development.
  - Added the [catlass_cppgen code generation framework](https://gitcode.com/cann/catlass/blob/v2.0.0/python/catlass_cppgen/README.md): a Python-based operator code generation framework for defining operator parameters, selecting optimization strategies, and automatically generating C++ kernel code for GEMM and GroupGEMM operators, with EVG and OpTensor API documentation and unit tests.
- More Examples (Nine New Examples: Six for Ascend 950 and Three for Atlas A2)
  - [Ascend 950 Svd Quant Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/61_ascend950_svd_quant_matmul/README.md): SVD low-rank decomposition quantization, combining a W4A4 quantized main path with high-precision computation in a low-rank side path.
  - [Ascend 950 Dual-Level Quant MX BatchMatmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/63_ascend950_dual_level_quant_mx_batch_matmul/README.md): Dual-level quantization with MXFP4 BatchMatmul.
  - [Ascend 950 Grouped MXFP8 Matmul SwiGLU MX Quant](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/README.md): Grouped MXFP8 matrix multiplication fused with SwiGLU and MXFP8 quantization.
  - [Ascend 950 Grouped MXFP8 Matmul Finalize Routing](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/README.md): Grouped MXFP8 matrix multiplication fused with Finalize Routing, with deterministic and non-deterministic variants.
  - [Ascend 950 A8W4 Grouped MX Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/README.md): A8W4 Grouped MX matrix multiplication for weight-only quantization scenarios.
  - [Ascend 950 Grouped Matmul Slice M Gelu](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/80_ascend950_grouped_matmul_slice_m_gelu/README.md): Grouped Matmul SliceM fused with Gelu.
  - [Atlas A2 Symm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/75_symm/README.md): Symmetric matrix multiplication.
  - [Atlas A2 Trmm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/76_trmm/README.md): Triangular matrix multiplication.
  - [Atlas A2 Planar Complex Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/77_planar_complex_matmul/README.md): Planar complex matrix multiplication with separate real and imaginary outputs.
- Tool Support
  - Expanded [optest](https://gitcode.com/cann/catlass/blob/v2.0.0/tests/optest/README.md) to 72 interfaces, adding 10 interfaces including Symm, Trmm, PlanarComplexMatmul, GroupedMatmulSliceMGelu, and A8W4GroupedMxMatmul.
  - Added Agent Skills for [CANNBot CATLASS C++](https://gitcode.com/cann/cannbot/blob/master/plugins-official/catlass-op-generator/quickstart.md) and CANNBot CATLASS Python operator generation toolchains, an [A2/A3-to-Ascend 950 migration workflow](https://gitcode.com/cann/catlass/tree/v2.0.0/.agents/skills/catlass-ascend950-migration), and a [Tile-level unit test writing skill](https://gitcode.com/cann/catlass/tree/v2.0.0/.agents/skills/catlass-tile-level-ut).
- Documentation
  - Updated the gcc version constraints in the README.
  - Updated Gemm/Tile UT dependency documentation.
- Bug Fixes & Optimizations
  - Optimized MLA with the `AMLATp1Spec` specialized Kernel template and a KV-split strategy for balanced core assignment, and fixed incorrect golden data type reads.
  - Fixed accuracy issues in the Ascend 950 A8W4 MxMatmul example for unaligned inputs.
  - Fixed accuracy issues in the Ascend 950 MxGroupedMatmulSliceM example for MxFp4 types and transB scenarios.
  - Fixed JIT compilation failures for unaligned shapes in Quant Optimized Matmul Tla under optest.

## CATLASS 1.X

### CATLASS 1.6.0

- Key Features
  - Added full support for **Ascend 950**, with a complete set of [Tile components](https://gitcode.com/cann/catlass/tree/v1.6.0/include/catlass/gemm/tile/ascend950) (see the [Gemm/Tile Template Overview](https://gitcode.com/cann/catlass/blob/v1.6.0/docs/zh/3_API/include/catlass/gemm/tile/README.md)) compatible with Atlas A2 and Atlas A3, and added [**unit tests**](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/unittest/catlass/gemm/tile/README.md) for all Gemm Tile components on **Atlas A2/Atlas A3** and **Ascend 950** architectures.
  - Added [**MXFP8 / MXFP4 quantization templates**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/53_ascend950_fp8_mx_matmul/README.md), supporting MXFP8/MXFP4 quantized matrix multiplication and A8W4 MX matrix multiplication on Ascend 950.
  - Added [**BlockMmad based on Mutex synchronization primitives**](https://gitcode.com/cann/catlass/blob/v1.6.0/include/catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp), simplifying Block component implementations.
  - Added the [**EVG declarative post-processing framework**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/64_ascend950_matmul_evg/README.md) for describing Epilogue logic declaratively.
  - Supported compilation with the Ascend C CMake build system.
- More Examples (16 New Ascend 950 Examples and One Atlas A2 and Atlas A3 Example)
  - [Ascend 950 Flash Attention Chunk Prefill](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/70_ascend950_flash_attention_chunk_prefill/README.md)
  - [Ascend 950 MXFP8 Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/53_ascend950_fp8_mx_matmul/README.md)
  - [Ascend 950 MXFP4 Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/54_ascend950_fp4_mx_matmul/README.md)
  - [Ascend 950 A8W4 MX Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/59_ascend950_a8w4_mx_matmul/README.md)
  - [Ascend 950 MXFP8 BatchMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/58_ascend950_fp8_mx_batch_matmul/README.md)
  - [Ascend 950 MXFP8/MXFP4 GroupedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/55_ascend950_mx_grouped_matmul_slice_m/README.md)
  - [Ascend 950 Matmul Full Dequant](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/57_ascend950_matmul_full_dequant/README.md)
  - [Ascend 950 Broadcast Matmul PerBlock Quant](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/62_ascend950_broadcast_matmul_perblock_quant/README.md)
  - [Ascend 950 Matmul EVG](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/64_ascend950_matmul_evg/README.md)
  - [Ascend 950 GroupedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/60_ascend950_grouped_matmul_slice_m/README.md)
  - [Ascend 950 BatchedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/67_ascend950_batched_matmul/README.md)
  - [Ascend 950 Matmul FullLoadA](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/73_ascend950_matmul_full_loadA/README.md)
  - [Ascend 950 StreamK Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/66_ascend950_streamk_matmul/README.md)
  - [Ascend 950 MultiCoreSplitkMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/68_ascend950_multi_core_splitk_matmul/README.md)
  - [Ascend 950 TailMultiCoreSplitkMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/69_ascend950_tail_multi_core_splitk_matmul/README.md)
  - [Ascend 950 Conv2d](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/56_ascend950_basic_conv2d_tla/README.md)
  - [Atlas A2 Quant Matmul Multi-Core Split-K](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/52_quant_multi_core_splitk_matmul_tla/README.md)
- Tool Support
  - Added the [operator-level test framework (optest)](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/optest/README.md), supporting JIT and Prebuilt modes, unified prebuilt kernel loading through weak symbols, and torch_catlass test interfaces for 63 existing and new examples, covering Matmul, GroupedMatmul, Flash Attention, and other operator types.
  - Enhanced [msTuner](https://gitcode.com/cann/catlass/blob/v1.6.0/tools/tuner/README.md) with support for -xasc compilation and Ascend 950 tuning.
  - Added [Agent Skill development tools](https://gitcode.com/cann/catlass/tree/v1.6.0/.agents), automating tasks such as generating torch interfaces and optest deliverables for examples.
  - Added CI/build infrastructure, including pre-commit scripts and dependency description files.
- Documentation
  - Added [EVG declarative post-processing framework documentation](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/2_Design/03_evg/01_evg_design.md).
  - Added [documentation for migrating existing non-TLA operators to Ascend 950](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md).
  - Expanded [Tile API documentation](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/3_API/README.md).
  - Added separate Chinese and English documentation directories, with multiple rounds of corrections and link fixes.
- Bug Fixes & Optimizations
  - Replaced fftsAddr with hardwareSyncAddr, obtained through aclrtGetHardwareSyncAddr.
  - Fixed RAW memory races caused by missing synchronization in EpilogueAtlasA2PerTokenDequant.
  - Fixed a bug with zN inputs in the TLA version of quant_matmul.
  - Fixed potentially unaligned compute lengths in RemovePaddingNDAndCast.
  - Fixed potential uint32/int32 overflows in some examples.

### CATLASS 1.5.0

- Key Features
  - Added support for the Ascend 950 architecture and its underlying template components.
  - TLA enhancement: introduced origin_shape, added APIs such as TileView, MakeTensorLike, and Tensor operator(), and improved layout and tensor expressions.
  - Matmul generalization project: The W8A8 Per-Token + Per-Channel dynamic quantization path and batch compilation are supported, and related design documents are supplemented.
  - FixPipe capability extension: Added/enhanced combined templates and examples such as Matmul FixPipe optimization and GMM + FixPipe + Dequant
  - Adapted to CANN 9.0.0.beta2. When g++ is linked with the BiSheng toolchain, profapi needs to be explicitly linked (compiler non-compatibility change description).
  - Added unit test and CI adaptation for Ascend 950.
- More Examples
  - [Ascend 950 Basic Matmul](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/43_ascend950_basic_matmul/README.md)
  - [Ascend 950 Matmul FixPipe Optimization](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/46_ascend950_matmul_fixpipe_opti/README.md)
  - [Ascend 950 Grouped Matmul SliceM Per-Token Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/47_ascend950_grouped_matmul_slice_m_per_token_dequant/README.md)
  - [Ascend 950 Grouped Matmul Per-Tensor & Per-Channel Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant/README.md)
  - [Ascend 950 Flash Attention Inference](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/49_ascend950_flash_attention_infer/README.md)
  - [Ascend 950 Basic Matmul GEMV](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/50_ascend950_basic_matmul_gemv/README.md)
  - [Ascend 950 Quant Matmul Per-Group & Per-Block TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/51_ascend950_quant_matmul_per_group_per_block_tla/README.md) (Per-Group x Per-Block Quantization)
  - [Quant Optimized Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/42_quant_optimized_matmul_tla/README.md)
  - [Quant Matmul Full LoadA TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/44_quant_matmul_full_loadA_tla/README.md)
  - [Strided Batched Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/45_strided_batched_matmul_tla/README.md)
  - [Matmul Generalization Project: dynamic W8A8 per-token quantization](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/103_dynamic_optimized_quant_matmul_per_token_basic/README.md)
- Tool Support
  - [shared\_lib](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/shared_lib/README.md): Added soname to output artifacts; shared libraries depended on by Python extensions are now switched to `.so` with version information.
  - [Python extension](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/python_extension/README.md): Supports passing build options to `build.sh` and supports asynchronous mode; updated device-side pointer access methods.
  - [msTuner\_CATLASS](https://gitcode.com/cann/catlass/tree/v1.5.0/tools/tuner/README.md): Extended GEMM configuration and search space; [MatmulGelu](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/27_matmul_gelu/README.md): example added for tuning demonstration.
- Documentation
  - Adjusted documentation directory and resource paths (e.g.,`figures` directory), with error fixes and content revisions.
  - Added related documentation for the generalization project, such as MultiCoreSplitK, StreamK, and Single Core Split-K.
  - Fixed incorrect statements in [ascendc\_dump documentation](https://gitcode.com/cann/catlass/blob/v1.5.0/docs/1_Practice/evaluation_tools/ascendc_dump.md)
- Bug Fixes & Optimizations
  - Fixed issues with TLA OriginShape and Flash Attention Golden, etc. Improved implementation paths for Ascend 950 FA's Block/Epilogue.
  - Adjusted blockLen calculation logic in [CopyGmToL1](https://gitcode.com/cann/catlass/blob/v1.5.0/include/catlass/gemm/tile/copy_gm_to_l1.hpp), eliminating unnecessary rounding up for alignment with `C0_NUM_PER_FRACTAL`.
  - NaN Special: Added zero-clear for Ki=0 scenarios in `exp11` and related paths to prevent dirty data from affecting model accuracy.
  - Fixed issues like CopyL0CToDstQuantMode. Corrected the Ascend 950 architecture identifier from 3501 to 3510.
  - Fixed indirect header file references, License comments, and multiple documentation typos. Continuously eliminated code specification warnings and style issues.

### CATLASS 1.4.0

- Key Features
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md) added:
    - `LocalPaddingCPaddingCommonMatmul` template, which uses local workspace to pad matrix C.
- More Examples
  - [StreamK Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md)
  - [W4A4 Low-Precision Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md)
  - [Matmul Operator L2 Tiling + Staggered Core Distribution](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/39_big_matmul_tla/README.md)
  - [Sparse Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md)
- Tool Support
  - Added the [shared\_lib](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/shared_lib/basic_matmul_shared_lib.cpp) usage example.
- Documentation
  - Added [detailed design document for the single-core split-K optimized matmul operator](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/contents/example_design/34_single_splitk_matmul.md), introducing the design concept and code breakdown for single-core split-K matrix multiplication.
  - Added [performance data display for Matmul/GroupedMatmul operator on the homepage](https://gitcode.com/cann/catlass/blob/v1.4.0/README.md).
  - [msdebug document](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/tools/msdebug.md) added driver support check.
- Bug Fixes & Optimizations
  - Optimized the [grouped\_matmul\_slice\_m](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/02_grouped_matmul_slice_m/README.md) example to support segmented `groupList` input.
  - Added 32Byte alignment constraint for `TileShape` in BlockMmad.
  - Fixed the data length issue in the [w4a8 matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/32_w4a8_matmul/w4a8.cpp) example.
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md) now supports scenarios where stride is greater than shape.
  - [msTuner\_CATLASS](https://gitcode.com/cann/catlass/tree/v1.4.0/tools/tuner/README.md) now supports tiling tuning for `quant Matmul`.
  - Fix the improper use of `cmath` on the device.
  - Fixed accuracy issues and naming errors in [Matmul Silu](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/28_matmul_silu/README.md).
  - Fixed vector synchronization issues in [cast\_int4\_to\_int8](https://gitcode.com/cann/catlass/blob/v1.4.0/include/catlass/gemm/tile/cast_int4_to_int8.hpp) component.
  - Fixed half data type usage issues in the [w8a16 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/30_w8a16_matmul/w8a16_matmul.cpp) operator.
  - Fixed logic issues in the [Single-Core Split-K Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/34_single_core_splitk_matmul/single_core_splitk.cpp) when `RemovePaddingNDAndCastC` is empty.

### CATLASS 1.3.0

- Key Features
  - Lowered the minimum required `CMake` version from 3.22 to 3.16.
  - Added support for [`FixPipe` inline quantization](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373).
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md) added:
    - `PaddingCommonMatmul`
    - `SmallMatmul`
    - `PaddingMultiCoreSplitkMatmul`
    - `PaddingStreamkMatmul`
    - `Single-Core Split-K Template Series`
    - `Dynamic Swizzle`
- More Examples
  - [Matmul operator for INT4 dequantization](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md)
  - [2D convolution operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md)
  - [Single-core split-K optimized matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/34_single_core_splitk_matmul/README.md)
- Tool Support
  - Added the [msOpGen](https://www.hiascend.com/document/detail/en/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0018.html) tool code example [basic_matmul](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/basic_matmul_aclnn.cpp) and [access document](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/README.md).
  - [msTuner_CATLASS](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md) added:
    - `GroupedMatmulSliceM operator`
    - `OptimizedMatmul operator`
- Documentation
  - Added [detailed design document for the INT8 type dequantization GroupedMatmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/example_design/10_grouped_matmul_slice_m_per_token_dequant.md), introducing the design concept and code breakdown for groupMatmul + epilogue type operators.
  - Added [matrix multiplication template summary document](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/matmul_template_summary.md),summarizing existing Matmul template designs in the template library.
  - Added [CommonMatmul description document](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/doc/CommonMatmul.md), introducing the basic templates in the generalized Matmul project.
- Bug Fixes & Optimizations
  - Fixed memory issues introduced when copying masks during softmax in the [Flash Attention inference operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/23_flash_attention_infer/README.md).
  - Fixed documentation errors.
    - [catlass_optimize_guidance.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/catlass_optimize_guidance.md)
    - [api.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/api.md)
    - [Template Library Quick Start.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/quickstart.md)
    - [tutorials.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/tutorials.md)
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md) updates:
    - Modified `TilingParams` reading method for better readability.
    - Optimized the original `Splitk ReduceAdd` to fully utilize the UB space.
    - Added Python environment detection in `CMakeLists.txt`.
  - Fixed the issue where the [optimized Matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/06_optimized_matmul/README.md) did not support PADDING_NZ in the kernel.
  - Optimized and reconstructed the [FP8 dequantized Matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/29_a2_fp8_e4m3_matmul/README.md) to better comply with the Prologue paradigm.
  - Fixed the accuracy issues of the [Matmul bias operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/20_matmul_bias/README.md) and added verification and interception for bf16.
  - Optimized simulation compilation logic and compilation issues in the A3 environment; the compilation logic for `simulator` mode is now the same as for the board-on mode.
  - [msTuner_CATLASS tool](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md) updates:
    - Added security checks for interface replacement, illegal characters, and maximum `groupCount` value.
    - Fixed the issue where the default `ffts_addr` parameter was blocked when dispatching some operators.
  - Changed the default number of consecutive times the cross-core flag can be set to avoid system freezes after exceeding the limit.

### CATLASS 1.2.0

- Key Features
  - Support for passing the compute platform architecture during operator compilation.
  - Added [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md) example:
    - Automatically determines Tiling parameters based on feature sizes.
    - You can select the best template from a set of preset operator templates.

- More Examples
  - [Flash Attention Inference Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer/README.md)
  - [3D Convolution Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias/README.md)
  - [Matrix A Full Load Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA/README.md)
  - [Small Matrix Optimization Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul/README.md)
  - [MatmulRelu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu/README.md)
  - [MatmulGelu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu/README.md)
  - [MatmulSilu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_silu/README.md)
  - [FP8 Dequantized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul/README.md)
  - [INT8 Dequantized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul/README.md)

- Tool Support
  - Updated the content of [Python API Calling](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/python_extension/README.md):
    - Adjusted the project organization structure.
    - Added support for transposed cases.
  - Added [`msTuner_CATLASS`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner/README.md) tool for automatic tiling tuning, which runs exhaustively within the search space to collect performance data.
  - Added support for enabling the [`msSanitizer`](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0039.html) address sanitizer tool (add --enable_mssanitizer to compile options).

- Documentation
  - Added [`catlass_optimize_guidance.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/contents/advanced/catlass_optimize_guidance.md) document, introducing common optimization methods for `Gemm` operators empowered by CATLASS.

- Bug Fixes & Optimizations
  - Optimized the [`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md) operator implementation to support any padding combinations.
  - Fixed the issue where the `msTuner_CATLASS` tool could not obtain the actual running `DeviceId` when the `ASCEND_RT_VISIBLE_DEVICES` environment variable was enabled.
  - Fixed the `Set/Wait` mismatch exception in the [PFA operator example](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/19_mla/README.md)for single-row data scenarios.
  - Fixed `Workspace` size calculation errors in the [`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md) operator example.
  - Isolated the code segments that enable the `AscendC::Dump` and `AscendC::print` functions.
  - Fixed the output zero-clearing behavior of the [`GroupedMatmulSliceK`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/05_grouped_matmul_slice_k/README.md) operator for the special case Ki=0, and adjusted the true value comparison logic to full (M, N, K) dimensions.
  - Fixed errors in documentation such as [`performance_tools.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tools/performance_tools.md) and [`tutorials.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tutorials.md).

### CATLASS 1.1.0

- Key Features
  - Provides engineering components for calling CATLASS operators from Python.
    - Can compile [pybind](https://github.com/pybind/pybind11) extensions and [PyTorch](https://pytorch.org/) extensions.
  - Supports operator simulation (by enabling the `--simulator` option during compilation).
  - Compilation process adapted to the [Bisheng](https://www.hiascend.com/cann/bisheng) compiler.

- More Examples
  - [MatmulBias Operator with Bias](https://gitcode.com/cann/catlass/blob/v1.1.0/examples/20_matmul_bias/README.md)
  - [Preload-Optimized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/21_basic_matmul_preload_zN/README.md) (co-developed with iFLYTEK)
  - [Split-K-Optimized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/22_padding_splitk_matmul/README.md) (co-developed with iFLYTEK)

- Tool Support
  - Added support for using [`AscendC::Dump`](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html) and [`AscendC::printf`](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html) for print debugging.
    - Added `--enable_ascendc_dump` and `--enable_print` to the compilation options to enable the preceding functions.
    - See [ascendc_dump](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/ascendc_dump.md) and [print](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/print.md).

- Documentation
  - Added [Tutorials quick start examples](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tutorials.md).
  - Added documentation for operator performance tuning using the [msProf tool](https://www.hiascend.com/document/detail/en/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html): [msProf](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tools/performance_tools.md) performance tuning.

- Bug Fixes & Optimizations
  - Optimized the AIC program at the `Kernel` level, adding `PIPE_ALL` to avoid impact on the whole network.
  - Optimized the implementation of the [optimized Matmul](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/06_optimized_matmul/README.md) operator to disable the AIV core in unnecessary padding scenarios.
  - Fixed the error that occurs when `nextBlock` is preloaded at the `Block` layer.
  - Isolated the `inline` definition of `AscendC` on the kernel side to prevent some standard libraries from being unavailable during heterogeneous programming.
  - Modified the redefinition issue of the `l2offset` setting.

### CATLASS 1.0.0

- Official open-source release of the [CATLASS](https://gitcode.com/cann/catlass/) template library

- Key Features
  - Provides hierarchical operator development capabilities including Kernel, Block, Tile, and Basic layers.

- Sample Reference
  - Provides operator examples including basic Matmul and various different optimization strategies.
