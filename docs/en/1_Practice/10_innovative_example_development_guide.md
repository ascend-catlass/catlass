# Innovative Example Development Process Guide

This document is intended for developers who need to develop innovative operator examples in the CATLASS template library. Using [example 44](../../../examples/44_quant_matmul_full_loadA_tla/README.md) `quant_matmul_full_loadA_tla` (quantized matrix multiplication with the A matrix fully loaded) as a case study, it explains the complete process from requirements analysis and solution design to component development and test integration, helping developers quickly master how to develop innovative examples.

## What Is an Innovative Example

Every example in the CATLASS template library is an innovation and extension built on existing examples, each carrying its own new features. The core idea of developing a new example is: **maximize reuse of existing components and only innovate where necessary**. When your operator requirements cannot be fully satisfied by combining existing components — for example, when you need a new data transfer strategy, new post-processing computation logic, or a new multi-core scheduling approach — you need to develop new Block-level components or Kernel logic based on existing examples.

Operator examples can be found in the [examples](../../../examples/) directory.

Take [example 44](../../../examples/44_quant_matmul_full_loadA_tla/README.md) as an example. It was not built from scratch; instead, it combines the features of two existing examples: the post-processing logic of the quantized matrix multiply (BlockEpilogue) reuses the per-token dequantization scheme of [example 42](../../../examples/42_quant_optimized_matmul_tla/README.md) (`42_quant_optimized_matmul_tla`), and the A-matrix full-load strategy (including BlockMmad and BlockScheduler) refers to [example 25](../../../examples/25_matmul_full_loadA/README.md) (`25_matmul_full_loadA`). Its BlockMmad component (`block_mmad_pingpong_full_loadA_tla.hpp`) is a TLA (Tensor Layout Abstraction, see [TLA Layout Design](../2_Design/02_tla/01_layout.md)) rewrite of the existing full-load component (`block_mmad_pingpong_full_loadA.hpp`), and the BlockScheduler (`GemmIdentityBlockSwizzleL1FullLoad`) directly reuses the full-load scheduling strategy of [example 25](../../../examples/25_matmul_full_loadA/README.md). [Example 44](../../../examples/44_quant_matmul_full_loadA_tla/README.md) did not add any component from scratch; all modules were completed by reusing or TLA-rewriting existing components.

## Design Stage

### Requirements Analysis

Before writing code, you first need to clarify the core requirements of the operator and **thoroughly investigate existing examples and template components** to determine which can be reused directly, which need to be rewritten by reference, and which must be developed from scratch:

- **Operator functionality**: clarify the mathematical definition and computation flow of the operator. Example 44 implements the quantized matrix multiplication `D = (A_int8*B_int8) * scale * perTokenScale`, with bf16 output.
- **Investigation of existing components**: traverse existing examples and the template components under `include/catlass/` to identify reusable resources. The investigation conclusions for example 44 are as follows:

    | Feature                         | Reference source                               | Reuse method               |
    | ---------------------------- | -------------------------------------- | ---------------------- |
    | Quantized matrix multiply + per-token dequantization | example 42 `42_quant_optimized_matmul_tla` | BlockEpilogue reused directly  |
    | A matrix fully loaded to L1                | example 25 `25_matmul_full_loadA`          | pipeline design idea referenced       |
    | Full-load BlockMmad (non-TLA)       | `block_mmad_pingpong_full_loadA.hpp`   | basis for the TLA rewrite            |
    | Multi-core scheduling in the full-load scenario             | example 25 `25_matmul_full_loadA`          | BlockScheduler reused directly |

- **Innovative features**: based on the investigation, identify the points that truly need innovation. The core innovation of example 44 is combining the two features of quantized matrix multiply and A-matrix full load, and reusing the BlockScheduler of example 25 so that the basic blocks handled by each core are distributed contiguously along N, maximizing the L1 reuse rate of the A matrix.
- **Applicable scenarios**: clarify the benefit scenarios and constraints of the new feature. A-matrix full load brings significant benefits when the N axis is large (the A matrix can be reused many times), but it requires the L1 space to be large enough to hold the data volume of `L1TileShape::M*K`; otherwise it cannot be used. This value is an element count, and the occupied bytes are `L1TileShape::M * K * sizeof(ElementA)`; the L1 capacity of Atlas A2 series products and Atlas A3 series products is 512 KB, and the fully loaded data must fit in the available L1 space. A large N axis means that multiple basic blocks can be split along N, that is, `problemShape.N > L1TileShape::N`; in this case the same A-matrix block can be reused by multiple N-direction basic blocks. When N does not exceed one basic block, the A matrix cannot be reused and performance may be lower than that of the ordinary template.

### Solution Design

Based on the requirements analysis and component investigation, design the overall implementation solution of the operator. The solution design of example 44 reflects the principle of "maximize reuse, minimize innovation":

**(1) Selection of template parameters**

Determine the core template parameters of the operator, including TileShape, data types, Dispatch policies, and so on. The key parameter choices of example 44 are as follows:

| Parameter           | Value            | Description                              |
| -------------- | --------------- | --------------------------------- |
| L1TileShape    | `128*256*512`   | L1-level tile size, M=128, N=256, K=512 |
| L0TileShape    | `128*256*128`   | L0-level compute tile size                  |
| ElementA/B     | int8_t          | quantized input matrices                      |
| ElementC       | int32_t         | accumulator type                        |
| ElementD       | bfloat16_t      | final output type                      |
| DispatchPolicy | `MmadFullLoadA` | pipeline policy for fully loading the A matrix               |

**(2) Component splitting and reuse**

Split the operator into component modules and clarify the reuse/reference/new relationship of each component:

| Component           | Role                                                 | Development method     | Reference source                                       |
| -------------- | ---------------------------------------------------- | ------------ | ---------------------------------------------- |
| BlockMmad      | A matrix fully loaded + B matrix pingpong transfer, performs block-level matrix multiply      | **TLA rewrite**  | `block_mmad_pingpong_full_loadA.hpp` (example 25) |
| BlockEpilogue  | dequantization post-processing (scale multiply + per-token multiply + type conversion) | **reused directly** | example 42 `42_quant_optimized_matmul_tla`         |
| BlockScheduler | multi-core task allocation, contiguous blocking in the full-load scenario                   | **reused directly** | example 25 `25_matmul_full_loadA`                  |
| Kernel         | assembles the above components, implements the AIC/AIV dual-core pipeline                    | **rewritten by reference** | the TLA Kernel structure of example 42                         |

**(3) Pipeline design**

Design the pipeline of data transfer and computation. The pipeline scheme of example 44 is shown in the following figure:

- At the start of computation, the complete data block of the A matrix (`L1TileShape::M*K`) is moved into L1 in one pass through MTE2
- Then the main loop is entered, transferring the K-dimension blocks of the B matrix into L1 with pingpong, while performing the L1→L0 data transfer and the Cube computation
- The A matrix resides in L1 and does not need to be transferred repeatedly, thus reducing MTE2 bandwidth pressure

<img src="https://raw.gitcode.com/user-images/assets/7631999/1de46727-7c46-411e-936c-7a437d951a3a/3e9c799e1de0405d89f07a6bfd7d7c54.png_tplv-a9rns2rl98-image-qvalue.png" alt="Example 44 pipeline design: the A matrix is moved into L1 in one pass and then stays resident; the B matrix is moved into L1 in pingpong blocks; data movement overlaps with Cube computation" width="60%">

### Documentation Design

The documentation structure of the example should be planned during the design stage. Every innovative example needs to include the following documents:

- **README.md**: a brief description of the operator, usage instructions, and a quick start guide
- **Design document (`${id}_${op_name}.md`)**: a detailed description of the prototype design, solution design, component implementation, and performance benefits
- **Code comments**: clear comments are needed for key interfaces and complex logic

## Development Stage

### Environment Preparation and Code Structure

Set up the development environment by referring to [Quick Start](./01_quick_start.md), then organize the code according to the following directory structure:

```text
examples/44_quant_matmul_full_loadA_tla/   # example directory
├── CMakeLists.txt                          # build configuration
├── README.md                               # usage instructions
├── 44_quant_matmul_full_loadA_tla.md       # design document
└── quant_matmul_full_loadA_tla.cpp         # Host-side assembly code

include/catlass/gemm/kernel/
└── quant_matmul_full_loadA_tla.hpp         # Kernel implementation

include/catlass/gemm/block/
├── block_mmad_pingpong_full_loadA_tla.hpp  # TLA-rewritten BlockMmad (based on block_mmad_pingpong_full_loadA.hpp)
└── block_swizzle.hpp                       # L1FullLoad Swizzle reused from example 25

include/catlass/epilogue/block/
└── block_epilogue_per_token_dequant_tla.hpp # BlockEpilogue reused from example 42
```

**Project registration (easy to miss):** Creating the example directory and its local `CMakeLists.txt` is not enough. The top-level [`examples/CMakeLists.txt`](../../../examples/CMakeLists.txt) decides which examples are built via `EXAMPLE_ATLASA2` / `EXAMPLE_ASCEND950`; if the example name is not appended to the matching list, `bash scripts/build.sh <example_name>` will not find the target. Register by product form, for example on Atlas A2 and Atlas A3:

```diff
set(EXAMPLE_ATLASA2
    00_basic_matmul
    # ...
+   44_quant_matmul_full_loadA_tla
)
```

For Ascend 950 examples, append to `EXAMPLE_ASCEND950` instead. For the local `CMakeLists.txt` pattern and the full registration steps, see [Host-side Example Assembly - Build and Run](./02_host_example_assembly.md#build-and-run).

### Host-Side Assembly

The Host-side code is responsible for device initialization, memory allocation, component assembly, and kernel invocation; for details, see [Host-layer Example Assembly](./02_host_example_assembly.md). During development, focus mainly on the following points:

**(1) Definition of data types and layouts**

Define the data types and memory layouts of all input and output tensors according to the operator requirements. Example 44 involves 5 tensors: the int8 A/B matrices, the float scale and perTokenScale, and the bf16 output D matrix.

**(2) Instantiation of component templates**

Assemble layer by layer in the order "DispatchPolicy → BlockMmad → BlockEpilogue → BlockScheduler → Kernel → Adapter":

1. First determine the DispatchPolicy (such as `MmadFullLoadA`), which controls the stage counts and pipeline behavior of the caches at each level
2. Then assemble BlockMmad based on the DispatchPolicy and TileShape, specifying the TileCopy component to handle data transfer
3. Next assemble BlockEpilogue, configuring the Tile components needed for dequantization (such as `TileRowBroadcastMul`)
4. Then select the appropriate BlockScheduler depending on whether the M dimension needs core splitting (use `GemmIdentityBlockSwizzleL1FullLoad` in the full-load scenario and `GemmIdentityBlockSwizzle` in the ordinary scenario)
5. Finally assemble BlockMmad, BlockEpilogue, and BlockScheduler into a Kernel, and complete the invocation through the DeviceGemm adapter

**(3) Selection of the core-splitting strategy**

Depending on the relationship between `problemShape.M` and `L1TileShape::M`, example 44 selects a different BlockScheduler on the Host side:

- When `M > L1TileShape::M`, the M direction needs core splitting, so the `GemmIdentityBlockSwizzleL1FullLoad` strategy is used to make the basic blocks handled by each core contiguous, improving inter-block reuse when the A matrix is fully loaded
- When `M <= L1TileShape::M`, the M direction does not need core splitting, and the ordinary `GemmIdentityBlockSwizzle` suffices

### Kernel-Layer Development

The Kernel layer is the execution entry of the operator, responsible for scheduler initialization, global tensor definition, and core loop control; for details, see [Kernel Code Development](./03_kernel_development.md). The key points of Kernel development for an innovative example are as follows:

**(1) Parameter structure design**

Define two levels of parameter structures, `Arguments` (user interface layer) and `Params` (kernel execution layer). The Arguments of example 44 contain problemShape (problem sizes M/N/K), aicCoreNum (the number of AIC cores participating in the computation, used by BlockScheduler to divide tasks), the input tensor group (A/B), and the output tensor group (scale/perTokenScale/D).

**(2) AIC core implementation**

The AIC core is responsible for the matrix multiply computation, and its core flow is:

- Initialize BlockScheduler and obtain the total loop count
- Define the global tensors pointing to GM
- Loop over each block: compute block coordinates → compute the GM offset → call BlockMmad to perform the matrix multiply → write the result to the Workspace

The Workspace is GM memory requested by the host through `GetWorkspaceSize` and allocated with `aclrtMalloc`. It is used to pass intermediate results between the AIC and AIV cores. Example 44 sets `workspaceStages = 2`, which allocates two basic-block-sized buffers per AIC so that AIC writes and AIV reads can overlap in a double-buffered pipeline.

**(3) AIV core implementation**

The AIV core is responsible for Epilogue post-processing, and its core flow is:

- Initialize the BlockEpilogue component
- Read the computation result of the AIC core from the Workspace
- Perform the dequantization computation (scale multiply + perTokenScale multiply + type conversion)
- Write the final result back to GM

**(4) AIC/AIV synchronization**

The CrossCoreFlag mechanism implements pipeline synchronization between the AIC and AIV cores: after finishing the computation of a block, the AIC core sets a flag; the AIV core waits for that flag before starting post-processing, and after finishing sets a flag to notify the AIC core that the Workspace stage can be reused.

### Block-Layer Component Development

The Block layer is the core of innovative example development. The following uses example 44 to explain the development points of the three types of components; for detailed development guides for each component, see the corresponding documents:

**(1) BlockMmad — matrix multiply computation component** (see [BlockMmad Code Development](./04_block_mmad_development.md) for details)

BlockMmad is responsible for managing the L1/L0 caches, performing data transfer, and executing Cube computation. The `BlockMmadTla` of example 44 (`block_mmad_pingpong_full_loadA_tla.hpp`) is a TLA rewrite of the existing full-load component `block_mmad_pingpong_full_loadA.hpp`:

- L1 cache allocation: the A matrix occupies the `L1TileShape::M*K` space (fully loaded), and the B matrix occupies the `L1TileShape::N*L1TileShape::K*2` space (double-buffer pingpong)
- The constructor completes the allocation of the L1/L0 buffers and the event initialization
- operator() implements: first load the complete A matrix into L1 in one pass → enter the K-dimension loop → pingpong-load the B matrix blocks → L1→L0 transfer → Cube computation → L0C write-back

The core difference from an ordinary pingpong BlockMmad is that the A matrix is transferred only once, and all subsequent K-dimension loops reuse the A matrix data in L1.

**(2) BlockEpilogue — post-processing component** (see [Epilogue Adaptation and Development](./07_epilogue_adaptation.md) for details)

BlockEpilogue is responsible for post-processing the matrix multiply result. The `BlockEpilogue` of example 44 directly reuses the per-token dequantization scheme of example 42:

- UB cache allocation: allocate double-buffer space for each of the C matrix, scale, perTokenScale, and the output D matrix
- Computation flow: load C and scale from GM into UB → execute tileRowBroadcastMul (per-channel multiply) → load perTokenScale → execute tileOneBlkColumnBroadcastMul (per-token multiply) → type conversion (int32→bf16) → write back to GM
- Overlap computation and transfer through UB double-buffer rotation

**(3) BlockScheduler — multi-core scheduling component** (see [BlockScheduler Code Development](./05_block_scheduler_development.md) for details)

BlockScheduler is responsible for distributing compute tasks to multiple AIC cores. Example 44 reuses `GemmIdentityBlockSwizzleL1FullLoad` of example 25; its core difference from an ordinary Swizzle is the block order:

- Ordinary Swizzle: blocks are distributed to each core alternately as `0-1-2-...-19-0-1-2-...`, so the basic blocks handled by each core are scattered
- L1FullLoad Swizzle: blocks are distributed contiguously as `0-0-...-0-1-1-...-1-2-2-...-19`, so the basic blocks handled by each core are contiguous along N, allowing the fully loaded A matrix to be fully reused by consecutive basic blocks

### Tile-Layer Component Development

Tile components are the lowest-level computation and data manipulation units in CATLASS and interact directly with the hardware; they mainly include `TileMmad` (matrix multiply computation) and `TileCopy` (data transfer between caches at each level). For a detailed development guide, see [Tile Component Code Development in Detail](./06_tile_development.md).

During Block-layer component development, you need to analyze whether the Tile components used inside BlockMmad and BlockEpilogue meet the requirements:

- **TileMmad**: responsible for the matrix multiply operation on the L0 cache. In most scenarios, the existing `TileMmad` templates (supporting various data type combinations such as int8/bf16/fp16) can be reused directly. TileMmad only needs to be extended when a new data type combination or a special computation pattern must be supported.
- **TileCopy**: responsible for data transfer between caches at each level, such as GM↔L1, L1↔L0, and L0→GM. The existing `TileCopy` templates cover common data transfer patterns (such as `CopyGmToL1`, `CopyL1ToL0A/B`, and `CopyL0CToGm`), and can usually be reused directly. New TileCopy components are only needed when a new data layout conversion or a special transfer strategy is required.

Taking example 44 as an example, the TileMmad and TileCopy used inside its BlockMmad and BlockEpilogue both directly reuse existing components, and no new Tile component needs to be developed. When designing your own example, you should first traverse the existing Tile components under the `include/catlass/gemm/tile/` directory and only decide whether to add new ones after confirming whether they meet the requirements.

### Compilation and Debugging

Before building, confirm the **project registration** step above is done (the example name is listed in `examples/CMakeLists.txt`); otherwise the build system will not pick up the target. Then from the repository root:

```bash
# Build the example; the executable is installed to output/bin
bash scripts/build.sh 44_quant_matmul_full_loadA_tla

# Change to the directory of the executable
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional, defaults to 0)
# The K-axis is limited by L1 capacity and must not exceed 2048
./44_quant_matmul_full_loadA_tla 512 4096 1024 0
```

On a successful build the log shows `[INFO] Target '44_quant_matmul_full_loadA_tla' built successfully`; on success the program prints `Compare success.`, indicating that the precision check passed.

During development, it is recommended to use the [performance tuning tools](./08_evaluation.md) for simulation verification, checking through the pipeline chart whether the overlap of data transfer and computation meets expectations.

## Testing Stage

### Precision Testing

- Use the benchmark functions in [`examples/common/golden.hpp`](../../../examples/common/golden.hpp), or implement a benchmark by referring to the function signatures there, and compare the error between the operator output and the benchmark result
- Execute at least 200 generalization precision test cases, covering different M/N/K combinations, including values below, equal to, and above the TileShape of each axis as well as non-aligned shapes
- Example 44 uses `golden::QuantMatmul` as the precision benchmark and compares the bf16 output against the criteria in [`compare_data.hpp`](../../../examples/common/golden/compare_data.hpp): the relative tolerance is $2^{-7}$ when the accumulation count is below 2048 and $2^{-6}$ otherwise, with each element judged by $|actual-expected| \le rtol \times \max(1/256, |expected|)$

### Performance Testing

- Compare performance with the benchmark operator to verify the benefits of the new feature. Example 44 uses [12_quant_matmul](../../../examples/12_quant_matmul/README.md) as the benchmark and improves performance by 5%~15% in scenarios with a large N axis
- Record the performance data, test environment, and benchmark information in the operator design document
- If the performance does not meet expectations, refer to [Performance Bottleneck Analysis and Optimization Techniques](./evaluation/bottleneck_analysis_and_optimization.md) for localization and optimization

## Integration Stage

### Code Specification Check

- Ensure the code conforms to the project code style (refer to the [.clang-format](../../../.clang-format) configuration)
- Check the naming conventions: the example directory uses the `${id}_${op_name}` format, and file names use the lowercase-with-underscores style
- Add the necessary copyright notice and License header

### Documentation Completion

- README.md contains a brief introduction to the operator, build and run steps, and parameter descriptions
- The design document contains the prototype design, solution design, component description, and performance data
- If necessary, supplement the usage instructions or API documentation of new components

### Submission Process

Refer to [Example Contribution Process](./09_example_contribution_guide.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md):

1. Create an Issue describing the design solution and innovative points of the new operator
2. Submit the code to a personal branch
3. Create a PR and fill in the template information in detail (including precision and performance test results)
4. Wait for code review and revise according to the feedback
5. Merge into the main branch after the review passes

## Summary

The core process of developing a CATLASS innovative example can be summarized in the following four steps:

1. **Investigate existing components**: before starting development, first traverse existing examples and the template components under `include/catlass/` to determine which can be reused directly, which need to be rewritten by reference, and which must be developed from scratch. This is the most critical step — thorough investigation minimizes duplicated development work.
2. **Identify the innovative points**: based on the investigation, determine the parts that truly need innovation. For example, the investigation conclusion of example 44 is: BlockEpilogue can directly reuse example 42, BlockMmad can be TLA-rewritten on top of the existing full-load component, and BlockScheduler can directly reuse example 25, so no module needs to be developed from scratch.
3. **Implement layer by layer**: complete the code of each layer top-down in the order Host→Kernel→Block→Tile. Prefer reusing and rewriting existing components, and develop new components only when necessary.
4. **Verify the benefits**: confirm that the new feature achieves the expected effect through precision and performance testing.

By combining quantized matrix multiply (reusing example 42) with A-matrix full load (referring to example 25), and completing all components through reuse or TLA rewrites, example 44 achieved a 5%~15% performance improvement in the quantized matrix multiply scenario with a large N axis, fully demonstrating the development principle of "maximize reuse, minimize innovation". When designing your own example, you should first complete a comprehensive component investigation and then determine the innovative points and development plan.
