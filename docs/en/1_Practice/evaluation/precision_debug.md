# Locating Precision Issues

## Before You Start

This document describes how to systematically locate the root cause of precision issues when precision comparison fails during CATLASS sample development. It first helps developers determine the problem type, then uses a decision tree and modular binary search to progressively narrow down the scope, and finally uses a symptom-cause lookup table and diagnostic patterns to locate and fix the specific cause.

Before reading this document, read [Precision Analysis Basics](./precision_analysis_basics.md) to understand CATLASS precision comparison methods and how to use the golden function.

## Classification of Precision Issues

Before locating the issue, determine which category the current precision issue falls into. Different categories require completely different troubleshooting paths.

### Complete Miscalculation

**Symptoms**: The NPU output differs significantly from the golden result. Almost all elements do not match, or the output shows obvious anomalous patterns (all zeros, all NaN, all Inf, random garbage, etc.).

**Common causes**:
- Computation starts before data movement completes (missing pipeline synchronization).
- Input data is not passed correctly (memory copy error, address offset error).
- Fundamental logic errors exist in the sample (incorrect formula, dimension swapped).
- Build cache has not been cleared; the actually running binary is an earlier version.

**Troubleshooting priority**: Clear the cache first, then investigate pipeline synchronization and data movement issues, and finally check logic errors.

### Precision Error

**Symptoms**: Most elements pass the comparison, only a few exceed the error tolerance, or the overall error is large but the numerical trend is correct (e.g., the output and the golden result are within the same order of magnitude).

**Common causes**:
- Accumulated rounding errors from low-precision data types (FP16/BF16)
- Intermediate computations not using higher-precision accumulation
- Numeric overflow (max. FP16 just reaching 65504)
- Catastrophic cancellation (Subtracting two nearly equal numbers causes loss of significance.)
- Precision characteristics of a specific API not matching expectations

**Troubleshooting priority**: Check the accumulator precision and overflow first, then investigate the behavior of specific APIs.

### Quick Determination

After running precision comparison, observe the number and distribution of error indices returned by `CompareData`:

| Symptom| Determination| Next Step|
| --- | --- | --- |
| Number of errors close to total element count, or output is all zeros/NaN/Inf| Complete miscalculation| Go to [Pre-checks](#pre-checks) → [Pipeline Synchronization Check](#missing-pipeline-synchronization)|
| Small proportion of errors relative to total element count (e.g., < 10%), with error values within a reasonable range| Precision error| Go to [Diagnostic Mode](#diagnostic-patterns)|
| Errors concentrated in specific locations (e.g., matrix edges, specific groups)| Boundary/Grouping issue| Go to [Modular Binary Search](#modular-binary-search)|

## Pre-checks

Before diving into investigation, complete the following pre-checks. These checks are low-cost but can eliminate many common issues.

### Validating Golden Code Correctness

**This is the first step in CATLASS precision debugging, and also the most easily overlooked step.** If the golden function itself is incorrect, all subsequent comparisons are meaningless.

Key check points:

1. **Is the correct golden function selected?** Confirm that the golden function used matches the sample's functionality. For example, matrix multiplication should use `ComputeMatmul` rather than `ComputeGemm` (the latter includes additional alpha/beta scaling).

2. **Is the data type for the golden function computation correct?** Floating-point golden functions must use higher precision for computation. Ensure that the `ElementGolden` template parameter is set to `float` rather than `half`.

   ```cpp
   // ✅ Correct: The golden function uses float for higher precision.
   std::vector<float> hostGolden(lenC);
   golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

   // ❌ Incorrect: The golden function uses half, introducing precision loss on the CPU.
   std::vector<fp16_t> hostGolden(lenC);
   golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
   ```

3. **Are the layout parameters correct?** Verify that `layoutA`, `layoutB`, and `layoutC` are consistent with the actual memory layout used by the sample. Swapping RowMajor and ColumnMajor is the most common error in the golden function.

4. **Does the golden function input data match the NPU input?** Verify that the `hostA`/`hostB` passed to the golden function is identical to the data copied to the device. If the host data is modified after copying and then passed to the golden function, the golden function will compute on different data than the NPU.

5. **Cross-validation with a simple case**: Manually compute the expected result for a simple case (e.g., M=N=K=2, with all data being 1.0) and compare with the golden function output to confirm that the golden function logic is correct.

### Clearing Build Cache

Uncleared build caches can cause modifications to not take effect, leading to repeated debugging of the same old code:

```bash
rm -rf build/
rm -rf output/
```

Alternatively, add the `--clean` option when building the sample.

After clearing, rebuild and run to confirm whether the issue still exists.

### Fixing a Minimum Reproducible Case

Reduce the issue to the minimum reproducible scale:

- Use the smallest M, N, K that reproduce the issue (e.g., M=N=K=16 or 32).
- Use a fixed random seed or fixed data instead of random data to ensure consistent results across runs.
- Simplify the layout combination (prefer RowMajor + RowMajor).

A minimal case reduces debugging data volume, shortens build-run cycles, and eliminates interference from multi-core interactions.

### Verifying That Modifications Have Taken Effect

Insert an explicit output (e.g., `std::cout << "check" << std::endl;`) in the code to confirm that the modified binary has been indeed executed.

## Modular Binary Search

### CATLASS Sample Classification: With or Without Tiling

First, a critical distinction: **Most CATLASS samples do not have an independent tiling step.** Only a few FlashAttention and dynamic matmul samples include an explicit tiling stage. Before starting binary search, confirm which category your sample falls into.

| Sample Type| Independent Tiling| Typical Sample| Call Chain|
| --- | --- | --- | --- |
| Basic/Quantized Matmul| ❌ No| `00_basic_matmul`, `01_batched_matmul`, `44_quant_matmul_full_loadA_tla`| Directly assemble Block components → Kernel → DeviceGemm|
| FlashAttention / MLA | ✅ Yes| `19_mla` (`mla_tiling.h/cpp`, `GetMLATilingParam()`), `23_flash_attention_infer` (`fai_tiling.cpp`, `GetFATilingParam()`), `40_flash_attention_infer_tla`| Tiling → Kernel launch|
| Dynamic Matmul| ✅ Yes| `102_dynamic_optimized_matmul` (`DoTiling` + `SelectKernel`), `103_dynamic_optimized_quant_matmul_per_token_basic`| DoTiling → SelectKernel → Launch |

> **Takeaways**
> - **For samples without tiling** (the majority), precision investigation goes directly to component binary search (Section 3.3). No need to consider tiling issues.
> - **For samples with tiling** (FA/MLA/dynamic Matmul), first determine whether the issue is related to tiling or the kernel, and then proceed to component binary search.

### CATLASS Template-based Layered Architecture

CATLASS adopts a template-based layered design. A complete sample consists of four layers: Device → Kernel → Block → Tile. Understanding this structure is the prerequisite for accurately locating faulty components.

Using `44_quant_matmul_full_loadA_tla` as an example, its complete component hierarchy is as follows:

```
Device layer: DeviceGemm<MatmulKernel>
    ↓ Assembles
Kernel layer: QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>
    ↓ Assembles
Block layer:
    ├── BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>
    │       ↓ Uses internally
    │   Tile layer:
    │       ├── TileCopy = PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>
    │       │              ← Data movement tile (moving the matrix A/B from GM to L1)
    │       └── (Implicit) MMAD computation tile          ← Matrix multiply-accumulate tile (responsible for cube computation from L1 to L0)
    │
    └── BlockEpilogue<EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD,
                       TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
                       EpilogueTileCopy, TileScheduler>
            ↓ Uses internally
        Tile layer:
            ├── TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>
            │              ← Row broadcast multiply tile (broadcasts scale along row direction and multiplies with result)
            ├── TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>
            │              ← Single-block broadcast tile
            ├── TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>
            │              ← Column broadcast multiply tile (broadcasts per-token scale along column direction and multiplies with result)
            ├── TileCopyDequantTla<ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale,
            │                       ElementPerTokenScale, LayoutTagPerTokenScale, ElementD, LayoutTagD>
            │              ← Dequantization copy tile (dequantizes accumulation result and converts it to output type)
            └── EpilogueHorizontalTileSwizzle  ← Tile scheduler (controls tile execution order and switching)
```

**Insight**: The root cause of precision issues can exist at any layer—pipeline synchronization and assembly logic at the kernel layer, computation logic and component use at the block layer, or specific implementations at the tile layer. The goal of binary search is to **progressively narrow the troubleshooting scope**, but note that **each layer itself may pose independent issues** (e.g., missing pipeline synchronization at the kernel layer, incorrect component use logic at the block layer). Therefore, during binary search, do not assume that the issue must be in the next layer. First verify whether the current layer itself has issues.

### Binary Search Strategy

#### Step 1 (only for FA/MLA/dynamic samples): Distinguish tiling vs. kernel issues.

Perform this step only if the sample includes an explicit tiling stage (see classification in Section 3.1):

- Replace the current tiling result with known correct tiling parameters → Does precision recover? Check the tiling (TileShape partitioning, address offset calculation, etc.).
- Precision still not as expected → Issue lies in kernel implementation. Proceed to Step 2.

For samples without tiling, go directly to Step 2.

#### Step 2: Perform block-level binary search (BlockMmad vs. BlockEpilogue).

This is the core step for CATLASS precision issue locating. Split the kernel into two block components and validate them separately.

**Method A: Replace BlockEpilogue with an identity epilogue.**

Use the simplest identity epilogue (no post-processing; directly writes back the accumulation result from BlockMmad). If precision recovers, the issue is in BlockEpilogue; otherwise, the issue is in BlockMmad.

```cpp
// Replace BlockEpilogue with identity epilogue.
// Original code (using sample 44):
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD,
    TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
    EpilogueTileCopy, TileScheduler>;
```

```cpp
// Use identity epilogue instead (data type conversion and write-back only, no quantization or broadcast).
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    SimpleDispatchPolicy, ElementC, void, void, ElementD,
    IdentityTile, IdentityTile, IdentityTile, SimpleTileCopy, SimpleScheduler>;
```

**Method B: Replace BlockMmad with a known correct implementation.**

Replace the current BlockMmad with a proven correct BlockMmad (e.g., `Gemm::Block::BlockMmad` extracted from `00_basic_matmul`). If precision recovers, the issue is in BlockMmad; otherwise, the issue is in BlockEpilogue.

#### Step 3: Perform tile-level binary search (inside the faulty block).

After locating the specific block, further perform binary search on its internal tile components:

**If the issue is in BlockMmad:**

```
BlockMmad
    ├── TileCopy (data movement tile)
    │   └── Replace it with simple DataCopy to validate data movement logic.
    │       - Check address offset, data size, and layout.
    │       - Check L1→L0 movement pipeline synchronization (SetFlag/WaitFlag).
    │
    └── MMAD computation tile
        └── Check the accumulator precision (whether FP32 accumulation is used).
            - Check mask parameters (whether tail block processing is missing).
            - Check Cube instruction parameters such as repeatTime and stride.
            - Check L0 C tiling strategy (l0CStages).
```

**If the issue is in BlockEpilogue** (using the quantization epilogue in sample 44 as an example):

```
BlockEpilogue
    ├── TileRowBroadcastMulTla → Replace it with element-wise Mul to validate the broadcast logic.
    │   - Check whether the broadcast dimension is correct (broadcasting scale along row direction).
    │
    ├── TileBroadcastOneBlkTla → Check whether the broadcast data block is correct.
    │
    ├── TileOneBlkColumnBroadcastMulTla → Replace it with element-wise Mul to validate column broadcast.
    │   - Check whether the broadcast dimension is correct (broadcasting per-token scale along column direction).
    │
    ├── TileCopyDequantTla → Replace it with common DataCopyPad to validate the dequantization logic.
    │   - Check the dequantization formula: output = accum * scale * per_token_scale
    │   - Check the RoundMode of the Cast operation (FP32 accumulation → BF16/FP16 output).
    │
    └── EpilogueHorizontalTileSwizzle → Check the tile scheduling order.
        - Check whether the tile execution order causes data overwriting or missing.
```

### Component Replacement Example

Using `44_quant_matmul_full_loadA_tla` as an example, this section shows how to replace specific components to narrow the troubleshooting scope:

| Layer| Component| Replacement| Validation|
| --- | --- | --- | --- |
| Block | BlockEpilogue | Replace with identity epilogue (no quantization, no broadcast, only type conversion + write-back)| Determine whether the issue is in epilogue or Mmad.|
| Block | BlockMmad | Replace with `Gemm::Block::BlockMmad` (non-TLA version, extracted from `00_basic_matmul`).| Verify basic Matmul logic.|
| Tile (inside BlockMmad)| TileCopy (PackedTileCopyTla)| Replaced with `Gemm::Tile::SimpleTileCopy`.| Verify the data transfer logic (address, layout, and synchronization).|
| Tile (inside BlockEpilogue)| TileCopyDequantTla | Replace with `DataCopyPad` (no dequantization).| Verify whether dequantization formula is the root cause.|
| Tile (inside BlockEpilogue)| TileRowBroadcastMulTla | Replace with element-wise Mul (no broadcast).| Verify whether the broadcast dimensions are correct.|

For dynamic samples such as `102_dynamic_optimized_matmul`, the module breakdown is as follows:

| Module| File/Function| Replaceability| Validation|
| --- | --- | --- | --- |
| Tiling| `include/do_tiling_b16.h` | Replaceable with manually computed tiling parameters| Compare with TilingParams fields.|
| Kernel selection| `include/select_kernel_b16.h` | Replaceable with a fixed kernel| Force specify TilingKey.|
| Wrapper launch| `impl/wrapper/*.cpp` (auto generated)| Replaceable with direct kernel template call| Bypass launch_map and instantiate directly.|
| Kernel implementation| `impl/kernel/*.h` | Replaceable with a simple kernel| Replace with the kernel of basic_matmul.|

### Binary Search Process

```
Precision comparison fails
    │
    ├─ Use a simple test case (M=N=K=16) to validate the golden function → If the function is incorrect, fix the function.
    │
    ├─ [For FA/MLA/dynamic samples only] Replace with known correct tiling parameters → Does precision recover?
    │   └─ Yes → Check the tiling (TileShape, address offset, etc.).
    │
    ├─ Modular binary search (when quick methods are ineffective)
    │   ├─ [For samples with tiling] Binary search on tiling vs. kernel
    │   ├─ Block-level binary search: BlockMmad vs. BlockEpilogue
    │   ├─ Tile-level binary search: Locate the specific tile inside the faulty block.
    │   └─ Perform binary search within the computation logic.
    │
    └─ Comparison (fallback method)
        ├─ Find the reference code that works properly (e.g., basic_matmul).
        └─ Look for differences module by module and line by line.
```

> **Keep in mind**: Do not blindly try-and-error. Before each modification, clearly state your assumption (e.g., "I think the problem lies in the XXX module"). After the modification, verify whether the assumption holds true. If consecutive modifications don't yield the expected result, you may troubleshoot the wrong way. Return to the decision tree and reassess.

## Diagnostic Patterns

The following diagnostic patterns cover the most common precision issues in CATLASS development. Each pattern provides a troubleshooting path from symptom to root cause.

### Pass for FP32 But Failure for FP16/BF16

**Symptom**: For the same sample, the precision comparison passes for the FP32 data type but fails for the FP16 or BF16.

**Troubleshooting**

```
Pass for FP32 but fail for FP16/BF16
    │
    ├─ Check the accumulator precision.
    │   └─ Does BlockMmad use FP32 accumulation?
    │       ├─ Yes → The accumulator is fine. Continue the troubleshooting.
    │       └─ No → Change to FP32 accumulation, the basic requirement for FP16/BF16 precision.
    │
    ├─ Check for numeric overflow.
    │   └─ FP16 maximum ≈ 65504, BF16 maximum ≈ 3.39e38
    │       Do intermediate results exceed the range?
    │       ├─ Yes → Scale down the input or use an intermediate type with higher precision.
    │       └─ No → Continue troubleshooting.
    │
    ├─ Check the Cast operation in the epilogue.
    │   └─ Conversion from FP32 accumulation result to FP16/BF16 output
    │       Is the correct RoundMode used?
    │       ├─ The default RoundMode may cause precision loss.
    │       └─ Try different RoundModes (e.g., RoundNearestEven vs. RoundTowardZero).
    │
    └─ Check for catastrophic cancellation.
        └─ Subtracting two nearly equal large numbers may cause severe loss of significance.
            FP16 has only 10 bits of mantissa, and BF16 has only 7 bits of mantissa.
            This issue is less noticeable with FP32 but is amplified under low precision.
```

### Failure for Specific Shapes or Parameter Ranges

**Symptom**: Precision issues only occur for specific shapes (e.g., non-aligned M/N/K dimensions, small shapes, large shapes) or specific parameter combinations.

**Troubleshooting**

```
Failure for specific shapes/parameters
    │
    ├─ Failure for small shapes (M, N, K < the corresponding dimension of TileShape)
    │ └─ Possible cause: DataCopy alignment issue
    │       ├─ DataCopy has a minimum movement granularity, which may not be met by small shapes.
    │       └─ Solution: Use DataCopyPad instead of DataCopy, or add the tail block processing logic.
    │
    ├─ Failure for non-aligned shapes (M, N, K not multiples of TileShape)
    │   └─ Possible cause: tail block processing logic error
    │       ├─ Are mask parameters correctly passed?
    │       ├─ Is the address offset calculation for tail blocks correct?
    │       └─ Is the data initialization (zeroing) for tail blocks correct?
    │
    ├─ Large shape failure
    │   └─ Possible cause: multi-core synchronization issue or out-of-bounds memory access
    │       ├─ Is the scheduling logic of BlockScheduler correct?
    │       ├─ Is the workspace size sufficient?
    │       └─ Is there a bank or address conflict?
    │
    └─ Failure of specific parameter combinations (e.g., specific batch size, specific number of heads)
        └─ Possible cause: parameter-dependent branch logic error
            ├─ Check the if/else branches related to that parameter.
            └─ Check whether the template specialization matches correctly.
```

### Obvious Abnormal Patterns in Output

**Symptom**: The output shows recognizable anomalous patterns rather than random errors.

| Pattern| Possible Cause| Troubleshooting|
| --- | --- | --- |
| All-0 output| Accumulator not initialized, data not moved, or kernel not executed| Check GlobalTensor.SetValue and the SetFlag/WaitFlag of DataCopy.|
| All-NaN output| Division by zero, sqrt of a negative number, or other invalid floating-point operations| Check the division, square root, and other operations in the epilogue.|
| All-Inf output| Numeric overflow| Check the range of intermediate computation results.|
| Random garbage output| Uninitialized memory or address offset error| Check workspace initialization and address calculation.|
| Some output regions correct, some incorrect| Block/Tile scheduling issue| Check the BlockScheduler and TileScheduler logic.|
| Output different from the golden result by a fixed multiple| Scale/Bias processing missing| Check whether the scale multiplication in the epilogue is missing.|
| Output matrix transposed| Layout parameters swapped| Check the RowMajor/ColumnMajor settings.|

## Common Pitfalls

The following are precision pitfalls that occur repeatedly in CATLASS development, listed in order of frequency.

### Missing Pipeline Synchronization

**Symptom**: The output is all zeros or partially zeros, or data appears mixed between old and new.

**Cause**: CATLASS uses pipelines to parallelize data movement and computation, using `SetFlag`/`WaitFlag` to control synchronization between pipelines. For example, after DataCopy moves data from GM to L1, it notifies the compute unit via `SetFlag<MTE2_MTE1>` that data is ready. The compute unit waits for the movement to complete via `WaitFlag<MTE2_MTE1>` before reading L1 data. If `SetFlag` or `WaitFlag` is missing, the compute unit may start reading before data movement completes, resulting in uninitialized or stale data.

**Troubleshooting**:
- Check whether the corresponding `SetFlag` operation (such as `SetFlag<MTE2_MTE1>`) exists after DataCopy completes.
- Check whether the corresponding `WaitFlag` operation (such as `WaitFlag<MTE2_MTE1>`) exists before MMAD computation.
- Check the L0 pipeline: Check whether `SetFlag<MTE1_M>` is set after DataCopy moves data from L1 to L0, and whether `WaitFlag<MTE1_M>` has been set before MMAD reads L0.
- Check whether the number of stages in each pipeline is properly configured (whether the event ID matches the number of stages).
- In cross-core synchronization scenarios, check whether `CrossCoreSetFlag` and `CrossCoreWaitFlag` appear in pairs.

### DataCopy Non-Alignment

**Symptoms**: Precision comparison fails for small shapes (M, N, K smaller than TileShape) but works for large shapes.

**Cause**: DataCopy has a minimum movement granularity requirement (typically 16B or 32B aligned). When the data size does not meet the requirement, DataCopy may move extra data (reading out-of-bounds data) or miss part of the data.

**Troubleshooting**:
- Check whether DataCopyPad (with padding) is used for non-aligned scenarios.
- Check whether masks are correctly used in tail block processing to limit the valid data range.

### Numeric Overflow

**Symptom**: Inf or excessively large values appear in the output.

**Cause:**
- FP16 maximum is approximately 65504, and BF16 maximum is approximately 3.39e38.
- The accumulated sum in matrix multiplication increases as the K dimension grows and can easily exceed the FP16 range.
- Intermediate computations (e.g., squaring, exponentiation) overflow even more easily.

**Troubleshooting**:
- Estimate the maximum possible values of intermediate results.
- Scale down the input data.
- Use FP32 as the intermediate accumulation type.

### Insufficient Accumulator Precision

**Symptoms**: Precision errors for FP16/BF16 are larger than expected, especially in large-K scenarios.

**Cause**: If BlockMmad uses FP16 or BF16 as the accumulator type (instead of FP32), each accumulation introduces rounding error. The larger K, the more severe the error accumulation.

**Troubleshooting**:
- Ensure that the `ElementC` template parameter of BlockMmad is `float` (FP32 accumulation).
- If FP32 accumulation is already in use but the issue persists, check the L0 C tiling strategy (l0CStages).

### Incorrect Quantization/Dequantization Formula in Epilogue

**Symptoms**: The output differs from the golden result by a fixed scaling factor, or error distribution shows systematic bias.

**Cause**: The epilogue in quantized Matmul includes operations such as scale multiplication and dequantization. A formula error (e.g., missing a scale factor, incorrect multiplication order) leads to systematic bias.

**Troubleshooting**:
- Manually derive the complete computation formula for the epilogue.
- Compare with the golden computation formula item by item.
- Validate the formula using simple data (e.g., all inputs = 1.0, all scales = 1.0).

### Layout Parameter Mismatch

**Symptoms**: The output matrix appears transposed, or errors are concentrated on specific dimensions.

**Cause**: RowMajor and ColumnMajor determine the data format in memory. If the assumed layout in the sample is inconsistent with the actual data layout, incorrect data will be read or written.

**Troubleshooting**:
- Ensure that LayoutA, LayoutB, and LayoutC are consistent in the golden function and the sample.
- Check the LayoutTag template parameters of DataCopy.

### Build Cache Not Cleared

**Symptoms**: No change in the issue after code modifications, as if modifications did not take effect.

**Cause**: The ATC compiler caches built kernels. If the cache is not cleared, even after modifying the source code, the old version still runs.

**Troubleshooting**: bash
rm -rf build/
rm -rf $HOME/atc_data/kernel_cache/
Alternatively, add the `--clean` option during build.

## Debugging Strategy Hierarchy

When facing a precision issue, use debugging strategies in the following hierarchy. Start with the lowest-cost quick methods, then progressively move to more systematic methods.

### Level 1: Quick Methods (Try First)

| Method| Scenario| Operation|
| --- | --- | --- |
| Clear build cache| Issue persists after any modification| `rm -rf build/ $HOME/atc_data/kernel_cache/` |
| Fix random seed| Issue is not consistently reproducible| Use fixed seed or fixed data instead of random data|
| Reduce issue scale| Large shape scenarios| Use the smallest reproducible shape (e.g., M=N=K=16)|
| Simplify data type| Fail for FP16/BF16| Test whether it passes for FP32 first.|
| Simplify layout| Complex layout combinations| Use RowMajor consistently|
| Check golden function| Uncertain which side the issue is on| Manually verify the golden function output using a simple case.|

### Level 2: Modular Binary Search (Core Method)

When quick methods cannot locate the issue, use modular binary search. This is the core strategy for CATLASS precision debugging.

**Binary Search Hierarchy (from Coarse to Fine):**

```
Level 1: [For samples with tiling] Tiling vs. kernel implementation
    └─ Replace with known correct tiling parameters to determine which side the issue lies on.
    └─ Note: The kernel layer itself may have issues in aspects such as pipeline synchronization and assembly logic. The tiling layer does not necessarily cause the issue.

Level 2: Kernel layer vs. Block layer
    └─ In addition to assembling block components, the kernel layer also has its own pipeline synchronization logic (SetFlag/WaitFlag),
       workspace management, and multi-core scheduling processes, all of which may introduce precision issues.
    └─ First, check whether the pipeline synchronization and assembly logic at the kernel layer are correct, and then proceed to block-level binary search.

Level 3: BlockMmad vs. BlockEpilogue
    └─ Use identity epilogue instead, or replace BlockMmad with a known correct version.
    └─ Note: Besides tile components, the block layer itself also has component use logic (e.g., tile assembly order,
       template parameter passing, DispatchPolicy selection, etc.) that may also introduce issues.

Level 4: Tile component binary search
    └─ Replace tile components one by one within the faulty block.

Level 5: Binary search inside computation logic
    └─ Perform binary search on the specific computation steps within the faulty tile.
```

**Binary Search Principles**:
- Change only one variable at a time to keep a clear track of the change outcomes.
- Preferentially replace with the simplest implementation (such as identity epilogue or simple DataCopy).
- Clearly state your assumption before each modification, and verify the assumption after the modification.

### Level 3: Comparison (Fallback)

When binary search cannot locate the issue, use the comparison method as a fallback.

**Procedure**:
1. Find a reference sample with similar functionality and normal precision (e.g., `00_basic_matmul`).
2. Look for the differences module by module from the top layer to the low layer:
   - Device layer: template parameters of DeviceGemm
   - Kernel layer: kernel assembly approach
   - Block layer: template parameters of BlockMmad and BlockEpilogue
   - Tile layer: implementation details of each tile component
3. Gradually align the current code with the reference code, making one change at a time and verifying the precision.
4. When precision recovers, the last modified item is the root cause.

**Although time-consuming, comparison is often the most reliable method when facing complex or subtle precision issues.**

### Strategy Selection Decision-Making Tree

```
Precision comparison fails
    │
    ├─ Can quick methods locate the issue?
    │   ├─ Yes → Fixed, done.
    │   └─ No → Proceed to modular binary search.
    │
    ├─ Modular binary search
    │   ├─ [With tiling] Tiling vs. Kernel → Locate the side.
    │   ├─ BlockMmad vs. BlockEpilogue → Locate the specific block.
    │   ├─ Tile component binary search → Locate the specific tile.
    │   └─ Computation logic binary search → Locate the specific code line.
    │
    └─ Unable to locate using binary search?
        └─ Use comparison to compare each module with the reference code that works properly.
```

CATLASS's modular architecture provides a natural advantage for precision debugging: every module can be independently replaced and validated. By fully utilizing this feature, combined with the golden function and decision-making tree, you can locate the causes of the majority of precision issues.

## Precision Tolerance Reference

The default precision tolerances for different data types in CATLASS are as follows. If the sample development plan has explicit precision requirements, those take precedence.

| Data Type| rtol | atol | Description|
| --- | --- | --- | --- |
| FP32 | 1e-5 | 1e-6 | High precision, the strictest tolerance|
| FP16 | 1e-3 | 1e-4 | Medium precision|
| BF16 | 1e-2 | 1e-3 | Low precision (mantissa: 7 bits only)|
| INT | - | 0 | Requires bitwise identical results|

The `CompareData` function of CATLASS dynamically adjusts rtol based on `computeNum` (usually the K dimension size).

| Computation Count| FP16/FP32 rtol | BF16 rtol |
| --- | --- | --- |
| < 2,048| 1/256 | 1/128 |
| ≥ 2,048| 1/128 | 1/64 |

## Debugging Checklist

When debugging precision issues each time, check the following items one by one:

**Before debugging**:
- [ ] Read [Precision Analysis Basics](./precision_analysis_basics.md) to understand precision comparison methods.
- [ ] Fix a minimal reproducible case (minimum shape, fixed data).
- [ ] Clear build cache (`rm -rf build/ $HOME/atc_data/kernel_cache/`).
- [ ] Confirm that the modified binary has been executed.

**Issue classification**:
- [ ] Determine whether it's a complete miscalculation or a precision error.
- [ ] Observe the distribution pattern of error elements.

**Golden function validation**:
- [ ] The correct golden function is selected (ComputeMatmul vs. ComputeGemm, etc.).
- [ ] The golden function uses higher-precision computation (ElementGolden = float).
- [ ] The layout parameters are consistent with those in the sample.
- [ ] The input data passed to the golden function is identical to the data copied to the NPU.

**Common pitfalls**:
- [ ] Pipeline synchronization issues (EnQue/DeQue after DataCopy?)
- [ ] DataCopy alignment issues (DataCopyPad used for small shapes?)
- [ ] GlobalTensor.SetValue issues
- [ ] Numerical overflow (FP16 max ≈ 65504)

**Modular binary search**:
- [ ] Try replacing tiling parameters (for samples with tiling).
- [ ] Try block-level binary search (BlockMmad vs. BlockEpilogue).
- [ ] Try tile-level binary search (locate the specific tile inside the faulty block).
- [ ] Try replacing with a simple kernel.

## Summary

Locating a precision issue in CATLASS goes in three steps: **classify, binary search, and diagnose**.

| Stage| Goal| Action|
| --- | --- | --- |
| Classify| Determine the nature of the issue| Distinguish complete miscalculation vs precision error; observe error distribution|
| Binary search| Narrow the scope| Validate golden function → [For samples with tiling] Binary search on tiling/kernel → Block-level binary search (BlockMmad vs BlockEpilogue) → Tile-level binary search (locating specific tile components)|
| Diagnose| Locate the specific cause| Find the root cause and fix based on the diagnostic patterns and common pitfalls.|

CATLASS's modular architecture provides a natural advantage for precision debugging: every module can be independently replaced and validated. By fully utilizing this feature, combined with the golden function and decision-making tree, you can locate the causes of the majority of precision issues.
