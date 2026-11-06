# Introduction to the CATLASS Single_core_splitK_Matmul Sample

## Sample Implementation

The CATLASS [34_single_core_splitk_matmul sample](../34_single_core_splitk_matmul/README.md) is an Ascend-native Matmul operator implemented using the CATLASS Gemm API. It is optimized for large-scale matrix computation scenarios. The core operator components include the following:

- **Example assembly**: [single_core_splitk.cpp](../34_single_core_splitk_matmul/single_core_splitk.cpp)
- **Kernel implementation**:
  - Main kernel file: [single_core_slicek_matmul.hpp](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
  - Reusable padding component: [padding_matmul.hpp](../../include/catlass/gemm/kernel/padding_matmul.hpp)

- **Block component**: [block_mmad_single_core_splitk.hpp](../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)

## Example Assembly

```mermaid

graph LR
    S [Construct input] --> GroupT [SingleSplitK operator]

    subgraph GroupT [SingleSplitK operator]
        direction LR
        A [Assemble `Padding` object]
        A --> B [Assemble `blockMmad`]
        B --> C [Assemble kernel]
    end

    GroupT --> X [Execute operation]
    X --> D [Precision verification and resource release]
```

Consistent with the development paradigm of standard template libraries, the execution flow of this sample is illustrated in the diagram above. A brief description is as follows:

<details>
<summary><strong>Construct Input</strong></summary>

_Generate the left and right matrices for computation._

- Calculate the dimension properties of each [input matrix](../34_single_core_splitk_matmul/single_core_splitk.cpp#L71).
- Generate data for each [input matrix on the host side](../34_single_core_splitk_matmul/single_core_splitk.cpp#L101).
- Construct the [input on the device](../34_single_core_splitk_matmul/single_core_splitk.cpp#L107).

</details>

<details>
<summary><strong>Assemble <code>Padding</code> Object</strong></summary>

_Assemble Padding objects to facilitate data movement alignment._

- Define different [`PaddingTag` configurations](../34_single_core_splitk_matmul/single_core_splitk.cpp#L90).
- Assemble the [Padding objects](../34_single_core_splitk_matmul/single_core_splitk.cpp#L95) for Matrix A/B via `PaddingBuilder`.
- Instantiate the predefined [PaddingC object](../34_single_core_splitk_matmul/single_core_splitk.cpp#L97).
- Determine whether to [enable the PaddingC object](../34_single_core_splitk_matmul/single_core_splitk.cpp#L98).

</details>

<details>
<summary><strong>Assemble <code>blockMmad</code></strong></summary>

_Assemble related Padding objects to facilitate data movement alignment._

- Define the [layout characteristics](../34_single_core_splitk_matmul/single_core_splitk.cpp#L78) for each input.
- Declare the corresponding [matrix data types](../34_single_core_splitk_matmul/single_core_splitk.cpp#L124).
- Set the [Dispatch policy](../34_single_core_splitk_matmul/single_core_splitk.cpp#L139) used to select the `BlockMmad` component.
- Set the L1 and L0 [tile sizes](../34_single_core_splitk_matmul/single_core_splitk.cpp#L136) to optimize data transfer from GM to L1.
- Use the optimized TileCopy component.
- [Assemble BlockMMAD](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L162) using the above template input parameters.

</details>

<details>
<summary><strong>Assemble and Execute <code>Kernel</code></strong></summary>

_Assemble the Kernel and instantiate the object to complete operator computation._

- Use the new S-shaped [swizzle policy](../34_single_core_splitk_matmul/single_core_splitk.cpp#L140).
- [Assemble the kernel](../34_single_core_splitk_matmul/single_core_splitk.cpp#L143) using the preceding template.
- [Pass the kernel to the adapter](../34_single_core_splitk_matmul/single_core_splitk.cpp#L160) and [instantiate it](../34_single_core_splitk_matmul/single_core_splitk.cpp#L148).
- Construct the [input arguments](../34_single_core_splitk_matmul/single_core_splitk.cpp#L163).
- [Pass arguments](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L202) to the kernel side.
- Call the kernel layer to [calculate workspace requirements](../34_single_core_splitk_matmul/single_core_splitk.cpp#L166).
- [Allocate workspace on the device side](../34_single_core_splitk_matmul/single_core_splitk.cpp#L166) (if necessary).
- [Initialize the operator with the adapter](../34_single_core_splitk_matmul/single_core_splitk.cpp#L166).
- [Execute the operator](../34_single_core_splitk_matmul/single_core_splitk.cpp#L166).

</details>

<details>
<summary><strong>Precision Verification and Space Release</strong></summary>

_Validate the final execution results and reclaim resources._

- Copy the operator output results [back to the host side](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L213).
- [Compute the golden benchmark](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L216).
- Perform [precision verification](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L221).
- [Free input/output buffers and workspace memory](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L228).

</details>

## Kernel Implementation

This section describes the structures and key functions at the kernel level, the simplified computing process for the AIC/AIV components, and the optimization strategies employed.

### Main Structures and Functions at the Kernel Level

The following structures and key functions are implemented in the [kernel layer](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp):

- [struct Params](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L86): parameters required for operator execution at runtime
- [struct Arguments](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L126): encapsulates the parameters passed from the host side.
- [static size_t GetWorkspaceSize](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L138): pre-calculates the space required for alignment.
- [static Params ToUnderlyingArguments](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L163): parses the input parameters on the host side into the `Params` structure on the operator side.
- [void operator()`<AscendC::AIV>`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L205): execution code for the AIV (Vector) part.
- [void operator()`<AscendC::AIC>`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L255): execution code for the AIC (Cube) part.

## AIV/AIC Computation Process

The following describes the specific execution steps performed by the AIV and AIC components at the kernel layer.

<details>
<summary><strong>Operations Performed by AIV</strong></summary>

- If alignment for Matrix A or B is enabled:
  - [Initialize GlobalTensor](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L208): `gmA`, `gmWA` (or `gmB`, `gmWB`)
  - [Instantiate the PaddingA](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L213) object to allocate Unified Buffer (UB) resources so that data can be managed before and after alignment.
  - Invoke the execution `operator()` of the PaddingA object to [run the data alignment](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L214) operation.
  - Perform [inter-core synchronization](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L229) and set the flag bit (`CrossCoreSetFlag`) to notify the AIC that alignment is complete.

- Result matrix write-out pipeline:
  - [Wait for the AIC](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L233) to complete computation (`CrossCoreWaitFlag`).
  - [Initialize GlobalTensor](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L235) destinations: `gmC` (target destination address) and `gmWC` (temporary workspace accumulation address).
  - Copy data to the target address while performing [data casting and layout precision mapping](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp) (`RemovePaddingNDAndCastC`).

</details>

<details>
<summary><strong>Operation Performed by AIC</strong></summary>

- If the left or right matrix requires alignment:
  - Perform inter-core synchronization and [wait for the AIV completion flag](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L258) (`CrossCoreWaitFlag`).
  - [Initialize GlobalTensor](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp): `gmWA`, `gmWB`.

- [Initialize `GlobalTensor`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp): `gmA`, `gmB`, `gmC`
- Initialize the [`BlockScheduler`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L261) and [`BlockMmad`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L302) objects.
- Fetch the current AIC identifier `coreIdx`, total AIC count `coreNum` (resolved within the [swizzle policy](../../include/catlass/gemm/block/block_swizzle.hpp)), and the required `coreLoops`.
- Enter the main loop (loop times `coreLoops`).
  - Calculate the current matrix read offsets [`gmOffsetA`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L333) and [`gmOffsetB`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L334), alongside the next data block offsets [`gmOffsetNextA`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L339) and [`gmOffsetNextB`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L340) (if double-buffering is enabled).
  > Note: Under the current optimized algorithm, the left matrix is overloaded, meaning `gmOffsetNextA` is not actively enabled.
  - Calculate [`needLoadNextA`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L325) and [`needLoadNextB`](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L326) to identify whether to preload.
  - Invoke `blockMmad` to execute an [AIC computation block](../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L343), resolving the corresponding fractal matrix operations on L1A and L1B caches.
  > Note: `blockMmad` dynamically evaluates the K-axis tiling state to decide whether to enable atomic additions on GM.
  - Set the synchronization flag bit to notify the AIV that computation is complete.
  - Disable atomic addition states.

</details>

### Single-Core Split-K Algorithm

<div style="display: flex; justify-content: center;">
    <img src="https://raw.gitcode.com/user-images/assets/7694484/6322a9e2-00e0-449b-8c35-f99fe5883bae/tmp1.jpg" width="85%" height="auto">
</div>

As illustrated above, compared to the classical matrix multiplication process, the single-core split-K `Matmul` template optimizes execution by reusing the left matrix. For a given AI Core, the fractal matrix fetched into the L1A buffer remains **resident**, significantly reducing the total volume of Global Memory read transactions.

To illustrate, the memory access pattern of a classical execution timeline involves a sequential staging loop: 1. Stage $A_0$ from GM to L1A, and $B_0$ from GM to L1B. 2. Stage $A_1$ from GM to L1A, and $B_1$ from GM to L1B. 3. ... 4. Stage $A_{n-1}$ from GM to L1A, and $B_{n-1}$ from GM to L1B.

Under this single-core split-K strategy, the workflow in an individual AI Core proceeds as follows:

<div style="display: flex; justify-content: center;">
<img src="https://raw.gitcode.com/user-images/assets/7694484/5f0dcf49-6c76-47ea-ac2f-70a88d2ca8bc/singleCoreSplitK_Arrange.png" width="85%" height="auto">
</div>

Obviously, this method can effectively reduce the transfer load of MTE2, but it will increase the load of moving the L0C computation results back to the GM. Therefore, atomic addition needs to be enabled on the GM.

```cpp
// Atomic set once in mmad level
if (atomicAdd) {
    AscendC::SetAtomicAdd<ElementAccumulator>();
} else {
    AscendC::SetAtomicNone();
}
```

- Increasing L1 utilization space
  Based on simple modeling (see the supplement below), the amount of data transferred from L0C back to the GM is proportional to $MNK/k_{\text{L1}}$, where $M$, $N$, and $K$ are the sizes of the input matrices, and $k_{\text{L1}}$ is the tile size on the K-axis in L1. Under the condition that the physical sizes of L1A and L1B are limited, a larger $k_{\text{L1}}$ can reduce the number of writes. It is recommended to configure `L1TileShape` according to the table below.

<details>

<summary><strong>Theoretical Modeling of Data Transfer</strong></summary>

First, let's analyze the memory access volume of the [basic Matmul operator](../00_basic_matmul/README.md). Assume that the matrix multiplication size is $(M, N, K)$, and the corresponding tile size on L1 is $m$, $n$, and $k$. Assuming they are perfectly aligned, the data volume transferred from GM to L1 for a single fractal block is $K(mk+kn)/k$. Combined with the total number of fractal blocks $MN/mn$, the total read access volume is $MNK(1/m+1/n)$. On the other hand, since accumulation is performed directly on L0C, the write-out data volume is simply $MN$.
Now consider the memory access pattern after applying this optimization strategy. Assuming the left matrix is reused, the total data read volume for a single core is $mK + KN$, making the total read access volume $(mK + KN)M/m$, which simplifies to $MNK(1/m+1/N)$. This is smaller than that of the basic Matmul. However, since the computation results cannot be fully accumulated on L0C and must be moved out immediately during computation, the write-out data volume becomes $MNK/k$, which is higher than the basic Matmul baseline.

| Category                    | Total Memory Access Operations |
| --------------------------- | ------------------------------ |
| Basic matrix multiplication | $MNK(1/m+1/n) + MN$            |
| Single-core split-K         | $MNK(1/m+1/N) + MNK/k$         |

</details>

| Data Type | `L1TileShape::M` | `L1TileShape::N` | `L1TileShape::K` |
| --------- | ---------------- | ---------------- | ---------------- |
| FP16/BF16 | 256              | 128              | 512              |
| FP32      | 256              | 128              | 256              |

- Data alignment
  If data type conversion is required, data alignment must be applied to maximize the data transfer bandwidth throughout the processing pipeline.

```cpp
// Core processing logic of RemovePaddingNDAndCast
uint32_t loopsPerTile = RoundUp(tileLen, COMPUTE_LENGTH);
uint32_t coreLoops = tilesPerAiv * loopsPerTile;

for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
 // Calculate the offset for gmWC data transfer.

 // Transfer data from gmWC (src) to inputBuffer.
 copyGm2Ub(inputBuffer, src, ubLayout, srcLayout);
 // ...

 // Cast to half.
 AscendC::Cast(outputBuffer, inputBuffer,AscendC::RoundMode::CAST_RINT, actualDataNum);
 // ...

 // Transfer data from outputBuffer out to gmC (dst).
 copyUb2Gm(dst, outputBuffer[bufferIndex], dstLayout, ubLayout);
 // ...
}

```

### Swizzle Layout

The swizzle policy on Matrix C adopts an S-shaped pattern. Compared with a Z-shaped swizzle layout, the S-shaped strategy saves one data transfer operation at line breaks. As shown in the figure below, based on this S-shaped swizzle policy, the fractal matrix group loaded into L1A and L1B for the previous step is `<A02, <B20, B21, B22, B23>>`, while the next group to be loaded is `<A12, <B20, B21, B22, B23>>` (taking a `SwizzleOffset` of `4` as an example). During the line break transition, the data residing on L1B can remain fixed, and only the left matrix needs to be re-read from GM, thereby minimizing the loading overhead.

<div style="display: flex; justify-content: center;">
    <img src="https://raw.gitcode.com/user-images/assets/7694484/45fcbe2f-bcfc-4bde-b11a-43361f09c1d5/singleCoreSplitK_Swizzle.png" width="35%" height="auto">
</div>

> Note: This swizzle policy is effective for large-scale matrices. The scheduling policy preferentially distributes tasks across different AI Cores.

### Performance Benefits

Actual measurements indicate that the single-core split-K algorithm yields positive performance gains compared to the basic Matmul operator in large-scale scenarios. As the dimension of the K-axis increases, the positive gains achieved by reducing GM-to-L1 transfers outweigh the negative overhead of repeated write-backs to GM, as outlined in the table below.

However, it should be noted that if the M and N dimensions are too small, or if the K dimension is relatively low, **load imbalance across cores** may occur.

| M    | N    | K     | Time (μs) | Benchmark Time (μs) | Speedup ratio |
| ---- | ---- | ----- | --------- | ------------------- | ------------- |
| 2048 | 4096 | 4000  | 261       | 445                 | 1.7049        |
| 4096 | 4096 | 8000  | 917       | 1231                | 1.3424        |
| 4096 | 4096 | 40000 | 4669      | 7775                | 1.6652        |
| 2048 | 4096 | 80000 | 16850     | 57144               | 1.7499        |

Note:

- The benchmark refers to the [BasicMatmul](../00_basic_matmul/README.md) operator.
- All recorded metrics represent the total execution time of the kernel function, collected with the [`msprof`](https://www.hiascend.com/en/document/redirect/CannCommunitymsOpProf) profiling tool.
- In the preceding test case, matrices A, B, and C are in `layout::RowMajor` format.
- Test environment: The NPU model is 910B2, and the CANN package version is 8.2.RC1.
