# Matrix Multiplication Template Summary

The `examples` directory in the repository contains multiple matrix multiplication sample templates. These are combinations of different matmul theoretical templates and engineering optimizations discovered in practice. After fully understanding each theoretical template and engineering optimization, developers can select an appropriate sample template based on their own problems, or even further combine them to create new sample templates not present in the repository, thereby achieving high-performance optimization for matrix multiplication.

Note that this document only summarizes samples related to matrix multiplication schemes. Other samples involving quantization, groupMatmul, and epilogues are not summarized here.

## Sample Template List

<details>
<summary><strong><font size="4">00_basic_matmul</font></strong></summary>

- Theoretical template: `Common`
- Engineering optimization: `Pipeline optimization (Multi-Buffer)`
- Key deliverables
  - host: [00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp)
  - kernel: [basic_matmul.hpp](../../../../include/catlass/gemm/kernel/basic_matmul.hpp)
  - blockMmad: [block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy: `MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">04_padding_matmul</font></strong></summary>

- Theoretical template: `Common`
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)`
  - `Read bandwidth optimization (padding) - PaddingMatrixND`
- Key deliverables
  - host: [04_padding_matmul](../../../../examples/04_padding_matmul/padding_matmul.cpp)
  - kernel: [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad: [block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy: `MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">06_optimized_matmul</font></strong></summary>

- Theoretical template: `Common`
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)`
  - `Pipeline optimization (Preload)`
  - `Read bandwidth optimization (padding) - PaddingMatrixNZ`
  - `Read bandwidth optimization (ShuffleK)`
  - `Read bandwidth optimization (instruction replacement in small-M scenarios)` (requires modifying the sample to enable)
- Key deliverables
  - host: [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)
  - kernel: [optimized_matmul.hpp](../../../../include/catlass/gemm/kernel/optimized_matmul.hpp)
  - Padding prologue component: [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad: [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy: `MmadAtlasA2Preload`
- ⚠️ Note: Even without `PaddingMatrixNZ` prologue, there is still overhead from MIX operator compilation and CV1:0 launch (greater than the overhead of launching only AIC).

</details>

<details>
<summary><strong><font size="4">09_splitk_matmul</font></strong></summary>

- Theoretical template: `MultiCoreSplitK`
- Engineering optimization: `Pipeline optimization (Multi-Buffer)`
- Key deliverables
  - host: [09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp)
  - kernel: [splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
  - blockMmad: [block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy: `MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">21_basic_matmul_preload_zN</font></strong></summary>

(This sample primarily demonstrates the adaptation method for NZ-layout inputs. It can also be adapted to ND-layout inputs, with no MIX operator compilation and launch overhead.)

- Theoretical template: `Common`
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)`
  - `Pipeline optimization (Preload)`
  - `Read bandwidth optimization (ShuffleK)`
- Key deliverables
  - host: [21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)
  - kernel: [basic_matmul_preload.hpp](../../../../include/catlass/gemm/kernel/basic_matmul_preload.hpp)
  - blockMmad: [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy: `MmadAtlasA2Preload`

</details>

<details>
<summary><strong><font size="4">22_padding_splitk_matmul</font></strong></summary>

- Theoretical template: `MultiCoreSplitK`
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)`
  - `Read bandwidth optimization (padding) - PaddingMatrixND`
- Key deliverables
  - host: [22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)
  - kernel: [padding_splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_splitk_matmul.hpp)
  - Padding prologue component: [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - SplitkReduceAdd epilogue component: [splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
  - blockMmad: [block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy: `MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">25_matmul_full_loadA</font></strong></summary>

(This sample and its related components only implement full loading of matrix A. To implement full loading of matrix B, refer to the key deliverables for self-development.)

- Theoretical template: `Common`
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)` (The fully loaded matrix A does not use multi-buffering in L1.)
  - `Read bandwidth optimization (L1 residency)`
- Key deliverables
  - host: [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp)
  - kernel: [matmul_full_loadA.hpp](../../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
  - blockMmad: [block_mmad_pingpong_full_loadA.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
- dispatchPolicy: `MmadAtlasA2FullLoadA`
- BlockScheduler: `GemmIdentityBlockSwizzleL1FullLoad`

</details>

<details>
<summary><strong><font size="4">31_small_matmul</font></strong></summary>

- Theoretical template: <idp:inline displayname="code" id="code85531551151311">Common</idp:inline>
- Engineering optimization:
  - <idp:inline displayname="code" id="code191623240184">Pipeline optimization (Multi-Buffer)</idp:inline>
  - `Scalar overhead reduction`
- Key deliverables
  - host: [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)
  - kernel: [small_matmul.hpp](../../../../include/catlass/gemm/kernel/small_matmul.hpp)
  - blockMmad: [block_mmad_small.hpp](../../../../include/catlass/gemm/block/block_mmad_small.hpp)
- dispatchPolicy: `MmadAtlasA2Small`
- BlockScheduler: not actually used in the kernel

</details>

<details>
<summary><strong><font size="4">34_single_core_splitk_matmul</font></strong></summary>

- Theoretical template: `SingleCoreSplitK`
- Engineering optimization:
  - <idp:inline displayname="code" id="code416252411819">Pipeline optimization (Multi-Buffer)</idp:inline>
  - <idp:inline displayname="code" id="code11507244191919">Read bandwidth optimization (padding) - PaddingMatrixNZ</idp:inline>
  - `Write bandwidth optimization`
- Key deliverables
  - host: [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
  - kernel: [single_core_slicek_matmul.hpp](../../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
  - Padding prologue component and RemovePaddingNDAndCast epilogue component: [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad: [block_mmad_single_core_splitk.hpp](../../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
- dispatchPolicy: `MmadAtlasA2SingleCoreSplitk`
- BlockScheduler: `SingleCoreSplitkGemmIdentityBlockSwizzle`

</details>

<details>
<summary><strong><font size="4">37_streamk_matmul</font></strong></summary>

- Theoretical template: StreamK (`tail-round multi-core split-K`); see the [design document](../../../../examples/102_dynamic_optimized_matmul/docs/en/StreamkMatmul_en.md)
- Engineering optimization:
  - `Pipeline optimization (Multi-Buffer)`
  - `Read bandwidth optimization (ShuffleK)`
- Key deliverables
  - host: [37_streamk_matmul](../../../../examples/37_streamk_matmul/streamk_matmul.cpp)
  - kernel: [streamk_matmul.hpp](../../../../include/catlass/gemm/kernel/streamk_matmul.hpp)
  - StreamkReduceAdd epilogue component: [streamk_matmul.hpp](../../../../include/catlass/gemm/kernel/streamk_matmul.hpp)
  - blockMmad: [block_mmad_streamk.hpp](../../../../include/catlass/gemm/block/block_mmad_streamk.hpp)
- dispatchPolicy: `MmadAtlasA2Streamk`
- BlockScheduler: `StreamkGemmIdentityBlockSwizzle`

</details>

## Theoretical Template List

<details>
<summary><strong><font size="4">Common</font></strong></summary>

### Tiling Modeling

<img src="https://raw.gitcode.com/user-images/assets/7801479/b1cb21ac-af83-4736-8582-4ed7392d766b/1common.png" width="80%">

The figure shows a conventional FP16 matrix multiplication (accumulation in FP32 on L0C). Define the following parameters:

- Problem shape: $M$, $N$, $K$
- TileShape when moving data into L1Cache: $m_1$, $n_1$, $k_1$
- TileShape when moving data into L0A/L0B and out of L0C: $m_0$, $n_0$, $k_0$

Core distribution is performed along the $M$ and $N$ directions. Tiling is done by $m_1$ and $n_1$. This produces $\frac{MN}{m_1n_1}$ basic task blocks, which are assigned to AIC cores for data movement and computation. Each basic task block needs to move $m_1K+Kn_1$ data elements, compute $m_1n_1$ results, and move them out. This yields the following constraints:

- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 = m_1$
- $n_0 = n_1$

### Amount of Data Read

Each basic task block needs to move $m_1K+Kn_1$ elements of data. The total amount of data read is:

$2Byte * [m_1K+Kn_1] * \frac{MN}{m_1n_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### Amount of Data Written

Each basic task block computes and moves out $m_1n_1$ results. The total amount of data written is:

$2Byte * MN$

### Amount of Computation

Each data point in the output matrix C requires $K$ multiply-add operations. The total amount of computation is fixed at:

$2MNK$

In most cases, computation time is rigid and depends only on the number of participating AIC cores. It is the same across all theoretical templates and will not be discussed further below.
</details>

<details>
<summary><strong><font size="4">MultiCoreSplitK</font></strong></summary>

### Tiling Modeling

<img src="https://raw.gitcode.com/user-images/assets/7801479/d4b0e2d2-4333-4df4-9af2-44654cc37e54/2multiCoreSplitK.png" width="80%">

The figure shows a conventional FP16 matrix multiplication (accumulation in FP32 on L0C). A total of 12 basic task blocks are tiled along the $MN$ direction. Assume that there are 24 physical AIC cores. In this case, the load is unbalanced. Therefore, the $K$ axis is tiled into two $k$ segments, producing 24 basic task blocks and achieving load balancing across the AIC cores. Define the following parameters:

- Problem shape: $M$, $N$, $K$
- TileShape when moving data into L1Cache: $m_1$, $n_1$, $k_1$
- TileShape when moving data into L0A/L0B and out of L0C: $m_0$, $n_0$, $k_0$
- <font color="red">Compared to the Common template</font>, a new tiling length $k$ along the $K$ direction is added.

When $m_1$ and $n_1$ are large, load imbalance may occur. That is, the number of task blocks tiled along the M and N directions is far less than the number of AIC cores, resulting in low read bandwidth (insufficient number of cores). Therefore, cores can be distributed along the K direction. This produces $\frac{MNK}{m_1n_1k}$ basic task blocks, which are assigned to AIC cores for data movement and computation. Each basic task block needs to move $m_1k+kn_1$ data elements, compute $m_1n_1$ results, and move them out. Hardware constraints are the same as those for the Common template.

### Amount of Data Read

Each basic task block needs to move $m_1K+Kn_1$ elements of data. The total amount of data read is the same as that for the `Common` template:

$2Byte * [m_1k+kn_1] * \frac{MNK}{m_1n_1k} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### Amount of Data Written

Each basic task block computes and moves out $m_1n_1$ results. It requires $\frac{K}{k}$ basic blocks to accumulate to obtain the final output for the $m_1n_1$ blocks of the output matrix C. The total amount of data written is:

$2Byte * MNK / k$

### Qualitative Analysis

Compared to the Common template, the amount of data read remains unchanged, the amount of data written increases, and there is overhead from ReduceAdd (including MIX operator compilation and launch overhead). However, more basic blocks are tiled, making load balancing easier.

</details>

<details>
<summary><strong><font size="4">SingleCoreSplitK</font></strong></summary>

### Tiling Modeling

<img src="https://raw.gitcode.com/user-images/assets/7801479/e16f5a39-2f7b-4a72-9d79-502cc8682e75/3singleCoreSplitK.png" width="80%">

The figure shows a conventional FP16 matrix multiplication (accumulation in FP32 on L0C). Define the following parameters:

- Problem shape: $M$, $N$, $K$
- TileShape when moving data into L1Cache: $m_1$, $n_1$, $k_1$
- TileShape when moving data into L0A/L0B and out of L0C: $m_0$, $n_0$, $k_0$

Compared to the Common template, to reduce the amount of data read and further increase $m_1$ and $n_1$ abstractly, consider directly computing the $m_1k_1$ tile with all corresponding $k_1n_1$ tiles (equivalent to scaling $n_1$ up to $N$). In this case, the output $m_0n_0$ tile cannot be resident in $L0C$ for accumulation and must be moved out in a timely manner, accumulating in `GM` via `atomicAdd`. Hardware constraints are as follows:

- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 <= m_1$
- $n_0 <= n_1$

### Amount of Data Read

Using the data read formula in the `Common` template, scale $n_1$ to $N$. Alternatively, from the perspective of tiling basic task blocks from matrix A, get $\frac{MK}{m_1k_1}$ basic blocks tiled. Each basic block moves in this matrix A tile and the corresponding entire B matrix tile. The amount of data moved is $m_1k_1+k_1N$:

$2Byte * [m_1k_1+k_1N] * \frac{MK}{m_1k_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{N}]$

### Amount of Data Written

Tile matrix A into basic task blocks, producing $\frac{MK}{m_1k_1}$ basic blocks. Each basic task block computes and moves out $m_1N$ results. The total amount of data written is:

$2Byte * MNK / k_1$

### Qualitative Analysis

Compared to the `Common` template, the amount of data moved in decreases, the amount of data written out increases, and there is no dependency on AIV.

</details>

## Engineering Optimization List

<details>
<summary><strong><font size="4">Pipeline Optimization (Multi-Buffer)</font></strong></summary>

### Problem Analysis

The following figure shows a simple scenario under a `Common` template. For a single AIC to process a basic task block C, the required matrix A/B tiles are small and can fit entirely in L1. When moving matrix A/B from L1 to L0, four times of partitioning are required.

<img src="https://raw.gitcode.com/user-images/assets/7801479/a62726da-f3c2-4c37-8a32-711046cfd239/8pingpong0.png" width="80%">

An example of the instruction pipeline diagram for each pipe is as follows:

<img src="https://raw.gitcode.com/user-images/assets/7801479/1dca48b1-2b5d-450c-9f66-18dbc1f7e2d1/8pingpong1.png" width="100%">

If, when loading data tiles into AIC's L1/L0A/L0B/L0C, you always try to fill all the space, it leads to serialized pipelines across different pipes, resulting in low efficiency.

### Optimization Solution

Use the conventional optimization technique Multi-Buffer. Enable multi-buffering in L1/L0A/L0B/L0C to make pipelines as parallel as possible to improve efficiency. This strategy is illustrated in the following figure:

<img src="https://raw.gitcode.com/user-images/assets/7801479/40315511-85cc-44ac-be3f-9efb6b5c0194/8pingpong2.png" width="80%">

An example of the instruction pipeline diagram for each pipe is as follows. `0` and `1` on MTE1 instructions indicate ping-pong pipelining:

<img src="https://raw.gitcode.com/user-images/assets/7801479/5917e1ca-dea0-4e00-8322-ef5ba2f32f46/8pingpong3.png" width="100%">

⚠️ Note that when combined with `L1 residency` optimization, disable multi-buffering for the resident tiles of matrix A/B.

### Code Location for the Feature

Since this is a conventional optimization technique, it is enabled in all blockMmad components.

</details>

<details>
<summary><strong><font size="4">Pipeline Optimization (Preload)</font></strong></summary>

### Problem Analysis

Through simulation pipeline analysis, issues were found with the ping-pong policy in blockMmad:

- On the MTE2 pipeline, there is a bubble between loading the last matrix A (matrix B) tile for the current matrix C basic block computation and loading the first matrix A (matrix B) tile for the next matrix C basic block computation.

### Optimization Solution

For the GM->L1 process, when reading $m_1k_1$ ($k_1n_1$) of the current round, compute the data read in the previous round (assuming one round of preloading and PRELOAD_STAGES = 1). The pseudocode for the steps is as follows:

```cpp
for ... {
    // Load data for the current round.
    copyGM2L1A
    copyGM2L1B
    preload_count++
    for (preload_count == PRELOAD_STAGES) {
        // Compute data from the previous PRELOAD_STAGES rounds.
        copyL12L0A
        copyL12L0B
        Mmad
    }
}
```

The following figure shows a simple scenario under the `Common` template. For a single AIC to process two basic task blocks C1 and C2, the required matrix A/B tiles need to be partitioned twice to fit in L1, and the matrix A/B L1 tiles need to be partitioned four times when moving from L1 to L0. (You can refer to the `Pipeline optimization (Multi-Buffer)` section above for a better understanding.) Below is a comparison of instructions for `MmadAtlasA2Pingpong`, `MmadAtlasA2Preload`, and `MmadAtlasA2PreloadAsync`:

- In `MmadAtlasA2Pingpong`, two blockMmad calls complete the computation of C1 and C2 separately.
- In `MmadAtlasA2Preload`, two blockMmad calls also complete the computation of C1 and C2 separately, but the GmToL1 movement of A3/B3 is advanced to the first blockMmad call.
- In `MmadAtlasA2PreloadAsync`, two blockMmad calls and one SynchronizeBlock call are used. The L1ToL0 movement, tileMmad, and C1 movement of A2/B2 are postponed from the first blockMmad call to the second. The L1ToL0 movement, tileMmad, and C2 movement of A4/B4 are postponed from the second blockMmad call to the SynchronizeBlock call.

<img src="https://raw.gitcode.com/user-images/assets/7801479/eca1e06a-7c9a-40d6-934b-6843f9229d7c/9preload0.png" width="100%">

An example of the instruction pipeline diagram for each pipe is as follows. Ultimately, the GmToL1 movement of A3/B3 blocks is advanced, reducing the movement bubble on the MTE2 pipeline:

<img src="https://raw.gitcode.com/user-images/assets/7801479/1edf8fb8-c8e2-4b84-b742-0e6c507efb64/9preload1.png" width="100%">

### Code Location for the Feature

- [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp), corresponding to dispatchPolicy: `MmadAtlasA2Preload`. You need to manually compute the information about the next block of preloaded data in the kernel.
- [block_mmad_preload_async.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async.hpp), corresponding to dispatchPolicy: `MmadAtlasA2PreloadAsync`. With asynchronous control in use, you do not need to manually compute the information about the next block of preloaded data. `Callback` can be passed after Mmad computation completes.
- [block_mmad_preload_async_with_callback.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp), corresponding to dispatchPolicy: `MmadAtlasA2PreloadAsyncWithCallback`. With asynchronous control in use, you do not need to manually compute the information about the next block of preloaded data. `Callback` can be passed before and after blockMmad computation.

</details>

<details>
<summary><strong><font size="4">Read Bandwidth Optimization (Padding)</font></strong></summary>

### Problem Analysis

When data read is the main pipeline, optimizing read bandwidth can yield performance gains. Using the FP16 matrix A as an example, the following low-bandwidth scenarios currently exist:

- **Low bandwidth due to Stride not being 512-byte aligned** When the movement parameter srcDValue (see [DataCopy-ND2NZ Movement with Channel Conversion](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_00127.html)) is not 512-byte aligned, the bandwidth decreases significantly.
- **Low bandwidth due to movement instruction restrictions**. For ND2NZ movement instructions, the srcDValue parameter is of type uint16, with a maximum value of 65535. When K > 65535, the movement instruction must be called repeatedly in the M direction with ndNum=1, reducing read bandwidth.
- **ND2NZ with channel conversion has bandwidth loss** compared to ND2ND (no layout conversion).

### Optimization Solution

For the above scenarios, an AIV helps rearrange the data format (a preprocessing action). When the rearrangement overhead is lower than the bandwidth loss, there is a performance gain. Based on complexity and the addressable scenarios, there are three different rearrangement methods.

#### PaddingMatrixND

<img src="https://raw.gitcode.com/user-images/assets/7801479/80501346-2ea2-42ba-8cc0-b6f614630606/4paddingND.png" width="100%">

Aligns the Stride direction to 512 bytes. This has the lowest implementation complexity and can handle bandwidth decreases caused by Stride misalignment.

#### PaddingMatrixBlockND

<img src="https://raw.gitcode.com/user-images/assets/7801479/9e5e0d37-ef6a-41e0-b54a-1f6d079e1179/4paddingBlockND.png" width="100%">

Rearranges data by $m_1*k_1$ as the "block" granularity. Within a block, it is row-major. Between blocks, it is also row-major. $k_1$ is 512-byte aligned. This has moderate implementation complexity and can handle bandwidth decreases caused by Stride misalignment and Stride exceeding 65535.

#### PaddingMatrixNZ

<img src="https://raw.gitcode.com/user-images/assets/7801479/11728de5-073d-481b-a030-b262f425161f/4paddingNZ.png" width="100%">

Rearranges data into the zN format, which poses the highest implementation complexity (among the padding strategies described). This is because the data layout is consistent with that in L1, resulting in the highest movement bandwidth. It can handle bandwidth decreases caused by Stride misalignment, Stride exceeding 65535, and ND2NZ with channel conversion.

<font color="red">In practice, different padding methods work for different cases. There is no globally go-to padding solution.</font>

### Code Location for the Feature

- [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp) contains the padding prologue component.
- For details about the actual adaptation, see [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp). Use `PaddingTag` and `PaddingBuilder` to assemble the padding prologues for matrix A and matrix B.

</details>

<details>
<summary><strong><font size="4">Read Bandwidth Optimization (ShuffleK)</font></strong></summary>

### Problem Analysis

Typically, all AIC cores start moving data from the first tile along the $K$ direction. This can result in multiple cores reading data from the same location in GM simultaneously, creating data read conflicts and reducing read bandwidth.

### Optimization Solution

<img src="https://raw.gitcode.com/user-images/assets/7801479/3b79fdb8-1154-46fa-bde8-057950d16b86/5shuffleK.png" width="100%">

Using the `Common` template as an example:

- $CoreX$ in $matC$ indicates that the basic block is assigned to the $X$-th AIC core for computation.
- $Aj$ indicates the $j$-th L1Tile basic block of matrix A partitioned along the $K$ axis under this $m_1$.
- $Bij$ indicates the $j$-th L1Tile basic block of matrix B partitioned along the $K$ axis under this $n_1$.
- The core distribution for the matC basic blocks uses Swizzle<2, 1>. For details, see [swizzle_explanation](./02_swizzle.md).

In the original solution shown on the left, both $Core2$ and $Core3$ move matrix A in the order of $A0$ -> $A1$ -> $A2$ -> $A3$, resulting in a data read conflict.

In the ShuffleK solution shown in the figure, the starting movement index $j$ is offset based on $CoreIdx$. $Core2$ moves matrix A in the order of $A2$ -> $A3$ -> $A0$ -> $A1$, and correspondingly moves matrix B in the order of $B02$ -> $B03$ -> $B00$ -> $B01$. $Core3$ moves matrix A in the order of $A3$ -> $A0$ -> $A1$ -> $A2$, and correspondingly moves matrix B in the order of $B13$ -> $B10$ -> $B11$ -> $B12$. This staggers the timing, avoiding access conflicts to the same addresses.

### Code Location for the Feature

- [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- [block_mmad_preload_async.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async.hpp)
- [block_mmad_preload_async_with_callback.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)

The above block-layer implementation code achieves staggering by setting the starting L1 index to `CoreIdx/kTileCount`:

```cpp
kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
startTileIdx = AscendC::GetBlockIdx();
firstTileIdx = startTileIdx % kTileCount;
```

</details>

<details>
<summary><strong><font size="4">Read Bandwidth Optimization (Instruction Replacement in Small-M Scenarios)</font></strong></summary>

### Problem Analysis

<img src="https://raw.gitcode.com/user-images/assets/7801479/e3a90904-677a-4973-b3b5-93e93ea072e2/6smallM.png" width="80%">

In matrix computation, when $M$ is small (for example, $M$ < 8), using DataCopy with ND2NZ with channel conversion (see [DataCopy - ND2NZ Movement with Channel Conversion](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_00127.html)) is not efficient.

### Optimization Solution

Use a `for` loop to move one row at a time, and call `DataCopy` for each row to perform multiple movements (using common strided movements).

### Code Location for the Feature

- [CopyGmToL1IntervalDataCopy](../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)
- For details about the actual adaptation, see [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp). Manually replace `using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;` in `struct TileCopyOpt` instead of using the default `using CopyGmToL1A = typename Base::CopyGmToL1A;`.

```diff
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    ...

-   // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;
+   using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

-   using CopyGmToL1A = typename Base::CopyGmToL1A;
+   // using CopyGmToL1A = typename Base::CopyGmToL1A;
    ...
};
```

</details>

<details>
<summary><strong><font size="4">Read Bandwidth Optimization (L1 Residency)</font></strong></summary>

### Optimization Solution

In practice, a tile can be made resident in L1, reducing repeated reads of tile data. This effectively improves read bandwidth. This feature needs to be implemented in conjunction with different theoretical templates.

### Code Location for the Feature

- For details about the `Common` template, see [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp) and related deliverables. This example optimizes performance in specific scenarios by fully loading matrix A on a single core or multiple cores along the M axis, and uses a dedicated swizzle policy to increase the reuse frequency of the fully loaded matrix A block in L1.
  - kernel: [matmul_full_loadA.hpp](../../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
  - blockMmad: [block_mmad_pingpong_full_loadA.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
  - dispatchPolicy: <idp:inline displayname="code" id="code413194620281">MmadAtlasA2FullLoadA</idp:inline>
  - BlockScheduler: <idp:inline displayname="code" id="code13616124752811">GemmIdentityBlockSwizzleL1FullLoad</idp:inline>
- The `single-core split-K` template ([34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)) considers the optimization of L1Tile block residency in its theoretical design.
  - kernel: [single_core_slicek_matmul.hpp](../../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
  - blockMmad: [block_mmad_single_core_splitk.hpp](../../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
  - dispatchPolicy: <idp:inline displayname="code" id="code94861649152910">MmadAtlasA2SingleCoreSplitk</idp:inline>
  - BlockScheduler: <idp:inline displayname="code" id="code17850135052918">SingleCoreSplitkGemmIdentityBlockSwizzle</idp:inline>

</details>

<details>
<summary><strong><font size="4">Scalar Overhead Reduction</font></strong></summary>

### Problem Analysis

In small-shape scenarios, for example, in the `Common` template:

- The number of cores in the $M$ and $N$ directions is less than the actual number of physical cores. Each AIC physical core processes a maximum of one basic task block.
- $k_1$ >= $K$. No partitioning is needed along the $K$ direction when moving from GM to L1.

In this case, the total kernel execution time is relatively small, and scalar overhead has a significant impact on performance.

### Optimization Solution

Reduce redundant scalar computations.

- Do not use BlockScheduler inside the kernel to assign task blocks to physical cores. Manually calculate the task block corresponding to each physical core.
- Eliminate the basic block loop inside the kernel (each AIC processes only one task block).
- Simplify offset-related calculations inside the kernel.
- Eliminate the $m$ and $n$ loops of L1A/L1B inside blockMmad.
- Simplify offset-related calculations inside blockMmad.

### Code Location for the Feature

- For details, see [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp). You can compare it with the deliverables of [00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp) to deepen your understanding.
  - kernel: [small_matmul.hpp](../../../../include/catlass/gemm/kernel/small_matmul.hpp)
  - blockMmad: [block_mmad_small.hpp](../../../../include/catlass/gemm/block/block_mmad_small.hpp)

</details>

<details>
<summary><strong><font size="4">Write Bandwidth Optimization</font></strong></summary>

### Problem Analysis

When data write is the main pipeline, optimizing the write bandwidth can yield performance gains.

- When `dstStride` is not 512-byte aligned during write, the bandwidth decreases significantly.
- Writing with NZ2ND with channel conversion incurs bandwidth loss.

### Optimization Solution

For the above scenarios, an AIV can be used to rearrange the data format. When the rearrangement overhead is lower than the bandwidth loss, there is a performance gain. Four rearrangement methods are provided below.
<img src="https://raw.gitcode.com/user-images/assets/7801479/89afcf74-a193-431b-aea3-a8de2abcb4f9/7rmPadding1.png" width="80%">

(↑) **Method 1**: Use a local workspace. Write ND to GM with 512-byte alignment. Then rearrange data in UB at block granularity before writing it back to GM.

<img src="https://raw.gitcode.com/user-images/assets/7801479/12665171-d86f-4208-8c11-3bef2359cfa1/7rmPadding2.png" width="80%">

(↑) **Method 2**: Use a full workspace. Write ND to GM with 512-byte alignment. After the entire result is written, start rearranging data in UB and write it to GM.

<img src="https://raw.gitcode.com/user-images/assets/7801479/13bd80b2-d6ab-4520-839c-cf5e700db0d5/7rmPadding3.png" width="80%">

(↑) **Method 3**: Use a local workspace. Write NZ to GM with 512-byte alignment. Then rearrange data in UB at block granularity before writing ND back to GM.

<img src="https://raw.gitcode.com/user-images/assets/7801479/204d24e9-5dc8-4318-bfe1-7f958598c514/7rmPadding4.png" width="80%">

(↑) **Method 4**: Use a full workspace. Write NZ to GM with 512-byte alignment. After the entire result is written, start rearranging data in UB and write ND back to GM.

### Code Location for the Feature

- [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp) implements `RemovePaddingNDAndCast` epilogue component that includes **Method 2**.
- For details about actual adaptation, see [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp).

</details>

## Brief Overview of Template Application

For details, see the select_kernel policy of [102_dynamic_optimized_matmul](../../../../examples/102_dynamic_optimized_matmul/include/select_kernel_b16.h).

<details>
<summary><strong><font size="4">Template Selection</font></strong></summary>

First, try to tune TileShape based on [00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp) and obtain a **performance baseline**. For details, see [Template Library Optimization Guide](../../1_Practice/11_matmul_optimization.md).

Then, identify whether the scenario fits each template and compare with the performance baseline:

- [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp):
  - Calculate the number of current basic task blocks taskBlocks.

    ```cpp
    taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
    ```

  - The number of basic task blocks is less than the number of AIC cores: $taskBlocks < aicCoreNum$
  - The $K$ axis is small: $K <= k_1$
- [09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp) or [22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp) (with padding prologue)
  - Select $m_1$, $n_1$, and $k_1$.
    - Set $m_1 = 128$, $n_1 = 256$, and $k_1 = 256$.
    - If either of the following conditions is met, change to $m_1 = 256$ and $n_1 = 128$.
      - Both matrices A and B are column-major.
      - Matrix A is column-major, matrix B is row-major, and $M > N$.
  - Calculate the number of current basic task blocks taskBlocks.

    ```cpp
    taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
    ```

  - The following two scenarios are met:
    - The number of basic task blocks is less than half of the number of AIC cores, and the $K$ axis is sufficiently large: $taskBlocks < aicCoreNum / 2, K > 5120$
    - The number of basic task blocks is less than 3, and the $K$ axis is not small: $taskBlocks <= 2, K > 1024$
- [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp) (with padding prologue) and [21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp) (manually change to ND input)
  - These feature stronger generalization and are suitable for the remaining scenarios.
  - If padding is not required, you are advised to use the [21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp) template to reduce the overhead of MIX operator compilation and launch.

⚠️ The applicable scenarios for the full-load feature in [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp) and the single-core split-K solution in [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp) are yet to refine.

</details>

<details>
<summary><strong><font size="4">Padding Selection</font></strong></summary>

Consider padding prologue when Stride is not 512-byte aligned. However, the overhead introduced by padding and the MIX operator compilation and launch overhead must be considered (the small-shape method in [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp) is not recommended for additional padding adaptation).

Developers are advised to try `PaddingMatrixND`, `PaddingMatrixBlockND`, and `PaddingMatrixNZ` themselves. In terms of generalization, `PaddingMatrixNZ` has more advantages.

</details>
