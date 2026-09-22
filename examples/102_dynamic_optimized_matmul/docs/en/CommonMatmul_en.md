# CommonMatmul

## 1 Template Description

In generalized Matmul, there are two CommonMatmul templates. One is the pure CUBE type, named `CommonMatmulKernel`, and the other is the MIX type, named `PaddingCommonMatmulKernel`. If the AIV is required for data format conversion (see section 2.3), `PaddingCommonMatmulKernel` is used. The two templates are distinguished because when compiling a kernel, you need to specify whether the kernel is of the MIX type or the pure CUBE or AIV type. The MIX operator requires both the AIC and AIV to be launched, and the launch overhead is higher than that of the pure CUBE or AIV. Especially in small-shape scenarios, the performance is significantly affected. If the AIV is not required, the AIV does not need to be launched. In this case, the performance of the pure CUBE CommonMatmulKernel is better.

![image-20251209154303323](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209154303323.png)

The typical feature of CommonMatmul is that the basic blocks on L1 are directly used to split matrices A, B, and C. Matrix C is first divided into several basic task blocks with a size of `L1TileM × L1TileN`. Then, matrix A is read to L1 at a granularity of `L1TileM × L1TileK`, and matrix B is read to L1 at a granularity of `L1TileK × L1TileN`. The complete computation result of each task block is generated on one AI Core. To avoid unnecessary repeated data movement from L1 to L0, in CommonMatmul, L0TileM = L1TileM, L0TileN = L1TileN, and L0TileK is set to the maximum value under the space constraint of L0.

## 2 Optimization

The optimization points described here are those that are not included in [00_basic_matmul](../../../00_basic_matmul/README_en.md) but are included in CommonMatmul. The basic optimization points in 00_basic_matmul are not described here. The optimization points used in 00_basic_matmul are a subset of those used in CommonMatmul.

### Preload

The pseudocode for implementing preload in CommonMatmul is as follows:

```c++
...
uint32_t kTileCount = CeilDiv(actualShape.k(), l1TileShape.k());
...
// k loop at the block level
for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
    // If the current core is computing the first C block and kLoopIdx is 0, the data of the current tile block needs to be loaded.
    // If the current core is not computing the first C block or kLoopIdx > 0, the data of the current tile block does not need to be loaded, because the data has been loaded in the previous iteration.
    if (isFirstBlock && kLoopIdx == 0) {
        // Calculate the address of the GM block to be loaded based on kLoopIdx.
        copyGmToL1A;
        copyGmToL1B;
    }

    // Intra-block preload. Preload the next A tile and B tile of the current C block to be computed.
    if ((kLoopIdx != kTileCount - 1)) {
        // Calculate the address of the GM block to be loaded based on kLoopIdx + 1, because the data of the next block to be computed is loaded.
        copyGmToL1A;
        copyGmToL1B;
    }

    // Inter-block preload. Preload the first A tile and B tile of the next C block to be computed.
    if ((kLoopIdx != kTileCount - 1) && hasNextBlock) {
        // Load the first A tile and B tile of the next C block to be computed.
        copyGmToL1A;
        copyGmToL1B;
    }

    ...
    tileMmad;
    ...
}
...
copyL0CToGm;
...
```

For details about the actual code implementation, see [block_mmad_dynamic_common.h](../../../../include/catlass/gemm/block/block_mmad_dynamic_common.hpp).

`isFirstBlock`: whether the current core is computing the first C block. If yes, the value is `true`.

`hasNextBlock`: indicates whether the current core has the next C block to be computed. If yes, the value is `true`.

The `isFirstBlock` and `hasNextBlock` parameters are transferred from the kernel layer to the block layer.

The core idea of preload is to load the data of the next tile to be computed before the current tile is computed. In this way, the data transfer instructions are issued ahead of time, minimizing MTE2 pipeline stalls.

For details about preload, see [Pipeline optimization (Preload) in Matrix Multiplication Template Summary](../../../../docs/en/2_Design/01_kernel_design/04_matmul_summary.md).

### ShuffleK

![image-20251209170344805](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209170344805.png)

When multiple cores read the same GM data simultaneously, memory access contention occurs, which reduces the read bandwidth.

Generally, all cores start computing from the first block in the K direction. To alleviate the preceding problem, different cores start computing from different blocks. In this way, cores no longer read the same GM data at the exact same time.

In the figure, the numbers in the C matrix are core IDs, and the numbers in the A and B matrices are task block IDs. The left figure shows the conventional computation mode. In the figure, core 2 and core 3 access the same GM data of matrix A in the sequence of 0, 1, 2, and 3 simultaneously.

The right figure shows the computation mode after ShuffleK is used for optimization. In this figure, core 2 accesses the tiles of matrix A in the sequence of 2, 3, 0, and 1, while core 3 accesses the tiles of matrix A in the sequence of 3, 0, 1, and 2. This temporally staggered access avoids same-address access conflicts.

The block computation sequence in the figure is `GemmIdentityBlockSwizzle<2,1>`. For details, see [the swizzle explanation](../../../../docs/en/2_Design/01_kernel_design/02_swizzle.md).

### Padding

On A2 or A3, when matrix A or B is in ND format (Row-Major or Column-Major), if the stride of the matrix is not 512-byte aligned, the bandwidth of the ND2NZ transfer interface will significantly decrease. To avoid this problem, the AIV is used to convert the data format (or pad the data) of matrix A or B in advance. The purpose is to avoid accessing GM data with a non-512-byte aligned stride during the GM2L1 transfer.

#### Padding Modes of Matrix A or B

Currently, generalized Matmul supports three padding modes. The enumerated values are defined in [Padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp).

```cpp
enum class PaddingTag {
    NO_PADDING,
    PADDING_ND,
    PADDING_BLOCK_ND,
    PADDING_NZ
};
```

The preceding padding operations are global operations. That is, for the entire matrix A or B, a workspace of the same size as the aligned matrix A or B needs to be allocated. The AIV performs the padding operation, and the AIC starts the Matmul computation after the operation is complete.

- PADDING_ND

  ![image-20251209193549117](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193549117.png)

  PADDING_ND simply aligns the stride direction to 512 bytes. For example, in the figure, the shape is 512 × 511, and the stride is 511. After PADDING_ND, the stride changes to 512, which is aligned to 512 bytes. (For the half precision data type, this requires the stride to be a multiple of 256.)

- PADDING_BLOCK_ND

  ![image-20251209193625747](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193625747.png)

  PADDING_BLOCK_ND rearranges the matrix data and reorganizes the data by L1Tile block granularity. If the original matrix is in Row-Major format, as shown in the figure, the data is arranged in Row-Major format within each tile block after reorganization, and the tile blocks themselves are also arranged in a Row-Major layout. This is a nested data format. The advantage of this approach is that, regardless of the original matrix stride, the stride within each tile block becomes L1TileK (which is typically 512-byte aligned). This avoids bandwidth degradation caused by an excessively large stride (greater than 65536) or misalignment.

- PADDING_NZ

  ![image-20251209193718695](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193718695.png)

PADDING_NZ directly converts the Row-Major matrix to the zN format (or nZ format for Column-Major). In this way, on-the-fly ND2NZ conversion is not required when data is read from GM to L1, bypassing all performance degradation issues associated with ND2NZ conversion hardware units.

PADDING_NZ has two implementations tailored to different scenarios:

- When the stride direction (the inner axis of the matrix) is large, the AIV reads 16 rows of data at a time via repeat-loop vector reads. Although the stride of the repeated vector instruction is misaligned, the data volume per instruction is large, so misalignment has minimal impact on bandwidth. The data is then rearranged into the zN format inside the UB using the Copy instruction and written out to the workspace.
- When the stride direction (the inner axis of the matrix) is small (typically less than 96), using the first scheme results in low AIV read efficiency. In this case, the misaligned matrix data is read continuously into the UB, and the [TransDataTo5HD](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0200.html) instruction is executed to perform two transposition operations to complete data padding, while simultaneously converting the ND data into the NZ format.

Because PADDING_NZ offers superior overall performance, it is the method actively used in generalized Matmul. The other two padding modes may provide better performance than PADDING_NZ for certain specific shapes, and can be evaluated during fine-tuning phases.

#### Padding Modes of Matrix C

Because the computation results of basic blocks are stored in the zN layout on the L0C, a data layout conversion from NZ to ND is required when transferring data from the L0C to the UB. If the stride of matrix C is not 512-byte aligned, significant bandwidth degradation will occur.

![image-20251210001807177](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210001807177.png)

To avoid this problem, the current implementation of CommonMatmul allocates a workspace of the same size (after alignment) as matrix C. After the computation on the L0C is complete, the Fixpipe interface writes the results to the workspace using the aligned stride. The AIV is then deployed to execute the unpadding operation, writing the final result back to GM C.

The conditions for determining whether to pad matrix C in CommonMatmul are as follows:

```c++
PaddingTag paddingTagC = PaddingTag::PADDING_NONE;
// The size of matrix C cannot be too small, and it must not be 256-byte aligned.
if (static_cast<size_t>(m) * n > 2048 * 2048 && n > 256 && (n % 128 != 0)) {
    size_t totalDataSize = static_cast<size_t>(m) * k * CeilDiv(n, n1) * 2
        + static_cast<size_t>(k) * n * CeilDiv(m, m1) * 2 + static_cast<size_t>(m) * n * 2;
    // The total read/write data volume must be less than the size of the L2 cache.
    if (totalDataSize < 192 * 1024 * 1024) { // L2 cache size
        paddingTagC = PaddingTag::PADDING_ND;
    }
}
```

If the total data volume exceeds the L2 cache capacity, the AIV will likely have to read data from the GM when removing the padding. In this scenario, the unpadding overhead increases sharply, causing the execution penalty to outweigh the performance benefits.

#### Padding Modeling (Determining Whether Matrix A or B Needs to Be Padded)

Whether a matrix is padded directly impacts Matmul performance. Padding introduces fixed overheads. If the bandwidth gains realized from an aligned padding layout cannot offset these execution penalties, the padding operation yields a net negative performance return. The performance overhead of padding is driven primarily by two factors:

- The launch overhead of the AIV: Whenever a MIX kernel is launched, it incurs a deterministic latency of roughly 1–5 μs.
- The execution overhead of the padding operation: The absolute clock cycles consumed by the AIV read, compute, and write pipelines.

Empirical testing reveals the following patterns:

- Generally, when a matrix layout is misaligned, padding is necessary to recover the degraded read bandwidth. However, if that misaligned matrix is only read once without data reuse throughout the Matmul computation, it should not be padded.
- If the stride length of the matrix is less than 64 bytes, the on-the-fly ND2NZ hardware conversion units suffer extreme bandwidth degradation. In this case, padding must be applied, even if the matrix exhibits no data reuse.
- If the matrix dimensions are too small, that is, rows and columns fall below specific thresholds), padding is counterproductive. Small matrices can reside completely within the L1 cache to eliminate redundant reads, and the thin padding gains cannot amortize the overhead.
- When matrix reuse is low and the stride length is already 256-byte aligned, bypassing the padding operation yields better performance.

Because defining fixed thresholds across diverse shapes is brittle, a rigid heuristic strategy cannot maximize performance. Instead, the padding process must be modeled mathematically.

Set the following parameters:

- `B_aiv`: single-core read bandwidth of the AIV.
- `B_aic512`: single-core read bandwidth of the AIC during an on-the-fly ND2NZ transfer with a 512-byte aligned stride.
- `B_aicunalign`: single-core read bandwidth of the AIC during an on-the-fly ND2NZ transfer with a non-512-byte aligned stride.
- `T_headcost`: the fixed execution latency penalty introduced by launching a MIX kernel.
- `M`, `N`, `K`, `m1`, `n1`, and `k1`: matrix dimensions and L1 tile sizes.
- `C`: number of CUBE cores; `V`: number of VECTOR cores.

The maximum number of computing rounds per core is:
<center>R_max = RoundUp(CeilDiv(M, m1) * CeilDiv(N, n1) / C)</center>

The maximum data read volume of matrices A and B per AIC core is:
<center>CRead_a = R_max × (m1 × K), CRead_b = R_max × (n1 × K)</center>

If both matrices A and B are padded, the data read volume of the AIV per core is:

<center>VRead_a = M × K / V, VRead_b = K × N / V</center>

The total time consumed by the AIC and AIV to read data when both matrices A and B are padded is:

<center>T_11 = (CRead_a + CRead_b) / B_aic512 + (VRead_a + VRead_b) / B_aiv + T_headcost</center>

If padding is not performed, the total read time is:

<center>T_00 = (CRead_a + CRead_b) / B_aicunalign</center>

If matrix A is padded but matrix B is not padded, the total read time is:

<center>T_10 = CRead_a / B_aic512 + VRead_a / B_aiv + CRead_b / B_aicunalign + T_headcost</center>

If matrix B is padded but matrix A is not padded, the total read time is:

<center>T_01 = CRead_b / B_aic512 + VRead_b / B_aiv + CRead_a / B_aicunalign + T_headcost</center>

Compare the four execution times T00, T01, T10, and T11. Whichever strategy yields the shortest duration is selected.

The parameters `B_aiv`, `B_aic512`, and `T_headcost` can be treated as constants for identical hardware configurations and are acquired through empirical profiling. Conversely, `B_{aicunalign}` varies dynamically based on the `nd2nz` structural parameters (`nValue`, `dValue`, and `srcDValue`), requiring curve fitting from empirical benchmark results to derive an analytical formula.

The formulation above represents a simplified model of the padding selection process. Real-world implementations incorporate more granular constraints. See [select_kernel_bf16.h](../../include/select_kernel_b16.h) for details. Note that the polynomial regression coefficients provided are strictly optimized for A2 and A3 architectures. Targeting other platforms requires standalone profiling and curve fitting.

### Read Optimization in Special Scenarios

#### Scenario 1

![image-20251210102254855](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210102254855.png)

The hardware `nd2nz` transfer can be bypassed in favor of a standard strided memory copy using `DataCopy`. This is structured as a loop that processes one row per iteration, invoking `DataCopy` with a `repeat` factor to handle the layout transformation. The implementation is as follows:

```c++
// Use DataCopy with common stride instead of ND2NZ. The following uses half as an example.
for (int i = 0; i < nValue; ++i) {
    AscendC::DataCopyParams dataCopyParams {
        (dValue + 15) / 16, // repeat times
        1, // Amount of data to be copied at a time, in 32 bytes
        0, // Stride on the GM, in 32 bytes
        (256 - 16) / 16 // Stride on the L1, in 32 bytes
    };
    AscendC::DataCopy(buffer[i * 16], gmSrc[i * srcDValue], dataCopyParams);
}
```

When `M` is exceptionally small (typically `M < 8`), the hardware execution overhead of on-the-fly `nd2nz` layout conversion is less efficient than executing explicit line-by-line copies. Consequently, CommonMatmul intercepts these shapes and deploys strided `DataCopy` operations to achieve higher overall instruction throughput.

#### Scenario 2

![image-20251210104718451](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210104718451.png)

As illustrated above, when `K = 16`, the structural layout of matrix data on the GM is naturally congruent with its target arrangement inside L1. Under these conditions, the on-the-fly nd2nz unit is redundant. The system defaults to a direct contiguous bulk memory copy, which inherently yields significantly higher effective bandwidth than on-the-fly nd2nz.

#### Scenario 3

When matrix A is in Column-Major order and `M = 1`, a naive GM2L1 read accesses exactly one element per row, crippling memory reading efficiency. To mitigate this, matrix A is handled as a `1 × K` Row-Major matrix during computation. Since matrix A represents a vector at this scale, Row-Major and Column-Major representations are functionally equivalent, but treating it as a row vector allows a contiguous layout that vastly increases read efficiency.

Similarly, when matrix B is Row-Major and `N = 1`, the logic modifies it to be processed as a `K ×` 1 Column-Major matrix. For details about the logic, see [select_kernel_bf16.h](../../include/select_kernel_b16.h).

For edge cases where `K = 1`, execution is handed over to specialized compute templates tailored for the AIV pipeline.
