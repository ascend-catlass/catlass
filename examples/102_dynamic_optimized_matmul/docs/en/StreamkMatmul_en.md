# StreamkMatmul

## 1 Template Description

The StreamkMatmul template is designed for more granular load balancing. Compared with the MultiCoreSplitkMatmul template, StreamkMatmul can evenly distribute the workload to all cores by partitioning along the K-dimension. For example, in a Matmul operation where M = 512, N = 2048, and K = 1280, assume the L1Tile is m1 = 128, n1 = 256, and k1 = 256. This results in 32 task blocks. With 20 available cores, the second round of computation would leave 8 cores idle under standard mapping. Using MultiCoreSplitkMatmul to split K once results in 64 tasks. However, in the fourth round, 16 cores remain idle, still resulting in load imbalance. StreamkMatmul redistributes the second-round tasks across all 20 cores to achieve strict load balancing.

For detailed principles of the StreamkMatmul template, refer to the paper [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598).

### Template Principles

![image-20260121101922888](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121101922888.png)

As shown in the figure, for a shape where M = 512, N = 2048, and K = 1280, there are 32 task blocks. With 20 cores, the computation requires two rounds. (For details on swizzling, see [the swizzle explanation](../../../../docs/en/2_Design/01_kernel_design/02_swizzle.md). In the first round, each core processes one task block and the load is balanced. However, in the second round, only 12 task blocks remain, leaving 8 cores idle as illustrated:

![image-20260121102302163](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121102302163.png)

There are 12 task blocks in total. Each core computes a full task block. Since K = 1280 and k1 = 256, each block is partitioned into five tiles along the K-dimension, accumulating in L0C to produce the final result for that block. If Stream-K logic is applied to the 12 remaining blocks in the second round, the 12 × 5 = 60 K-tiles are distributed across 20 cores, with each core processing 3 tiles:

![image-20260121103528628](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121103528628.png)

This ensures the second-round load is perfectly balanced. Some cores will process the tail K-tiles of one task block and the head K-tiles of the subsequent block (e.g., core 1). Because these segments belong to different result blocks, their partial sums must be stored separately in the workspace.

![image-20260121114730452](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121114730452.png)

Each core is allocated two workspace slots. Total workspace size is 2 × m1 × n1 × sizeof(ElementAccumulator) × CoreNum. The workspace size is fixed and independent of the input shape. Upon completion of the Matmul kernel, the AIV units accumulate partial sums to produce the final result. For instance, if task block 20 consists of two partial sums computed by core 0 and core 1, the two AIVs associated with core 0 perform the reduction. If task block 21 is split across core 1, core 2, and core 3, the four AIVs for core 1 and core 2 handle the reduction.

### Key Optimization: Tail-Round Splitting

Only the final round of tasks is split. In all preceding rounds, the K-dimension is not partitioned, and results are written directly to GM_C. During the tail round, each core writes its partial sums to the workspace, which are then reduced by the corresponding AIVs. Since non-tail rounds are inherently balanced, bypassing K-splitting for them reduces synchronization and accumulation overhead.

### Key Optimization: Early Execution of the Tail Round

The tail round is advanced to the second-to-last position in the execution sequence:

![image-20260121121815177](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121121815177.png)

By initiating the tail round early, the Vector engine can begin partial sum accumulation in parallel with the Cube engine's remaining computations, effectively masking the Vector reduction overhead.

### Other Optimizations

The StreamkMatmul template incorporates existing CommonMatmul optimizations, including [Preload, ShuffleK, Padding, and specialized read optimizations](./CommonMatmul_en.md).

## Application Scenarios

1. Scenarios with significant load imbalance in the tail round.
