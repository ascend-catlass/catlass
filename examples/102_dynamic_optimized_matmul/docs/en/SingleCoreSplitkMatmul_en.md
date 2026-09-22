# SingleCoreSplitkMatmul

## Template Description

![image-20260210002058199](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210002058199.png)

CommonMatmul requires that the complete computation result of each task block be generated within a single AI Core. To avoid unnecessary repeated data movement from L1 to L0 and to complete the entire K-direction reduction on L0C, CommonMatmul enforces that L0TileM = L1TileM and L0TileN = L1TileN, while L0TileK is maximized based on the spatial constraints of L0C.

The capacity of L0C is smaller than that of L1 (for example, on A2, L0C is 128 KB while L1 is 256 KB). When scaling up L1TileM and L1TileN, the L0C capacity typically reaches its spatial bottleneck first. For example, consider a common L1TileShape where the data type is half, L1TileM = 128, L1TileN = 256, and L1TileK = 256. This precisely saturates L0C (`128 × 256 × sizeof(float) = 128 KB` for accumulation). However, L1 only uses `128 × 256 × 2 × 2 + 256 × 256 × 2 × 2 = 384 KB` (under double-buffering layouts). If the tile shape scales up to L1TileM = 256, L1TileN = 256, and L1TileK = 256, L1 becomes fully utilized, but L0C cannot accommodate the resulting `256 × 256 × sizeof(float) = 256 KB` basic blocks. This scenario requires single-core Split-K tiling, where partial sums are temporarily staged to Global Memory (GM) and reduced with atomic addition. In the single-core Split-K template, `L0TileM < L1TileM` and `L0TileN < L1TileN` are permitted. As illustrated above, where `L1TileM = 2 × L0TileM` and `L1TileN = L0TileN`, L0C generates a partial sum block of size L0TileM × L0TileN per iteration, which is then written and accumulated into the workspace on GM with atomic addition. In this diagram, the Matmul operator generates 8 partial sums in total, and the corresponding spatial locations undergo 3 atomic updates to yield the final output.

The performance gains of single-core Split-K stem from scaling up L1TileM. Based on geometric modeling, larger L1TileM dimensions reduce input data reloading from GM. However, because the generated partial sums cannot complete their reduction within L0C, they must be written out to GM for atomic reduction, introducing additional write-out overhead.

## Optimization

Three SingleCoreSplitK template variants are derived based on distinct data-loop scheduling strategies.

### SingleCoreSplitkKLoopMiddleMatmul

![image-20260210193559636](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210193559636.png)

Pseudocode description:

```c++
for gmTileN1 in N:
    for m1 in gmTileM1:
        for k1 in K:
            loadL1A(m1,k1)
            for n1 in gmTileN1:
                loadL1B(k1,n1)
                for m0 in m1:
                    for k0 in k1:
                        loadL0A(m0, k0)
                            for n0 in n1:
                                loadL0B(k0, n0)
                                mmad(A, B)
                                writeC(m0, n0)
```

### Engineering optimization

- L1 residency

  For single-core Split-K templates, keeping blocks resident in L1 is a primary optimization. As shown in the loop structure, tile blocks of Matrix A can be pinned inside L1. For example, on core 0, an L1 block of Matrix A is multiplied against two successive L1 blocks of Matrix B to generate partial sums. Thus, Matrix A needs to be loaded from GM only once, reducing memory traffic.

- Disabling double-buffering for resident matrix

  Because Matrix A remains resident within the L1 buffer, allocating a double-buffer structure for it is unnecessary. The reclaimed L1 footprint can be reallocated to scale up total tile dimensions.

- Localized workspace buffering to improve write hit rate

  Each AI Core is allocated a fixed-size workspace partition. Because successive Fixpipe instructions stream write-outs into the exact same workspace address space, the data frequently hits the underlying hardware cache, maximizing write bandwidth.

- Disabling unitflag and enabling L0C double-buffering

  By optimizing down L0TileM or L0TileN, double-buffering can be enabled directly on L0C. This achieves tight pipeline interleaving between MMAD compute execution and Fixpipe write-out operations, providing superior instruction masking compared to enabling unitflag.

- Aligned write-out

  Since final reduction relies on atomic additions in the workspace, Fixpipe operations streaming out to the workspace have their address strides aligned to 512-byte boundaries. This fully saturates the NZ-to-ND write-out bandwidth.

- SetMMLayoutTransform

### Advantages and Disadvantages

Advantage: `SingleCoreSplitkKLoopMiddleMatmul` facilitates the configuration of localized workspace partitions. Because Fixpipe consistently targets a highly localized workspace memory range, the write path maintains an exceptional cache hit rate, maximizing write-out bandwidth.

Disadvantage: Each core is assigned a relatively large spatial tile of Matrix C, which increases vulnerability load imbalance.

### SingleCoreSplitkKLoopOuterMatmul

![image-20260210194055831](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210194055831.png)

Pseudocode description:

```c++
for k1 in K:
    for m1 in M:
        loadL1A(m1,k1)
        for n1 in N:
            loadL1B(k1,n1)
              for m0 in m1:
                for k0 in k1:
                    loadL0A(m0, k0)
                    for n0 in n1:
                        loadL0B(k0, n0)
                        mmad(A, B)
                        writeC(m0, n0)
```

For `SingleCoreSplitkKLoopOuterMatmul`, a global workspace matching the total element count of Matrix C must be allocated whenever the accumulation data type differs from the final output precision, or when Matrix C dimensions violate hardware alignment rules.

### Engineering optimization

- Optimizations such as L1 residency, disabling double-buffering for resident tensors, Disabling unitflag and enabling L0C double-buffering, aligned write-outs, and `SetMMLayoutTransform` map identically to those in `SingleCoreSplitkKLoopMiddleMatmul`.

- Load balancing

  In the `SingleCoreSplitkKLoopOuterMatmul` template, basic blocks of Matrix C are interleaved evenly across all active AI Cores with a swizzle pattern.

### Advantages and Disadvantages

Advantage: `SingleCoreSplitkKLoopOuterMatmul` achieves more uniform load balancing than `SingleCoreSplitkKLoopMiddleMatmul`.

Disadvantage: This strategy requires backing workspace storage equivalent to the full footprint of Matrix C. The kernel computes an entire horizontal layer of partial sums across Matrix C before advancing to the next layer down the K-axis. If Matrix C is exceptionally large, this execution footprint thrashes the cache, drastically degrading write-out bandwidth.

### SingleCoreSplitkForSmallKMatmul

This template targets a specialized scenario. When K is highly constrained (K ≤ L1TileK), tiling along the K-direction becomes unnecessary, bypassing the requirement for workspace-allocated atomic accumulation. For example, when source matrices A and B are provided in half precision, if Matrix C matches native alignment constraints, the half precision data can be streamed directly out to the memory space allocated for Matrix C. If Matrix C exhibits non-alignment, the half data blocks are instead staged out to a 512-byte stride-aligned workspace before a downstream AIV flushes the data back to GM C.

![image-20260210212742124](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210212742124.png)
