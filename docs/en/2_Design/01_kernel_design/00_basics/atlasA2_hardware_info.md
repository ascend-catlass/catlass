# Atlas A2 Hardware Basics

## Programming Model

<https://www.hiascend.com/en/document/redirect/CannCommunityAscendCOnHost>

## Architecture Diagram

<https://www.hiascend.com/en/document/redirect/CannCommunityAscendCOnHost>

## Memory Units

| Level     | Subdivision | Space                            | Description                                                                                                                       |
| --------- | ----------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| GM        |             |                                  |                                                                                                                                   |
| L2 Cache  |             |                                  |                                                                                                                                   |
| L1 Buffer |             | 512KB                            | Temporarily stores data reused repeatedly inside AI Core; minimum access granularity is cacheline-32B; usually split into 4 x 128KB, L1A pingpong, L1B pingpong |
| L0 Buffer | L0A/L0B     | 64KB each<br />usually split into two 32KB each | Input of Cube<br />usually split into 4 x 32KB, L0A pingpong, L0B pingpong                                                |
|           | L0C         | 128KB                            | Cube output,<br />double buffer can be set explicitly; usually enabled via unit-flag for compute-while-move                    |
|           | BT          | 1KB                              | Stores bias                                                                                                                       |
|           | FB          | 7KB                              | Stores quantization parameters, relu parameters                                                                                   |
| UB        |             | 192KB<br />usually split into two 96KB | Vector storage                                                                                                               |

| Logical Position | Physical Position | Load Instruction                     | Supported Format                     |
| ---------------- | ----------------- | ------------------------------------ | ------------------------------------ |
| GM               | GM                | DataCopy                             | RowMajor/ColumnMajor/nZ/zN            |
| A1/B1/C1         | L1                | DataCopy                             | zN(RowMajor)/nZ(ColumnMajor)/ND(m=1) |
| A2               | L0A               | LoadData                             | zZ                                   |
| B2               | L0B               | LoadData                             | nZ                                   |
| CO1              | L0C               | -                                    | zN                                   |
| CO2              | BiasTable Buffer  | DataCopy                             | -                                    |
| C2PIPE2GM        | Fixpipe Buffer    | DataCopy                             | -                                    |
| tbufVECIN        | UB                | GM->UB DataCopy<br />UB->UB DataCopy | ND/NZ                                |
| tbufVECOUT       | UB                |                                      | ND/NZ                                |
| tbufVECCALC      | UB                |                                      | ND/NZ                                |
|                  |                   |                                      |                                      |

## Deriving TileShape from Hardware Capacity

The table above lists physical capacities for L1 / L0A / L0B / L0C. What developers need in practice is: under a **chosen DispatchPolicy + BlockMmad**, given the data type and per-level buffer stages, which `L1TileShape` / `L0TileShape` values are legal. Constants are defined in [`Arch::AtlasA2`](../../../../../include/catlass/arch/arch.hpp) (`L1_SIZE=512KB`, `L0A/L0B_SIZE=64KB`, `L0C_SIZE=128KB`).

### Prerequisite: DispatchPolicy and stages assumed below

The formulas and reference table below use the most common basic-Matmul baseline:

| Item | Value | Notes |
| --- | --- | --- |
| DispatchPolicy | [`MmadAtlasA2Pingpong`](../../../../../include/catlass/gemm/dispatch_policy.hpp) (`STAGES = 2`) | Default in examples such as [00_basic_matmul](../../../../../examples/00_basic_matmul/) |
| BlockMmad | Matching pingpong specialization | L1 / L0A / L0B share the unified `STAGES`; see [`block_mmad_pingpong.hpp`](../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp) |
| L1A / L1B stages | **2** | Two tiles each (double buffer) |
| L0A / L0B stages | **2** | Same as above |
| L0C stages | **1** | On the pingpong path L0C is usually single-buffered; usage counts one accumulator tile (`ENABLE_UNIT_FLAG` enables fine-grained Fixpipe overlap but still uses a single-tile capacity check) |

In general: `bytes per tile * stages at that level <= physical capacity of that level`. Other policies may split stages per level (e.g. `L1A_STAGES` / `L1B_STAGES` / `L0A_STAGES` / `L0B_STAGES` / `L0C_STAGES`; see [Dispatch Policies](../03_dispatch_policies.md)), so **do not assume the factor 2 below**.

> **Always size tiles against the DispatchPolicy and BlockMmad you actually use:** read per-level stages (or the unified `STAGES`) from the Policy, then rewrite the usage formulas for that BlockMmad’s L1 layout (full-load A, asymmetric A/B pingpong, extra bias/scale space, etc.) and verify with `static_assert` or a manual check. The formulas below use pingpong as an example.

### Usage formulas (`MmadAtlasA2Pingpong`; L1/L0A/L0B stages=2, L0C stages=1)

Let `sizeof(Element)` be the element size in bytes; the L0C accumulator is typically `fp32` (4 bytes):

```text
L1 usage = (L1.M * L1.K + L1.K * L1.N) * sizeof(Element) * 2   <= 512 KB   # * L1 stages
L0A usage = L0.M * L0.K * sizeof(Element) * 2                   <= 64 KB    # * L0A stages
L0B usage = L0.K * L0.N * sizeof(Element) * 2                   <= 64 KB    # * L0B stages
L0C usage = L0.M * L0.N * 4                                     <= 128 KB   # * L0C stages(=1); mmad accumulator as fp32
```

Additional engineering constraints: each axis is typically a **multiple of 16**; a common library simplification is `L0.M = L1.M`, `L0.N = L1.N`, with `L0.K = L1.K / 4`.

### Common data-type reference (still under `MmadAtlasA2Pingpong`, L0B stages=2)

| Input dtype | sizeof | Typical L1TileShape (M,N,K) | Typical L0TileShape (M,N,K) | L0B usage (*2 stages) | Example |
| ----------- | ------ | --------------------------- | --------------------------- | ------------------- | ------- |
| fp16 / bf16 | 2 | `<128,256,256>` | `<128,256,64>` | `64*256*2*2 = 64 KB` | [00_basic_matmul](../../../../../examples/00_basic_matmul/) |
| int8 | 1 | `<128,256,512>` | `<128,256,128>` | `128*256*1*2 = 64 KB` | [12_quant_matmul](../../../../../examples/12_quant_matmul/) |
| fp32 | 4 | `<128,128,256>` | `<128,128,64>` | `64*128*4*2 = 64 KB` | [Matmul Optimization - TileShape Constraints - Scenario 2](../../../1_Practice/11_matmul_optimization.md#tileshape-constraints) |

For the same capacity budget, smaller elements allow a larger `K` (or `N`). Therefore **do not reuse an int8 TileShape unchanged for fp16**: applying int8's `L0TileShape<128,256,128>` to fp16 yields

```text
L0B = 128 * 256 * 2 * 2 = 128 KB greater than 64 KB   # overflow
```

When changing dtype, shrink `L0.K` or `L0.N` at least by the `sizeof` ratio and scale `L1.K` accordingly (in the example above, set `L0.K=64` and `L1.K=256`, matching the common fp16 combo). If you only change L0 while keeping int8's `L1.K=512`, L1 usage becomes 768KB and exceeds 512KB. Then re-check L1 / L0A / L0C with the same formulas.

### Quick checklist

1. Confirm the **DispatchPolicy** and **BlockMmad** in use; read L1A/L1B, L0A/L0B, and L0C stages (or the unified `STAGES`) - do not assume they equal 2.
2. Pick the dtype and write down `sizeof(Element)`; form usage as `bytes per tile * stages at that level` (same shape as above, only the stage factors change).
3. Constrain `L0.M/N/K` so L0A/L0B stay within 64KB and L0C within 128KB; then choose `L1.K` (often about `4 * L0.K` under pingpong) and verify L1 <= 512KB.
4. Apply that BlockMmad’s extra constraints (full-load, asymmetric stages, prologue/bias space, etc.), then re-check after switching dtype or copying TileShape from another example.

For full worked examples (fp16 / fp32 / int8) and later tiling tuning, see [Matmul Optimization Guide - TileShape Constraints](../../../1_Practice/11_matmul_optimization.md#tileshape-constraints). For Policy field meanings, see [Dispatch Policies](../03_dispatch_policies.md). When first filling TileShape on the host side, also see [Host-side Example Assembly](../../../1_Practice/02_host_example_assembly.md).
