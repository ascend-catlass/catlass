# Atlas A2 硬件基础信息

## 编程模型

<https://www.hiascend.com/document/redirect/CannCommunityAscendCOnHost>

## 架构图

<https://www.hiascend.com/document/redirect/CannCommunityAscendCOnHost>

## 内存单元

| 层级      | 层级细分 | 空间                             | 说明                                                                                                |
| --------- | -------- | -------------------------------- | --------------------------------------------------------------------------------------------------- |
| GM        |          |                                  |                                                                                                     |
| L2 Cache  |          |                                  |                                                                                                     |
| L1 Buffer |          | 512KB                            | 暂存AI Core内部反复使用数据，最小访问粒度cacheline-32B，通常切为4个128KB，L1A pingpong，L1B pingpong |
| L0 Buffer | L0A/L0B  | 分别64KB<br />一般各分为两个32KB | Cube的输入<br />通常切分为4个32KB， L0A pingpong，L0B pingpong                                      |
|           | L0C      | 128KB                            | Cube输出，<br />可显式设置doublebuffer，一般通过unit-flag开启边算边搬                               |
|           | BT       | 1KB                              | 存放bias                                                                                            |
|           | FB       | 7KB                              | 存放量化参数，relu参数                                                                              |
| UB        |          | 192KB<br />一般分为两个96KB      | Vector存储                                                                                          |

| 逻辑位置    | 物理位置        | 搬入指令                             | 支持格式                             |
| ----------- | --------------- | ------------------------------------ | ------------------------------------ |
| GM          | GM              | DataCopy                             | RowMajor/ColumnMajor/nZ/zN           |
| A1/B1/C1    | L1              | DataCopy                             | zN(RowMajor)/nZ(ColumnMajor)/ND(m=1) |
| A2          | L0A             | LoadData                             | zZ                                   |
| B2          | L0B             | LoadData                             | nZ                                   |
| CO1         | L0C             | -                                    | zN                                   |
| CO2         | BiasTableBuffer | DataCopy                             | -                                    |
| C2PIPE2GM   | FixpipeBuffer   | DataCopy                             | -                                    |
| tbufVECIN   | UB              | GM->UB DataCopy<br />UB->UB DataCopy | ND/NZ                                |
| tbufVECOUT  | UB              |                                      | ND/NZ                                |
| tbufVECCALC | UB              |                                      | ND/NZ                                |
|             |                 |                                      |                                      |

## 从硬件容量反推 TileShape

上表给出了 L1 / L0A / L0B / L0C 的物理容量，但开发时真正要用的是：在**选定的 DispatchPolicy + BlockMmad** 下，给定数据类型与各级 Buffer 的 stages，哪些 `L1TileShape` / `L0TileShape` 合法。常量定义见 [`Arch::AtlasA2`](../../../../../include/catlass/arch/arch.hpp)（`L1_SIZE=512KB`，`L0A/L0B_SIZE=64KB`，`L0C_SIZE=128KB`）。

### 前提：公式所基于的 DispatchPolicy 与 stages

下文占用公式与对照表均以基础 Matmul 最常见的组合为基准：

| 项 | 取值 | 说明 |
| --- | --- | --- |
| DispatchPolicy | [`MmadAtlasA2Pingpong`](../../../../../include/catlass/gemm/dispatch_policy.hpp)（`STAGES = 2`） | [00_basic_matmul](../../../../../examples/00_basic_matmul/) 等样例默认策略 |
| BlockMmad | 与该 Policy 匹配的 pingpong 特化 | L1 / L0A / L0B 按统一 `STAGES` 开缓冲；见 [`block_mmad_pingpong.hpp`](../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp) |
| L1A / L1B stages | **2** | 各保留 2 份 Tile（double buffer） |
| L0A / L0B stages | **2** | 同上 |
| L0C stages | **1** | pingpong 路径下 L0C 通常单缓冲；占用按 1 份累加器 Tile 计（启用 `ENABLE_UNIT_FLAG` 时与 Fixpipe 细粒度并行，仍按单份容量约束） |

各级占用的一般形式为：`单份 Tile 字节数 * 该级 stages <= 该级物理容量`。换用其他 Policy 时，stages 可能按级拆分（如 `L1A_STAGES` / `L1B_STAGES` / `L0A_STAGES` / `L0B_STAGES` / `L0C_STAGES`，见 [Dispatch 策略](../03_dispatch_policies.md)），**不能直接套用下面的系数 2**。

> **实际选型必须以所用 DispatchPolicy 与 BlockMmad 为准**：先从 Policy 读出各级 stages（或统一 `STAGES`），再结合该 BlockMmad 的 L1 划分方式（是否全载 A、A/B 是否对称 pingpong、是否另占 bias/scale 等）重写占用式并做 `static_assert` / 手工验算。下文公式以 pingpong 为例进行计算。

### 占用公式（`MmadAtlasA2Pingpong`，L1/L0A/L0B stages=2，L0C stages=1）

设元素字节数为 `sizeof(Element)`，累加器在 L0C 上一般为 `fp32`（4 Byte）：

```text
L1 占用 = (L1.M * L1.K + L1.K * L1.N) * sizeof(Element) * 2   <= 512 KB   # * L1 stages
L0A 占用 = L0.M * L0.K * sizeof(Element) * 2                   <= 64 KB    # * L0A stages
L0B 占用 = L0.K * L0.N * sizeof(Element) * 2                   <= 64 KB    # * L0B stages
L0C 占用 = L0.M * L0.N * 4                                     <= 128 KB   # * L0C stages(=1)；mmad 累加器按 fp32
```

另有工程约束：各轴取值一般为 **16 的倍数**；库内常见简化是 `L0.M = L1.M`、`L0.N = L1.N`，并推荐 `L0.K = L1.K / 4`。

### 常见数据类型对照（仍基于 `MmadAtlasA2Pingpong`，L0B stages=2）

| 输入 dtype | sizeof | 常用 L1TileShape (M,N,K) | 常用 L0TileShape (M,N,K) | L0B 占用（*2 stages） | 参考样例 |
| ---------- | ------ | ------------------------ | ------------------------ | ----------------- | -------- |
| fp16 / bf16 | 2 | `<128,256,256>` | `<128,256,64>` | `64*256*2*2 = 64 KB` | [00_basic_matmul](../../../../../examples/00_basic_matmul/) |
| int8 | 1 | `<128,256,512>` | `<128,256,128>` | `128*256*1*2 = 64 KB` | [12_quant_matmul](../../../../../examples/12_quant_matmul/) |
| fp32 | 4 | `<128,128,256>` | `<128,128,64>` | `64*128*4*2 = 64 KB` | [矩阵乘优化指引 - TileShape约束 - 场景二](../../../1_Practice/11_matmul_optimization.md#tileshape约束) |

同容量下，元素越小，同样字节预算能放下的 `K`（或 `N`）越大。因此 **int8 样例的 Tile 不能原样套到 fp16**：若把 int8 的 `L0TileShape<128,256,128>` 直接用于 fp16，则

```text
L0B = 128 * 256 * 2 * 2 = 128 KB 大于 64 KB   # 超限
```

改 dtype 时至少按 `sizeof` 比例缩小 `L0.K` 或 `L0.N`，并同步缩放 `L1.K`（上例可改为 `L0.K=64`、`L1.K=256`，对应常用 fp16 组合）；若只改 L0 而保留 int8 的 `L1.K=512`，L1 占用会到 768KB 超过 512KB。改完后用同样公式复核 L1 / L0A / L0C。

### 速查步骤

1. 确认所用 **DispatchPolicy** 与 **BlockMmad**，读出 L1A/L1B、L0A/L0B、L0C 的 stages（或统一 `STAGES`），勿默认等于 2。
2. 选定 dtype，写出 `sizeof(Element)`；按「单份 Tile * 该级 stages」写占用式（与上式同形，仅替换 stages 系数）。
3. 约束 `L0.M/N/K`：L0A/L0B 不超 64KB，L0C 不超 128KB；再定 `L1.K`（pingpong 下常约 `4 * L0.K`），检查 L1 <= 512KB。
4. 对照该 BlockMmad 的额外约束（全载、非对称 stages、prologue/bias 占位等）修正后，再切换 dtype 或从其他样例拷贝 TileShape 并验算。

更完整的算例（含 fp16 / fp32 / int8 展开）与后续 tiling 调参见 [矩阵乘基础优化指引 - TileShape约束](../../../1_Practice/11_matmul_optimization.md#tileshape约束)；DispatchPolicy 字段含义见 [Dispatch 策略](../03_dispatch_policies.md)；Host 侧首次填写 TileShape 时可对照 [Host侧样例组装](../../../1_Practice/02_host_example_assembly.md)。
