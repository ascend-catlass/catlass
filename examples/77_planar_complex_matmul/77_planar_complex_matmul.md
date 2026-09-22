# PlanarComplexMatmul

## 功能说明

 - 算子功能：完成平面复数矩阵乘计算。复数矩阵以实部、虚部分离的 planar complex 形式输入，样例输出实部与虚部两路结果。
 - 计算公式：

$$
    \begin{aligned}
    A &= A_{real} + iA_{imag} \\
    B &= B_{real} + iB_{imag} \\
    C &= A \times B \\
    C_{real} &= A_{real} \times B_{real} - A_{imag} \times B_{imag} \\
    C_{imag} &= A_{real} \times B_{imag} + A_{imag} \times B_{real}
    \end{aligned}
$$

  其中 `A_real`、`A_imag` 是形如 `(m, k)` 的左矩阵实部和虚部，`B_real`、`B_imag` 是形如 `(k, n)` 的右矩阵实部和虚部，`C_real`、`C_imag` 是形如 `(m, n)` 的输出矩阵实部和虚部。

## 参数说明

以下是本样例可执行文件的运行参数：

| 参数名 | 描述 | 约束 |
| ----- | -------- | ------ |
| `m` | 复数矩阵乘中左矩阵 A 的行 | 正整数 |
| `n` | 复数矩阵乘中右矩阵 B 的列 | 正整数 |
| `k` | 复数矩阵乘中左矩阵 A 的列，也即右矩阵 B 的行 | 正整数 |
| `deviceId` | 使用的 NPU 卡 ID（默认 0） | 在设备 NPU 有效范围内 |
| `--datapath` | 输入数据与输出数据目录 | 可选；未指定时仅随机生成输入并统计 kernel 耗时 |

PlanarComplexMatmul 所涉及的关键模板参数如下：

| 模板参数 | 说明 | 当前样例取值 |
| ----- | -------- | -------------- |
| `ElementA` | 左矩阵实部/虚部的数据类型 | `half` |
| `ElementB` | 右矩阵实部/虚部的数据类型 | `half` |
| `ElementC` | 输出矩阵实部/虚部的数据类型 | `float` |
| `LayoutA` | 左矩阵排布方式 | `layout::RowMajor` |
| `LayoutB` | 右矩阵排布方式 | `layout::ColumnMajor` |
| `LayoutC` | 输出矩阵排布方式 | `layout::RowMajor` |
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |
| `L1TileShape` | L1 tile 形状 | `tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<256>>` |
| `L0TileShape` | L0 tile 形状 | `tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<64>>` |
| `DispatchPolicy` (Four-Pass) | 4-pass 路径的 MMAD 调度策略 | `Gemm::MmadPingpong<ArchTag, true>` |
| `DispatchPolicy` (Fused) | Fused 路径的 MMAD 调度策略 | `Gemm::MmadPlanarComplexFused<ArchTag, true>` |

## 约束说明

 - 输入矩阵实部和虚部均为 fp16，输出实部和虚部均为 fp32。
 - `B_real`、`B_imag` 在设备侧按 `layout::ColumnMajor` 读取；使用 `gen_data_compare.py` 校验时脚本会将 NumPy 生成的 B 矩阵转置后写入输入文件。
 - 样例根据 shape 在 Host 侧选择执行路径：当 `k >= 6000` 且每个 AIC core 分到的 MN tile 数不少于 3 时选择 Four-Pass，否则选择 Fused。
 - 样例根据 `m` 与 `n` 的关系选择对 `A_imag` 或 `B_imag` 取负后写入 workspace，用于计算 `C_real` 中的负号项。

## 具体设计方案

### Host 层

#### 参数解析

`77_planar_complex_matmul` 命令执行参数：

```text
m, n, k, [device_id], [--datapath DATA_PATH]
```

Host 层除了常规的 GEMM shape，还需要根据m, n, k,动态选择路径：

1. 选择执行路径（Four-Pass 或 Fused）；
2. 选择对 `A_imag` 还是 `B_imag` 取负（`NEGATE_A`）；
3. 选择 block swizzle 方向（`m >= n` 时行优先扫描，`m < n` 时列优先扫描）。

#### 路径选择

Host 侧基于 cost-model 选择 kernel 变体：

```text
K >= 6000 AND per_core >= 3 tiles  -> Four-Pass
否则                                -> Fused
```

- `coreLoops = CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N)`
- `perCore = coreLoops / aicCoreNum`

Four-Pass 将 L2 工作集锁定为 2 个矩阵（每 pass 只读写一路 C），适合 K 大、per-core tile 多的场景；Fused 单遍完成，适合 K 小或 per-core tile 少的场景。

#### NEGATE_A 选择

`C_real = A_real * B_real - A_imag * B_imag` 中的负号项通过预先对 `A_imag` 或 `B_imag` 取负实现：

- `m < n`：对 `A_imag` 取负（`NEGATE_A=true`），workspace 尺寸为 `M*K*half`；
- `m >= n`：对 `B_imag` 取负（`NEGATE_A=false`），workspace 尺寸为 `K*N*half`。

选择较小的一侧可以减少 workspace 开销和 AIV 取负工作量。

#### device memory 与 copy

设备内存分配：

```text
deviceAReal, deviceAImag          // 输入 A 实部/虚部
deviceBReal, deviceBImag          // 输入 B 实部/虚部
deviceCReal, deviceCImag          // 输出 C 实部/虚部
deviceWorkspace                   // AIV 取负后的 signed 工作区
```

Host 通过 `aclrtGetHardwareSyncAddr` 获取硬件同步地址 `hardwareSyncAddr`，用于 Mix kernel 内 AIV<->AIC 跨核同步。Host 将 problemShape、4 路输入指针、2 路 signed 占位（`nullptr`，由 kernel 层根据 `NEGATE_A` 替换为 workspace 指针）、2 路输出指针传入 `PlanarKernel::Arguments`，再通过 `DeviceGemm<PlanarKernel>` 启动 kernel：

```cpp
using DeviceOp = Gemm::Device::DeviceGemm<PlanarKernel>;
DeviceOp op;
op.Initialize(arguments, deviceWorkspace);
op(stream, aicCoreNum, hardwareSyncAddr);   // hardwareSyncAddr 用于跨核同步
```

### Kernel 层

#### 统一 kernel 模板

```cpp
Gemm::Kernel::PlanarComplexGemm<
    USE_FOUR_PASS,
    NEGATE_A,
    BlockMmadFourPass,
    BlockMmadFused,
    BlockScheduler>
```

kernel 通过 `USE_FOUR_PASS` 编译期开关选择 block 类型，未选中的路径以 `void` 传入，不会被实例化。

#### AIV 取负预处理

`NegateMatrixAiv` 是 kernel 内的 AIV 组件：

- 输入：原始 `A_imag`（或 `B_imag`）GM tensor；
- 输出：取负后的 signed GM workspace；
- 单缓冲：src/dst 各占 UB 的一半，compute（Muls）相对 GM 带宽可忽略，双缓冲无收益。

AIV 路径在 Mix kernel prologue 阶段执行取负，AIC 路径通过 `ptrAImagSigned`/`ptrBImagSigned` 消费结果。

#### Four-Pass 编排（`USE_FOUR_PASS=true`）

4 次顺序 `BlockMmad` 调用，fixpipe atomic-add 把交叉项累加回 C：

```text
pass1: C_real  = A_real * B_real              (无 atomic)
pass2: C_real += signed_imag_cross_term       (atomic add)
pass3: C_imag  = A_imag * B_real              (无 atomic)
pass4: C_imag += A_real * B_imag              (atomic add)
```

其中 pass 2 的 `signed_imag_cross_term` 为：

\- `NEGATE_A = true`：`A_imag_signed * B_imag`

\- `NEGATE_A = false`：`A_imag * B_imag_signed`

#### Fused 编排（`USE_FOUR_PASS=false`）

单遍 K-loop，C_real 与 C_imag 分时复用同一块 L0C：

```text
Stage 1 (C_real): 2K 个子迭代
  even sub: A_real * B_real           -> l0C (initC)
  odd  sub: AImagSigned * BImagSigned -> l0C (accumulate)
  ... FixPipe l0C -> GM_C_real

Stage 2 (C_imag): 2K 个子迭代
  even sub: A_imag * B_real -> l0C (initC)
  odd  sub: A_real * B_imag -> l0C (accumulate)
  ... FixPipe l0C -> GM_C_imag
```

C_real 的 FixPipe 与 C_imag 首个子迭代的 MTE2（GM->L1）重叠，隐藏 fixpipe 延迟。

### Block 层

#### Four-Pass block

Four-Pass 复用通用 BlockMmadTla。kernel 层负责 4 次调用的编排和 atomic-add。

#### Fused block

Fused 使用 `BlockMmadTla` 针对 `MmadPlanarComplexFused` policy 的偏特化（`block_mmad_planar_complex_fused_tla.hpp`）：

1. **4 路输入 tensor**：`A_real`、（`A_imag` or `A_imag_signed`）、`B_real`、(`B_imag` or `B_imag_signed`)。block 对 `NEGATE_A` 无感知，始终从 Signed 槽位读 C_real 交叉项，从原始槽位读 C_imag。
2. **L1 4 槽 K-pingpong**：`[A_K0 | A_K1 | B_K0 | B_K1]`，A/B 槽位通用，GM 来源按子迭代交替。
3. **L0A/L0B 双缓冲 pingpong**：重叠 L1->L0 搬运与 Cube MMAD。
4. **L0C 单缓冲**：C_real FixPipe 完成后 C_imag 才开始，分时复用。
5. **K-shuffle**：`ENABLE_SHUFFLE_K=true` 时按 `GetBlockIdx()` 偏移 K tile 顺序，分散 L2 访问热点。

### DispatchPolicy 设计

#### Fused: `MmadPlanarComplexFused`

模板参数：

| 参数 | 说明 | 当前样例取值 |
|---|---|---|
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |
| `ENABLE_SHUFFLE_K` | 是否启用 K 维 shuffle | `true` |

## 空间分配

### Tile Shape 设计

```cpp
using L1TileShape = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<256>>;  // M, N, K
using L0TileShape = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<64>>;   // M, N, K
```

### 存储空间计算

数据类型：`ElementA = ElementB = half`（2 字节），`ElementAccumulator = float`（4 字节），pingpong `STAGES = 2`。硬件 buffer 容量：`L1 = 512 KB`、`L0A = 64 KB`、`L0B = 64 KB`、`L0C = 128 KB`。

**Fused 路径 `<128, 256, 256> / <128, 256, 64>`：**

| Buffer | 单级尺寸 | 计算 | 单级容量 | ×2 stages | 硬件上限 | 利用率 |
|---|---|---|---|---|---|---|
| L1A | 128 × 256 | × 2B | 64 KB | 128 KB | - | - |
| L1B | 256 × 256 | × 2B | 128 KB | 256 KB | - | - |
| **L1 合计** | - | - | 192 KB | **384 KB** | 512 KB | **75%** |
| L0A | 128 × 64 | × 2B | 16 KB | **32 KB** | 64 KB | **50%** |
| L0B | 64 × 256 | × 2B | 32 KB | **64 KB** | 64 KB | **100%** |
| L0C | 128 × 256 | × 4B | 128 KB | - (单缓冲) | 128 KB | **100%** |

L0C 单缓冲（C_real/C_imag 分时复用）打满 128 KB；L0B 双缓冲打满 64 KB；L1 合计 384/512 KB（75%）。L0A 利用率 50%，留出余量给对角 tile 等特殊处理。

**GM workspace（AIV 取负）：**

- `NEGATE_A=true`（`m < n`）：`M × K × 2B`（对 `A_imag` 取负）
- `NEGATE_A=false`（`m >= n`）：`K × N × 2B`（对 `B_imag` 取负）

Host 选择较小的一侧取负以减少 workspace 开销。
