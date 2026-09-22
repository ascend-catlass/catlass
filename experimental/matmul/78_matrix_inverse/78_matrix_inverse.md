# 矩阵求逆算子（78_matrix_inverse）设计文档

本文档用于说明 `experimental/matmul/78_matrix_inverse` 矩阵求逆算子示例所依赖的 Catlass GEMM 模板库能力、外部接口、分层设计方案。

## 功能说明

### 算子功能

| 项 | 说明 |
|------|------|
| 功能 | 计算 N×N 矩阵 A 的逆，A ← A⁻¹ |
| 布局 | 固定行主序（`layout::RowMajor`） |
| 数据类型 | 由模板参数 `Element` 决定（当前实例化为 `float`） |
| 输入 | 矩阵 A（GM）、主元数组 `ipiv`（int32） |
| 输出 | A 原位覆盖为 A⁻¹；`ipiv` 写出 LU 主元信息 |
| 算法 | 分块 GETRF（部分主元 LU）+ 分块 GETRI（TRTRI / ApplyLInverse / 列交换） |

核心实现位于 `include/catlass/gemm/kernel/matrix_inverse.hpp`，类 `Catlass::Gemm::Kernel::MatrixInverse`。

### 计算流程

基于 LU 分解与部分选主元（LU decomposition with partial pivoting）：

$$
\begin{aligned}
PA &= LU \\
A^{-1} &= U^{-1} L^{-1} P
\end{aligned}
$$

其中 $P$ 为排列矩阵，$L$ 为单位下三角矩阵，$U$ 为上三角矩阵。

**算法总览**：

```text
输入 A ──→ GETRF（LU 分解）──► LU ──→ GETRI（求逆）──► A⁻¹
```

计算分为四个阶段：

| 阶段 | 操作 | 说明 |
|------|------|------|
| GETRF | $PA = LU$ | LU 分解，含部分选主元 |
| TRTRI | $A \leftarrow U^{-1}$ | 上三角矩阵求逆（分块算法），结果存入 A |
| ApplyL | $A \leftarrow A \times L^{-1}$ | 应用 L 逆矩阵（分块算法） |
| SwapCols | $A^{-1} = A \times P$ | 按 P 置换列，得到最终逆矩阵 |

### 约束说明

- 输入矩阵必须是方阵（行数等于列数），即 $N \times N$
- 输入矩阵必须是非奇异矩阵（行列式不为零）
- 为保证数值稳定性，本样例生成对角占优（diagonally dominant）的随机测试矩阵

## 参数说明

### 运行参数

以下是本样例可执行文件的运行参数：

| 参数名 | 描述 | 约束 |
| ----- | -------- | ------ |
| `N` | 方阵的行/列数 | 正整数，矩阵必须非奇异 |
| `device_id` | 使用的 NPU 卡 ID（默认 0） | 在设备 NPU 有效范围内 |

### 模板参数

MatrixInverse kernel 模板定义：

```text
template <class ArchTag_, class Element_, class BlockMmad_, class BlockScheduler_>
class MatrixInverse
```

| 参数 | 说明 | 当前样例取值 |
| ----- | -------- | -------------- |
| `ArchTag_` | 硬件架构标签 | `Arch::AtlasA2` |
| `Element_` | 元素类型 | `float` |
| `BlockMmad_` | 块级 GEMM 计算单元（决定 L1/L0 分块与搬运策略） | `BlockMmadTla<...>` |
| `BlockScheduler_` | 块调度器（多核 tile 分发） | `GemmIdentityBlockSwizzle<>` |

### 参数结构

算子提供 Host 端 `Arguments` 与设备端 `Params` 两套一一对应的参数结构：

| 字段 | 类型（Params / Arguments） | 含义 |
| ------ | -------------------------- | ---- |
| `N` | `uint32_t` | 矩阵维度 N（N×N） |
| `ptrA` | `GM_ADDR` / `uint8_t*` | 矩阵 A 全局内存地址（输入兼输出） |
| `layoutA` | `LayoutA` | 矩阵布局描述 |
| `ptrIpiv` | `GM_ADDR` / `uint8_t*` | 主元索引数组地址（int32） |
| `ptrWorkspace` | `GM_ADDR` / `uint8_t*` | 工作空间地址 |

## 具体设计方案

### AIC / AIV 异构分工

| 阶段 | AIC（Cube 核，多核并行） | AIV（Vector 核，Core 0 串行） |
|------|------------------------|------------------------------|
| GETRF | `TrsmLeftLowerUnitGemm`（L21 消元）、`SchurGemmToWorkspace`（Schur 补） | `PanelGetrf`（面板 LU + 主元）、`ApplyRowSwaps`（行交换）、`ComputeInvLDiagGETRF`、Schur epilogue |
| GETRI | `TrsmTempGemmToWorkspace`、`InvertUpperTriGemmToWorkspace`、`DtrmmGemmToWorkspace`、`ApplyLInverseGemmToWorkspace`、`ApplyLIntraBlockGemm` | `TRTRIdiag`（对角块求逆）、`ComputeInvLDiag`、三角拷贝/取反/列交换 |

**设计要点**：把适合 Cube 单元的密集 GEMM 交给 AIC 多核并行加速，把逐列主元选取、向量化的标量消元、三角结构化拷贝交给 AIV。

### Host 层

#### 参数解析

`78_matrix_inverse` 命令执行参数：

```text
N, device_id
```

矩阵求逆仅需矩阵维度 N 和设备 ID，相比普通 GEMM 参数更简单。

#### 测试数据构造

host 侧生成对角占优的随机矩阵以确保数值稳定性：

1. 生成范围 $[-1.0, 1.0]$ 的随机数据填充矩阵
2. 将对角元素加强：`A[i][i] += N`，确保对角占优性质

对角占优矩阵具有良好的条件数，能保证 LU 分解的数值稳定性。

### Kernel 层

#### 并行化策略

矩阵求逆采用 AIV/AIC 混合并行：

| 执行单元 | 负责阶段 | 并行度 |
|---------|---------|--------|
| AIV core 0 | Panel LU、行交换、TRTRI 对角块、列交换 | 串行 |
| AIV 全核 | GETRF Schur epilogue、TRTRI epilogue、ApplyLInverse epilogue | 多核并行 |
| AIC 全核 | TRSM、Schur GEMM、TRTRI GEMM、ApplyLInverse GEMM | 多核并行 |

通过 AIV/AIC 协同，将 $O(N^3)$ 的 GEMM 计算剥离到 AIC，充分利用 Cube 加速能力。

#### Kernel 定义

MatrixInverse kernel 针对不同核心类型（AIC/AIV）有特化实现。

### AIC 核心设计

#### GETRF 阶段 AIC 职责

AIC 负责：

1. **TRSM**：用已计算的 $L_{diag}^{-1}$ 更新右侧矩阵
2. **Schur GEMM**：计算 Schur 补项 $A_{22} - L_{21}U_{12}$

#### TRTRI 阶段 AIC 职责

AIC 负责 TRTRI 的 GEMM 部分，将密集矩阵乘法剥离到 Cube。

#### ApplyLInverse 阶段 AIC 职责

AIC 负责：

1. 块间 GEMM：更新右侧列块
2. 块内 GEMM：应用当前块的 $L^{-1}$

### AIV 核心设计

#### GETRF 阶段 AIV 职责

**Core 0** 负责：

1. Panel LU 分解（含选主元）
2. 应用行交换到已处理和未处理区域
3. 计算 $L_{diag}$ 的逆

**全核** 负责 Schur epilogue：多核并行执行 `gmA -= gmInvLDense`

#### TRTRI 阶段 AIV 职责

**Core 0** 负责：

1. 复制 LU 到 workspace（L 严格下三角、U 上三角为稠密格式）
2. 对角块求逆（原地计算 $U_{diag}^{-1}$）
3. DTRMM 结果取反拷贝

**全核** 负责 epilogue：多核并行执行 `gmInvUDense -= gmGemmTemp`

#### ApplyLInverse 阶段 AIV 职责

**全核** 负责 epilogue：多核并行执行 `gmA -= gmInvU`

**Core 0** 负责计算 $L_{diag}$ 的逆

#### SwapColumns 阶段

**Core 0** 串行执行：按 pivot 信息的逆序交换列，完成 $A^{-1} = U^{-1}L^{-1}P$。

### DispatchPolicy 设计

#### MmadPingpong 模板参数含义

| 参数 | 说明 | 当前样例取值 |
|---|---|---|
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |
| `ENABLE_UNIT_FLAG` | 是否启用 unit 标记 | `true` |
| `useHF32` | 是否使用 HF32 | `false` |

## 空间分配

### Tile Shape

| 常量 | 取值 | 含义 |
|------|-----|------|
| NB | 64 | 面板分块大小，GETRF/GETRI 外层步进单位 |
| L1TileShape | 128×128×256 | L1 层 GEMM 分块（M×N×K） |
| L0TileShape | 128×128×64 | L0/Cube 层 GEMM 分块（M×N×K） |

Tile Shape 由 BlockMmad 模板参数决定，在示例代码中实例化指定。

### Workspace 布局

工作空间在 GM 上连续划分，总大小为 `2×N×N + 2×N×NB` 个元素字节：

| 区域 | 大小 | 用途 |
|------|-----|------|
| gmInvLDense | N×N | GETRF Schur GEMM / L 因子（ApplyLInverse 用） |
| gmInvUDense | N×N | U 因子 / invU（TRTRI 用） |
| gmGemmTemp | N×NB | 各阶段 GEMM 临时输出 |

## 同步与 Cache 一致性

### AIV/AIC 同步点

AIV 与 AIC 之间通过 `AscendC::SyncAll<false>()` 同步。每个 GETRF 迭代包含 4 个同步点（B1-B4），确保阶段间数据依赖正确。

### Cache 操作

硬件架构中缓存一致性需要软件显式维护。

部分关键 cache 操作位置：

1. AIV Core 0 PanelGetrf 后：flush gmA
2. AIV 全核 epilogue 后：flush gmA
3. AIV TRTRIdiag 后：flush gmInvUDense
4. AIV ComputeInvLDiag 后：flush gmGemmTemp

## 使用示例

### 编译

```bash
bash scripts/build.sh 78_matrix_inverse
```

### 运行

计算 128×128 矩阵的逆：

```bash
./78_matrix_inverse 128 0
```

计算 512×512 矩阵的逆：

```bash
./78_matrix_inverse 512 0
```

### 预期输出

成功执行输出：

```text
Matrix Inverse: N=128, device=0
Kernel time: X.XX ms
Compare success.
```
