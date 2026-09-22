# 对称矩阵乘算子（symm）设计文档

本文档用于说明 `./examples/75_symm` 对称矩阵乘算子示例所依赖的 Catlass GEMM 模板库能力、外部接口、分层设计方案。

## 功能说明

 - 算子功能：完成对称矩阵乘计算。当左操作数或右操作数为对称矩阵时，仅存储上三角或下三角作为有效数据，另一半通过对称性推导得到，从而减少访存。
 - 计算公式：

$$
    \begin{aligned}
    \text{Left:  } C_{M \times N} &= A_{M \times K} \times B_{K \times N}, \quad A = A^T, \quad M = K \\
    \text{Right: } C_{M \times N} &= B_{M \times K} \times A_{K \times N}, \quad A = A^T, \quad K = N
    \end{aligned}
$$

  其中左乘时 `A` 为对称矩阵（shape `M x K`，满足 `M == K`），右乘时 `A` 为对称矩阵（shape `K x N`，满足 `K == N`）。对称矩阵有效三角区域由 `symm_fill` 参数指定：`UPPER` 表示上三角有效，`LOWER` 表示下三角有效。

 - 四种模式：

| side | fill | 数学语义 | shape 约束 |
|---|---|---|---|
| LEFT | UPPER | `C = A * B`，`A` 上三角有效 | `M == K` |
| LEFT | LOWER | `C = A * B`，`A` 下三角有效 | `M == K` |
| RIGHT | UPPER | `C = B * A`，`A` 上三角有效 | `K == N` |
| RIGHT | LOWER | `C = B * A`，`A` 下三角有效 | `K == N` |

## 参数说明

以下是本样例可执行文件的运行参数：

| 参数名 | 描述 | 约束 |
| ----- | -------- | ------ |
| `m` | 输出矩阵 C 的行数 | 正整数 |
| `n` | 输出矩阵 C 的列数 | 正整数 |
| `k` | GEMM 归约维度 | 正整数 |
| `device_id` | 使用的 NPU 卡 ID（默认 0） | 在设备 NPU 有效范围内 |
| `symm_side` | 对称矩阵所在侧：`0=LEFT`，`1=RIGHT`（默认 0） | 0 或 1 |
| `symm_fill` | 有效三角区域：`0=LOWER`，`1=UPPER`（默认 1） | 0 或 1 |

SymmMatmul 所涉及的关键模板参数如下：

| 模板参数 | 说明 | 当前样例取值 |
| ----- | -------- | -------------- |
| `ElementA` | 左矩阵数据类型 | `float` |
| `ElementB` | 右矩阵数据类型 | `float` |
| `ElementC` | 输出矩阵数据类型 | `float` |
| `LayoutA` | 左矩阵排布方式 | `layout::RowMajor` |
| `LayoutB` | 右矩阵排布方式 | `layout::RowMajor` |
| `LayoutC` | 输出矩阵排布方式 | `layout::RowMajor` |
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |

## 约束说明

 - 输入矩阵均为 fp32，对称矩阵的无效三角区域在 host 侧通过镜像补全。
 - 左乘要求 `M == K`，右乘要求 `K == N`，这是因为对称矩阵必须是方阵。
 - 对称矩阵通过 `tla::MakeLayout` 构造 RowMajor（direct 路径）和 ColumnMajor（transpose 路径）两套 TLA layout，kernel 侧根据 tile 位置在两条路径间切换。
 - side 和 fill 作为编译期模板参数确定，避免 device 端运行时分支。

## 具体设计方案

### Host 层

#### 参数解析

`symm` 命令执行参数：

```text
m, n, k, device_id, symm_side, symm_fill
```

原因是对称矩阵乘不仅有矩阵乘本身的 shape，还需要描述：

1. 对称矩阵在左边还是右边；
2. 对称矩阵有效存储区域是上三角还是下三角。

因此 host 层需要将运行时输入映射为四种编译期 kernel 实例：

```text
LEFT  + UPPER
LEFT  + LOWER
RIGHT + UPPER
RIGHT + LOWER
```

#### shape 校验

普通 GEMM 只需要满足：

```text
A: M x K
B: K x N
C: M x N
```

对 `M`、`N`、`K` 之间没有额外相等关系。

对称矩阵乘需要额外校验：

- `LEFT`：`M == K`
- `RIGHT`：`K == N`

这是因为对称矩阵必须是方阵。

#### host 数据构造

`symm` 生成随机数据后，还需要根据 side/fill 对对称矩阵做镜像补全。

左乘时，对称矩阵是 `hostA`：

- upper 模式：保留上三角，用上三角补下三角；
- lower 模式：保留下三角，用下三角补上三角。

右乘时，对称矩阵是 `hostB`：

- upper 模式：保留上三角，用上三角补下三角；
- lower 模式：保留下三角，用下三角补上三角。

### Kernel 层

#### 对称矩阵乘 kernel

`symm` 使用统一的对称矩阵乘 kernel producer：

```cpp
Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<
    Side,
    FillMode,
    BlockMmad,
    BlockScheduler>
```

它与 `BasicMatmul` 的主要区别是：

1. **增加 side/fill 编译期参数**

   `Side` 决定对称矩阵在左操作数还是右操作数。

   `FillMode` 决定有效三角区域是 upper 还是 lower。

2. **根据 side 构造不同 layout**

   左乘：

   ```text
   LayoutAP: 对称矩阵 A 的 direct 路径布局
   LayoutAQ: 对称矩阵 A 的 transpose 路径布局
   LayoutB : 普通矩阵 B 布局
   ```

   右乘：

   ```text
   LayoutA : 普通矩阵 B 布局，即代码中的左操作数 A
   LayoutBP: 对称矩阵 A 的 direct 路径布局
   LayoutBQ: 对称矩阵 A 的 transpose 路径布局
   ```

3. **调用 block 接口**

   左右乘均通过 `operator()` 调用，block 内部以参数顺序区分左右：

   左乘（先传对称矩阵，再传非对称矩阵）：

   ```cpp
   blockMmad.template operator()<UPPER_STORAGE>(
       tensorBlockSym, tensorBlockSymQ, tensorBlockNonSym, tensorBlockC, ...);
   ```

   右乘（先传非对称矩阵，再传对称矩阵）：

   ```cpp
   blockMmad.template operator()<UPPER_STORAGE>(
       tensorBlockNonSym, tensorBlockSym, tensorBlockSymQ, tensorBlockC, ...);
   ```

### Block 层

#### 左侧对称 block

左侧对称使用：

```cpp
Gemm::Block::BlockMmadPingpongSymmLeftTla
```

它针对 `C = A * B` 设计，其中 `A` 位于 A 侧。核心区别是：

1. **A 侧有 direct / transpose 两条读取路径**

   - `LayoutAP`：按 A 的普通行主序 direct 读取；
   - `LayoutAQ`：按列主序/转置语义读取对称位置。

2. **根据 `iTile` 和 `kTile` 判断读取路径**

   对于左乘，输出 block 的行 tile 为 `iTile`，归约 tile 为 `kTile`。对称矩阵访问的是：

   ```text
   A(iTile, kTile)
   ```

   upper 模式下：

   - `kTile >= iTile`：位于上三角或对角，direct 读取；
   - `kTile < iTile`：位于下三角，需要转置到上三角对应位置读取。

   lower 模式下：

   - `kTile <= iTile`：位于下三角或对角，direct 读取；
   - `kTile > iTile`：位于上三角，需要转置到下三角对应位置读取。

3. **对角 tile 的 L1 内补全**

   对角 tile 同时包含上三角和下三角。由于 valid fill 只保证一半三角有效，因此加载对角 tile 后需要在 L1 内补全另一半：

   - upper：用 `A(c,r)` 补 `A(r,c)`；
   - lower：用 `A(r,c)` 补 `A(c,r)`。

   这样 MMAD 看到的是完整对称 tile，不需要在 MMAD 内部做三角判断。

#### 右侧对称 block

右侧对称使用：

```cpp
Gemm::Block::BlockMmadPingpongSymmRightTla
```

它针对 `C = B * A` 设计，其中 `A` 位于 B 侧。核心区别是：

1. **B 侧有 direct / transpose 两条读取路径**

   - `BTypeP/LayoutBP`：按右侧对称矩阵的普通方向读取；
   - `BTypeQ/LayoutBQ`：按转置方向读取对称位置。

2. **根据 `jTile` 和 `kTile` 判断读取路径**

   对于右乘，输出 block 的列 tile 为 `jTile`，归约 tile 为 `kTile`。对称矩阵访问的是：

   ```text
   A(kTile, jTile)
   ```

   upper 模式下：

   - `kTile <= jTile`：位于上三角或对角，direct 读取；
   - `kTile > jTile`：位于下三角，需要转置到上三角对应位置读取。

   lower 模式下：

   - `kTile >= jTile`：位于下三角或对角，direct 读取；
   - `kTile < jTile`：位于上三角，需要转置到下三角对应位置读取。

3. **对角 tile 的 L1 内补全**

   与左乘类似，对角 tile 需要在 L1 内完成无效半边补全，以保证后续 L0/MMAD 仍使用完整密集 tile。

4. **为什么右乘需要独立 block**

   左乘对称性作用在 A 侧，影响 L1A/L0A 的搬运和 layout；右乘对称性作用在 B 侧，影响 L1B/L0B 的搬运和 layout。

   虽然判断逻辑形式相似，但实际涉及的 buffer、layout、copy primitive、MMAD operand 类型不同。因此 block 层保留两个实现更清晰，也更便于分别调试和优化。

### DispatchPolicy 设计

#### MmadPingpongSymmLeft/Right模板参数含义

| 参数 | 说明 | 当前样例取值 |
|---|---|---|
| `ArchTag` | 目标架构，控制硬件特性和指令选择 | `Arch::AtlasA2` |
| `ENABLE_UNIT_FLAG` | 是否启用 unit 标记（L0C 写回时的尾块标记） | `true`（默认 `false`） |

两个 dispatch policy 内部统一硬编码 `STAGES = 2`，所有 buffer（L1A、L1B、L0A、L0B）共用同一份 pingpong 级数。

#### Pingpong 流水

SYMM 沿用与普通 GEMM 一致的四级流水结构，对称矩阵的 direct/transpose 路径选择和跨 block 预取均集成在 MTE2 preload 阶段中。流水流程如下：

```text
MTE2:  GM  → L1       预取阶段，含以下两步决策：
         ├─ 对称矩阵 direct/transpose 路径选择：根据 iTile/kTile（或 jTile/kTile）判断当前
         │    tile 位于上三角还是下三角，选择 direct（RowMajor）或 transpose（ColumnMajor）
         │    路径从 GM 搬运对称矩阵到 L1；
         │   - 对角 tile 搬运后在 L1 内完成对称补全（用有效三角补无效三角）
         ├─ 跨 block preload：当前 block 最后一个 k-tile 预取时，若 hasNextBlock 为真，
         │   则转而预取下一个 block 的第一个 k-tile，隐藏跨 block 的 GM→L1 延迟。
MTE1:  L1  → L0       将 L1 tile 拆分为 L0 tile 搬运到计算 buffer
M:     L0  → MMAD     矩阵乘累加（此时 L0 中已是完整密集 tile，无需感知对称性）
FIX:   L0C → GM       结果写回全局内存
```

pingpong 的级数由 dispatch policy 硬编码的 `STAGES = 2` 决定。Block 内为每级维护独立的 buffer 和硬件 event：

```text
l1ATensorList[STAGES]    l1AEventList[STAGES]
l1BTensorList[STAGES]    l1BEventList[STAGES]
l0ATensorList[STAGES]    l0AEventList[STAGES]
l0BTensorList[STAGES]    l0BEventList[STAGES]
```

event 按流水方向配对，实现阶段间同步与切换：

- `l1AEvent` / `l1BEvent`：MTE2 ↔ MTE1（GM→L1 完成后通知 L1→L0 可消费）
- `l0AEvent` / `l0BEvent`：M ↔ MTE1（L1→L0 完成后通知 MMAD 可消费，MMAD 完成后通知可写下一轮）
- 对角 tile 在 L1 补全完成后才释放 event 给 MTE1，确保 MTE1 消费的是完整对称 tile。
- Kernel `operator()` 末尾调用 `PipeBarrier<PIPE_ALL>()` 等待所有 block 的流水线排空后才返回。

## 空间分配

### Tile Shape 设计

左乘 SYMM tile：

```cpp
using L1TileShape = Shape<128, 256, 128>;
using L0TileShape = Shape<128, 256, 32>;
```

左乘要求 `L1TileShape::M == L1TileShape::K`（M=K=128 方形 tile），这是因为左侧对称矩阵位于 `M x K` 平面，只有 M tile 和 K tile 对齐为方形才能用 `iTile/kTile` 正确判定三角位置。

右乘 SYMM tile：

```cpp
using L1TileShape = Shape<256, 128, 128>;
using L0TileShape = Shape<256, 128, 32>;
```

右乘要求 `L1TileShape::K == L1TileShape::N`（K=N=128 方形 tile），因为右侧对称矩阵位于 `K x N` 平面。右乘将 M tile 扩大到 256，在 M 较大、K=N 中等的场景（如 M=4096）可将 block 数量从 192 减半到 96。

### 存储空间计算

数据类型为 float（4 字节），pingpong STAGES = 2，硬件 buffer 容量为 L1 = 512 KB、L0A = 64 KB、L0B = 64 KB、L0C = 128 KB。

**左乘 <128, 256, 128> / <128, 256, 32>：**

| Buffer | 单级尺寸 | 计算 | 单级容量 | ×2 stages | 硬件上限 | 利用率 |
|---|---|---|---|---|---|---|
| L1A | 128 × 128 | × 4B | 64 KB | — | — | — |
| L1B | 256 × 128 | × 4B | 128 KB | — | — | — |
| **L1 合计** | — | — | 192 KB | **384 KB** | 512 KB | **75%** |
| L0A | 128 × 32 | × 4B | 16 KB | **32 KB** | 64 KB | **50%** |
| L0B | 32 × 256 | × 4B | 32 KB | **64 KB** | 64 KB | **100%** |
| L0C | 128 × 256 | × 4B | 128 KB | — | 128 KB | **100%** |

**右乘 <256, 128, 128> / <256, 128, 32>：**

| Buffer | 单级尺寸 | 计算 | 单级容量 | ×2 stages | 硬件上限 | 利用率 |
|---|---|---|---|---|---|---|
| L1A | 256 × 128 | × 4B | 128 KB | — | — | — |
| L1B | 128 × 128 | × 4B | 64 KB | — | — | — |
| **L1 合计** | — | — | 192 KB | **384 KB** | 512 KB | **75%** |
| L0A | 256 × 32 | × 4B | 32 KB | **64 KB** | 64 KB | **100%** |
| L0B | 32 × 128 | × 4B | 16 KB | **32 KB** | 64 KB | **50%** |
| L0C | 256 × 128 | × 4B | 128 KB | — | 128 KB | **100%** |

两种模式下 L1 占用 384/512 KB（75%），L0C 均打满 128 KB（100%），L0A/L0B 分别在一种模式下达到 100% 利用率。当前 tile shape 已将各硬件 buffer 推至接近极限，达到硬件最优配置。

此外，对称矩阵乘只将上三角（或下三角）作为有效数据来源：direct 路径从有效三角区域直读，transpose 路径将位于无效三角区域的 tile 重定向到转置后的有效三角位置读取。这意味着 GM 中对称矩阵的实际 "热数据" 仅集中在有效三角区域（约为完整矩阵的一半），缩小了 GM→L1 搬运时的数据工作集，提高了 L2 cache 的命中率。

## 使用示例

### 编译

```bash
bash scripts/build.sh 75_symm
```

### 运行

左乘，上三角：

```bash
.output/bin/75_symm 768 4096 768 0 0 1
```

左乘，下三角：

```bash
.output/bin/75_symm 768 4096 768 0 0 0
```

右乘，上三角：

```bash
.output/bin/75_symm 4096 768 768 0 1 1
```

右乘，下三角：

```bash
.output/bin/75_symm 4096 768 768 0 1 0
```
