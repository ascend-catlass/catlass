
# Conv优化设计

## 卷积到矩阵乘的映射

Block 层通过 Load3D 接口将卷积的输入展开为矩阵乘：

```text
M = 输出空间位置，主要对应 Ho * Wo
N = 输出通道 Cout
K = Cin * Kd * Kh * Kw
```

一个 L0 计算块可以理解为：

```text
A(M, K) * B(K, N) -> C(M, N)
```

其中：

- A 是 feature map 经 GM→AL1 布局转换和 Load3D 展开后得到的数据。
- B 是 weight 经 GM→BL1→L0B 转换后得到的数据。
- C 是 L0C 中的 FP32 累加结果，最后通过 Fixpipe 转成 FP16/BF16 并写回 NCDHW。

## 主要硬件流水

```text
Feature GM (NCDHW) --MTE2/Dn2Nz--> AL1 --MTE1/Load3D--> L0A
                                                       \
                                                        MMAD --> L0C --Fixpipe--> Output GM (NCDHW)
                                                       /
Weight GM (NCDHW)  --MTE2/Dn2Nz--> BL1 --MTE1--------> L0B
```

因此，Conv3D 的性能不只取决于 Cube 计算量，还取决于：

- GM→L1 搬运是否重复。
- Dn2Nz 描述符是否过于零碎。
- AL1/BL1 是否能重用。
- MTE2、MTE1、Cube 和 Fixpipe 是否能并行。
- 分核后是否存在空闲核或尾波不均衡。

## 1. A/B 矩阵驻留

A 代表 feature map，B 代表 weight。在卷积中，有较多的数据需要进行重复计算。启用驻留，则会极大的减少MTE2的搬运开销，是影响计算性能的重要因素。Host 侧会评估四种 L1 状态：

| 驻留状态 | 含义 | 主要收益 |
|---|---|---|
| `ResidentAB` | 当前单核工作区需要的 A 和 B 都能完整放入 L1 | 两边都不需要反复从 GM 加载 |
| `ResidentA` | A 能完整驻留，B 分段流式加载 | 同一 feature patch 可以服务多个 Cout tile |
| `ResidentB` | B 能完整驻留，A 分段流式加载 | 同一份 weight 可以服务多个 M/D tile |
| `StreamAB` | A/B 都需要分段加载 | 依靠双缓冲隐藏两边的搬运 |

## 2. 驻留与 stage 的联动

由于启用双缓冲会使AB分块的大小减半。因此保持开启双缓冲会使原本可以启用驻留的分块由于分块大小减半从而无法驻留。但驻留减少的MTE2搬运开销对性能的提升远大于启用双缓冲实现的性能提升。因此将双缓冲参数调整为host侧动态启用。

AL1 和 BL1 的 stage 是独立选择的：

- 某一边完整驻留时，优先使用 1-stage，把更多 L1 容量留给驻留数据。
- 某一边需要分段流式加载时，优先使用 2-stage，用 ping/pong 隐藏下一段 GM→L1 延迟。

因此常见组合是：

| 驻留 | 推荐 stage | 说明 |
|---|---|---|
| ResidentAB | A1/B1 | A/B 都已驻留，无需为下一段保留空闲缓冲 |
| ResidentA | A1/B2 | A 使用单份驻留，B 使用双缓冲 |
| ResidentB | A2/B1 | A 使用双缓冲，B 使用单份驻留 |
| StreamAB | A2/B2 | A/B 使用独立双缓冲 |

这种方式可以在驻留收益和编译规模之间取平衡。

## 3. M-mode 和 HW-mode

在结果矩阵的划分上，实现了两种模式。其中HWmode主要是将结果矩阵从HW两个维度上进行划分。而Mmode主要是将HW两个维度展平成一个维度进行划分。mmode的展平计算可以更好启用连续的数据搬运，减少搬运开销。

### HW-mode

HW-mode 在 Ho 和 Wo 两个维度上切矩形输出块。每个任务有明确的 `tileH` 和 `tileW`，Block 根据这个矩形计算所需的输入。

优点：

- 占用更少的AL1，M-mode搬运需要将计算涉及到的H行整个Wi搬运至L1A中，会增加额外的搬运字节数。而HW-mode则只需搬运对应的H行W列即可。
- 减少AL1的存储空间更有利于B能驻留计算

代价：

- 搬运不连续，对于跨行W采用的是非连续搬运，搬运时间开销大

### M-mode

M-mode 把输出空间位置看成矩阵乘的 M 维：

优点：

- AL1 尽量保留完整输入行，完整行可以进行连续搬运且启用更少的搬运指令
- 同样的，搬出Fixpipe 可以按连续 M 写回，减少搬运开销

代价：

- 占用较多AL1空间可能导致原本可以驻留的权重无法驻留，增加大量搬运时长。

## 4. MGroup与DGroup

MGroup 将相邻的多个空间微块交给同一个 AIC 连续处理。让 Kernel 的一个 work region 包含多个仍然由 Block 分别计算的 M tile。这样有利于权重复用，当 weight 已经完整驻留 BL1 时，连续 M tile 可以共享同一份 B。

DGroup 将相邻的多个 Dout 微块交给同一个 AIC 连续处理。当 B 驻留时，多个输出深度位置可以共享同一份 weight。

## 5.M-first与N-first

M-fitst：单核块间计算时，当前 N/BL1 块保持不变，先遍历多个 M 和 D 位置。因此B 驻留时优先选 M-first，减少 weight 从 GM 到 BL1 的重复搬运。

N-first：单核块间计算时，当前 M/AL1 块保持不变，先遍历多个 N/Cout 位置。因此A 驻留时优先选 N-first，减少 feature map 从 GM 到 AL1 的重复搬运。

## 6.动态Tiling

CANN基线aclnnConvolution的Tiling如下：
```text
输入 shape 和属性
        │
        ▼
查询进程内 Tiling Cache
        │未命中
        ▼
查询 Runtime Knowledge Base
        │未命中
        ▼
执行公式化 Fast Tiling
        │
        ├─ 选择 M-mode / HW-mode
        ├─ 选择分核方式
        ├─ 选择 L0 tile
        ├─ 判断 A/B 驻留
        ├─ 选择 L1 tile
        ├─ 选择 M-first / N-first
        └─ 选择 double buffer
```

由于是否驻留可以节省较大的搬运开销，因此使用动态tiling调整AB分块大小可以有效调整驻留空间大小，将不同shape的情况进行动态调整优化。

在进行Tiling策略选择时还会计算不同情况下的计算代价，通过搬运量，带宽，以及计算量从多个角度综合考虑应该使用哪一种分核方式以及策略。因此CATLASS设计中也应该采用动态Tiling来实现探寻shape间的最优优化策略。

## 7.Basic / split-K 分核调度

**Basic 路径**：任务空间 = `ceil(m/M_TILE) × ceil(n/N_TILE)` 的 M×N 网格。每个任务是一个 (mTile, nTile) 块，核认领后在块内部把整个 K 循环走完。适合 M×N 分块数已经足够多的 shape——并行度高，且没有跨任务归约。

**split-K 路径**：任务空间 = `ceil(m/M_TILE) × ceil(n/N_TILE) × splitkFactor`，K 被切成 `splitkFactor` 份。每个任务是一个 (mTile, nTile, kSlice)，各算一个 K 片段的 partial sum，然后原子加累加进 C。适合 M×N 分块数少的 shape（例如 cout 小、cin 小的点），用 K 方向切片把核填满。

## 8.Conv3d DGrad的dY 和 W 的数据格式处理

算子的输入是 NCHW 格式的 dY 和 OIHW 格式的 W，但 Cube 计算时更适合使用按通道分块后的数据。因此，在正式计算前，需要选择合适的数据准备方式。

当前实现提供两种方式：

- **Direct**：不提前转换整份数据。计算当前分块时，只读取需要的部分，在L1中进行随路转换。
- **Packed**：先在UB将数据转换为更适合 Cube 使用的排列，保存回GM中，后续计算直接读取。

dY 和 W 可以分别选择处理方式，因此形成以下四种组合：

| dY | W | 适用情况 | 选择原因 |
|---|---|---|---|
| Direct | Direct | 计算量较小、Shape 不规则，或者只需要部分数据 | 不生成输入临时空间，只处理当前分块真正需要的数据 |
| Direct | Packed | dY 适合按分块处理，而 W 会被多个空间分块重复使用 | dY 不做全量转换，W 只转换一次并重复使用 |
| Packed | Direct | dY 会被多个通道分块重复使用，而 W 较小或只需要部分卷积核位置 | 减少 dY 的重复整理，W 则按实际需要读取 |
| Packed | Packed | Shape 规则、计算任务较多，dY 和 W 都会被多次使用 | 前期转换成本可以由后续大量计算共同分担 |

Direct 的优点是准备过程简单、workspace 较小；不足是不同计算分块可能重复进行边界判断、补零和数据整理。
Packed 会增加一次数据转换和 workspace 读写，但后续计算能够直接读取排列规则的数据。当同一份数据会被多次使用时，这种方式通常更有优势。
选择的基本原则是：如果数据转换一次后能够被多次使用，就优先考虑 Packed；如果数据使用次数较少，或者只需要其中一小部分，就优先考虑 Direct。
