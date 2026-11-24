# IterBatchBroadcastMatmul Example Readme

> **注意**：本样例位于 `experimental/` 目录下，未登记在 `examples/CMakeLists.txt` 的
> `EXAMPLE_ASCEND950` 列表中，`catlass_examples` 全量构建不包含本样例。单独编译运行时，先将样例目录复制到
> `examples/`，在 `EXAMPLE_ASCEND950` 列表中加入 `85_ascend950_iterbatch_broadcast_matmul`，并通过
> `-DCATLASS_ARCH=3510` 指定 Ascend 950 架构。
>
> 本样例的 optest 测试用例保存在当前目录，不会被全局 CI 自动收集；jit Kernel 源码在
> `tests/optest/kernels/85_ascend950_iterbatch_broadcast_matmul/`，与本目录样例源码相互独立。

## 功能说明

- 算子功能：实现广播批量矩阵乘，把 batch 维的 B 个矩阵乘打包执行，其中 A 或 B 可以只有一份并在 batch 维广播（broadcast）——对每个输出 batch 计算 $C_b = A_{b_a} \times B_{b_b} + \mathrm{Bias}_b$，允许 A 或 B 只有一份（batch 维广播）、被全部 batch 共享。
- 计算公式：

  $$
    \begin{aligned}
    C_{b} &= \mathrm{cast}\big(\mathrm{op}(A_{b_a}) \cdot \mathrm{op}(B_{b_b}) + \mathrm{Bias}_{b}\big) \\
    b_a &= \begin{cases} 0, & \mathrm{batchA} = 1 \\ b, & \mathrm{batchA} = \mathrm{batchCount} \end{cases} \\
    b_b &= \begin{cases} 0, & \mathrm{batchB} = 1 \\ b, & \mathrm{batchB} = \mathrm{batchCount} \end{cases}
    \end{aligned}
  $$

  符号与命令行参数对照：

  | 公式符号 | 含义 | 命令行参数 |
  | --- | --- | --- |
  | $b$ / batchCount | 输出 batch 索引 / 总数 | `batch_count` |
  | batchA | A 的 batch 数，取 1 或 batchCount | `broadcast_a`（1 = 广播） |
  | batchB | B 的 batch 数，取 1 或 batchCount | `broadcast_b`（1 = 广播） |
  | $\mathrm{Bias}_b$ | 共享 `[N]` 或逐 batch `[batchCount, N]` | `bias_mode`（1 / 2） |
  | $\mathrm{op}$ | 可选转置，由输入布局决定；可执行样例固定 RowMajor 不转置 | — |

- 手算示例（batchCount=2，B 广播，共享 bias，M=N=K=2）：

  设 $A_0=\begin{bmatrix}1&2\\3&4\end{bmatrix}$、$A_1=\begin{bmatrix}5&6\\7&8\end{bmatrix}$、$B=\begin{bmatrix}1&1\\1&1\end{bmatrix}$、$\mathrm{Bias}=\begin{bmatrix}1&2\end{bmatrix}$，则

  $$
  C_0 = A_0 B + \mathrm{Bias} = \begin{bmatrix}4&5\\8&9\end{bmatrix},\quad
  C_1 = A_1 B + \mathrm{Bias} = \begin{bmatrix}12&13\\16&17\end{bmatrix}
  $$

  同一份 B 参与 b=0 与 b=1 两次计算。若为逐 batch bias（`bias_mode=2`），则 b=0 加 `Bias[0]`、b=1 加 `Bias[1]`，上例中 Bias 形状变为 `(2, N)`。
- 适用场景：batch 数多、单个矩阵小、操作数在 batch 维广播的批量乘加，例如多组查询共享同一份权重或键值矩阵。若改用普通 bmm 接口，广播操作数要么先物化成 batchCount 份（显存与 GM 带宽随 batch 数线性增长），要么逐 batch 重复装载同一份矩阵；本算子在片上只装载一份、组内复用，装载开销与 batch 数解耦。
- 不适用场景：单个矩阵大到放不进 L1 tile（m/n/k 超界，见"使用范围说明"）时，请改用 88 号样例的 tile 分块路径；操作数无任何广播的普通 bmm，广播复用没有收益，直接使用通用 bmm 即可。
- 与 88 号样例（bmm_broadcast）的边界：88 号按 M/N tile 分块调度、L1 中只驻留当前 tile，单矩阵可以很大；本算子整组 A/B 驻留 L1、按 batch 分组复用广播操作数，专攻"矩阵小、batch 多"的场景。一句话：矩阵大走 88 号的 tile 复用，矩阵小、batch 多走本算子的 batch 复用。batch 分组方式与缓冲规划行为见"使用范围说明"。
- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batchA, M, K)` | `float` | 批量输入矩阵A，layout为`RowMajor` |
| `B` | Input | `(batchB, K, N)` | `float` | 批量输入矩阵B，layout为`RowMajor` |
| `Bias` | Input | `(N,)` 或 `(batchCount, N)` | `float` | bias_mode=1时为共享bias`(N,)`；bias_mode=2时为逐batch bias`(batchCount, N)`；bias_mode=0时不使用 |
| `C` | Output | `(batchCount, M, N)` | `float` | 批量输出矩阵C，layout为`RowMajor` |

其中`batchA`/`batchB`只能取`1`（batch维广播）或`batchCount`（逐batch独立）。

## 使用范围说明

- 数据类型：算子支持 fp16/bf16/fp32 三种输入输出类型，累加类型为 FP32。可执行样例固定实例化为 fp32，输入输出矩阵均按`RowMajor`布局排布；DispatchPolicy的`useHF32`可开启HF32（仅float类型支持），本样例默认关闭。经 optest 接口（`torch_catlass.ascend950_iterbatch_broadcast_matmul`）调用时支持 fp16/bf16/fp32：tile shape 以 half 为基准，K tile 按数据位宽缩放（fp16/bf16 不变，fp32 减半，即 L1 `k <= 64`、L0 `k <= 16`）。
- 基础约束：
  - `batchCount`、`M`、`N`、`K`均为正数
  - `broadcast_a` / `broadcast_b`只能取 0（逐batch独立）或 1（batch维广播）
  - `bias_mode`只能取 0（无bias）、1（共享bias`[N]`）或 2（逐batch bias`[batchCount, N]`）

- 空间约束（Ascend 950 的`L1_SIZE`为512KB，`L0A_SIZE`、`L0B_SIZE`为64KB，`L0C_SIZE`为256KB）：
  - 首期走小矩阵路径：整组完整的 A/B 矩阵驻留 L1（即`m <= L1_TILE_M`，`n <= L1_TILE_N`，`k <= L1_TILE_K`），K 维仅在 L0 层切砖
  - batch 维采用两级组织：**L1 组**（B1 = `iterBatchL1` 个 batch 一组，广播侧操作数每组只装载一份物理数据，组内复用）与 **L0 子组**（B0 = `iterBatchL0` 个 batch 一组，每个 batch 拥有独立的 L0C 累加区），约束`1 <= B0 <= B1`，不要求整除；传 0 表示自动规划，自动规划的`iterBatchL1`额外满足`ceil(batchCount / coreNum)`并行下界，保证 grid-stride 启动时所有 Cube 核满载
  - 缓冲规划与拒绝行为：分配设备内存前，`IterBatchBroadcastMatmulTla::MakePlan` 会根据矩阵形状（batch_count、M、N、K）与 batch 分组参数计算 L1/L0A/L0B/L0C 各级缓冲字节数，并逐一与硬件容量比对；用户显式指定的 iterBatch 放不下、形状超出 L1 tile、或任一级越界时，程序打印`BlockMmad::ResourcePlan::Status`状态名与各级资源预算明细后以非零返回码退出，不会自动调小参数，由用户调整后重新运行
- 样例 Tile Shape：`L1TileShape<128, 128, 128>`、`L0TileShape<128, 128, 32>`。默认配置（batch=8，M=N=K=128，A逐batch、B广播）下自动规划得到`iterBatchL1 = 2`、`iterBatchL0 = 2`，此时 L1 占用 393216 / 524288（75%）。


## 使用示例

### 命令行参数

```bash
85_ascend950_iterbatch_broadcast_matmul [batch_count] [m] [n] [k] [broadcast_a] [broadcast_b] [bias_mode] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch_count` | 无 | batch 总数 |
| `m` | 无 | 单个矩阵的 M 维大小 |
| `n` | 无 | 单个矩阵的 N 维大小 |
| `k` | 无 | 单个矩阵的 K 维大小 |
| `broadcast_a` | `0` | 0 = A 逐 batch 独立，1 = A 在 batch 维广播 |
| `broadcast_b` | `1` | 0 = B 逐 batch 独立，1 = B 在 batch 维广播 |
| `bias_mode` | `0` | 0 = 无 bias，1 = 共享 bias `[N]`，2 = 逐 batch bias `[batchCount, N]` |
| `deviceId` | `0` | 指定运行设备ID |

其中`broadcast_a`、`broadcast_b`、`bias_mode`与`deviceId`为可选参数，合法调用形式为`batch_count m n k`、`batch_count m n k broadcast_a broadcast_b bias_mode`或`batch_count m n k broadcast_a broadcast_b bias_mode deviceId`。

### 执行示例

1. 将样例加入 `examples` 构建目录后，进入项目根目录编译样例代码（本样例为Ascend 950算子，需指定`-DCATLASS_ARCH=3510`）。

    ```bash
    bash scripts/build.sh 85_ascend950_iterbatch_broadcast_matmul -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./85_ascend950_iterbatch_broadcast_matmul 8 128 128 128 0 1 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    IterBatch plan: iterBatchL1 (L1 group) = 2, iterBatchL0 (L0 sub-group) = 2, L1 used = 393216 / 524288, L0A used = 65536 / 65536, L0B used = 32768 / 65536, L0C used = 131072 / 262144
    IterBatch execution plan: iterBatchL1 (L1 group) = 1, iterBatchL0 (L0 sub-group) = 1
    Compare success.
    ```

    程序输出 `Compare success.` 表示数值结果校验通过。

对应的 Python optest 接口为：

```python
torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
```

optest 测试用例保存在当前 experimental 样例目录，不会被全局 CI 自动收集。复验时可将
`test_85_ascend950_iterbatch_broadcast_matmul.py` 放入 `tests/optest/tests/` 后，按
`tests/optest/README.md` 的步骤执行。
