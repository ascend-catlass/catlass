# ScatterRowAdd Example Readme

## 功能说明

- 算子功能：对 `[M, V]` 的输出按行做索引散写累加，即把每行 `S` 个槽位上的值加到该行指定列上；基于 CATLASS DSL 实现的纯 Vector（AIV）kernel，入口为 `scatter_row_add` 模块的 `run` 函数。
- 计算公式：

  $$
  out[m, ids[m, s]] \mathrel{+}= values[m, s],
  \qquad 0 \le m < M,\; 0 \le s < S,\; 0 \le ids[m, s] < V
  $$

  同一行的多个槽位落在同一列时，按槽位下标升序依次累加；`ids[m, s]` 为负或大于等于 `V` 时跳过该槽位。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = scatter_row_add.run(global_col_ids, values, num_cols, out=None)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `global_col_ids` | Input | `[M, S]` | int32 | 每行 `S` 个目标列下标；越界槽位跳过 |
| `values` | Input | `[M, S]` | float32 | 每行 `S` 个待累加值 |
| `num_cols` | Input | 标量 | `int` | 输出列宽 `V`，必须为正 |
| `out` | InOut | `[M, V]` | float32 | 可选；传入时在其上累加，不传时由内部置零 |
| `y` | Output | `[M, V]` | float32 | 输出，与 `out` 为同一缓冲区 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `global_col_ids` 与 `values` 必须同为 `[M, S]` 二维张量；`S` 可为 0 |
| dtype | `global_col_ids` 为 int32（其它整型会被转换），`values` 与 `out` 为 float32 |
| 输入布局 | 输入必须连续；`out` 传入时也必须连续 |
| 列宽上限 | `V` 必须是 8 的倍数（保证每行 32 字节对齐），并受 UB tile 预算限制：`16 × V × 4` 字节不得超过 UB 可用容量，即 `V ≤ 3552`；超过时需在调用方按列切分 |
| 确定性 | 同一行由同一线程处理，重复下标按槽位升序累加，结果确定 |

## 代码组织

kernel 按行 tile 划分，每个 block 处理若干 16 行的 tile：

1. `MTE2`：把该 tile 的 `out` 从 GM 载入 UB（`out` 未传入时改为下一步）
2. `SIMD`：`out` 未传入时，用向量填充把 UB tile 置零
3. `SIMT`：一行一线程，线程内按槽位升序扫描 `S`，命中本行有效列的槽位在 **UB 内**累加
4. `MTE3`：整 tile 写回 GM

随机读写只发生在 UB，GM 只做规整的整块搬运。三个阶段的交界处各有一个 `pipe_barrier`：载入/置零完成后才允许散写，散写完成后才允许写回，否则会读到未完成的数据。

### 融入其他算子

本 kernel 的散写段可以独立抽出，拼接到其他 Vector 阶段的输出上：

- 若上游结果已经在 UB 中（例如 GEMM 的 epilogue 刚算完的 fp32 tile），直接在该 UB tile 上做第 3 步即可，**不需要再从 GM 读取，也不需要中间写回**
- 上游的规整运算（缩放、激活）与下游的后处理（缩放、`bf16` 舍入）分别位于散写之前和之后，由拼接方自行安排；散写本身不改变 dtype，也不做舍入
- 若散写目标只在 `V` 的某个分片内，调用方按分片裁好 `global_col_ids` 后再进入本段

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from scatter_row_add.scatter_row_add import run

torch.npu.set_device(0)
torch.manual_seed(0)
m, v, s = 128, 512, 8
ids = torch.randint(0, v, (m, s), dtype=torch.int32, device="npu:0")
values = torch.randn(m, s, dtype=torch.float32, device="npu:0")

y = run(ids, values, v)
print(y.shape, y.dtype)
PY
```

执行结果如下，说明样例执行成功：

```text
torch.Size([128, 512]) torch.float32
```

## 验证方法

- 与 `torch.index_add_` 对照：把越界下标过滤掉后，`base.index_add_(1, ids_clamped, values)` 应与本算子输出逐元素一致（`atol=0`）
- 专项用例：同一行重复下标、全部越界、混合越界、`S=0`、`M=0`、`V` 非 32 对齐
- 确定性：同输入连续执行两次，输出逐位一致
- 累加语义：传入非零 `out` 时，结果应等于 `out + scatter`，而不是覆盖
