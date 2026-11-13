# MatmulRowExpScatterAdd Example Readme

## 功能说明

- 算子功能：对 `x @ weightᵀ` 的结果执行逐行指数变换和梯度缩放，再累加选中列的梯度；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。
- 计算公式：

  $$
  \begin{aligned}
  C &= \operatorname{bfloat16}(X W^{\mathsf{T}}) \\
  E[m, n] &= \exp(\operatorname{float32}(C[m, n]) * invT - LSE[m]) \\
  r[m] &= \sum_{s=0}^{S-1} U[m, s] \\
  Y[m, n] &= \operatorname{bfloat16}\left(
    \left(-E[m, n] * r[m] + \sum_{s:\,I[m,s]-col\_start=n} U[m, s]\right) * invT
  \right)
  \end{aligned}
  $$

  其中 `X` 为 `[M, H]`、`W` 为 `[V, H]`，分别对应 `x` 和 `weight`；`M` 为行数，`H` 为内积维度，`V` 为当前列分片宽度，`S` 为每行选中槽位数。`I`、`U`、`LSE` 分别对应 `global_col_ids`、`values`、`row_logsumexp`，`invT = 1 / temperature`，`col_start = shard_index * V`。

  `row_logsumexp` 由调用方提供。片外列跳过 scatter-add，但对应的 `values` 仍参与行求和；重复下标按槽位顺序依次累加。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 代码组织

| 文件 | 概述 |
| ----- | ----- |
| [matmul_row_exp_scatter_add.py](matmul_row_exp_scatter_add.py) | 矩阵乘、逐行指数变换与选中列梯度累加的融合实现。 |

## 参数说明

```python
y = matmul_row_exp_scatter_add.run(
    x, weight, global_col_ids, row_logsumexp, values,
    temperature, shard_index=0, out=None,
)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, H]` | `bfloat16` | 左操作数 |
| `weight` | Input | `[V, H]` | `bfloat16` | 当前列分片的权重；矩阵乘使用其转置 |
| `global_col_ids` | Input | `[M, S]` | `int32` | 选中列的全局下标；片外下标跳过 scatter-add |
| `row_logsumexp` | Input | `[M]` | `float32` | 与温度缩放后矩阵乘结果对应的逐行对数归一化偏移 |
| `values` | Input | `[M, S]` | `float32` | 选中槽位的梯度值，所有槽位均参与行求和 |
| `temperature` | Input | 标量 | `float` | 结果缩放温度，必须为正数 |
| `shard_index` | Input | 标量 | `int` | 当前列分片编号，默认为 `0`，必须为非负整数 |
| `out` | Output | `[M_pad, V_pad]` | `bfloat16` | 可选输出缓冲区，默认为 `None`，由 host 分配 |
| `y` | Output | `[M, V]` | `bfloat16` | 输出缓冲区的有效区域视图，可能不连续 |

`M_pad`、`V_pad` 分别为 `M`、`V` 向上补齐到 256 的结果。

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`V`、`S` 为非负整数，`H` 为正整数；可非 tile 整数倍，host 侧自动补零 |
| dtype | `x` 与 `weight` 必须同为 `bfloat16`，`global_col_ids` 为 `int32`，`row_logsumexp` 与 `values` 为 `float32`；输出固定 `bfloat16` |
| 设备 | 所有输入与可选 `out` 位于同一 NPU，执行前将该 NPU 设为当前设备 |
| 输出缓冲区 | 显式传入的 `out` 必须为连续张量，shape 为 `[M_pad, V_pad]` |
| 编译 | 每个设备上，按补齐后的 `(M, H, V)` 与 `S` 编译一个 kernel，编译产物按 shape 缓存 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 CATLASS DSL 环境，导入样例模块后调用 `run`。脚本根据输入编译 kernel、执行 NPU 计算并返回结果：

```python
import torch
import torch_npu  # noqa: F401

import matmul_row_exp_scatter_add

torch.npu.set_device(0)
torch.manual_seed(0)
m, h, v, s = 128, 256, 512, 8
temperature = 0.8
x = (torch.randn(m, h, device="npu:0") * 0.1).bfloat16()
weight = (torch.randn(v, h, device="npu:0") * 0.1).bfloat16()
global_col_ids = torch.randint(0, v, (m, s), dtype=torch.int32, device="npu:0")
values = torch.randn(m, s, dtype=torch.float32, device="npu:0")

# 单分片示例：调用方预先计算归一化偏移。
logits = (x.float() @ weight.float().T).bfloat16().float()
row_logsumexp = torch.logsumexp(logits / temperature, dim=-1)
y = matmul_row_exp_scatter_add.run(
    x, weight, global_col_ids, row_logsumexp, values, temperature,
)
print(y.shape, y.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([128, 512]) torch.bfloat16
```

## 性能调度与写回

kernel 在编译期根据 `m_tiles = ceil(M / 256)`、`v_tiles = ceil(V / 256)` 选择调度，按下表顺序匹配。矩阵乘的 M/N tile 保持为 256，L0 的 K 步长保持为 64；`H` 的 Padding 粒度为 256，与 L1 的 K 块大小 `tile_k` 分开。

| 条件 | `(swizzle_m, swizzle_n)` | `tile_k` |
| ----- | ----- | ----- |
| `m_tiles >= 32` 且 `v_tiles >= 256` | `(32, 2)` | 128 |
| `m_tiles >= 8` 且 `v_tiles >= 128` | `(8, 4)` | 128 |
| `m_tiles >= 32` | `(2, 8)` | 128 |
| 其余 | `(2, 8)` | 256 |

`(2, 8)` 的两个维度分别限制为不超过 `m_tiles`、`v_tiles`。分档阈值来自 Ascend 950PR、28 个 Cube Core、`H=2560`、`S=128` 的经验调优，核数通过 `torch.npu.get_device_properties(device).cube_core_num` 查询；迁移到其他 shape 或设备时需重新测量。例如 `M=8192, V=16384, tile_k=128` 的扫描中，使用打包 DMA 写回时，`(2, 8)` 为 1.990 ms，`(32, 2)` 为 2.328 ms。

输出在 SIMD 中完成 RNE（最近偶数舍入），精确转换为 bf16 后通过 `DIST_PACK_B32` 紧凑存入 UB，由 MTE3 批量写回 GM。`mutex_output` 保护 VECTOR 与 MTE3 之间的缓冲区复用，GM 元素偏移使用 64 位整数。矩阵乘结果与最终输出的两次 RNE、重复下标按槽位顺序累加的规则保持不变。
