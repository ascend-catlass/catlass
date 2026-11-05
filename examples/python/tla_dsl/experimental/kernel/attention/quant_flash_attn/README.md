# QuantFlashAttn Example Readme

## 功能说明

- 算子功能：计算投机解码草稿阶段的固定窗口 Q8KV8 注意力前向，将分页 KV cache 寻址、QK、在线 softmax 与 PV 融合到一个 CATLASS DSL AIC/AIV mixed kernel 中。
- 计算公式：所有草稿 query 行共用一个绝对窗口 `[max(prefix - window + 1, 0), prefix + right)`，窗口内非因果。记 `t` 为逻辑 KV token，`phys(t) = block_table[t / page_size]`、`off(t) = t mod page_size`，对每个 query 行 `r`、head `h`：

  $$
  \begin{aligned}
  s_{r,h,t} &= \frac{1}{16}\, q[r,h,:] \cdot K[phys(t), off(t), :] \\
  p_{r,h,t} &= \operatorname{round\_e4m3}\left(\exp(s_{r,h,t} - \max_{t} s_{r,h,t})\right) \\
  O_{r,h,:} &= \frac{\sum_t p_{r,h,t}\, V[phys(t), off(t), :]}{\sum_t \exp(s_{r,h,t} - \max_{t} s_{r,h,t})}
  \end{aligned}
  $$

  QK/PV 以 E4M3 值参与矩阵乘、FP32 累加；softmax 全程 FP32，`scale = 256 ** -0.5 = 0.0625`；分子在 PV 前舍入到 E4M3，分母保持舍入前的 FP32 行和；整行位于窗口外时输出 0。

- 支持产品型号：Ascend 950PR&950DT 系列产品

## 参数说明

```python
output = quant_flash_attn.run(
    q, k_cache, v_cache, block_table, kv_lens,
    q_positions, k_positions, window, window_right=None,
)
```

记 `Tq` 为草稿 query token 数，`H` 为 query head 数，`P` 为物理页数，`kv_len` 为 KV cache 有效长度。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[Tq, H, 256]` | `float8_e4m3fn` | 草稿 query，已由上游量化为无 scale E4M3 |
| `k_cache` | Input | `[P, 1, 128, 256]` | `float8_e4m3fn` | 分页 K 物理页池 |
| `v_cache` | Input | `[P, 1, 128, 256]` | `float8_e4m3fn` | 分页 V 物理页池，shape 与 dtype 必须和 `k_cache` 一致 |
| `block_table` | Input | `[n_pages]` | `int32` | 逻辑页到物理页的映射 |
| `kv_lens` | Input | `[1]` | `int32` | KV cache 有效长度 |
| `q_positions` | Input | `[Tq]` | `int32` | 草稿 query 绝对位置，从 `prefix` 开始连续 |
| `k_positions` | Input | `[kv_len]` | `int32` | KV 绝对位置，分页连续，要求 `k_positions[i] == i` |
| `window` | Input | 标量 | `int` | 左窗口大小 |
| `window_right` | Input，可选 | 标量 | `int` 或 `None` | 右窗口（草稿行数）；`None` 时取 `Tq` |
| `output` | Output | `[Tq, H, 256]` | `bfloat16` | 注意力结果 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 维度 | `head_dim` 固定为 256；`Hkv = 1`；`page_size = 128` |
| dtype | `q`、`k_cache`、`v_cache` 必须为 `float8_e4m3fn`；四个位置/索引张量必须为 `int32` |
| 请求数 | 单请求接口，`kv_lens` 恰好一个元素 |
| 窗口 | `window > 0`，`right >= Tq`；窗口为 `[max(prefix - window + 1, 0), prefix + right)` |
| 位置 | `q_positions` 从 `prefix` 起连续；`k_positions[i] == i`，且 `kv_len <= n_pages * 128` |
| 页表 | 使用到的物理页号必须位于 `[0, P)` 内 |
| 布局与设备 | 输入张量位于同一 NPU |
| 数值行为 | 窗口外 mask 为 `-3e38`；整行位于窗口外时输出精确 0 |

## 使用示例

```python
import torch
import torch_npu  # noqa: F401

import quant_flash_attn

tq, heads, page_size, pages = 7, 128, 128, 33
kv_len, window = 4102, 4096
q = torch.randn(tq, heads, 256, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
k_cache = torch.randn(pages, 1, page_size, 256, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
v_cache = torch.randn(pages, 1, page_size, 256, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
block_table = torch.arange(pages, dtype=torch.int32).npu()
kv_lens = torch.tensor([kv_len], dtype=torch.int32).npu()
q_positions = torch.arange(kv_len - tq, kv_len, dtype=torch.int32).npu()
k_positions = torch.arange(kv_len, dtype=torch.int32).npu()

out = quant_flash_attn.run(
    q, k_cache, v_cache, block_table, kv_lens,
    q_positions, k_positions, window=window, window_right=tq,
)
print(out.shape, out.dtype)
```
