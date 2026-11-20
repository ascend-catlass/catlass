# Quant Block Sparse Attention（MXFP8）

## 功能说明

MXFP8 块稀疏注意力前向。输入 Q/K/V 为 E4M3FN，反量化 scale 为 E8M0，输入量化由调用方完成。每个 query block 通过 `block_indices` 指定可见的 KV blocks，仅计算已选有效 token 的注意力：

```text
Q = decode_e4m3fn(q) * decode_e8m0(q_dequant_scale)
K = decode_e4m3fn(k) * decode_e8m0(k_dequant_scale)
V = decode_e4m3fn(v) * decode_e8m0(v_dequant_scale)

S = FP16(softmax_scale * Q @ selected_K.T)
W = group32_quantized_exp(S)
output = sum(W * selected_V) / sum(W)
```

计算使用原生 MXFP8 QK/PV 矩阵乘法和 FP32 累加。Q/K 沿通道轴每 32 个元素共享 scale，V 沿序列轴每 32 个元素共享 scale。GQA 的 KV head 为 `query_head // (QH // KVH)`。

连续、完整长度 BNSD 的 attention 主计算在一个融合内核中完成。BSND、TND 和含实际长度的 padded batch 通过 NPU 布局复制、逐序列调用及结果回写适配。全部运行时代码位于 [quant_block_sparse_attn.py](quant_block_sparse_attn.py)，软最大值、概率量化与输出归约均为 TLA 实现，不再依赖 AscendC extern 源。

### 长序列分派

当 block 为 128×128、D=128、Q 长度至少 8192、索引槽数至少 64，且 `|softmax_scale| <= 1e10` 时，入口检查本次输入的反量化 Q/K 范数：

```text
max_row ||Q||₂ * max_row ||K||₂ * |softmax_scale| <= 30
```

满足范围检查时使用整行公共指数 K=48 的累加路径；否则回退到在线重缩放路径。范数计算、归约和判断会启动额外设备内核，并通过一次 host 读取选择分支，因此该路径不是单内核调用，也不是无 host 同步的执行路径。

## 参数说明

```python
output = quant_block_sparse_attn(
    q, k, v, block_indices, block_shape,
    q_dequant_scale, k_dequant_scale, v_dequant_scale,
    q_input_layout="BNSD", kv_input_layout=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    softmax_scale=None, output_dtype="bfloat16", return_lse=False,
)
```

记 B 为 batch 数，QH/KVH 为 query/KV head 数，Sq/Sk 为定长或 padded Q/K 长度，Tq/Tk 为 packed 总 token 数，D 为 head dimension。`block_shape=[X,Y]`；定长的 `Nq=ceil(Sq/X)`，TND 的 `Nq=ceil(max(各序列实际Q长度)/X)`。

| 参数                      | 类型 / 默认值                            | 说明                                                                     |
| ----------------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| `q`                     | E4M3FN Tensor                       | query，布局及 shape 见下表                                                    |
| `k`                     | E4M3FN Tensor                       | key，与 Q 使用相同布局和 D                                                      |
| `v`                     | E4M3FN Tensor                       | 与 K 同 shape，value dimension 同样为 D                                      |
| `block_indices`         | INT32/INT64 Tensor                  | `[B,Nq,Ksel]` 或 `[B,MH,Nq,Ksel]`；`MH=1` 共享稀疏模式，`MH=QH` 逐 query head 指定 |
| `block_shape`           | 长度为 2 的整数序列                         | Q/KV 块大小 `[X,Y]`，两个值均为正的 64 倍数                                         |
| `q_dequant_scale`       | UINT8 或 E8M0 Tensor                 | Q 沿 D 轴的 group32 反量化乘数                                                 |
| `k_dequant_scale`       | UINT8 或 E8M0 Tensor                 | K 沿 D 轴的 group32 反量化乘数                                                 |
| `v_dequant_scale`       | UINT8 或 E8M0 Tensor                 | V 沿序列轴的 group32 反量化乘数                                                  |
| `q_input_layout`        | str，默认 `"BNSD"`                     | 支持 `"BNSD"`、`"BSND"`、`"TND"`                                           |
| `kv_input_layout`       | str 或 `None`                        | 默认与 Q 一致；显式传入时必须与 Q 相同                                                 |
| `actual_seq_lengths`    | 整数序列、一维 INT32/INT64 Tensor 或 `None` | 定长布局传每条 Q 序列的实际长度；TND 必填，传不含前导 0 的累计结束位置                               |
| `actual_seq_lengths_kv` | 同上                                  | K/V 的实际长度或累计结束位置；与 Q 的长度参数同时提供或同时省略                                    |
| `softmax_scale`         | 有限浮点数或 `None`                       | 默认 `D**-0.5`                                                           |
| `output_dtype`          | dtype 或名称，默认 `"bfloat16"`           | 支持 `float16`、`bfloat16`、`float32`                                      |
| `return_lse`            | bool，默认 `False`                     | 为 `True` 时返回 `(output, lse)`，否则仅返回 output                              |

### 张量布局

| 张量      | BNSD                      | BSND                      | TND                             |
| ------- | ------------------------- | ------------------------- | ------------------------------- |
| Q       | `[B,QH,Sq,D]`             | `[B,Sq,QH,D]`             | `[Tq,QH,D]`                     |
| K/V     | `[B,KVH,Sk,D]`            | `[B,Sk,KVH,D]`            | `[Tk,KVH,D]`                    |
| Q scale | `[B,QH,Sq,D/64,2]`        | `[B,Sq,QH,D/64,2]`        | `[Tq,QH,D/64,2]`                |
| K scale | `[B,KVH,Sk,D/64,2]`       | `[B,Sk,KVH,D/64,2]`       | `[Tk,KVH,D/64,2]`               |
| V scale | `[B,KVH,ceil(Sk/64),D,2]` | `[B,ceil(Sk/64),KVH,D,2]` | `[sum_b ceil(Sk_b/64),KVH,D,2]` |
| output  | 与 Q 同 shape               | 与 Q 同 shape               | 与 Q 同 shape                     |
| LSE     | `[B,QH,Sq,1]`             | `[B,Sq,QH,1]`             | `[Tq,QH,1]`                     |

scale 末维的 2 对应同一 64 元素范围内的两个 group32。TND 的 V scale 在每条序列起点重新分组，各序列分别补齐到 64 后拼接。

`block_indices` 中的 KV block 编号相对于当前序列起点。负数为 padding；非负编号排序、去重后参与计算。全负行或 `Ksel=0` 表示空选择。空稀疏行、空 KV 序列和 padded query 的 output 为 0，LSE 为 `-inf`。

## 约束说明

- 依赖 CATLASS DSL、PyTorch 和 torch\_npu，目标设备为 Ascend 950，编译架构固定为 `3510`。
- Q/K/V 必须为 `torch.float8_e4m3fn`，scale 必须为 `torch.uint8` 或 `torch.float8_e8m0fnu`，所有输入 Tensor 位于同一 NPU。完整长度 BNSD 主路径要求 Q/K/V、scale 和索引连续。
- D 仅支持 64/128，B、QH、KVH 必须为正，`QH % KVH == 0`；K/V shape 相同。输出 dtype 不改变输入和概率的 MXFP8 计算精度。
- 块大小 X/Y 必须为正的 64 倍数。序列长度无需与块大小对齐，尾块仅处理有效 token；支持零长度序列。
- 固定布局的实际长度位于 `[0,padded长度]`。TND 的两组累计结束位置均长为 B，单调不减，末项分别等于 Tq/Tk；不包含前导 0。Tensor 形式的长度参数会读取到 host。
- 稀疏索引只接受三维或四维形式，query-block 维必须等于 Nq，mask-head 维只能为 1 或 QH。非负索引必须处于对应序列可用的 KV 块范围内。
- E8M0 编码 `0..254` 表示 `2**(byte-127)`，`255` 表示 NaN，不能作为有效输入。入口不扫描 scale 字节和索引数值，调用方必须保证合法。
- 解码后的输入和 scale 应为有限值；FP16 score、指数基点及中间概率需处于可表示范围。浮点累加顺序、指数近似和极端指数的下溢会影响结果。
- 只实现无额外 token mask 的前向，不提供 causal 参数或反向接口。含范数检查的路径存在 host 同步；不提供图捕获支持保证。

## 使用示例

```python
import torch
import torch_npu
from quant_block_sparse_attn import quant_block_sparse_attn

# 已由调用方完成 MXFP8 量化：q/k/v 为 E4M3FN，scale 为 E8M0 字节
block = 128
blocks = (sk + block - 1) // block
selected = max(1, round(blocks * (1 - sparsity)))
indices = torch.stack([
    torch.randperm(blocks)[:selected].sort().values for _ in range((sq + block - 1) // block)
]).to(torch.int32)[None, None].npu()

out = quant_block_sparse_attn(
    q, k, v, indices, [block, block],
    q_scale, k_scale, v_scale,
    q_input_layout="BNSD", output_dtype="bfloat16", return_lse=False,
)
```
