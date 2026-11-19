# SplitQkvRmsnormRope Example Readme

## 功能说明

- 算子功能：对已融合的 QKV 输入按 `[q_hidden_size, kv_hidden_size, kv_hidden_size]`
  切分为 Q、K、V；对 Q、K 逐 head 做不带均值的 RMSNorm，乘 head 权重（可选加 bias）；
  再对每个 head 的前 `rope_dim` 维做 NeoX 风格 RoPE；V 直接拷出。基于 CATLASS DSL
  实现的单个纯 AIV kernel。
- 计算公式：

  $$
    \text{var} = \frac{1}{d}\sum_{i=0}^{d-1} x_i^2,\qquad
    y_i = \frac{x_i}{\sqrt{\text{var} + \epsilon}} \cdot w_i + b_i
  $$

  归一化结果按 bfloat16 舍入后做旋转，`half = rope_dim // 2`，
  `cos, sin = cache[pos, :half], cache[pos, half:]`：

  $$
    \begin{aligned}
    y'_i &= y_i \cos_i - y_{i+half} \sin_i \\
    y'_{i+half} &= y_{i+half} \cos_i + y_i \sin_i
    \end{aligned}
    \qquad i \in [0, half)
  $$

  `rope_dim < head_dim` 时（partial RoPE），`[rope_dim, head_dim)` 原样保留。
  其中 $x$ 为单个 head 的输入向量，$d$ 为 `head_dim`，$w$ / $b$ 为该 head 的权重与偏置。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 参数说明

SplitQkvRmsnormRope 的输入输出参数如下：

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `input` | Input | `(T, q_hidden_size + 2 * kv_hidden_size)` | `bfloat16` | QKV 融合输入 |
| `cos_sin_cache` | Input | `(max_positions, rope_dim)` | `bfloat16` | 每行 `[cos | sin]` |
| `positions` | Input | `(T,)` | `int64` | 每个 token 的位置索引 |
| `q_weight` / `k_weight` | Input | `(head_dim,)` | `bfloat16` | 逐 head 的 RMSNorm 权重 |
| `q_bias` / `k_bias` | Input | `(head_dim,)` | `bfloat16` | 可选，归一化后的偏置 |
| `q_hidden_size` | 属性 | 整数 | — | `q_heads * head_dim` |
| `kv_hidden_size` | 属性 | 整数 | — | `kv_heads * head_dim` |
| `head_dim` | 属性 | 整数 | — | 单个 head 宽度 |
| `eps` | 属性 | 浮点 | — | RMSNorm 数值稳定项 |
| `q` / `k` / `v` | Output | `(T, q_hidden_size)` / `(T, kv_hidden_size)` / `(T, kv_hidden_size)` | `bfloat16` | 旋转后的 Q/K 与拷贝的 V |

## 约束说明

- `input` 必须是 2 维张量，最后一维必须等于 `q_hidden_size + 2 * kv_hidden_size`；
  `positions` 的长度必须等于 `input` 的行数。内核按输入 extent 推导行数，形状不符会
  多写（越界）或少写（返回未初始化数据），因此入口直接拒绝。
- `q_hidden_size % head_dim == 0`、`kv_hidden_size % head_dim == 0`。
- `q_hidden_size` 必须是 `kv_hidden_size` 的整数倍（GQA / MQA 几何）。
- `rope_dim` 必须为正偶数且不超过 `head_dim`，并且 `head_dim % (rope_dim // 2) == 0`
  （设备侧用 tile 下标定位每个 head 的 RoPE 前后半，head 起点必须 tile 对齐）。
- `positions` 的每个元素必须落在 `[0, cos_sin_cache.shape[0])` 内。
- `q_bias` / `k_bias` 必须同时提供或同时省略，否则入口直接抛 `ValueError`；
  当前样例仅覆盖 `bfloat16` 输入。
- 设备侧在 fp32 下完成 RMSNorm 与 RoPE 的乘加，归一化结果与旋转结果各做一次
  bfloat16 舍入，与 Ascend Triton 后端的数值路径一致。

## 使用示例

### 命令行参数

```text
split_qkv_rmsnorm_rope.py [-h] [--device DEVICE] [--tokens TOKENS]
                          [--q-heads Q_HEADS] [--kv-heads KV_HEADS]
                          [--head-dim HEAD_DIM] [--rope-dim ROPE_DIM]
                          [--eps EPS] [--bias] [--cache-rows CACHE_ROWS]
                          [--seed SEED] [--block-num BLOCK_NUM]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
| ----- | -------- | ------ |
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--tokens` | `128` | token 数，即输入行数。 |
| `--q-heads` | `32` | Q head 数，`q_hidden_size = q_heads * head_dim`。 |
| `--kv-heads` | `8` | KV head 数，`kv_hidden_size = kv_heads * head_dim`。 |
| `--head-dim` | `128` | 单个 head 的宽度。 |
| `--rope-dim` | `128` | RoPE 宽度，须为正偶数、`<= head_dim`，且 `head_dim % (rope_dim // 2) == 0`；小于 `head_dim` 时为 partial RoPE。 |
| `--eps` | `1e-6` | RMSNorm 的数值稳定项。 |
| `--bias` | `False` | 是否对归一化结果额外加 `q_bias` / `k_bias`。 |
| `--cache-rows` | `16384` | `cos_sin_cache` 的行数（最大位置数）。 |
| `--seed` | `0` | 生成随机输入所用的种子。 |
| `--block-num` | `-1` | 启用的 AIV 核数；该值会实际传入 kernel，`-1` 表示自动探测可用核数（满核）。 |

### 执行示例

进入项目根目录，执行算子样例程序。脚本会生成随机输入、完成 kernel 编译与执行，
并与 golden 参考实现做精度比较。

```bash
cd /path/to/catlass
python examples/python/tla_dsl/experimental/kernel/vec/split_qkv_rmsnorm_rope/split_qkv_rmsnorm_rope.py --device 0
```

执行结果如下，说明样例执行成功，精度通过：

```text
--- tokens=128 q_heads=32 kv_heads=8 head_dim=128 rope_dim=128 bias=False blocks=64 ---
accuracy out=q pass=True matched=1.000000 max_abs=0.015625
accuracy out=k pass=True matched=1.000000 max_abs=0.015625
accuracy out=v pass=True matched=1.000000 max_abs=0.000000
validation pass=True q=4096 kv=1024 slots=4 rows_per_group=2
```

其中 `matched` 为落在 `atol = 1e-3`、`rtol = 2^-7` 容差内的元素占比（阈值 `0.999`），
`max_abs` 为最大绝对误差，具体数值随输入与运行环境变化，以 `pass=True` 为准。

其它配置示例：

```bash
# 单 token MHA 几何
python .../split_qkv_rmsnorm_rope.py --tokens 1 --q-heads 16 --kv-heads 8

# partial RoPE + bias
python .../split_qkv_rmsnorm_rope.py --tokens 512 --rope-dim 64 --bias

# MQA 几何
python .../split_qkv_rmsnorm_rope.py --tokens 2048 --kv-heads 1
```

## 实现要点

- **pingpong 双缓冲**：外层动态 group 循环，内层 `tla.range_constexpr` 静态 slot 循环，
  每个 slot 持有 row / q / k 三把 mutex，使一个 slot 的 MTE2/MTE3 搬运与另一个
  slot 的向量计算重叠。
- **bf16 even/odd 拆分**：利用 AVE bf16→fp32 cast 的奇偶 lane 打包，把 128 lane
  拆成两组 fp32 向量做归约，权重、bias、cos/sin 提前按同样方式拆好。
- **host 侧固定开销裁剪**：按 `(地址, 元素数, dtype)` 缓存张量绑定、按 (线程, device)
  缓存 stream 句柄、按 `(artifact, 各 buffer 的地址/元素数/dtype)` 缓存 launch payload
  （元素数入键，避免地址被复用而继承了旧的行数），避免小 token 场景下 host 开销超过
  kernel 时间。多流调用方需在切换流后调用 `reset_stream_cache()`，或以
  `SQR_STREAM_CACHE=0` 启动以禁用 stream 缓存。
