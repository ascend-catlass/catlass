# MoE Route Example Readme

## 功能说明

- 算子功能：MoE 的 gate（softmax top-k 路由）与 combine（按路由权重加权专家输出）融合为**单个 kernel**，一次 launch 完成，入口为 `moe_route` 模块的 `run` 函数，基于 CATLASS DSL 实现的单个纯 AIV kernel。一个 AIV 块认领一段连续 token，单个 token 的处理链是
  `MTE2(logits) -> VECTOR(softmax + top-k) -> SCALAR(读回 id) -> MTE2(K 条专家整行) -> VECTOR(K 路乘加) -> MTE3(w / idx / y)`。
- 计算公式：对每个 token `b`（`p` 为概率，`e[b, j]` 为第 `j` 大的专家 id）：

  $$
  \begin{aligned}
  p_{b,:} &= \mathrm{softmax}(\mathrm{logits}_{b,:}) \\
  w_{b,j}, e_{b,j} &= \text{top-}K(p_{b,:}) \\
  w_{b,:} &\gets w_{b,:} / \textstyle\sum_j w_{b,j} \quad (\text{RENORM} = 1) \\
  y_{b,:} &= \sum_{j < K} w_{b,j} \cdot \mathrm{experts}_{e_{b,j},\,b,\,:}
  \end{aligned}
  $$

  全程 FP32 计算；`RENORM = 0` 时省掉归一化，整段在编译期消除。
- 高性能设计：
  - **单 kernel，中间结果不出 UB**：top-k 的权重 / id 打包在 UB 的 64-lane 缓冲里，`local_mem_bar(VEC_STORE, SCALAR_LOAD)` 之后由标量单元直接读回 lane `0..K-1`，省掉一次 kernel launch、一次概率写回和一次 host 侧 gather 索引的 materialize。
  - **一个 64-lane 向量装下一个 token 的全部 logits**：行按 `E_PAD = ceil(E/64) * 64` 补齐留在 UB，GM 侧始终是稠密的 `[B, E]`，行首字节偏移不要求 32 B 对齐。
  - **两遍定 softmax，省掉一趟写回**：第一遍求行 max，第二遍把 `exp(x - max)` 直接原地覆盖 logits 并累加分母；因为 exp 单调，top-k 选择直接在 exp 值上做，不必再单独存一份概率。
  - **选择 = max + lane 位置扫描**：每选出一个 top 值，只做一次全行 `max` 和一次「等于最大值的最小 lane 号」扫描（`min` 归约），`K` 个结果打包进同一个 64-lane 向量的 lane `0..K-1`，权重、id 各只写一次。选出后按 **lane 号**（而不是按取值）抹掉该 lane 再进入下一轮，因此概率并列时按专家 id 从小到大依次入选，不会像「按值抹除」那样把并列项一起抹掉。
  - **数据相关 gather 交给 MTE2**：一个 token 的 `K` 条专家行在 GM 上互不相邻，但每条行本身连续。用标量读回的 `idx` 直接算出 `idx * B + b` 行的 GM 地址，`K` 条整行 DMA 连续下发；vector 核不做逐元素 4 B GM 读，吞吐由 DMA 而不是标量 load 决定。
  - **K 条行乘加全程在寄存器里完成**：每个 chunk 读入 `K` 条行的同一段，累加器留在向量寄存器中，只把最终结果写回行 0 的同一段，`K` 条行只占 `K` 个 UB 行缓冲、只做一次 UB 写回，没有 `K` 次 UB 往返；权重用 UB→Reg 的 `gather` 按 lane 广播，不占额外缓冲。
  - **per-token mutex 的多段流水**：`BUFS` 个 token 槽位各挂一个 mutex，MTE2 搬第 `t+1` 个 token、VECTOR 算第 `t` 个、MTE3 写第 `t-1` 个重叠执行；槽位复用由 mutex 而不是程序序控制，`BUFS` 按 `E_PAD + K * D_PAD` 的 UB 预算自动在 1..4 之间选择。标量段锁在 `SCALAR`、解锁在 `MTE2` 上，保证标量读等 vector store 落地、下一段 VECTOR 等 DMA 落地。
  - **尾块只收窄 mask**：`E` / `D` 不是 64 的倍数时，最后一段按整 64 lane 计算，只在 store 时用 `update_mask` 收窄；UB 行宽固定 `E_PAD` / `D_PAD`，GM 行宽是稠密的 `E` / `D`，所以 `E` / `D` 只要求是正整数，不要求 32 B 对齐。
  - **无 padding 干扰**：padding lane 在求 max / 求分母 / 选择前都显式置为 `-inf`，而 `exp(x - max)` 的最大值恒为 1，所以 padding 的 `-inf` 永远不会被选中。
  - **`experts` 以 `[H * B, D]` 的 2-D 视图传入**（`from_dlpack` 不支持 3-D layout），连续输入下就是零拷贝视图，行号在 kernel 内算出。
  - **相对 `torch_npu` 组合实现（`softmax -> topk -> renorm -> 按 id gather -> 加权求和`，其中 `topk` 占主要耗时）**：上述设计把整条链压成一次 kernel launch，top-k 无排序、无概率写回，`K` 条专家行由 DMA 整行搬运，因此 device 侧耗时明显低于组合实现。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y, w, idx = run(logits, experts, k=2, renorm=True)
```

`B` 为 token 数，`E` 为 logits 的专家数（门控宽度），`H` 为 `experts` 的专家数，`D` 为每个专家输出的通道数，`K` 为 top-k。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `logits` | Input | `[B, E]` | `float32` | 门控 logits；`E` 任意（非 64 倍数走尾块路径）；非连续输入内部自动 `contiguous()`。 |
| `experts` | Input | `[H, B, D]` | `float32` | 专家输出，按 token 序排列（`experts[h, b, :]` 是专家 `h` 对 token `b` 的输出）；需 `H >= E`；`D` 任意；非连续输入内部转成 `[H*B, D]` 连续视图。 |
| `k` | 属性 | 标量 | Python `int` | top-k，取值区间 `[1, 64]`，默认 `2`。 |
| `renorm` | 属性 | 标量 | Python `bool` | 是否对 top-k 权重归一化，默认 `True`。 |
| `y` | Output | `[B, D]` | `float32` | combine 结果。 |
| `w` | Output | `[B, K]` | `float32` | top-k 权重（`renorm=True` 时已归一化）。 |
| `idx` | Output | `[B, K]` | `int32` | top-k 专家 id，取值 `[0, E)`。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `logits` 必须为 2 维；`experts` 必须为 3 维 `[H, B, D]`，且 `experts.shape[1] == logits.shape[0]`。 |
| 专家数 | 必须 `H >= E`：`idx` 的取值是 logits 的 lane 号 `[0, E)`，`H < E` 会用越界 id 去 gather 专家行（host 侧直接报错）。 |
| top-k | `1 <= K <= 64`（一个 64-lane 向量同时装下全部 top-k 结果），且 `K <= E`。 |
| dtype | `logits`、`experts`、`y`、`w` 均为 `float32`，`idx` 为 `int32`。 |
| UB 占用 | 每个在飞 token 槽位 `4 * (E_PAD + K * D_PAD + 128)` 字节（`E_PAD`/`D_PAD` 为 64 对齐），需 `E_PAD + K * D_PAD <= 63360`；`BUFS` 取 `max(1, min(4, (UB_BYTES - 8192) // per_slot))`，即 `K * D_PAD` 越大流水越浅，最少单缓冲。 |
| 数值行为 | 设备侧全程 FP32；softmax 用「行 max + exp 求和」两遍，top-k 权重误差约 `1e-6`，`idx` 精确匹配。概率完全并列时取专家 id 最小者优先（等价于稳定降序排序），`torch.topk` 的并列顺序由后端实现决定，可能与本算子不同。 |
| 编译 | 按 `(B, E, K, D, renorm)` 缓存编译产物，形状或参数变化会重编译。 |
| 启动缓存 | host 侧按 5 个张量地址 + 形状缓存绑定张量与 launch payload；切换 `torch.npu.stream` 后需调用 `reset_stream_cache()`。 |
| 执行 | 纯 AIV；block 数取 `vector_core_num` 并缓存，每个块认领一段连续 token。 |
| 硬件 | 编译选项固定 `--npu-arch 3510`。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from moe_route.moe_route import run

torch.npu.set_device(0)
torch.manual_seed(0)
b, e, k, d = 37, 100, 3, 63          # E / D 都不是 64 的倍数，走尾块路径
logits = torch.randn(b, e, dtype=torch.float32, device="npu:0")
experts = torch.randn(e, b, d, dtype=torch.float32, device="npu:0")   # [H, B, D]

y, w, idx = run(logits, experts, k=k, renorm=True)

p = torch.softmax(logits.cpu(), dim=-1)
ref_w, ref_i = torch.topk(p, k, dim=-1)
ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
sel = experts.cpu().permute(1, 0, 2)
rows = torch.arange(b).unsqueeze(1)
ref_y = (ref_w.unsqueeze(-1) * sel[rows, ref_i]).sum(dim=1)
print(y.shape, w.shape, idx.dtype)
print(f"w err {(w.cpu() - ref_w).abs().max().item():.2e}")
print(f"idx err {(idx.cpu() - ref_i.to(torch.int32)).abs().max().item()}")
print(f"y err {(y.cpu() - ref_y).abs().max().item():.2e}")
PY
```

```text
torch.Size([37, 63]) torch.Size([37, 3]) torch.int32
w err 5.96e-08
idx err 0
y err 2.38e-07
```
