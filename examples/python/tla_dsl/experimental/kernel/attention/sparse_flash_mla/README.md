# SparseFlashMLA Example Readme

## 功能说明

- 算子功能：稀疏 MLA（Multi-head Latent Attention）注意力前向。结构参照
  arch35 `SparseFlashMla` 算子的分支划分与流水形态，`out` 与参考算子逐位一致
  （150 例回归口径：50 条官方 workload + 100 条扩展语料）。支持
  ori / cmp 双路 KV（compressed KV 按 `cmp_ratio` 压缩）、token 级稀疏索引、
  分页 KV cache、dense / causal / 滑窗掩码，并可写回 FP32 的 lse。
- 计算公式：对每个 query row `r` 与 head `h`，可见列集合由 ori 段与 cmp 段拼接而成：

  ```text
  dense ori 段：
    ori_mask_mode=0  [0, T2)
    ori_mask_mode=3  [0, min(T2 - S1 + r + 1, T2))            # right-down causal
    ori_mask_mode=4  [max(thr - win_left - 1, 0),
                      min(thr + win_right, T2))               # 滑窗
  sparse ori 段：按 ori_sparse_indices 前缀取，-1 终止，按段边界过滤
  cmp 段：[0, thr_c)，causal 时 thr_c = min((restored_cmp - S1 + r + 1) // cmp_ratio, T2c)
  ```

  $$
  \begin{aligned}
  s_j &= scale \cdot q[r,h,:] \cdot k_j \\
  m &= \max_j s_j \quad (\text{有 sinks 时 } m = \max(m,\ sink_h)) \\
  p_j &= e^{s_j - m} \\
  O[r,h,:] &= \frac{\sum_j \operatorname{round}_{io}(p_j)\, k_j}{\sum_j p_j + e^{sink_h - m}}
  \end{aligned}
  $$

  P 在 PV 前按输入 dtype 舍入、PV 用 FP32 累加；sinks 只进入 max 与分母。可见集为空的行
  输出 0 且不写 lse；lse 为 `m + log(denom + 1e-10)`。

- 支持产品型号：Ascend 950PR / 950DT（`--npu-arch 3510`）

## 实现结构

样例按工作负载模式拆成 **5 个分支 kernel**，由 host 侧按输入形态
自动选路，每次 `run` 只 launch 其中一个：

| 分支 kernel | 输入形态 |
| --- | --- |
| `sparse_flash_mla_swa_kernel` | 无 `cmp_kv`、无稀疏索引（dense / causal / 滑窗） |
| `sparse_flash_mla_hca_kernel` | 有 `cmp_kv`、无稀疏索引 |
| `sparse_flash_mla_ori_sparse_kernel` | 只有 `ori_sparse_indices` |
| `sparse_flash_mla_csa_kernel` | 只有 `cmp_sparse_indices` |
| `sparse_flash_mla_ori_cmp_sparse_kernel` | `ori_sparse_indices` 与 `cmp_sparse_indices` 同时给出 |

5 个 kernel 共享 3 个模块：`smla_csa_common`（常量、缓冲分配、同步对象、物理地址
表尺寸等公共定义）、`smla_csa_block_cube`（AIC 侧 QK / PV 与 L1 装载）、
`smla_csa_block_vector`（AIV 侧 softmax、O 累加、稀疏 gather 与地址表生成）。

关键实现形态：

- **4 深扁平流水**：tile 沿 `stage0 load(x) → stage1 vec1(x-1) → stage2 bmm1(x-2) →
  stage3 bmm2/vec2(x-3)` 连续推进，tile 序跨任务连续，任务边界不做整体同步；
  kernel 尾部用哨兵步补 `PRELOAD_NUM` 个 body 排空。AIC / AIV 通过 cross-core flag
  与 L1 / L0 / GM 槽位环形复用交接，全程无整核 SyncAll（除 Init 阶段的一次性清零）。
- **AIV 稀疏 gather + GM 暂存槽**：稀疏路由下 AIV 按索引把 KV 行搬进 workspace 的
  GM 暂存槽（每槽 `3 × 128 × 512` 元素），AIC 再从槽装 L1，避免 AIC 逐行寻址；
  非 PA 且 `is_vec_s2phyaddr=1` 的实例由 AIV 预生成 int64 物理地址表，gather 只读表。
- **split-G**：`N1 > 64` 时按 `g_split=2` 编译，相邻两个 block 处理同一 token 的
  两个 G 半区并共享一份 GM 暂存；此时每个 AIC 对的 4 个 AIV 各写 1/4 象限，跨块
  保护用与参考算子同构的 MTE2 节流屏障。
- **约束是编译期折叠的**：布局、掩码模式、`g_split`、`is_vec_s2phyaddr` 等均由
  constexpr 分支在追踪期选定，每个实例的指令流只含自己那条路径。

## 参数说明

```python
out, lse = sparse_flash_mla.run(
    q, ori_kv, cmp_kv,
    ori_sparse_indices, cmp_sparse_indices,
    ori_block_table, cmp_block_table,
    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
    seqused_q, seqused_ori_kv, seqused_cmp_kv,
    cmp_residual_kv, ori_topk_length, cmp_topk_length,
    sinks, metadata, softmax_scale, cmp_ratio,
    ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
    layout_q, layout_kv, return_softmax_lse,
)
```

记 `B` 为 batch，`T1` / `S1` 为 query 行数，`T2` / `S2` 为 ori KV token 数，`T2c` / `S2c`
为 cmp KV token 数，`N1` 为 query head 数，`D = 512`。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | BSND `[B, S1, N1, 512]` / TND `[T1, N1, 512]` | `float16` / `bfloat16` | 连续行优先张量 |
| `ori_kv` | Input | BSND `[B, S2, 1, 512]` / TND `[T2, 1, 512]` / PA `[blocks, page, 1, 512]` | 与 `q` 相同 | 原始 KV；MLA 只有 1 个 KV head |
| `cmp_kv` | Input，可选 | 与 `ori_kv` 同布局 | 与 `q` 相同 | compressed KV；省略时不得给 cmp 掩码与 cmp 索引 |
| `ori_sparse_indices` | Input，可选 | 与 `q` 同前缀，末维为索引宽度 | `int32` | 每行选中的 ori token；`-1` 终止，越界跳过 |
| `cmp_sparse_indices` | Input，可选 | 与 `q` 同前缀，末维为索引宽度 | `int32` | 每行选中的 cmp token；要求给出 `cmp_kv` |
| `ori_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 ori KV 的页映射 |
| `cmp_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 cmp KV 的页映射 |
| `cu_seqlens_q` | Input，可选 | `[B + 1]` | `int32` | TND query 必需 |
| `cu_seqlens_ori_kv` | Input，可选 | `[B + 1]` | `int32` | 非分页 TND ori KV 必需 |
| `cu_seqlens_cmp_kv` | Input，可选 | `[B + 1]` | `int32` | 非分页 TND cmp KV 必需 |
| `seqused_q` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 query 长度 |
| `seqused_ori_kv` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 ori KV 长度；分页路径必需 |
| `seqused_cmp_kv` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 cmp KV 长度；分页路径必需 |
| `cmp_residual_kv` | Input，可选 | `[B]` | `int32` | cmp causal 阈值使用的残差长度 |
| `ori_topk_length` | Input，可选 | BSND `[B, S1]` / TND `[T1]` | `int32` | 每行 ori 索引的有效前缀长度 |
| `cmp_topk_length` | Input，可选 | BSND `[B, S1]` / TND `[T1]` | `int32` | 每行 cmp 索引的有效前缀长度 |
| `sinks` | Input，可选 | `[N1]` | `float32` | 每 head 的 sink 值，只进 max 与分母 |
| `metadata` | Input | `[1024]` | `int32` | 预留元数据张量；当前样例只校验，不参与计算 |
| `softmax_scale` | Input | 标量 | Python `float` | QK 缩放因子 |
| `cmp_ratio` | Input | 标量 | Python `int` | compressed KV 压缩比，正整数 |
| `ori_mask_mode` | Input | 标量 | Python `int` | `0` 全量 / `3` causal / `4` 滑窗 |
| `cmp_mask_mode` | Input | 标量 | Python `int` | `0` 全量 / `3` causal |
| `ori_win_left` / `ori_win_right` | Input | 标量 | Python `int` | 滑窗左右宽度，`-1` 表示不限制 |
| `layout_q` | Input | 标量 | Python `str` | `"BSND"` 或 `"TND"` |
| `layout_kv` | Input | 标量 | Python `str` | `"BSND"`、`"TND"` 或 `"PA_BBND"` |
| `return_softmax_lse` | Input | 标量 | Python `bool` | 是否写回 `lse` |
| `out` | Output | 与 `q` 相同 | 与 `q` 相同 | attention 结果 |
| `lse` | Output | TND `[1, T1, N1]` / BSND `[B, 1, S1, N1]` | `float32` | `max + log(sum + 1e-10)`；关闭时为空张量 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| head 维度 | 固定 512 |
| head 数 | `N1 ≥ 1`；KV head 数固定为 1 |
| dtype | `q`、`ori_kv`、`cmp_kv` 必须同为 `float16` 或 `bfloat16`；索引与长度张量为 `int32`；`sinks` 为 `float32` |
| 布局 | `layout_q ∈ {BSND, TND}`；`layout_kv ∈ {BSND, TND, PA_BBND}`；分页 KV 与 query 布局相互独立（`BSND` / `TND` query 都可配分页 KV）；非分页时 `layout_q == layout_kv` |
| 分页 KV | 需要 `block_table` 与对应 `seqused_*`；ori / cmp 两侧页大小各自独立，分别由对应 `KV` 张量的第 1 维给出 |
| 掩码 | `ori_mask_mode ∈ {0, 3, 4}`，`cmp_mask_mode ∈ {0, 3}`；无 `cmp_kv` 时 `cmp_mask_mode` 必须为 0 |
| 压缩比例 | `cmp_ratio ≥ 1`；cmp causal 阈值按 `restored_cmp = T2c * cmp_ratio + residual` 计算 |
| 张量布局 | `q` 必须连续；所有张量位于同一 NPU |
| ragged BSND | `layout_q == BSND` 且 `seqused_q` 有小于 `S1` 的 batch 时，入口按有效行成包送入 kernel、跑完散射回物理位置；`out` 与 `lse` 的 padding 行写 0。非分页 `BSND` KV 的 `seqused_ori_kv` / `seqused_cmp_kv` 同理 |
| 执行形态 | 5 个 kernel 按输入形态自动选路，每次 `run` 只 launch 一个：无 `cmp_kv` 且无稀疏索引 → SWA；有 `cmp_kv` 无稀疏索引 → HCA；只有 `ori_sparse_indices` → ORI_SPARSE；只有 `cmp_sparse_indices` → CSA；两者都有 → ORI_CMP_SPARSE。`N1 > 64` 时按 `g_split=2` 编译，相邻两核共享 GM 暂存 |
| 稀疏路由 | 给出 `ori_sparse_indices` 与 `cmp_kv` 却缺 `cmp_sparse_indices` 时直接报错，不会静默退化到其它路由 |
| topk 长度 | 稀疏路由必须给对应 `*_topk_length`：CSA 需要 `cmp_topk_length`，ORI_SPARSE 需要 `ori_topk_length`，ORI_CMP_SPARSE 两者都要；缺失即报错 |
| lse | `return_softmax_lse=True` 时 5 个分支 kernel 都写回 FP32 的 `lse`；关闭时 `lse` 返回空张量，内部仍按同形分配以保证 UB 划分一致 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`。下面的 TND
稠密示例（SWA 路由）在 950PR 上编译并执行：

```python
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

import sparse_flash_mla

T1, T2, N1, D = 192, 1024, 32, 512
device = "npu:0"
q = torch.randn(T1, N1, D, dtype=torch.bfloat16, device=device)
ori_kv = torch.randn(T2, 1, D, dtype=torch.bfloat16, device=device)
cu_seqlens_q = torch.tensor([0, T1], dtype=torch.int32, device=device)
cu_seqlens_kv = torch.tensor([0, T2], dtype=torch.int32, device=device)
metadata = torch.zeros(1024, dtype=torch.int32, device=device)

out, lse = sparse_flash_mla.run(
    q, ori_kv, None,            # cmp_kv
    None, None,                 # ori/cmp_sparse_indices
    None, None,                 # ori/cmp_block_table
    cu_seqlens_q, cu_seqlens_kv, None,
    None, None, None,           # seqused_q / seqused_ori_kv / seqused_cmp_kv
    None, None, None,           # cmp_residual_kv / ori/cmp_topk_length
    None, metadata, D ** -0.5,  # sinks / metadata / softmax_scale
    1, 0, 0, -1, -1,            # cmp_ratio / ori,cmp_mask_mode / win_left,right
    "TND", "TND", False,        # layout_q / layout_kv / return_softmax_lse
)
print(out.shape, out.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([192, 32, 512]) torch.bfloat16
```

其他形态只需替换参数组合，host 会自动切到对应分支 kernel：`ori_mask_mode=3` 打开
causal、`=4` 配合 `ori_win_left` / `ori_win_right` 打开滑窗；给出 `cmp_kv` 与
`cmp_ratio` 走 HCA 路由；给出 `ori_sparse_indices` / `cmp_sparse_indices` 走对应稀疏
路由（需同时给 `*_topk_length`）；`layout_kv="PA_BBND"` 配合 `block_table` /
`seqused_*` 走分页 KV；任意路由下 `return_softmax_lse=True` 都返回 FP32 的 `lse`。

## 文件组织

```text
sparse_flash_mla/
├── README.md                                  # 本文件
├── sparse_flash_mla.py                        # host 入口：参数校验、选路、tiling/workspace、launch
├── smla_csa_common.py                         # 公共定义：常量、缓冲分配、同步对象、地址表尺寸
├── smla_csa_block_cube.py                     # AIC 侧：QK / PV、L1 装载与槽位管理
├── smla_csa_block_vector.py                   # AIV 侧：softmax、O 累加、稀疏 gather、物理地址表
├── sparse_flash_mla_swa_kernel.py             # SWA 分支 kernel（无 cmp、无稀疏索引）
├── sparse_flash_mla_hca_kernel.py             # HCA 分支 kernel（有 cmp、无稀疏索引）
├── sparse_flash_mla_csa_kernel.py             # CSA 分支 kernel（仅 cmp 稀疏索引）
├── sparse_flash_mla_ori_sparse_kernel.py      # ORI_SPARSE 分支 kernel（仅 ori 稀疏索引）
└── sparse_flash_mla_ori_cmp_sparse_kernel.py  # ORI_CMP_SPARSE 分支 kernel（ori + cmp 双稀疏索引）
```
