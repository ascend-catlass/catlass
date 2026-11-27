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

## 性能优化（2026-11）

本样例在保持数学语义与精度的前提下做了以下优化（详见关联 Issue）：

1. **长序列前缀查表**：SWA/HCA 每个 task 的 tile 前缀由 host 一次计算并按内容缓存，kernel 由
   O((N/q_bn)^2) 循环改为 O(1) 查表（门控 N≥4096）。最长例 63 ms → 3.6 ms。
2. **FlashDecoding**：KV 维切分 + fp32 部分 softmax 归并（sink 只计一次），按 tile 区间的 unit 规划
   与 LPT 均衡，并支持 PA_BBND；两个 AIV 子块写回时按有效行守卫（`sub * rows < m` 在追踪期折叠，
   覆盖 `N1` 为奇数时子块行数不齐的情形），偶数 `m` 实例的指令流与优化前逐字一致。
3. **CSA 稀疏路径**：`N1=32` 场景两个 token 合批（`m` 32→64）、PA 分页装载每页单描述符、anchor
   切分与归并 3 趟合 1。
4. **host 快路径**：按输入签名缓存整份 launch 计划，合并 D2H、池化 workspace/临时缓冲；
   ragged 场景自动回退。host 开销 209 µs → 30 µs。

验证：250 例泛化精度全部通过（TND/BSND 与 TND/BSND/PA_BBND 组合、bf16/fp16、含 lse）；
本实现与同 shape 基线的 kernel 耗时几何平均为 0.82；内存检测 24 例，连续 20 次调用分配增量为
0 B，5 次重复输出逐位一致。详见关联 Issue。

### CSA 生产与 DeepSeek 泛化深度优化（Ascend 950DT，2026-11）

针对 DeepSeek CSA（`H=64/1, D=512, SWA=128, topk=512, cmp_ratio=1/2, sinks, TND/BSND, bf16/fp16`）与分页生产（`PA_BBND`）场景，在保持 5 模式通用路径不变、零环境变量默认生效的前提下完成以下优化：

1. **AIV 直写 L1 与批量 MLP Gather**：TND/BSND 与 PA 路径均由 AIV 直接将稀疏 `cmp_kv` 经 `rot2 + pitch17` 换形写入 L1 3 槽环（省去 GM 中转），TND 采用 4-MLP 批量 gather，PA 采用单描述符页表访存。
2. **8 槽 N-Split 64 宽 Cube Ring 与原生 CATLASS-DSL 向量核**：BMM2 沿 D 拆为 8×64 宽段、`P` 常驻 `L0A`、`unit_flag=0b11` 与 `FixPipe` 逐段重叠；向量侧 `vec1`（4-row 交错 `DIST_DINTLV_B32` 单趟 softmax）、`vec2`（4-wide 后更新累乘/归一化）、`sink` 初始化与 `rot2 + pitch17` KV 换形（`BlockStoreParams`）全部使用原生 CATLASS-DSL SIMD 编写，末 tile `ub_o16` 别名至 `ub_otmp_full` 使 `o_free` 提前于 `copy_out` 释放，配合边界感知的 split-pipeline 调度隐藏 `copy_out` 写回。
3. **自适应 L2 缓存策略与多 token HBM `Q` 分槽预取**：CSA 输出写回与 `Q` bypass 均采用原生 `tla.copy(..., cache_policy=CachePolicy.DISABLE)`（`l2CacheCtl=4`）避免污染 L2；主机侧按工作集下传三态 `Q_L2_BYPASS`（`0`=默认缓存；`1`=超长上下文小 T 仅 `DISABLE`；`2`=多 token 大工作集 `DISABLE` + `opt_q_pre3` 分槽预取 + `r2_j==0` 提前发射 `Q half0`），`idx` DMA 保持 L2 缓存；全仓 CSA 新增 `@tla.extern` 精简至 7 个（4 个动态步长/页表 gather、2 个 `set_loop3_para` 外提 FixPipe、1 个 `K` `l2CacheCtl=1` ND2NZ）。
4. **消除主循环 `MOV CTRL` 串行刷新**：BMM1 与 BMM2 统一采用 `ComputeOrder.M_FIRST`，反汇编确认主循环内 `MOV CTRL` 归零。
5. **消除尾行补零 L2 拥塞、小头数 `cube128` 与自适应核调度**：
   - 将 `tnd_gather_tail_rows_ub_l2_1` / `pa_gather_tail_rows_ub_l2_1` 中 64 个 AIV 并发读同一 1 KB `_kv_zero` GM 地址导致的 L2 bank 拥塞，改为原生 CATLASS-DSL SIMD 本地 UB 清零（`_ub_zero_tail_rows`），零 GM/L2 流量且保持 `0×NaN` 安全；
   - 对 `m <= 32 && g_split == 1` 新增 `opt_cube128`（BMM1/BMM2 改为 `4×128`，MMAD 16→8、FixPipe 9→5），并对 `m == 1` 开放 `aic_fix_l0c_to_ub_uf3`；
   - 新增 CPU-only、零同步的 `_csa_swiz_on_host`：tile 均匀形状保持连续分块（`SWIZ_ON=1`）保留 L2 局部性，长 `S_q` 因果三角不均（`>1.08×`）自动切换轮询（`SWIZ_ON=0`）。
6. **稳态显存修复**：ragged BSND 打包张量、HCA 奇数 head padding 与各 role workspace 池均按 role 复用，避免绑定缓存与池重分配导致的显存锯齿增长。

零环境变量验收（Ascend 950DT，10-call `msprof` median，`ub_static_bytes = 247,296 B < 253,952 B`）：
- **200 例泛化语料**（5 模式 × TND/BSND/PA_BBND × bf16/fp16，含 LSE）：精度 **200/200 PASS**，连续 20 次调用稳态显存增量 **200/200 为 0 B**；195 例与同机 `cannbot-dsl 0.8.0` 对比总量 **29.26 ms vs 31.49 ms（0.929×）**、几何均值比 **0.760**、中位比 **0.906**、胜场 **132/195（67.7%）**，其中 CSA 路由 94 例总量 **22.35 ms vs 23.07 ms（0.969×）**、几何均值比 **0.909**、61/94 胜；
- **5 模式回归**：`reg12` 11/11 PASS；
- **DeepSeek 生产与非 PA 泛化**（双 seed `1000/1001` 全部 PASS）：`ProdA 30.07 µs`、`ProdB 30.18 µs`、decode `T4 14.94 µs` / `G1 14.99 µs`、spec-decode `T1~T3 27.43~27.56 µs` / `G2 28.12 µs`、prefill `T5_w128 110.42 µs` / `G6 110.73 µs` / `G3 116.35 µs` / `T6 207.80 µs` / `G7 215.97 µs`。

### 后续可优化方向与已排除路径

**可继续推进的方向（按性价比排序）：**

1. **P1 — `BMM2` 首段 `V0` 提前装载 + 首 token `Q` 分半启动**：`BMM2 d_tile=0` 所需的 `V0` 早在 `BMM1` 阶段已驻留 L1，可在 `wait_p_ready` 之前预装至 `L0B` 隐藏搬运延迟；首 token 冷启动阶段仅等待 `Q half0` 即可开始 `c=0..3`，同时并行装载 `Q half1`。预期全用例再降 `0.5~1.5 µs`，改动局部且风险低。
2. **P1 — `PA_BBND` 页表标量与 `aiv_scalar` 专项压缩**：当前 `ProdA/ProdB` 的 `aiv_scalar`（`17.9 µs`）显著高于同算力 `TND`（`11.5 µs`），可将 `block_table` 查表与页号/页内偏移计算进一步下沉合并至 C++ extern 或向量化。预期 `ProdA/ProdB` 再降 `1.0~1.8 µs`（逼近 `28.5 µs`）。
3. **P2 — `SWA=128`（`ori_mask_mode=4`）接入 FlashDecoding**：当前 FD（KV 维切分）仅对 `ori_mask_mode=3` 生效；若泛化场景涵盖 `T < 32` 的低并发 decode（如 `B=1~16, S=1`），放开 `mask_mode=4` 的 FD 切分可将空闲 AIC 核利用起来，预期提速 `30%~50%`。
4. **P3（低性价比，暂不推荐）— `Q` 384+128 非对称 L1 划分**：需将 `l1_p` 从双缓冲 `32 KB` 压缩至单缓冲 `16 KB` 并重构 `P` 环跨核同步，仅对 64K 长上下文 `G4` 约 `0.8~1.5 µs` 潜在收益，复杂度与稳定性风险较高。

**已消融验证并排除的方向（避坑记录）：**

- `SCHED=7/8/9`（将 `vec2` 提前至 `Stage 2.5 vsstb` 之前）：推迟下一轮 `vec1`（`p_ready` 更关键），端到端全线退化 `+13~15 µs`。
- `ori_kv` 卸载至 `AIV -> L1`：虽降低 `AIC MTE2`，但推高 `AIV MTE2` 约 `+16 µs`，端到端反慢 `+1.1~2.4 µs`（证实 AIC 硬件 `nd2nz` 直读稠密 `ori_kv` + AIV 处理稀疏 `cmp_kv` 的异构分工更优）。
- `cmp_sparse_indices`（`idx` DMA）启用 L2 bypass：256 B 小包退化为 HBM miss，`G7` 慢 `+6.0 µs`。
- `R2_AIC_SWIZZLE=4`（4 路核重排）与 `R2_AIC_SWIZZLE=0`（轮转）：破坏 L2 局部性，长上下文退化 `+2.3~6.4 µs`。
- Half-D 双跨核 flag 对（`o_ready_hi / o_free_hi`）：在 `SCHED=4` 下可降 `c_fix`，但在更优的 `SCHED=6` 下无端到端净收益。
