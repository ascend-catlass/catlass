# Flash Attention 新增模板总结

本目录（04_contrib/00_XAttention）收录个人仓在 CATLASS 主干仓基础上**新增的 23 个 Flash Attention 相关模板**（10 个 GEMM + 13 个 Epilogue）的设计文档。23 个模板按算子形态划分为五个系列，每个系列配备一篇设计文档。

## 系列文档索引

| 文档 | 系列 | 场景 | 新增模板 |
| --- | --- | --- | --- |
| [01_fa_unshared_kernel.md](./01_fa_unshared_kernel.md) | FA Unshared | 非 shared expert（unshared）场景的 Flash Attention，MHA/GQA 基础形态 | 3 个 |
| [02_fai_split_row_kernel.md](./02_fai_split_row_kernel.md) | FAI SplitRow | 推理 Flash Attention，按行切分免 Combine 的 split-KV 变体 | 4 个 |
| [03_xfai_kernel.md](./03_xfai_kernel.md) | XFAI | Atlas A2 推理 Flash Attention，AIC QK→AIV Softmax→AIC PV 跨核流水，支持 Paged KV | 5 个 |
| [04_fd_kernel.md](./04_fd_kernel.md) | FD | XFAI 的深度性能演化版（Flash Attention-Decode 形态），含通用 CombineScale | 3 个 |
| [05_xa_tla_kernel.md](./05_xa_tla_kernel.md) | XA TLA | Atlas A5（Ascend 950）TLA 指令版 Flash Attention，shared/unshared 双形态 | 8 个 |

## 新增模板清单

### GEMM 侧（10 个，`include/catlass/gemm/block/`）

| 模板文件 | 所属系列 |
| --- | --- |
| `block_mmad_unshared_fa_qk.hpp` | FA Unshared |
| `block_mmad_unshared_fa_pv.hpp` | FA Unshared |
| `block_mmad_fai_qk_split_row.hpp` | FAI SplitRow |
| `block_mmad_fai_pv_split_row.hpp` | FAI SplitRow |
| `block_mmad_xfai_qk.hpp` | XFAI |
| `block_mmad_xfai_pv.hpp` | XFAI |
| `block_mmad_xa_shared_qk_tla.hpp` | XA TLA（shared） |
| `block_mmad_xa_shared_pv_tla.hpp` | XA TLA（shared） |
| `block_mmad_xa_unshared_qk_tla.hpp` | XA TLA（unshared） |
| `block_mmad_xa_unshared_pv_tla.hpp` | XA TLA（unshared） |

### Epilogue 侧（13 个，`include/catlass/epilogue/block/`）

| 模板文件 | 所属系列 |
| --- | --- |
| `block_epilogue_fa_unshared_softmax.hpp` | FA Unshared |
| `block_epilogue_online_softmax_copy_glm.hpp` | FAI SplitRow |
| `block_epilogue_rescale_o_no_split_row.hpp` | FAI SplitRow |
| `block_epilogue_xfai_online_softmax.hpp` | XFAI |
| `block_epilogue_xfai_rescale_o.hpp` | XFAI |
| `block_epilogue_xfai_combine_scale.hpp` | XFAI |
| `block_epilogue_online_softmax_FD.hpp` | FD |
| `block_epilogue_rescale_o_FD.hpp` | FD |
| `block_epilogue_combine_scale.hpp` | FD |
| `block_epilogue_xa_shared_softmax_ascend950.hpp` | XA TLA |
| `block_epilogue_xa_unshared_softmax_ascend950.hpp` | XA TLA |
| `block_epilogue_xa_shared_rescale_ascend950.hpp` | XA TLA |
| `block_epilogue_xa_combine_scale_ascend950.hpp` | XA TLA |

## 系列速查

- **01 FA Unshared**：QK/PV 两个 `BlockMmad` + 一个 UnsharedSoftmax `BlockEpilogue`，结构最简的入门形态；
- **02 FAI SplitRow**：按行切分的 split-KV 免合并变体（`CopySumMax` 记录行统计量替代 Combine 步骤）；
- **03 XFAI**：AIC/AIV 跨核流水"QK→OnlineSoftmax→PV→RescaleO"，`l1BufAddrStart` 共享 L1，`PAGED_CACHE_FLAG` 原生分页 KV，`CombineScale` 收口 split-KV 合并；
- **04 FD**：XFAI 的深度性能演化版，dm 按周期分区等优化，`CombineScale` 通用化为 `EpilogueAtlasA2CombineScale`；
- **05 XA TLA**：面向 Atlas A5（Ascend 950）的 TLA 指令实现，shared/unshared 两种 expert 形态各配 QK/PV/Softmax 模板。

各系列的 DispatchPolicy 定义、算法设计（数据布局/流水组织/跨核同步）、系列间差异对比与基于 xllm-ops（https://gitcode.com/xLLM-AI/xllm_ops，`x_attention` / `x_flash_attention_infer`）的真实工程使用示例，详见对应设计文档。
