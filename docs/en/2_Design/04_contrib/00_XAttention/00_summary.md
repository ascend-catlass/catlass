# Flash Attention New Templates Summary

This directory (04_contrib/00_XAttention) collects the design documents for **23 Flash Attention related templates newly added** to the personal repository on top of the CATLASS main repository (10 GEMM + 13 Epilogue). The 23 templates are divided into five series by operator form, with one design document for each series.

## Series Document Index

| Document | Series | Scenario | New templates |
| --- | --- | --- | --- |
| [01_fa_unshared_kernel.md](./01_fa_unshared_kernel.md) | FA Unshared | Flash Attention for non-shared expert (unshared) scenarios, the basic MHA/GQA form | 3 |
| [02_fai_split_row_kernel.md](./02_fai_split_row_kernel.md) | FAI SplitRow | Inference Flash Attention, a split-KV variant that splits by row and avoids Combine | 4 |
| [03_xfai_kernel.md](./03_xfai_kernel.md) | XFAI | Atlas A2 inference Flash Attention with AIC QK → AIV Softmax → AIC PV cross-core pipeline, supporting Paged KV | 5 |
| [04_fd_kernel.md](./04_fd_kernel.md) | FD | A deeply performance-evolved version of XFAI (FlashAttention-Decode form), including a general CombineScale | 3 |
| [05_xa_tla_kernel.md](./05_xa_tla_kernel.md) | XA TLA | Atlas A5 (Ascend 950) TLA instruction version of Flash Attention, in both shared/unshared forms | 8 |

## New Template List

### GEMM side (10, `include/catlass/gemm/block/`)

| Template file | Series |
| --- | --- |
| `block_mmad_unshared_fa_qk.hpp` | FA Unshared |
| `block_mmad_unshared_fa_pv.hpp` | FA Unshared |
| `block_mmad_fai_qk_split_row.hpp` | FAI SplitRow |
| `block_mmad_fai_pv_split_row.hpp` | FAI SplitRow |
| `block_mmad_xfai_qk.hpp` | XFAI |
| `block_mmad_xfai_pv.hpp` | XFAI |
| `block_mmad_xa_shared_qk_tla.hpp` | XA TLA (shared) |
| `block_mmad_xa_shared_pv_tla.hpp` | XA TLA (shared) |
| `block_mmad_xa_unshared_qk_tla.hpp` | XA TLA (unshared) |
| `block_mmad_xa_unshared_pv_tla.hpp` | XA TLA (unshared) |

### Epilogue side (13, `include/catlass/epilogue/block/`)

| Template file | Series |
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

## Series Quick Reference

- **01 FA Unshared**: two `BlockMmad` for QK/PV plus one UnsharedSoftmax `BlockEpilogue`, the simplest introductory form;
- **02 FAI SplitRow**: a row-split split-KV variant that avoids merging (`CopySumMax` records row statistics instead of the Combine step);
- **03 XFAI**: AIC/AIV cross-core pipeline "QK → OnlineSoftmax → PV → RescaleO", `l1BufAddrStart` shared L1, `PAGED_CACHE_FLAG` native paged KV, `CombineScale` closing out the split-KV merge;
- **04 FD**: a deeply performance-evolved version of XFAI, with optimizations such as partitioning dm by cycle, and `CombineScale` generalized to `EpilogueAtlasA2CombineScale`;
- **05 XA TLA**: a TLA instruction implementation for Atlas A5 (Ascend 950), with QK/PV/Softmax templates for the shared and unshared expert forms.

For the DispatchPolicy definitions of each series, the algorithm design (data layout / pipeline organization / cross-core synchronization), the differences between series, and real engineering usage examples based on xllm-ops (https://gitcode.com/xLLM-AI/xllm_ops, `x_attention` / `x_flash_attention_infer`), see the corresponding design documents.
