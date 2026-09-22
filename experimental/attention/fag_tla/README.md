# Atlas A2 FlashAttentionScoreGrad (FAG TLA)

> **注意**：本样例位于 `experimental/` 目录下。如需独立编译运行，请先将样例目录复制到
> `examples/`，并在 `examples/CMakeLists.txt` 的 `EXAMPLE_ATLASA2` 列表中添加 `fag_tla`.

## 功能说明

本样例实现 FlashAttention 反向算子（FlashAttentionScoreGrad），完成注意力反向传播的
梯度计算：

```text
(dq, dk, dv) = FAG(dout, q, k, v, out, softmax_lse, cu_seq_qlen, cu_seq_kvlen)
```

- 输入：输出梯度 `dout`、前向 query/key/value、前向输出 `out`、前向 log-sum-exp
  `softmax_lse`，以及 TND 变长布局的前缀和序列长度表。
- 输出：`dq`/`dk`/`dv` 三路梯度。
- 数据布局为 TND（变长序列），dtype 支持 half/bf16，head_dim 支持 64/128/192/256。
- 在单次 MIX kernel（AIC + AIV）中完成三路 matmul 与 softmax 梯度计算，走确定性 DQKV
  路径（IS_DTM=1）。

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| `batch` | 变长序列条数 |
| `qSeqlen` | Q 序列长度（TND 下由前缀和表覆盖） |
| `qHeadNum` | Q 头数 |
| `qkHeadDim` | Q/K 每头维度（64/128/192/256） |
| `kvSeqlen` | KV 序列长度（TND 下由前缀和表覆盖） |
| `kvHeadNum` | KV 头数 |
| `vHeadDim` | V 每头维度 |
| `window_size_left` / `window_size_right` | 滑动窗口（当前无 mask 时不生效） |
| `layout` | 布局标签，本样例固定 `TND` |
| `isDeterministic` | 是否确定性 tiling（TND 路径下置 1） |
| `--dtype` | `half` 或 `bf16` |
| `--datapath` | 输入数据目录 |
| `--device` | NPU 设备编号（默认 0） |

## 约束说明

- 目标架构为 Atlas A2（`CATLASS_ARCH=2201`）。
- TND 布局下 `cu_seq_qlen` / `cu_seq_kvlen` 为 int64 前缀和（cumsum）数组。
- `scale = 1 / sqrt(qkHeadDim)`，`keepProb = 1.0`（无 dropout）。

## 代码组织

算子目录只保留主入口 host 代码、kernel 代码、tiling 代码、数据生成脚本和公共定义代码；
epilogue 等组件统一放在 `include/catlass/` 下，由 `fag_tla_kernel.cpp` 直连引用
（FAG epilogue 依赖算子目录的 `kernel_common_fag.hpp`，故未注册进全局聚合头 `block_epilogue.hpp`）。

```text
fag_tla
├── CMakeLists.txt
├── fag_main.cpp              # host main：tiling + kernel launch + golden 对比
├── fag_tla_kernel.cpp        # FlashAttentionScoreGrad kernel
├── fag_tiling.cpp/.h         # host 侧 tiling
├── softmax_tiling.cpp        # softmax 梯度 tiling
├── kernel_common_fag.hpp     # kernel/host/epilogue 公共定义（FAGTilingData、DBParams、布局常量、FAGKernelParams 等）
├── gen_data.py               # 输入/输出数据生成
└── test_87_fag_tla.py        # optest 测试用例（不会被全局 CI 自动收集）

include/catlass/epilogue/block/   # 本算子贡献的组件
├── fag_sfmg.h                           # SoftmaxGradFront 辅助实现
├── block_epilogue_fag_pre.hpp           # workspace 清零 epilogue
├── block_epilogue_fag_sfmg.hpp          # softmax 梯度 epilogue
├── block_epilogue_fag_op.hpp            # 反向主计算 epilogue（SabVec）
├── block_epilogue_fag_post.hpp          # fp32 workspace 回写 epilogue
└── block_epilogue_fag_deterministic_add.hpp  # 确定性累加 epilogue
```

## 使用示例

1. 复制到 `examples/` 并加入 `EXAMPLE_ATLASA2` 列表后编译：

   ```bash
   bash scripts/build.sh fag -DCATLASS_ARCH=2201
   ```

2. 生成数据并运行：

   ```bash
   python gen_data.py
   ./output/bin/fag <batch> <qSeqlen> <qHeadNum> <qkHeadDim> <kvSeqlen> <kvHeadNum> <vHeadDim> \
       <window_left> <window_right> TND 1 --dtype half --datapath <data_dir>
   ```

   程序输出 `Compare dq/dk/dv success.` 表示与 golden 对齐。

## optest 测试用例

`test_87_fag_tla.py` 保存在本 experimental 样例目录，不会被全局 CI 自动收集。复验时先
按 `tests/optest/README.md` 完成编译与安装，再把 `test_87_fag_tla.py` 复制到
`tests/optest/tests/` 后执行：

```bash
pytest tests/test_87_fag_tla.py -v
```