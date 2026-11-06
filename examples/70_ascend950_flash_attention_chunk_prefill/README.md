# FlashAttentionChunkPrefill Example Readme

## 功能说明

- 算子功能：完成 Ascend 950 上的 Flash Attention chunk-prefill 计算，仅支持 mask + PA（分页 KV cache）场景，支持 GQA 与变长序列
- 计算公式：

  $$
    \begin{aligned}
    O &= \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{d}} + M\right) \times V
    \end{aligned}
  $$

  其中 $Q$ 是形如 `(batch*qSeqlen, numHeads, qkHeadSize)` 的查询矩阵，$K$、$V$ 为分页 KV cache，形如 `(numBlocks, kvHeads, blockSize, headDim)`，$d$ 为 `qkHeadSize`，$M$ 为底对齐的 causal 掩码（仅支持 mask 场景，`maskType` 固定为 1，不支持无掩码）；GQA 场景下 `numHeads` 为 `kvHeads` 的正整数倍。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `query` | Input | `(batch*qSeqlen, numHeads, qkHeadSize)` | `fp16/bf16` | TND 布局，layout 固定 `RowMajor` |
| `keyCache` | Input | `(numBlocks, kvHeads, blockSize, qkHeadSize)` | `fp16/bf16` | layout 固定 `ColumnMajor` |
| `valueCache` | Input | `(numBlocks, kvHeads, blockSize, vHeadSize)` | `fp16/bf16` | layout 固定 `RowMajor` |
| `batch` | Attr | - | `int` | `batch ≥ 1`，推荐 `1 ≤ batch ≤ 16` |
| `qSeqlen` | Attr | - | `int` | `qSeqlen ≤ kvSeqlen`，推荐 `1 ≤ qSeqlen ≤ 16384` |
| `kvSeqlen` | Attr | - | `int` | `kvSeqlen ≥ qSeqlen`，推荐 `1 ≤ kvSeqlen ≤ 16384` |
| `kvHeads` | Attr | - | `int` | `kvHeads ≥ 1` 且须整除 `numHeads`（GQA），推荐 `1 ≤ kvHeads ≤ 8` |
| `numHeads` | Attr | - | `int` | `numHeads` 为 `kvHeads` 的正整数倍，`query.size(1)` 须与之一致，推荐 `1 ≤ numHeads ≤ 8` |
| `blockSize` | Attr | - | `int` | 取值 {128, 256, 512, 1024}，与 `key/valueCache.size(2)` 一致 |
| `maskType` | Attr | - | `int` | 固定为 1（causal，chunk-prefill 底对齐），仅支持 mask 场景，不支持无 mask |
| `qkHeadSize` | Attr | - | `int` | 取值 {64, 128, 192}，与 `query.size(2)`、`keyCache.size(3)` 一致 |
| `vHeadSize` | Attr | - | `int` | 取值 {64, 128}，与 `valueCache.size(3)` 一致 |
| `numBlocks` | Attr | - | `int` | 等于 `keyCache.size(0)`，且 `numBlocks ≥ batch * ceil(kvSeqlen / blockSize)` |
| `blockTable` | Input | `(batch, ceil(kvSeqlen/blockSize))` | `int32` | 元素为 KV cache 物理块号，条数 `≥ batch * ceil(kvSeqlen / blockSize)` |
| `actualSeqLengths` | Input | `(batch+1,)` | `int64` | Q 长度累积前缀和（cumsum），末项 = 总 token 数 = `query.size(0)` |
| `actualSeqLengthsKV` | Input | `(batch,)` | `int64` | 逐 batch KV 长度 |
| `output` | Output | `(batch*qSeqlen, numHeads, vHeadSize)` | `fp16/bf16` | layout 固定 `RowMajor` |

## 使用范围说明

本样例仅支持 mask + PA 场景：掩码固定为 causal（`maskType=1`，chunk-prefill 底对齐），不支持无掩码；KV cache 固定为分页布局并经 `blockTable` 寻址，不支持非分页（dense）场景。已支持特性如下：

|            特性             |          对应参数          |
| :-------------------------: | :------------------------: |
|          数据类型           |    dtype="half"/"bf16"     |
|      不同batch序列可变      |      isVariedLen=0/1       |
|          blocksize          | blocksize=128/256/512/1024 |
|         qk_head_dim         |   qkHeadSize=64/128/192    |
|         v_head_dim          |      vHeadSize=64/128      |
| kvlayout支持PAGE_ND/PAGE_NZ |   cacheLayout="nd"/"nz"    |

推荐范围：

- 序列长度满足 `1 ≤ qSeqlen ≤ kvSeqlen ≤ 16384`。上限依据：单 case 数据量 `batch*qSeqlen*numHeads*qkHeadSize + batch*ceil(kvSeqlen/blockSize)*blockSize*kvHeads*(qkHeadSize+vHeadSize)` 随 seqlen 线性放大，kernel 中间 workspace（qkOut/smOnline/pvOut/Update）同步膨胀，seqlen 再增加单case数据量即到GB 级，易 OOM（aclError：207001）。
- `batch` 满足 `1 ≤ batch ≤ 16`：kernel 无硬上限，但与 seqlen/heads 共享同一条数据量公式，开大 batch 需等比缩小 seqlen 或 numHeads。
- 头数满足 `1 ≤ kvHeads ≤ numHeads ≤ 8` 且 `numHeads % kvHeads == 0`：除整除关系外无其它硬约束，heads 放大等价于放大 Q 数据量。
- `qkHeadSize`、`vHeadSize`、`blockSize` 为 kernel 模板实例化的离散取值（{64,128,192} / {64,128} / {128,256,512,1024}），无连续范围。
- `query`/`keyCache`/`valueCache` 张量的数值范围推荐在 `[-1, 1]`以内，避免过大的动态范围致使绝对误差超阈值上限。

## 使用示例

### 命令行参数

先执行 `gen_data.py` 生成测试数据，再执行算子可执行文件，两者的 shape 参数须一致。

```bash
python gen_data.py [batch] [qSeqlen] [kvSeqlen] [numHeads] [kvHeads] [qkHeadSize] [vHeadSize] [isVariedLen] [dtype] [numBlocks] [innerPrec] [blockSize] [cacheLayout]
./70_ascend950_flash_attention_chunk_prefill [batch] [qSeqlen] [kvSeqlen] [numHeads] [kvHeads] [qkHeadSize] [vHeadSize] [isVariedLen] [numBlocks] [blockSize] [--dtype DTYPE] [--cache_layout CACHE_LAYOUT] [--device DEVICE_ID]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | batch size |
| `qSeqlen` | 无 | 每 batch 的 query 序列长度 |
| `kvSeqlen` | 无 | 每 batch 的 KV 序列长度 |
| `numHeads` | 无 | query 头数，须为 `kvHeads` 的正整数倍（GQA） |
| `kvHeads` | 无 | KV 头数 |
| `qkHeadSize` | 无 | Q/K 头维，取值 64/128/192 |
| `vHeadSize` | 无 | V 头维，取值 64/128 |
| `isVariedLen` | 无 | 0=各 batch 等长；1=各 batch 变长 |
| `dtype` | `half` | 数据类型 `half`/`bf16`（gen_data.py 为位置参数，可执行文件为 `--dtype` 选项） |
| `numBlocks` | 无 | KV cache 物理块数，`≥ batch * ceil(kvSeqlen / blockSize)` |
| `innerPrec` | 无 | 仅 gen_data.py 使用，对应 kernel 逻辑固定为 0 |
| `blockSize` | 无 | KV cache 块大小，取值 128/256/512/1024 |
| `cacheLayout` | `nd` | KV cache 布局 `nd`/`nz`（gen_data.py 为位置参数，可执行文件为 `--cache_layout` 选项） |
| `device` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本用例为 Ascend 950 算子，编译时需添加 `-DCATLASS_ARCH=3510`。

    ```bash
    bash scripts/build.sh 70_ascend950_flash_attention_chunk_prefill -DCATLASS_ARCH=3510
    ```

2. 进入样例目录，执行 `gen_data.py` 生成测试数据。执行后会在当前路径生成 `data` 目录，包含算子的输入数据和用于精度验证的 golden 数据。

    ```bash
    cd examples/70_ascend950_flash_attention_chunk_prefill
    python gen_data.py 1 567 1000 8 1 128 128 0 half 8 0 128 nd
    ```

3. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序，输入 shape 需与第 2 步生成数据的 shape 一致。

    ```bash
    cd ../../../output/bin
    ./70_ascend950_flash_attention_chunk_prefill 1 567 1000 8 1 128 128 0 8 128 --dtype half --cache_layout nd --device 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
