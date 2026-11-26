# FlashAttentionInfer Example Readme

## 功能说明

- 算子功能：完成 Ascend 950 上推理场景的 Flash Attention 计算，KV cache 支持 Paged/非 Paged 布局，支持 GQA/MQA 与 causal mask
- 计算公式：

  $$
    O = \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{d}} + M\right) \times V
  $$

  其中 $Q$ 是形如 `(qTokenNum, numHeads, headDim)` 的查询矩阵，$K$、$V$ 为 KV cache，Paged 布局下形如 `(numBlocks, blockSize, kvHeads, headDim)` 并经 `blockTable` 寻址，$d$ 为 `headDim`，$M$ 为掩码（`maskType=0` 时无掩码，`maskType=1` 时为 causal 掩码）；GQA/MQA 场景下 `numHeads` 为 `kvHeads` 的正整数倍。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `query` | Input | `(qTokenNum, numHeads, headDim)` | `fp16/bf16` | layout 固定 `RowMajor` |
| `keyCache` | Input | `(numBlocks, blockSize, kvHeads, headDim)` | `fp16/bf16` | KV cache 的 K |
| `valueCache` | Input | `(numBlocks, blockSize, kvHeads, headDim)` | `fp16/bf16` | KV cache 的 V |
| `blockTable` | Input | `(batch, ceil(kvSeqlen/blockSize))` | `int32` | 元素为 KV cache 物理块号 |
| `qSeqLengths` | Input | `(batch,)` | `int64` | 逐 batch 的 Query 序列长度 |
| `kvSeqLengths` | Input | `(batch,)` | `int64` | 逐 batch 的 KV 序列长度 |
| `mask` | Input | `(qTokenNum, kvSeqlen)` | `uint8` | causal 掩码，仅 `maskType=1` 时需要 |
| `output` | Output | `(qTokenNum, numHeads, headDim)` | `fp16/bf16` | 注意力计算结果，layout 固定 `RowMajor` |
| `batch` | Attr | - | `int` | `batch ≥ 1`，推荐 `1 ≤ batch ≤ 16` |
| `qSeqlen` | Attr | - | `int` | 推荐 `1 ≤ qSeqlen ≤ 16384` |
| `kvSeqlen` | Attr | - | `int` | 推荐 `1 ≤ kvSeqlen ≤ 16384` |
| `numHeads` | Attr | - | `int` | `≥ kvHeads` 且为 `kvHeads` 的正整数倍（GQA/MQA），推荐 `1 ≤ numHeads ≤ 16` |
| `kvHeads` | Attr | - | `int` | `kvHeads ≥ 1` 且整除 `numHeads`，推荐 `1 ≤ kvHeads ≤ 16` |
| `headDim`（`embeddingSize`） | Attr | - | `int` | `headDim ≤ 128`，推荐 `128` |
| `blockSize` | Attr | - | `int` | KV cache 分块大小，固定为 `128` |
| `numBlocks` | Attr | - | `int` | `numBlocks ≥ batch * ceil(kvSeqlen / blockSize)` |
| `isVariedLen` | Attr | - | `int` | 是否使用变长序列，当前仅支持 `0` |
| `maskType`（`sparseMode`） | Attr | - | `int` | `0`：无 mask；`1`：causal mask（左上/右下） |
| `cacheMode` | Attr | - | `int` | `0`：非 Paged Attention；`1`：Paged Attention |
| `dtype` | Attr | - | `string` | 数据类型 `half`/`bf16` |

## 使用范围说明

本样例面向 Ascend 950 推理场景的 Flash Attention 计算：

- `headDim ≤ 128`（当前模板的 BlockMmadPV 仅输出到 UB，更大的头维需扩展输出到 GM），推荐使用 `128`。
- `isVariedLen` 当前仅支持 `0`（各 batch 等长）。
- `numHeads ≥ kvHeads` 且 `numHeads` 为 `kvHeads` 的正整数倍（GQA/MQA）。
- `cacheMode=1` 时 KV cache 为 Paged 布局并经 `blockTable` 寻址，`cacheMode=0` 为非 Paged 布局。
- `Q`/`K`/`V` 张量的数值范围推荐在 `[-1, 1]` 以内，避免过大的动态范围致使绝对误差超阈值上限。

## 使用示例

### 命令行参数

先执行 `gen_data.py` 生成测试数据，再执行算子可执行文件，两者的 shape 参数须一致。

```bash
python gen_data.py [batch] [qSeqlen] [kvSeqlen] [numHeads] [kvHeads] [headSize] [isVariedLen] [maskType] [cacheMode] [dtype]
./49_ascend950_flash_attention_infer [batch] [qSeqlen] [kvSeqlen] [numHeads] [kvHeads] [headSize] [isVariedLen] [maskType] [cacheMode] [--dtype DTYPE] [--datapath DATA_PATH] [--device DEVICE_ID]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | batch 大小 |
| `qSeqlen` | 无 | query 序列长度 |
| `kvSeqlen` | 无 | key/value 序列长度 |
| `numHeads` | 无 | query 头数，须为 `kvHeads` 的正整数倍（GQA/MQA） |
| `kvHeads` | 无 | key/value 头数 |
| `headSize` | 无 | 头维度（`embeddingSize`），最大支持 128 |
| `isVariedLen` | 无 | 是否使用变长序列，当前仅支持 0 |
| `maskType` | 无 | mask 类型，0 表示无 mask，1 表示使用 causal mask（左上/右下） |
| `cacheMode` | 无 | 缓存模式，0 表示非 Paged Attention，1 表示 Paged Attention |
| `dtype` | `half` | 数据类型 `half`/`bf16`（gen_data.py 为位置参数，可执行文件为 `--dtype` 选项） |
| `datapath` | `../../examples/49_ascend950_flash_attention_infer/data` | 输入数据路径（可执行文件选项） |
| `device` | `0` | 指定运行设备ID（可执行文件选项） |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本用例为 Ascend 950 算子，编译时需添加 `-DCATLASS_ARCH=3510`。

    ```bash
    bash scripts/build.sh 49_ascend950_flash_attention_infer -DCATLASS_ARCH=3510
    ```

2. 进入样例目录，执行 `gen_data.py` 生成测试数据。执行后会在当前路径生成 `data` 目录，包含算子的输入数据和用于精度验证的 golden 数据。

    ```bash
    cd examples/49_ascend950_flash_attention_infer
    python gen_data.py 1 177 512 1 1 128 0 1 1 half
    ```

3. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序，输入 shape 需与第 2 步生成数据的 shape 一致。

    ```bash
    cd ../../output/bin
    ./49_ascend950_flash_attention_infer 1 177 512 1 1 128 0 1 1 --dtype half --device 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
