# Mla Example Readme

## 功能说明

- 算子功能：完成 MLA（Multi-head Latent Attention）decode 阶段的注意力计算，KV cache 采用 Paged 布局，`kvHeads` 固定为 `1`（MQA），`numHeads` 取 `16/32/64/128` 分别对应模型侧 TP8/TP4/TP2/TP1 场景，`qSeqlen` 支持 `1~4`，覆盖常规 decode 与 MTP 场景
- 计算公式：

  $$
    \begin{aligned}
    S &= \frac{Q_{nope}K_{nope}^{\mathrm{T}} + Q_{rope}K_{rope}^{\mathrm{T}}}{\sqrt{D + D_r}} \\
    O &= \mathrm{softmax}(S) \times V
    \end{aligned}
  $$

  其中 $Q_{nope}$、$Q_{rope}$ 分别是形如 `(qTokenNum, numHeads, D)`、`(qTokenNum, numHeads, D_r)` 的查询矩阵的非 Rope 与 Rope 部分；$K_{nope}$、$K_{rope}$ 为分页 KV cache，形如 `(numBlocks, blockSize, kvHeads, D)`、`(numBlocks, blockSize, kvHeads, D_r)`；$V$ 与 $K_{nope}$ 共用同一份数据（MLA 吸收形式），$O$ 是形如 `(qTokenNum, numHeads, D)` 的输出矩阵。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `query` | Input | `(qTokenNum, numHeads, 512)` | `fp16/bf16` | Query 非 Rope 部分，`qTokenNum` 为各 batch `qSeqlen` 之和 |
| `queryRope` | Input | `(qTokenNum, numHeads, 64)` | `fp16/bf16` | Query Rope 部分 |
| `keyCache` | Input | `(numBlocks, blockSize, 1, 512)` | `fp16/bf16` | K 非 Rope 部分，同时作为 V |
| `keyRopeCache` | Input | `(numBlocks, blockSize, 1, 64)` | `fp16/bf16` | K Rope 部分 |
| `blockTable` | Input | `(batch, ceil(kvSeqlen/blockSize))` | `int32` | 元素为 KV cache 物理块号 |
| `qSeqLengths` | Input | `(batch,)` | `int32` | 逐 batch 的 Query 序列长度 |
| `kvSeqLengths` | Input | `(batch,)` | `int32` | 逐 batch 的 KV 序列长度 |
| `qNtokens` | Input | `(1,)` | `int32` | Query 总 token 数 `qTokenNum` |
| `output` | Output | `(qTokenNum, numHeads, 512)` | `fp16/bf16` | 注意力计算结果 |
| `batch` | Attr | - | `int` | `batch ≥ 1`，推荐 `1 ≤ batch ≤ 16` |
| `qSeqlen` | Attr | - | `int` | 每 batch 的 Query 序列长度，取值 `1~4`（decode/MTP） |
| `kvSeqlen` | Attr | - | `int` | `128 ≤ kvSeqlen ≤ 16384` 且 `kvSeqlen ≥ qSeqlen` |
| `numHeads` | Attr | - | `int` | 取值 {16, 32, 64, 128}，取 `128` 时走 TP1 特化 kernel |
| `kvHeads` | Attr | - | `int` | 固定为 `1`（MQA，整除 `numHeads`） |
| `headDim`（`D`） | Attr | - | `int` | Query/Key 非 Rope 部分头维度，固定为 `512` |
| `qkRopeDim`（`D_r`） | Attr | - | `int` | Query/Key Rope 部分头维度，固定为 `64` |
| `blockSize` | Attr | - | `int` | Paged KV cache 分块大小，固定为 `128` |
| `numBlocks` | Attr | - | `int` | `numBlocks ≥ batch * ceil(kvSeqlen / blockSize)` |
| `dtype` | Attr | - | `string` | 数据类型 `half`/`bf16` |

## 使用范围说明

本样例面向 MLA decode 场景，使用 Paged KV cache 承载长上下文：

- `numHeads` 取值 {16, 32, 64, 128}，分别对应模型侧 TP8/TP4/TP2/TP1 场景；取 `128` 时走 TP1 特化 kernel。通用 kernel（`numHeads ∈ {16,32,64}`）的 `batch` 上限受 AIC 核数限制，TP1 特化 kernel 要求 `batch ≤ 40`。
- `qSeqlen` 支持 `1~4`，覆盖常规 decode 与 MTP 场景；`kvSeqlen` 支持 `128~16384`。
- kernel 支持 `sparseMode` `0`（无 mask）/`1`（causal mask）两种掩码模式，当前样例的 `gen_data.py` 固定按 `sparseMode=0` 生成数据。
- `Q`/`K`（及其 Rope 部分）张量的数值范围推荐在 `[-5, 5]` 以内，避免过大的动态范围致使绝对误差超阈值上限。

## 使用示例

### 命令行参数

先执行 `gen_data.py` 生成测试数据，再执行算子可执行文件，两者的 shape 参数须一致。

```bash
python gen_data.py [batch] ["qSeqlenList"] ["kvSeqlenList"] [numHeads] [numBlocks] [blockSize] [dtype]
./19_mla [batch] ["qSeqlenList"] ["kvSeqlenList"] [numHeads] [numBlocks] [blockSize] [--dtype DTYPE] [--datapath DATA_PATH] [--device DEVICE_ID] [--output OUTPUT_PATH]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | batch 数量，`qSeqlenList`/`kvSeqlenList` 的元素个数须等于 `batch` |
| `qSeqlenList` | 无 | 逗号分隔的逐 batch Query 序列长度，如 `"1,2,3,4"`，每个取值 1~4 |
| `kvSeqlenList` | 无 | 逗号分隔的逐 batch KV 序列长度，如 `"128,256,512,1024"`，每个取值 128~16384 |
| `numHeads` | 无 | Query 头数，取值 16/32/64/128 |
| `numBlocks` | 无 | KV cache 物理块数，`≥ batch * ceil(kvSeqlen / blockSize)` |
| `blockSize` | 无 | KV cache 分块大小，当前仅支持 128 |
| `dtype` | `half` | 数据类型 `half`/`bf16`（gen_data.py 为位置参数，可执行文件为 `--dtype` 选项） |
| `datapath` | `../../examples/19_mla/data` | 输入数据路径（可执行文件选项） |
| `device` | `0` | 指定运行设备ID（可执行文件选项） |
| `output` | 无 | 输出落盘路径，不指定时不落盘（可执行文件选项） |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 19_mla
    ```

2. 进入样例目录，执行 `gen_data.py` 生成测试数据。执行后会在当前路径生成 `data` 目录，包含算子的输入数据和用于精度验证的 golden 数据。

    ```bash
    cd examples/19_mla
    python gen_data.py 4 "1,2,3,4" "128,256,512,1024" 16 64 128 half
    ```

3. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序，输入 shape 需与第 2 步生成数据的 shape 一致。

    ```bash
    cd ../../../output/bin
    ./19_mla 4 "1,2,3,4" "128,256,512,1024" 16 64 128
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
