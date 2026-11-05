# HstuInfer Example Readme

## 功能说明

本样例实现了 HSTU（Hierarchical Sequential Transduction Units，用于生成式推荐的分层序列转导架构）推理场景下的注意力计算算子（不携带rab编码），运行于 Ascend 950（CATLASS_ARCH=3510），基于 CATLASS 模板库的 Cube-Vector（CV）融合框架开发。

与标准 Attention 不同，HSTU 不做 softmax 归一化，而是对 QK^T 结果做带缩放的 SiLU 激活，计算公式如下：

```text
S = Q · K^T                         // Cube 侧 MMAD，fp32 累加
P = siluScale · S · sigmoid(S)      // Vector 侧逐元素 SiLU（可选 causal mask）
O = P · V                           // Cube 侧 MMAD
```

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

以下是本样例涉及的张量类型：

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `Q` | Input | `(totalQTokens, numHeads, headDim)`（TND）或 `(numHeads, totalQTokens, headDim)`（NTD） | half | Query 张量，`totalQTokens` 为各 batch Q 序列长度之和 |
| `K` | Input | 非分页：`(totalKvTokens, kvHeads, headDim)`（TND/NTD 关系同 `Q`）；分页：`(numBlocks, pagedBlockSize, kvHeads, headDim)`（NHD） | half | Key 张量 / Paged KV Cache |
| `V` | Input | 同 `K` | half | Value 张量 / Paged KV Cache |
| `q_seqlen` | Input | `(batch + 1,)` | int64 | Q 序列长度前缀和（cu_seqlen），支持变长 |
| `kv_seqlen` | Input | `(batch + 1,)` | int64 | KV 序列长度前缀和（cu_seqlen），支持变长 |
| `block_table` | Input | `(batch, maxNumBlocks)` | uint32 | 逻辑 block 到物理 block 的映射（仅分页模式） |
| `O` | Output | 同 `Q` | half | HSTU 推理结果，布局与 `Q` 相同 |

以下是本样例涉及的标量参数：

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `batch` | Attr | 标量 | int | batch 数量 |
| `qSeqlen` | Attr | 标量 | int | 最大 Q 序列长度 |
| `kvSeqlen` | Attr | 标量 | int | 最大 KV 序列长度 |
| `numHeads` | Attr | 标量 | int | Q 头数 |
| `kvHeads` | Attr | 标量 | int | KV 头数 |
| `embeddingSize` | Attr | 标量 | int | 头维度 headDim |
| `isVariedLen` | Attr | 标量 | int / bool | 变长序列标志：`0`：定长；`1`：随机变长 |
| `siluScale` | Attr | 标量 | float | SiLU 缩放系数 |
| `layout` | Attr | 标量 | string | Q/O 布局：`NTD` 或 `TND`；分页时 KV 自动切换为 `NHD` |
| `dtype` | Attr | 标量 | string | 固定为 `half`（当前样例仅以 `half` 实例化） |
| `maskType` | Attr | 标量 | int | 掩码模式：`0`：无 mask；`1`：causal 乘法掩码 |
| `pagedBlockSize` | Attr | 标量 | int | `0`：非分页；`>0`：Paged KV Cache 的 block 大小 |

本样例使用一组静态 Tiling 值（见 [hstu_infer_launcher.hpp](launcher/hstu_infer_launcher.hpp)），当前配置为：

```cpp
static constexpr uint32_t headDimCfg = 256;

const uint32_t qkL1TileM = 96;
const uint32_t qkL1TileN = 256;
const uint32_t qkL1TileK = headDimCfg;   // 256
const uint32_t qkL0TileM = 96;
const uint32_t qkL0TileN = 256;
const uint32_t qkL0TileK = 64;

const uint32_t pvL1TileM = 96;
const uint32_t pvL1TileN = headDimCfg;   // 256
const uint32_t pvL1TileK = 256;          // 必须等于 qkL1TileN
const uint32_t pvL0TileM = 96;
const uint32_t pvL0TileN = 256;
const uint32_t pvL0TileK = 64;
```

需满足约束：

- `qkL1TileN` 必须与 `pvL1TileK` 相等（P 由 Silu 阶段直接产出，作为 PV 的 K 维）；
- `qkL1TileM/N` 必须与 Silu Epilogue 的 UB Tile M/N 一致；
- 各级 buffer（L1/L0A/L0B/L0C/UB）大小不得超出硬件容量。

调整 `headDimCfg` 等参数时需满足上述约束，且 headDim 与输入的 embeddingSize 保持一致可获得最优性能。

## 使用范围说明

以下是当前样例参数所使用的参数约束及推荐场景：

| 参数 | 说明 | 约束 | 推荐范围 |
| --- | --- | --- | --- |
| `head_dim`（`embeddingSize`） | 头维度 | `1 ≤ head_dim ≤ 256` | `1` / `64` / `128` / `256` |
| `num_heads` | Q 头数 | `≥ 1` | `1` ~ `16` |
| `num_kv_heads`（`kvHeads`） | KV 头数 | 需满足 `kvHeads == numHeads`（MHA，不支持 GQA / MQA） | 等于 `num_heads` |
| `batch` | batch 数量 | `≥ 1` | `1` ~ `16` |
| `qSeqlen` | 每 batch 最大 Q 序列长度 | `≥ 1`；变长时各 batch 独立取值 | `1` ~ `16384` |
| `kvSeqlen` | 每 batch 最大 KV 序列长度 | 需满足 `kvSeqlen ≥ qSeqlen` | `1` ~ `16384` |
| `block_size`（`pagedBlockSize`） | Paged KV Cache 的 block 大小 | 非分页为 `0`；分页固定 `128` | `128` |
| `mask_type`（`maskType`） | 掩码模式 | `0`（无 mask）或 `1`（causal 乘法掩码） | `0` / `1` |
| `silu_scale`（`siluScale`） | SiLU 缩放系数 | 计算无溢出的实数（`P = silu_scale · S · sigmoid(S)`） | `0.1` |
| `is_varied_len`（`isVariedLen`） | 变长序列标志 | `0`：定长；`1`：随机变长 | `0` / `1` |

推荐范围：

- `Q`/`K`/`V` 张量的数值范围推荐在 `[-1, 1]`以内，避免过大的动态范围致使绝对误差超阈值上限

## 使用示例

### 命令行参数

本样例测试分为数据生成和算子执行两部分

1. 数据生成脚本（[`gen_data.py`](./gen_data.py)）参数

    ```bash
    python gen_data.py batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale layout str_dtype  mask_type paged_block_size data_path
    ```

    上述命令行参数具体说明如下：

    | 序号 | 参数 | 说明 |
    | --- | --- | --- |
    | 1 | batch | batch 数 |
    | 2 | q_seqlen | 每 batch 最大 Q 序列长度 |
    | 3 | kv_seqlen | 每 batch 最大 KV 序列长度 |
    | 4 | num_head | Q 头数 |
    | 5 | kv_heads | KV 头数 |
    | 6 | embedding_size | 头维度（headDim） |
    | 7 | is_varied_len | 0：定长；1：随机变长（kv_seq >= q_seq） |
    | 8 | silu_scale | SiLU 缩放系数 |
    | 9 | layout | `NTD` 或 `TND` |
    | 10 | str_dtype | `half` 或 `bf16` |
    | 11 | mask_type | 0：无 mask；1：causal mask |
    | 12 | paged_block_size | 0：非分页；>0：Paged KV Cache 的 block 大小 |
    | 13 | data_path | 数据输出目录 |

    执行该脚本后，在数据输出目录下会生成如下文件：

    | 文件 | 数据类型 | Shape | 说明 |
    | --- | --- | --- | --- |
    | `q.bin` | half | `[totalQTokens, numHeads, headDim]`（TND）或 `[numHeads, totalQTokens, headDim]`（NTD） | Query |
    | k.bin / v.bin | half | 非分页：`[totalKvTokens, kvHeads, headDim]`（TND/NTD 转置关系同上）；分页：`[numBlocks, pagedBlockSize, kvHeads, headDim]` | Key / Value |
    | q_seqlen.bin | int64 | `[batch + 1]` | Q 序列长度前缀和（cu_seqlen） |
    | kv_seqlen.bin | int64 | `[batch + 1]` | KV 序列长度前缀和（cu_seqlen） |
    | q_ntokens.bin / kv_ntokens.bin | int32 | `[1]` | Q / KV 总 token 数 |
    | block_table.bin | uint32 | `[batch, maxNumBlocks]` | 分页 KV 的逻辑 block 到物理 block 映射（仅分页模式） |
    | golden.bin | fp32 | 同 q.bin | 参考输出，用于精度比对 |

2. 可执行文件参数

```bash
83_ascend950_hstu_infer batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale
       [--dtype DTYPE] [--layout LAYOUT] [--datapath DATA_PATH] [--device DEVICE_ID]
       [--mask MASK_TYPE] [--paged_block_size PAGED_BLOCK_SIZE]
```

上述命令行参数具体说明如下：

| 参数 | 说明 |
| --- | --- |
| batch / qSeqlen / kvSeqlen | 需与 gen_data.py 一致（变长时为各 batch 的最大长度） |
| numHeads / kvHeads / embeddingSize | 头数与头维度，需与 gen_data.py 一致 |
| isVariedLen | 是否变长，需与 gen_data.py 一致 |
| siluScale | SiLU 缩放系数 |
| --dtype | `half` |
| --layout | Q/O 布局：`NTD` / `TND`（分页时 KV 自动切换为 NHD） |
| --datapath | 数据目录，默认 `./examples/83_ascend950_hstu_infer/data` |
| --device | 设备 ID，默认 0 |
| --mask | 0：无 mask；1：causal mask，需与 gen_data.py 一致 |
| --paged_block_size | 0：非分页；>0：启用 Paged KV Cache，需与 gen_data.py 一致 |

## 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh -DCATLASS_ARCH=3510 83_ascend950_hstu_infer
    ```

2. 执行 `gen_data.py` 生成测试样例，执行后会在指定路径下生成 `data/` 目录，包含算子输入数据和用于精度验证的 golden 数据（见后续脚本 `gen_data()` 部分）。

3. 切换到可执行文件的编译目录 `output/bin`，执行编译后的算子样例（见后续脚本 `run_kernel()` 部分）。

4. 执行结果如下，说明精度比对成功：

```text
Compare success.
```

以下是一个完整的 shell 脚本示例：

```shell
batch=2
qSeqlen=256
kvSeqlen=256
numHeads=8
kvHeads=8
headSize=256
isVariedLen=0
siluScale=0.003
layout="NTD"
dtype="half"
maskType=0
pagedBlockSize=0          # >0 时启用 Paged KV Cache，如 128
device=0
dataPath="./examples/83_ascend950_hstu_infer/data"

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh -DCATLASS_ARCH=3510 83_ascend950_hstu_infer
}

function gen_data() {
    python3 examples/83_ascend950_hstu_infer/gen_data.py \
        $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $siluScale \
        "$layout" "$dtype" $maskType $pagedBlockSize "$dataPath"
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads \
        ' D=' $headSize ' isVariedLen=' $isVariedLen ' siluScale=' $siluScale \
        ' layout=' $layout ' mask=' $maskType ' pagedBlockSize=' $pagedBlockSize
    output/bin/83_ascend950_hstu_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize \
        $isVariedLen $siluScale --device $device --dtype $dtype --layout $layout \
        --mask $maskType --paged_block_size $pagedBlockSize
}

build
gen_data
run_kernel
```

## 实现原理

### 算子特性

| 特性 | 说明 |
| --- | --- |
| 硬件 | Ascend 950（Atlas A5，`CATLASS_ARCH=3510`） |
| 核函数形态 | mix 核（`__mix__(1, 2)`）：1 个 AIC（Cube）+ 2 个 AIV 子块（Vector） |
| 数据类型 | half（fp16），S/P 中间结果分别为 fp32 / half |
| Q/O 布局 | `NTD`（numHeads, seqLen, headDim）或 `TND`（seqLen, numHeads, headDim） |
| KV 布局 | 非分页：`NTD` / `TND`；分页：`NHD`（numBlocks, blockSize, kvHeads, headDim） |
| Paged KV Cache | 通过 `--paged_block_size` 启用，基于 block table 间接寻址 |
| 变长序列 | 通过 `q_seqlen` / `kv_seqlen` 前缀和（cu_seqlen）数组描述每个 batch 的实际长度 |
| Mask | `--mask 1` 启用 causal 掩码（乘法掩码，`mask[i,j] = 1 if j <= i`） |
| CV 流水 | 2 级 CV stage（`CV_STAGES=2`）乒乓流水，PV 提前一拍发射（`PRE_LAUNCH_NUM=1`） |

### 总体流程

```text
                 GM (q / k / v)
                      │
        ┌─────────────┴──────────────┐
        │       Cube (AIC)           │        Vector (AIV0 / AIV1)
        │  ┌──────────────────────┐  │        ┌──────────────────────────┐
        │  │ QK MMAD: S = Q·K^T   │──┼──────► │ Silu: P = scale·S·σ(S)   │
        │  │ S(fp32) 写入 UB       │  │  flag  │ P(half) 写入 L1 (A1)     │
        │  └──────────────────────┘  │        │ （可选 causal mask）      │
        │  ┌──────────────────────┐  │        └───────────┬──────────────┘
        │  │ PV MMAD: O = P·V     │◄─┼──────◄─────────────┘
        │  │ O 写回 GM             │  │  flag
        │  └──────────────────────┘  │
        └────────────────────────────┘
```

- **QK 阶段（Cube）**：Q 每任务加载一次到 L1 并跨 KV 块复用，K 从 GM 加载，`S = Q·K^T` 以 fp32 累加并写入 UB。
- **Silu 阶段（Vector）**：从 UB 读取 S，做 `siluScale · S · sigmoid(S)` 逐元素激活（maskType=1 时同时应用 causal 乘法掩码），结果 P（half）写入 L1 的 A1 位置。
- **PV 阶段（Cube）**：以 P 为矩阵 A、V 为矩阵 B，累加计算 `O = P·V` 并写回 GM。PV 的 L0C 结果常驻（`ENABLE_PV_RESIDENT_L0C = true`），跨 KV block 累加。

### 跨核同步

采用 `CROSS_CORE_SYNC_MODE_4` 硬件 flag 实现 Cube 与 Vector 间的生产者-消费者同步：

- `QK_TO_SILU_FLAG_ID`（AIC → AIV）：Cube 通知 Vector S 数据就绪；
- `SILU_TO_PV_FLAG_ID`（AIV → AIC）：Vector 通知 Cube P 数据就绪；
- 两个 AIV 子块（AIV0/AIV1）的 flag 以 `FLAG_ID_MAX = 16` 偏移区分，避免硬件 flag 竞争。

### 任务划分

- 以 `(qSeqBlock, qHead)` 为最小任务粒度，`qSeqBlock` 大小为 `QK L1TileM`；
- 所有 core 以 `globalTaskIdx = coreIdx` 起步、步长 `coreNum` 静态领取任务；
- KV 方向按 `QK L1TileN` 分块循环，通过 `cvStageId` 在 2 级 CV stage 间乒乓切换；
- maskType=1 时，`validKvSeqlen = min(kvSeqlen, Q_BLOCK_SIZE * qSeqBlockIdx + actualQSeqLen)`，跳过当前 Q 块全被掩蔽的 KV 块。

### Paged KV Cache

启用分页（`--paged_block_size > 0`）时：

- K/V 按 `NHD` 布局组织：`(numBlocks, pagedBlockSize, kvHeads, headDim)`；
- 通过 `block_table`（shape 为 `[batch, maxNumBlocks]` 的 uint32 数组）完成逻辑 KV 序列到物理 block 的映射；
- Kernel 内由 `PagedTensor` 封装 block table 间接寻址，逐 block 完成 K/V 的搬运。
