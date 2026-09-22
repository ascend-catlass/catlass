# HSTU Infer Example ReadMe（Ascend950）

## 算子介绍

本样例实现了 HSTU（Hierarchical Sequential Transducer Units）推理场景下的注意力计算算子，运行于 Ascend950（CATLASS_ARCH=3510），基于 CATLASS 模板库的 Cube-Vector（CV）融合框架开发。

与标准 Attention 不同，HSTU 不做 softmax 归一化，而是对 QK^T 结果做带缩放的 SiLU 激活，计算公式如下：

```
S = Q · K^T                         // Cube 侧 MMAD，fp32 累加
P = siluScale · S · sigmoid(S)      // Vector 侧逐元素 SiLU（可选 causal mask）
O = P · V                           // Cube 侧 MMAD
```

### 算子特性

| 特性 | 说明 |
| --- | --- |
| 硬件 | Ascend950（Atlas A5，`CATLASS_ARCH=3510`） |
| 核函数形态 | mix 核（`__mix__(1, 2)`）：1 个 AIC（Cube）+ 2 个 AIV 子块（Vector） |
| 数据类型 | half（fp16），S/P 中间结果分别为 fp32 / half |
| Q/O 布局 | `NTD`（numHeads, seqLen, headDim）或 `TND`（seqLen, numHeads, headDim） |
| KV 布局 | 非分页：`NTD` / `TND`；分页：`NHD`（numBlocks, blockSize, kvHeads, headDim） |
| Paged KV Cache | 通过 `--paged_block_size` 启用，基于 block table 间接寻址 |
| 变长序列 | 通过 `q_seqlen` / `kv_seqlen` 前缀和（cu_seqlen）数组描述每个 batch 的实际长度 |
| Mask | `--mask 1` 启用 causal 掩码（乘法掩码，`mask[i,j] = 1 if j <= i`） |
| CV 流水 | 2 级 CV stage（`CV_STAGES=2`）乒乓流水，PV 提前一拍发射（`PRE_LAUNCH_NUM=1`） |

## 代码组织

```
├── 83_ascend950_hstu_infer
│   ├── CMakeLists.txt              # CMake 编译文件（mix 核，L2_CACHE_HINT）
│   ├── gen_data.py                 # 测试数据与 golden 数据生成脚本
│   ├── ascend950_hstu_infer.cpp    # 主程序：参数解析、数据搬运、精度比对
│   ├── kernel
│   │   └── hstu_infer.hpp          # Kernel 主体（HstuInfer：任务调度 + CV 流水 + 跨核同步）
│   ├── launcher
│   │   └── hstu_infer_launcher.hpp # Host 侧封装（Tile/DispatchPolicy 配置、kernel 发射）
│   └── README.md
```

## 实现原理

### 总体流程

```
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

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#样例编译)。

- 先执行 `gen_data.py` 生成测试样例，执行后会在指定路径下生成 `data` 目录，包含算子输入数据和用于精度验证的 golden 数据。

- 然后执行算子，注意执行算子的输入 shape 需与第一步生成数据的 shape 一致。

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

执行结果如下，说明精度比对成功：

```
Compare success.
```

### gen_data.py 参数

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

### 可执行文件参数

```
Usage: 83_ascend950_hstu_infer batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale
       [--dtype DTYPE] [--layout LAYOUT] [--datapath DATA_PATH] [--device DEVICE_ID]
       [--mask MASK_TYPE] [--paged_block_size PAGED_BLOCK_SIZE]
```

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

## 输入输出数据

`gen_data.py` 在 data 目录下生成如下文件：

| 文件 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- |
| q.bin | half | `[totalQTokens, numHeads, headDim]`（TND）或 `[numHeads, totalQTokens, headDim]`（NTD） | Query |
| k.bin / v.bin | half | 非分页：`[totalKvTokens, kvHeads, headDim]`（TND/NTD 转置关系同上）；分页：`[numBlocks, pagedBlockSize, kvHeads, headDim]` | Key / Value |
| q_seqlen.bin | int64 | `[batch + 1]` | Q 序列长度前缀和（cu_seqlen） |
| kv_seqlen.bin | int64 | `[batch + 1]` | KV 序列长度前缀和（cu_seqlen） |
| q_ntokens.bin / kv_ntokens.bin | int32 | `[1]` | Q / KV 总 token 数 |
| block_table.bin | uint32 | `[batch, maxNumBlocks]` | 分页 KV 的逻辑 block 到物理 block 映射（仅分页模式） |
| golden.bin | fp32 | 同 q.bin | 参考输出，用于精度比对 |

输出 O 与 Q 同 shape、同布局。

## Tile 参数说明

样例运行 Tile 参数固定为一组静态值（见 [hstu_infer_launcher.hpp](launcher/hstu_infer_launcher.hpp)），非性能最优，若有需要调优请联系样例开发咨询内置 Tile 调优参数。当前配置为：

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

约束条件（kernel 内 static_assert 检查）：

- `qkL1TileN` 必须与 `pvL1TileK` 相等（P 由 Silu 阶段直接产出，作为 PV 的 K 维）；
- `qkL1TileM/N` 必须与 Silu Epilogue 的 UB Tile M/N 一致；
- 各级 buffer（L1/L0A/L0B/L0C/UB）大小不得超出硬件容量。

调整 `headDimCfg` 等参数时需满足上述约束，且 headDim 与输入的 embeddingSize 保持一致可获得最优性能。
