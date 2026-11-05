# XAttention Example Readme

## 功能说明

- 算子功能：计算单步 decode attention。同一 batch 内的多个 beam 共享 shared KV，每个 beam 另有独立的 unshared KV。分别计算两部分 attention，再根据 softmax 的 row max / row sum 合并输出，等价于对两段有效 KV 拼接后做一次 attention。

  对 batch b、beam r、Query head h，令 g=numHeads/kvHeads、j=floor(h/g)，取该 KV head 的有效 shared KV 和前 decodeStep 个 unshared KV：

  $$
  K=[K^{shared}_{b,:,j,:};K^{unshared}_{b,r,j,:decodeStep,:}],\quad
  V=[V^{shared}_{b,:,j,:};V^{unshared}_{b,r,j,:decodeStep,:}]
  $$

  $$
  O_{b,r,h,:}=\operatorname{softmax}(s Q_{b,r,h,:}K^T)V,\quad s=1/\sqrt{128}.
  $$

  shared KV 只取 shared_kv_lens[b] 个有效 token，上式省略分页映射。支持 MHA、MQA、GQA，不额外施加 attention mask。

- 样例支持产品型号：Atlas A2系列产品、Atlas A3系列产品（`CATLASS_ARCH=2201`），至少两个 Cube 核。

| cacheMode | shared KV | unshared KV | 索引方式 |
|---|---|---|---|
| 0 | Paged Attention | 连续内存 | shared 按 128-token block 索引 |
| 1 | 连续内存 | Paged Attention | unshared 按 request 索引 |

两种模式互斥，不支持两部分同时分页或同时连续。

## 样例参数说明

记 B=batch、R=beamSize、S=sharedKvSeqLen、H=numHeads、N=kvHeads、D=embeddingSize=128、C=maxDecodeStep、T=decodeStep、L=ceil(S/128)、P=B*L。下表为数据生成器的逻辑 shape，均为连续行主序，最右维连续；Q/K/V/O 的 dtype 必须一致。

| 属性 | 参数（数据文件） | shape | dtype | layout | 说明 |
|---|---|---|---|---|---|
| Input | query | (B,R,H,D) | float16 / bfloat16 | BRHD | 每个 beam 一个 Query token |
| Input | shared_key / shared_value，mode 0 | (P,128,N,D) | 同 query | block-token-head-dim | K/V shape 相同 |
| Input | shared_key / shared_value，mode 1 | (B,S,N,D) | 同 query | batch-token-head-dim | 各 batch 等长、紧密存放 |
| Input | unshared_key / unshared_value | (B,R,N,C,D) | 同 query | batch/request-beam-head-token-dim | mode 0 按 batch，mode 1 按 request；样例 request 数为 B |
| Input | shared_block_table | (B,L) | int32 | 行主序 | mode 0 使用，索引范围 [0,P) |
| Input | unshared_block_table | (B,) | int32 | 一维 | mode 1 使用，索引范围 [0,B) |
| Input | shared_kv_lens | (B,) | int32 | 一维 | 样例所有元素均为 S |
| Input | decode_step | (1,) | int32 | 一维 | 所有 batch/beam 共用 T |
| Output | output | (B,R,H,D) | 同 query | BRHD | attention 结果，样例在内存中校验 |

输入文件名为表中名称加 `.bin`；`golden.bin` 是 float32 精度校验参考结果。样例生成并读取两个 block table 文件，kernel 只使用当前模式对应的表。

PyTorch 接口的 query/output shape 为 `(B*R,H,D)`；mode 0 的 unshared KV 为 `(B*R,N,C,D)`；mode 1 的 shared KV 为 `(B*S,N,D)`，unshared KV 为 `(request_count,R,N,C,D)`。这些 reshape 不改变元素顺序。接口为 `torch_catlass.x_attention`，参数名采用 snake_case。

## 使用范围说明

推荐范围：

| 项目 | 约束 |
|---|---|
| batch、beamSize | 正整数 |
| numHeads、kvHeads | 正整数，H % N == 0，1 <= H/N <= 128 |
| embeddingSize | 固定 128 |
| sharedKvSeqLen | 正整数，不要求为 128 的倍数；分页最后一块允许不满 |
| decodeStep、maxDecodeStep | 1 <= T <= C <= 256；T 是有效长度，C 是缓存容量 |
| cacheMode | 只能为 0 或 1 |
| dtype | float16 或 bfloat16；命令行名称为 half / bf16 |
| mask | 不支持 attention mask；仅通过长度限定有效 KV |
| 长度布局 | 样例和 ATK 使用等长 shared KV，shared_kv_lens[:] = S，batch 之间不额外 padding |
| 索引范围 | B*R*H*129 <= UINT32_MAX；S <= INT32_MAX，B*ceil(S/128) <= INT32_MAX；还需满足设备可用内存 |
| 输入文件 | shape、dtype 和长度元数据须与命令行一致；变更参数后须重新生成数据 |

129=D+1 来自 kernel 中 unshared 输出与 row max 的 uint32 偏移，是组合规模约束。128 的 GQA 上限不代表 H、N 或 R 各自最大只能为 128。ATK YAML 的取值是测试采样集合，不是完整支持范围。

128-token block 仅适用于 mode 0 的 shared KV。mode 1 的 unshared 分页按 request 取整段缓存，容量由 C 决定，不要求 C=128。

推荐使用场景：

- Query、shared KV 张量的数值范围推荐在 `[-5, 5]`以内，避免过大的动态范围致使绝对误差超阈值上限
- 多 beam 共享同一前缀、各自保留短 decode 上下文的单步推理；共享前缀可避免为每个 beam 重复存储 shared KV。
- 已使用 shared 分页缓存时选择 mode 0；shared KV 连续存放、unshared KV 按 request 管理时选择 mode 1。
- MHA（H=N）、MQA（N=1）或 GQA 均须满足上述范围。建议先用下方小规模示例验证，再按设备内存与业务实际 shape 测试性能。

推荐场景的参数范围与样例支持范围相同；这里不承诺特定 shape 的性能优势。不适用于多 Query token 的 prefill、任意 attention mask、head dim 非 128 或 decode 容量超过 256 的调用。

## 使用示例

### 命令行参数说明

```text
78_x_attention batch beamSize sharedKvSeqLen numHeads kvHeads embeddingSize maxDecodeStep decodeStep cacheMode
               [--dtype half|bf16] [--datapath DATA_PATH] [--device DEVICE_ID]
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| batch / beamSize | batch 数 / 每个 batch 的 beam 数 | 必填 |
| sharedKvSeqLen | 每个 batch 的 shared KV 有效长度 | 必填 |
| numHeads / kvHeads | Query head 数 / KV head 数 | 必填 |
| embeddingSize | 每个 head 的维度，必须为 128 | 必填 |
| maxDecodeStep / decodeStep | unshared 缓存容量 / 有效长度 | 必填 |
| cacheMode | 缓存模式 0 或 1 | 必填 |
| --dtype | Q/K/V dtype | half |
| --datapath | 输入及 golden 目录，相对于执行目录 | ../../examples/78_x_attention/data |
| --device | 可见 NPU 设备编号，必须有效 | 0 |

数据生成命令使用相同顺序的九个数值参数，dtype 是最后一个**必填位置参数**，另支持 `--output DIR`（默认脚本所在目录下的 data）。

### 执行示例

准备匹配 Atlas A2系列产品、Atlas A3系列产品的 CANN 环境；生成数据需要 Python 3、NumPy，bf16 另需 ml_dtypes。从 CATLASS 仓库根目录执行：

1. 加载环境并编译样例。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh  # 按实际 CANN 安装路径调整
    bash scripts/build.sh 78_x_attention -DCATLASS_ARCH=2201
    ```

2. 生成输入数据和参考结果。

    ```bash
    python3 examples/78_x_attention/gen_data.py 1 4 512 8 2 128 32 16 0 half
    ```

3. 切换到 output/bin 并执行相同参数。

    ```bash
    cd output/bin
    ./78_x_attention 1 4 512 8 2 128 32 16 0 --dtype half --device 0
    ```

4. 输出如下表示精度验证通过：

```text
Compare success.
```

验证 mode 1 / bf16 时，从仓库根目录重新生成并执行：

```bash
python3 examples/78_x_attention/gen_data.py 1 4 512 8 2 128 32 16 1 bf16
cd output/bin
./78_x_attention 1 4 512 8 2 128 32 16 1 --dtype bf16 --device 0
```
