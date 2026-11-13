# Mega Recurrent KDA 接口、设计与优化

## 接口契约

语义基线为 cannbot-dsl 提交 `9180a51235853c5e8dd1dcc2846b7e3861c1c521`。

### 接口签名

保留基线 20 参数顺序、返回值和状态副作用；主实现位于 `kernel/`，公开入口为 `mega_recurrent_kda.py`。

```python
def mega_recurrent_kda(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    recurrent_state,
    conv_state_indices,
    ssm_state_indices,
    conv_num_accepted_tokens,
    num_accepted_tokens,
    A_log,
    dt_bias,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
) -> torch.Tensor:
    ...
```

`P=N*D`。权重逻辑布局均为输出通道在前，矩阵乘使用 `x @ weight.T`。

| 参数 | dtype / 存储 | shape / 行为 |
|---|---|---|
| hidden_states | BF16 / ND | `[B,8,7168]`，BSH |
| qkv_projection_weight | BF16 / FRACTAL_NZ | `[3*P,7168]`，Q、K、V 顺序 |
| decay_projection_a_weight | BF16 / FRACTAL_NZ | `[128,7168]` |
| decay_projection_b_weight | BF16 / FRACTAL_NZ | `[P,128]` |
| beta_projection_weight | BF16 / FRACTAL_NZ | `[N,7168]`，保留物理 padding |
| output_gate_projection_weight | BF16 / FRACTAL_NZ | `[P,7168]` |
| output_norm_weight | FP32 / ND | `[128]`，各 head 共享 |
| output_projection_weight | BF16 / FRACTAL_NZ | `[7168,P]` |
| conv1d_weight | BF16 / ND | 标准 `[3,N,4,128]`；兼容 `[4,3*P]`、`[3,4,128]` |
| conv_state | BF16 / ND | `[C,L,3*P]`，原地更新 |
| recurrent_state | FP32 / ND | `[R,N,128,128]`，原地更新 |
| conv_state_indices | INT32 / ND | `[B]` |
| ssm_state_indices | INT32 / ND | `[B*8]`，batch-major |
| conv_num_accepted_tokens | INT32 / ND | `[B]`，选择 Conv 历史 |
| num_accepted_tokens | INT32 / ND | `[B]`，选择 recurrent 初始状态 |
| A_log | FP32 / ND | `[N]` |
| dt_bias | FP32 / ND | `[N,128]` |
| scale | FP32 scalar | 有限，常用 `128**-0.5` |
| lower_bound | FP32 scalar | `[-5,0]` |
| rms_norm_eps | FP32 scalar | 有限且大于 0，默认 `1e-5` |

返回连续 BF16 `[B,8,7168]`；除 `conv_state` 和 `recurrent_state` 外不修改输入。

### 支持范围

固定 `H=7168`、`S=8`、`D=128`、`N=96/TP`，模型 batch 为 1～32，算子输入 `B=model_batch*TP`。

| TP | N | P | 算子 B |
|---|---|---|---|
| 4 | 24 | 3072 | 4,8,...,128 |
| 8 | 12 | 1536 | 8,16,...,256 |
| 16 | 6 | 768 | 16,32,...,512 |

单卡处理 TP-local 权重与状态；算子内部不进行跨卡 collective。一次 mixed AIC/AIV kernel 完成 QKV、decay A/B、beta、gate 投影，causal Conv1D+SiLU，Q/K 归一化，recurrent KDA，RMSNorm、sigmoid gating 和输出投影。

沿用基线的当前 stream 配额语义：有效 AIC 配额截断到 32，结果必须属于 `{8,16,24,28,32}`，AIV 配额至少为两倍。标准输入路径不得在上述计算链之外额外启动计算 kernel。NZ 转换在初始化时完成；兼容 Conv 权重布局的转换不属于融合 kernel，严格单 kernel 验证采用标准 Conv 布局。

### 计算语义

以下按每个 batch、head、token 表述，state 存储方向为 `[V,K]`，两维均为 128。

1. `raw_qkv=FP32(x @ Wqkv.T)`。对 Q/K/V 分别执行 4-tap causal depthwise Conv，再做 `SiLU(a)=a/(1+exp(-a))`，结果转 BF16。当前 token 参加 Conv 的值沿用 FP32 raw projection；写回 Conv 历史的 raw QKV 转 BF16。
2. `a=BF16(x @ Wa.T)`；`g=BF16(a @ Wb.T)`；`raw_beta=BF16(x @ Wbeta.T)`；`z=BF16(x @ Wgate.T)`。
3. Q/K 从 BF16 转 FP32，分别除以 `sqrt(sum(v*v)+1e-6)`。Q 再乘 `scale`。
4. `log_decay=lower_bound*sigmoid(exp(A_log)*(g+dt_bias))`；`decay=exp(log_decay)`；`beta=sigmoid(raw_beta)`，均 FP32。
5. 读取选中的 FP32 state `M`；`Md=M*decay[None,:]`；`delta=beta*(v-Md@k)`；`Mnew=Md+delta[:,None]*k[None,:]`；`o=Mnew@q`。每个 token 写入对应 FP32 snapshot。
6. `norm=o/sqrt(mean(o*o)+rms_norm_eps)*output_norm_weight`；`gated=BF16(norm*sigmoid(z))`。
7. 将各 head 的 gated output 拼成 P 通道，`out=BF16(gated @ Wo.T)`。

计算顺序、BF16 舍入点和状态写回范围采用上述固定 DSL 版本的 snapshot 语义；阶段间的数据流和同步约束见 [设计文档](#整体设计)。

### 边界行为

- 所有 Tensor 连续并位于同一 NPU。六个投影权重传入 FRACTAL_NZ allocation-root Tensor，不能传 view；Parameter 先 `.detach()`。hidden、Conv 权重及各状态为 ND。
- shape、dtype、连续性、设备和 scalar 非法时抛异常；不接受 S/H/D/TP 的其他值或空 batch。
- Conv 槽位范围 `[0,C)`；槽 0 为不可写空槽，对应 Conv 输出全零。同一调用非零 Conv 索引互不重复。
- Conv 历史起点 `offset=max(conv_num_accepted_tokens[b]-1,0)`；`L>=10` 且 `offset+3<=L`。选中历史的最后两行写到 `[0:2]`，本轮 8 个 raw QKV 写到 `[2:10]`；其他行及未选槽保持不变。
- recurrent 槽位范围 `[0,R)`；槽 0 有效，`R>=B*8`。初始 state 索引为 `ssm_state_indices[b*8+max(num_accepted_tokens[b]-1,0)]`，该 token 偏移必须位于 `[0,8)`。token t 写入 `ssm_state_indices[b*8+t]`。
- 不同 batch 的 recurrent 槽集合不能相交。索引和计数值由调用方保证有效，不为 host 验证把设备张量复制回 CPU。全部 8 个 proposal token 均计算，accepted 计数不截断输出。
- 状态池未选区域保持逐元素完全不变。共享同一 workspace 的调用由调用方串行化。
- FP32 state 专指 recurrent state；conv state 是 BF16。BF16 权重专指六个投影和 Conv 权重；RMSNorm 权重、A_log、dt_bias 是 FP32。

## 整体设计

本设计对应 C25 前向实现。公开入口、Tensor 布局和状态副作用见 [接口说明](#接口契约)，分块路由和输出权重复用见 [优化方案](#优化方案)。

### 执行边界

目标设备为 Ascend 950。标准 Conv 权重布局 `[3,N,4,128]` 下，host 准备 workspace、绑定 Tensor 并发射一次 C1V2 mixed AIC/AIV kernel；每个 AIC 配对两个 AIV。计算链包括全部投影、Conv1D、SiLU、recurrent KDA、RMSNorm、sigmoid gating 和输出投影。

固定 `H=7168`、`S=8`、`D=128`，`N=96/TP`，`TP` 为 4、8 或 16。定义 `B=model_batch*TP`、`M=B*8`、`P=N*128`，其中 model_batch 为 1～32。该入口使用 TP-local 权重与状态，不包含跨卡通信。

六个投影权重使用 BF16 / FRACTAL_NZ，hidden 和 Conv state 使用 BF16 / ND，recurrent state 使用 FP32。host 只进行输入元数据校验、内存分配、编译和发射。兼容 Conv 布局需要额外转换，严格单 kernel 路径使用标准布局。

### 阶段划分

| 阶段 | 执行单元 | 输入与输出 | 实现位置 |
| --- | --- | --- | --- |
| S0 | AIC | hidden 投影生成 raw QKV、decay A、beta、gate | `kernel/mega_recurrent_kda.py` |
| S1 | AIV | 消费 FP32 raw QKV，执行四抽头 Conv 和 SiLU，生成 BF16 QKV 并更新 Conv state | `kernel/mega_recurrent_kda.py` |
| S2 | AIC | BF16 decay A 乘第二级权重，生成 BF16 decay 投影 | `kernel/mega_recurrent_kda.py` |
| S3 | AIV | Q/K 归一化、八步 recurrent KDA、RMSNorm 和 gating，写 FP32 snapshots 与 BF16 gated output | `kernel/mega_recurrent_kda.py` |
| S4 | AIC | gated output 乘输出权重，返回 BF16 `[B,8,7168]` | `kernel/mega_recurrent_kda.py` |

`kernel/mega_recurrent_kda.py` 集中 AIV 辅助函数、Conv/recurrent 阶段、AIC 投影资源与调度，以及融合入口。文件内部按上述顺序组织，保留全部函数边界和同步事件；公开入口仍只发射一次 `mega_kernel`。

S0 的 raw QKV 通过片上双缓冲交给 S1，不写完整 raw QKV GM 中间张量。S2 完成后，AIV 使用全核同步保证 QKV、decay、beta 和 gate 投影均可见。启用流水时，S3 按 128 行块发布 gated output，S4 逐块消费；其余输入走完整阶段同步路径。

### 数值与状态语义

矩阵乘采用 BF16 输入、FP32 累加，保留各阶段规定的 BF16 舍入点。raw QKV 的当前 token 保留 FP32 参与 Conv，写入 Conv 历史前转 BF16。Q/K 归一化、decay、beta、recurrent 更新、RMSNorm 和 sigmoid gating 使用 FP32 中间运算。

recurrent state 的每个 head 存储方向为 `[V,K]`。一次调用顺序计算八个 token，读取 accepted 计数指定的初始 snapshot，并逐 token 写回对应状态槽；accepted 计数不截断输出。Conv 槽 0 是空槽，recurrent 槽 0 有效。索引范围、槽位互斥和未选区域保持条件见接口说明，由调用方保证设备索引值合法。

Conv 和 recurrent 状态池的地址乘法在 INT64 中完成，避免大池访问的元素偏移先在 INT32 中溢出。kernel 不改变输入索引的 INT32 类型。

### Workspace 与生命周期

host 为每次调用独立分配一个 BF16 workspace。定义 `NB=16*ceil(N/16)`，各区按 256 个 BF16 元素，即 512 B，对齐。

| 区域 | 逻辑形状 | 未对齐字节数 |
| --- | --- | ---: |
| Conv 后的 QKV | `[M,3*P]` | `6*M*P` |
| decay A | `[M,128]` | `256*M` |
| decay B 输出 | `[M,P]` | `2*M*P` |
| beta | `[M,NB]` | `2*M*NB` |
| gate 投影 | `[M,P]` | `2*M*P` |
| gated output | `[M,P]` | `2*M*P` |

令 `A(v)=512*ceil(v/512)`，workspace 大小为 `A(6*M*P)+A(256*M)+3*A(2*M*P)+A(2*M*NB)` B。最终输出另分配 `2*M*7168` B；输入与两个状态池由调用方提供。

host 将 artifact、绑定参数、输出、workspace、views 和输入数据保存在同一 prepared tuple 中，直到发射调用返回。执行仍遵循当前 NPU stream 的异步语义，不增加同步或设备计算。片上资源由 `allocate_projection_resources` 与 `allocate_recurrent_resources` 分阶段复用。

### 同步约束

- S0/S1 使用 ready/free 双缓冲协议：AIC 完成 raw QKV 交付后通知 AIV，AIV 消费完成后归还 slot；buffer 复用必须等待相应消费与搬运完成。
- S0 与 S2 之间、S2 与 S3 之间保留全核同步。空任务核同样参与相位栅栏。
- S3 的 state ping-pong buffer 同时服务下一 token 的读取与 snapshot 写回，复用前等待对应搬运完成。
- 流水路径使用两组 ready/ack 通知发布 128 行 gated output。ack 释放通知槽，完整 GM gated buffer 不覆盖复用。
- S4 的 L1、L0A/L0B 和 L0C 分别通过搬运完成、MMAD 完成和 FIX 完成事件保护；退出阶段等待在途事件完成。

C25 的输出权重常驻布局及事件顺序详见优化方案。源文件中的独立阶段 kernel 入口用于阶段隔离，公开前向入口仅发射 `mega_kernel`。

## 优化方案

本文描述 C25 已实现的优化及其适用范围。接口与状态语义见 [接口说明](#接口契约)，整体数据流见 [设计文档](#整体设计)。`M=B*8`，`P=N*128`，`C` 表示当前 stream 的有效 AIC 配额。

### 已实现的路由

以下条件直接对应公开入口的编译参数，未满足条件时使用默认路径。

| 参数 | 启用条件与取值 | 默认值 |
| --- | --- | --- |
| `initial_n` | `C in {8,16,24,28}` 且 `M>=256` 时为 384 | 128 |
| `pipelined` | `C in {8,16,24,28}` 且 `M>=256` | False |
| `output_n` | `C=28` 且 `M>=256` 时为 256 | 128 |
| `dual_m` | `C=28`、`N=6` 且 `M>=2048` | False |
| `output_resident` | `C=28`、`N=6` 且 `M>=256` | False |

三 head 投影通过共享 hidden panel 减少重复输入搬运。双 M 路径在两个行块间共享权重，仍保留各输出累加器及原有 Conv 交付顺序。S3/S4 流水按 128 行发布和消费 gated output，宽输出 tile 使用 256 列。这些策略改变分块与搬运生命周期，不改变数学公式、BF16 舍入点或八次 FP32 snapshot。

32 AIC 配额保留默认路由；本次实测设备通过运行时查询与 `npu-smi` 确认为 28 AIC / 56 AIV，32 AIC 未实测。

### C25：输出投影权重常驻 L1

当 `N=6` 时，输出投影的归约维为 `P=768`。在 28 AIC、256 列 tile 下，`7168/256=28`，每个 AIC 固定负责连续的 256 个输出通道。该列片在全部 128 行块之间复用，因此每核只需从 GM 加载一次 `[256,768]` BF16 输出权重。

布局由 `kernel/mega_recurrent_kda.py` 中的 `prefetch_output_weight` 与 `output_resident_tile` 的指针偏移和 Tensor 形状定义。以下为该实现的 L1 分区，单位为 B，区间左闭右开；它描述代码使用的布局，而非所有设备的通用容量规格。

| 对象 | 字节区间 | 大小 |
| --- | --- | ---: |
| gated A slot 0 | `[0,65536)` | 64 KiB |
| gated A slot 1 | `[65536,131072)` | 64 KiB |
| 常驻输出权重 | `[131072,524288)` | 384 KiB |

两个 A slot 各容纳 `[128,256]` BF16。权重先按紧凑 `zNUnAlign` 布局写入 L1，再通过 `nZ` 视图提供 `[64,256]` 的 L0B 切片。归约维按三个 K256 panel、每个 panel 四个 K64 子块递增遍历，FP32 累加完成后由 FIX 转为 BF16 输出。

权重预取使用既有 `ld0` 事件，先完成 MTE2 set / MTE1 wait，再初始化 A slot 的 free 事件。首个 A copy 必须等待该 free 事件，避免在权重通知尚未消费时复用 `ld0`。权重区在所有输出行块完成前只读；每块 gated 数据仍需等待 AIV ready 与 AIC 全核栅栏，不以预取完成替代生产者同步。

该路径不增加 GM workspace、cross flag 或 kernel 发射次数。`N=12/24`、其他 AIC 配额和 `M<256` 保持原输出投影路径。

### 搬运量与收益边界

以 TP16、model_batch=16、C=28 为例，`B=256`、`M=2048`，共 16 个 128 行块。完整输出权重大小为 `7168*768*2=11,010,048` B。按每行块重新加载一次计算，逻辑权重读取量从 `176,160,768` B 减至 `11,010,048` B，减少 `165,150,720` B。

这是按 tile 调度推导的逻辑搬运量，不等于 HBM 实际流量或耗时收益。L2 Cache 命中、S3/S4 重叠、首块预取和各流水共享带宽均会影响实测结果。

固定 C25 已完成 48 个性能点：`TP{4,8,16} * model_batch{1,8,16,32} * AIC{8,16,24,28}`。每点以普通 msprof 的正式单 kernel Task Duration 分别采样五次取中位数，加速比为 cannbot-dsl 耗时除以 C25 耗时。

- 9 点达到至少 1.0 倍，39 点未达标；逐点验收未通过。
- 48 点加速比的等权算术平均为 0.948971 倍，范围为 0.743840～1.388773 倍。
- TP16、model_batch=16、C=28 的 DSL / C25 中位数为 1157.252 / 1171.052 μs，加速比为 0.988216 倍。

上述数据来自 [PR #1725 的逐 case 性能表](https://gitcode.com/cann/catlass/merge_requests/1725)，对应性能报告提交 `a20f02c2`。实测环境为 Ascend 950PR_957c、CANN toolkit 9.2.0、torch / torch_npu 2.13.0、CATLASS DSL 2.2.1.dev20261112；基线固定为 cannbot-dsl 提交 `9180a51235853c5e8dd1dcc2846b7e3861c1c521`。每个 case 的双方在同一卡上采样，TP4 使用 device7，TP8/TP16 使用 device3；设备未独占，存在其他用户瞬时负载的限制。

### 后续优化方向

以下是待验证方向，不属于当前实现或已获得的收益：

1. 优先分析 TP4/TP8 的投影重复读取与任务分配，分别评估更大 `P` 下的分块和分段权重复用，不能直接套用 K768 的常驻布局。
2. 分离 S0 投影、recurrent 更新和 S3/S4 交接的耗时，判断改动影响的是独立瓶颈还是已被重叠覆盖的阶段。
3. 对输入预取与更深流水重新验证 buffer 生命周期、通知回绕、空任务核和尾块，再测完整适用域。

每项改动均需保留完整输出与两状态池精度检查，并以全部 48 点逐点至少 1.0 倍为性能目标；单点收益或平均值不能替代该目标。
