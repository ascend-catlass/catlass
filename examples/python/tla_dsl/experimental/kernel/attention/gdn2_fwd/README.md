# 实验性 GDN2 前向（CATLASS Python DSL）

本实验性样例在 Ascend 950PR 上实现 Gated DeltaNet 2（GDN2）定长前向，使用两个 CATLASS Python DSL kernel：共享因子预计算和分块状态递推。Host 负责参数校验、分配、编译缓存和启动。

## 支持规格

| 项目 | 支持范围 |
|---|---|
| 布局 | 连续 BNSD，所有 Tensor 位于同一 NPU |
| Q/K | BF16，`[B, HK, T, 128]` |
| V、输出 o | BF16，`[B, HV, T, 128]` |
| g、b | BF16 或 FP32，`[B, HG/HB, T, 128]`，各自选择 HK 或 HV 个头 |
| w_gate | BF16 或 FP32，`[B, HV, T, 128]` |
| 初始和最终状态 | FP32，K-first，`[B, HV, 128, 128]` |
| 头映射 | `HV/HK ∈ {1,2,4}`，`1 ≤ HV ≤ 96`；Q/K 共用 HK 个头 |
| 维度和长度 | `K=V=128`，`B≥1`，`T≥1`，支持非整块尾部 |
| chunk_size | 64 或 128；物理递推块为 64 |

这里的头映射是 Grouped Value Attention（GVA），每个 value head 使用 `hk=hv//(HV/HK)` 对应的 Q/K；不支持任意 GQA 头配置。

`g` 是已经激活的逐 K 通道自然对数衰减，`b` 是逐 K 通道 erase gate，`w_gate` 是逐 V 通道 write gate。输入值域是调用方前置条件：所有值有限，`g≤0`、`b∈[0,2]`、`w_gate∈[0,1]`。Host 不回读 Tensor 检查值域，不执行 gate 激活或 Q/K 归一化。

仅返回 `o` 和可选的最终状态。不返回反向缓存 `Aqk/Akk/w/u/qg/kg/v_new/g_cumsum/h`，不提供 `disable_recompute`、`allow_neg_eigval`、反向、变长 packed 序列、CP 或 V-first 状态。输入 `w_gate` 不是 WY 表示中的缓存 `w`，最终状态不是逐 chunk 的状态缓存 `h`。不支持标量 beta 自动广播。

## 文件组织

样例包含两个 Python 文件和本说明文档，设计与优化方案分别见[设计](#设计)和[优化方案](#优化方案)。

| 文件 | 功能 |
|---|---|
| `__init__.py` | 公开 Host 接口、参数校验、输出与 workspace 分配、32 项 LRU 编译缓存及当前 stream 上的启动 |
| `gdn2_fwd.py` | 公共 Tensor/view、分量和同步辅助函数，以及 prepare/recurrent 两个设备 kernel 的完整实现 |
| `README.md` | 支持规格、调用方式、精度约束、计算与同步设计、优化方案及性能结果 |

Host 先建立两个设备 kernel 的绑定和编译产物，再按 prepare、recurrent 的顺序启动。文件归并不改变设备 kernel 数量与调用顺序。

## 环境与调用

已验证环境为 Ascend 950PR、CANN toolkit 9.2.0、`ascend-catlass-dsl==2.2.0`、Python 3.12.15、torch 2.13.0+cpu、torch_npu 2.13.0。编译选项为 `--npu-arch 3510`。

初始化 CANN 环境后，在 CATLASS 仓库根目录设置包搜索路径：

```bash
export PYTHONPATH="$PWD/examples/python/tla_dsl/experimental/kernel/attention${PYTHONPATH:+:$PYTHONPATH}"
```

Q/K/V 和 gates 按支持规格准备并放置在同一健康 NPU 上：

```python
from gdn2_fwd import gdn2_fwd

o, final_state = gdn2_fwd(
    q, k, v, g, b, w_gate,
    scale=None,
    initial_state=None,
    output_final_state=True,
    chunk_size=64,
)
```

`scale=None` 使用 `128**-0.5`；`initial_state=None` 使用全零初态。`output_final_state=False` 时第二项为 `None`。输出与输入不别名，接口不修改输入，也不构建反向计算图。Host 使用输入设备的当前 stream，调用方按需执行设备同步。

## 数值约束

独立 CPU FP64 逐 token 标杆用于验证输出和最终状态。误差判定为 `abs(actual-golden) <= atol + rtol*abs(golden)`。
BF16 使用 `atol=rtol=0.0156`、最大绝对误差上限 1；FP32 使用 `atol=1.53e-5`、`rtol=0.000977`、最大绝对误差上限 0.01。输出全局及每个 head 匹配率至少 99.9%。关键区域包括序列首尾、32/64/128-token 块边界的前后 token 及不足整块的尾部；这些区域和最终状态的匹配率要求为 100%。

## 语义与 API 依据

数学语义参考 [FLA 固定版本 GDN2 forward](https://github.com/fla-org/flash-linear-attention/blob/2a3e61a965f801caed462f5d963d584767bd0db6/fla/ops/gdn2/chunk_fwd.py)。本实现扩展 BNSD、GVA 1/2/4 和逻辑 chunk_size 128。
设备调用遵循 [CATLASS Python API 固定版本](https://gitcode.com/cann/catlass/blob/60524ac206c3027124f5b3ce8b1e5c6879eca1ba/python/tla_dsl/docs/zh/api/index.md) 与已安装 DSL 2.2.0 的共同可用接口。

## 设计

### 计算语义

状态 `H` 为 K-first FP32 矩阵。对每个 token，逐通道的 `g/b/w_gate` 分别控制衰减、擦除和写入：

```text
D = exp(g)[:, None] * H_previous
erase = (b * k)^T @ D
delta = w_gate * v - erase
H = D + k[:, None] * delta[None, :]
o = scale * q^T @ H
```

独立 CPU 标杆按 token 顺序使用 FP64 累积，返回 BF16 输出和 FP32 最终状态。设备使用物理时间块 `C=64`，逻辑 chunk_size 128 由两个物理块组成。中间表示允许舍入误差，最终精度按 [数值约束](#数值约束) 的混合容差判定。

### 分块与任务映射

令 `M=ceil(T/64)`、`HF=max(HK, HG, HB)`、`F=B*HF*M`。prepare 的每个 work 产生一份因子，由 value heads 和 V 切片共享。factor 编号为 `f=block_idx+i*block_num`，双缓冲 slot 为 `i%2`。

recurrent 从设备查询 Cube Core 数 `Nc`。令 `N=B*HV`、`J=floor(N/Nc)*Nc`、`G=J+4*(N-J)`。前 J 个 head 使用 V128 tile，剩余 head 各拆成四个 V32 tile；两个循环使用独立循环变量。每个任务沿 M 个物理时间块递推。主 shape 在 28 个 Cube Core 上形成 44 个任务，最长 core 处理 160 列。

两 AIV 各处理 32 个 token 行和 64 个状态 K 行。当前接受版本在 S9 由两 AIV 各加载完整 64 行 V/W，再消费各自的 32 行。GM 的地址乘积使用 Int64；Host 检查 signed-64-bit 字节数上限。

### 数据流

prefix 与 suffix 在 K 通道上独立累计。块内矩阵使用分层三角逆；64 阶逆由四个 16 阶叶子和 32 阶合并构造。下表中的 N 表示负 W，不是 head 数。

| Stage | 执行单元 | 计算或发布 |
|---|---|---|
| S0 | Vector | decay、prefix、Qg、Bg、Ki、Kt、end；Qg 高分量，Kt 两分量 |
| S1 | Cube | QK score、严格下三角 L 和 safe 标记 |
| S2 | Vector | score 掩码、稳定回退、四个 16 阶三角逆叶子 |
| S3 | Cube | 两个 16 阶合并中间矩阵 |
| S4 | Cube | 合并两个 32 阶逆并补写 GM |
| S5 | Cube | 32 阶合并中间矩阵 |
| S6 | Cube | 完整 64 阶逆 A，写 L1 和 GM |
| S7 | Cube | `W=A@Bg` |
| S8 | Vector | `N=-W`，编码为 Nh/Nl 两个 BF16 分量 |
| S9 | Vector / AIC MTE2 | WV/end 加载；Cube 侧预取共享因子 |
| S10 | Cube | `U=A@WV`，FP32 留在 L0C |
| S11 | Cube | 按固定顺序累积 `Nh@Hh, Nh@Hl, Nl@Hh, Nl@Hl`，得到 delta |
| S12 | Cube / FIX | delta 同时发布到 FP32 L1 和每个 AIV 的 UB |
| S13 | Cube / Vector | `O=Qg_high@H_high + Aq@delta`；发布 O_ready 后由 Vector 输出 BF16 o |
| S14 | Vector | delta 编码为三个 BF16 分量，写独立 L1 池 |
| S15 | Cube | 五项 BF16 MMAD 得到 P，FP32 累积 |
| S16 | Vector | `H_new=H_old*end+P`；更新状态分量和可选最终状态 |

S13 与 S14 可重叠。S13 完成后通过独立的 `O_ready` 通知，Vector 输出写回可与 Cube 的 S15 重叠。QgH 使用 BF16 high 分量，Aq@delta 使用 `HF32_NEAREST_EVEN`；U、逆矩阵、W 及分量乘法使用 `HF32_DISABLE`。

对 FP32 数 x，两分量表示为：

```text
x0 = BF16_CAST_ROUND(clamp(x, -BF16max, BF16max))
r1 = x - FP32(x0)
x1 = BF16_CAST_ROUND(r1)
x2 = BF16_CAST_ROUND(r1 - FP32(x1))  # only for delta
```

Kt 保存 x0/x1，delta 保存 x0/x1/x2；残差的两次减法不重排。`CAST_ROUND` 的中点舍入为 ties-away。P 的累积顺序固定如下，首项 `init_c=True`，后续项均为 False：

```text
P = (((Kt0.T@D0 + Kt0.T@D1) + Kt1.T@D0) + Kt1.T@D1) + Kt0.T@D2
```

状态更新保持独立 FP32 乘法和加法；最终 BF16 输出使用 round-to-nearest-even。prepare 在 prefix 过小或 Q/K 超出安全范围时使用稳定 score 路径，避免先倒数再处理下溢。

### C53 的流水与 Vector 行配对

prepare 在当前 work 的 score 和三角求解期间提前准备下一 work 的 Bg/Ki/Kt，复用原双缓冲空间；没有改变因子公式。每个 AIV 的两个 16 阶逆叶子在互不相交的 SIMD lane 中按同一动态行循环求解，保持各自递推顺序，随后展开回 64 阶布局。

V32 分支把相邻两行映射到一次 64-lane FP32 Vector 运算，覆盖 WV、delta 分量、状态更新和输出。两行的 gate 分别广播后按 lane 选择，存储用两个互斥掩码写入 native 布局；不跨行归约。V128 分支沿用原运算。状态更新仍为独立 FP32 乘法和加法，P 的五项累积顺序及全部冻结精度阈值不变。

`O_ready` 在 O 的 FIX 写回及本地完成等待之后发布。Vector 输出只使用 UB 的 O/output 区域，Cube 同时计算的 P 使用独立 L0C/UB 区域。后续状态更新仍等待 H，下一轮 O 的覆盖受原 D/H 往返约束。

### Workspace 与片上空间

每次调用新建八份 workspace，分配和字节计数共用 `__init__.py` 中的 layout 定义。每 factor 为 115200 Byte，主 shape 共 309.375 MiB，不包含 API 输入、输出、初末状态和系统 workspace。

| Tensor | 每 factor shape | dtype |
|---|---|---|
| Qg | `[64,128]` | BF16 |
| Kt0 / Kt1 | 各 `[64,128]` | BF16 |
| end | `[1,128]` | FP32 |
| Aq / A | 各 `[64,64]` | FP32 |
| Nh / Nl | 各 `[64,128]` | BF16 |

prepare 分配 UB 224 KiB、L1 432 KiB、L0A 32 KiB、L0B 64 KiB、L0C 64 KiB。两个 L1 factor slot 各 184 KiB，原始 Bg 另外使用两个 32 KiB 槽。Kt 两个 RowMajor UB staging 分别位于 72–80 KiB 和 80–88 KiB；Kt 不写 prepare L1。Qg/Bg/Ki 使用直接 native 分量写回；Aq 保留已有 L1 副本。

recurrent 分配 UB 240 KiB、L1 496 KiB、L0A/L0B 各 64 KiB、L0C 96 KiB。地址以 KiB 表示：

| 空间 | 区间 | 内容 |
|---|---|---|
| UB | 0–48 / 48–96 | raw V/W 双槽，delta 编码复用当前槽的前三个 8 KiB 池 |
| UB | 96–112 | native WV staging |
| UB | 112–144 | FP32 Hold |
| UB | 144–160 / 160–176 | native Hh/Hl |
| UB | 176–208 | P；前 16 KiB 分时保存 delta |
| UB | 208–224 / 224–232 | FP32 O / BF16 o |
| UB | 232–233 | 两个 end 槽 |
| L1 | 0–128 / 128–256 | factor 双槽：Qg/Nh/Nl/Kt0/Kt1/空洞/Aq/A，各 16 KiB |
| L1 | 256–320 | 单 H 槽，两个 BF16 分量 |
| L1 | 320–352 / 352–384 | FP32 WV 双槽 |
| L1 | 384–416 / 416–448 | FP32 delta 双槽 |
| L1 | 448–496 | delta 三个 BF16 分量 |
| L0C | 0–32 / 32–96 | U/delta/O 复用池 / P 池 |

V32 tile 使用实际宽度确定分量偏移，保留最大容量分配。P 的两个 A 平面占 L0A 0–32 KiB，三个 B 平面最多占 L0B 0–48 KiB，最后一次 MMAD 完成后才释放全部输入。

### 同步与生命周期

prepare 使用 P0/P1、Q、leaf0/leaf1、W 跨核通知实现双 slot 流水。下一 work 的 S0 分段插入当前 score、逆矩阵和 W 的消费区间；本地 MTE2/Vector/MTE3 与 Cube/FIX 的完成条件独立闭合。GM 的 A 叶子写在 leaf-ready 前，避免与 Cube 补角写回竞争。

recurrent 每个 group 使用 WV0/WV1、H、D、O_ready、group_done 六个跨核通知：

```text
Cube: prefetch factors0
      wait factors; prefetch next factor; wait WV; U
      wait H; predict; FIX delta to L1 and UB; publish D
      compute O; publish O_ready; wait D; compute P; publish H; release factor
      wait final H; drain factor credits; ALL; group_done

Vector: initialize H; publish H; produce WV0; publish WV0
        prefetch next WV when present
        wait D; encode delta; publish D
        wait O_ready; write o
        wait H; update H; publish H
        wait group_done before reusing pools for another group
```

- delta FIX 写 UB 176 KiB 前已等待上一轮 H，因此上一轮 P 的 Vector 读取已结束；它不覆盖下一块预取正在使用的 WV staging。
- P FIX 等待两 AIV 返回 D，保证 delta 源读和三个 BF16 分量的 MTE3 发布已完成。
- H 单槽只在 P 完成后更新，此时当前 predict/O 的 MTE1 读者已结束；下一 predict 必须等待新 H。
- raw 的 n+2 复用发生在当前 D/P/H 往返后，保证 delta native staging 的 MTE3 读已结束。
- 零有效输出行的 AIV 仍更新负责的状态 K 行并参与全部通知。退出时分别排空本地 pipe 和跨核通知。

### Host 生命周期与验证

Host 在启动第一个 kernel 前建立所有绑定并获取两个编译产物；两次 launch 之间不插入编译、绑定或数值计算。LRU 缓存只保存编译产物，每次调用创建新绑定，避免保留输入地址。调用使用输入设备的当前 stream。

运行参数校验由 `__init__.py` 提供，Host 不依赖 CPU 标杆或测试模块。精度约束见 [数值约束](#数值约束)，优化方案及性能口径见 [优化方案](#优化方案)。

## 优化方案

### 当前方案与约束

当前实现为 C53/v104。共享因子由 prepare kernel 生成，recurrent kernel 沿时间块递推状态。核心数据流、内存布局及同步关系见 [设计](#设计)。

主规格为 `B=1, HK=16, HV=32, T=11264, K=V=128, chunk_size=64`，连续 BNSD，Q/K/V/g/b/w_gate 均为 BF16，使用非零 FP32 初态并返回最终状态。GVA 定义为 `HV/HK`，支持 1/2/4；HV 上限为 96。

所有优化保持 CPU FP64 逐 token 语义、固定误差容限和关键边界/最终状态 100% 匹配要求。状态更新使用独立 FP32 乘法和加法，BF16 输出使用 round-to-nearest-even。性能采纳不能替代精度验收。

### 已采用的主要优化

| 方案 | 实现方式 | 作用与边界 |
|---|---|---|
| 完整块和尾块分流 | 完整块省去重复清零与掩码处理，尾块保留显式保护 | 限制无效地址访问，减少完整块上的 Vector 工作 |
| 延迟等待 | 将 H-ready 等待移至首次消费 H 的位置 | 无依赖的预处理可先执行，消费者仍等待数据完成 |
| 共享因子预计算（C26） | 按 `HF=max(HK,HG,HB)` 生成因子，由 value heads/V 切片复用 | 避免共享头重复求解；完整计时包含 prepare 与 recurrent |
| 物理块 64（C29） | 四个 16 阶叶子构成分层三角逆，逻辑块 128 由两个物理块组成 | 减少状态编码和交接次数；不改变逐 token 递推语义 |
| Cube 内形成 delta（C28） | 用 U 初始化 L0C，按固定顺序累加负 W/H 的四项乘积 | 减少中间结果的 Vector 往返 |
| V128/V32 混合调度（C30/C35） | 整批 head 使用 V128，剩余 head 拆为四个 V32 任务 | 根据设备核数缩短最长任务；任务数并非越少越好 |
| L0 输入复用（C31） | 预载 Nh/Nl 和 Hh/Hl，连续完成四项乘法后释放 | 减少重复 MTE1 搬运；保持乘法累加顺序 |
| 直接生成 native 布局（C36/C37/C38/C39） | Vector 在 UB 中按 Cube 消费布局生成分量，再整块搬入 L1 | 减少细碎 DMA、布局重排和相关同步 |
| 五项 P（C41/C44） | Kt 两个 BF16 分量、delta 三个 BF16 分量，五项 FP32 累加 | 保留状态精度，减少第三个 Kt 分量及对应乘法 |
| prepare 流水（C49） | 在当前 score/求逆期间准备下一 work 的 Bg/Ki/Kt | 复用双缓冲空间，保护跨 work 的生产者和消费者 |
| 叶子逆配对（C51） | 一个向量的互不重叠 lane 同时处理两个独立 16 阶叶子 | 保持每个叶子的行递推顺序，无跨叶子归约 |
| 输出与 P 重叠（C52） | O 的 FIX 完成后发布独立 `O_ready`，Vector 输出与 Cube P 重叠 | 输出区与 P/H 区分离；后续状态更新仍等待 H |
| V32 行配对（C53） | 相邻两行放进一个 64-lane 向量，以互斥掩码写回 | 两行 gate 分别广播，无跨行归约；V128 路径保持 |
| Host 编译产物缓存 | 32 项 LRU 只保留可执行产物，每次创建新绑定 | 输入地址和 Tensor 不随缓存保留；使用当前 stream |

P 的累积顺序固定为：

```text
P = (((Kt0.T@D0 + Kt0.T@D1) + Kt1.T@D0) + Kt1.T@D1) + Kt0.T@D2
```

### 关键优化的实测对照

下表每行均使用该轮同一设备上的对照和候选，单位为 μs。各行可能来自不同设备和采样时段，收益不可相加，也不以跨行耗时差作为优化收益。历史汇总值保留原记录精度。

| 候选 | 对照中位数 | 候选中位数 | 结果 |
|---|---:|---:|---|
| C26：共享因子与状态分离 | 5382.305 | 4895.334 | 采用 |
| C29：物理块 64 | 4895.334 | 3924.530 | 补齐 E 的第四项后采用 |
| C28：Cube 内形成 delta | 3924.530 | 3614.649 | 采用 |
| C35：V32 尾部切片 | 3205.306 | 3022.021 | 采用 |
| C36：UB 重排后整块 DMA | 3022.021 | 2800.843 | 采用 |
| C44：P 六项缩减至五项 | 2479.038 | 2258.126 | 采用 |
| C49：prepare 流水 | 2226.716 | 2045.592 | 采用 |
| C51：叶子逆配对 | 2051.562 | 2030.593 | 采用 |
| C52：输出与 P 重叠 | 2020.026 | 1921.178 | 采用 |
| C53：V32 行配对 | 1923.849 | 1906.715 | 采用 |

### 未采用方案与限制

| 方案 | 结果 | 结论 |
|---|---|---|
| P 双分量四项（C40） | 长序列 g=0 的 FP32 最终状态未满足冻结误差界 | 不通过精度，未进入设备实现 |
| 全面使用双 BF16 分量（C24） | 完整前向 5573.030 → 6086.540 μs | 编码/搬运抵消 Cube 收益，回退 |
| 独立 head 交错（C54） | 1910.628 → 2057.359 μs | 精度通过但性能回退 |
| Qg/Kt native GM（C48） | 1909.304 → 1899.285 μs，收益小于组内极差 | 未证实稳定收益，回退 |
| U 的四项 BF16（C55） | 1948.269 → 2163.484 μs | 局部矩阵乘加快未转化为完整前向收益，回退 |
| Ki 独立 UB（C50） | 1906.793 → 1908.115 μs | 无完整前向收益，回退 |
| H 分量 UB pitch 调整（C56） | 1912.589 → 1907.129 μs，收益 5.460 μs，小于最大组内极差 14.781 μs | 未证实稳定收益，回退 |

后续可重新评估 prefix/Qg 计算融合、无消费者的 L1 搬运消除以及 Kt 的搬运重叠，但这些方案尚无新的完整前向接受结果。先检查保留实现的瓶颈，再选一个候选修改；局部 pipe 忙碌时间存在重叠，不能直接相加或相减推导端到端收益。

### 当前性能与测量方法

环境为 Ascend 950PR、CANN toolkit 9.2.0、CATLASS DSL 2.2.0。设备核数由 `torch.npu.get_device_properties(device).cube_core_num` 查询；记录中的设备为 28 个 Cube Core。每组五个独立进程，每进程预热三次，取第四次完整前向；普通 msprof 使用 `--ai-core=off --task-time=on --ascendcl=on`。

每次耗时取 `max(prepare + recurrent 的 Task Duration 之和, recurrent 结束时间 - prepare 开始时间)`。从原始十进制时间戳精确作差后计算中位数，不计 Host 编译、输入生成及 CPU 标杆。共享服务器每次采样前后检查占用；不同服务器/设备的样本不混合成一组。

| GVA | HK | HV | T | 完整前向中位数（μs） | 记录 |
|---:|---:|---:|---:|---:|---|
| 2 | 16 | 32 | 11264 | 1909.211 | C53 在新服务器设备 5 的五次采样 |
| 1 | 16 | 16 | 11264 | 1965.778 | 已提交 C53 包在设备 2 的五次采样 |
| 4 | 16 | 64 | 11264 | 3096.294 | 已提交 C53 包在设备 2 的五次采样 |

三组均使用 B=1、K=V=128、chunk_size=64、BF16、BNSD，g/b 按 HK 共享，非零 FP32 初态，返回最终状态。GVA=1/4 的长序列精度检查通过；GVA=2 与另外两组不是同卡同时采样，不据此解释 GVA 间的精确加速比例。

GVA=2 的五次数据为 1911.513、1909.211、1907.734、1907.196、1909.961 μs。早期浮点时间戳汇总曾记为 1909.304 μs；精确重算后的 0.093 μs 差异不属于优化收益。

### 三文件组织阶段的历史回归

在此前三文件组织阶段，`kernel/` 将 Host、prepare 和 recurrent 分别放在 `__init__.py`、`prepare.py` 和 `recurrent.py`。保留当前入口可达的 72 个函数与常量定义，移除 35 个不可达旧定义；kernel 目录从 14 个文件、4609 行缩减为 3 个文件、2665 行。公开调用、计算和同步顺序保持不变。

在相同 Ascend 950PR 服务器的设备 6 上，对归并前提交 `1d617c1` 与三文件版本交替采样，各五个独立进程；沿用上述完整前向计时与占用检查。主规格为 B=1、HK=16、HV=32、T=11264、K=V=128、chunk_size=64、BF16，g/b 按 HK 共享，非零 FP32 初态且返回最终状态。

| 版本 | 完整前向中位数（μs） | 最小值（μs） | 最大值（μs） |
|---|---:|---:|---:|
| 归并前 | 3764.378 | 3762.144 | 3768.443 |
| 三文件版本 | 3762.026 | 3760.336 | 3762.975 |

中位耗时变化为 -0.0625%，差异小于组内极差，本次不计作性能优化收益。该组用于同卡重构回归；归并前版本在本组也有同等耗时，不将其与设备 5 的历史 1909.211 μs 跨卡相减。不同采样组耗时差异的原因未作归因。

该阶段的三文件版本通过 91 项冻结精度/接口测试及 `python -O` 下的 16 项缓存/stream 检查。测试和采样证据独立保存，运行包不依赖这些文件。

### 两文件代码组织的性能回归

当前代码由 `__init__.py` 和 `gdn2_fwd.py` 组成，分别提供 Host 接口和全部设备实现。公开调用方式、两个设备 kernel 的启动顺序及 72 个函数与常量定义保持不变；同名矩阵乘辅助函数增加阶段前缀，避免符号覆盖。

在相同 Ascend 950PR 服务器的设备 3 上，对归并前提交 `43c4232` 和两文件代码版本交替采样，各五个独立进程。规格、预热、完整设备前向计时和占用检查沿用本节方法：B=1、HK=16、HV=32、T=11264、K=V=128、chunk_size=64、BF16，g/b 按 HK 共享，非零 FP32 初态且返回最终状态。

| 版本 | 完整前向中位数（μs） | 最小值（μs） | 最大值（μs） |
|---|---:|---:|---:|
| 归并前 | 1947.935 | 1940.443 | 1952.098 |
| 两文件代码版本 | 1942.515 | 1936.456 | 1954.381 |

中位耗时变化为 -0.2782%，差异小于组内极差，不计作性能优化收益。这组数据用于同卡结构回归，不与其他设备或采样时段的历史耗时混合。

两文件代码版本在设备 3 上通过 91 项冻结精度/接口测试及 `python -O` 下的 16 项缓存/stream 检查。测试脚本和原始采样独立保存，不属于运行包依赖。

### 性能基线

按 FLA 固定提交 `af7337db98fafd902f69e382dde993a0e426ea89` 比较 GDN 与 GDN2 的完整前向主矩阵算量。GDN 的范围包含 `chunk.py` 调用的状态和输出阶段。

主规格令 C=64，按 HV 逻辑展开共享 Q/K，块头数 `N=B*HV*(T/C)=5632`。统一以稠密等效矩阵乘计数，乘加记 2 FLOPs，两者主矩阵总量均为 `N*(6*C*C*K + 4*C*C*V + 6*C*K*V)=64.961380352 GFLOPs`；计入相同三角逆合并后均为 65.699577856 GFLOPs。

因此算量比为 1.0，由 GDN 完整前向参考耗时 1146 μs 折算的 GDN2 等效基线为 1146 μs。设备 5 的历史主规格记录对应性能比为 `1146/1909.211=0.6002`；0.8 的目标对应耗时不超过 1432.5 μs，尚未达到。

该基线不包含逐通道 gate、exp、cast、访存和同步的时间加权，也不以本实现的额外分量乘法放大算法基线；它不是 GDN2 手写实测或严格理论耗时界。该数值只适用于上述 HV=32 的主规格，不直接用于 GVA=1/4 的达标判断。
