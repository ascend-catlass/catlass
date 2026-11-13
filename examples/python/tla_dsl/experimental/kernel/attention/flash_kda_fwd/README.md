# Flash KDA forward

计算配方来自固定 [cannbot-dsl 源码](https://gitcode.com/cann/cannbot-dsl/tree/fc438d3ef03129929eb8faae3906365b3162f163/samples/attention/flash_kda_fwd)。chunk 为 64 个 token，head dimension 为 128。公开状态按 `[Dv,Dk]` 存储。

[优化方案](#优化方案)集中说明性能问题的证据、已采用的优化、精度适用条件、平均加速比和下一轮实验顺序。

## 阶段与数据类型

| 阶段 | 计算与输出 | 数值类型 |
| --- | --- | --- |
| 预处理 | Q/K L2 norm、gate、beta、chunk 内顺序前缀和 | 向量运算 FP32；g 先 RNE 转 BF16 |
| score | 带 gate 差值的 QK、KK，形成因果三角矩阵 | BF16 操作数、FP32 累加 |
| inverse | 16-token 对角块求逆，再合并至 32 和 64 | 局部 FP16 交接、FP32 累加；persistent 叶子前代使用 FP32 |
| projection | 计算 W/U、qg/kg，投影 chunk 入口状态 | BF16 操作数、FP32 累加 |
| recurrence | `v_new = u - w @ H.T`，计算 out，推进状态 | H 持续保存为 FP32；矩阵操作数快照 BF16 |
| writeback | 输出、可选反向缓存、chunk 入口 h 与最终状态 | 与本文接口定义一致 |

两类模板保留参考实现的不同舍入路径。staged 的 W/U 经 BF16 GM 交接；persistent 保留 FP32 L0C 中的 U_pre，再计算 `(-W) @ H`。不以统一 FP32 中间量替换这两条已验证的数值路径。

## 模板与调度

staged 将 chunk 预处理与跨 chunk 递推分开，使用全核同步保证生产的数据可见。workspace 容量为 1024 个槽，值维切片支持 128/64/32/16。persistent 按序列和 V 头保持状态在 UB 中，依次推进 chunk，并重叠下一块预取、前缀计算和当前递推。

Host 只计算整数调度、验证形状和分配空间，不进行输入数据的数值运算。`__init__.py` 的调度分区移植固定参考的 ABI4 调度模型，按设备实际 Cube Core 数派发；调度测试独立核对原始 861 槽控制版本与参考 ABI，并检查生产版本的 1024 槽调度。

推理模板覆盖只在内部 norm、bounded gate、`lower_bound == -5`、`abs(scale) <= 0.125` 且各序列长度为 64 整数倍时选择 persistent，其余输入保留原自动调度。persistent 的共享 score 14-panel 路径还要求不保存矩阵、不启用负特征值；其他情况保留 18-panel 路径。较弱 gate（例如 `lower_bound=-0.01`）不能套用该优化，因为两种计算配方在长序列上有可测数值差异。

## 同步与缓存

MTE、Vector、Cube 和 Fixpipe 之间用设备事件与跨核同步建立数据依赖。共享 score 的 RHS 预取必须在首次等待 `ps_left_free0` 之后，这个等待也保证前一 chunk 输出所占用的 L0B 已释放。跨 chunk 的 FP32 状态不被 BF16 矩阵快照覆盖。

Host 缓存仅包含编译结果和常量 16 阶单位矩阵；不同 stream 使用常量前等待创建 stream 的完成事件，所有本次调用 Tensor 记录消费 stream。工作区与中间缓存按调用独立分配。双 stream 测试验证常量发布顺序和两次输出的数值结果。

## 代码组织与使用

运行包仅有 `__init__.py` 和 `flash_kda_fwd.py` 两个 Python 文件，另保留本文及 [NOTICE](NOTICE)，共 4 个文件。运行不依赖 cannbot-dsl。

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 公开入口 `flash_kda_fwd`、参数检查、ABI4 整数调度、工作区和输出分配、模板选择、JIT 启动、常量缓存与跨 stream 生命周期；按参数检查、调度、调用入口分区 |
| `flash_kda_fwd.py` | 全部设备端向量/矩阵计算、状态递推、同步和 staged/persistent kernel |
| `README.md` | 使用示例、接口与缓存契约、设计、优化方案、精度和完整性能矩阵 |
| `NOTICE` | 固定参考提交、源目录、版权和许可归属 |

验证环境为 Ascend 950、CANN toolkit 9.2.0、torch `2.13.0+cpu` / torch_npu `2.13.0`、官方 CATLASS DSL `2.2.1.dev20261112`。先激活匹配的 Python 环境并加载 CANN 的 `set_env.sh`，再从仓库根目录执行：

```bash
python -m pip install --extra-index-url https://ascend-catlass.github.io/simple/ ascend-catlass-dsl==2.2.1.dev20261112
cd examples/python/tla_dsl/experimental/kernel/attention
```

在该目录运行以下 Python 代码即可通过包入口调用：

```python
import torch
import torch_npu
from flash_kda_fwd import flash_kda_fwd

torch.npu.set_device(0)
q = torch.randn((1, 2, 128, 128), device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn((1, 4, 128, 128), device="npu", dtype=torch.bfloat16)
g = torch.randn(v.shape, device="npu", dtype=torch.float32)
beta = torch.randn(v.shape[:-1], device="npu", dtype=torch.float32)
A_log = torch.zeros(4, device="npu", dtype=torch.float32)
with torch.inference_mode():
    out, final_state, *cache = flash_kda_fwd(
        q, k, v, g, beta, A_log=A_log, layout="BNSD",
        lower_bound=-5.0, output_final_state=True,
    )
torch.npu.synchronize()
```

设备端使用单个 `flash_kda_fwd.py`，依次分为共用地址/Vector/Cube/状态原语、预处理与 score、staged 两阶段及调度、persistent 专用数学与流水/调度。单文件组织保留各函数和两个 kernel 类的计算、舍入及同步边界，不把 staged 与 persistent 合并成同一种数值路径。Host metadata 逻辑位于 `__init__.py` 的调度分区，调用方仍使用原 `flash_kda_fwd` 包入口。

## 接口与计算定义

### 接口签名

Python 入口如下：

```python
def flash_kda_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    metadata: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    scale: Optional[float] = 128 ** -0.5,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> tuple[Optional[Tensor], ...]:
    ...
```

固定返回 16 项，未启用的槽位返回 `None`，不改变顺序：

```text
(out, final_state, h, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_sigmoid)
```

`metadata` 是为调用位置兼容保留的参数，本版本仅接受 `None`，调度由 Host 生成；非 `None` 在 Host 抛出 `ValueError`。不将参考 DSL 的 ABI4 调度 Tensor 直接传给新内核。性能适配器分别准备各自调度数据，不改变数学输入。

训练/推理判定沿用参考实现：`inference = torch.is_inference_mode_enabled()`；`torch.no_grad()` 不会自动切换为推理模式。本样例提供 forward 和所需反向缓存，不包含 backward 内核或自动求导注册。

#### 输入

记 `Dk=Dv=128`，`B` 为逻辑序列数，`S` 为 padded 存储长度，`T` 为 packed token 总数。`Nqk` 是共享 Q/K 头数，`Nv` 是 V/状态头数。

| 输入 | 类型 | 形状 / 规则 |
|---|---|---|
| q、k | BF16 | BNSD: `[B,Nqk,S,128]`；BSND: `[B,S,Nqk,128]`；TND: `[T,Nqk,128]` |
| v、g | v 为 BF16；g 为 BF16 或 FP32 | 对应布局的 `[B,Nv,S,128]`、`[B,S,Nv,128]` 或 `[T,Nv,128]` |
| beta | BF16 或 FP32 | v 去掉最后一维 |
| initial_state | FP32，可选 | `[B,Nv,Dv,Dk]`；缺省为零状态 |
| A_log | FP32 | `[Nv]`；启用 gate 计算时必需 |
| dt_bias | FP32，可选 | `[Nv,128]`；缺省按零偏置 |
| cu_seqlens | INT32，可选 | `[B+1]`；TND 必需，首项 0、严格递增、末项 T |
| seqused | INT32，可选 | `[B]`；padded 布局的有效长度；若与 cu_seqlens 同时提供，两者长度必须一致 |

输入 Tensor 连续且位于同一 NPU；长度 Tensor 也在该设备上。不修改任何输入，不将输出别名到输入。`scale=None` 等价于 `128**-0.5`，其他值须为有限实数。所有开关须为 bool。

#### 输出和缓存

CHUNK 固定为 64。padded 布局 `J=ceil(S/64)`；TND 的 chunk 槽位容量 `J=ceil(T/64)+B`，序列 b 的首槽为 `floor(cu_seqlens[b]/64)+b`。h 是每个 chunk **进入计算前**的状态，最后状态单独由 final_state 返回。

| 输出 | dtype / shape | 返回条件 |
|---|---|---|
| out | BF16，与 v 同形状、同布局 | 始终返回 |
| final_state | FP32 `[B,Nv,128,128]`，末两维 `[Dv,Dk]` | `output_final_state=True` |
| h | FP32 `[B,J,Nv,128,128]`；TND `[J,Nv,128,128]` | `return_intermediate_states=True`，训练和推理均支持 |
| Aqk、Akk | BF16 `[B,J,Nv,64,64]`；TND `[J,Nv,64,64]` | 训练模式 |
| w、u、qg、kg、v_new | BF16，与 v 同形状、同布局；Q/K 按 GQA 映射展开到 Nv | 训练模式且 `disable_recompute=True` |
| g_cumsum | FP32 `[B,J,Nv,64,128]`；TND `[J,Nv,64,128]` | 训练模式且 `disable_recompute=True` |
| q_norm、k_norm | BF16；padded 固定 `[B,Nqk,S,128]`；TND `[T,Nqk,128]` | 训练模式且 `use_qk_l2norm_in_kernel=True` |
| q_rstd、k_rstd | FP32；padded `[B,Nqk,S]`；TND `[T,Nqk]` | 同 norm |
| beta_sigmoid | FP32；padded `[B,S,Nv]`；TND `[T,Nv]` | 训练模式且 `use_beta_sigmoid_in_kernel=True` |

`disable_recompute` 只控制六项缓存的保存，不改变 out、final_state、Aqk/Akk 或所请求 h 的数学定义。关闭相应输出时可以省略写回，但不能省略计算依赖。未提供逐缓存 mask；可选性采用上表的组合开关。

### 支持范围

#### 平台和形状

- A5 / Ascend 950，CANN 9.2.0，CATLASS DSL `2.2.1.dev20261112`。
- `1 <= Nqk <= Nv <= 96`，`Nv % Nqk == 0`。V 头 j 使用 Q/K 头 `floor(j / (Nv/Nqk))`。
- BNSD、BSND、TND；B 和各序列有效长度均为正整数，支持长度不为 64 整数倍的尾块。
- B、S/T 不增加模型特有的固定上限；Host 使用足够宽的整数检查元素数、字节数、工作区和设备寻址范围，内存不足报告实际分配错误。
- FP32 initial_state、final_state、h；跨 chunk 的持久状态和状态更新保持 FP32。矩阵乘法的 BF16 操作数副本不能覆盖 FP32 持久状态。

#### 计算语义

以下公式给出实数算法定义。浮点数值验收对齐固定 NPU 参考，容差见[优化方案](#优化方案)。

1. 与指定 NPU 参考保持输入转换一致：g 先转为 BF16，再升为 FP32 参与 gate；beta 转为 FP32。因而“g 兼容 FP32”表示接受该输入类型，不承诺保留超过 BF16 的输入有效位。
2. 启用 L2 归一化时，`rstd(x)=rsqrt(sum(x*x)+1e-6)`，`x_norm=x*rstd(x)`；Q/K 归一化结果以 BF16 供后续矩阵计算使用。关闭时直接使用传入 Q/K。
   关闭内核归一化时，调用方须提供外部 L2 归一化后的 BF16 Q/K；Host 不重复归一化。未归一化的大范数输入不在已验证精度范围内。
3. 令 `a=exp(A_log)`、`x=float(bfloat16(g))+dt_bias`。启用 gate 时：`lower_bound=None` 使用 `ell=-a*softplus(x)`；否则使用 `ell=lower_bound*sigmoid(a*x)`，且 `-5 <= lower_bound < 0`。关闭 gate 时，g 表示已经激活的逐维 log decay，A_log/dt_bias/lower_bound 不参与计算。
4. 启用 beta 激活时，`b=sigmoid(beta)`，`allow_neg_eigval=True` 则 `b=2*sigmoid(beta)`。关闭时直接使用 beta；此时不允许 `allow_neg_eigval=True`。
5. 每个 V 头的状态 `H` 按 `[Dv,Dk]` 存放，逐 token 的定义为：

```text
H_decay = H_prev * exp(ell_t)[None, :]
delta_t = b_t * (v_t - H_decay @ k_t)
H_t     = H_decay + delta_t[:, None] * k_t[None, :]
out_t   = scale * (H_t @ q_t)
```

6. chunk 内从零开始累计 `G_i=sum(ell_0...ell_i)`，`Gamma_i=exp(G_i)`。令三角矩阵：

```text
L[i,j]   = b_i * sum_d(k_i[d] * k_j[d] * exp(G_i[d]-G_j[d]))  if i>j, else 0
Akk      = inverse(I + L)
Aqk[i,j] = scale * sum_d(q_i[d] * k_j[d] * exp(G_i[d]-G_j[d])) if i>=j, else 0
w        = Akk @ (b[:,None] * k * Gamma)
u        = Akk @ (b[:,None] * v)
qg       = q * Gamma
kg       = k * exp(G_last[None,:] - G)
v_new    = u - w @ H_chunk_in.T
out      = (scale * qg) @ H_chunk_in.T + Aqk @ v_new
H_next   = H_chunk_in * exp(G_last)[None,:] + v_new.T @ kg
```

公开 w 为正号定义；内部采用负 w 时必须在输出前恢复。qg 不包含 scale；Aqk 包含 scale；Akk 为单位下三角逆矩阵，不是未求逆的 KK score。g_cumsum 保存 G，h 保存 H_chunk_in。实现避免先生成不必要的大指数再相乘，使用因果 gate 差值的稳定求值路径。

### 边界行为

- 长度 1、15、16、17、63、64、65、127、128、129 纳入精度测试；尾块只处理有效 token，padding 不参与状态更新。TND 序列之间不共享状态。
- padded out 的无效 token 置零；缓存只保证有效 token、有效 chunk/矩阵行列的数值。未使用 chunk 槽及无效矩阵行列不供调用方读取，不纳入误差统计；g_cumsum 的尾 token 置零。
- 空 batch、零有效长度、D 非 128、头数不整除或 Nv 超过 96、不合法布局/dtype/shape/stride/device，以及无效长度前缀，在启动计算内核前抛出 `ValueError`。
- TND 的 cu_seqlens 末项必须等于 T；padded 中 cu_seqlens 差分须不大于 S。若 cu_seqlens 和 seqused 同时存在，逐序列长度必须相等。缺省 padded 长度为 S。
- gate 开启但 A_log 缺失、非法 lower_bound、非有限 scale、非 bool 开关，以及 `allow_neg_eigval=True` 与 beta 激活关闭的组合，抛出 `ValueError`。gate 关闭时忽略 A_log、dt_bias、lower_bound。
- 模型有效输入要求有限数；关闭 gate 的 g 应为非正 log decay，关闭 beta 激活的 beta 应为已激活系数。NaN/Inf 的传播不属于精度达标域，不静默清洗或更改数学定义。
- 初始状态省略等价于 FP32 全零；初始状态不被原地更新。最终状态是最后有效 token 后的状态。h 的第一个有效 chunk 等于该序列 initial_state。
- `disable_recompute`、`allow_neg_eigval` 分别覆盖 True/False；全部缓存逐项检查 shape、dtype、返回条件、有效域精度；h 同时覆盖训练和推理。错误和边界测试不能由性能 case 替代。

## 优化方案

本方案说明已经实现的优化及后续实验顺序。计算基础是固定 cannbot-dsl `fc438d3ef03129929eb8faae3906365b3162f163` 的 staged/persistent 方案；不会将继承的算法结构全部计为新增优化。阶段与数值类型见[设计](#阶段与数据类型)，精度与性能组合见本文。

### 目标与不变条件

主目标是完整 forward 的 20 个冻结组合逐项不慢于固定 NPU 参考。平均值用于概括结果，不能抵消任何单项回退。每项加速比为参考设备耗时中位数除以候选中位数，双方各五个独立进程；Host API 耗时单列。

保持 FP32 持续状态、BF16 主矩阵操作数与 FP32 累加，以及原参考的 FP16 局部逆矩阵交接点。BF16/FP32 g/beta、缓存、h、尾块和 GQA 契约不变；所有有效输出元素必须满足固定 NPU 默认容差。改变融合、调度或舍入点后重新实测，不从输入 dtype 推断数值一定合格。

### 优化依据与优先顺序

初次流水翻译的 P01/cache 完整设备耗时为 708.292 μs，同轮参考为 419.090 μs。早期拆分诊断中，两轮合计 Stage1 为 384.134 μs，Stage2 为 320.075 μs；显式 store mask 后对应为 312.371 μs 和 199.434 μs。采样及来源报告哈希随原始诊断材料归档。

这里的 Stage1 是预处理、score、inverse 和 W/U；Stage2 是跨 chunk 状态递推及输出。上述计时来自隔离诊断 kernel，包含拆分边界开销，只用于定位优化方向；既不是最终版本的阶段耗时，也不能相加后冒充最终融合 kernel 的精确分摊。Vector/Cube 流水可重叠，profiler 利用率也不能直接转成串行阶段百分比。

据此先处理 Vector 端多余指令与中间读写，再恢复计算/搬运重叠，最后调整已验证分支的操作数数量和模板选择。

### 已实现的措施

| 位置 | 措施 | 降低的开销 | 约束与实现入口 |
| --- | --- | --- | --- |
| Vector 存储 | 对宽度和分配已证明的普通 SIMD store 显式指定 mask | 动态子视图自动生成的边界谓词指令 | 保留有效行/尾列处理；`FIRST_ELEMENT_B32` 路由不能加 mask。见 `flash_kda_fwd.py` 的 Vector/状态原语分区 |
| gate、norm、操作数 | 寄存器内保留所需舍入；合并归一化/score 生产、直接打包 NZ；完整 64 行使用静态循环，尾块走动态有效行路径 | VF 调用、重复 UB 读写和行内分支 | 顺序 FP32 prefix 和 BF16 RNE 位置不变。见 `flash_kda_fwd.py` 的 `prefix_inputs`、`pack_recurrent`、`persistent_pipeline_chunk` |
| staged 流水 | 保留源方案的延迟 W/U 生产、Q/K/G/score 双缓冲；工作区由 861 扩至 1024 槽；按值维切片成本调度 | 不必要的轮次与交接等待 | 全核边界必须排空；DV=128/64/32/16 均保留。见 `flash_kda_fwd.py` 的 `stage_one_pipeline`、`StagedMetadataKernel` 和 `__init__.py` 的调度分区 |
| persistent 流水 | 预取下一 chunk Q/K/V/g/beta；双 L1 递推槽及双 gamma 支持下一 prefix/score 与当前递推重叠 | DMA、Vector 与 Cube 串行等待 | 当前数据的所有读者释放后才覆盖；保留 FP32 驻留状态。见 `flash_kda_fwd.py` 的 `persistent_pipeline_chunk`、`prefetch_half` |
| 依赖与状态发布 | KK 完成即交付叶子求逆；Aqk 提前发布；状态增量完成后先交付 Vector；FP32 状态更新同时生成下一块 BF16 快照 | 等待无关输出完成、重复状态快照转换 | 按真实消费者收紧事件依赖，不直接删除同步。见 `flash_kda_fwd.py` 的 persistent score/状态及 staged 递推分区 |
| 共享 score | 满足条件的 persistent 分支由 18 个 panel 减至 14 个，右侧从 10 份减至 6 份；使用 G16/G48 中心 | 每个 AIV 每 chunk 少 4096 个 exp 元素及 4 份 RHS 生产/搬运；score 行数 288→224 | 有效 score 数据量减少 22.2%，不等于 kernel 耗时减少 22.2%。见下一节与 `flash_kda_fwd.py` 的 `persistent_score_mmads` |
| 模板选择 | 在已验证的整 chunk 推理分支选择 persistent，其余使用原自动调度 | staged 中间 GM 交接与跨阶段同步 | 精度条件见下一节；不能对所有 gate 值强制切换。见 `__init__.py` |
| 参数与常量 | 同 head 任务复用 A_log/dt_bias；按设备缓存 16 阶单位矩阵并用事件保证跨 stream 发布 | 重复参数加载与逐调用 eye 设备任务 | 不缓存 Q/K/V/g/beta、输入状态或数值输出。见 `__init__.py` 和 `flash_kda_fwd.py` 的 `StagedMetadataKernel` |

这组措施包含对参考方案的忠实恢复和 CATLASS DSL 的专门优化。最终 20 项结果验证的是组合方案；没有完整逐项消融数据的措施，不单独声明其百分比收益，也不将不同轮次的改善相加。

### 共享 score 和模板选择的精度边界

两项选择分别判断：

- 推理改选 persistent：内部 Q/K norm、bounded gate、`lower_bound == -5`、`abs(scale) <= 0.125`，且所有序列有效长度为 64 的整数倍。
- persistent 14-panel：不保存 Aqk/Akk、内部 Q/K norm、bounded gate、`lower_bound == -5`、`allow_neg_eigval=False`、`abs(scale) <= 0.125`。该判断自身不要求全部序列整 chunk，仍需正确处理尾块。

未满足条件时保留原自动模板或 18-panel 计算路径。曾将适用范围放宽到弱 gate，但 `S=2048, lower_bound=-0.01` 的扩展检查出现 final_state 超出固定容差，故最终限制为 `lower_bound == -5`，没有放宽阈值或删除该输入支持。

RHS 预取必须在首次 `ps_left_free0` 等待之后进行，因为这一等待同时释放前一 chunk 输出占用的 L0B；仅靠矩阵公式等价不能证明流水复用安全。

### 当前收益与平均加速比

单文件版本 `585fce4` 重新测量的完整 20 组合等权算术平均为 **1.340 倍**，几何平均为 **1.322 倍**；最小 1.013 倍、最大 1.842 倍，20/20 项通过。统计对象是每项五进程中位数的比值。以下表格为该单文件版本的测量，不复用原分模块版本的采样；后续静态整改及 Host/文档合并未重新采集完整性能矩阵。

| 模式 | 组合数 | 等权算术平均 | 几何平均 |
| --- | ---: | ---: | ---: |
| infer | 5 | 1.352 | 1.310 |
| recompute | 5 | 1.208 | 1.199 |
| cache | 5 | 1.360 | 1.353 |
| cache_h | 5 | 1.441 | 1.437 |
| 全部 | 20 | 1.340 | 1.322 |

令 `B_i`、`C_i` 分别为参考与候选的设备耗时中位数，每项比值为 `r_i=B_i/C_i`，等权算术平均为 `sum(r_i)/20=1.340216`，几何平均为 `exp(sum(log(r_i))/20)=1.322031`。计算使用未舍入数值；完整输入与单项耗时见下文。

### 下一轮优化计划（尚未执行）

| 优先级 | 目标与当前余量 | 实验 | 通过依据 |
| --- | --- | --- | --- |
| 1 | P03/infer 1.013 倍；P05/infer 1.044 倍 | 在最终源码上重新采集 score/inverse、prefix、recurrence 的等待与资源占用，先定位最长依赖链，再分别试验预取距离和事件发布位置 | 先定向精度，再同轮五进程完整 forward；增益须区别于采样波动，且其余组合不回退 |
| 2 | P01/P02 recompute 1.049/1.061 倍 | 复核 staged 的 DV 成本、轮次与缓存写回重叠；对单个因素做消融 | 不改变公开缓存条件和三角矩阵计算；保留所有 DV/尾块回归及 20 项验收 |
| 3 | P05 变长链与 Host 开销 | 诊断不同核的链长度/有效 token/等待，不只按 chunk 总数推断均衡；Host 整数调度与分配耗时单独测量 | 设备任务和同步 API 耗时分别报告；不缓存数据相关结果，不改变计时范围 |

P03 启用 `allow_neg_eigval`，不能直接套用已有 14-panel 快路径；P05 已使用可适用的共享 score 路径，不能把再开启同一优化列为新收益。此前独立 Q/K norm 预处理、部分同步协议删减、任务成本微调等实验未显示稳定的推理收益，不作为默认后续路线；重新尝试时须有新的 profiler 证据。

每次实验只修改一个因素：固定输入/参考/源码 → 定向精度 → 同轮诊断 → 完整 20 项精度 → 每后端五独立进程性能。保留每项原始样本和源码哈希，拒绝用平均值掩盖单项失败。仓库的 200 场景、专项内存和确定性检查与性能优化分别跟踪，尚未完成。

### 精度与测量环境

实测环境：Ascend 950、CANN toolkit 9.2.0、CATLASS DSL `2.2.1.dev20261112`，torch `2.13.0+cpu` / torch_npu `2.13.0`。数值与性能参考固定为 cannbot-dsl `fc438d3ef03129929eb8faae3906365b3162f163` 的 `samples/attention/flash_kda_fwd/`。

数值判据为 `abs(actual-reference) <= atol + rtol*abs(reference)`。主要 BF16 输出及 FP32 状态、h、g_cumsum 使用 `atol=rtol=5e-3`；Q/K norm 使用 `1e-3`；rstd 和 beta_sigmoid 使用 `1e-6`。所有有效元素必须通过，不额外设置最大绝对误差上限；shape、dtype、有限值、三角结构、首个 h 和 padding 另外检查。FP64 数学标杆仅作诊断，数值验收对齐上述固定 NPU 实现。

20 个模型组合共 9,414,450,896 个有效元素通过；模型 out 最大绝对误差 0.0009765625，final_state 为 0.00091552734375。19 个接口用例包含长序列弱 gate、尾块、变长、三种布局、外部归一化和缓存开关；另有 39 个启动前拒绝检查、双 stream、16 个调度检查。API 用例中的 h 不保证逐位相同，均按固定容差判断。


### 完整性能矩阵

所有组合均为 BF16 Q/K/V、FP32 g/beta、非零 FP32 initial_state，启用 L2/gate/beta sigmoid，`lower_bound=-5`，请求 final_state，scale 默认。

| case | layout | B | 有效长度 | Nqk / Nv | allow_neg_eigval |
| --- | --- | --- | --- | --- | --- |
| P01 | BNSD | 1 | 2048 | 8 / 32 | False |
| P02 | BSND | 1 | 8192 | 32 / 96 | False |
| P03 | BNSD | 1 | 16384 | 96 / 96 | True |
| P04 | BSND | 4 | 2048 | 8 / 32 | True |
| P05 | TND | 4 | 63, 1025, 2048, 4097 | 8 / 32 | False |

infer 为推理无 h；recompute 为训练重算；cache 为训练保存缓存；cache_h 在 cache 基础上保存 h。每个后端运行五个独立进程，预热和编译后用普通 msprof 统计一次完整 forward 的全部设备任务耗时之和，取五次中位数。baseline 复用按其 API 预建的 metadata；候选内部生成调度。metadata 构建和同步 Python API 时间单独采集，不混入下表的设备任务耗时。

原参考训练分支不导出 h；对 cache_h 的参考仅将验证模块中的 `chunk_states=inference and return_intermediate_states` 改为 `chunk_states=return_intermediate_states`。原件和导出副本逐文件校验，设备数学计算不变，双方均执行状态写出。

| case | mode | baseline (μs) | CATLASS DSL (μs) | 加速比 |
| --- | --- | ---: | ---: | ---: |
| P01 | infer | 299.082 | 253.169 | 1.181 |
| P01 | recompute | 346.462 | 330.267 | 1.049 |
| P01 | cache | 428.661 | 353.673 | 1.212 |
| P01 | cache_h | 520.839 | 400.021 | 1.302 |
| P02 | infer | 3325.402 | 1979.340 | 1.680 |
| P02 | recompute | 3837.305 | 3618.363 | 1.061 |
| P02 | cache | 4905.670 | 3933.499 | 1.247 |
| P02 | cache_h | 6029.713 | 4405.852 | 1.369 |
| P03 | infer | 4048.355 | 3995.751 | 1.013 |
| P03 | recompute | 6201.541 | 4586.489 | 1.352 |
| P03 | cache | 7831.319 | 5627.291 | 1.392 |
| P03 | cache_h | 11276.767 | 6935.120 | 1.626 |
| P04 | infer | 1274.320 | 691.909 | 1.842 |
| P04 | recompute | 1487.641 | 1268.198 | 1.173 |
| P04 | cache | 1848.438 | 1387.662 | 1.332 |
| P04 | cache_h | 2204.817 | 1526.958 | 1.444 |
| P05 | infer | 567.467 | 543.723 | 1.044 |
| P05 | recompute | 1050.314 | 747.400 | 1.405 |
| P05 | cache | 1600.629 | 989.697 | 1.617 |
| P05 | cache_h | 1662.221 | 1135.720 | 1.464 |

20/20 项加速比不低于 1.0；最低为 P03/infer 的 1.013，最高为 P04/infer 的 1.842。最低项余量为 1.3%，重新测量受设备负载和环境影响，不推广到未测输入。

本轮在共享服务器的 NPU 4 上采集，期间观察到同卡其他作业及高耗时样本。保留全部五次采样，按预定中位数口径统计，未按结果筛除样本；这些数值不代表独占环境性能，也不用于声称文件合并产生加速收益。

### 验证范围

单文件版本 `585fce4` 在 Ascend 950 上使用独立编译缓存重新验证，通过 19 项接口、39 项异常输入、16 项调度、双 stream、完整 20 个模型组合精度及每后端五个独立进程的完整 forward 性能测量。该次验证的运行包包含 5 个 Python 文件，全部设备端实现位于当时的 `kernel.py`（现命名为 `flash_kda_fwd.py`）；合并时 102 个函数/类 AST 与合并前一致，Host metadata AST 及 API 非导入逻辑也一致。

版本 `c86efb2` 针对 `G.FNM.07` 删除 5 处未使用的 Tensor 视图构造：`stage_one_pipeline` 的 `nt`、`cast`、`gate_bf16`，以及 `stage_two_group` 和 `persistent_pipeline_chunk` 的 `snapshot`。这些变量没有读取点；移除对应赋值后，其余 AST 与 `585fce4` 一致，UB 地址偏移、计算和同步逻辑未修改。整改后在 Ascend 950 上重新通过 19 项接口和双 stream 检查，当时全部 5 个运行源码文件的 SHA256 与候选包一致；容差保持不变。

当前四文件版本使用 `__init__.py` 集中实现 Host 逻辑。文件合并时，28 个函数及相关常量的非导入 AST 与 `c86efb2` 一致，`flash_kda_fwd.py` 的可执行 AST 也一致。后续仅在公开入口增加调度返回结构、状态码校验，以及 `ValueError`、`RuntimeError` 的调度失败上下文，按原类型保留原因链重新抛出；这些调用边界检查之外的 AST 与 `dba2379` 一致，设备文件逐字节不变。公开导入方式不变；在 Ascend 950 上重新通过 19 项接口、39 项异常输入、16 项调度和双 stream 检查，两个 Python 文件的 SHA256 与候选包一致。另有 5 项调度故障注入覆盖两类异常、无返回值、返回数量错误和非零状态码，均在编译前拒绝。文档和源码组织合并不计为新增性能优化。

上述四文件版本的完整接口、异常、调度及故障注入回归对应 `828b3ae`。设备文件重命名为 `flash_kda_fwd.py` 后，内容与原设备文件逐字节一致，Host 仅调整导入与说明；在 Ascend 950 上重新验证新模块导入、公开函数入口和 JIT 实际执行，通过 19 项接口及双 stream 检查。本次重命名未重新采集完整模型和性能矩阵。

当前覆盖 20 个模型组合和 19 个接口用例，尚未达到仓库要求的至少 200 个精度场景；尚未执行专项内存检测和完整确定性检测。空 Tensor、NaN/Inf 标量 scale 等入口拒绝，不等同于所有 Tensor 特殊值场景。上述未完成项不记为通过。

测试与采集脚本、逐样本结果、参考文件哈希、源码对应关系及静态检查记录单独归档，不属于运行包内容。本文保留环境、输入、容差、计时定义及完整单项中位数，用于界定性能结论的适用范围。
