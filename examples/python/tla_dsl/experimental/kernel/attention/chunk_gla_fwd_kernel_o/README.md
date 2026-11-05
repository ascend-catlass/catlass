# ChunkGlaFwdKernelO Example Readme

## 功能说明

- 算子功能：计算 chunk 内门控线性注意力的输出，将门控 Q 与历史状态 h 的矩阵乘法、因果 A 与 V 的矩阵乘法及输出累加融合到一个 CATLASS DSL AIC/AIV mixed kernel 中。
- 计算公式：对每个 chunk 和 head，数学表达式为：

  $$
  O = \mathrm{scale}\,(Q \odot 2^G)H + \operatorname{tril}(A)V
  $$

  其中 Q、G、V 的单个 chunk shape 为 `(BT, D)`，H 为 `(D, D)`，A 为 `(BT, BT)`；`tril` 保留下三角及对角线，输出 O 为 `(BT, D)`。

  kernel 使用 FP16/BF16 矩阵输入及 FP32 累加，最终转换为与 Q 相同的 dtype。实际计算包含门控 Q、H 和因果 A 的低精度转换，公式表示数学语义，不表示逐位相同的浮点运算顺序。

- 支持产品型号：Ascend 950PR、950DT 系列产品；本次实际验证使用 Ascend 950PR。

## 参数说明

ChunkGlaFwdKernelO 的输入输出参数如下。记 `NT` 为所有 batch 或变长序列的 chunk 总数，`BT=chunk_size`。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | `(B, T, H, D)`（BSND）或 `(T, H, D)`（TND） | `float16` / `bfloat16` | query 张量。 |
| `v` | Input | 与 q 相同 | 与 q 相同 | value 张量，当前要求 value 维度等于 D。 |
| `g` | Input | 与 q 相同 | 与 q 相同 | 门控张量，kernel 内计算 `exp2(g)`。 |
| `A` | Input | `(B, T, H, BT)`（BSND）或 `(T, H, BT)`（TND） | `float32` | chunk 内系数矩阵；kernel 使用其下三角部分。 |
| `h` | Input | `(NT, H, D, D)` | `float32` | 每个 chunk、每个 head 对应的历史状态。 |
| `scale` | Input | 标量 | Python `float` | 门控 Q 与历史状态乘积的缩放因子。 |
| `cu_seqlens` | Input，可选 | `(序列数 + 1,)` | 整数 Tensor | TND 模式的累计序列边界；BSND 模式为 `None`。命令行以逗号分隔的整数传入。 |
| `o` | Output | 与 q 相同 | 与 q 相同 | 融合计算结果。 |

BSND 模式下 `NT=B*T/BT`；TND 模式下 `NT` 为各序列的 chunk 数之和，满足对齐约束时等于 `T/BT`。

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 输入维度 | B、T、H 为正整数；D 支持 64 或 128；q、v、g 的 shape 和 dtype 一致。 |
| chunk 大小 | `chunk_size` 支持 64 或 128。 |
| BSND | T 必须能被 chunk 大小整除。 |
| TND | 命令行 B 必须为 1；T 表示所有序列的 token 总数；必须传入 `--cu-seqlens`。 |
| 序列边界 | `cu_seqlens` 至少包含两个元素，首项为 0、末项为 T、严格递增，每条序列长度均能被 chunk 大小整除。 |
| 状态与系数 | h 的 shape 必须为 `(NT, H, D, D)`；A 的最后一维必须等于 BT。 |
| 缩放 | scale 必须为有限数值。 |
| 布局与设备 | 输入张量位于同一 NPU；host 将公开布局打包为连续 task stream。 |

当前不支持 `state_v_first=True`。命令行不提供显式 `chunk_indices` 参数，TND 调用根据序列边界生成 chunk 索引。

D=128 直接由 kernel 执行宽矩阵 MMA，不拆分成多次算子调用。BT128 的因果 A 临时缓冲区在 host 侧初始化为零，确保 kernel 未写入的右上角 64×64 分块不会携带旧数据。

## 使用示例

### 命令行参数

```bash
python3 chunk_gla_fwd_kernel_o.py B T H D [deviceId] [选项]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `B` | 必填 | BSND 输入 batch 大小；TND 模式必须填写 1。 |
| `T` | 必填 | BSND 单个 batch 的序列长度；TND 所有序列的 token 总数。 |
| `H` | 必填 | 输入的 head 数。 |
| `D` | 必填 | 输入特征维度，支持 64 或 128。 |
| `deviceId` | `0` | 指定运行设备 ID；省略时默认使用设备 0。 |
| `--mode` | `bsnd` | 输入布局，支持 `bsnd` 和 `tnd`。 |
| `--chunk-size` | `64` | chunk 大小，支持 64 或 128。 |
| `--dtype` | `float16` | q、v、g 和输出的 dtype，支持 `float16` 或 `bfloat16`；A、h 输入为 FP32。 |
| `--scale` | `0.125` | state 分支的缩放因子。 |
| `--cu-seqlens` | `None` | TND 模式必填的累计序列边界，以逗号分隔，例如 `0,128,256`；BSND 模式不能指定该选项。 |
| `--block` | `28` | kernel 启动的 block 数量。 |
| `--seed` | `42` | 生成随机输入所用的种子。 |
| `--rtol` | `0.04` | 精度比较的相对误差容限。 |
| `--atol` | `0.04` | 精度比较的绝对误差容限。 |
| `--force-recompile` | `False` | 是否强制重新编译 kernel；指定时为 `True`。 |
| `--no-cache` | `False` | 是否禁用 kernel 编译缓存；默认使用缓存，指定时禁用。 |
| `--cache-dir` | 脚本目录下的 `artifacts/runtime-cache/mixed-generalized` | kernel 编译缓存根目录。显式传入相对路径时，相对于当前工作目录解析。 |
| `--build-only` | `False` | 只编译 kernel，不执行精度校验；与 `--dump-tlair` 互斥。 |
| `--dump-tlair` | `False` | 输出 TLAIR，不执行 kernel；与 `--build-only` 互斥。 |

### 执行示例

先加载配套的 CANN、PyTorch、torch_npu 和 CATLASS DSL 环境。进入项目根目录，切换到样例目录并执行：

```bash
cd examples/python/tla_dsl/experimental/kernel/attention/chunk_gla_fwd_kernel_o
python3 chunk_gla_fwd_kernel_o.py 1 128 2 64 --chunk-size 64 --force-recompile --no-cache
```

上述命令省略设备参数，默认使用设备 0。q、v、g 的 shape 为 `(1, 128, 2, 64)`，A 为 `(1, 128, 2, 64)`，h 为 `(2, 2, 64, 64)`，输出类型为 FP16。脚本根据参数生成输入、编译并执行 kernel，与 CPU 参考实现比较。

执行结果如下，说明样例执行成功，精度通过：

```text
accuracy name=o pass=True max_abs=0.0078125 mean_abs=2.0812585717067122e-06 max_rel=0.0009132419945672154 mismatch=0/16384 matched_ratio=1.000000000 first_mismatch=None
validation uuid=direct-bsnd-float16-bt64 pass=True
```

`max_abs` 为最大绝对误差，`mismatch` 为不满足容差条件的元素数。具体误差值可能随输入和运行环境变化，以输出为有限数且 `validation ... pass=True` 表示验证通过。

运行 TND、BF16、BT128 用例，两个序列长度均为 128：

```bash
python3 chunk_gla_fwd_kernel_o.py 1 256 2 64 --mode tnd --cu-seqlens 0,128,256 --chunk-size 128 --dtype bfloat16
```

仅编译或查看 TLAIR：

```bash
python3 chunk_gla_fwd_kernel_o.py 1 128 2 64 --build-only
python3 chunk_gla_fwd_kernel_o.py 1 128 2 64 --dump-tlair
```
