# NLLLoss Example Readme

## 功能介绍

本样例实现 FP32 `NLLLoss` 与 `CrossEntropy` 前向，入口为
`nll_loss.nll_loss.run`。输入 `x` 的形状为 `[N, C]`，每行输出一个 loss；`sum` 与
`mean` 在 host 侧对长度为 `N` 的中间结果完成最终归约。

```text
NLL:  loss_i = -x[i, t_i] * w[t_i]

CE:   m_i    = max_j x[i, j]
      lse_i  = m_i + log(sum_j exp(x[i, j] - m_i))
      loss_i = (lse_i - x[i, t_i]) * w[t_i]
```

- `from_logits=True` 执行稳定的 log-sum-exp；`from_logits=False` 表示输入已经是
  log-probability，只需 gather 目标类别。
- `ignore_index` 行在 host 侧把 target 改写为 0，避免 kernel 越界；这些行随后从
  `sum/mean` 中剔除，`reduction="none"` 时对应输出置 0。
- `weight=None` 时不加载类别权重；`weight` 非空时按 target 类别逐行施加。
- 当前支持 Ascend 950PR&950DT系列产品，编译目标为 `--npu-arch 3510`。

## 高性能设计

- **One-row-one-owner**：每行由单个 AIV block 独占，按 block 步长处理多行。行内
  max/sum 不需要跨 block 归约，也没有 atomic 更新。
- **双槽 row pipeline**：每个 block 使用两份 padded UB 行和两份标量输出槽。当前行
  由 VECTOR 扫描时，MTE2 可填充另一槽，MTE3 可排空已完成槽；末尾统一做一次 pipe
  barrier，而不是每行等待全部 pipe。
- **64-lane SIMD 扫描**：每次处理一个 64-element FP32 chunk。logits 路径先求行
  maximum，再计算 `exp(x-max)` 的累加和，避免数值溢出。尾块使用独立 mask，head 与
  tail 分别归约后合并，padding lane 不进入 max、sum 或输出。
- **设备侧热点**：C 宽的 log-sum-exp、目标 gather 和类别权重 gather 全部在 Device
  完成；host 只处理 O(N) 的掩码和最终标量归约。
- **显式 mutex 流水**：两个输入槽分别使用独立 `MTE2 -> VECTOR` mutex，两个输出槽分别
  使用独立 `VECTOR -> MTE3` mutex。不依赖 `set_flag/wait_flag`。
- **Host 快路径**：Device 产物按 `(N, C, HAS_WEIGHT, FROM_LOGITS)` 缓存；重复调用复用
  `_Tensor` 描述符、预打包 launch payload、target sanitize 结果和标量 reduction scratch。
  缓存键包含输入版本和目标 stream；输入原地更新或切换 stream 时会使缓存失效。
- **复杂度**：Device 为 `O(N*C)`；host 归约为 `O(N)`。

## 代码组织

```text
nll_loss
├── README.md
├── bench_nll_loss.py  # 配对性能基准
└── nll_loss.py        # TLA kernel、host wrapper 和精度自检
```

## 参数说明

```python
loss = run(
    x,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    from_logits=False,
)
```

| 参数 | 方向 | 形状 / 类型 | 说明 |
| --- | --- | --- | --- |
| `x` | Input | `[N, C]`, `float32` | logits 或 log-probability |
| `target` | Input | `[N]`, `int64` | 类别索引；可包含 `ignore_index` |
| `weight` | Input | `[C]` 或 `None`, `float32` | 类别权重 |
| `ignore_index` | 属性 | Python `int` 或 `None` | 忽略的目标值 |
| `reduction` | 属性 | `"none" / "sum" / "mean"` | 输出归约方式 |
| `from_logits` | 属性 | Python `bool` | 是否执行 log-sum-exp |
| 返回值 | Output | `[N]` 或标量, `float32` | 每行 loss 或归约结果 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `x` 必须为 2-D，`target` 必须为 1-D 且长度等于 `x.shape[0]` |
| 特征维 | `C >= 1`；支持任意非 64 倍数，尾块按 64-lane mask 处理 |
| 行宽对齐 | GM 行宽可为非 32Byte 对齐；UB 使用 `C_PAD = ceil(C/64)*64` 的 padded stride |
| UB 容量 | 两个 `C_PAD` FP32 row slot 加两个标量输出槽；当前上板覆盖至 `C=4096` |
| dtype | `x`、`weight` 为 FP32；`target` 会转换为 int64 |
| device | `x`、`target`、`weight` 必须位于同一个 NPU |
| target 范围 | 每个元素必须位于 `[0, C)`，或等于 `ignore_index` |
| ignore 处理 | ignore 行由 host 掩码；kernel 只接收替换为 0 的安全 target |
| weighted mean | 分母为目标类别权重之和，和 PyTorch 语义一致 |
| 并行方式 | 按行分配 AIV，不做跨 block 标量归约 |
| 同步 | 输入和输出槽分别使用独立 `tla.mutex` ownership 周期 |
| 执行环境 | 需要可用 CATLASS Python、CANN、torch_npu 和 Ascend 950 NPU |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
source <CANN_INSTALL_PATH>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
timeout -k 5 60s python \
    examples/python/tla_dsl/experimental/kernel/vec/nll_loss/nll_loss.py
```

脚本会执行以下精度检查：

1. logits + `ignore_index` 的 `cross_entropy(mean)`；
2. 重复调用与缓存快路径；
3. 默认 stream 与显式 stream 切换；
4. 非连续输入和 FP16 weight 的原地更新；
5. log-probability 的 `nll_loss(mean)`；
6. 含权重的 `cross_entropy(sum)`；
7. 含权重的 `cross_entropy(mean)`；
8. `reduction="none"` 的逐行输出。

在 Python 中直接调用：

```python
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.nll_loss.nll_loss import run

logits = torch.randn(128, 1024, device="npu:0")
target = torch.randint(0, 1024, (128,), device="npu:0")

print(run(logits, target, from_logits=True))
print(run(logits, target, from_logits=True, reduction="none").shape)
```

## 验证要求

- 修改 kernel 同步后，必须至少运行样例文件自带的全部精度用例。
- 单例执行必须使用 `timeout -k 5 60s`；超过 60 秒按失败处理，不得直接增大超时。
- 更换目标架构时必须同步修改 `--npu-arch`，当前样例只验证 Ascend 950 `3510`。

## 验证结果

- 尾块：`C=1..64` 全部 64 个宽度和 `C=65..128` 全部 64 个宽度均通过，覆盖
  4Byte~512Byte 行宽、非 32Byte 对齐、`ignore_index` 与 weighted mean；代表性
  `C=257,1025` 也通过。
- 性能：使用 `bench_nll_loss.py` 对 `torch.nn.functional.cross_entropy` 做同进程交替
  配对测试。脚本计时 `run(...)` 的完整调用，包含 host 校验、launch 和归约成本。

```bash
timeout -k 5 60s python \
    examples/python/tla_dsl/experimental/kernel/vec/nll_loss/bench_nll_loss.py \
    --device 0 --n 257 --c 1024
```
