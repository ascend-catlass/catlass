# MatmulAddChain Example Readme

## 功能说明

- 算子功能：完成链式矩阵乘加 `out = ((Q @ K) + X1) @ V + X2`，即先做一次矩阵乘并加上偏置矩阵 `X1`，再以结果为左操作数做第二次矩阵乘并加上 `X2`；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel，入口为 `matmul_add_chain` 模块的 `run` 函数。
- 计算公式：

  $$
  out = ((Q \times K) + X_1) \times V + X_2
  $$

  其中 $Q \in \mathbb{R}^{M \times K_1}$、$K \in \mathbb{R}^{K_1 \times N_1}$、$V \in \mathbb{R}^{N_1 \times N_2}$、$X_1 \in \mathbb{R}^{M \times N_1}$、$X_2 \in \mathbb{R}^{M \times N_2}$，输出 $out \in \mathbb{R}^{M \times N_2}$。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
out = matmul_add_chain.run(q, k, v, x1, x2)
```

`M` 为批行数，`K1` 为第一级内积维度，`N1` 为第一级输出宽度（即第二级内积维度），`N2` 为最终输出宽度。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[M, K1]` | `float32` / `float16` / `bfloat16` | 第一级左矩阵，行优先连续 |
| `k` | Input | `[K1, N1]` | 同 `q` | 第一级右矩阵，行优先连续 |
| `v` | Input | `[N1, N2]` | 同 `q` | 第二级右矩阵，行优先连续 |
| `x1` | Input | `[M, N1]` | 同 `q` | 第一级加项 |
| `x2` | Input | `[M, N2]` | 同 `q` | 第二级加项 |
| `out` | Output | `[M, N2]` | 同 `q` | 链式结果 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`K1`、`N1`、`N2` 为正整数；可非 tile 整数倍，尾块由 origin_shape 裁剪与补零处理（已实测 8~512、奇数 M、各维尾块） |
| dtype | 全部输入同一 dtype（`float32` / `float16` / `bfloat16`），输出与输入同 dtype；fp16/bf16 的窄化为 round-to-nearest-even |
| 布局 | 全部输入为 row-major (ND) 连续张量 |
| 设备 | 全部输入位于同一 NPU |
| 编译 | 每个 `(M, K1, N1, N2, dtype)` 组合编译一次并缓存（约 0.5s）；输出与 workspace 缓冲按 shape 复用 |

## 使用示例

导入样例模块后调用 `run`：

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/matmul:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from matmul_add_chain.matmul_add_chain import run

m, k1, n1, n2 = 64, 64, 64, 64
dtype = torch.bfloat16
q = torch.randn(m, k1, dtype=dtype, device="npu:0") * 0.1
k = torch.randn(k1, n1, dtype=dtype, device="npu:0") * 0.1
v = torch.randn(n1, n2, dtype=dtype, device="npu:0") * 0.1
x1 = torch.randn(m, n1, dtype=dtype, device="npu:0") * 0.1
x2 = torch.randn(m, n2, dtype=dtype, device="npu:0") * 0.1

out = run(q, k, v, x1, x2)
torch.npu.synchronize()

ref = ((q.float() @ k.float()).to(dtype) + x1).to(dtype)
ref = ((ref.float() @ v.float()).to(dtype) + x2).to(dtype)
print(out.shape, out.dtype, (out.float() - ref.float()).abs().max())
PY
```

执行结果如下，说明样例执行成功（误差在 bf16 舍入量级内）：

```text
torch.Size([64, 64]) torch.bfloat16 tensor(0., device='npu:0')
```
