# MatmulExpScaleUb Example Readme

## 功能说明

- 算子功能：对 `x @ wᵀ` 的 fp32 累加结果直接做逐行 epilogue（减去行标量、`exp`、乘以行标量），在同一个 kernel 内把结果窄化为 `bfloat16` 写回 GM，不落 fp32 中间结果；基于 CATLASS DSL 实现的 Cube/Vector mixed kernel，单 `@tla.kernel`、单次 launch，入口为 `matmul_exp_scale_ub` 模块的 `run` 函数。
- 计算公式：

  $$
  Y[m, n] = \mathrm{bf16\_rne}\bigl(\exp\bigl(\mathrm{float32}(x[m, :] \cdot w[n, :]) - row\_logsumexp[m]\bigr) \times row\_scale[m]\bigr),
  \qquad 0 \le m < M,\; 0 \le n < V
  $$

  其中 $x$ 为 `[M, H]`、$w$ 为 `[V, H]`，分别对应入参 `x` 与 `weight`；矩阵乘在 L0C 中以 fp32 累加，epilogue 全程在 fp32 下进行，只有最终结果做一次 bf16 舍入（向偶舍入）。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = matmul_exp_scale_ub.run(x, weight, row_logsumexp, row_scale)
```

`M` 为行数，`H` 为内积维度，`V` 为输出列数。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ---- |
| `x` | Input | `[M, H]` | bfloat16 | 左操作数；`H` 必须为 128 的整数倍 |
| `weight` | Input | `[V, H]` | bfloat16 | 右操作数，按行存放，kernel 以其转置视图 `[H, V]` 作为 B；`V` 为任意正整数，`V % 16 != 0` 时 `run` 内部把行补零到 16 的整数倍（见约束说明） |
| `row_logsumexp` | Input | `[M]` | float32 | 逐行减去的标量 |
| `row_scale` | Input | `[M]` | float32 | 逐行相乘的标量 |
| `y` | Output | `[M, V]` | bfloat16 | 输出；是内部 `[M_pad, V]` 缓冲区的 `[0:M]` 视图（`M` 已对齐时 `M_pad == M`） |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `M` 任意；`M % 128 != 0` 时 host 侧补齐到 `M_pad`（`x` 补零、`row_logsumexp` 补 0、`row_scale` 补 1），`M % 128 == 0` 时直接绑定输入，不新建、不拷贝（仅输入本身不连续时 `contiguous()` 会拷贝）；`H` 为 128 的正整数倍，不满足时抛出 `ValueError`；`V` 为任意正整数 |
| `V` 方向尾块 | `V % 256 != 0` 时最后一列 tile 只有 `V_real = V - (⌈V / 256⌉ - 1) * 256` 个有效列（`1 ≤ V_real ≤ 255`）。kernel 把该 tile 在 Cube 侧的 N 维取 `N_ext = ⌈V_real / 16⌉ * 16`（MMAD 与 L0B 的分形数），fixpipe 只搬 `⌈V_real / 8⌉ * 8` 列，vector 与写回只处理 `V_real` 列；`y` 按 `[M_pad, V]` 分配，不补列 |
| `weight` 行补齐 | `V % 16 != 0` 时 `run` 先把 `weight` 补零到 `⌈V / 16⌉ * 16` 行再绑定，使尾块 B tile `[128, N_ext]` 完整落在缓冲区内（见「结构与载荷」第 3 条）；`V % 16 == 0` 时不因尾块额外拷贝 `weight`（仅 `weight` 本身不连续时 `contiguous()` 会拷贝）。补零行只参与 kernel 不写出的尾列计算 |
| dtype | `x`、`weight` 必须为 `bfloat16`；`row_logsumexp`、`row_scale` 必须为 `float32`；输出固定 `bfloat16` |
| 输入布局 | 不要求连续，内部按需 `contiguous()`；`weight` 不需要在 host 侧转置 |
| 设备 | 四个输入必须位于同一 NPU；`run` 会先把当前设备切到该 NPU 再分配与 launch |
| 补齐行 | `M_pad` 中 `[M, M_pad)` 的行按 `exp(0 - 0) * 1` 计算得到 1.0，被返回视图裁掉，调用方不可见；`M % 128 == 0` 时不存在该段 |
| 数值 | 矩阵乘 fp32 累加；`exp` 与缩放在 fp32 下计算；最终 `CAST_RINT` 舍入到 bf16。与“fp32 logits + fp32 exp 再转 bf16”的参考值逐元素相差不超过 1 个 bf16 ulp |
| 编译 | 每个 `(device, M_pad, H, V)` 组合编译一个 kernel，产物按该键缓存 |
| 不覆盖 | 整行归约 / softmax 归一化、列主序 A、多阶段单 launch |

## 代码组织

kernel 的 tiling 为 `BM/BN/BK/SK = 128/256/128/32`，每个任务是一个 `[BM, N_ext]` 输出 tile。任务列表按列方向分两趟（pass）：第 0 趟是所有满列 tile（列索引 `nt < ⌈V / BN⌉ - 1`，`N_ext = BN`），第 1 趟是尾列 tile（`nt = ⌈V / BN⌉ - 1`，每个行 tile 一个，`N_ext = ⌈V_real / 16⌉ * 16`）；`V % 256 == 0` 时第 1 趟的 tile 同样是满 tile。两趟由 `tla.range_constexpr` 在前端展开，每趟的 tile 形状都是编译期常量。cube 区与 vector 区用同一组 `block_idx / block_num`（`block_num` 为 cube 核数）按相同顺序遍历两趟，`sub_block_idx` 区分 `SPLIT_M` 之后各 AIV 负责的半块。每个 tile 依次经过：

1. `cube`：MTE2 把 A 的 `[BM, BK]`（RowMajor → zN）与 B 的 `[BK, N_ext]`（weight 的 ColumnMajor `[H, V]` 视图 → nZ）搬进 L1 双缓冲，MTE1 按 `SK` 分块搬进 L0A/L0B 双缓冲，MMAD 在 L0C 中 fp32 累加；最后一次 MMAD 以 `unit_flag=0b11` 收尾。B tile 先按 `[BK, BN]` 定位到第 `nt` 列 tile 的起始列，再从中切出 `[BK, N_ext]`（`tile_view` 的坐标按 tile 形状放大，直接以 `[BK, N_ext]` 取第 `nt` 个 tile 会落到第 `nt * N_ext` 列）。
2. `fixpipe`：`CopyL0C2DstParams(unit_flag=3, l0c2ub_mode=SPLIT_M)` 把 `[BM, N_ext]` tile 按行对半落到同一 core 两个 AIV 的 UB。目的视图形状为 `[BM, N_ext]`、`origin_shape` 为 `(BM, V_real)`，fixpipe 的 `nSize` 随之为 `⌈V_real / 8⌉ * 8`。
3. `vector`：SIMT 一线程一行把 `row_logsumexp`、`row_scale` 读进 UB；SIMD 用 `NormalLoadParams(load_dist=DIST_BRC_B32)` 把行标量广播到 64 条 lane，逐 `[1, 64]` 片段做减法、`exp`、乘法，再用 `CastParams(RegSlot.ZERO, SatMode.NOSAT, RoundMode.CAST_RINT)` 与 `NormalStoreParams(store_dist=StoreDist.DIST_PACK_B32)` 窄化成 bf16 写入 UB。每行只遍历 `⌈V_real / 64⌉` 个片段，片段掩码由 `tla.update_mask` 从循环携带的剩余列数得到：满片段全真，最后一个片段只覆盖有效 lane。bf16 输出 tile 的行距按 64 个 bf16（128 Byte）补齐，最后一个片段中越过 `V_real` 的 lane 落在这段填充里，不写回 GM（存储掩码保持全真，原因见「结构与载荷」第 4 条）。
4. `MTE3`：把 `[BM/2, V_real]` 的 bf16 半块按行距 `V` 写回 GM（满 tile 时 `V_real = BN`）。

cube 与 vector 之间靠一个 `exchange` cross flag 交接：vector 区先 `set`，fixpipe `wait` 后写 UB 再 `set`，vector `wait` 后消费，写回 GM 并 `pipe_barrier` 之后再 `set` 归还，因此 fp32 暂存半块不会被下一次 fixpipe 提前覆盖；`ub_out` mutex 负责 bf16 半块在 VECTOR 与 MTE3 之间的交接。

### 结构与载荷

以上流水线、交接、`SPLIT_M` 与 bf16 落盘是本样例的**结构**，可以原样复用；vector 区的数学段（减行标量 → `exp` → 乘行标量）是**载荷**，可整体替换为其它逐行变换，但要保留“SIMT 读行标量 + SIMD 广播消费”的形态，以继续覆盖行循环、广播与窄化路径。

结构依赖四条实测结论，任一条不满足都会挂死或写坏数据，源码注释里均已标出：

- `tla.tile_view` 的 `coord` 是 **tile 索引**，前端会乘以 tile shape 得到元素偏移；若误传元素行号，偏移会被放大 tile 行数倍，表现为 errcode 82 / 507015 或挂死。GM 输出半块改用 `tla.make_tensor(ptr + row * stride + col, layout(...))`，偏移是元素粒度，不存在这一换算；尾块 B tile 则先按 `[BK, BN]` 定位再切窄。
- `origin_shape` **裁剪 DMA**：传输长度按 `origin_shape` 计算。Ascend 950PR、ascend-catlass-dsl 2.2.1 哨兵实验：shape 64 行、origin 32 行的 UB → GM 与 GM → UB 拷贝都只搬 32 行，第 33～64 行保持哨兵值；cube 侧 GM → L1 的 ND2NZ 也只读 origin 内的列，origin 之外的 GM 行未被读取。被裁掉的目的区域不会写入数据，其内容不确定（实验中 L1 尾列读回为 0）。因此尾块不依赖裁剪：有效宽度全部显式给出（Cube 侧 `N_ext`、fixpipe 与 vector 侧 `V_real`），B tile 由 host 补零保证完整在界内。
- A tile 与 C tile 的 `origin_shape[0]` 对应的 16 行分形数不一致时（`ceil(acc_M / 16) > ceil(A_M / 16)`），MMAD 会忙等、AI Core 利用率持续 100%、无错误码；对 B 的 N 方向同样成立。A 补零到整 `BM` 块即可消除；B 必须从 weight 的逻辑 `[H, V]` 视图切片，保证 `[BK, N_ext]` tile 不会越过 `H`，也不会越过 weight 缓冲区（见「`weight` 行补齐」）。
- `DIST_PACK_B32` 存储按 B32 寄存器槽压紧低半部 bf16，**一次 `[1, 64]` fp32 片段的存储需要 128 位掩码全真才会写出全部 64 个值**：只置低位 64 lane 时实际只写 32 个，其余位置保留陈旧字节（表现为每个 256 列 tile 的最后 32 列结果不确定）。因此 bf16 输出用 `tla.create_mask(pattern=tla.mask.ALL)` 全真掩码写出，越过 `V_real` 的 lane 靠 `ub_y` 行距补齐到整 64 个 bf16（128 Byte）来隔离——这些 lane 落在填充区，MTE3 只搬 `V_real` 列，永远读不到；fp32 片段上的部分掩码只用于计算，不用于存储。

## 使用示例

以下命令在 CATLASS 仓库根目录执行，前提是已安装 `torch_npu` 与 CATLASS DSL 的 Python 包（`catlass.tla` 可导入）并加载了 CANN toolkit 的环境变量：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/matmul/matmul_exp_scale_ub:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from matmul_exp_scale_ub import run

torch.npu.set_device(0)
torch.manual_seed(0)
m, h, v = 1000, 256, 1100
x = torch.randn(m, h, dtype=torch.bfloat16, device="npu:0")
weight = torch.randn(v, h, dtype=torch.bfloat16, device="npu:0") * 0.05
row_logsumexp = torch.rand(m, dtype=torch.float32, device="npu:0") * 5.0
row_scale = torch.randn(m, dtype=torch.float32, device="npu:0")

y = run(x, weight, row_logsumexp, row_scale)
torch.npu.synchronize()
print(y.shape, y.dtype)

# kernel 保留 fp32 的矩阵乘结果；bf16 输入的 x @ weight.T 会先把结果舍入到 bf16
logits = x.float() @ weight.float().T
ref = (logits - row_logsumexp[:, None]).exp().mul(row_scale[:, None]).to(torch.bfloat16)
diff = (y.float() - ref.float()).abs()
ulp = torch.pow(2.0, torch.floor(torch.log2(ref.float().abs().clamp_min(2**-126))) - 7)
print("max_abs", diff.max().item(), "max_ulp", (diff / ulp).max().item())
PY
```

`v = 1100` 时最后一列 tile 是 `V_real = 76` 的尾块；`v = 1024` 则没有尾块。执行结果如下，说明样例执行成功（`max_ulp` 是逐元素按 bf16 ulp 归一化的最大误差，两种 `v` 都为 1.0；Ascend 950PR 实测）：

```text
torch.Size([1000, 1100]) torch.bfloat16
max_abs 0.00390625 max_ulp 1.0
torch.Size([1000, 1024]) torch.bfloat16
max_abs 0.00390625 max_ulp 1.0
```

## 验证方法

- 与 torch 对照：参考值取 `(x.float() @ weight.float().T - row_logsumexp[:, None]).exp().mul(row_scale[:, None]).to(torch.bfloat16)`。kernel 保留 fp32 的矩阵乘结果，而 torch 对 bf16 输入的 `x @ weight.T` 会先把 logits 舍入到 bf16，用后者做参考会把参考值自身的舍入算进误差（在 Ascend 950PR 上实测，该参考值与 fp64 真值的偏差可达 1.3%）；逐元素比较时应不超过 1 个 bf16 ulp。
- 补齐行检查：`M % 128 != 0` 时返回值是 `[M_pad, V]` 缓冲区的 `[0:M]` 视图，`y._base[M:M_pad]` 应全为 `1.0`；`M % 128 == 0` 时 `M_pad == M`，没有补齐行，也不新建输入副本（`x`、`row_logsumexp`、`row_scale` 已连续时直接绑定，可用 `torch.npu.memory_allocated()` 在调用前后的差值确认只新分配了输出 `[M, V]`）。
- 专项用例：`M=128`（单 tile）、`M=1024`（M 已对齐，走零拷贝绑定）、`M=1000`（M 尾块，`M_pad=1024`）、`M=1`、`H=256`（K 方向两个 tile）、`V=1024`（无 V 尾块）、`V=1100`（`V_real=76`）、`V=1040`（`V_real=16`）、`V=1025`、`V=1027`、`V=1032`（`V_real` 分别为 1、3、8）、`V=255`、`V=300`（单列 tile 且带尾块，`V_real` 分别为 255、44）、`M=1000`、`H=256`、`V=1100`（M 与 V 双尾块）、非连续的 `x` / `weight`、`H` 不满足约束时的 `ValueError`。Ascend 950PR 上实测上述用例逐元素误差不超过 1 个 bf16 ulp，与参考值逐位一致的元素占比不低于 99.99%。
- 确定性：同输入连续执行两次，输出逐位一致。
