# 基础 Mixed（Cube + Vector）端到端示例

本目录下提供的系列样例演示 **CATLASS Python** 下基础 CV（Cube + Vector）融合算子的计算过程。

## 功能说明

基础 CV 融合算子实现形如 $(m, k)$ 和 $(k, n)$ 的两矩阵乘法后与加和矩阵逐元素相加，输出形如 $(m, n)$，计算公式为：

$$
\begin{aligned}
D &= A \times B + C \\
D_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j} + C_{i,j}
\end{aligned}
$$

- [`basic_mixed_fixpipe_nz2dn`](basic_mixed_fixpipe_nz2dn.py) 实现基本矩阵乘功能，无后续加和操作。
- [`basic_mixed_fixpipe_per_tensor`](basic_mixed_fixpipe_per_tensor.py) 与 [`basic_mixed_fixpipe_per_channel`](basic_mixed_fixpipe_per_channel.py) 在 `L0C`→`UB` 通路上启用 FIXPIPE 量化，同样不含加和。

与基础矩阵乘相比，CV 融合除 AIC 上的矩阵计算外，还包含 AIV 上的加法运算，二者之间存在两条数据通路：

- `L0C` → `UB` 数据通路：矩阵乘结果计算完成后启动 FIXPIPE 由 L0C 搬出到 UB (Unified Buffer)，可便于后续 Vector 运算；
- `UB` → `L1` 数据通路：矩阵 A 从 GM 加载到 UB 后，通过该通路搬运到 L1 上，可随路转换为 `zN` / `zNUnAlign` 排布。

## 代码组织

本目录组织结构如下所示：

```plain
./basic_mixed
├── basic_mixed.py
├── basic_mixed_fixpipe_nz2dn.py
├── basic_mixed_fixpipe_per_channel.py
├── basic_mixed_fixpipe_per_tensor.py
├── basic_mixed_l0c2ub_b8_unitflag_pad.py
├── basic_mixed_mutex.py
├── basic_mixed_store_zN.py
├── basic_mixed_store_zNUnAlign.py
├── basic_mixed_ub2l1.py
└── README.md
```

各子文件所承载样例特性概述如下：

| 文件                                                                          | 概述                                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [**`basic_mixed.py`**](basic_mixed.py)                                 | 基础 CV 融合示例，CUBE 单元矩阵乘计算完成后经`L0C`→`UB` 通路搬移到 UB，AIV 上执行逐元素加法，是其他变体的基准版本。 |
| [**`basic_mixed_mutex.py`**](basic_mixed_mutex.py)                     | 使用`Mutex` 原语实现流水同步的 CV 融合示例。                                                                           |
| [**`basic_mixed_fixpipe_nz2dn.py`**](basic_mixed_fixpipe_nz2dn.py)     | 使用`L0C`→`UB` 通路定向搬运到 AIV0 上进行计算。                                                                     |
| [**`basic_mixed_fixpipe_per_tensor.py`**](basic_mixed_fixpipe_per_tensor.py) | `L0C`→`UB` 通路上启用 FIXPIPE 的 per-tensor 量化，整块 tile 共用一个标量 scale，按块编号奇偶分派 AIV。 |
| [**`basic_mixed_l0c2ub_b8_unitflag_pad.py`**](basic_mixed_l0c2ub_b8_unitflag_pad.py) | 演示 `unit_flag` + UB `i8`/`u8` 时如何把写出轴（Col→M / Row→N）上对齐到 32，避免 Fixpipe/mmad 分形计数不一致卡死。 |
| [**`basic_mixed_fixpipe_per_channel.py`**](basic_mixed_fixpipe_per_channel.py) | 同上，但每个输出通道一个 scale，dequant tensor 按块由 GM 搬入 L1。 |
| [**`basic_mixed_ub2l1.py`**](basic_mixed_ub2l1.py)                     | 启用`UB`→`L1` 通路的 CV 融合示例，AIV 将矩阵 A 从 GM 经 UB 搬运到 L1，不改动数据排布。                              |
| [**`basic_mixed_store_zN.py`**](basic_mixed_store_zN.py)               | 启用`UB`→`L1` 通路的 CV 融合示例，AIV 将矩阵 A 转为 `zN` 排布后再写入 L1。                                        |
| [**`basic_mixed_store_zNUnAlign.py`**](basic_mixed_store_zNUnAlign.py) | 同上，但使用`zNUnAlign` 排布，M 轴不对齐到分形大小。                                                                   |

## 约束说明

所有样例要求`M > 0`、`N > 0`、`K > 0`。下表是各脚本`run()`实际检查的shape范围；默认值不是额外的固定尺寸限制。

| 脚本 | M | N | K |
| --- | --- | --- | --- |
| `basic_mixed.py` | `M = 32` | `N = 32` | `1 <= K <= 32` |
| `basic_mixed_mutex.py` | `M = 32` | `N = 32` | `1 <= K <= 32` |
| `basic_mixed_fixpipe_nz2dn.py` | `M = 60` | `N = 100` | `1 <= K <= 64` |
| `basic_mixed_store_zN.py` | `2 <= M <= 64`，且M为偶数 | `1 <= N <= 64` | `1 <= K <= 256` |
| `basic_mixed_store_zNUnAlign.py` | `2 <= M <= 64`，且M为偶数 | `1 <= N <= 64` | `1 <= K <= 256` |
| `basic_mixed_ub2l1.py` | M为正偶数，且满足下述分形存储及容量限制 | 满足下述分形存储及容量限制 | 满足下述分形存储及容量限制 |
| `basic_mixed_fixpipe_per_tensor.py` | 正整数，支持跨tile计算 | 正整数，支持跨tile计算 | `1 <= K <= 64` |
| `basic_mixed_fixpipe_per_channel.py` | 正整数，支持跨tile计算 | 正整数，支持跨tile计算 | `1 <= K <= 64` |
| `basic_mixed_l0c2ub_b8_unitflag_pad.py` | `1 <= M <= 64` | `1 <= N <= 64` | `1 <= K <= 64` |

- `basic_mixed.py`和`basic_mixed_mutex.py`的Vector输出覆盖范围固定，K不分段；不能仅改变GM张量形状来扩大M/N/K。
- `basic_mixed_fixpipe_nz2dn.py`的输出视图和stride使用固定M/N，K由输入决定但须放入一个tile。
- 两个store示例通过两个AIV子块各处理`M // 2`行，奇数M会漏行；kernel只处理一个`64*64*256`的tile。M无需是16的倍数。
- 两个量化示例遍历M/N网格，K不切分。per-tensor的单字节输出还要求：RowMajor输出时`N % 32 == 0`，ColumnMajor输出时`M % 32 == 0`。
- b8 pad示例在ColumnMajor输出时将M补齐到32的倍数，在RowMajor输出时将N补齐到32的倍数；补齐后的cube frame须落入`64*64`的L0 tile。
- `basic_mixed.py`、`basic_mixed_mutex.py`和`basic_mixed_fixpipe_nz2dn.py`使用`f32`。UB→L1及两个store示例的类型选择见各脚本的`--dtype`；量化输出类型由各脚本的`--dtype-ab`、`--dtype-c`组合决定。
- 尺寸校验发生在设备初始化和张量分配之前；数据类型、布局和精度仍须满足对应通路的要求，shape校验不代表所有组合均已经过验证。

### ub2l1的分形存储与容量限制

本示例按逻辑元素数分配片上空间，不额外为分形尾块扩容。令`s`为输入元素的字节数，`C0 = 32 / s`，`R(x,a) = ceil(x/a)*a`，须满足：

- L1A/L0A：`R(M,16)*R(K,C0) <= M*K`。
- L1B：`R(K,16)*R(N,C0) <= K*N`。
- L0B：`R(K,C0)*R(N,16) <= K*N`。
- L0C：`R(M,16)*R(N,16) <= M*N`。

这些检查防止分形补齐后的物理占用超过实际分配空间。通过上述检查后，还须满足：

| 存储层 | 本示例使用的容量判据 |
| --- | --- |
| L1 | `R(M*K*s,512) + K*N*s <= 512*1024`字节 |
| L0A | `M*K*s <= 64*1024`字节 |
| L0B | `K*N*s <= 64*1024`字节 |
| L0C | `M*N*4 <= 256*1024`字节 |
| UB | `(M/2)*K*s <= 248*1024`字节 |

容量依据为本仓库`python/tla_dsl/catlass/base_dsl/arch.py`中Ascend 950的片上容量定义；UB扣除编译器默认保留的8 KiB。示例未传入释放保留空间的编译选项。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

```text
basic_mixed.py [-h] [--device DEVICE] [--m M] [--n N] [--k K]
               [--layout-a {row,col}] [--layout-b {row,col}]
               [--block-num BLOCK_NUM]
               [--sentinel SENTINEL]
```

上述命令行参数具体说明如下：

| 参数                            | 默认值                                                  | 说明                                                                                 |
| ------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--device`                    | `0`                                                   | 上板执行使用的 NPU 设备号。                                                          |
| `--m` / `--n` / `--k`     | `32`, `32`, `32`                                  | 问题尺寸，各脚本的允许范围见约束说明                                                                   |
| `--layout-a` / `--layout-b` | `"row"` / `"row"`                                   | 左、右矩阵 A、B 的数据排布格式，可选`"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--block-num`                 | `-1`                      | 启用的核数，`-1` 表示自动探测可用核数（满核）。                                               |
| `--sentinel`                  | `-9.0`                                                | Kernel 启动前预先填入结果矩阵的值。                                                   |

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 基础测试 (默认 m=32, n=32, k=32)
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py


# FIXPIPE NZ2DN
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_nz2dn.py

# 使用 Mutex 同步
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_mutex.py

# 启用 UB -> L1 通路
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_ub2l1.py

# 启用 UB -> L1 通路并做 zN 排布转换
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_store_zN.py
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_store_zNUnAlign.py

# FIXPIPE per-tensor 量化（可选 --layout-c col 走转置通路）
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_tensor.py

# FIXPIPE per-channel 量化
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_fixpipe_per_channel.py

# L0C -> UB b8 unit_flag
python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed_l0c2ub_b8_unitflag_pad.py
```

执行测试后，预期输出：

```plain
--- mnk=(<m>, <n>, <k>) ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

- 上述`<m>`、`<n>`、`<k>`为回显占位符，实际值由命令行参数决定，且须满足对应脚本的约束。

其中 `passed` 结果为 `True` 或 `False` 表明 NPU 计算结果与 golden 参考值精度校验是否通过；`cache_dir` 是指定的缓存目录，`cache_key` 是编译缓存的哈希值。

---

## 特性 Kernel 介绍

以下针对本目录下的特性 Kernel 做简略介绍。

### basic_mixed_mutex

**文件**：`basic_mixed_mutex.py`

功能与 `basic_mixed.py` 相同，使用 `Mutex` 原语替代 `set`/`wait` 实现 Cube 侧与 Vector 侧流水间的同步。以 GM 至 L1A 的数据搬运为例，关键代码如下：

```python
import catlass.tla as tla

# ...

mutex_l1a = tla.mutex(resource="l1a", id=0)

# ...
mutex_l1a.lock(pipe=tla.arch.MTE2)
tla.copy(l1_a, gm_a)
mutex_l1a.unlock(pipe=tla.arch.MTE2)
```

## basic_mixed_fixpipe_nz2dn

**文件**：`basic_mixed_fixpipe_nz2dn.py`

使用 `L0C`→`UB` 通路的基础矩阵乘示例。矩阵乘结果通过 `NO_SPLIT_VEC_0` 模式定向搬运到 AIV0，随后以列优先（ColumnMajor）排布写回 GM，关键代码如下：

```python
import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode

# ...
ub_c = tla.make_tensor_like(ub_c_ptr, l0_c, tla.arch.ColumnMajor)
tla.copy(ub_c, l0_c, params=CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0))
```

## basic_mixed_fixpipe_per_tensor

**文件**：`basic_mixed_fixpipe_per_tensor.py`

在 `L0C`→`UB` 通路上启用 FIXPIPE 的 per-tensor 量化：一个标量 `quant_scale` 在搬出 L0C 时作用于整块 tile。kernel 遍历 tile 网格（K 轴不切分），并按块的线性编号奇偶将每个任务块交给一个 AIV：偶数块归 AIV0、奇数块归 AIV1，因此 FIXPIPE 需按奇偶分别指定 `NO_SPLIT_VEC_0` / `NO_SPLIT_VEC_1`：

```python
import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode, QuantMode

# ...
if block_linear % 2 == 0:
    tla.copy(
        ub_c,
        l0_c,
        params=CopyL0C2DstParams(
            unit_flag=0b11,
            l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0,
            relu_enable=relu_enable,
            quant_mode=QuantMode.PER_TENSOR,
            quant_scale=quant_scale,
        ),
    )
```

`quant_scale` 是运行时标量（普通 kernel 参数，不是 `tla.Constexpr`），因此一份编译产物可服务任意 scale。`--layout-c col` 时同一次搬运经转置（nz2dn）FIXPIPE 写出，量化与转置在一趟内完成。

执行后输出形如 `--- mnk=(128,128,64) layout=row/row/row dtype=f32/f32 acc=f32 dst=f16 relu=False quant_scale=<scale> ---`，随后为 `passed=True`。

## basic_mixed_fixpipe_per_channel

**文件**：`basic_mixed_fixpipe_per_channel.py`

与 `basic_mixed_fixpipe_per_tensor` 相同，但 scale 是逐输出通道的张量而非标量。每个通道的 f32 scale 打包成一个 `uint64` 字，作为 FIXPIPE 的 dequant tensor 由 GM 搬到 L1：

```python
import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode, QuantMode

# ...
l1_quant = tla.make_tensor(
    l1_quant_ptr,
    tla.make_layout(
        tla.make_shape(gm_c.origin_shape[1]),
        tla.make_stride(1),
        layoutTag=tla.arch.VectorLayout,
    ),
)
tla.copy(l1_quant, gm_quant)
# ...
tla.copy(
    ub_c,
    l0_c,
    params=CopyL0C2DstParams(
        unit_flag=0b11,
        l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0,
        relu_enable=relu_enable,
        quant_mode=QuantMode.PER_CHANNEL,
        quant_tensor=l1_quant,
    ),
)
```

每个任务块覆盖的列区间不同，因此 dequant tensor 需按块重新搬运一次，并用一对 flag（`quant_full` / `quant_free`）与 FIXPIPE 同步。`uint64` 只在 L1 上的 rank-1 `VectorLayout` 一种形态被接受，且只能通过 `tla.make_tensor` 构造。

scale 由 host 侧随机生成，没有对应命令行参数；输出行因此不含 `quant_scale` 字段。

## basic_mixed_ub2l1

**文件**：`basic_mixed_ub2l1.py`

启用 `UB`→`L1` 通路的 CV 融合示例。AIV 侧将矩阵 A 从 GM 加载到 UB，再通过 `UB`→`L1` 通路搬运到 L1A（不改动数据排布），并通过 `cross_core` flag 与 AIC 侧同步：

```python
import catlass.tla as tla

# ...
tla.copy(ub_a, gm_a)                                   # GM -> UB
tla.cross_core_wait_flag(ub2l1_ready, tla.arch.MTE3)
tla.copy(l1_a, ub_a)                                   # UB -> L1
tla.cross_core_set_flag(ub2l1_done, tla.arch.MTE3)
```

## basic_mixed_store_zN

**文件**：`basic_mixed_store_zN.py`

在 `basic_mixed_ub2l1` 基础上，AIV 侧先将 RowMajor 数据重排为 `zN` 排布再写入 L1。数据重排通过 `tla.vec.func(mode="simd")` 逐行逐列执行，以 `BlockStore(block_stride=...)` 随路完成排布转换：

```python
import catlass.tla as tla
from catlass.params import BlockStoreParams

# ...
ub_a_zN = tla.make_tensor_like(ub_a_zN_ptr, ub_a, tla.arch.zN)
# ...
with tla.vec.func(mode="simd"):
    a_zN_chunk.store(a_chunk.load(), params=BlockStoreParams(block_stride=block_stride))
```

## basic_mixed_store_zNUnAlign

**文件**：`basic_mixed_store_zNUnAlign.py`

与 `basic_mixed_store_zN` 类似，但使用 `zNUnAlign` 排布，M 轴不对齐到分形大小，搬运时的 stride 是运行时变量，适用于非对齐场景。使用该排布：

```python
import catlass.tla as tla

# 创建 zNUnAlign 排布的目的 tla.Tensor
# ub_a_tile 为排布前源 tla.Tensor
ub_a_zN_full = tla.make_tensor_like(ub_a_zN_ptr, ub_a_tile, tla.arch.zNUnAlign)

# 通过 tla.tile_view 获取有效数据的 tla.Tensor 表示
ub_a_zN = tla.tile_view(
   ub_a_zN_full,
   tla.make_shape(gm_a.origin_shape[0], gm_a.origin_shape[1]),
   tla.make_coord(0, 0),
)
```
