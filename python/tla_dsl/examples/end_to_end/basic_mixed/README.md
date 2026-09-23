# 基础 Mixed（Cube + Vector）端到端示例

本目录下提供的系列样例演示 **CATLASS DSL** 下基础 CV（Cube + Vector）融合算子的计算过程。

## 功能说明

基础 CV 融合算子实现形如 $(m, k)$ 和 $(k, n)$ 的两矩阵乘法后与加和矩阵逐元素相加，输出形如 $(m, n)$，计算公式为：

$$
\begin{aligned}
D &= A \times B + C \\
D_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j} + C_{i,j}
\end{aligned}
$$

- [`basic_mixed_fixpipe_nz2dn`](basic_mixed_fixpipe_nz2dn.py) 实现基本矩阵乘功能，无后续加和操作。

与基础矩阵乘相比，CV 融合除 AIC 上的矩阵计算外，还包含 AIV 上的加法运算，二者之间存在两条数据通路：

- `L0C` → `UB` 数据通路：矩阵乘结果计算完成后启动 FIXPIPE 由 L0C 搬出到 UB (Unified Buffer)，可便于后续 Vector 运算；
- `UB` → `L1` 数据通路：矩阵 A 从 GM 加载到 UB 后，通过该通路搬运到 L1 上，可随路转换为 `zN` / `zNUnAlign` 排布。

## 代码组织

本目录组织结构如下所示：

```plain
./basic_mixed
├── basic_mixed.py
├── basic_mixed_fixpipe_nz2dn.py
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
| [**`basic_mixed_ub2l1.py`**](basic_mixed_ub2l1.py)                     | 启用`UB`→`L1` 通路的 CV 融合示例，AIV 将矩阵 A 从 GM 经 UB 搬运到 L1，不改动数据排布。                              |
| [**`basic_mixed_store_zN.py`**](basic_mixed_store_zN.py)               | 启用`UB`→`L1` 通路的 CV 融合示例，AIV 将矩阵 A 转为 `zN` 排布后再写入 L1。                                        |
| [**`basic_mixed_store_zNUnAlign.py`**](basic_mixed_store_zNUnAlign.py) | 同上，但使用`zNUnAlign` 排布，M 轴不对齐到分形大小。                                                                   |

## 约束说明

- 本目录下示例固定采用 `f32` 数据类型。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../docs/zh/quick_start.md)完成部署。

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
| `--m` / `--n` / `--k`     | `32`, `32`, `32`                                  | 矩阵乘加的问题大小                                                                   |
| `--layout-a` / `--layout-b` | `"row"` / `"row"`                                   | 左、右矩阵 A、B 的数据排布格式，可选`"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--block-num`                 | `-1`                      | 启用的核数，`-1` 表示自动探测可用核数（满核）。                                               |
| `--sentinel`                  | `-9.0`                                                | Kernel 启动前预先填入结果矩阵的值。                                                   |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 基础测试 (默认 m=32, n=32, k=32)
python examples/end_to_end/basic_mixed/basic_mixed.py


# 使用 Mutex 同步
python examples/end_to_end/basic_mixed/basic_mixed_mutex.py

# 启用 UB -> L1 通路
python examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py

# 启用 UB -> L1 通路并做 zN 排布转换
python examples/end_to_end/basic_mixed/basic_mixed_store_zN.py
python examples/end_to_end/basic_mixed/basic_mixed_store_zNUnAlign.py
```

执行测试后，预期输出：

```plain
--- mnk=(<m>, <n>, <k>) ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

- 上述 `<m>`， `<n>`， `<k>` 等为占位符，具体依赖外部参数或环境变量传入。

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
