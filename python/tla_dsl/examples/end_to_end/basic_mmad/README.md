# 基础 MMAD 端到端示例

本目录下提供的系列样例演示 **CATLASS DSL** 下基础矩阵乘的计算。

## 功能说明

基础矩阵乘算子实现形如$(m, k)$和$(k, n)$的两矩阵的乘法运算，输出形如$(m, n)$，计算公式为：

$$
\begin{aligned}
C &= A \times B \\
C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
\end{aligned}
$$

## 代码组织

本目录组织结构及文件概述如下：

```plain
./basic_mmad
├── basic_matmul_atomic_add.py
├── basic_matmul_auto_sync.py
├── basic_matmul_mutex.py
├── basic_matmul_mutex_with.py
├── basic_matmul.py
├── basic_mmad_ptr.py
└── README.md
```

| 文件                                                                  | 概述                                                                                                                      |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [**`basic_matmul.py`**](basic_matmul.py)                       | 基础矩阵乘示例，支持动态场景，含双缓冲优化，是其他变体的基准版本。                                                        |
| [**`basic_matmul_atomic_add.py`**](basic_matmul_atomic_add.py) | 启用原子加的矩阵乘示例，每个 tile 块计算后立刻搬移，在 GM 上进行累加，`DTYPE_C` 固定为累加类型（即 `tla.Float32`）。  |
| [**`basic_matmul_mutex.py`**](basic_matmul_mutex.py)           | 使用`Mutex` 原语的矩阵乘示例，不同于`set`/`wait` 原语， `Mutex` 原语同步不再区分源和目的指令。                    |
| [**`basic_matmul_mutex_with.py`**](basic_matmul_mutex_with.py) | 同样是利用`Mutex` 原语的矩阵乘示例，使用 `with tla.mutex_guard(...)` 表达，替换 `mutex.lock/unlock` 的做法。        |
| [**`basic_matmul_auto_sync.py`**](basic_matmul_auto_sync.py)   | 使用`@tla.kernel(auto_sync="v0")` 自动插入核内 mutex 的独立端到端用例，覆盖 MMAD/FIX unit flag 协议且无需手写本地同步。 |
| [**`basic_mmad_ptr.py`**](basic_mmad_ptr.py)                   | 矩阵乘最小集示例，显式构造`tla.Tensor` 供 kernel 计算，仅支持静态场景。                                                 |

## 约束说明

- 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
| ----------- | ----------- | ----------- |
| f16         | f16         | f32         |
| f16         | f16         | f16         |
| bf16        | bf16        | f32         |
| bf16        | bf16        | bf16        |
| f32         | f32         | f32         |

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../docs/zh/quick_start.md)完成部署。

### 命令行参数

```text
basic_matmul.py [-h] [--device DEVICE] [--m M] [--n N] [--k K]
                [--layout-a {row,col}] [--layout-b {row,col}]
                [--dtype-a {f16,bf16,f32}]
                [--dtype-b {f16,bf16,f32}]
                [--dtype-c {f16,bf16,f32}]
                [--relu-enable {0,1}]
                [--block-num BLOCK_NUM]
```

上述命令行参数具体说明如下：

| 参数                                          | 默认值                                                  | 说明                                                                                       |
| --------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `--device`                                  | `0`                                                   | 上板执行使用的 NPU 设备号。                                                                |
| `--m`                                       | `256`                                                 | 矩阵乘左矩阵 A 的行数                                                                      |
| `--n`                                       | `512`                                                 | 矩阵乘右矩阵 B 的列数                                                                      |
| `--k`                                       | `1024`                                                | 矩阵乘累加轴的大小                                                                         |
| `--layout-a` / `--layout-b`               | `"row"` / `"row"`                                   | 左、右矩阵 A、B 的数据排布格式，可选`"row"` 或 `"col"`，表示行优先或列优先布局。       |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f32"`                       | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围包括`"f16"`, `"bf16"` 和 `"f32"` 。 |
| `--relu-enable`                             | `0`                                         | fixpipe是否开启随路relu                                                                       |
| `--block-num`                               | `-1`                                        | 启用的核数，`-1` 表示自动探测可用核数（满核）。                                               |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 基础测试 (默认 m=256, n=512, k=1024)
python examples/end_to_end/basic_mmad/basic_matmul.py

# 指定 NPU ID 以及核数 (1卡1核)
python examples/end_to_end/basic_mmad/basic_matmul.py --block-num 1 --device 1

# 指定输入矩阵的尺寸、排布以及数据类型
python examples/end_to_end/basic_mmad/basic_matmul.py \
  --m 256 --n 512 --k 128 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

执行测试后，预期输出：

```plain
--- mnk=(<m>, <n>, <k>) layout=<layout_a>/<layout_b> dtype=<dtype_a>/<dtype_b>/<dtype_c> ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。

---

## 特性 Kernel 介绍

以下针对本目录下的特性Kernel做简略介绍。

## basic_matmul_atomic_add

**文件**：`basic_matmul_atomic_add.py`

使用 **原子加（Atomic Add）** 将K 维度的 Tile 分块在 GM 上进行累加，通过Cube计算单元每个分块计算完毕后立即通过 FIX 流水写回 GM。其中原子加特性关键代码如下：

```python
import catlass.tla as tla

# ...

# is_first_l1_tile 判断是否为首块累加的K Tile ( `k_l1 == 0` )
if not is_first_l1_tile:
   # 非首个 Tile 块启用原子加
   tla.copy(
      gm_c_by_core,
      l0_c,
      tla.params.CopyL0C2DstParams(
            unit_flag=0b11,
            atomic_mode=tla.params.AtomicMode.ADD
      )
   )
else:
   # 第一个 Tile 块搬出直接覆写 GM C
   tla.copy(
      gm_c_by_core,
      l0_c,
      tla.params.CopyL0C2DstParams(unit_flag=0b11)
   )
```

启用原子加的操作在`tla.copy(...)`中通过向 `CopyL0C2DstParams` 参数中传入 `atomic_mode` 实现，目前支持的模式包括 `AtomicMode.ADD` 和 `AtomicMode.NONE` （默认行为，无原子操作）。

## basic_matmul_mutex

**文件**：`basic_matmul_mutex.py`

`Mutex` 原语是一种互斥锁，是面向数据层面的依赖，较 `set` / `wait` 原语可以更简洁地描述同步关系。

可以使用显式 Mutex 锁/解锁 （`mutex.lock(pipe)` / `mutex.unlock(pipe)`）操作，当某条流水线申请到对某个`mutex`资源的 `lock` 操作后，该锁供其所有，阻塞其他申请该 `mutex` 资源的流水，直至其释放（ `unlock` 操作），`lock` / `unlock` 操作必须配对。

以 GM 向 L1A 的数据搬运操作为例，示例如下：

```python
import catlass.tla as tla

# ...

# mutex 资源申请
mutex_l1a = tla.mutex(resource="l1a", id=0)

# ...

# 启动 MTE2 源流水的数据搬运 (GM->L1A)
mutex_l1a.lock(pipe=tla.arch.MTE2)
tla.copy(l1_a, gm_a_by_l1)
mutex_l1a.unlock(pipe=tla.arch.MTE2)

# ...

# mutext_l1a 释放后（L1A数据准备完成）
# 可以启动 MTE1 源流水的数据搬运 (L1A->L0A)
mutex_l1a.lock(pipe=tla.arch.MTE1)
mutex_l0a.lock(pipe=tla.arch.MTE1)
tla.copy(l0_a, l1_a_by_l0)
mutex_l0a.unlock(pipe=tla.arch.MTE1)
mutex_l1a.unlock(pipe=tla.arch.MTE1)
```

---

## basic_matmul_mutex_with

**文件**：`basic_matmul_mutex_with.py`

使用 `with tla.mutex_guard(...)` **上下文管理器**实现 `Mutex` 同步，省去 `mutex.lock(pipe)` 和 `mutex.unlock(pipe)` 的手动管理。

仍以GM 至 L1A 的数据搬运过程为例，相应写法是：

```python
import catlass.tla as tla

# ...
with tla.mutex_guard(mutex_l1a):
   tla.copy(l1_a, gm_a_by_l1)
```

这一操作与手动 `lock` / `unlock` 操作等价，由 mutex 串行化。

---

## basic_matmul_auto_sync

**文件**：`basic_matmul_auto_sync.py`

在装饰器 `@tla.kernel` 传入 `auto_sync` 参数（目前支持 `auto_sync="v0"` ），可自动处理多级流水线间的同步，覆盖 MMAD/FIX unit flag 协议且无需手写本地同步，示例如下：

```python
@tla.kernel(auto_sync="v0")
def kernel_func(...):
   # ...
```
