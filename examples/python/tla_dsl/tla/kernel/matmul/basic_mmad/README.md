# 基础 MMAD 端到端示例

本目录下提供的系列样例演示 **CATLASS Python** 下基础矩阵乘的计算。

## 功能说明

- 算子功能：完成基础矩阵乘计算
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵。

- 支持产品型号：Ascend 950PR/950DT 系列产品

## 代码组织

本目录组织结构及文件概述如下：

```plain
./basic_mmad
├── basic_matmul.py
├── basic_matmul_atomic_add.py
├── basic_matmul_auto_sync.py
├── basic_matmul_bias.py
├── basic_matmul_fixpipe_per_channel.py
├── basic_matmul_fixpipe_per_tensor.py
├── basic_matmul_l0c2l1.py
├── basic_matmul_mutex.py
├── basic_matmul_mutex_with.py
├── basic_mmad_ptr.py
└── README.md
```

| 文件 | 概述 |
| --- | --- |
| [**`basic_matmul.py`**](basic_matmul.py) | 基础矩阵乘示例，支持动态场景，含双缓冲优化，是其他变体的基准版本。 |
| [**`basic_matmul_atomic_add.py`**](basic_matmul_atomic_add.py) | 启用原子加的矩阵乘示例，每个 tile 块计算后立刻搬移，在 GM 上进行累加，`DTYPE_C` 固定为累加类型（即 `tla.Float32`）。 |
| [**`basic_matmul_auto_sync.py`**](basic_matmul_auto_sync.py) | 使用 `@tla.kernel(auto_sync="v0")` 自动插入核内 mutex 的独立端到端用例，覆盖 MMAD/FIX unit flag 协议，无需手动插入同步。 |
| [**`basic_matmul_bias.py`**](basic_matmul_bias.py) | 带偏置项的矩阵乘示例，bias 经 `L1 → biasbuf` 通道加载，bias 类型与 Cube 累加类型一致，为 i32 或 f32。 |
| [**`basic_matmul_fixpipe_per_channel.py`**](basic_matmul_fixpipe_per_channel.py) | 带随路量化的矩阵乘（`per-channel`）量化示例，按输出通道应用 `u64` 量化参数，支持随路 ReLU。 |
| [**`basic_matmul_fixpipe_per_tensor.py`**](basic_matmul_fixpipe_per_tensor.py) | 带随路量化的矩阵乘（`per-tensor`）量化示例，使用标量 `quant_scale`/`quant_offset`，支持随路 ReLU。 |
| [**`basic_matmul_l0c2l1.py`**](basic_matmul_l0c2l1.py) | L0C→L1 回搬示例，连续计算 `E = (A × B) × D`，固定 f32 静态场景。 |
| [**`basic_matmul_mutex.py`**](basic_matmul_mutex.py) | 使用 `Mutex` 原语的矩阵乘示例，不同于 `set`/`wait` 原语，`Mutex` 原语同步不再区分源和目的指令。 |
| [**`basic_matmul_mutex_with.py`**](basic_matmul_mutex_with.py) | 同样为利用 `Mutex` 原语的矩阵乘示例，使用 `with tla.mutex_guard(...)` 表达。 |
| [**`basic_mmad_ptr.py`**](basic_mmad_ptr.py) | 矩阵乘最小集示例，显式构造 `tla.Tensor` 供 kernel 计算，仅支持静态场景。 |

## 样例参数说明

以下为本目录下演示的基础矩阵乘示例 kernel 所需要的参数：

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `a_tensor`        | Input  | `(m,k)` | `f16/bf16/f32/i8/f8e4m3fn/f8e5m2` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`  |
| `b_tensor`        | Input  | `(k,n)` | `f16/bf16/f32/i8/f8e4m3fn/f8e5m2` | 右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，支持的数据组合类型参考[使用范围说明](#使用范围说明) |
| `c_tensor`        | Output | `(m,n)` | `f16/bf16/f32/i32` | 矩阵乘结果，layout支持`RowMajor`（默认）和`ColumnMajor` |


针对 [`basic_matmul_bias`](basic_matmul_bias.py) ，带偏置参数：

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `bias_tensor`        | Input  | `(n,)` | `f32`/`i32` | 偏置项，layout固定为`VectorLayout`  |

针对 [`basic_matmul_fixpipe_per_channel`](basic_matmul_fixpipe_per_channel.py) 及 [`basic_matmul_fixpipe_per_tensor`](basic_matmul_fixpipe_per_tensor.py) ，带量化参数：

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `quant_tensor`        | Input  | `(n,)` | `u64` | Per-channel 量化系数（`tla.UInt64` 类型视图） |
| `quant_scale`         | Input(Scalar) | `(1)` | `f32` | Per-tensor 量化系数 |
| `quant_offset`         | Input(Scalar) | `(1)` | `i32` | Per-tensor 量化偏置 |

部分样例包含下述参数，做下述说明（具体也请通过查询[命令行参数](#命令行参数)确认）：

| 参数       | 属性 |shape|dtype| 说明                                                      |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `tiling`  | Attr | - | - | Tiling参数（包含L1/L0 的TileShape） |
| `swizzle` | Attr | - | - | Swizzle参数（包含方向与偏移） |
| `acc_is_int`  | Attr | - | - | 指示矩阵乘计算是否为 i8@i8->i32 |
| `hf32_mode`  | Attr | - | - | 矩阵计算启用Hifloat32的模式 |
| `relu_enable`  | Attr | - | - | 输出是否启用ReLu |


## 使用范围说明

`basic_matmul.py` 支持并通过精度校验的数据组合如下：

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
| ----------- | ----------- | ------------  |
| f16         | f16         | f16/bf16/f32  |
| bf16        | bf16        | f16/bf16/f32  |
| f32         | f32         | f32           |
| i8          | i8            | i32         |
| f8e4m3fn/f8e5m2 | f8e4m3fn/f8e5m2 | f32   |

说明：

- `A`/`B` 为 fp8 时两种编码可以混用（如 `f8e4m3fn × f8e5m2`），输出类型必须为 `f32`。
- [`atomic_add`](basic_matmul_atomic_add.py) / [`auto_sync`](basic_matmul_auto_sync.py) / [`mutex`](basic_matmul_mutex.py) / [`mutex_with`](basic_matmul_mutex_with.py) 无 i8、f8 通路，输入/输出仅支持 f16/bf16/f32；其中 `atomic_add` 的 `--dtype-c` 仅支持 `f32`。
- [`basic_matmul_bias`](basic_matmul_bias.py) 的类型支持：

   | `DTYPE_A` | `DTYPE_B` | `DTYPE_C` | `DTYPE_bias` |
   | --------- | --------- | ---------  | ---------   |
   | f16         | f16           | f32 | f32         |
   | bf16        | bf16          | f32 | f32         |
   | f32         | f32           | f32 | f32         |
   | f8e4m3fn/f8e5m2 | f8e4m3fn/f8e5m2 | f32   | f32 |
   | i8          | i8            | i32         | i32 |
- [`basic_matmul_fixpipe_per_channel`](basic_matmul_fixpipe_per_channel.py) 及 [`basic_matmul_fixpipe_per_tensor`](basic_matmul_fixpipe_per_tensor.py) 的类型支持为：

   | `DTYPE_A` | `DTYPE_B` | `DTYPE_C` | `DTYPE_quant` |
   | ----------- | ----------- | -----------  | -----------  |
   | f32         | f32         | f32/f16/bf16/i8/u8 | per-channel：`u64`<br> per-tensor：`f32`（scale）+ `i32`（offset） |
   | i8          | i8          | f16/bf16/i8/u8 | per-channel：`u64`<br> per-tensor：`f32`（scale）+ `i32`（offset） |

   per-channel 的 `u64` 量化参数由样例内部构造。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

以下为 [`basic_matmul`](basic_matmul.py)的命令行参数，本目录下的其他样例一般包含这些参数：

```text
basic_matmul.py [-h] [--device DEVICE] [--m M] [--n N] [--k K]
                [--layout-a {row,col}] [--layout-b {row,col}]
                [--layout-c {row,col}]
                [--dtype-a {f16,bf16,f32,i8,f8e4m3fn,f8e5m2}]
                [--dtype-b {f16,bf16,f32,i8,f8e4m3fn,f8e5m2}]
                [--dtype-c {f16,bf16,f32,i32}]
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
| `--layout-c`                                | `"row"`                                             | 结果矩阵 C 的数据排布格式。选`"col"` 时，L0C→GM 走转置搬出（nz2dn）通路，直接写出 C 的列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f32"`                       | 左、右矩阵 A、B 和结果矩阵 C 的数据类型。A/B 可选 `"f16"`、`"bf16"`、`"f32"`、`"i8"`、`"f8e4m3fn"`、`"f8e5m2"`；C 可选 `"f16"`、`"bf16"`、`"f32"`、`"i32"`。除 fp8 两种编码可混用外，A/B 需为同一类型。i8 路由要求 A/B 均为 i8 且 C 为 i32；fp8 路由要求 A/B 均为 fp8 且 C 为 f32。 |
| `--relu-enable`                             | `0`                                         | fixpipe是否开启随路relu                                                                       |
| `--block-num`                               | `-1`                                        | 启用的核数，`-1` 表示自动探测可用核数（满核）。                                               |

说明：
- [`basic_matmul_bias`](basic_matmul_bias.py) 不包含 `--layout-a`/`--layout-b`/`--layout-c`/`--dtype-c`/`--relu-enable`；A/B/C 固定 RowMajor。
- [`basic_matmul_fixpipe_per_channel`](basic_matmul_fixpipe_per_channel.py) 及 [`basic_matmul_fixpipe_per_tensor`](basic_matmul_fixpipe_per_tensor.py) 额外提供：
  - `--dtype-ab {f32,i8}`：A/B 共用一种数据类型。
  - `--dtype-c`：取值随 `--dtype-ab` 变化；`f32` 输入可选 `f32/f16/bf16/i8/u8`，`i8` 输入可选 `f16/bf16/i8/u8`。
  - `--layout-c {row,col}`：结果矩阵布局。
  - `--relu-enable {0,1}`：是否启用随路 ReLU。
  - `--quant-scale` / `--quant-offset`：仅 `basic_matmul_fixpipe_per_tensor.py` 提供，分别为 f32/i32 标量；`quant_offset` 仅在 i8/u8 输出时生效。`basic_matmul_fixpipe_per_channel.py` 的量化参数由样例内部构造，无额外命令行参数。

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 基础测试 (默认 m=256, n=512, k=1024)
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py

# 指定 NPU ID 以及核数 (1卡1核)
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --block-num 1 --device 1

# 指定输入矩阵的尺寸、排布以及数据类型
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py \
  --m 256 --n 512 --k 128 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

执行测试后，预期输出：

```plain
--- mnk=(<m>, <n>, <k>) layout=<layout_a>/<layout_b>/<layout_c> dtype=<dtype_a>/<dtype_b>/<dtype_c> ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。

---

## 特性 Kernel 介绍

以下针对本目录下的特性Kernel做简略介绍。

### basic_matmul_atomic_add

**文件**：[`basic_matmul_atomic_add.py`](./basic_matmul_atomic_add.py)

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

### basic_matmul_mutex

**文件**：[`basic_matmul_mutex.py`](./basic_matmul_mutex.py)

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

### basic_matmul_mutex_with

**文件**：[`basic_matmul_mutex_with.py`](./basic_matmul_mutex_with.py)

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

### basic_matmul_auto_sync

**文件**：[`basic_matmul_auto_sync.py`](./basic_matmul_auto_sync.py)

在装饰器 `@tla.kernel` 传入 `auto_sync` 参数（目前支持 `auto_sync="v0"` ），可自动处理多级流水线间的同步，覆盖 MMAD/FIX unit flag 协议且无需手写本地同步，示例如下：

```python
@tla.kernel(auto_sync="v0")
def kernel_func(...):
   # ...
```

---

### basic_matmul_bias

**文件**：[`basic_matmul_bias.py`](./basic_matmul_bias.py)

带偏置项的基础矩阵乘示例，bias 为形如 `(n,)` 的一维向量，经 `GM → L1 → biasbuf` 通道加载。bias 的数据类型跟随累加类型：浮点路由下为 `f32`，i8 路由下为 `i32`。

在计算时，先加载当前 AIC 核所需要的 bias，并在 `init_c=True` 的首个 MMAD 中参与累加。关键代码如下：

```python
# 在 L1/Biasbuf 上申请空间
l1_bias_ptr = tla.allocate(_tiling.l1_tn, bias_dtype, tla.AddressSpace.l1, 512)
l0_bias_ptr = tla.allocate(_tiling.l0_tn, bias_dtype, tla.AddressSpace.biasbuf, 512)

# bias tile 使用 VectorLayout 标记
l1_bias = tla.make_tensor_like(l1_bias_ptr, gm_bias_by_core, tla.arch.VectorLayout)
l0_bias = tla.make_tensor_like(l0_bias_ptr, l1_bias, tla.arch.VectorLayout)


# 每个 block 仅首个 MMAD 携带 bias
if init_c:
   tla.wait_flag(l1_bias_data_ready)
   tla.wait_flag(l0_bias_available)
   tla.copy(l0_bias, l1_bias)
   tla.set_flag(l1_bias_available)

   # 带 bias 的 MMAD 计算
   tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag, bias=l0_bias)
   tla.set_flag(l0_bias_available)
```

### basic_matmul_fixpipe_per_channel

**文件**：[`basic_matmul_fixpipe_per_channel.py`](./basic_matmul_fixpipe_per_channel.py)

该矩阵乘示例在 L0C→GM 经 FIXPIPE 完成完成随路量化（`QuantMode.PER_CHANNEL`），按输出通道应用量化参数。每个输出通道对应一个 `u64` 量化字：低 32 位为 f32 scale），bit 46 为 8 bit 目的类型的符号选择位，[45:37] 为 9 bit 有符号 offset。

通过 `tla.params.CopyL0C2DstParams` 参数指定量化模式及张量，可同时启用随路 ReLU。关键代码如下：

```python
tla.copy(
   gm_c_by_core,
   l0_c,
   tla.params.CopyL0C2DstParams(
      unit_flag=0b11,
      relu_enable=relu_enable,
      quant_mode=tla.params.QuantMode.PER_CHANNEL,
      quant_tensor=l1_quant,
   ),
)
```

量化参数由样例内部构造，无额外命令行参数。

### basic_matmul_fixpipe_per_tensor

**文件**：[`basic_matmul_fixpipe_per_tensor.py`](./basic_matmul_fixpipe_per_tensor.py)

数据通路与随路量化（`QuantMode.PER_CHANNEL`）版本一致，区别在于量化模式为 `QuantMode.PER_TENSOR`： 共用一个标量 `quant_scale`，可选 `quant_offset` （仅在目的类型为 8 bit 时生效）。使用时同样通过 `tla.params.CopyL0C2DstParams` 指定相关入参，关键代码如下：

```python
tla.copy(
   gm_c_by_core,
   l0_c,
   tla.params.CopyL0C2DstParams(
      unit_flag=0b11,
      relu_enable=relu_enable,
      quant_mode=tla.params.QuantMode.PER_TENSOR,
      quant_scale=quant_scale,
      quant_offset=quant_offset if dst_is_b8 else None,
   ),
)
```

### basic_matmul_l0c2l1

**文件**：[`basic_matmul_l0c2l1.py`](./basic_matmul_l0c2l1.py)

演示以 L0C→L1 回搬衔接的连续矩阵乘，计算 `E = (A * B) * D`：第一次 MMAD 的结果保留在 L0C，按 `L0Clayout → zN` 搬回 L1，再作为第二次 MMAD 的输入。两次 MMAD 复用同一组 L0A/L0B/L0C 空间，样例仅支持静态场景。

```python
# 第一次 MMAD 计算：C = A @ B
tla.mmad(l0c_c, l0a_a, l0b_b, init_c=True)

# L0C -> L1
l1_c = tla.make_tensor_like(l1c_ptr, l0c_c, tla.arch.zN)
tla.copy(l1_c, l0c_c)

# 第二次 MMAD 计算：E = C @ D
tla.mmad(l0c_e, l0a_c, l0b_d, init_c=True)
tla.copy(gm_e, l0c_e)
```

该样例中矩阵尺寸固定为 `M/N/K = 50/60/64`（L1/L0 tile 取 64）。

### basic_mmad_ptr

**文件**：[`basic_mmad_ptr.py`](./basic_mmad_ptr.py)

矩阵乘的最小集示例：不依赖动态 GM，输入尺寸全部静态，显式构造 `tla.Tensor` 供 kernel 计算。其中对 A 使用基址 `ptr` 加偏移、步长为 `K_DIM` 的 `make_tensor` 表达子矩阵视图，对 B 使用 `tile_view` 截取 tile。

```python
# 通过 ptr + offset 显式建立 tla.Tensor
gm_a = tla.make_tensor(
   32 + lhs.ptr + 32 * K_DIM,
   tla.make_layout(
      tla.make_shape(L1_M_DIM, L1_K_DIM), tla.make_stride(K_DIM, 1)
   ),
)
# 通过 tile_view "切分" 所需的 tla.Tensor
gm_b = tla.tile_view(rhs, tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_coord(1, 0))
gm_c = tla.make_tensor(
   out.ptr,
   tla.make_layout(
      tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_stride(L1_N_DIM, 1)
   ),
)
```

输入缓冲按 `M/N/K = 64/64/64` 构造，单 AIC 核实际计算 `A[32:,32:] * B[32:,:32]`，仅支持静态场景。
