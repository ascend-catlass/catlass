# 基础 VADD 端到端示例

本目录下样例演示 **CATLASS Python** 下基础向量加法的计算。

## 功能说明

向量加法算子实现两个一维向量的逐元素加法，计算公式为：
$$
\begin{aligned}
C &= A + B
\end{aligned}
$$

## 代码组织

本目录组织结构如下所示：

```plain
./basic_vadd
├── basic_vadd.py
└── README.md
```

各子文件所承载样例特性概述如下：

| 文件 | 概述 |
|------|------|
| [**`basic_vadd.py`**](basic_vadd.py) | 基础向量加法示例，含 4 个 kernel 变体：flag 同步（默认）、mutex（`--use-mutex`）、mutex_with（`--use-mutex-with`）和 atomic_add（`--use-atomic-add`）。 |

## 约束说明

 - 加和向量 A、B 的数据类型要求一致，所支持的数据类型及分块lane大小如下。

| dtype | VL_ELE | 浮点/整数 | 校验方式 |
|-------|--------|----------|---------|
| `f32` | 64 | 浮点 | 容差 |
| `f16` | 128 | 浮点 | 容差 |
| `i8` | 256 | 整数 | 精确匹配 |
| `i16` | 128 | 整数 | 精确匹配 |
| `i32` | 64 | 整数 | 精确匹配 |

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)一节的有关内容。

### 命令行参数

```text
basic_vadd.py [-h] [--device DEVICE] [--n N]
              [--block-num BLOCK_NUM]
              [--dtype {f32,f16,i8,i16,i32}]
              [--use-mutex | --use-mutex-with | --use-atomic-add]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--n` | `400` | 向量长度。 |
| `--block-num` | `-1` | 启用的核数，`-1` 表示自动探测可用核数（满核）。 |
| `--dtype` | `"f32"` | 数据类型，可选 `"f32"`、`"f16"`、`"i8"`、`"i16"`、`"i32"`。 |
| `--use-mutex` | `False` | 切换到显式 Mutex `lock` / `unlock` 同步（执行 `basic_vadd_mutex`）。 |
| `--use-mutex-with` | `False` | 切换到 `with tla.mutex_guard(...)` 同步（执行 `basic_vadd_mutex_with`）。 |
| `--use-atomic-add` | `False` | 切换到原子加同步（执行 `basic_vadd_atomic_add`）。 |

> `--use-mutex`、`--use-mutex-with`、`--use-atomic-add` 三者互斥。

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 基础测试 (默认 f32, n=400)
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py

# 指定向量长度和 dtype
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py --n 256 --dtype f16

# 使用 Mutex 同步
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py --use-mutex
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py --use-mutex-with

# 使用原子加
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py --use-atomic-add
```

执行测试后，预期输出：

```plain
--- dtype=<dtype> n=<n> ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

- 上述 `<dtype>`， `<n>` 等为占位符，具体依赖外部参数或环境变量传入。

其中 `passed` 结果为 `True` 或 `False` 表明 NPU 计算结果与 golden 参考值精度校验是否通过；`cache_dir` 是指定的缓存目录，`cache_key` 是编译缓存的哈希值。

---

## 特性 Kernel 介绍

以下针对本文件中的特性 Kernel 做简略介绍。

### basic_vadd_mutex

**文件**：[`basic_vadd.py`](basic_vadd.py#L67)

使用显式 `mutex.lock(pipe)` / `mutex.unlock(pipe)` 实现同步，以 GM 至 UB 的搬运为例：

```python
import catlass.tla as tla

# ...
mutex_ub_a = tla.mutex(resource="ub_a", id=0)

mutex_ub_a.lock(pipe=tla.pipes.MTE2)
tla.copy(ub_a, gm_a)
mutex_ub_a.unlock(pipe=tla.pipes.MTE2)
```

### basic_vadd_mutex_with

**文件**：[`basic_vadd.py`](basic_vadd.py#L127)

使用 `with tla.mutex_guard(...)` 上下文管理器替代手动 `lock`/`unlock`：

```python
import catlass.tla as tla

# ...
with tla.mutex_guard(mutex_ub_a):
    tla.copy(ub_a, gm_a)
```

### basic_vadd_atomic_add

**文件**：[`basic_vadd.py`](basic_vadd.py#L175)

先将 A 写回 GM 覆盖目的位置，再将 B 写回 GM 上同一片区域，并开启原子加操作，关键代码如下：

```python
import catlass.tla as tla

# ...
tla.copy(gm_c, ub_a)  # C = A
tla.pipe_barrier(tla.pipes.MTE3)
tla.copy(gm_c, ub_b, tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD))
```
