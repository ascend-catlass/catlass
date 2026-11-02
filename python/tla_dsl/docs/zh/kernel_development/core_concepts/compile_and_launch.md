---
nav_order: 45
---

# 编译与启动

本文说明 Host 侧使用 `tla.compile` 和启动 kernel 时可观察到的行为，包括编译样本、
编译缓存、重复启动和 stream 选择。Host tensor 的创建方式见
[Tensor 接入](tensor_binding.md)，接口完整签名见
[程序生命周期参考](../../api/runtime/index.md)。

tuple/list、dataclass 等复合参数的传递方式见 [结构化 Kernel 参数](structured_arguments.md)。

---

## 基本流程

先用 Host tensor 作为类型样本完成编译，再用绑定了真实 NPU 缓冲的 tensor 启动：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# vadd 是一个由 @tla.kernel 装饰的函数。
torch.npu.set_device(0)
a = torch.rand(64, dtype=torch.float32, device="npu")
b = torch.rand(64, dtype=torch.float32, device="npu")
out = torch.zeros(64, dtype=torch.float32, device="npu")

ta = from_dlpack(a, layout_tag=tla.arch.RowMajor)
tb = from_dlpack(b, layout_tag=tla.arch.RowMajor)
tout = from_dlpack(out, layout_tag=tla.arch.RowMajor)

compiled = tla.compile(vadd, ta, tb, tout, options="--npu-arch 3510")
compiled(ta, tb, tout, block_num=1)
torch.npu.synchronize()
```

`tla.compile` 的参数用于确定 kernel 的编译类型。编译样本可以是 `from_dlpack` 绑定的
真实 tensor，也可以是没有设备指针的 `make_fake_tensor`；启动时则必须传入绑定了真实
NPU 缓冲的 tensor。

默认 layout 是静态的，shape、stride 等信息会参与编译。若同一份编译产物需要处理不同
shape，应在编译和启动所用的 Host tensor 上正确标记动态维度，详见
[静态与动态 Layout](layout.md)。

---

## 以类实例组织 Kernel

编译期配置（tile size、stage 数、向量长度等）可以放在实例里，与 kernel 和编译入口封装成同一个抽象。kernel 写成实例方法，第一个参数 `self` 不进入启动 ABI，`self.attr` 在编译时折叠为常量：

```python
from functools import cache

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

VL_ELE = 64               # f32 的 SIMD 向量长度


class VectorAddWithStaticLength:
    """gm_c = gm_a + gm_b，长度 n_ele 在编译期确定。"""

    def __init__(self, n_ele: int, vl_ele: int = VL_ELE) -> None:
        self.n_ele = n_ele
        self.vl_ele = vl_ele

    @tla.jit
    def __call__(self, gm_a, gm_b, gm_c):
        self.kernel(gm_a, gm_b, gm_c)

    @tla.kernel
    def kernel(self, gm_a, gm_b, gm_c):
        ub_a = tla.make_tensor_like(
            tla.allocate(self.n_ele, tla.Float32, tla.AddressSpace.ub, 256),
            gm_a, tla.arch.RowMajor,
        )
        ...                                    # 受篇幅限制，省略 kernel 主逻辑

        with tla.vec.func(mode="simd"):
            for i in tla.range(self.n_ele // self.vl_ele):
                ...                            # 循环上界为编译期常量

    @cache
    @staticmethod
    def compile(n_ele: int, vl_ele: int = VL_ELE):
        fake = make_fake_tensor(tla.Float32, (n_ele,), (1,))
        return tla.compile(
            VectorAddWithStaticLength(n_ele, vl_ele), fake, fake, fake,
            options="--npu-arch 3510",
        )


compiled = VectorAddWithStaticLength.compile(n_ele=384)
compiled(gm_a, gm_b, gm_c, block_num=1)
```

### 编译入口的写法

| 写法 | 入口如何确定 | 何时使用 |
|------|--------------|----------|
| `tla.compile(实例, gm_a, gm_b, gm_c)` | 取类自己的 `@tla.jit __call__`，编译它一行转发到的 kernel 方法，方法名不限 | 默认写法 |
| `tla.compile(实例.kernel, gm_a, gm_b, gm_c)` | 直接点名 kernel 方法；从实例读到它时，该实例随属性读取一并作为 receiver | 类里有多个 kernel，需要指定其中一个 |

两种写法编译出的产物相同：实例作为 receiver 排在编译类型样本的第 0 位，落进 kernel 的 `self` 参数，不占启动 ABI。`n_ele` / `vl_ele` 是编译期输入，不同实例得到不同的编译产物；`gm_a`、`gm_b`、`gm_c` 仍是启动时传入的 tensor。

完整 kernel 与可运行版本见 [vector_add_with_static_length.py](../../../../../../examples/python/tla_dsl/dsl_tutorials/kernel_in_class/vector_add_with_static_length.py)。

约束：

- kernel 的第一个参数是 `self` 时，它只承载编译期配置，不占启动 ABI。判定按位置和名字进行：位置 0 且名为 `self` 即为接收者（`cls` 需要 `classmethod` 宿主），不区分该函数是否定义在类里。
- `@tla.jit __call__` 必须是 `self.<kernel>(...)` 形式的纯转发，并按自身参数顺序传参；其它形式的 `@tla.jit __call__` 函数体不支持。

入口不合法时的报错：

| 情况 | 报错内容 |
|------|----------|
| 类里没有 `@tla.jit __call__` | `declares no @tla.jit __call__`，并给出正确写法 |
| `__call__` 不是一行纯转发 | 指出具体原因：语句数、关键字实参、实参不是自身参数名、参数顺序不符等 |
| 转发目标不是本类的 `@tla.kernel` 方法 | `not one of that class's @tla.kernel methods` |
| 转发参数个数与 kernel 的运行时参数个数不一致 | `forwards N parameters to 'kernel', which takes M (self excluded)` |

---

## 用户需要认识的对象

编译和启动过程中会遇到以下对象：

| 对象 | 如何得到 | 用户如何使用 |
|------|----------|--------------|
| `TlaJitFunction` | Python 函数经 `@tla.kernel` 装饰后得到 | 作为 `tla.compile` 的第一个参数；通常不需要手工构造 |
| `JitCompiledFunction` | `tla.compile(...)` 的返回值 | 直接调用以启动 kernel；首次调用延迟创建 executor，后续调用复用它 |
| `JitExecutor` | `JitCompiledFunction` 首次启动时在内部创建 | 承载已加载的 binary 和启动状态；用户无需直接构造或调用 |
| `TlaExecutionResult` | 调用 `compiled(...)` 后返回 | 查看本次启动的运行信息；计算结果仍写入 kernel 的输出 tensor |

### `JitCompiledFunction`

`JitCompiledFunction` 表示一次 `tla.compile` 的结果。它持有与设备无关的编译产物，
但刚返回时尚未创建 executor，也没有加载 binary。

用户常用的操作包括：

- `compiled(*args, block_num=...)`：使用内部 executor 启动；首次调用时才创建该
  executor，后续调用复用它；
- `compiled.cache_key`：查看本次编译对应的缓存 key；
- `compiled.kernel_binary_path`：查看设备二进制路径；
- `compiled.artifacts`：在调试时查看 MLIR、后端 IR、缓存目录等编译产物信息。

每次调用 `tla.compile(...)` 都返回新的 `JitCompiledFunction`。即使编译缓存命中，新的
对象也不会继承旧对象已经创建的 executor 或启动状态。

### `JitExecutor`

`JitExecutor` 是 `JitCompiledFunction` 内部使用的执行对象。首次调用 `compiled(...)`
时，它负责加载 binary；后续调用复用同一个 executor 和已加载的执行状态。

```python
result = compiled(ta, tb, tout, block_num=1)
```

用户不需要获取或直接调用 `JitExecutor`。重复启动、参数传递和 stream 选择都通过
`JitCompiledFunction` 完成。

`JitModule`、`ExecutionArgs` 等对象用于承载框架准备好的启动信息。普通用户无需直接
构造或修改它们；编译使用 `tla.compile(...)`，启动时直接调用返回值即可。

---

## 编译缓存

编译缓存默认启用。每次调用 `tla.compile` 都会先根据 kernel 和编译类型样本生成编译
表示，并据此计算 cache key。影响 cache key 的主要内容包括：

- kernel 代码，以及静态 shape、`constexpr` 等编译期输入；
- kernel mode、目标架构和编译选项；
- 编译工具链和运行 ABI 版本；
- kernel 使用的外部 Ascend C 源码。

设备指针、输入数据、`block_num` 和 stream 是启动状态，不进入 cache key。因此，仅
更换输入缓冲不会触发重新编译；改变静态 shape、`constexpr`、目标架构或 kernel 代码
通常会产生新的 key。

缓存按以下顺序查找：

1. **进程内缓存**：同一进程中 key 相同且编译产物仍存在时，直接复用。
2. **磁盘缓存**：进程内未命中时，检查缓存目录中已持久化的二进制及其描述信息。
3. **重新编译**：两级缓存均未命中时运行后端编译，并更新磁盘和进程内缓存。

文件缺失或缓存版本过期按未命中处理；缓存描述损坏或启动 ABI 无效时会报错，不会静默
使用不可信的旧产物。

无论是否命中缓存，每次显式调用 `tla.compile(...)` 都会返回一个新的
`JitCompiledFunction`：

| 会复用 | 不会复用 |
|------|------|
| 编译后的设备二进制及其启动描述 | 先前创建的 executor |
| 缓存中的编译诊断和产物信息 | 已加载的执行状态和 stream |

也就是说，缓存命中只省去重复的后端编译，不会把上一次启动使用的 executor 或 stream
带到新的 `JitCompiledFunction` 中。

常用缓存开关如下，完整列表见 [环境变量](env_vars.md)：

| 环境变量 | 作用 |
|------|------|
| `CATLASS_DSL_CACHE=0` | 关闭编译缓存 |
| `CATLASS_DSL_CACHE_DIR=/path/to/cache` | 指定磁盘缓存目录 |
| `CATLASS_DSL_FORCE_RECOMPILE=1` | 忽略已有缓存并强制重新编译 |

---

## 首次启动与重复启动

| 调用方式 | 行为 |
|------|------|
| 首次调用 `compiled(...)` | 延迟创建 executor，并加载 binary |
| 再次调用同一个 `compiled(...)` | 复用 executor 和已加载的执行状态 |

重复启动时直接复用 `compiled`：

```python
compiled(ta, tb, tout, block_num=1)
compiled(ta2, tb2, tout2, block_num=1)
```

---

## 选择 stream

未显式传入 `stream=` 时，executor 会在**每次启动**时查询其设备上的当前 NPU stream。
因此，同一个 executor 可以跟随 stream 上下文变化：

```python
with torch.npu.stream(stream_a):
    compiled(ta, tb, tout, block_num=1)

with torch.npu.stream(stream_b):
    compiled(ta2, tb2, tout2, block_num=1)
```

显式传入 `stream=` 时，以该参数为准。stream 的变化不会触发重新编译，也不会创建新的
executor。

---

## 使用建议

- 重复启动：直接复用同一个 `compiled(...)`。
- 不同 stream：复用同一个 `compiled`，在启动前切换 stream 上下文。
- 仅需类型样本：用 `make_fake_tensor` 编译，启动时换成真实绑定 tensor。
- 跨 shape 复用：正确标记动态 layout；不要依赖缓存忽略静态 shape 差异。
- 配置封装：把编译期配置放进实例属性，用类实例作为 `tla.compile` 的第一个参数。
