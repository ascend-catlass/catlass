---
nav_order: 46
---

# 结构化 Kernel 参数

Kernel 参数可以是 Tensor、标量，也可以是包含它们的 tuple/list、NamedTuple、
dataclass。用户直接传递 Python 结构，Python 负责展开参数并在编译函数体时
恢复对应结构，无需手写设备 ABI。

本文说明使用方式。Tensor 创建见 [Tensor 接入](tensor_binding.md)，编译对象与 stream
见 [编译与启动](compile_and_launch.md)，内部机制见
[结构化参数设计](../../dsl_development/feature_development/structured_arguments.md)。

## 最小用法

下面用一个嵌套 tuple/list 传入辅助 Tensor 和标量。kernel 读取第一个元素，
预期结果为 `1.0 + 2.0 + 0.5 = 3.5`：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def add_bias(x: tla.Tensor, aux, out: tla.Tensor):
    bias = aux[0]
    limit = aux[1][0]
    out[0] = x[0] + bias[0] + limit

torch.npu.set_device(0)
x = torch.tensor([1.0], dtype=torch.float32, device="npu")
bias = torch.tensor([2.0], dtype=torch.float32, device="npu")
out = torch.zeros_like(x)

tx = from_dlpack(x, layout_tag=tla.arch.RowMajor)
tb = from_dlpack(bias, layout_tag=tla.arch.RowMajor)
tout = from_dlpack(out, layout_tag=tla.arch.RowMajor)

aux = (tb, [0.5, None])
compiled = tla.compile(add_bias, tx, aux, tout, options="--npu-arch 3510")
compiled(tx, aux, tout, block_num=1)
torch.npu.synchronize()
assert out.item() == 3.5
```

编译和启动都传入完整的 `aux`，不能在启动时改写为 `*aux`。
这里 `None` 保留结构位置，但不向设备传递数据。

## 支持的类型

| 参数类型 | 使用规则 |
| --- | --- |
| `tla.Tensor` | 传入绑定了设备缓冲的 Host Tensor |
| TLA Numeric | 使用自身标量类型，如 `tla.Int64(3)` |
| Python `bool/int/float` | 分别按 i1/i32/f32 处理 |
| `tuple/list` | 支持递归嵌套；编译和启动的容器类型、长度须一致 |
| `NamedTuple` | 保留具体类型和字段顺序，可通过字段名访问 |
| `dataclass` | 按字段顺序处理，允许混合动态字段和 `Constexpr` 字段 |
| `None` | 保留逻辑位置，不产生设备参数 |
| Dynamic-GM Tensor | 支持顶层或嵌套传入，沿用 Tensor API 的动态描述符规则 |

容器的子项也必须属于支持类型。runtime 参数树拒绝 dict、set、循环引用和任意自定义对象。这里的容器限制不用于定义顶层 `Constexpr` 的取值范围。

`make_fake_tensor()` 可用于编译，但不能作为启动时的数据。Tensor 无论位于顶层还是
容器中，提取启动指针时都要求已绑定设备缓冲。

Python `int`、`float` 的子类（包括 `IntEnum`）默认按基础数值传递，分别使用
`i32`、`f32`；`bool` 使用 `i1`。子类的额外属性和方法不随标量传入设备。
数值超过声明类型范围时，启动前报错，不自动扩宽。

dataclass 允许定制 `frozen` 和 `kw_only`，其他装饰器选项须保持默认值，
不支持 `slots=True`。字段 Numeric 注解不负责自动转换 Python 数值：
需要 i64 时传入 `tla.Int64(...)`，而不是只给 Python int 字段标注 Int64。

tuple/list 子类按 `type(value)(children)` 重建，NamedTuple 按
`type(value)(*children)` 重建；子类须能通过该构造方式还原，额外实例状态不随容器传入。
dataclass 按声明字段读取属性，允许使用类默认值；无法读取的声明字段或额外实例字段会报错。
重建按全部声明字段调用 `cls(**fields)`，构造器必须接受这些字段名作为关键字参数。
使用自动生成构造器时，`field(init=False)` 不满足这一要求。
派生属性声明为字段只是必要条件；重建还会在编译期执行构造器及其调用的
`__post_init__` 等逻辑，这些逻辑须能处理重建后的 Tensor/Numeric 值。

## NamedTuple 与动静结合的 dataclass

以下片段沿用上例中的 `tb`、`tout`：

```python
from dataclasses import dataclass
from typing import NamedTuple

class BiasAndLimit(NamedTuple):
    bias: tla.Tensor
    limit: float

@dataclass
class Config:
    enabled: tla.Constexpr[bool]
    values: BiasAndLimit

@tla.kernel
def configured_bias(config, out: tla.Tensor):
    if config.enabled:
        out[0] = config.values.bias[0] + config.values.limit

config = Config(True, BiasAndLimit(tb, 0.5))
compiled = tla.compile(configured_bias, config, tout, options="--npu-arch 3510")
compiled(config, tout, block_num=1)
```

`enabled` 决定编译时生成的代码，`bias` 和 `limit` 在每次启动时传入。
修改静态配置后应重新调用 `tla.compile`。

### 哪些参数在启动时省略

只有**顶层形参声明为 `Constexpr`**，才从 compiled object 的启动签名中移除：

```python
@tla.kernel
def scaled(x: tla.Tensor, out: tla.Tensor, enabled: tla.Constexpr[bool]):
    if enabled:
        out[0] = x[0]

compiled = tla.compile(scaled, tx, tout, True, options="--npu-arch 3510")
compiled(tx, tout, block_num=1)
```

普通空 tuple/list、顶层 `None`、全静态 dataclass 都保留 Python 参数位置，
即使内部没有动态字段。要把整个配置作为编译期参数，应标注顶层 `Constexpr`。
启动时缺参、多参或结构不匹配会报错。

## Dynamic-GM 参数

动态标记作用于 Tensor，与它位于顶层还是容器中无关：

```python
dynamic_x = tx.mark_compact_shape_dynamic(0)
aux = (dynamic_x, [0.5, None])
compiled = tla.compile(add_bias, tx, aux, tout, options="--npu-arch 3510")
compiled(tx, aux, tout, block_num=1)
```

动态维度和布局约束见 [静态与动态 Layout](layout.md)。
普通 Tensor 与 Dynamic-GM 可以组合传入；描述符的物理字段由 Python 处理。
混合参数示例见 [mixed_memref_arguments.py](../../../../../../examples/python/tla_dsl/dsl_tutorials/structured_arguments/mixed_memref_arguments.py)。

## 已编译对象的复用

| 变化 | 处理方式 |
| --- | --- |
| Tensor 内容或地址变化，类型约束不变 | 直接启动，使用当前地址 |
| 运行时标量值变化，类型不变 | 直接启动，使用当前值 |
| 已标记动态的 shape/stride/origin 数值变化，动态模式和约束不变 | 直接启动，使用当前描述符 |
| 静态 shape/stride、dtype、rank、layout、坐标或对齐约束变化 | 重新 compile；启动时不逐项比较这些编译约束 |
| 容器类型、长度、字段顺序或 None 位置变化 | 重新 compile |
| NamedTuple/dataclass 换成另一个类，包括同名不同类 | 重新 compile |
| 静态配置变化 | 必须重新 compile；已有编译对象不比较启动时的 Constexpr 字段值，仍执行原特化代码 |

启动器检查容器结构、叶子种类和 ABI 打包约束。普通 Tensor 与 Dynamic-GM 的完整
类型和布局契约由调用方保证，传入不兼容的 Tensor 可能造成错误结果或越界。
普通 None 保留结构位置检查；声明为 Constexpr 的 None 按静态字段处理。
已编译对象使用编译时的字段定义；不要在编译后修改参数类的字段定义或注解。

重新调用 `tla.compile` 不一定重新编译设备二进制：生成相同 IR 且编译配置一致时，
可以复用产物，但使用本次参数的结构绑定。

## 高级用法：启动叶子适配

编译样例仍须符合前述支持类型。启动时，普通指针叶子可使用提供单个有效地址编码的
pointer provider；描述符叶子则通过 `build_memref_launch_fields()` 提供 canonical
字段，适用于 Dynamic-GM 及混合入口中的静态 memref。启动器检查提取能力、字段数量
和编码，不限定 provider 的 Python 类；调用方负责使这些值满足已编译 kernel 的契约。

既有 `register_jit_arg_adapter` 可将单个启动叶子转换成 Tensor、Numeric 或
pointer provider，转换结果仍须满足该叶子的 ABI。Numeric 和已提供
`__c_pointers__()` 的值直接使用，不被注册 adapter 覆盖。
该接口只适配启动叶子，不用于注册自定义容器或展开规则。

容器重建发生在编译期，Python 对象本身不会复制到设备；结构化参数用于传入字段，
不是设备侧可变 Python 对象的状态传递协议。
