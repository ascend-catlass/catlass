---
nav_order: 50
---

# 结构化 Kernel 参数设计

## 背景与目标

一个 Python 逻辑参数可以包含多个 Tensor、标量和静态字段，而设备函数接收的是
确定类型的形式参数，启动器填充的是物理 ABI 字段。结构识别、设备参数生成与
启动打包必须使用一致的顺序，否则即使各字段类型合法，也可能传错值。

本设计用参数树保存逻辑结构，用叶子绑定连接设备类型，再由 TypeBridge 生成的
`KernelAbiLayout` 确定启动字段。使用方法和能力边界见
[结构化 Kernel 参数](../../kernel_development/core_concepts/structured_arguments.md)。

## 模块与职责

以下路径相对 `python/tla_dsl/catlass/`：

| 模块 | 职责 |
| --- | --- |
| `base_dsl/utils/tree_utils.py` | 通用节点、树结构、统一回放与 unflatten |
| `base_dsl/runtime/argument_tree.py` | 识别 TLA 参数，构建树并校验运行时对象 |
| `base_dsl/runtime/jit_arg_adapters.py` | dataclass 字段拆分与重建，保留既有 host launch adapter |
| `base_dsl/runtime/argument_binding.py` | 叶子到 MLIR/ABI 的物理绑定描述 |
| `execution_lowering.py` | 一次分配叶子槽位，构造 proxy 并调用 unflatten，再生成函数体 |
| `execution.py`、`base_dsl/jit_executor.py` | 产物缓存、运行时参数校验、payload 打包和启动 |
| `csrc/mlir/lib/Tools/TypeBridge.cpp`（相对 DSL 根目录） | 根据 lowered 参数类型与后端调用约定生成 ABI 字段 |

## 参数树与叶子绑定

| 对象 | 保存什么 |
| --- | --- |
| `NodeType` | 一类节点的 `to_iterable/from_iterable` 拆分与重建规则 |
| `PyTreeDef` | 当前参数的节点类型、元数据和有序子树 |
| `Leaf` | runtime 绑定序号，或 constexpr 静态值、Unit 位置 |
| `RuntimeLeafBinding` | 叶子路径、MLIR 类型和 lowering 类别 |
| `RuntimeArgumentTree` | 一个顶层参数的树和有序运行时绑定 |
| `KernelAbiLayout` | 最终入口的物理字段、偏移、宽度和逻辑分组 |

以 `aux = (bias, [0.5, None])` 为例：

```text
tuple
├── child[0] Tensor  → runtime leaf 0
└── child[1] list
    ├── child[0] f32 → runtime leaf 1
    └── child[1] Unit → 无 runtime leaf
```

运行时叶子采用深度优先、从左到右的顺序。节点元数据保存具体 Python 类型、
字段名或索引和静态字段位置，不保留容器样例作为重建模板。

树负责“字段在哪里”，绑定负责“字段是什么设备类型”，ABI 布局负责“启动时写到哪里”。

建树时按 DFS 顺序生成 `RuntimeLeafBinding`，运行时 `Leaf.binding_index` 指向
对应绑定。MLIR 类型保存在 binding 中，启动时不重新推导类型。
lowering 按绑定类别生成实际参数类型列表，并按列表长度分配槽位：
普通叶子为 1 个 block argument，Dynamic-GM 为 3 个。
`RuntimeLeafBinding.kind` 选择叶子的 lowering 和重建方式，不表示最终物理 ABI 类型。

`TreeMetadata` 保存节点类型、字段名、子项键和静态字段位置。
子项数量直接取子项键的长度，静态字段名按位置取得；静态值保存在 constexpr 叶子中。

## 编译与启动流程

沿用用户指南的 `add_bias(x, aux, out)`，完整逻辑叶子顺序是
`[x, bias, 0.5, out]`。

### 绑定 Python 调用并建树

`tla.compile` 通过 `_bind_kernel_call_args()` 按函数签名绑定位置参数、关键字参数和
默认值，分离编译选项，将实参原样交给 lowering。无参调用使用 `type_args=None`；
包含 None 或空容器的调用仍保留对应实参。类型与结构是否合法由 lowering 和参数树判断。
只有顶层 `Constexpr` 从启动签名移除；普通零叶子结构仍保留逻辑位置。

`build_runtime_argument_tree()` 对每个运行时顶层参数建树。容器由
`NodeType.to_iterable()` 提供元数据和子项，统一遍历负责递归、路径和循环检查。
NamedTuple 在普通 tuple 之前识别，以保存具体类型。字段级 Constexpr 和 Unit
保留在树中，但不生成运行时绑定。

### 分配 MLIR 参数并重建对象

`execution_lowering.py` 中的 `_build_runtime_physical_argument_layout()`
按绑定顺序分配 block arguments，即设备函数的形式参数。

普通 Tensor、scalar 分配一个参数；Dynamic-GM 分配三个；
Constexpr/Unit 不分配。Tensor 绑定转换为可索引的 proxy，scalar 转换为相应
Numeric 值，随后通过树重建 tuple/list/NamedTuple/dataclass。

因此，编译用户函数体时的 `aux` 是：

```text
(bias_proxy, [scalar_proxy, None])
```

`aux[0][0]` 访问的是设备参数对应的 Tensor proxy，运算生成 TLA IR。
Python 容器重建发生在编译期间。

dataclass 按全部声明字段调用 `cls(**fields)` 重建，构造器须接受这些字段关键字。
因此，使用自动生成构造器的 `field(init=False)` 不满足重建约定。
派生属性声明为字段只是必要条件；构造器及其调用的 `__post_init__` 等逻辑也在
编译期执行，须能处理重建后的 Tensor/Numeric 值。字段的读取允许使用类默认值。

### 生成产物与 ABI 描述

TLA IR 经 lowering 后，TypeBridge 根据 lowered 参数类型和当前后端调用约定生成
`KernelAbiLayout`，后端编译器生成设备二进制。布局由 TypeBridge 推导，并非 hivmc
导出的最终二进制 ABI 清单；TypeBridge 和 host packer 须共同遵循后端调用约定。
Frontend 的 block argument 与物理字段不总是一一对应：
memref 可能继续展开为多个地址、尺寸和步长字段。

host packer 按 `KernelAbiLayout` 准备字段组和打包器，确定 payload 大小与字段位置。

### 回放当前参数并启动

`ExecutionArgs` 的树路径先校验 Python 逻辑参数数量，再按编译时的树回放当前对象。
全部参数为普通 Tensor/scalar 时，每个顶层参数就是一个运行时叶子，可直接交给
prepared packer 校验数量、类型和范围并打包。

容器回放调用 `to_iterable(value, expected_metadata)`，使用保存的元数据读取当前子项，
检查具体类型、长度和 None/Unit 位置。dataclass 按保存的字段顺序通过
`getattr()` 取值，声明字段无法读取或存在额外实例字段时会报错。
Constexpr 字段不进入 payload，也不参与启动值比较；修改静态配置后须重新 compile。
遍历函数维护路径和循环检测集合，共用一个叶子输出列表。

展平后应用既有 launch adapter，再由 prepared packer 按 ABI 顺序写入缓冲区。
adapter 只转换单个启动叶子，不参与建树；Numeric 和已有 `__c_pointers__()` 的
provider 保持原值。pointer/scalar 的种类、位宽和范围由 packer 检查；memref
字段组通过 `build_memref_launch_fields()` 提取并检查 canonical 字段数量和编码。
provider 按这些 ABI 接口接入，无需限定其 Python 类。

底层 Tensor 在提取指针或描述符时检查绑定状态，描述符提取还检查非空 shape 和
支持的 rank。地址、标量值和动态尺寸来自当前调用。
启动器不逐项比较完整的 shape、stride、dtype 或 layout；调用方仍负责满足
已编译 kernel 的静态类型与布局契约，违反约束可能造成错误结果或越界。

## Dynamic-GM 的物理映射

Dynamic-GM 仍是一个逻辑 Tensor；它需要传递动态 shape、stride 和 origin：

| 层次 | 普通 Tensor | Dynamic-GM |
| --- | --- | --- |
| 参数树 | 一个 Tensor 叶子 | 一个 Dynamic-GM 叶子 |
| Frontend block arguments | 一个 Tensor 参数 | 一个动态 memref 和两个 origin index |
| 最终 host payload | pointer 或静态 memref 字段组 | canonical descriptor 字段组 |

`_materialize_tree_dynamic_gm_descriptors()` 使用动态形式参数构造 descriptor，
`_rebuild_tree_argument_proxies()` 将 Tensor proxy 放回原结构。

在本 DSL 使用的 hivmc 混合入口调用约定下，普通静态 GM memref 也展开为
rank-R descriptor。TypeBridge 与 host packer 必须共同遵循：

- Dynamic-GM：统一 rank-4 memref 的 11 个字段，加两个 origin，共 13 个值。
- 静态 rank-R memref：allocated、aligned、offset、R 个 size、R 个 stride，
  共 `3 + 2R` 个值。
- 纯静态 pointer 入口：继续传单独地址。

host Tensor 的 `build_memref_launch_fields()` 提供统一 13 值 tuple。
packer 准备阶段校验静态 rank（1～4）、字段顺序、宽度和分组，保存字段选择索引。
例如 rank-1 静态 memref 选择索引 `(0, 1, 2, 3, 7)`，不能取前五项。

Dynamic-GM 直接打包全部 13 个值；多个描述符分别形成独立字段组。

这样嵌套位置只影响逻辑路径，不改变该 Tensor 的描述符规则。
可运行用例见 [mixed_memref_arguments.py](../../../../examples/end_to_end/structured_arguments/mixed_memref_arguments.py)。

## 缓存与绑定生命周期

```text
当前编译参数 → 当前参数树与 TLA IR
                       ↓
            以 IR、选项和工具链查询产物缓存
                       ↓
            设备二进制 + KernelAbiLayout
                       ↓
            配上本次签名与参数树 → compiled object
```

内存和磁盘命中均使用本次请求的参数绑定，不沿用旧请求的 Python 类型模板。
tuple/list 或不同 dataclass 生成相同 IR 时可以共享设备产物，但各自校验自身结构。

Tensor 地址、version、内容不进入 artifact key。静态字段值按引用保存在参数树中并
参与 tracing；修改静态配置后重新 compile，已有编译对象继续使用原特化代码。
设备 ABI 和缓存格式的兼容标记隔离不兼容产物。

现有 compiled object 的 NamedTuple/dataclass 校验采用当前进程的具体类型身份；
持久缓存不编码 Python 类型地址。合法的新编译请求可绑定自己的同名类型并复用
相同设备产物。

## 维护入口

结构、重建和 launch 规则对应的测试位于：

- [test_struct_like_jit_arguments.py](../../../../tests/test_struct_like_jit_arguments.py)：参数树、类型校验和设备侧重建。
- [test_execution_args.py](../../../../tests/test_execution_args.py)：逻辑调用参数与启动 payload。
- [test_dynamic_gm_launch_abi.py](../../../../tests/test_dynamic_gm_launch_abi.py)：描述符字段组与混合 ABI。
- [test_execution_hivm_runtime.py](../../../../tests/test_execution_hivm_runtime.py)：编译、缓存和执行器。
