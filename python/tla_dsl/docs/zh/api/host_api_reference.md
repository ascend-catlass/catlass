---
nav_order: 15
---

<!--
手工维护的中文 Host API 参考。
英文源：docs/en/api/host_api_reference.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# TLA DSL Host API 参考

本文档介绍 **TLA DSL 的 Host 侧 API**（通常以 `import catlass.tla as tla` 导入）。
内容覆盖：`@tla.kernel` / `@tla.jit` / `@tla.extern` 装饰器、Host 侧 `@dataclass` 打包、`tla.compile` /
`JitCompiledFunction` 启动、Host tensor。
使用流程见 [编译与启动](../kernel_development/core_concepts/compile_and_launch.md)，环境变量见
[环境变量](../kernel_development/core_concepts/env_vars.md)。Kernel 侧接口见
[Kernel API 参考](kernel_api_reference.md)。

接口说明与调用示例来自各 API 源码 docstring；这些接口均在 Python Host 脚本中、
`@tla.kernel` 函数体**外**调用。

DLPack 接入教程见 [Host Tensor 接入](../kernel_development/core_concepts/tensor_binding.md)。
动态 layout 编程见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

---

## 目录

- [装饰器](#装饰器)
- [编译与启动](#编译与启动)
  - [编译](#编译)
  - [启动](#启动)
  - [查看 IR](#查看-ir)
- [Host Tensor](#host-tensor)
  - [创建与绑定](#创建与绑定)
  - [动态 Layout](#动态-layout)

---

## 装饰器

Host 侧 `@tla.kernel` 入口、`@tla.jit` device helper、`@tla.extern` Ascend C 复用，以及 Host 侧 `@dataclass` 打包。
被装饰的 kernel 函数体在 Host 端不执行。

### `kernel`

**源码：** [`catlass.dsl.kernel`](../../../catlass/catlass_dsl/catlass.py#L27)

功能说明：

将 Python 函数标注为 TLA Kernel 入口。函数体在 Host 端不执行。
返回 `TlaJitFunction`。不允许直接调用 kernel；应先用 `tla.compile`
显式编译，再调用返回的 `JitCompiledFunction` 启动。

函数原型：

```python
tla.kernel(fn: Callable[..., Any] | None = None, *, auto_sync: str | None = None) -> TlaJitFunction | Callable[[Callable[..., Any]], TlaJitFunction]
```

参数说明：

- *`fn`*（`Callable[..., Any] | None`）：被装饰的函数。用 `@tla.kernel` 或
  `@tla.kernel(auto_sync=...)`；只有无法使用装饰器语法时才手写 `tla.kernel(fn)`。
- *`auto_sync`*（`str | None`）：可选。`"v0"` 为受支持的 `tla.copy`、
  `tla.mmad` 和 `tla.vec.func` 访存启用实验性自动核内同步；默认 `None`
  （同步仍由用户显式控制）。

约束说明：

- 被装饰的函数不能用 Python 的 `async def` 定义。
- 关键字选项有白名单：当前仅支持 `auto_sync`。
- `auto_sync` 只能是 `"v0"` 或 `None`。
- 使用 `auto_sync="v0"` 时：
  - 只生成单个 AIC 或 AIV 内部不同硬件流水之间的核内同步。AIC/AIV
    核间同步和 `tla.vec.func` 内部需要的线程级同步仍由开发者显式处理。
  - 不能与核内 `tla.flag` / `tla.set_flag` / `tla.wait_flag`、`tla.mutex` /
    `tla.mutex_guard` 或 `tla.call_extern` 混用。
  - 被自动同步保护的片上 tensor 必须来自 `tla.allocate`。不支持保护通过
    `tla.make_ptr` 从片上裸地址构造的 tensor。
  - 当前 AutoSync 不支持将 UB `tla.scalar_load` / `tla.scalar_store` 直接写在
    `tla.vector` 下；启用 AutoSync 时须将其放在 `tla.vec.func` 内。不启用
    AutoSync 时，这类写法仍然合法。
  - 支持通过运行时条件选择由 `tla.allocate` 创建的 buffer，但不支持在循环
    迭代间把携带的 pointer 切换到另一块 allocation，也不支持多个条件 buffer
    在不同分支中使用不一致的 allocation 顺序。
  - `tla.mmad` 的 `unit_flag` 必须能静态证明为始终等于 0 或始终位于
    `{2, 3}`；L0C copy 的 `unit_flag` 只支持 0 或 3。
  - `tla.print_tensor` 和 `tla.debug_print` 不会获得自动同步。
- 详细设计和限制说明见 [AutoSync 设计](../dsl_development/feature_development/auto_sync_design.md)。
- 启动前必须调用 `tla.compile(kernel, *sample_args)`；直接调用装饰后的
  kernel 会抛出 `TypeError`。

**Kernel 入参类型**

| 类别 | 类型 | 启动时传入 |
| --- | --- | --- |
| Tensor | `tla.Tensor` | 是 |
| Python 标量 | `bool` / `int` / `float` | 是 |
| `tla` 标量 | `Bool`、`Int8/16/32/64`、`UInt8/16/32/64`、`Float16/32`、`BFloat16` | 是 |
| 编译期常量 | `tla.Constexpr[...]` | 否 |
| 编译期函数 | `tla.Constexpr[Callable[...]]` 或 `tla.Constexpr` | 否；见 [Constexpr Callable 入参](#constexpr-callable-入参) |
| 结构体 | 字段类型属于上表的 `@dataclass` 实例 | 按字段展开；Constexpr 字段 launch 时不需要传入 |

调用示例：

```python
@tla.kernel
def vadd(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(src, dst)

compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
```

---

### `jit`

**源码：** [`catlass.dsl.jit`](../../../catlass/catlass_dsl/catlass.py#L114)

功能说明：

将 Python 函数标注为 device 侧 DSL helper。

- 在 `@tla.kernel` 降级过程中被调用时，函数体内联进该 kernel 的 device IR。
- 可作为 [Constexpr Callable](#constexpr-callable-入参) kernel 入参；也可在 kernel 内按名调用。
- 在 Host 上直接调用时按普通 Python 执行。

函数原型：

```python
tla.jit(fn: Callable[..., Any] | None = None) -> Callable[..., Any]
```

参数说明：

- *`fn`*（`Callable[..., Any] | None`）：被装饰的函数。用 `@tla.jit`；只有无法使用
  装饰器语法时才手写 `tla.jit(fn)`。

约束说明：

- 可用普通 `def` 定义；不可使用 `async def`。
- 不接受关键字选项。
- 无独立的 `dump_mlir` / `compile`。
- helper 之间不可递归调用。
- helper 可包含需框架改写的控制流：动态 `if` / `while`、`tla.range` 等；
  在 `@tla.kernel` 降级内联时一并处理。

调用示例：

```python
@tla.jit
def apply_abs(value):
    return tla.abs(value)

@tla.kernel
def k(src: tla.Tensor, dst: tla.Tensor) -> None:
    ...
    y = apply_abs(x)  # 按名调用，compile 时内联

compiled = tla.compile(k, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
```

---

### Constexpr Callable 入参

对应 [`kernel`](#kernel) 入参表中「编译期函数」一行（`tla.Constexpr[Callable[...]]`，或实参为可调用对象的 `tla.Constexpr`）；不含「编译期常量」。

传递形态：外层 `def`、`lambda`、`functools.partial`，或 `@tla.jit` 装饰的函数。
在 `tla.compile` 时传入，`compiled(...)` 启动时不再传入；不同函数对象对应不同特化。

函数体语义（普通 `def` / `lambda` / `partial`）：

- 只在 `tla.compile` / `dump_mlir` 时执行；其中的 DSL 操作写入当前 kernel 的 device IR。`compiled(...)` 启动时不再执行该函数体。
- 在 kernel 内调用时，函数体只能使用 [Kernel API](kernel_api_reference.md) 中的接口，具体约束见对应条目。
- 任意 Host 侧 Python 不构成设备计算语义，包括但不限于第三方库、文件/网络 I/O、对本机状态的依赖，以及把 DSL 值当 Host 张量或容器使用。
- 不支持 TLA 控制流：`tla.range`、动态 `if` / `while` 等。

`@tla.jit` 作入参时的函数体约束见 [`jit`](#jit)。

**Kernel 体内可调用**

- 外层普通 `def`：同本节「函数体语义」。
- `@tla.jit` helper：见 [`jit`](#jit)；降级时内联进当前 kernel IR。
- Constexpr Callable 入参：同本节。

调用示例：

```python
def abs_epilogue(value):
    return tla.abs(value)

@tla.kernel
def transform(src: tla.Tensor, dst: tla.Tensor, epilogue: tla.Constexpr) -> None:
    tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            dst_tile.store(epilogue(tile.load()))

compiled_ep = tla.compile(transform, tx, ty, abs_epilogue, options="--npu-arch 3510")
compiled_ep(tx, ty, block_num=1)  # 不再传 abs_epilogue
```

---

### `dataclass`

**源码：** [`dataclasses.dataclass`](../../../catlass/base_dsl/runtime/jit_arg_adapters.py#L135)

功能说明：

用 Python 标准库 `@dataclass` 在 Host 侧打包 kernel 入参。可在 Host 上创建实例后传给
`tla.compile` / 启动；也可在 kernel 内构造字段实例。

函数原型：

```python
dataclasses.dataclass(cls: type, *, frozen: bool = False, kw_only: bool = False) -> type
```

参数说明：

- *`frozen`*（`bool`）：为 `True` 时实例不可变。默认 `False`。
- *`kw_only`*（`bool`）：为 `True` 时字段必须按关键字传入。默认 `False`。

约束说明：

- 作为 kernel 入参时，只支持设置 `frozen` / `kw_only`；其它 stdlib 选项
  （如 `slots=True`、`init=False`）会在编译期报错。
- 支持的字段类型：

  | 类别 | 类型 | 约束 |
  | --- | --- | --- |
  | Tensor | `tla.Tensor` | 普通 Tensor 和受支持的 Dynamic-GM 描述符均可作为字段 |
  | Python 标量 | `bool` / `int` / `float` | — |
  | `tla` 标量 | `Bool`、`Int8/16/32/64`、`UInt8/16/32/64`、`Float16/32`、`BFloat16` | — |
  | 编译期常量 | `tla.Constexpr[...]` | 不进入 kernel ABI / IR，且在 kernel 内只读 |
  | 嵌套结构 | tuple/list/NamedTuple/dataclass | 按有序叶子递归绑定 |
  | 空值 | `None` | 保留结构位置，不生成运行时 payload |

完整用法见[结构化 kernel 参数](../kernel_development/core_concepts/structured_arguments.md)。

调用示例：

```python
from dataclasses import dataclass
import catlass.tla as tla

@dataclass(frozen=True, kw_only=True)
class TilingData:
    TILE_M: tla.Constexpr[int]
    tiling_int: int
    out: tla.Tensor

@tla.kernel
def struct_arg_kernel(tiling: TilingData) -> None:
    # TILE_M 为编译期常量；tiling_int 为运行时标量。
    ...

tiling = TilingData(TILE_M=128, tiling_int=64, out=tout)
artifact = tla.compile(struct_arg_kernel, tiling, options="--npu-arch 3510")
artifact(tiling, block_num=1)
```

---

### `extern`

**源码：** [`catlass.tla.ffi.extern`](../../../catlass/tla/ffi.py#L70)

功能说明：

声明由内联 Ascend C 源实现的 Ascend C ABI 入口。被装饰的 Python 函数体
不会执行；其名字与注解描述的是在 TLA kernel 内调用时发出的符号与参数 ABI。

函数原型：

```python
tla.extern(*, source: str, name: str | None = None, include_dirs: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] = ()) -> Callable[[Callable[..., None]], ExternFunction]
```

参数说明：

- *`source`*（`str`）：非空的 Ascend C 翻译单元源，用于定义所声明的 C ABI 符号。
- *`name`*（`str | None`）：可选 C 符号名。默认使用被装饰的 Python 函数名。
- *`include_dirs`*（`str | os.PathLike[str] | Sequence[str | os.PathLike[str]]`）：
  可选的头文件搜索目录，或按序目录列表。相对路径相对声明所在文件解析。

约束说明：

- `name` 必须是合法 C 标识符，且 `source` 非空。
- 参数必须是位置参数、无默认值，并用
  `tla.Pointer[dtype, address_space]` 或具体 TLA 数值类型注解。
- 返回注解必须是 `None`。
- 调用必须落在恰好一个 `tla.cube()` 或 `tla.vector()` 区域内，且在
  `tla.vec.func()` 之外。
- 同一 kernel 内，一个符号只能对应一个 extern 声明对象；共享同一份
  `source` 的声明必须使用完全相同的有序 `include_dirs`。

调用示例：

```python
SOURCE = r'''
#include "kernel_operator.h"
extern "C" {
[aicore] __attribute__((always_inline)) void store_value(
    uint64_t dst_addr, int32_t value) {
  auto dst = reinterpret_cast<__gm__ int32_t *>(dst_addr);
  dst[0] = value;
}
}
'''

@tla.extern(source=SOURCE, include_dirs="include")
def store_value(
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    value: tla.Int32,
) -> None: ...

@tla.kernel
def kernel(dst: tla.Tensor) -> None:
    with tla.cube():
        store_value(dst.ptr, 42)
```

---

## 编译与启动

将装饰后的 kernel 编译为设备二进制并启动。同一份二进制需要多次启动时，用
`tla.compile` 获取可调用的 `JitCompiledFunction`；直接调用它时会延迟创建并复用内部
executor。缓存 / 架构 / IR dump 等非函数参数见
[环境变量](../kernel_development/core_concepts/env_vars.md)。

### 编译

生成设备二进制。日常入口是 `tla.compile`；
`TlaJitFunction.compile` 是装饰后函数上的底层辅助接口。

#### `compile`

**源码：** [`catlass.base_dsl.compiler.CompileCallable.__call__`](../../../catlass/base_dsl/compiler.py#L11)

功能说明：

编译 `@tla.kernel` 函数，返回可调用的
`JitCompiledFunction`。这是公开的 `tla.compile` 入口；调用返回的对象即可启动
（`compiled(*tensors, block_num=...)`）。适合编译一次、同一二进制多次启动。

函数原型：

```python
tla.compile(func: Any, *args: Any, **kwargs: Any) -> JitCompiledFunction
```

参数说明：

- *`func`*（`TlaJitFunction`）：被 `@tla.kernel` 装饰的函数。必填。
- *`args`*（`Any`）：作为编译类型样本的 Host tensor / 标量 / `@dataclass` 实例
  （如 `from_dlpack` 或 `make_fake_tensor` 的返回值）。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名，
  全部可用取值见[编译选项](#编译选项)。
  缓存 / IR dump / 强制重编译由 `CATLASS_DSL_*` 环境变量控制。

约束说明：

- `func` 必须是 `@tla.kernel` 得到的 `TlaJitFunction`。
- `args` 只作编译期类型样本，不必绑定 NPU 缓冲（`make_fake_tensor` 合法）。
- 用 `options="--npu-arch 3510"` 指定芯片名；不支持的取值在编译时报错。
- `block_num` / `stream` 等启动参数写在返回的编译函数上，而不是
  `tla.compile`。

调用示例：

```python
compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
compiled(tx, ty, block_num=1)  # 同一份二进制再次启动
```

---

#### `TlaJitFunction.compile`

**源码：** [`catlass.dsl.TlaJitFunction.compile`](../../../catlass/dsl.py#L198)

功能说明：

编译当前 `@tla.kernel` 函数并返回 `JitCompiledFunction`。
日常 Host 入口是 `tla.compile(fn, *args, options=...)`；只有已持有
`TlaJitFunction` 并需要直接获取编译函数所有者时才调用 `.compile()`。

函数原型：

```python
TlaJitFunction.compile(*, type_args: Sequence[Any] | None = None, **kwargs: Any) -> JitCompiledFunction
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：作为编译类型样本的 Host tensor / 标量。
  可选，默认 `None`（不做张量特化）。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名，
  全部可用取值见[编译选项](#编译选项)。
  缓存 / IR dump 由 `CATLASS_DSL_*` 环境变量控制。

约束说明：

- `type_args` 只作编译期类型样本，不必绑定 NPU 缓冲（`make_fake_tensor` 合法）。
- 用 `options="--npu-arch 3510"` 指定芯片名；不支持的取值在编译时报错。

调用示例：

```python
compiled = my_kernel.compile(
    type_args=[tx, ty],
    options="--npu-arch 3510",
)
```

---

#### 编译选项

`tla.compile` 与 `TlaJitFunction.compile` 的 `options` 接受一个字符串，按空格
分隔解析。可用选项如下：

| 选项 | 形式 | 说明 |
| --- | --- | --- |
| `--npu-arch <芯片名>` | 键值 | 指定目标芯片，例如 `3510`。 |
| `--cce-disable-asc-reserved-ubuf` | 开关 | 释放编译器为 Ascend C 预留的 2 KB Unified Buffer。 |
| `--cce-disable-vf-stack-reserved-ubuf` | 开关 | 释放编译器为 VF 栈预留的 6 KB Unified Buffer。 |

约束说明：

- 键值选项必须带取值，缺少取值时编译期报错。
- 开关选项不带取值：写出即生效，不需要再写 `1` 或 `true`。
- Unified Buffer 共 256 KB，编译器默认从中预留 8 KB（Ascend C 2 KB 与 VF 栈
  6 KB），kernel 只能用剩下的 248 KB。上面两个开关分别释放这两段，释放后
  kernel 可用的 Unified Buffer 相应变大。
- **`--cce-disable-vf-stack-reserved-ubuf` 有风险。** VF 栈是编译器溢出（spill）
  向量寄存器的去处；释放之后，若 kernel 触发溢出，编译器没有地方可写，而且
  没有任何检查会报错——结果是静默写坏 Unified Buffer 中相邻的数据。只在确认
  kernel 不会溢出时使用。
- `tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)` 返回整块 256 KB，
  而不是扣掉预留之后的大小。kernel 若按这个值划分缓冲又没有带上这两个开关，
  申请量就会超过实际可用，启动时被拒绝。

调用示例：

```python
compiled = tla.compile(
    matmul_add,
    ta, tb, tc,
    options=(
        "--npu-arch 3510"
        " --cce-disable-asc-reserved-ubuf"
        " --cce-disable-vf-stack-reserved-ubuf"
    ),
)
```

---

### 启动

在 NPU 上运行已编译的 kernel。直接调用 `tla.compile` 返回的
`JitCompiledFunction`。

#### `JitCompiledFunction.__call__`

**源码：** [`catlass.base_dsl.jit_executor.JitCompiledFunction.__call__`](../../../catlass/base_dsl/jit_executor.py#L614)

功能说明：

在 NPU 上启动已编译 kernel，传入运行时入参与 `block_num`、`stream` 等启动参数。
executor 和 binary 在首次调用时延迟加载，后续调用直接复用。

函数原型：

```python
JitCompiledFunction.__call__(*launch_args: Any, *, block_num: int | None = None, args: Sequence[Any] | None = None, **launch_kwargs: Any) -> TlaExecutionResult
```

参数说明：

- *`launch_args`*（`Any`）：位置形式的运行时 kernel 入参，与 `@tla.kernel`
  签名对应（已绑定的 Host tensor、标量或 `@dataclass` 实例）。与 `args=` 互斥。
- *`block_num`*（`int | None`）：启动的 block 数。可选，默认 `1`；传入时须为 `int`。
- *`args`*（`Sequence[Any] | None`）：显式运行时实参序列。可选，默认 `None`。
  不能与非空的 `*launch_args` 同时使用。
- *`stream`*（`Any`，经 `**launch_kwargs`）：可选 ACL stream 句柄。
  省略时使用执行器所在设备的当前 stream。
- *`ub`*（`int`，经 `**launch_kwargs`）：为 kernel 中 `tla.arch.get_dyn_ub`
  声明的动态 UB 区域实际提供的字节数，与 CuTe DSL 用 `launch(smem=...)`
  指定动态 shared memory 的方式对应。可选，默认 `0`。kernel 未声明该区域时
  传入会报错；超过 `artifact.max_ub_bytes` 也会报错。按 kernel 实际使用量
  申请即可：launch 只提供所申请的字节数，SIMT 场景下未被占用的 UB 会成为
  Data Cache，申请过多会拖慢访存。

约束说明：

- `*launch_args` 与 `args=` 不能同时非空（`TlaUnsupportedAbiError`）。
- tensor 启动实参须为已绑定的 NPU 缓冲（`from_dlpack`）；
  `make_fake_tensor` 仅用于编译样本。
- `block_num` 须为 `int`（默认 `1`）。

调用示例：

```python
compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
compiled(args=(tx, ty), block_num=1)
```

---

### 查看 IR

导出前端 TLA IR，不生成设备二进制，也不启动。

#### `TlaJitFunction.dump_mlir`

**源码：** [`catlass.dsl.TlaJitFunction.dump_mlir`](../../../catlass/dsl.py#L259)

功能说明：

返回该 kernel 的 TLA IR（`tlair`）MLIR 文本。不编译设备二进制，也不 launch。

函数原型：

```python
TlaJitFunction.dump_mlir(*, type_args: Sequence[Any] | None = None) -> str
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：类型样本，用法同 `.compile()`。
  可选，默认 `None`。

约束说明：

- `type_args` 规则与 `.compile()` 相同。
- 返回的是前端 TLA IR（`tlair`），不是
  `JitCompiledFunction.artifacts.LLVM` 中的 HIVM/LLVM 形式。

调用示例：

```python
text = my_kernel.dump_mlir(type_args=[fa, fb])
print(text[:500])
```

---

## Host Tensor

构造 Host 侧 `tla.Tensor`，并可将静态 layout 尺寸标为动态，使同一份编译产物可在不同 shape 下运行。详见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

### 创建与绑定

用 `from_dlpack` 绑定真实 NPU 缓冲，或用 `make_fake_tensor` 造仅含元数据的类型样本。

#### `from_dlpack`

**源码：** [`catlass.tla.runtime.from_dlpack`](../../../catlass/tla/runtime.py#L625)

功能说明：

将 DLPack NPU tensor **零拷贝**绑定为 TLA Host tensor。返回对象与 `tensor_dlpack` 共享同一块设备缓冲。

函数原型：

```python
tla.from_dlpack(tensor_dlpack: object, *, layout_tag: Any, origin_shape: Any | None = None, assumed_align: int | None = None, stream: int | None = -1, element_type: type | None = None) -> _Tensor
```

参数说明：

- *`tensor_dlpack`*（`object`）：实现了 `__dlpack__()` 的对象。须为 Ascend/NPU
  缓冲（如 `torch_npu`）。CPU / NumPy 不可用。必填。
- *`layout_tag`*（`tla.arch.*`）：布局标签，如 `tla.arch.RowMajor`、
  `tla.arch.ColumnMajor`、`tla.arch.zN`。必填。
- *`origin_shape`*（`tuple | int | None`）：逻辑 origin，Python int 树。
  可选；省略时由 DLPack 物理 shape 与 `layout_tag` 推导。不是 Kernel 的
  `tla.make_shape`。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。
- *`stream`*（`int | None`）：传给 `__dlpack__(stream=...)`。默认 `-1`（不做流同步）。
  `None` 表示省略 `stream` 参数。
- *`element_type`*（`type | None`）：可选。覆盖从 DLPack 推导出的元素类型；
  默认 `None` 表示沿用 DLPack。当 DLPack 无法表达真实类型时使用（例如 fp8），
  传入 `tla.Float8E4M3FN` / `Float8E5M2`。须与导出缓冲的每元素位宽一致。

约束说明：

- 所有权遵循 DLPack consumer 约定：capsule 会被消费，返回的 Host tensor
  销毁时调用 deleter；同时会保留对 `tensor_dlpack` 的引用，因此
  `from_dlpack(x.contiguous().to(device), ...)` 这类临时源是安全的。
- capsule 仅能消费一次；再次传入已消费的 capsule 会抛 `RuntimeTensorError`。
  需要再次绑定时请重新调用 `from_dlpack`。
- 二维 `RowMajor` 须先 `tensor.contiguous()`；二维 `ColumnMajor` 须先
  `tensor.permute(1, 0).contiguous()`。物理布局不符时抛 `RuntimeTensorError`。
  显式传入 `origin_shape` 则跳过该检查。
- 默认得到静态 layout。跨 shape 复用编译产物时再调用
  `mark_layout_dynamic` / `mark_compact_shape_dynamic`。
- 若指定 `element_type`，其每元素位宽须与导出的 DLPack 缓冲一致。

调用示例：

```python
tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(
    y.permute(1, 0).contiguous(),
    layout_tag=tla.arch.ColumnMajor,
)
```

---

#### `make_fake_tensor`

**源码：** [`catlass.tla.runtime.make_fake_tensor`](../../../catlass/tla/runtime.py#L887)

功能说明：

构造仅含元数据、不绑定设备缓冲的 Host tensor（`data_ptr == 0`）。
用于无需 NPU 时给 `tla.compile` 提供类型样本。真实缓冲请用 `from_dlpack`。

函数原型：

```python
tla.make_fake_tensor(dtype: Any, shape: Any, stride: Any, *, layout_tag: Any | None = None, addrspace: Any = AddressSpace.gm, origin_shape: Iterable[Any] | None = None, coord: Iterable[Any] | None = None, assumed_align: int | None = None) -> _Tensor
```

参数说明：

- *`dtype`*：元素类型，如 `tla.Float16` / `tla.Float32`。必填。
- *`shape`*（`int | tuple`）：逻辑 shape 树（zN 等物理布局用嵌套 tuple）。必填。
- *`stride`*（`int | tuple`）：stride 树，结构须与 `shape` 一致。必填。
- *`layout_tag`*：`tla.arch` 标签。可选，默认 `tla.arch.RowMajor`。
- *`addrspace`*：地址空间。可选，默认 `AddressSpace.gm`。
- *`origin_shape`*（`int | tuple | None`）：逻辑 origin。可选，默认等于 `shape`。
- *`coord`*（`int | tuple | None`）：坐标树。可选；省略时由 layout 推导（通常为零）。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。

约束说明：

- `shape` / `stride` / `origin_shape` / `coord` 须为 Python int 树，
  不能是 Kernel 侧 `tla.make_shape` / `tla.make_stride` / `tla.make_coord`。
- 始终未绑定，不能直接 launch；真实缓冲须改用 `from_dlpack`。
- 显式传入的 `shape` / `stride` 按原样使用（不做 layout remap）。

调用示例：

```python
fa = make_fake_tensor(tla.Float16, (128, 64), (64, 1))
fzn = make_fake_tensor(
    tla.Float16,
    ((16, 2), (16, 4)),
    ((16, 256), (1, 512)),
    layout_tag=tla.arch.zN,
    origin_shape=(32, 64),
)
```

---

### 动态 Layout

将静态 layout 尺寸标为动态。详见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

#### `Tensor.mark_layout_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_layout_dynamic`](../../../catlass/tla/runtime.py#L268)

功能说明：

将所有 shape 维标为动态，使一份 artifact 可接受不同 extents。
stride 除 leading 维（保持 `1`）外均变为动态；广播 stride `0` 保留。
对应的 `origin_shape` 叶节点也变为动态，编译类型不再依赖具体 DLPack 尺寸。

函数原型：

```python
tensor.mark_layout_dynamic(leading_dim: int | None = None) -> '_Tensor'
```

参数说明：

- *`leading_dim`*（`int | None`）：stride 为 `1` 的 leading 维索引。
  可选，默认 `None`（由 `layout_tag` 或紧凑 stride 顺序推断）。

约束说明：

- 原地修改并返回 `self`（可链式调用）。
- 各 `coord` 叶节点必须为 `0`；切片子视图会失败。
- `leading_dim` 对应维的 stride 必须为 `1`。
- NZFamily 布局下，每组两个物理 shape 叶节点对应一个逻辑 `origin_shape` 轴。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_layout_dynamic()
artifact = tla.compile(my_kernel, ta, options="--npu-arch 3510")
```

---

#### `Tensor.mark_compact_shape_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_compact_shape_dynamic`](../../../catlass/tla/runtime.py#L366)

功能说明：

将指定的一个紧凑 shape 维（`mode`）标为动态。以该维为因子的 major 维 stride
也会变为动态。对应的 `origin_shape` 叶节点同步标记，编译类型不再依赖具体尺寸。

函数原型：

```python
tensor.mark_compact_shape_dynamic(mode: int, stride_order: tuple[int, ...] | None = None) -> '_Tensor'
```

参数说明：

- *`mode`*（`int`）：要标记为动态的扁平 shape 叶节点索引（从 0 开始）。必填。
- *`stride_order`*（`tuple[int, ...] | None`）：紧凑 stride 顺序（外层 → 内层）。
  可选；省略时由当前 stride 推断。

约束说明：

- 原地修改并返回 `self`。
- 各 `coord` 叶节点必须为 `0`。
- `stride_order` 须为 `range(rank)` 的一个排列。
- NZFamily 布局下，物理维 0/1 对应逻辑 M，物理维 2/3 对应逻辑 N。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_compact_shape_dynamic(mode=0)
```

---
