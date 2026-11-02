---
nav_order: 1
---

<!--
手工维护的中文 程序生命周期参考。
英文源：docs/en/api/runtime/kernel_entry.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# Kernel 入口与语言边界

[程序生命周期](index.md) | [API 速查](../quick_reference.md)

Host 侧 `@tla.kernel` 入口、`@tla.jit` device helper、`@tla.extern` Ascend C 复用，以及 Host 侧 `@dataclass` 打包。
kernel 与 extern 声明体不在 Host 端作为普通 Python 执行。`@tla.jit` helper 在
kernel lowering 期间内联；在 Host 顶层调用时按普通 Python 执行。

## `dataclasses.dataclass`

**源码：** [`dataclasses.dataclass`](../../../../catlass/base_dsl/runtime/jit_arg_adapters.py#L135)

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

完整用法见[结构化 kernel 参数](../../kernel_development/core_concepts/structured_arguments.md)。

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

## `tla.kernel`

**源码：** [`catlass.dsl.kernel`](../../../../catlass/catlass_dsl/catlass.py#L27)

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
- 详细设计和限制说明见 [AutoSync 设计](../../dsl_development/feature_development/auto_sync_design.md)。
- 启动前必须调用 `tla.compile(kernel, *sample_args)`；直接调用装饰后的
  kernel 会抛出 `TypeError`。

**Kernel 入参类型**

| 类别 | 类型 | 启动时传入 |
| --- | --- | --- |
| Tensor | `tla.Tensor` | 是 |
| Python 标量 | `bool` / `int` / `float` | 是 |
| `tla` 标量 | `Bool`、`Int8/16/32/64`、`UInt8/16/32/64`、`Float16/32`、`BFloat16` | 是 |
| 编译期常量 | `tla.Constexpr[...]` | 否 |
| 编译期函数 | `tla.Constexpr[Callable[...]]` 或 `tla.Constexpr` | 否；见 [`Constexpr Callable arguments`](#constexpr-callable-arguments) |
| 结构体 | 字段类型属于上表的 `@dataclass` 实例 | 按字段展开；Constexpr 字段 launch 时不需要传入 |

调用示例：

```python
@tla.kernel
def copy_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)

@tla.kernel(auto_sync="v0")
def copy_kernel_auto(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)

compiled = tla.compile(copy_kernel, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
```

---

## `tla.jit`

**源码：** [`catlass.dsl.jit`](../../../../catlass/catlass_dsl/catlass.py#L114)

功能说明：

将 Python 函数标注为 device 侧 Python helper。

- 在 `@tla.kernel` 降级过程中被调用时，函数体内联进该 kernel 的 device IR。
- 可作为 [`Constexpr Callable arguments`](#constexpr-callable-arguments) kernel 入参；也可在 kernel 内按名调用。
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

## Constexpr Callable arguments

对应 [`kernel`](#tlakernel) 入参表中「编译期函数」一行（`tla.Constexpr[Callable[...]]`，或实参为可调用对象的 `tla.Constexpr`）；不含「编译期常量」。

传递形态：外层 `def`、`lambda`、`functools.partial`，或 `@tla.jit` 装饰的函数。
在 `tla.compile` 时传入，`compiled(...)` 启动时不再传入；不同函数对象对应不同特化。

函数体语义（普通 `def` / `lambda` / `partial`）：

- 只在 `tla.compile` / `dump_mlir` 时执行；其中的 Python 操作写入当前 kernel 的 device IR。`compiled(...)` 启动时不再执行该函数体。
- 在 kernel 内调用时，函数体只能使用 [语言参考](../language/index.md) 中的接口，具体约束见对应条目。
- 任意 Host 侧 Python 不构成设备计算语义，包括但不限于第三方库、文件/网络 I/O、对本机状态的依赖，以及把 Python 值当 Host 张量或容器使用。
- 不支持 TLA 控制流：`tla.range`、动态 `if` / `while` 等。

`@tla.jit` 作入参时的函数体约束见 [`jit`](#tlajit)。

**Kernel 体内可调用**

- 外层普通 `def`：同本节「函数体语义」。
- `@tla.jit` helper：见 [`jit`](#tlajit)；降级时内联进当前 kernel IR。
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

## `tla.extern`

**源码：** [`catlass.tla.ffi.extern`](../../../../catlass/tla/ffi.py#L70)

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
