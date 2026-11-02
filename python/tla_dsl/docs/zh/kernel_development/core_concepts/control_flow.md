---
nav_order: 20
---

# 控制流

TLA kernel 函数体由 Python 在前端构建设备 IR 时执行。这使得普通、可检查的 Python staging 代码可用于在操作降级之前推导静态配置。

## Python staging

Kernel 与 `@tla.jit` 函数可以调用裸 Python 函数、lambda、嵌套函数、闭包、类构造器、property 与绑定方法，只要它们的值保持为 Python 编译期值。裸辅助函数在调用方活跃的前端上下文中按普通 Python 执行，因此它的 TLA 操作会直接产生 IR，无需仅为了调用 TLA API 而加 `@tla.jit`。

裸辅助函数永远不会被独立变换，因此它自身的 TLA 运行时控制流不受支持：请把此类控制流保留在带装饰器的调用方中。装饰辅助函数的组合留给后续控制流更新。

```python
class TileConfig:
    def __init__(self, width: int):
        self.width = width

    @property
    def half_width(self) -> int:
        return self.width // 2


@tla.kernel
def kernel(limit: int) -> None:
    config = TileConfig(64)
    offset = (lambda value: value + 1)(config.half_width)
    tla.make_coord(offset, limit)
```

## 运行时控制流与提升

由 Python 值支撑的条件与循环边界会形成运行时控制流。在此类区域内写入的值会成为运行时状态。在运行时 `if`、`tla.range` 循环、`while` 或展开的 Python `with` 区域中赋值之前，请使用 `tla.as_numeric(...)` 显式提升 Python 标量：

```python
@tla.kernel
def kernel(limit: int) -> None:
    index = tla.as_numeric(0)
    if limit > 0:
        index = index + 1
    tla.make_coord(index, 0)
```

用对字面量 list/tuple 元素的推导式组合固定运行时集合，并将每个携带值用 `tla.as_numeric` 包裹：

```python
state = [tla.as_numeric(value) for value in (0, 1)]
```

容器字面量、部分提升的集合与动态集合迭代对象不是运行时提升形式。

在运行时控制流内被更新之后，再读取编译期绑定是不受支持的。请保留提升后的值，或在运行时区域内计算后续使用。

## 对象边界

当 dataclass 字段本身是受支持的运行时值时，dataclass 值可以在运行时控制流中构造、携带与重新绑定。其他用户定义对象仅用于 staging：它们不能作为 kernel ABI 参数、不能传给 `tla.as_numeric(...)`、也不能在运行时控制流中被修改。设备可见的修改请使用 tensor 存储，且只提升受支持的标量值或用受支持的推导式组装的固定集合。

## `@tla.jit`

`@tla.jit` 是 kernel lowering 期间使用的 Python 辅助函数，可以从
`@tla.kernel` 中调用，但当前不支持调用或启动 `@tla.kernel`。当前不支持直接调用
`@tla.kernel`；须先通过 `tla.compile(kernel, *sample_args)` 显式编译，
再调用返回的编译函数。
