---
nav_order: 10
---

# 语法约束

CATLASS Python 采用 Python 作为 kernel 描述语言，但受 NPU 架构、性能等因素限制，CATLASS Python 并不支持完整的 Python 语言特性。只有部分 Python 写法能够被映射为有意义的 NPU 指令。本文介绍这些限制。

## 基本概念

### `@tla.kernel` 与 `@tla.jit` 装饰器

并非所有Python代码都会被CATLASS Python翻译成NPU指令，只有被 `@tla.kernel` 或 `@tla.jit` 装饰的函数才会被前端处理，转换为对应的NPU指令。

- `@tla.kernel`：NPU kernel 入口，当前不支持直接调用。须先通过
  `compiled = tla.compile(kernel, *sample_args)` 显式编译，再通过
  `compiled(*runtime_args, block_num=...)` 启动。
- `@tla.jit`：kernel lowering 期间可调用的辅助函数，当前不支持调用或启动
  `@tla.kernel`。

两者装饰的Python函数遵循统一的降级——编译路径，以下的语法约束对两者同样适用。

### 值、分支与循环的动态与静态

编译、运行两个步骤将Python中的值、分支与循环区分为动态与静态。判断一个值、分支或循环是动态还是静态，看它的值或条件是否依赖运行时才确定的数值。

| 类别 | 判定 | 例子 |
| --- | --- | --- |
| 静态 | 编译期确定 | 字面量、`tla.Constexpr[T]` 参数、`tla.const_expr(...)` 的值 |
| 动态 | 运行时确定 | 张量元素、`tla.arch.block_idx()`、`tla.arch.block_num()`、动态循环的循环变量及其运算结果（运行时变量类型为 `tla.Int32`/`tla.Bool` 等 `tla.*` 标量类型） |

由此区分两类控制流：

- **动态分支/循环**：`if/while` 的条件、`for` 的边界依赖运行时变量时，编译为NPU分支/循环指令，在NPU上执行，每次运行可能走不同路径；受动态区域约束（不能提前退出、携带状态结构一致等）。
- **编译期分支/展开**：条件或边界编译期已知（`if True:`、`if tla.const_expr(...)`、边界为常数的 `for i in range(n)`），编译期直接选择一侧或**展开**循环体，不产生NPU分支/循环指令，不受动态区域约束，可参与编译期常量展开。

循环的动静判断：边界编译期已知用 `range`/`range_constexpr`（编译期展开）；边界依赖运行时变量用 `tla.range`（动态循环）。

### 编译期求值

kernel 中出现的普通 Python 函数（模块级函数、不在白名单里的内置函数）不会被编译成 NPU 指令。它们在翻译阶段按普通 Python 直接执行——参数都是编译期常数时，结果会代入后续代码（如 `for i in range(4)` 在翻译时直接展开成 4 条语句）；参数是运行时变量时，该调用不会产生 NPU 上的计算，不可用于设备操作。

需要显式声明"这个参数是编译期常数"时，用注解 `tla.Constexpr[T]`（如 `reverse: tla.Constexpr[bool]`）；`tla.const_expr(...)` 将条件标记为编译期可求值。

**示例：host 函数与内建 `range` 的编译期常量展开**

=== "原始代码"

    ```python
    @tla.jit
    def host_square(x: int) -> int:
        return x * x

    @tla.kernel
    def const_expand_kernel() -> None:
        n = host_square(3)      # 参数为字面量，编译期求值为 9
        for i in range(n):      # 内建 range，编译期展开
            tla.make_coord(i, 0)
    ```

=== "编译期展开后"

    ```python
    @tla.kernel
    def const_expand_kernel() -> None:
        tla.make_coord(0, 0)
        tla.make_coord(1, 0)
        tla.make_coord(2, 0)
        tla.make_coord(3, 0)
        tla.make_coord(4, 0)
        tla.make_coord(5, 0)
        tla.make_coord(6, 0)
        tla.make_coord(7, 0)
        tla.make_coord(8, 0)
    ```

### 编译期展开

需要小规模、有意的编译期展开时，使用 `tla.range_constexpr(...)`。边界须为编译期 Python 整数（也可来自 `tla.as_numeric(...)` 等编译期 Numeric 值）；Python 在构建 kernel 时执行循环并展开循环体。迭代次数达到 64 次及以上时，CATLASS 会在展开前发出一条 `DSLOptimizationWarning`，但**仍会继续展开**。不需要展开时应优先使用 `range(...)` 或 `tla.range(...)`。

```python
@tla.kernel
def static_stages(count: tla.Constexpr[int]) -> None:
    for stage in tla.range_constexpr(count):
        emit_stage(stage)
```

编译期 `while` 使用 `while tla.const_expr(condition):`，告警行为相同；条件写错可能导致编译无法终止，有界重复更推荐 `tla.range_constexpr(...)`。

### 规划中的语义变更

以下语义变更已在评审中确认，合入代码后本文档相应章节将同步更新。当前编写 kernel 时建议直接采用目标写法，避免后续迁移：

- **`range` 语义调整**：`range`（Python 内建）将由"编译期展开"改为执行期（动态）语义；未来 `tla.range == range`。
- **编译期展开显式化**：需要编译期展开的循环将统一显式写 `tla.range_constexpr`（见 2.1），不再依赖内建 `range`。
- **`@tla.jit` 要求**：进入 kernel 的普通 Python 函数将要求显式加 `@tla.jit` 装饰（见 1.3），不再允许裸模块级函数隐式参与。

## 控制流限制

### 三种 for

| 写法 | 语义 | 适用场景 |
| --- | --- | --- |
| `for i in tla.range(...)` | 动态循环，编译为NPU循环指令 | 边界为运行期确定的运行时变量 |
| `for i in range(...)` | 编译期展开（host 迭代） | 边界为编译期常数 |
| `for i in tla.range_constexpr(...)` | 编译期展开（显式声明） | 同上 |

约束：循环目标须为简单局部变量名；循环变量在循环结束后不可使用；`tla.range` 支持 `start`/`stop`/`step` 三参数与负步长。

### 动态 if 与 while

- 条件依赖运行时变量时，`if`/`while` 编译为NPU分支/循环；两个分支、循环体与条件区域都会生成NPU代码。
- 动态 if/while 体内的赋值目标受限，按出现位置区分：

| 语句 | 动态 if/while 体内 | 动态 for 体内 |
| --- | --- | --- |
| 局部名赋值 `x = v`、复合赋值 `x += 1` | 支持 | 支持 |
| 元组/列表解包 `(a, b) = v` | 支持 | 支持 |
| 张量下标写入 `out[i] = v` | 支持（切片除外） | 支持（切片除外） |
| 属性赋值 `obj.attr = v` | 不支持 | 不支持 |
| `del` | 不支持 | 不支持 |
| 调用 kernel 内定义的局部函数/闭包 | 不支持 | 不支持 |

状态变更统一通过局部名重新绑定实现，不依赖对象属性或容器元素变更；张量下标写入 `out[i] = v` 是例外（支持，切片赋值 `out[a:b] = v` 除外）。

注：动态 for 体内对属性赋值、`del` 等不做前端校验，缺少诊断不代表支持。

### 提前退出与 else

动态区域内 `return`/`break`/`continue`/`raise` 均不支持；动态 `for`/`while` 不支持 `else` 子句。NPU循环没有跳出指令与调用栈，提前退出改用条件跳过，或用标志变量记录结果、循环结束后读取。

### 变量作用域与携带状态

- 区域外使用的变量须在进入区域前初始化。
- 循环变量在循环结束后不可使用。
- 在分支或循环之间传递的变量（携带变量），各路径赋值的结构与类型须一致；支持的容器为 tuple/list/dict/dataclass 等。

### 编译期分支

条件为字面量 `True`/`False` 或 `tla.const_expr(...)` 时，`if` 在编译期选择分支，不生成NPU分支；两侧分支体仍会被检查。配合 `tla.Constexpr[T]` 参数用于编译期分发。

### with 区域

NPU 中存在cube核与vector核，在Python中被抽象为两个可使用`with`进入的区域。

- **Cube 区域**：用 `with tla.cube(...)` 进入，承载cube核上的运算。
- **Vector 区域**：用 `with tla.vector(...)` 进入，承载vector核上的运算。
  - **Vector SIMD VF 子区域**： 用 `tla.vec.func(mode="simd")` 进入，承载[Reg Vector 计算](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/%E6%A6%82%E8%BF%B0.md)子函数。

约束：

- 一个 `with` 仅允许一个上下文管理器；多个上下文管理器时不报错，但区域不会生成。
- 区域内定义的临时值不可在区域外使用。
- 区域内对外层变量的重新绑定由前端自动处理（生成 `nonlocal`），无需手写。

## 常用内容速查

### 表达式运算符支持

| 运算符 | 支持情况 |
| --- | --- |
| `+` `-` `*` `/` `//` `%`、一元负号 `-x` | 支持 |
| 位运算 `& \| ^ ~ << >>` | 支持 |
| 比较 `== != < <= > >=` | 支持 |
| `is`/`is not`/`in`/`not in` | 部分支持（仅适用于 host 值） |
| `and or not`（短路）、条件表达式 `x if c else y` | 支持 |
| `**`（幂） | 部分支持（仅浮点类型） |
| 下标读取 `meta[i]` | 部分支持：张量/元组等运行时变量支持动态下标；**Python list 不支持动态下标**（下标为NPU上的数值时），仅支持编译期常数下标（由 `range`/`tla.range`/`tla.range_constexpr` 创建的列表表达式同理） |

### 内置关键字支持

Python 的 36 个内置关键字全部列出如下，按功能分类。两个支持列的通用约定（动静定义见 1.2）：

- **静态支持**＝按 Python（host）语义在编译期求值，不生成NPU指令；
- **动态支持**＝作用于运行时变量，降级为NPU指令；
- "部分支持"表示存在例外，"不支持"表示该用法会报错或没有NPU语义，具体见说明列；"—"表示该关键字没有对应的动静用法。

#### 字面量

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `True`/`False` | 支持 | — | Python 布尔字面量，降级为布尔常量。作为 `if` 条件时属于编译期分支（见 2.5），不生成NPU分支指令 |
| `None` | 部分支持 | 不支持 | 没有对应的NPU类型，不能作为标量参与计算。仅在编译期有占位语义：张量 `data_ptr=None` 视为未绑定（归一为 `0`）、索引中的动态维度标记、作为内核实参时跳过类型推断 |

#### 逻辑运算

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `and`/`or` | 支持 | 支持 | 与 Python 一致支持短路求值：`a and b` 等价于 `b if a else a`，`a or b` 等价于 `a if a else b`。操作数为动态值时惰性计算，只执行实际需要的一侧；结果类型是操作数的类型，不一定是布尔 |
| `not` | 支持 | 支持 | 逻辑取反，结果恒为布尔（`Bool`） |

#### 比较与成员

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `is`/`is not` | 支持 | 不支持 | 仅适用于 host 值，按 Python 身份比较（`left is right`）求值；运行时变量上没有"身份"概念，不生成NPU指令 |
| `in`/`not in` | 支持 | 不支持 | 仅适用于 host 值，按 Python 成员测试求值；运行时变量上的成员判断需改用其他方式。注意 `for x in ...` 的迭代语义见"循环"分类 |

#### 分支

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `if`/`elif`/`else` | 支持 | 支持 | 条件编译期已知（字面量、`tla.Constexpr[T]` 参数、`tla.const_expr(...)`）时编译期直接选择一侧，不生成NPU分支（两侧分支体仍会被检查）；条件依赖运行时变量时编译为NPU分支。动态分支内赋值的约束见 2.2 |
| `match`/`case`（软关键字） | 不支持 | 不支持 | 模式匹配不在支持范围内（动态区域内会报错） |

#### 循环

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `for` | 支持 | 支持 | 边界为编译期常数时用 `range`/`tla.range_constexpr` 在编译期展开循环体；边界依赖运行时变量时用 `tla.range` 编译为NPU循环。循环目标须为简单局部变量名，循环变量在循环结束后不可使用 |
| `while` | 支持 | 支持 | 条件编译期已知时按 host 循环执行（不生成NPU指令）；条件依赖运行时变量时编译为NPU循环。动态 while 同样受携带状态、不能提前退出等约束 |
| `break`/`continue` | 支持 | 不支持 | 只在编译期展开的 host 循环内有效（作用于展开过程）；动态循环没有跳出指令，会报错，提前退出请改用条件跳过或标志变量（见 2.3） |
| `else`（`for`/`while` 的 else） | 支持 | 不支持 | `if` 的 `else` 见"分支"分类；`for`/`while` 的 `else` 仅静态循环可用（按 host 语义），动态循环报 for-else/while-else 错误 |

#### 函数与作用域

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `def` | 支持 | 不支持 | 支持 `@tla.kernel`/`@tla.jit` 根函数与模块级函数；kernel 内定义的局部函数不能从动态区域调用（会报错），请将公共逻辑提取到模块顶层 |
| `lambda` | 支持 | 不支持 | 仅按普通 Python 求值，不生成NPU指令；动态条件内不允许使用 |
| `return` | 支持 | 不支持 | kernel 函数体的正常返回可用；动态区域内不允许提前返回（无调用栈、无跳出指令），需用条件跳过改写 |
| `yield`/`yield from` | 不支持 | 不支持 | 生成器语义不在支持范围内 |
| `async`/`await` | 不支持 | 不支持 | 异步语义不在支持范围内 |
| `global`/`nonlocal` | 不支持 | 不支持 | 手写没有NPU语义；`with` 区域对外层变量的重绑定由前端自动生成 `nonlocal`，无需手写（见 2.6） |

#### 区域与上下文

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `with` | 支持 | — | 用于进入NPU区域：`tla.cube`（cube 核）、`tla.vector`/`tla.vec.func`（vector 核）。区域结构编译期确定，一个 `with` 仅一个上下文管理器；区域内的临时值不可在区域外使用 |
| `as` | 支持 | 不支持 | 随 `with ... as x`（目标须为局部变量名）或 `import ... as` 使用；`except ... as` 因 `except` 不支持而不可用 |

#### 模块与类

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `import`/`from` | 支持 | 不支持 | 模块级 `import` 可用，编译期（宿主进程内）加载，不在NPU侧执行；`from x import y` 同理 |
| `class` | 部分支持 | 不支持 | kernel 内嵌套类定义不支持；模块级 dataclass 可作为动态区域的携带状态容器（见 2.4） |

#### 异常与断言

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `try`/`except`/`finally` | 部分支持 | 不支持 | 动态区域内报错；区域外按 host 语义编译期执行，结果不可依赖，不建议用异常做流程控制 |
| `raise` | 部分支持 | 不支持 | 区域外按 host 语义执行（命中即编译期抛错）；动态区域内与 `return`/`break`/`continue` 一并禁止 |
| `assert` | 部分支持 | 不支持 | 编译期按 host 求值，失败直接导致编译报错；作用于运行时变量时没有NPU语义 |

#### 变量与内存

| 关键字 | 静态支持 | 动态支持 | 说明 |
| --- | --- | --- | --- |
| `pass` | 支持 | 支持 | 空操作，不产生任何指令，可用于占位 |
| `del` | 部分支持 | 不支持 | 动态 if/while/for 体内报 "does not support deletion"；区域外按 host 语义执行（删除编译期局部变量） |

### 内置函数支持

对内置函数的支持可以看成一种白名单：只有下表列出的内置函数会进入 kernel 编译，其余不进入编译、按上面的 host 函数规则在降级时求值。完整清单与语义见 [Python 内置函数官方文档](https://docs.python.org/zh-cn/3.13/library/functions.html#import__)。

| 进入编译的内置函数 | 支持情况 | 进入编译的方式 | 参考 |
| --- | --- | --- | --- |
| `any()`/`all()`/`bool()` | 部分支持（要求单参数且无关键字） | 重定向到NPU实现 | `ast_preprocessor.py:35-41`、`tla_ast_decorators.py:533-550` |
| `min()`/`max()` | 部分支持（要求无关键字） | 重定向到NPU实现 | `ast_preprocessor.py:35-41`、`tla_ast_decorators.py:559-567` |
| `abs(x)` | 支持 | 经 `__abs__` 生成NPU运算 | `typing.py:804-821` |
| `pow(x, y)` | 部分支持（仅浮点，同 `**`） | 经 `__pow__` 生成NPU运算 | `typing.py:861` |
| `range()` | 部分支持（边界须为编译期常数） | 参与编译期循环展开 | `ast_preprocessor.py:305-474` |

## 实例

示例取自 `tla/kernel` 与 `dsl_tutorials` 并精简，完整可运行版本见对应文件。

### 动态 if：分支内改变量须先初始化

动态 `if` 中，分支外要使用的变量须先初始化、再在分支内重新赋值。

```python
@tla.kernel
def dynamic_if_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # 先初始化
        if i == 0:
            coord = i + 1             # 分支内重新赋值
        else:
            coord = i + 2             # 另一分支同样赋值
        tla.make_coord(coord, 0)      # 分支结束后使用
```

错误写法：

```python
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1             # 仅在分支内定义
        tla.make_coord(coord, 0)      # SyntaxError: ... must be initialized before the if
```

### 动态 if：分支间传值

各分支给同一组变量赋不同值，分支结束后统一使用。

```python
@tla.kernel
def select_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # 先初始化
        offset = i + 1
        if i == 0:
            coord = i + 2             # 各分支给同一组变量赋不同值
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset) # 分支结束后使用
```

### 动态 for：携带状态跨迭代

动态循环不支持提前退出；循环内重新赋值的变量在迭代间自动传递，各次赋值的类型须一致。

```python
@tla.kernel
def carried_state_kernel(mem_src: tla.Tensor, mem_out: tla.Tensor) -> None:
    with tla.vector():
        src_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        out_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        src_ub = tla.make_tensor_like(src_ptr, mem_src, tla.arch.RowMajor)
        out_ub = tla.make_tensor_like(out_ptr, mem_out, tla.arch.RowMajor)
        tla.copy(src_ub, mem_src)
        with tla.vec.func(mode="simd"):
            value = src_ub.load()     # 先初始化
            for _ in tla.range(2):    # 动态循环
                value = tla.abs(value)  # 状态跨迭代传递
            out_ub.store(value)
        tla.copy(mem_out, out_ub)
```

错误写法：

```python
    with tla.vec.func(mode="simd"):
        for _ in tla.range(2):
            break                     # SyntaxError: ... does not support return, break, continue, or raise
```

### 动态 for：按分块循环

按 tile 分块循环，每次迭代处理一个分块。

```python
@tla.kernel
def tile_loop_kernel(mem: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        src_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        src_ub = tla.make_tensor_like(src_ptr, mem, tla.arch.RowMajor)
        dst_ub = tla.make_tensor_like(dst_ptr, out, tla.arch.RowMajor)
        tla.copy(src_ub, mem)
        with tla.vec.func(mode="simd"):
            for i in tla.range(0, 4, 1):              # 按分块循环
                src_tile = tla.tile_view(src_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile = tla.tile_view(dst_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile.store(tla.abs(src_tile.load()))
        tla.copy(out, dst_ub)
```

### 编译期分支

条件编译期已知时，编译期直接选择一侧，不生成NPU分支。

```python
@tla.kernel
def constexpr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):          # 编译期分支
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)
```

### with 区域

NPU操作放在 `tla.vector()` 区域内，vector 计算包在 `tla.vec.func(mode="simd")` 中。

```python
@tla.kernel
def vec_region_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    with tla.vector():
        a_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        b_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        c_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        a_ub = tla.make_tensor_like(a_ptr, mem_a, tla.arch.RowMajor)
        b_ub = tla.make_tensor_like(b_ptr, mem_b, tla.arch.RowMajor)
        c_ub = tla.make_tensor_like(c_ptr, mem_c, tla.arch.RowMajor)
        tla.copy(a_ub, mem_a)
        tla.copy(b_ub, mem_b)
        with tla.vec.func(mode="simd"):
            c_ub.store(a_ub.load() + b_ub.load())
        tla.copy(mem_c, c_ub)
        tla.pipe_barrier(tla.pipes.ALL)
```

### struct-like args

用`@dataclass`创建的类实例可以作为kernel入参，也可以在kernel内创建。

```python
from __future__ import annotations
from dataclasses import dataclass
import catlass.tla as tla

@dataclass(frozen=True, kw_only=True)
class TilingData:
    TILE_M: tla.Constexpr[int]    # 编译期常量，不进 ABI,生成的IR不占用入参
    tiling_gm_out: tla.Tensor     # 支持tla.Tensor入参
    tiling_int16: tla.Int16       # Int16 标量
    tiling_float: tla.Float32     # Float32 标量
    tiling_int: int               # Python int, 编译期为int，运行期为tla.Int32

@dataclass(frozen=True)
class Info:
    tile_m: int
    tile_n: int

def print_tiling(tiling: TilingData):
    tla.print("tiling_int={}", tiling.tiling_int)

@tla.kernel
def struct_arg_kernel(tiling: TilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_int16
    ptr = tla.allocate(tiling.TILE_M, ...)
    print_tiling(tiling)    # 在kernel内作为其他函数的参数使用，跟普通变量一致
    info = Info(tile_m=tiling.tiling_int, ...)  # 在kernel侧实例化


tiling = TilingData(TILE_M=128, ...)
artifact = tla.compile(struct_arg_kernel, tiling, out)
artifact(tiling, out)
```

**实例使用范围**:

- 在host侧创建实例，作为kernel入参，字段类型支持标量和tensor
- 在kernel侧创建实例

**dataclass 允许自定义的选项**：仅 `frozen/kw_only`，其他参数不支持修改为非默认值。`frozen=True`实例化后不支持修改字段值，`kw_only=True`时实例化时必须显式指定key，例如`tiling=Tiling(TILE_M=128, tiling_int16=128, ...)`，不支持`tiling=TilingData(128, 128, ...)`

**`tla.Constexpr[...]` 字段**：

- 视作编译期常量：kernel内只读，可用于`tla.allocate`、`tla.range_constexpr`等。
- 不进入 kernel ABI / IR：`tla.func` 无对应块参数。

**入参支持的类型**：

- `tla.*` 标量：`tla.Bool`、`tla.Int8/16/32/64`、`tla.UInt8/16/32/64`、`tla.Float16/32`、`tla.BFloat16`。
- 纯 Python 标量：`bool`→`i1`、`int`→`i32`、`float`→`f32`。
- `tla.Tensor`。

**注意**：字段实际类型由**字段值**决定（与 Python 动态语义一致），构造 dataclass 时不会强制转换成注解类型。

### 错误信息速查

| 错误信息（关键词） | 原因 | 处理 |
| --- | --- | --- |
| `does not support return, break, continue, or raise` | 动态区域内提前退出 | 条件跳过，或用标志变量记录结果 |
| `does not support for-else`/`while-else` | 动态循环带 else | 移除 else，逻辑移入循环体 |
| `requires a simple local name target` | for 目标为 `pair[0]` 等 | 使用单个局部变量名 |
| `induction variables cannot be used after the loop` | 循环结束后使用循环变量 | 循环内将所需值存入预先初始化的变量 |
| `must be initialized before the if/loop` | 变量仅在分支或循环内定义，区域外使用 | 在区域前初始化 |
| `only supports assignments to local names or tuples/lists` | if/while 体内写 `out[i] =`、`obj.attr =` 等 | 存储移出 if/while 体，或用局部名重绑 |
| `calling active local callable` | 动态区域内调用局部函数 | 内联，或移到模块顶层 |
| `structure`/`expected i32` | 各分支或迭代间变量结构或类型不一致 | 各路径保持结构与类型一致 |
| `'**' is only supported for float types` | 整型幂运算 | 使用乘法 |

## 参考资料

- 控制流使用示例：[`scalar_index_control_flow.py`](../../../../../../examples/python/tla_dsl/dsl_tutorials/tensor_index/scalar_index_control_flow.py)、[`basic_vadd.py`](../../../../../../examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py)、[`register_control_flow.py`](../../../../../../examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/register_control_flow.py)
- 控制流 AST 实现：[`ast_preprocessor.py`](../../../../catlass/base_dsl/ast_preprocessor.py)、[`tla_ast_decorators.py`](../../../../catlass/tla_ast_decorators.py)
- 控制流测试用例参考：[`test_frontend_branching.py`](../../../../tests/test_frontend_branching.py)、[`test_frontend_for_range.py`](../../../../tests/test_frontend_for_range.py)、[`test_frontend_while.py`](../../../../tests/test_frontend_while.py)、[`test_frontend_with.py`](../../../../tests/test_frontend_with.py)
