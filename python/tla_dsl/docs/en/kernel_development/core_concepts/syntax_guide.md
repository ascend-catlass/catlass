---
nav_order: 10
---

# Syntax Constraints

CATLASS Python uses Python as the kernel description language, but due to NPU architecture, performance considerations, and other factors, CATLASS Python does not support the full set of Python language features. Only part of the Python code can be mapped to meaningful NPU instructions. This document introduces these restrictions.

## Basic Concepts

### `@tla.kernel` and `@tla.jit` decorators

Not all Python code is translated by CATLASS Python into NPU instructions. Only functions decorated with `@tla.kernel` or `@tla.jit` are processed by the frontend and converted into the corresponding NPU instructions.

- `@tla.kernel`: the NPU kernel entry point; direct invocation is currently unsupported. It must first be compiled explicitly with `compiled = tla.compile(kernel, *sample_args)`, then launched with `compiled(*runtime_args, block_num=...)`.
- `@tla.jit`: a helper function that can be called during kernel lowering; calling or launching an `@tla.kernel` from it is currently unsupported.

Python functions decorated with either follow a unified lowering/compilation path, so the syntax constraints below apply equally to both.

### Dynamic and static values, branches, and loops

The compile and run steps divide values, branches, and loops in CATLASS Python into dynamic and static. Whether a value, branch, or loop is dynamic or static depends on whether its value or condition depends on a number determined only at runtime.

| Category | Determination | Examples |
| --- | --- | --- |
| Static | Determined at compile time | literals, `tla.Constexpr[T]` parameters, the value of `tla.const_expr(...)` |
| Dynamic | Determined at runtime | tensor elements, `tla.arch.block_idx()`, `tla.arch.block_num()`, the induction variable of a dynamic loop and its computed results (runtime variable types are `tla.*` scalar types such as `tla.Int32`/`tla.Bool`) |

This distinguishes two kinds of control flow:

- **Dynamic branch/loop**: when the condition of `if/while` or the bounds of `for` depend on a runtime variable, it compiles to NPU branch/loop instructions and executes on the NPU, potentially taking a different path on each run; it is subject to dynamic-region constraints (no early exit, consistent carried-state structure, etc.).
- **Compile-time branch/expansion**: when the condition or bounds are known at compile time (`if True:`, `if tla.const_expr(...)`, `for i in range(n)` with constant bounds), the compile time directly selects one side or **expands** the loop body, generating no NPU branch/loop instructions, is not subject to dynamic-region constraints, and can participate in compile-time constant expansion.

Determining whether a loop is dynamic or static: when the bounds are known at compile time, use `range`/`range_constexpr` (compile-time expansion); when the bounds depend on a runtime variable, use `tla.range` (dynamic loop).

### Compile-time evaluation

Ordinary Python functions that appear in a kernel (module-level functions, built-in functions not on the whitelist) are not compiled into NPU instructions. They are executed directly as ordinary Python during the translation stage: when the arguments are all compile-time constants, the result is substituted into the subsequent code (for example, `for i in range(4)` is expanded directly into 4 statements at translation time); when an argument is a runtime variable, the call produces no computation on the NPU and cannot be used for device operations.

When you need to explicitly declare that "this argument is a compile-time constant", use the annotation `tla.Constexpr[T]` (such as `reverse: tla.Constexpr[bool]`); `tla.const_expr(...)` marks a condition as compile-time evaluable.

**Example: compile-time constant expansion of a host function and the built-in `range`**

=== "Original code"

    ```python
    @tla.jit
    def host_square(x: int) -> int:
        return x * x

    @tla.kernel
    def const_expand_kernel() -> None:
        n = host_square(3)      # the argument is a literal, evaluated to 9 at compile time
        for i in range(n):      # built-in range, expanded at compile time
            tla.make_coord(i, 0)
    ```

=== "After compile-time expansion"

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

### Compile-time expansion

When a small, intentional compile-time expansion is needed, use `tla.range_constexpr(...)`. The bounds must be compile-time Python integers (they may also come from compile-time Numeric values such as `tla.as_numeric(...)`); Python executes the loop while building the kernel and expands the loop body. When the iteration count reaches 64 or more, CATLASS emits one `DSLOptimizationWarning` before expansion but **still continues expanding**. When expansion is not needed, prefer `range(...)` or `tla.range(...)`.

```python
@tla.kernel
def static_stages(count: tla.Constexpr[int]) -> None:
    for stage in tla.range_constexpr(count):
        emit_stage(stage)
```

A compile-time `while` uses `while tla.const_expr(condition):`, with the same warning behavior; an incorrect condition may make compilation unable to terminate, so `tla.range_constexpr(...)` is more recommended for bounded repetition.

### Planned semantic changes

The following semantic changes have been confirmed in review, and the corresponding sections of this document will be updated when the code lands. When writing kernels now, it is recommended to adopt the target form directly to avoid later migration:

- **`range` semantics adjustment**: `range` (the Python built-in) will change from "compile-time expansion" to execution-time (dynamic) semantics; in the future `tla.range == range`.
- **Explicit compile-time expansion**: loops that need compile-time expansion will uniformly be written explicitly as `tla.range_constexpr` (see 2.1), no longer relying on the built-in `range`.
- **`@tla.jit` requirement**: ordinary Python functions entering a kernel will be required to be explicitly decorated with `@tla.jit` (see 1.3), no longer allowing bare module-level functions to participate implicitly.

## Control-Flow Restrictions

### The three kinds of for

| Form | Semantics | Applicable scenario |
| --- | --- | --- |
| `for i in tla.range(...)` | dynamic loop, compiled to NPU loop instructions | bounds are runtime variables determined at run time |
| `for i in range(...)` | compile-time expansion (host iteration) | bounds are compile-time constants |
| `for i in tla.range_constexpr(...)` | compile-time expansion (explicitly declared) | same as above |

Constraints: the loop target must be a simple local variable name; the induction variable cannot be used after the loop; `tla.range` supports the three parameters `start`/`stop`/`step` and negative steps.

### Dynamic if and while

- When the condition depends on a runtime variable, `if`/`while` compiles to NPU branches/loops; both branches, the loop body, and the condition region generate NPU code.
- Assignment targets inside a dynamic if/while body are restricted, distinguished by where they appear:

| Statement | Inside a dynamic if/while body | Inside a dynamic for body |
| --- | --- | --- |
| Local name assignment `x = v`, augmented assignment `x += 1` | Supported | Supported |
| Tuple/list unpacking `(a, b) = v` | Supported | Supported |
| Tensor subscript write `out[i] = v` | Supported (except slices) | Supported (except slices) |
| Attribute assignment `obj.attr = v` | Not supported | Not supported |
| `del` | Not supported | Not supported |
| Calling a local function/closure defined in the kernel | Not supported | Not supported |

State changes are realized uniformly by rebinding local names, not by depending on object attribute or container element changes; tensor subscript writes `out[i] = v` are the exception (supported, except slice assignment `out[a:b] = v`).

Note: attribute assignment, `del`, and the like inside a dynamic for body are not frontend-validated; the absence of a diagnostic does not mean they are supported.

### Early exit and else

Inside a dynamic region, `return`/`break`/`continue`/`raise` are all unsupported; dynamic `for`/`while` do not support an `else` clause. NPU loops have no jump-out instruction and no call stack, so rewrite early exit as a conditional skip, or record the result in a flag variable and read it after the loop.

### Variable scope and carried state

- Variables used outside a region must be initialized before entering the region.
- The induction variable cannot be used after the loop.
- For variables passed between branches or loop iterations (carried variables), the structure and type assigned on each path must be consistent; supported containers are tuple/list/dict/dataclass, etc.

### Compile-time branches

When the condition is the literal `True`/`False` or `tla.const_expr(...)`, `if` selects the branch at compile time and generates no NPU branch; both branch bodies are still checked. It is used together with `tla.Constexpr[T]` parameters for compile-time dispatch.

### with regions

The NPU has a cube core and a vector core, abstracted as two regions that can be entered with `with`.

- **Cube region**: entered with `with tla.cube(...)`, hosting operations on the cube core.
- **Vector region**: entered with `with tla.vector(...)`, hosting operations on the vector core.
  - **Vector SIMD VF sub-region**: entered with `tla.vec.func(mode="simd")`, hosting [Reg Vector computation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/%E6%A6%82%E8%BF%B0.md) sub-functions.

Constraints:

- A single `with` allows only one context manager; with multiple context managers, no error is reported, but the region is not generated.
- Temporary values defined inside a region cannot be used outside the region.
- Rebinding an outer variable inside a region is handled automatically by the frontend (generating `nonlocal`) and does not need to be written by hand.

## Quick Reference

### Expression operator support

| Operator | Support |
| --- | --- |
| `+` `-` `*` `/` `//` `%`, unary minus `-x` | Supported |
| Bitwise `& \| ^ ~ << >>` | Supported |
| Comparison `== != < <= > >=` | Supported |
| `is`/`is not`/`in`/`not in` | Partially supported (only applies to host values) |
| `and or not` (short-circuit), conditional expression `x if c else y` | Supported |
| `**` (power) | Partially supported (float types only) |
| Subscript read `meta[i]` | Partially supported: runtime variables such as tensors/tuples support dynamic subscripts; **Python list does not support dynamic subscripts** (when the subscript is a number on the NPU), and only compile-time constant subscripts are supported (the same applies to list expressions created by `range`/`tla.range`/`tla.range_constexpr`) |

### Built-in keyword support

All 36 built-in Python keywords are listed below, classified by function. The general conventions for the two support columns (definitions of static/dynamic are in 1.2):

- **Static support** = evaluated at compile time with Python (host) semantics, generating no NPU instructions;
- **Dynamic support** = acts on runtime variables and is lowered to NPU instructions;
- "Partially supported" means there are exceptions, "Not supported" means the usage reports an error or has no NPU semantics; see the notes column for details; "—" means the keyword has no corresponding static/dynamic usage.

#### Literals

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `True`/`False` | Supported | — | Python boolean literals, lowered to boolean constants. As an `if` condition it is a compile-time branch (see 2.5) and generates no NPU branch instruction |
| `None` | Partially supported | Not supported | There is no corresponding NPU type, so it cannot participate in computation as a scalar. It only has placeholder semantics at compile time: tensor `data_ptr=None` is treated as unbound (normalized to `0`), a marker for a dynamic dimension in an index, and skipping type inference when used as a kernel argument |

#### Logical operations

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `and`/`or` | Supported | Supported | Supports short-circuit evaluation consistent with Python: `a and b` is equivalent to `b if a else a`, `a or b` is equivalent to `a if a else b`. When the operands are dynamic values, it is evaluated lazily and only the side actually needed is executed; the result type is the operand type and is not necessarily boolean |
| `not` | Supported | Supported | Logical negation; the result is always boolean (`Bool`) |

#### Comparison and membership

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `is`/`is not` | Supported | Not supported | Only applies to host values, evaluated as Python identity comparison (`left is right`); runtime variables have no concept of "identity" and generate no NPU instruction |
| `in`/`not in` | Supported | Not supported | Only applies to host values, evaluated as Python membership testing; membership checks on runtime variables must use other approaches. Note that the iteration semantics of `for x in ...` are described in the "Loops" category |

#### Branching

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `if`/`elif`/`else` | Supported | Supported | When the condition is known at compile time (literals, `tla.Constexpr[T]` parameters, `tla.const_expr(...)`), the compile time directly selects one side and generates no NPU branch (both branch bodies are still checked); when the condition depends on a runtime variable, it compiles to an NPU branch. Constraints on assignments inside a dynamic branch are described in 2.2 |
| `match`/`case` (soft keywords) | Not supported | Not supported | Pattern matching is out of scope (it reports an error inside a dynamic region) |

#### Loops

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `for` | Supported | Supported | When the bounds are compile-time constants, use `range`/`tla.range_constexpr` to expand the loop body at compile time; when the bounds depend on a runtime variable, use `tla.range` to compile to an NPU loop. The loop target must be a simple local variable name, and the induction variable cannot be used after the loop |
| `while` | Supported | Supported | When the condition is known at compile time, it executes as a host loop (generating no NPU instruction); when the condition depends on a runtime variable, it compiles to an NPU loop. A dynamic while is likewise subject to carried-state and no-early-exit constraints |
| `break`/`continue` | Supported | Not supported | Only valid inside a compile-time-expanded host loop (acting on the expansion process); dynamic loops have no jump-out instruction and report an error, so rewrite early exit as a conditional skip or a flag variable (see 2.3) |
| `else` (the else of `for`/`while`) | Supported | Not supported | For the `else` of `if`, see the "Branching" category; the `else` of `for`/`while` is available only for static loops (with host semantics), and a dynamic loop reports a for-else/while-else error |

#### Functions and scope

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `def` | Supported | Not supported | Supports `@tla.kernel`/`@tla.jit` root functions and module-level functions; a local function defined inside a kernel cannot be called from a dynamic region (it reports an error), so extract common logic to the module top level |
| `lambda` | Supported | Not supported | Only evaluated as ordinary Python, generating no NPU instruction; not allowed inside a dynamic condition |
| `return` | Supported | Not supported | A normal return from a kernel body is available; early return inside a dynamic region is not allowed (no call stack, no jump-out instruction) and must be rewritten as a conditional skip |
| `yield`/`yield from` | Not supported | Not supported | Generator semantics are out of scope |
| `async`/`await` | Not supported | Not supported | Asynchronous semantics are out of scope |
| `global`/`nonlocal` | Not supported | Not supported | Written by hand they have no NPU semantics; rebinding an outer variable inside a `with` region is generated automatically by the frontend as `nonlocal` and does not need to be written by hand (see 2.6) |

#### Regions and context

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `with` | Supported | — | Used to enter NPU regions: `tla.cube` (cube core), `tla.vector`/`tla.vec.func` (vector core). The region structure is determined at compile time, and a single `with` has only one context manager; temporary values inside the region cannot be used outside it |
| `as` | Supported | Not supported | Used with `with ... as x` (the target must be a local variable name) or `import ... as`; `except ... as` is unavailable because `except` is not supported |

#### Modules and classes

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `import`/`from` | Supported | Not supported | Module-level `import` is available and is loaded at compile time (in the host process), not executed on the NPU side; `from x import y` works the same way |
| `class` | Partially supported | Not supported | Nested class definitions inside a kernel are not supported; a module-level dataclass can serve as a carried-state container for a dynamic region (see 2.4) |

#### Exceptions and assertions

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `try`/`except`/`finally` | Partially supported | Not supported | It reports an error inside a dynamic region; outside a region it executes at compile time with host semantics, its result cannot be relied on, and exceptions are not recommended for flow control |
| `raise` | Partially supported | Not supported | Outside a region it executes with host semantics (a hit raises at compile time); inside a dynamic region it is forbidden together with `return`/`break`/`continue` |
| `assert` | Partially supported | Not supported | Evaluated at compile time with host semantics, and a failure directly causes a compilation error; it has no NPU semantics when acting on a runtime variable |

#### Variables and memory

| Keyword | Static support | Dynamic support | Notes |
| --- | --- | --- | --- |
| `pass` | Supported | Supported | A no-op that produces no instruction, usable as a placeholder |
| `del` | Partially supported | Not supported | Inside a dynamic if/while/for body it reports "does not support deletion"; outside a region it executes with host semantics (deleting a compile-time local variable) |

### Built-in function support

Support for built-in functions can be seen as a whitelist: only the built-in functions listed in the table below enter kernel compilation; the rest do not enter compilation and are evaluated at lowering time according to the host function rules above. For the complete list and semantics, see the [official Python built-in functions documentation](https://docs.python.org/zh-cn/3.13/library/functions.html#import__).

| Built-in function entering compilation | Support | How it enters compilation | Reference |
| --- | --- | --- | --- |
| `any()`/`all()`/`bool()` | Partially supported (requires a single argument and no keywords) | redirected to an NPU implementation | `ast_preprocessor.py:35-41`, `tla_ast_decorators.py:533-550` |
| `min()`/`max()` | Partially supported (keywords are not allowed) | redirected to an NPU implementation | `ast_preprocessor.py:35-41`, `tla_ast_decorators.py:559-567` |
| `abs(x)` | Supported | generates an NPU operation through `__abs__` | `typing.py:804-821` |
| `pow(x, y)` | Partially supported (float only, same as `**`) | generates an NPU operation through `__pow__` | `typing.py:861` |
| `range()` | Partially supported (bounds must be compile-time constants) | participates in compile-time loop expansion | `ast_preprocessor.py:305-474` |

## Examples

The examples are taken from `tla/kernel` and `dsl_tutorials` and trimmed; for complete runnable versions, see the corresponding files.

### Dynamic if: a variable changed inside a branch must be initialized first

In a dynamic `if`, a variable to be used after the branch must first be initialized, then reassigned inside the branch.

```python
@tla.kernel
def dynamic_if_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        if i == 0:
            coord = i + 1             # reassign inside the branch
        else:
            coord = i + 2             # the other branch assigns too
        tla.make_coord(coord, 0)      # used after the branch
```

Incorrect form:

```python
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1             # defined only inside the branch
        tla.make_coord(coord, 0)      # SyntaxError: ... must be initialized before the if
```

### Dynamic if: passing values between branches

Each branch assigns different values to the same set of variables, which are used uniformly after the branch.

```python
@tla.kernel
def select_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        offset = i + 1
        if i == 0:
            coord = i + 2             # branches assign different values to the same set of variables
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset) # used after the branch
```

### Dynamic for: carrying state across iterations

A dynamic loop does not support early exit; variables reassigned inside the loop are passed between iterations automatically, and the type must be consistent on every assignment.

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
            value = src_ub.load()     # initialize first
            for _ in tla.range(2):    # dynamic loop
                value = tla.abs(value)  # state is passed across iterations
            out_ub.store(value)
        tla.copy(mem_out, out_ub)
```

Incorrect form:

```python
    with tla.vec.func(mode="simd"):
        for _ in tla.range(2):
            break                     # SyntaxError: ... does not support return, break, continue, or raise
```

### Dynamic for: tiled loop

Loop over tiles, processing one block per iteration.

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
            for i in tla.range(0, 4, 1):              # tiled loop
                src_tile = tla.tile_view(src_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile = tla.tile_view(dst_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile.store(tla.abs(src_tile.load()))
        tla.copy(out, dst_ub)
```

### Compile-time branch

When the condition is known at compile time, the compile time directly selects one side and generates no NPU branch.

```python
@tla.kernel
def constexpr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):          # compile-time branch
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)
```

### with regions

NPU operations are placed inside a `tla.vector()` region, with vector computation wrapped in `tla.vec.func(mode="simd")`.

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

A class instance created with `@dataclass` can be used as a kernel argument, and can also be created inside the kernel.

```python
from __future__ import annotations
from dataclasses import dataclass
import catlass.tla as tla

@dataclass(frozen=True, kw_only=True)
class TilingData:
    TILE_M: tla.Constexpr[int]    # compile-time constant, not in the ABI; takes no argument slot in the generated IR
    tiling_gm_out: tla.Tensor     # supports tla.Tensor arguments
    tiling_int16: tla.Int16       # Int16 scalar
    tiling_float: tla.Float32     # Float32 scalar
    tiling_int: int               # Python int: compile-time int, runtime tla.Int32

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
    print_tiling(tiling)    # can be passed to other functions inside the kernel, just like a plain variable
    info = Info(tile_m=tiling.tiling_int, ...)  # instantiated on the kernel side


tiling = TilingData(TILE_M=128, ...)
artifact = tla.compile(struct_arg_kernel, tiling, out)
artifact(tiling, out)
```

**Instance usage scope**:

- Create an instance on the host side and pass it as a kernel argument; the field types support scalars and tensors
- Create an instance on the kernel side

**Options a dataclass allows customizing**: only `frozen/kw_only`; the other parameters cannot be changed from their defaults. With `frozen=True` field values cannot be modified after instantiation; with `kw_only=True` the keys must be specified explicitly at instantiation, for example `tiling=Tiling(TILE_M=128, tiling_int16=128, ...)`, not `tiling=TilingData(128, 128, ...)`

**`tla.Constexpr[...]` fields**:

- Treated as compile-time constants: read-only inside the kernel, usable in `tla.allocate`, `tla.range_constexpr`, etc.
- Do not enter the kernel ABI / IR: `tla.func` has no corresponding block argument.

**Supported argument types**:

- `tla.*` scalars: `tla.Bool`, `tla.Int8/16/32/64`, `tla.UInt8/16/32/64`, `tla.Float16/32`, `tla.BFloat16`.
- Plain Python scalars: `bool`→`i1`, `int`→`i32`, `float`→`f32`.
- `tla.Tensor`.

**Note**: the actual field type is determined by the **field value** (consistent with Python's dynamic semantics); constructing the dataclass does not coerce values to the annotated type.

### Error message quick reference

| Error message (keyword) | Cause | Fix |
| --- | --- | --- |
| `does not support return, break, continue, or raise` | early exit inside a dynamic region | skip with a condition, or record the result in a flag variable |
| `does not support for-else`/`while-else` | dynamic loop with else | remove else, move the logic into the loop body |
| `requires a simple local name target` | for target such as `pair[0]` | use a single local variable name |
| `induction variables cannot be used after the loop` | using the induction variable after the loop | store the needed value in a variable initialized beforehand inside the loop |
| `must be initialized before the if/loop` | variable defined only inside a branch or loop, used outside the region | initialize it before the region |
| `only supports assignments to local names or tuples/lists` | writing `out[i] =`, `obj.attr =`, etc. inside an if/while body | move the store out of the if/while body, or rebind a local name |
| `calling active local callable` | calling a local function inside a dynamic region | inline it, or move it to the module top level |
| `structure`/`expected i32` | variable structure or type differs across branches or iterations | keep the structure and type identical on every path |
| `'**' is only supported for float types` | integer power | use multiplication |

## References

- Control-flow usage examples: [`scalar_index_control_flow.py`](../../../../../../examples/python/tla_dsl/dsl_tutorials/tensor_index/scalar_index_control_flow.py), [`basic_vadd.py`](../../../../../../examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py), [`register_control_flow.py`](../../../../../../examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/register_control_flow.py)
- Control-flow AST implementation: [`ast_preprocessor.py`](../../../../catlass/base_dsl/ast_preprocessor.py), [`tla_ast_decorators.py`](../../../../catlass/tla_ast_decorators.py)
- Control-flow test case references: [`test_frontend_branching.py`](../../../../tests/test_frontend_branching.py), [`test_frontend_for_range.py`](../../../../tests/test_frontend_for_range.py), [`test_frontend_while.py`](../../../../tests/test_frontend_while.py), [`test_frontend_with.py`](../../../../tests/test_frontend_with.py)
