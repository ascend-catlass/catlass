---
nav_order: 10
---

# Syntax Constraints

## Basic Concepts

### `@tla.kernel` and `@tla.jit`

`@tla.kernel` is the device-kernel entry point. Direct invocation is currently
unsupported. Compile it explicitly with
`compiled = tla.compile(kernel, *sample_args)`, then launch with
`compiled(*runtime_args, block_num=...)`. `@tla.jit` is a Python helper callable
from a kernel and is preprocessed before lowering; calling or launching an
`@tla.kernel` from it is currently unsupported.

Bare Python helpers may also be called while a kernel is emitting device-time code. They execute in the caller's active emission context and may directly emit supported TLA operations. Use `@tla.jit` when the helper itself needs Python control-flow preprocessing. A bare helper called from device-time control flow must not declare `global` or `nonlocal`; calling a kernel as a helper and recursive `@tla.jit` calls are unsupported.

At compile time, kernels and helpers may also use ordinary Python functions, lambdas, nested functions, closures, class constructors, properties, and bound methods to derive compile-time configuration.

### Compile-time and device-time values

Compile-time values are ordinary Python values known while a specialized kernel is built, including literals and parameters or dataclass fields annotated with `tla.Constexpr[T]`. A `Constexpr` annotation classifies that input as compile-time-only: it has no device ABI slot.

Device-time values are available when the kernel executes, including tensor elements, `tla.arch.block_idx()`, `tla.arch.block_num()`, and values derived from them. A device-time predicate has type such as `tla.Bool`; a counted-loop induction value has type `tla.Int32`.

`tla.const_expr(value)` is a function for a control-flow condition. It requires a compile-time Python value and returns a Python `bool`; it rejects device-time values. It is separate from `tla.Constexpr[T]`: the annotation classifies an input, while the function makes a condition explicitly compile-time. A literal `if True` or `if False` is already compile-time and needs neither form.

| Form | Evaluated when | Result |
| --- | --- | --- |
| `if True:` / `if False:` | Compile time | Only the selected path is emitted |
| `if tla.const_expr(flag):` | Compile time | Only the selected path is emitted |
| `if predicate:` | Device time | Device-time branch |
| `while condition:` | Device time | Device-time loop |
| `for i in range(...):` | Device time | Counted device-time loop |
| `for i in tla.range(...):` | Device time | Counted device-time loop |
| `for i in tla.range_constexpr(...):` | Compile time | Python expands the body |

`range(...)` and `tla.range(...)` are device-time loops even when their bounds are Python integers. `tla.range_constexpr(...)` is the explicit compile-time expansion form.

### Expressions and Operators at a Glance

| Operator | Support |
| --- | --- |
| `+` `-` `*` `/` `//` `%`, unary minus `-x` | Supported |
| Bitwise `& \| ^ ~ << >>` | Supported |
| Comparisons `== != < <= > >=` | Supported |
| `is`/`is not`/`in`/`not in` | Partially supported for compile-time Python values |
| `and` `or` `not` (short-circuit), conditional expression `x if c else y` | Supported for applicable device-time values |
| `**` (power) | Partially supported (float types only) |
| Subscript read `meta[i]` | Supported as documented by the value type |

### Compile-time expansion

Use `tla.range_constexpr(...)` for small, intentional compile-time expansion. Its bounds must be compile-time integers and Python executes the loop while building the kernel. CATLASS emits one `DSLOptimizationWarning` before expanding a loop with 64 or more iterations, but continues the expansion. Prefer `range(...)` or `tla.range(...)` when expansion is not needed.

```python
@tla.kernel
def static_stages(count: tla.Constexpr[int]) -> None:
    for stage in tla.range_constexpr(count):
        emit_stage(stage)
```

Compile-time `while` uses `while tla.const_expr(condition):`. It has the same warning behavior, but an incorrect condition can prevent compilation from terminating; prefer `tla.range_constexpr(...)` for bounded compile-time repetition.

### Glossary

| Term | Meaning |
| --- | --- |
| Compile time | Python builds a specialized kernel |
| Device time | The generated kernel executes on the device |
| Carried value | Existing device-time state updated across a branch or loop |
| Device-time region | An `if`, `for`, `while`, or documented TLA `with` scope controlled at device time |

## Control-Flow Constraints

### Counted `for` loops

Both `range` forms accept one, two, or three bounds, including a device-time or negative step. Loop-tuning keywords are not part of the supported contract. The target must be one simple local name and is local to the loop; it cannot be used afterward. Device-time `for-else` is unsupported.

### Device-time `if` and `while`

Device-time `if`, `elif`, and `else` are supported. An `if` without `else` preserves an incoming carried device-time value on the false path. Device-time `while` carries compatible values updated by its condition and body; nested supported control flow is allowed. `while-else` is unsupported.

Read an outer compile-time value freely. An assignment to an existing compile-time binding inside a device-time region must explicitly promote the right-hand side with `tla.as_numeric(...)` (or a directly imported `as_numeric(...)`). Promotion occurs at the assignment, not when the value is created. New names first defined in a device-time region are local to that region and cannot be used afterward.

Supported tensor and pointer subscript stores are device-time side effects. Attribute assignment, `del`, and starred or chained assignment are unsupported.

### Early Exit and `else`

`return`, `break`, `continue`, and `raise` are unsupported in device-time control-flow bodies. Rewrite early exit as a condition or carried flag. Device-time `for-else` and `while-else` are unsupported.

### Variable Scope and Carried State

An existing device-time value may be rebound across branches and loops when every path preserves compatible leaf types and the same collection layout. Tensors, pointers, and documented CATLASS values may be carried.

A tuple, list, or dictionary that begins as compile-time state must be reconstructed as the entire fixed structure in the device-time assignment, applying `tla.as_numeric(...)` to every value leaf. Its source iterable and dictionary keys must be compile-time fixed; list length, dictionary keys and ordering, and leaf types must remain stable.

Dataclass values whose fields are supported device-time values may be constructed, carried, and rebound through device-time branches and loops. Their dataclass type, field layout, and compatible leaf types must remain stable. Objects implementing `__extract_mlir_values__` and `__new_from_mlir_values__` are also supported carried structures.

`tla.as_numeric(...)` promotes a scalar, not an arbitrary object. `set`, `frozenset`, and ordinary user classes cannot be promoted; do not mutate unpromoted Python containers or object attributes in device-time control flow.

### Compile-Time Branches

A literal Boolean condition is compile-time directly. Use `tla.const_expr(...)` when an otherwise Python expression should be explicitly evaluated as a compile-time condition. `tla.Constexpr[T]` parameters and fields are common sources of those values, but are not required by either form. Only the selected path becomes device code.

### `with` Regions

The Ascend NPU has two compute regions; device operations must be placed in their corresponding region:

- **Cube region**: hosts matrix operations such as `tla.mmad`, entered with `with tla.cube(...)`.
- **Vector region**: hosts element-wise and vector operations, entered with `with tla.vector(...)`. Use `tla.vec.func(mode="simd")` for a SIMD Vector Function sub-region.

Only documented TLA context managers are supported. One `with` may have one context manager. Device-time `with` scopes follow the same promotion and carried-state rules above; newly defined names do not escape the scope.

## Keywords at a Glance

| Keyword | Support | Notes |
| --- | --- | --- |
| `pass` | Supported | Emits nothing |
| `if`/`elif`/`else` | Supported | Dynamic branch or compile-time branch |
| `for`/`while` | Supported | `range` and `tla.range` are device-time; `tla.range_constexpr` is explicit compile-time expansion |
| `with` | Partially supported (only `tla.cube`/`tla.vector`/`tla.vec.func`) | Device region; one context manager per with |
| `def` | Partially supported | Bare Python and `@tla.jit` helpers follow the helper boundary in 1.1 |
| `lambda` | Partially supported | Valid for compile-time Python staging |
| `return`/`break`/`continue`/`raise` | Not supported | No early exit in dynamic regions |
| `del` | Partially supported (outside dynamic if/while bodies) | Assignment-target check |
| `global`/`nonlocal` | Not supported (hand-written) | Pass values explicitly |
| `import` | Not supported | Loaded at compile time, not executed on the device |
| `class` | Not supported | Nested definitions |
| `assert`/`try`/`except`/`finally`/`match`/`case` | Not supported | No compile-time error, but the result cannot be relied on |
| `async`/`await` | Not supported | — |
| `yield`/`yield from` | Not supported | — |

## Examples

The examples below are trimmed from `tla/kernel` and `dsl_tutorials`; see the corresponding files for complete runnable versions.

### Dynamic if: initialize before rebinding in the branch

Dynamic `if` branches are converted into standalone functions, so a variable must be initialized first and then rebound inside the branch.

```python
@tla.kernel
def dynamic_if_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        if i == 0:
            coord = i + 1             # rebind inside the branch
        else:
            coord = i + 2             # the other branch assigns too
        tla.make_coord(coord, 0)      # used after the branch
```

Incorrect:

```python
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1             # defined only inside the branch
        tla.make_coord(coord, 0)      # SyntaxError: ... must be initialized before the if
```

### Dynamic if: passing values between branches

Each branch assigns different values to the same set of variables, which are used after the branch.

```python
@tla.kernel
def select_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        offset = i + 1
        if i == 0:
            coord = i + 2             # branches assign different values
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset) # used after the branch
```

### Dynamic for: carrying state across iterations

Dynamic loops do not support early exit; variables rebound inside the loop are passed across iterations automatically, and the type must stay the same on every assignment.

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
                value = tla.abs(value)  # carried across iterations
            out_ub.store(value)
        tla.copy(mem_out, out_ub)
```

Incorrect:

```python
    with tla.vec.func(mode="simd"):
        for _ in tla.range(2):
            break                     # SyntaxError: ... does not support return, break, continue, or raise
```

### Dynamic for: tiled loops

Loop over tiles, processing one tile per iteration.

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

When the condition is known at compile time, the branch is selected at compile time and no device branch is generated.

```python
@tla.kernel
def constexpr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):          # compile-time branch
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)
```

### with regions

Device operations are placed inside a `tla.vector()` region, with vector computation wrapped in `tla.vec.func(mode="simd")`.

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

Class instances created with `@dataclass` can be used as kernel arguments, and can also be created inside the kernel.

Dataclasses are also valid device-time carried state when their fields are
supported values. Rebind the complete instance rather than mutating its
structure, and preserve its type and field layout across every path and
iteration:

```python
@dataclass
class LoopState:
    index: tla.Int32
    limit: tla.Int32


@tla.kernel
def carry_dataclass(limit: tla.Int32) -> None:
    state = LoopState(tla.as_numeric(0), limit)
    if limit > 0:
        state = LoopState(state.index + 1, state.limit)
    for _ in range(limit):
        state = LoopState(state.index + 1, state.limit)
    consume(state.index)
```

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
    tiling_int: int               # python int: compile-time int, runtime tla.Int32

@dataclass(frozen=True)
class Info:
    tile_m: int
    tile_n: int

def print_tiling(tiling: TilingData):
    tla.print("tiling_int={}", tiling.tiling_int)

@tla.kernel
def struct_arg_kernel(tiling: TilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_int16
    ptr = tla.allocate(tiling.TILE_M, tla.Int32, tla.AddressSpace.ub, 256)
    print_tiling(tiling)    # can be passed to other functions inside the kernel, just like a plain variable
    info = Info(tile_m=tiling.tiling_int, tile_n=tiling.tiling_int)


# Construct `tiling` and `out` with application-specific supported values.
```

**Usage scope**:

- Create an instance on the host side and pass it as a kernel argument, fields should be numeric and tensor
- Create an instance on the kernel side

**Customizable dataclass options**: only `frozen/kw_only`; the other options cannot be changed from their defaults. With `frozen=True` field values cannot be modified after instantiation; with `kw_only=True` fields must be passed by keyword, e.g. `tiling=TilingData(TILE_M=128, tiling_int16=128, ...)`, not `tiling=TilingData(128, 128, ...)`.

**`tla.Constexpr[...]` fields**:

- Treated as compile-time constants: read-only inside the kernel; usable in `tla.allocate`, `tla.range_constexpr`, etc.
- Do not enter the kernel ABI / IR: `tla.func` has no corresponding block argument.

**Supported argument types**:

- `tla.*` scalars: `tla.Bool`, `tla.Int8/16/32/64`, `tla.UInt8/16/32/64`, `tla.Float16/32`, `tla.BFloat16`.
- Plain Python scalars: `bool`→`i1`, `int`→`i32`, `float`→`f32`.
- `tla.Tensor`.

**Note**: the actual field type is determined by the **field value** (consistent with Python's dynamic semantics); constructing the dataclass does not coerce values to the annotated type.

### Error Reference

| Error message (keyword) | Cause | Fix |
| --- | --- | --- |
| `does not support return, break, continue, or raise` | Early exit inside a dynamic region | Skip with a condition, or record the result in a flag variable |
| `does not support for-else`/`while-else` | Dynamic loop with else | Remove else, move the logic into the loop body |
| `requires a simple local name target` | for target such as `pair[0]` | Use a single local variable name |
| `induction variables cannot be used after the loop` | Using the induction variable after the loop | Store the needed value in a variable initialized beforehand inside the loop |
| `must be initialized before the if/loop` | Variable defined only inside a branch/loop, used outside | Initialize it before the region |
| assignment-target diagnostic | Attribute, deletion, starred, or chained assignment in a device-time region | Rebind supported local state; supported tensor subscript stores remain valid |
| `calling active local callable` | Calling a local function inside a dynamic region | Inline it, or move it to module level |
| `structure`/`expected i32` (TlaCoreAPIError) | Carried state structure or type differs across branches/iterations | Keep the structure and leaf types identical on every path |
| `'**' is only supported for float types` | Integer power | Use multiplication |

## References

- Control-flow examples: `examples/python/tla_dsl/dsl_tutorials/tensor_index/scalar_index_control_flow.py`, `examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py`, `examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/register_control_flow.py`
- Control-flow frontend implementation: `catlass/base_dsl/ast_preprocessor.py`, `catlass/tla_ast_decorators.py`
- Control-flow test cases: `tests/test_frontend_branching.py`, `tests/test_frontend_for_range.py`, `tests/test_frontend_while.py`, `tests/test_frontend_with.py`
