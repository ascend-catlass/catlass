---
nav_order: 20
---

# Control Flow

TLA kernel bodies are executed by Python while the frontend builds device IR. This makes ordinary, inspectable Python staging code useful for deriving static configuration before an operation is lowered.

## Python staging

Kernels and `@tla.jit` functions may call bare Python functions, lambdas, nested functions, closures, class constructors, properties, and bound methods while their values remain Python compile-time values. A bare helper executes as ordinary Python in the caller's active frontend context, so its TLA operations emit IR directly. It does not need `@tla.jit` merely to call TLA APIs.

A bare helper is never transformed independently. Its own TLA runtime control flow is therefore unsupported: keep such control flow in the decorated caller. Decorated-helper composition is reserved for the next control-flow follow-up.

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

## Runtime control flow and promotion

Conditions and loop bounds backed by Python values create runtime control flow. Values written in such a region become runtime state. Promote a Python scalar explicitly with `tla.as_numeric(...)` before assigning to it in a runtime `if`, `tla.range` loop, `while`, or outlined Python `with` region:

```python
@tla.kernel
def kernel(limit: int) -> None:
    index = tla.as_numeric(0)
    if limit > 0:
        index = index + 1
    tla.make_coord(index, 0)
```

Compose a fixed runtime collection with a comprehension over a literal list/tuple of elements, wrapping every carried value with `tla.as_numeric`:

```python
state = [tla.as_numeric(value) for value in (0, 1)]
```

Container literals, partially promoted collections, and dynamic collection iterables are not runtime promotion forms.

Reading a compile-time binding after it has been updated inside runtime control flow is unsupported. Keep the promoted value or compute the later use inside the runtime region.

## Object boundary

Dataclass values may be constructed, carried, and rebound in runtime control flow when their fields are otherwise supported runtime values. Other user-defined objects are staging-only: they cannot be kernel ABI arguments, passed to `tla.as_numeric(...)`, or mutated in runtime control flow. Use tensor stores for device-visible mutation and promote only supported scalar values or fixed collections assembled with the supported comprehensions.

## `@tla.jit`

`@tla.jit` is a Python helper used while a kernel is being lowered. It may be called from `@tla.kernel`, but calling or launching an `@tla.kernel` from it is currently unsupported. Direct `@tla.kernel` invocation is also currently unsupported. Compile explicitly with `tla.compile(kernel, *sample_args)`, then invoke the returned compiled function.
