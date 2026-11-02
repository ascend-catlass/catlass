---
nav_order: 46
---

# Structured Kernel Arguments

Kernel arguments can be Tensors or scalars, or tuple/list, NamedTuple, or
dataclass values that contain them. The user passes the Python structures directly, and CATLASS Python expands the arguments and
restores the corresponding structures while compiling the function body, with no hand-written device ABI needed.

This document explains the usage. For Tensor creation see [Tensor Integration](tensor_binding.md); for compiled objects and streams
see [Compile and Launch](compile_and_launch.md); for the internal mechanism see
[Structured Arguments Design](../../dsl_development/feature_development/structured_arguments.md).

## Minimal Usage

The following passes an auxiliary Tensor and a scalar through a nested tuple/list. The kernel reads the first element,
and the expected result is `1.0 + 2.0 + 0.5 = 3.5`:

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

Both compilation and launch pass the complete `aux`; it cannot be rewritten to `*aux` at launch.
Here `None` preserves the structural position but passes no data to the device.

## Supported Types

| Argument type | Usage rules |
| --- | --- |
| `tla.Tensor` | pass a Host Tensor bound to a device buffer |
| TLA Numeric | use its own scalar type, such as `tla.Int64(3)` |
| Python `bool/int/float` | handled as i1/i32/f32 respectively |
| `tuple/list` | supports recursive nesting; the container type and length must be consistent between compilation and launch |
| `NamedTuple` | preserves the concrete type and field order; accessible by field name |
| `dataclass` | processed in field order; mixing dynamic fields and `Constexpr` fields is allowed |
| `None` | preserves the logical position and produces no device argument |
| Dynamic-GM Tensor | supported at the top level or nested; follows the Dynamic-GM Tensor API's dynamic descriptor rules |

The children of a container must also be of supported types. The runtime argument tree rejects dict, set, cyclic references, and arbitrary user-defined objects. The container restrictions here do not define the value range of top-level `Constexpr`.

`make_fake_tensor()` can be used for compilation but cannot serve as launch-time data. Whether a Tensor is at the top level or
inside a container, extracting a launch pointer requires a bound device buffer.

Subclasses of Python `int` and `float` (including `IntEnum`) are passed as the base numeric type by default, using
`i32` and `f32` respectively; `bool` uses `i1`. A subclass's extra attributes and methods are not passed to the device with the scalar.
When a numeric value exceeds the range of the declared type, an error is reported before launch; it is not widened automatically.

A dataclass allows customizing `frozen` and `kw_only`; the other decorator options must keep their default values, and
`slots=True` is not supported. Numeric field annotations are not responsible for automatically converting Python numbers:
to get i64, pass `tla.Int64(...)` rather than only annotating a Python int field as Int64.

tuple/list subclasses are reconstructed as `type(value)(children)`, and NamedTuple as
`type(value)(*children)`; the subclass must be restorable by that construction form, and extra instance state is not passed with the container.
A dataclass reads attributes by declared field and allows class defaults; declared fields that cannot be read or extra instance fields report an error.
Reconstruction calls `cls(**fields)` with all declared fields, so the constructor must accept these field names as keyword arguments.
With the auto-generated constructor, `field(init=False)` does not satisfy this requirement.
Declaring a derived property as a field is only a necessary condition; reconstruction also executes the constructor and the
`__post_init__` and other logic it calls at compile time, and these must be able to handle the reconstructed Tensor/Numeric values.

## NamedTuple and a dataclass combining static and dynamic fields

The following snippet reuses `tb` and `tout` from the example above:

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

`enabled` determines the code generated at compile time, while `bias` and `limit` are passed on each launch.
After changing the static configuration, call `tla.compile` again.

### Which arguments are omitted at launch

Only **top-level parameters declared as `Constexpr`** are removed from the compiled object's launch signature:

```python
@tla.kernel
def scaled(x: tla.Tensor, out: tla.Tensor, enabled: tla.Constexpr[bool]):
    if enabled:
        out[0] = x[0]

compiled = tla.compile(scaled, tx, tout, True, options="--npu-arch 3510")
compiled(tx, tout, block_num=1)
```

A plain empty tuple/list, a top-level `None`, and a fully static dataclass all keep their Python argument position,
even if they contain no dynamic fields. To make the whole configuration a compile-time argument, annotate the top level as `Constexpr`.
A missing argument, extra argument, or structure mismatch at launch reports an error.

## Dynamic-GM Arguments

The dynamic marking acts on the Tensor, regardless of whether it is at the top level or inside a container:

```python
dynamic_x = tx.mark_compact_shape_dynamic(0)
aux = (dynamic_x, [0.5, None])
compiled = tla.compile(add_bias, tx, aux, tout, options="--npu-arch 3510")
compiled(tx, aux, tout, block_num=1)
```

For dynamic dimensions and layout constraints, see [Static and Dynamic Layout](layout.md).
Ordinary Tensors and Dynamic-GM can be passed in combination; the physical fields of the descriptor are handled by CATLASS Python.
For a mixed-argument example, see [mixed_memref_arguments.py](../../../../../../examples/python/tla_dsl/dsl_tutorials/structured_arguments/mixed_memref_arguments.py).

## Reusing a Compiled Object

| Change | Handling |
| --- | --- |
| Tensor contents or address change, type constraints unchanged | launch directly, using the current address |
| runtime scalar value changes, type unchanged | launch directly, using the current value |
| marked-dynamic shape/stride/origin values change, dynamic modes and constraints unchanged | launch directly, using the current descriptor |
| static shape/stride, dtype, rank, layout, coordinate, or alignment constraints change | recompile; these compile constraints are not compared item by item at launch |
| container type, length, field order, or None position changes | recompile |
| NamedTuple/dataclass replaced with another class, including a different class with the same name | recompile |
| static configuration changes | must recompile; an existing compiled object does not compare the Constexpr field values at launch and still executes the original specialized code |

The launcher checks the container structure, leaf kinds, and ABI packing constraints. The complete
type and layout contract of ordinary Tensors and Dynamic-GM is guaranteed by the caller; passing an incompatible Tensor may cause wrong results or out-of-bounds access.
An ordinary None preserves the structural position check; a None declared as Constexpr is treated as a static field.
A compiled object uses the field definitions from compile time; do not modify the field definitions or annotations of the argument class after compilation.

Calling `tla.compile` again does not necessarily recompile the device binary: when it generates the same IR with a consistent compilation configuration,
the artifact can be reused, but with the structure binding of the current arguments.

## Advanced Usage: Launch Leaf Adaptation

Compile samples must still conform to the supported types described above. At launch, ordinary pointer leaves may use a
pointer provider that supplies a single valid address encoding; descriptor leaves provide canonical
fields through `build_memref_launch_fields()`, which applies to Dynamic-GM and static memrefs in mixed entries. The launcher checks the extraction capability, field count,
and encoding, and does not restrict the provider's Python class; the caller is responsible for making these values satisfy the contract of the compiled kernel.

The existing `register_jit_arg_adapter` can convert a single launch leaf into a Tensor, Numeric, or
pointer provider; the conversion result must still satisfy that leaf's ABI. Numerics and values that already provide
`__c_pointers__()` are used directly and are not overridden by a registered adapter.
This interface only adapts launch leaves and is not used to register custom containers or expansion rules.

Container reconstruction happens at compile time, and the Python object itself is not copied to the device; structured arguments are used to pass fields,
not as a state-passing protocol for device-side mutable Python objects.
