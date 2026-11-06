# Tensor `tla.print` example

This example compiles and launches one single-block kernel that prints a known
16-element prefix from a static or dynamic-shaped GM-resident `float32[8,4]`
tensor, or from a static/dynamic-shaped AIV UB view. The output verifier
requires exactly one native CANN record for ordinary cases and prints only the
stable public fields:

```text
tla.print dtype=float32 subblock=0 shape=[8,4] count=16 values=[0.0, ..., 15.0]
compile_ok=True
launch_ok=True
output_ok=True
```

Kernels use `with tla.vector():` (AIV). Vector-core output includes the logical
C310 vector `subblock` (`0` or `1`). AIC/AIV is inferred from the region at
compile time; there is no Host `--core-type` switch.

Run GM:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --dynamic-shape --device 0 --block-num 1
```

Exercise a tensor print inside runtime control flow. `--enabled 0`
takes the false branch and validly emits no tensor record. With `--enabled 1`,
the runtime-bounded loop emits one record for each executed iteration; this
example verifies each available record rather than requiring a fixed count:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case dynamic-control-flow --enabled 0 --repeats 2 --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case dynamic-control-flow --enabled 1 --repeats 2 --device 0 --block-num 1
```

Exercise raw physical-prefix semantics with column-major GM storage. The
logical output shape remains `[8,4]`, while values follow the contiguous
transposed backing buffer:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --layout column-major --device 0 --block-num 1
```

Exercise the exact 262,112-element GM capacity boundary (redirect the large
canonical record when running in automation):

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case capacity --device 0 --block-num 1 > capacity.log
```

Run the UB base-address and aligned-offset cases. The kernel explicitly
uses a 32-byte row width and completes the producer-side GM-to-UB copy before
printing:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case base --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case aligned-offset --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case dynamic --device 0 --block-num 1
```

Native AIC L1 printing uses CANN's C310 debug bus and opens the required
DebugTunnel automatically. The battery covers both layouts, all dtypes,
aligned offsets, repeated calls, cache reuse, and a two-core block-dependent
case. The host state depends on CANN's private DebugTunnel ABI and has been
validated with CANN 9.2.0. A CANN upgrade requires checking that ABI and validating
L1/L0C printing again.

Run it with `--storage l1 --layout zn --arch-scope aic.c310`.

Native AIC L0C printing dumps one 16x16 CO1 view. `float32` and `int32` are
native accumulator types; other dtypes are byte-preserving views. It shares
L1's DebugTunnel/FIFO envelope, while CANN performs the transfer without a
debug-bus address.

Run it with `--storage l0c --dtype f32 --arch-scope aic.c310`.

## Support matrix

| Property | Supported | Rejected |
| --- | --- | --- |
| Core scope | AIV for GM/UB; AIC for L1/L0C (inferred from IR) | Host `--core-type` / `compile(core_type=...)` |
| Launch grid | One or more blocks for static print sites | Multi-block dynamic-control-flow example |
| Storage | GM; 32-byte-aligned effective UB address; aligned A1 L1 address; aligned CO1 L0C fractal | L0A/L0B or host invocation |
| Dtype | GM/UB/L1/L0C: `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16`, `u32` | Other dtypes |
| Shape | Rank-1/rank-2 static or runtime-shaped tensors | Empty, rank above 2, or mismatched runtime metadata |
| Length | Static or integer-SSA 1–262,112 element prefix, no greater than runtime tensor size | Zero, negative, over 262,112, or over tensor size |
| Dynamic control flow | GM print sites under runtime `if` and `tla.range` | multi-block or dynamic-shape variants of this example case |
| Layout | GM/UB TLA layouts; L1 `zN [M,C0]` and `nZ [C0,N]`, where C0 is 32 bytes; one aligned 16x16 `L0Clayout` tile | Other L1/L0C layouts or partial L0C tiles |
| Baseline | Ascend 950PR, validated with CANN 9.2.0 | Other device/CANN combinations are not declared |

`tla.print(value, length, /)` derives dtype and the concrete runtime shape from
the tensor. Length is an element count, not a byte count, and may be a Python
integer or integer SSA value. Dynamic-shaped tensors require an explicit
length. Static tensors may omit it when the complete tensor fits the
262,112-element `float32` FIFO capacity. The maximum
comes from the 1 MiB CANN ring, its 48-byte shape TLV, 72-byte tensor TLV, and
32-byte payload alignment.

For a print site nested in dynamic control flow, the disabled path may emit no
record and a loop may emit the same static site repeatedly. The FIFO is
best-effort: once it fills, later records can be absent. The example accepts
those zero/repeated outcomes but rejects every emitted record whose public
dtype, position, subblock, shape, count, or values do not match the call site.
Malformed FIFO data and unknown print identities remain execution errors.

The dynamic examples pass both the first extent and print length as scalar
kernel arguments. Current pointer-only host tensor arguments do not carry
runtime memref extents.

Like Ascend C `DumpTensor`, values are read as a contiguous physical prefix
from the effective address. The logical tensor shape controls display grouping
but does not gather through strides or reorder packed storage. Runtime guards
reject invalid counts and misaligned effective addresses by emitting no native
record, which the host reports as an execution error. `tla.print` does not
insert producer synchronization; callers must complete writes to UB before
printing. L1 requires the MTE2-to-MTE1 handoff; L0C requires MMAD-to-FIX.
CANN publishes A1 directly and converts one CO1 fractal to ND. L0C tile
coordinates remain relative to the root accumulator and its stride.
