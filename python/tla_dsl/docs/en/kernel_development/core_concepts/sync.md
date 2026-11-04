---
nav_order: 25
---

# Synchronization Guide

By default, CATLASS DSL does **not** insert synchronization between hardware pipes.
`tla.copy`, `tla.mmad`, and the compute inside `tla.vec.func` are lowered to instructions
on different pipes, and those pipes run asynchronously. Only an explicit `tla.flag` /
`set_flag` / `wait_flag` (or a cross-core `cross_flag` / `mutex`) guarantees that the
producer finishes before the consumer starts.

Incorrect synchronization usually does **not** produce a compile error. On device it
shows up as:

- a hang (waiting for a flag that is never set, or that was already consumed);
- intermittent wrong results (most iterations are correct, a few tiles read stale data);
- compile-time errors (event ids exhausted, duplicate flag names, a `wait_flag` with no
  matching `set_flag`).

Compile-time checking is **one-directional**: it reports a `wait_flag` that has no
`set_flag`, but it does not catch a redundant `set_flag`, reversed set/wait order, a missing
pre-loop prime or post-loop drain, or overlapping lifetimes of one flag. "It compiles"
therefore does not prove the synchronization protocol is correct.

This page is organized as: decide whether synchronization is needed -> pick an interface ->
write it -> diagnose failures. For signatures see the
[Synchronization API reference](../../api/language/sync.md).

## 1. Hardware model: pipes

Inside one AIC/AIV there are several independent pipes. Instructions on the same pipe
execute in order; different pipes have no ordering guarantee.

| Pipe | TLA token | Instructions |
| --- | --- | --- |
| PIPE_S | `tla.arch.SCALAR` | Scalar arithmetic, tensor subscript access, scalar load/store in `vec.func` |
| PIPE_V | `tla.arch.VECTOR` | Vector compute (SIMD arithmetic, `tile.load/store`, ...) |
| PIPE_M | `tla.arch.CUBE` | `tla.mmad` |
| PIPE_MTE1 | `tla.arch.MTE1` | L1 -> L0A/L0B movement |
| PIPE_MTE2 | `tla.arch.MTE2` | GM -> L1/L0/UB movement |
| PIPE_MTE3 | `tla.arch.MTE3` | UB -> GM movement |
| PIPE_FIX | `tla.arch.FIX` | L0C -> GM/UB movement |

Instructions on one pipe are naturally ordered, so **only cross-pipe accesses need
attention**. Vector/scalar visibility inside `tla.vec.func` is a different layer; use
`local_mem_bar` (section 8) when needed.

## 2. Which pipe does each operation use

The first step is to identify the producer and consumer pipes. For `tla.copy`, the pipe is
determined by the **source** address space:

| Operation | Pipe | Note |
| --- | --- | --- |
| `tla.copy(dst_l1, src_gm)` | MTE2 | Source is GM |
| `tla.copy(dst_l0, src_l1)` | MTE1 | Source is L1 |
| `tla.copy(dst_ub, src_gm)` | MTE2 | Source is GM |
| `tla.copy(dst_gm, src_ub)` | MTE3 | Source is UB |
| `tla.copy(dst_gm, src_l0c)` | FIX | Source is L0C |
| `tla.mmad(...)` | CUBE | |
| SIMD compute, `tile.store` / `tile.load` | VECTOR | Inside `tla.vec.func(mode="simd")` |
| `tensor[i]` read/write, scalar arithmetic | SCALAR | |
| Scalar load/store in `tla.vec.func` | SCALAR | |

Note: when the source of `tla.copy` is L0A/L0B there is no inferable pipe
(`auto_sync` reports an error); with manual synchronization you must determine its pipe
yourself.

## 3. Deciding whether synchronization is needed

For every producer-write / consumer-read pair:

1. **Same pipe?** No synchronization needed.
2. **Is there a data dependency?** See the table below.
3. **Can the consumer run first?** If it can, synchronize; when unsure, assume it is
   needed.

| Dependency | Meaning | Typical case | Flag to insert |
| --- | --- | --- | --- |
| RAW (read after write) | Consumer reads data just written | Compute right after a `copy` into UB | Producer pipe -> consumer pipe |
| WAR (write after read) | Producer overwrites a buffer the consumer still reads | The next `copy` overwrites the UB tile just computed | Consumer pipe -> producer pipe |
| WAW (write after write) | Two writes to the same memory | Two `copy` calls into the same UB tile | Earlier write -> later write |

The most common mistake is handling RAW but forgetting WAR. Double buffering needs both;
see [section 5](#5-double-buffering-ping-pong).

## 4. In-core synchronization: `tla.flag` + `set_flag` / `wait_flag`

The semantics are:

- `set_flag` waits for all preceding instructions **on the source pipe** and then sets the
  flag. It does **not** block subsequent instructions on the source pipe;
- `wait_flag` runs on the **destination pipe**. If the flag is not set, it blocks later
  instructions on that pipe; once set, the flag is cleared and execution continues.

Rules:

- `tla.flag(name, src_pipe, dst_pipe, id=-1)` only creates a handle. Create it near the top
  of the kernel body, before entering `tla.cube()` / `tla.vector()`;
- `tla.set_flag` / `tla.wait_flag` must appear inside `tla.cube()` or `tla.vector()` and
  must be paired;
- A flag name can be defined only once, and `set_flag` / `wait_flag` must use the same
  handle;
- A flag is a one-shot event: one `set` corresponds to one `wait`. Setting the same flag
  twice without an intervening wait is undefined behavior and can hang a later
  `pipe_barrier(ALL)`;
- Each `(src_pipe, dst_pipe)` pair has only **8 hardware event ids (0..7)**. The default
  `id=-1` auto-allocates and skips explicitly used ids. Explicit ids may be reused, but the
  two synchronization lifetimes must not overlap.

Minimal example (GM -> UB -> compute -> GM):

```python
@tla.kernel
def vadd(x: tla.Tensor, y: tla.Tensor, out: tla.Tensor) -> None:
    # Buffer and tile-view creation omitted: x_ub / y_ub / out_ub come from tla.make_tensor_like
    # Data ready: after MTE2 writes UB, VECTOR may read it.
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    # Compute done: after VECTOR writes UB, MTE3 may read it.
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    with tla.vector():
        tla.copy(x_ub, x_gm)          # MTE2
        tla.copy(y_ub, y_gm)          # MTE2
        tla.set_flag(ub_loaded)       # MTE2 predecessors completed
        tla.wait_flag(ub_loaded)      # VECTOR waits for data

        with tla.vec.func(mode="simd"):
            out_ub.store(x_ub.load() + y_ub.load())   # VECTOR

        tla.set_flag(vec_done)        # VECTOR predecessors completed
        tla.wait_flag(vec_done)       # MTE3 waits for compute
        tla.copy(out_gm, out_ub)      # MTE3

        tla.pipe_barrier(tla.pipes.ALL)   # drain, see section 8
```

WAR case (producer may overwrite only after the consumer is done):

```python
# Only after VECTOR has read ub may MTE2 write the next tile.
ub_free = tla.flag("ub_free", tla.arch.VECTOR, tla.arch.MTE2)
...
tla.set_flag(ub_free)
tla.wait_flag(ub_free)
tla.copy(ub, gm_next)
```

## 5. Double buffering (ping-pong)

Double buffering needs two flags per buffer:

- `loaded[b]`: the buffer holds ready data (producer pipe -> consumer pipe);
- `available[b]`: the buffer is free and may be overwritten (**the pipe that reads it** ->
  producer pipe).

Two things are easy to miss, and either one breaks the kernel:

1. **Preset `available` before the loop**, otherwise the first
   `wait_flag(available[0])` waits forever;
2. **Drain after the loop** with `pipe_barrier(tla.pipes.ALL)` so all issued moves and
   compute complete.

```python
@tla.kernel
def vadd_pingpong(x: tla.Tensor, y: tla.Tensor, out: tla.Tensor) -> None:
    num_tiles = 4

    loaded = (
        tla.flag("loaded_0", tla.arch.MTE2, tla.arch.VECTOR, id=0),
        tla.flag("loaded_1", tla.arch.MTE2, tla.arch.VECTOR, id=1),
    )
    store_done = (
        tla.flag("store_done_0", tla.arch.VECTOR, tla.arch.MTE3, id=0),
        tla.flag("store_done_1", tla.arch.VECTOR, tla.arch.MTE3, id=1),
    )
    available = (
        tla.flag("available_0", tla.arch.MTE3, tla.arch.MTE2, id=0),
        tla.flag("available_1", tla.arch.MTE3, tla.arch.MTE2, id=1),
    )

    with tla.vector():
        # 1. Initial state: both buffers are writable in the first round.
        tla.set_flag(available[0])
        tla.set_flag(available[1])

        # range_constexpr expands at compile time, so slot is a compile-time int
        # and can index the flag tuples.
        for i in tla.range_constexpr(num_tiles):
            slot = i % 2

            # 2. WAR: wait until the previous round finished reading/storing.
            tla.wait_flag(available[slot])
            tla.copy(ub_x[slot], gm_x[i])     # MTE2
            tla.copy(ub_y[slot], gm_y[i])     # MTE2

            # 3. RAW: publish the data, then let VECTOR read it.
            tla.set_flag(loaded[slot])
            tla.wait_flag(loaded[slot])
            with tla.vec.func(mode="simd"):
                ub_out[slot].store(ub_x[slot].load() + ub_y[slot].load())

            # 4. RAW: compute done, MTE3 may read and store to GM.
            tla.set_flag(store_done[slot])
            tla.wait_flag(store_done[slot])
            tla.copy(gm_out[i], ub_out[slot])  # MTE3

            # 5. WAR: store done, the buffer is reusable.
            tla.set_flag(available[slot])

        # 6. Drain: ensure the final store really completes.
        tla.pipe_barrier(tla.pipes.ALL)
```

> Note: this example treats the x/y/out buffers as one slot, so `available` waits for the
> `MTE3` store to finish before releasing the next `MTE2` load. That is the most
> conservative form. If you only guard the input buffers, change the direction to
> `VECTOR -> MTE2` and set it after compute; the output buffer then needs its own flag pair.

Constraint: the flag handle must be selectable at compile time. When the loop bound is a
runtime value (`tla.range`), you cannot index a Python tuple with the loop variable.
Unroll the buffers into compile-time branches, or use `tla.range_constexpr` (which expands
the loop body).

In addition, the operand of `set_flag` / `wait_flag` must be defined directly by `tla.flag`.
If control flow such as `scf.if` yields the handle and turns it into a block argument,
lowering fails with `expected ... to have a 'flag' attribute or a tla.flag operand`.
`cross_flag` has a structural conversion that allows more dynamic handles; in-core flags
have no equivalent path.

## 6. Cross-core synchronization: `tla.cross_flag`

When AIC and AIV, or different AIVs, share data, in-core flags do not apply; use cross-core
flags. `tla.cross_flag(name, mode=...)` creates the handle, and
`cross_core_set_flag` / `cross_core_wait_flag` specify the pipe inside
`tla.cube()` / `tla.vector()` and must be paired.

| mode | Scope | Meaning |
| --- | --- | --- |
| 0 | All AICs (or all AIVs) across AI Cores | Global barrier; every participant must set before any proceeds |
| 1 | Between the AIVs of one AI Core | Both AIVs must set before either proceeds |
| 2 (default) | AIC <-> all AIVs in one core | AIC set releases both AIVs; both AIVs must set to release AIC |
| 4 | AIC <-> one AIV in one core | Addressed by `aiv_id` (0/1); Ascend 950 only |

Rules and limits:

- Modes 0/1/2 use cross flag ids 0..15. In mode 4, AIC<->AIV0 uses 0..15 and AIC<->AIV1 uses
  16..31. A kernel may have at most **16 cross flags**, otherwise compilation fails with
  `cross flag id exhausted`;
- `pipe` for set/wait does not support `ALL`; common choices are `tla.arch.CUBE`,
  `tla.arch.FIX`, `tla.arch.VECTOR`, `tla.arch.MTE2`;
- Mode 4 requires an explicit `aiv_id=0/1`. Write `aiv_id=i` on the AIC side, and write both
  0 and 1 on the AIV side; the compiler generates a `sub_block_idx` guard so each AIV only
  executes its own instruction at runtime;
- With `@tla.kernel(core_ratio=tla.arch.C1V1)`: mode 4 allows only `aiv_id=0`;
  `cross_core_wait_flag` must use `tla.arch.SCALAR`; cube results must stage
  `L0C -> GM -> UB` (tightly coupled `L0C -> UB` is illegal). The compiler neither rewrites a
  wrong wait pipe nor enforces this rule at compile time: a wrong pipe usually still compiles,
  then fails or hangs on device (see `basic_mixed_core_ratio_11.py`).

Two-phase mode-4 handshake (AIC delivers a token -> AIV acks -> AIC releases again). The
example below is for two AIVs (not `C1V1`); under `C1V1`, follow the rules above and address
only `aiv_id=0` with `tla.arch.SCALAR` on the wait:

```python
aic_to_aiv = tla.cross_flag("aic_to_aiv", mode=4)
aiv_to_aic = tla.cross_flag("aiv_to_aic", mode=4)

with tla.cube():
    tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=0)
    tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=1)
    # Proceed only after both AIVs have acknowledged.
    tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=0)
    tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=1)

with tla.vector():
    tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=0)
    tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=1)
    ...
    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)
```

For a complete runnable version see
[`cross_flag_two_way.py`](https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py).

## 7. Resource mutex: `tla.mutex`

Flags and mutexes solve different problems:

| | flag (`tla.flag`) | mutex (`tla.mutex`) |
| --- | --- | --- |
| Semantics | One-shot event: one set per wait | Counting lock: Lock blocks until every Unlock with the same id |
| Direction | One-way (source pipe -> destination pipe) | Any direction; can express "wait until all readers leave" |
| Use case | Fixed producer/consumer order | Many consumers, buffer reuse, reverse synchronization |

```python
mtx = tla.mutex("l1_buf", id=0)     # id must be explicit

with tla.cube():
    with tla.mutex_guard(mtx):
        tla.copy(l1_a, gm_a)        # the guard block must emit copy or mmad
```

Rules and limits:

- **`id` must be explicit**: bitcode lowering does not support auto allocation, so
  `id=-1` fails with `mutex id auto allocation is not implemented`. `id` ranges over
  `0..31`, and no id in that range is reserved;
- `pipe` depends on the enclosing region: `SCALAR`/`CUBE`/`MTE1`/`MTE2`/`MTE3`/`FIX` in
  `tla.cube()`; `SCALAR`/`VECTOR`/`MTE2`/`MTE3` in `tla.vector()`. There is no `ALL`
  form;
- A mutex works only within one core. Use `cross_flag` across cores;
- Do not call `mutex_lock` / `mutex_unlock` explicitly inside a `mutex_guard` block.

## 8. `pipe_barrier` and `local_mem_bar`

`tla.pipe_barrier(pipe)` blocks the **given pipe** until the instructions already issued on
that pipe complete; it does not wait for other pipes. Use `tla.pipes.ALL` to drain every
pipe. It must be inside `tla.cube()` / `tla.vector()`, and Ascend 950 supports only a limited
set of pipes:

| Region | Allowed pipes |
| --- | --- |
| `tla.cube()` | `CUBE`, `MTE1`, `MTE2`, `FIX`, `ALL` |
| `tla.vector()` | `MTE2`, `MTE3`, `ALL` (no `pipe_barrier<PIPE_V>`) |

Therefore do not write `tla.pipe_barrier(tla.arch.VECTOR)` inside a vector region; use
`tla.pipe_barrier(tla.pipes.ALL)` when you need a full drain.

`tla.local_mem_bar(src, dst)` is only valid inside `tla.vec.func`. It orders vector and
scalar accesses within the SIMD region, and only 12 `MemType` pairs are supported. Common
ones:

```python
tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.SCALAR_LOAD)
tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.SCALAR_ALL)
```

It **only affects the inside of `vec.func`**: after `vec.func` returns, the outer code still
needs normal flags so scalar and other pipes observe the complete result (see
[`load_and_store_scalar_after_reduction.py`](https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_and_store_scalar_after_reduction.py)).
Thread synchronization inside a SIMT region uses `tla.arch.sync_threads()`; it is not a
cross-core or pipe barrier.

## 9. Automatic synchronization: `@tla.kernel(auto_sync="v0")`

`auto_sync="v0"` automatically inserts mutex-style synchronization for in-core accesses to
`tla.copy`, `tla.mmad`, and `tla.vec.func`. It suits regular move-compute pipelines.
Limits:

- It only generates synchronization **inside one AIC or AIV**. Cross-core synchronization
  and thread synchronization inside `tla.vec.func` remain explicit;
- It cannot be mixed with manual `tla.flag`/`set_flag`/`wait_flag` or
  `tla.mutex`/`mutex_guard`, and `tla.call_extern` is unsupported;
- Protected on-chip tensors must come from `tla.allocate`, not `tla.make_ptr`;
- A carried on-chip pointer must not be switched to another `tla.allocate` allocation root
  inside a loop;
- `tla.mmad` `unit_flag` must be provably always 0 or always 2/3; L0C copy supports only 0
  or 3;
- When one instruction conditionally selects several buffers, every branch must lock
  allocations in the same order;
- UB `tla.scalar_load`/`scalar_store` must be placed inside `tla.vec.func`;
- `tla.print_tensor` / `tla.debug_print` are not automatically synchronized.

Design details and static-analysis constraints are in the
[AutoSync design document](../../dsl_development/feature_development/auto_sync_design.md).
When in doubt, write synchronization manually: manual synchronization has deterministic
behavior, while AutoSync has a clearly bounded coverage.

## 10. Common errors

| Symptom / error | Cause | Fix |
| --- | --- | --- |
| `event id exhausted (max 7)` | More than 8 flags on one pipe pair | Reuse flags, reduce concurrent buffers, or merge same-direction syncs |
| `cross flag id exhausted` | More than 16 cross flags | Merge syncs with the same mode, or reuse flags |
| `wait_flag used without set_flag for 'xxx'` | A wait without a matching set, or a typo | Check the name and the pairing |
| `mutex id auto allocation is not implemented` | `tla.mutex` left at the default `id=-1` | Pass an explicit `id=0..31` |
| `in vector pipe_barrier only support [...]` | Unsupported pipe (e.g. PIPE_V) in a vector region | Use `MTE2`, `MTE3`, or `ALL` |
| Duplicate flag name fails compilation | The same `tla.flag` name is defined twice | Keep flag names unique |
| Hang | Initial `available` flag not preset, or set/wait not paired | Preset before the loop; check pairing |
| Intermittent wrong results | Missing WAR sync, or no final `pipe_barrier(ALL)` | Add buffer-free flags and a drain |
| Cross-core handshake fails | Wrong mode, missing `aiv_id`, or wait pipe `VECTOR` under 1:1 | Recheck mode and `core_ratio` rules in section 6 |

## 11. Pre-submit checklist

- [ ] For every `tla.copy` / `tla.mmad` / `vec.func` consumer, the producer pipe is known;
- [ ] Every cross-pipe data dependency has a set/wait pair referencing the same flag;
- [ ] Double buffering presets `available` before the loop and re-sets it each round;
- [ ] A pipe pair has no more than 8 flags, and there are no more than 16 cross flags;
- [ ] Every `pipe_barrier` pipe is allowed in its region;
- [ ] The kernel ends with `pipe_barrier(tla.pipes.ALL)`;
- [ ] Cross-core kernels check mode, `aiv_id`, and `core_ratio`;
- [ ] AutoSync is not mixed with manual flags or mutexes.
