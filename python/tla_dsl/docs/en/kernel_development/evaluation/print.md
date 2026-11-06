---
nav_order: 30
---

# Using the Print Debugging Interface in CATLASS Python

CATLASS Python provides the `tla.print` interface for debugging output inside a kernel. `tla.print` supports the following forms:

| Usage       | Form                                              | Description                                 |
| ----------- | ------------------------------------------------- | ------------------------------------------- |
| Scalar printing | `tla.print(value)` / `tla.print(fmt, *args)`      | Prints a single scalar, plain string, or format string |
| Tensor printing | `tla.print(tensor)` / `tla.print(tensor, length)` | Prints a contiguous physical prefix from the Tensor view starting address |
| SIMD register printing | `tla.print(values)` / `tla.print(values, length)` | Prints elements of a `VectorSSA` value |

Scalars, Tensors and registers support `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16` and `u32`.

`tla.print` accepts positional arguments only, and `length` must be an integer element count. For a Tensor, omit `length` only when its shape is static and all its elements fit one call; otherwise specify the length explicitly.

Here, "outer" means code inside `tla.cube()` or `tla.vector()` but outside `tla.vec.func`.

## Usage Examples

The following examples use `basic_mmad_kernel` from `examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py`.

### Scalar Printing

Scalar printing is suitable for inspecting runtime integers, floating-point values, block scheduling, and execution flow.

```diff
# examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(123)  # Print a constant
+       tla.print("Hello world")  # Print a string
        tla.set_flag(l1a0_available)
        ...

        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
+           tla.print(
+               "tile={} row={} col={}", block_linear, block_row, block_col
+           )  # Print a formatted string
            ...
            for k_l1 in k_l1_range:
                ...
                for k_l0 in k_l0_range:
                    ...
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
+                   if k_l1 == 0 and k_l0 == 0:
+                       tla.print("first mmad done")
```

The code above shows three forms of scalar printing:

1. `tla.print(123)`: Prints a single scalar.
2. `tla.print("Hello world")`: Prints a plain string.
3. `tla.print("tile={} ...", ...)`: Uses formatted printing for runtime values such as `block_linear`, `block_row`, and `block_col`.

Plain strings and format strings must be compile-time ASCII strings. A format string accepts at most eight `{}` scalar fields; `{{` and `}}` print literal braces.
A complete SIMD formatted record is limited to 2024 bytes, including its header, argument slots, string terminator and alignment. Larger records produce a compile-time error.

### Tensor Printing

Tensor printing is suitable for inspecting data in GM or UB.

```diff
# examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(gm_a, 256)
        tla.set_flag(l1a0_available)
        ...
```

`tla.print(gm_a, 256)` prints 256 consecutive elements of GM input A starting from its valid address. With the default Basic MMAD parameters, `gm_a` has dtype `f16` and shape `[256, 1024]`.

### Tensor and Register Printing Support

| Call location | Printable data | Per-call range |
| --- | --- | --- |
| Outer `tla.cube()` | GM Tensor | Rank-1/rank-2 Tensor, at most 262112 elements |
| Outer `tla.vector()` | GM or UB Tensor | Rank-1/rank-2 Tensor, at most 262112 elements |
| SIMD `tla.vec.func` | Register | One 256-byte register: at most 64 FP32 or 128 FP16 elements |
| SIMD `tla.vec.func` | UB Tensor | Rank-1/rank-2 Tensor, at most 1952 data bytes: 488 FP32 or 976 FP16 elements |
| SIMT `tla.vec.func` | GM or UB Tensor | Statically shaped rank-1/rank-2 Tensor, at most 262112 elements |

The limits above have different sources. A vector register on the current Ascend 950 target holds 256 bytes. In CANN 9.2, `asc_debug_types.h` reserves 2048 UB bytes for VF debugging. The 24-byte control header leaves 2024 bytes for a complete record; subtracting the 72-byte Tensor record header leaves 1952 bytes for data.

The 262112-element GM/UB limit follows the uniform safety limit used by outer printing: subtract the 48-byte shape record header and 72-byte Tensor record header from a 1 MiB FIFO, round the data area down to a 32-byte boundary, and count FP32 elements. The same element limit applies to other supported dtypes and SIMT.

Rank-1/rank-2 support and the unsigned 32-bit encoding of SIMT shape dimensions are software conventions of the current frontend and log protocol.

- Statically known Tensor dimensions must be positive.
- Omitting `length` prints all Tensor elements, which requires a static shape and an element count within the per-call limit in the table.
- A compile-time integer `length` must be positive and must not exceed the Tensor element count or per-call limit; otherwise compilation fails. A nonpositive runtime length prints nothing.
- Inside SIMD/SIMT, a positive runtime length is clamped to the Tensor element count and per-call limit. Outside `vec.func`, a runtime length exceeding either limit emits no Tensor record.
- GM/UB Tensor printing reads a contiguous physical prefix from the view's starting address, without following strides or rearranging layouts. For example, with underlying data `[0, 1, 2, 3, 4, 5, 6, 7]`, a shape-4, stride-2 view has logical elements `[0, 2, 4, 6]`, while `tla.print(view, 4)` prints `[0, 1, 2, 3]`. The caller ensures that this contiguous memory is readable. In SIMT, reading `view[index]` and printing that scalar follows the view's stride.
- Outer UB printing requires the view starting address to be 32-byte aligned. An unaligned address emits no Tensor record.
- SIMT Tensors require static shapes with each dimension in `1..4294967295`; larger dimensions produce a compile-time error.
- Omitting a register length uses its compile-time valid element count, or its physical capacity when that count is dynamic. An explicit static length is in `1..physical capacity`; a runtime length is clamped to that capacity, and a nonpositive length prints nothing. Values in uninitialized register elements are unspecified.

### Synchronization and Buffer Reuse

Inside a SIMD/SIMT `vec.func`, the library handles print reads, log writes and publication, using the necessary memory barriers. A successfully written log keeps the values read by that call; later overwrites of the source UB do not change it. Before printing, complete transfers or writes from other threads/cores. Other execution units must not overwrite the same data while printing reads it.

Within the same SIMD `vec.func`, printing uses internal barriers to order its reads against preceding and subsequent Vector stores, so no extra `tla.local_mem_bar` is needed for printing. The following code is inside `tla.vector()`, both UB Tensors have been allocated, and the input is ready:

```python
with tla.vec.func(mode="simd"):
    values = input_ub.load() * 2.0
    tla.print(values, 4)
    output_ub.store(values)
    tla.print(output_ub, 4)
    output_ub.store(values + 1.0)
```

The UB log retains the first four elements of `values`; the last store does not change the saved record.

In SIMT, `tla.print` runs only in the calling thread. Select printing threads with a condition such as `if thread == 0:`. The library protects log writes with atomics and memory barriers, without an implicit block-wide barrier. Printing only a thread's own scalar values does not require an extra block-wide barrier. Before printing shared UB written by other threads, the whole thread block executes a barrier. If other threads will overwrite that UB afterwards, all threads execute another barrier after printing to wait for the read to finish:

```python
with tla.vec.func(mode="simt", thread_block_dim=64):
    thread, _, _ = tla.arch.thread_idx()
    shared_ub[thread] = source_gm[thread] * 2.0
    tla.arch.sync_threads()
    if thread == 0:
        tla.print(shared_ub, 8)
    tla.arch.sync_threads()
    shared_ub[thread] = shared_ub[thread] + 1.0
    output_gm[thread] = shared_ub[thread]
```

The example is inside `tla.vector()`. Each GM/UB Tensor has 64 FP32 elements, and UB is allocated outside `vec.func`. Both barriers are outside `if thread == 0:` and are executed by every thread in the block. Placing them inside the branch can deadlock because the other threads do not reach them. If no other thread will overwrite the data afterwards, printing alone does not require the second barrier.

Users still express dependencies across transfers, pipelines, cores, and streams. Internal print barriers protect only print accesses. Printing changes timing; the algorithm must remain correctly synchronized after printing is removed.

## Building and Running

Scalar, GM/UB Tensor, and VF prints can be combined in one kernel without additional compilation options:

```bash
cd examples/python/tla_dsl/tla/kernel/matmul/basic_mmad

# Launch a single block to make the output easier to inspect
python basic_matmul.py --device 0 --block-num 1
```

## Example Output

### Scalar Printing Output

Scalar, plain-string, and formatted output all use the `TLA printf:` prefix. The following shows the outer print format:

```text
TLA printf: core=0 block=0 x=123
TLA printf: core=0 block=0 Hello world
TLA printf: core=0 block=0 tile=0 row=0 col=0
TLA printf: core=0 block=0 first mmad done
...
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

SIMT scalar output includes block and thread coordinates. For example, thread 0 in `simt_print.py` prints:

```text
TLA printf: block=(0, 0, 0) thread=(0, 0, 0) tid=0 x=0.000000
```

### Tensor Printing Output

Outer and SIMD Tensor output uses the `DumpTensor:` prefix. The following shows the format for `tla.print(gm_a, 256)`, with some values omitted:

```text
DumpTensor: call=0, block=0, data_type=float16, position=GM, shape=[256,1024] dump_size=256 [-1.66015625, -2.26953125, 3.0703125, 0.9375, ...]
```

`shape` is the input Tensor view's shape, and `dump_size` counts the elements printed by this call. For a prefix, these element counts may differ. SIMD register output uses position `REG`; Vector records in a MIX kernel also include `subblock` to distinguish Vector subblocks of the same logical block.

SIMT Tensor output also uses `DumpTensor:`, with block/thread coordinates. For example, the first eight shared-UB elements in `simt_print.py` are:

```text
DumpTensor: block=(0, 0, 0) thread=(0, 0, 0) dtype=float32 shape=(64,) [0, 2, 4, 6, 8, 10, 12, 14]
```

Actual values and artifact paths depend on the input and runtime environment. Output order across threads, cores, and print channels is unspecified.

## Notes

- Printing is for debugging and adds log buffers, data transfers, and synchronization. The host synchronizes the launch stream before reading output. Remove print calls before production runs or performance measurements; skipping them through a runtime condition can retain these costs.
- SIMD printing requires CANN VF debug support and the default reserved UB configuration.
- Tensor printing supports GM, UB, and L1/L0C Tensors in outer `tla.cube()` regions on C310. For L1, only a 32-byte-aligned zN/nZ dense prefix is supported, the C0 dimension must occupy 32 bytes, and at most 8 elements can be printed; for L0C, only a 32-byte-aligned L0Clayout 16x16 complete fractal (256 elements) is supported. Both support `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16`, and `u32`. This local-buffer path depends on CANN's private DebugTunnel ABI and has been validated with CANN 9.2.0. A CANN upgrade requires checking that ABI and validating L1/L0C printing again. L1 additionally requires the driver to provide `halGetMaxResMapType`, `halResMap`, and `halResUnmap`. A kernel cannot combine L1 and L0C printing. L0A/L0B are not currently supported.
- If an error or interruption leaves it unknown whether the device has finished accessing print buffers, the runtime retains them to avoid freeing memory still in use. Continue debugging in a new process to avoid accumulating device allocations through repeated launches. Confirmed completion allows normal release.
- Logs store complete records. If the remaining space cannot hold one record, existing complete records are kept and kernel computation continues, as follows.

| Print location | Log capacity | When space is insufficient |
| --- | --- | --- |
| Outer and SIMD | Shared data area with 108 fixed 1 MiB partitions: 108 MiB total, plus control metadata | That partition stops accepting later records and reports `output truncated` with its core identifier; other partitions continue |
| SIMT | A separate 1 MiB data area per launch, shared by all blocks/threads, plus a 64-byte control header | Drops the call that does not fit and subsequent calls; reports their count as `dropped N print call(s)` |

Capacity includes record headers and alignment. One Tensor print is one record; it does not retain only part of the requested elements. Outer Tensor-only kernels reject statically known aggregate output exceeding capacity before launch.

Fewer print calls or a shorter `length` reduce log volume. The example option `--block-num 1` reduces the number of cores printing concurrently; it does not reduce the fixed outer/SIMD log allocation or guarantee fewer printed lines within one tile.

Complete VF calls are shown in the [printing examples](../../../../../../examples/python/tla_dsl/dsl_tutorials/debug_print/README_en.md). CANN debugging interfaces:

- [Ascend C DumpTensor](https://www.hiascend.com/en/document/redirect/CannCommunityascendcapidumptensor)
- [Ascend C printf](https://www.hiascend.com/en/document/redirect/CannCommunityascendcapiprintf)
