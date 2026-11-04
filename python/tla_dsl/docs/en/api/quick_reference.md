---
nav_order: 5
---

# CATLASS Python API Quick Reference

This task-based index covers common APIs for explicit hardware programming with CATLASS Python. It is not a complete API listing. See the [Program Lifecycle Reference](runtime/index.md) and [Language Reference](language/index.md) for full signatures and constraints.

Context labels: **Host** means a Python script outside `@tla.kernel`; **kernel** means inside a `@tla.kernel` function; **Cube** and **Vector** identify `tla.cube()` and `tla.vector()` regions; **SIMD** and **SIMT** identify the corresponding `tla.vec.func` modes.

## 1. Getting Started

| Entry | Purpose | Context or key constraint |
| --- | --- | --- |
| [Quick Start](../quick_start.md#catlass-python-quick-start) | Install the package and run the first kernel. | Host; check the product, CANN, and Python requirements. |
| [Syntax Constraints](../kernel_development/core_concepts/syntax_guide.md#dynamic-and-static-values-branches-and-loops) | Distinguish compile-time values and expansion from runtime values and control flow. | Kernel and `@tla.jit` helper code supports only the documented Python syntax. |
| [Layout](../kernel_development/core_concepts/layout.md#overview) | Understand static and dynamic layouts, logical extents, and physical storage. | Host layout metadata participates in compilation; Kernel code reads the resulting layout. |
| [Compile and Launch](../kernel_development/core_concepts/compile_and_launch.md#basic-flow) | Follow the compile-then-launch flow. | Host; compile with type samples and launch with tensors bound to real NPU buffers. |

## 2. Host Lifecycle

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`@tla.kernel`](runtime/kernel_entry.md#tlakernel) | Declare a kernel entry and obtain a `TlaJitFunction`. | Host decorator; the body is not executed on the Host, and direct invocation is disabled. Compile before launch. |
| [`@tla.jit`](runtime/kernel_entry.md#tlajit) | Declare a device helper inlined during kernel lowering. | Declared on the Host; no standalone `compile` or `dump_mlir`, and no recursive helper calls. |
| [`@tla.extern`](runtime/kernel_entry.md#tlaextern) | Declare an Ascend C ABI entry implemented by inline source. | Host declaration; calls belong in one Cube or Vector region and outside `tla.vec.func`. |
| [`tla.compile`](runtime/compile_launch.md#tlacompile) | Compile a kernel from type samples and return a `JitCompiledFunction`. | Host; samples need not be bound NPU buffers. Launch options such as `block_num` and `stream` belong to the compiled call. |
| [`tla.JitCompiledFunction.__call__`](runtime/compile_launch.md#tlajitcompiledfunction__call__) | Launch a compiled kernel and reuse its executor and binary. | Host; tensor arguments must be bound NPU buffers. `make_fake_tensor` results cannot be launched. |
| [`tla.from_dlpack`](runtime/host_tensor.md#tlafrom_dlpack) | Bind a DLPack tensor as a TLA Host tensor without copying device data. | Host; `layout_tag` is required. NPU buffers support launch; CPU buffers provide compile-only metadata. |
| [`tla.make_fake_tensor`](runtime/host_tensor.md#tlamake_fake_tensor) | Create a metadata-only compile sample. | Host; no device buffer and no launch. Shape-related inputs are Python integer trees, not Kernel layout values. |

## 3. Tensor, Layout, and Pointer

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`tla.make_shape`](language/layout_tensor.md#tlamake_shape) | Build a logical or nested physical shape. | Kernel; `zN`, `nZ`, `L0Clayout`, and `zNUnAlign` use nested physical shapes with explicit layout construction. |
| [`tla.make_coord`](language/layout_tensor.md#tlamake_coord) | Build a coordinate tree. | Kernel; the consuming API determines whether coordinates denote tiles or elements. |
| [`tla.make_stride`](language/layout_tensor.md#tlamake_stride) | Build a stride tree. | Kernel; nesting and values must match the shape and layout tag. |
| [`tla.make_layout`](language/layout_tensor.md#tlamake_layout) | Compose shape, stride, logical origin, and layout tag. | Kernel; physical layouts require nested shape and stride trees. |
| [`tla.make_layout_with_tag`](language/layout_tensor.md#tlamake_layout_with_tag) | Remap a flat 2D logical extent using an element type and layout tag. | Kernel; on-chip layouts only, for use with `make_tensor`. |
| [`tla.tile_view`](language/layout_tensor.md#tlatile_view) | Create a tile view without copying the source buffer. | Kernel; coordinates use tile granularity. |
| [`tla.get_tile`](language/layout_tensor.md#tlaget_tile) | Extract a flat 2D sub-tile that shares the source buffer. | Kernel; coordinates are element offsets. Cropping trims the extent but does not clamp out-of-range offsets. |
| [`tla.make_tensor`](language/layout_tensor.md#tlamake_tensor) | Combine a pointer, layout, and optional coordinate into a tensor. | Kernel; pointer dtype and address space must match. Runnable kernels require backing storage. |
| [`tla.make_tensor_like`](language/layout_tensor.md#tlamake_tensor_like) | Reuse reference-tile metadata on another pointer, optionally changing the layout tag. | Kernel; destination pointers must be on-chip. |
| [`tla.make_ptr`](language/layout_tensor.md#tlamake_ptr) | Construct a typed pointer from an address. | Kernel; integer address width must match the target address space. |
| [`tla.recast_ptr`](language/layout_tensor.md#tlarecast_ptr) | Reinterpret a pointer's element type. | Kernel; address and swizzle remain unchanged. |

## 4. Data Movement and Matrix Compute

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`tla.copy`](language/data_movement.md#tlacopy) | Copy tiles between supported global and on-chip memory routes. | Kernel / Cube or Vector; route, layout, dtype, alignment, and origin-shape rules apply. Not available inside SIMT. |
| [`tla.Tensor.load`](language/data_movement.md#tlatensorload) | Load a UB tile into vector or mask SSA. | Kernel / Vector / SIMD; inside `tla.vec.func`, with a UB source tile. |
| [`tla.Tensor.store`](language/data_movement.md#tlatensorstore) | Store vector or mask SSA into a UB tile. | Kernel / Vector / SIMD; inside `tla.vec.func`, with a UB destination tile. |
| [`tla.Tensor.fill`](language/data_movement.md#tlatensorfill) | Fill an L1 tile with zero. | Kernel / Cube; destination must be L1 `zN` or `nZ`, with the documented packed fp4/fp8 element types. |
| [`tla.Tensor.__getitem__`](language/data_movement.md#tlatensor__getitem__) | Read one element as a scalar. | Kernel; GM or UB, `RowMajor` or `ColumnMajor`, logical rank 1 or 2. UB access requires Vector; SIMT uses per-thread element access. |
| [`tla.Tensor.__setitem__`](language/data_movement.md#tlatensor__setitem__) | Write one scalar element. | Kernel; same tensor restrictions as element reads. A `Numeric` value must match the tensor element type exactly. |
| [`tla.mmad`](language/matrix.md#tlammad) | Multiply and accumulate matrix tiles. | Kernel / Cube; accumulator and operands must be matching L0 tiles with supported dtype combinations. |
| [`tla.mmad_mx`](language/matrix.md#tlammad_mx) | Multiply and accumulate microscaling matrix tiles. | Kernel / Cube; matching L0 tiles, fp32 accumulator, and both operands loaded by `tla.copy(..., scale=...)`. |

## 5. SIMD and SIMT Vector Compute

Both modes run in `tla.vec.func` nested inside `tla.vector()`:

- **SIMD** operates on whole vector registers and masks. Stage data in UB and use `Tensor.load` / `Tensor.store`; this mode fits regular, dense vector work.
- **SIMT** expresses the program of one thread through scalar arithmetic and element access. It uses `thread_block_dim`, permits only scalar runtime captures, and does not allow register-vector operations, mask operations, `Tensor.load` / `Tensor.store`, or `tla.copy` inside the region.

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`tla.vec.func`](language/scopes.md#tlavecfunc) | Select `simd` or `simt` vector execution. | Kernel / Vector; SIMT requires `thread_block_dim`, with at most 2048 threads per block. SIMD rejects this argument. |
| [`tla.create_mask`](language/simd/mask.md#tlacreate_mask), [`tla.update_mask`](language/simd/mask.md#tlaupdate_mask) | Create fixed-pattern or tail masks. | Kernel / Vector / SIMD; register-vector masks are not available in SIMT. |
| [`tla.add`](language/simd/arithmetic.md#tlaadd), [`tla.sub`](language/simd/arithmetic.md#tlasub), [`tla.mul`](language/simd/arithmetic.md#tlamul), [`tla.div`](language/simd/arithmetic.md#tladiv) | Perform element-wise arithmetic. | Kernel / Vector; SIMD uses `VectorSSA`, while SIMT uses per-thread `Numeric` scalars. Execution masks apply to vector forms. |
| [`tla.max`](language/simd/arithmetic.md#tlamax), [`tla.min`](language/simd/arithmetic.md#tlamin) | Compute element-wise maxima or minima. | Kernel / Vector; SIMD vector and SIMT scalar forms have their own dtype rules. |
| [`tla.exp`](language/simd/arithmetic.md#tlaexp), [`tla.log`](language/simd/arithmetic.md#tlalog), [`tla.sqrt`](language/simd/arithmetic.md#tlasqrt), [`tla.abs`](language/simd/arithmetic.md#tlaabs) | Apply unary math operations. | Kernel / Vector; SIMD uses vectors and SIMT uses scalars. Scalar `exp` and `log` are available only in SIMT. |
| [`tla.cmp`](language/simd/compare_select.md#tlacmp) | Compare vector elements and produce a `MaskSSA`. | Kernel / Vector / SIMD; SIMT uses scalar comparison operators that produce `Bool`. |
| [`tla.where`](language/simd/compare_select.md#tlawhere) | Select between two values under a condition. | Kernel / Vector; SIMD uses `MaskSSA`, while SIMT uses a per-thread `Bool`. |
| [`tla.full`](language/simd/fill.md#tlafull), [`tla.arange`](language/simd/fill.md#tlaarange) | Create a constant vector or a lane-index sequence. | Kernel / Vector / SIMD; these register-vector constructors are rejected in SIMT. |
| [`tla.VectorSSA.reduce`](language/simd/discrete_aggregate.md#tlavectorssareduce) | Reduce a vector under a mask to a one-lane fragment. | Kernel / Vector / SIMD; a mask is required. Use `full` to broadcast the fragment. |
| [`tla.VectorSSA.to`](language/simd/type_conversion.md#tlavectorssato) | Convert register-vector element types. | Kernel / Vector / SIMD; supported source and destination types are listed in the full entry. |
| [`tla.Numeric.to`](language/values_types.md#tlanumericto) | Convert a scalar element type. | Kernel; in SIMT this emits the per-thread conversion. There is no per-thread equivalent of register-vector `tla.cast`. |

## 6. Synchronization, Resources, and Debugging

### Synchronization

If you are unsure when synchronization is required, or whether to use a flag or a mutex,
start from the [Synchronization Guide](../kernel_development/core_concepts/sync.md).

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`tla.flag`](language/sync.md#tlaflag), [`tla.set_flag`](language/sync.md#tlaset_flag), [`tla.wait_flag`](language/sync.md#tlawait_flag) | Synchronize producer and consumer pipes within a core. | Kernel / Cube or Vector; pair set and wait operations on the flag. |
| [`tla.cross_flag`](language/sync.md#tlacross_flag), [`tla.cross_core_set_flag`](language/sync.md#tlacross_core_set_flag), [`tla.cross_core_wait_flag`](language/sync.md#tlacross_core_wait_flag) | Synchronize across cores. | Kernel / Cube or Vector; set and wait specify their pipes. Mode 4 requires `aiv_id` 0 or 1. |
| [`tla.pipe_barrier`](language/sync.md#tlapipe_barrier) | Insert a barrier on a hardware pipe. | Kernel / Cube or Vector; not available inside SIMT. |
| [`tla.mutex`](language/sync.md#tlamutex), [`tla.mutex_guard`](language/sync.md#tlamutex_guard), [`tla.mutex_lock`](language/sync.md#tlamutex_lock), [`tla.mutex_unlock`](language/sync.md#tlamutex_unlock) | Protect a named resource with explicit locking or a scoped guard. | Kernel / Cube or Vector for lock operations; a guard block must emit `copy` or `mmad` and must not explicitly lock or unlock. |
| [`tla.local_mem_bar`](language/sync.md#tlalocal_mem_bar) | Order accesses between local-memory types. | Kernel / Vector / SIMD; inside `tla.vec.func`, with a supported `MemType` pair. Not available in SIMT. |
| [`tla.arch.sync_threads`](language/system.md#tlaarchsync_threads) | Synchronize threads in one SIMT block. | Kernel / Vector / SIMT; every thread must reach the barrier. It is not a cross-core or pipe barrier. |

### Resources and Debugging

| API | Purpose | Context or key constraint |
| --- | --- | --- |
| [`tla.allocate`](language/resources.md#tlaallocate) | Allocate on-chip scratch and return a typed pointer. | Kernel; static shape and on-chip address space only. Allocate SIMT scratch outside the SIMT region. |
| [`tla.arch.block_idx`](language/system.md#tlaarchblock_idx), [`tla.arch.block_num`](language/system.md#tlaarchblock_num) | Read the launch block index or block count. | Kernel; these describe AI Core blocks, not SIMT threads. |
| [`tla.arch.thread_idx`](language/system.md#tlaarchthread_idx), [`tla.arch.thread_block_dim`](language/system.md#tlaarchthread_block_dim) | Read SIMT thread coordinates or thread-block extents. | Kernel / Vector / SIMT only. |
| [`tla.arch.get_capacity_in_bytes`](language/system.md#tlaarchget_capacity_in_bytes) | Query an on-chip memory capacity for the compile target. | Host or kernel; evaluated as a compile-time constant. On-chip address spaces only; compiler reservations still apply to usable UB. |
| [`tla.arch.get_dyn_ub`](language/system.md#tlaarchget_dyn_ub) | Obtain the dynamic UB pointer whose size is supplied at launch. | Kernel; at most one region per kernel. Pass launch `ub=` for the required bytes; kernel-side extents are not bounds-checked. |
| [`tla.TlaJitFunction.dump_mlir`](runtime/compile_launch.md#tlatlajitfunctiondump_mlir) | Return frontend TLA IR without building or launching a binary. | Host; type samples follow the same rules as compilation. |
| [`tla.print`](language/debug.md#tlaprint) | Print scalars or supported physical tensor prefixes from a kernel. | Kernel / Cube or Vector; not available inside SIMT. Tensor location, layout, dtype, prefix, and output limits apply. |
