---
nav_order: 30
---

# Calling bc to Integrate the Entire Python Backend — Taking tla.copy as an Example

---

## Background

### The conventional lowering path

In a pure MLIR compiler, the standard lowering path for `tla.copy` is to use a RewritePattern in a conversion pass to **expand it directly** into HIVM DMA ops:

```
tla.copy  (high-level move op)
   │
   │  expanded directly by the conversion pass's RewritePattern
   ▼
HIVM DMA op   (hivm.hir.load / nd2nz)
   │
   │  HIVM -> Standard -> LLVM
   ▼
LLVM IR (DMA intrinsic)
   │
   ▼
kernel.o
```

**"How to move" is entirely decided by the MLIR C++ lowering pattern** — each additional `(source layout, destination layout, element type)` combination requires writing another piece of construction logic in the pass.

### The C++ version already implements these move specializations

The CATLASS C++ template library (repository root `include/catlass`) was built for high-performance matmul-like operators, and data movement is one of its core capabilities. Taking GM->L1 as an example, `TileCopyTla` in [copy_gm_to_l1.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp) is a template partially specialized over multiple dimensions `(Arch, source Tensor, destination Tensor)`:

```cpp
// Partial specialization #1: RowMajor input -> zN output, via Nd2Nz
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, ...>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, TPosition::GM>,
    tla::Tensor<LocalTensor<ElementDst>,  LayoutDst, CoordDst, TPosition::A1>,
    std::enable_if_t<isRowMajor<LayoutSrc> && iszN<ElementDst, LayoutDst>>> {
    void operator()(dstTensor, srcTensor, ...) {
        AscendC::Nd2NzParams intriParams;
        intriParams.nValue = ...;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;  // C0 alignment conversion
        ...
        AscendC::DataCopy(dst, src, intriParams);   // Nd2Nz variant
    }
};
// Partial specialization #2: zN -> zN, via ordinary DataCopy(repeatParams)
// Partial specialization #3: ColumnMajor -> nZ
// Partial specialization #4/#5: float4_e2m1x2 / float4_e1m2x2 narrow-precision packing
// Partial specialization #6+: mixed element types ……
```

This one file alone has **8+ partial specializations**, handling RowMajor->zN, zN->zN, ColumnMajor->nZ, float4 narrow-precision packing, mixed element types, and other combinations, each precisely constructing `Nd2NzParams` / `DataCopyParams` and handling the C0 size, alignment, and stride conversions. There is also a set of similar templates:

- [copy_l0c_to_gm.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp) (`CopyL0CToGmTla`, via `AscendC::Fixpipe`, supporting f32->f16/bf16 quantization and the `unit_flag` feature)
- [copy_l0c_to_ub.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp) (`CopyL0CToUBTla`, with split_m / split_n)
- [copy_l1_to_l0a.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp) / [copy_l1_to_l0b.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp) (L1 -> L0A/L0B)
- vector-side [copy_gm_to_ub_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub_tla.hpp) / [copy_ub_to_gm_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm_tla.hpp) / [copy_ub_to_l1_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp)

These templates are the fruit of C++ library tuning and encode a large amount of hardware detail (C0 stride, the choice between Nd2Nz and DataCopy, Fixpipe's quantization/ReLU/unit_flag). **If CATLASS Python reimplemented them in MLIR passes, it would be enormous duplicated effort and prone to diverging from the C++ library implementation — the same move operator would have two sets of logic, with extremely high long-term maintenance cost.**

### The reuse idea: the contract lives in MLIR, the implementation in C++ templates, and bc is the bridge

The choice of CATLASS Python is to **reuse rather than reimplement**. The reuse mechanism decouples "what to call" from "how it is implemented":

| Layer | Responsibility | Location |
|----|------|---------|
| MLIR lowering | Decides only "which stub to call and what payload to pass" (the **contract**) | passes such as [TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) |
| C++ stub | Unpacks the payload and calls the corresponding `TileCopyTla` template (the **implementation entry**) | [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) and similar |
| C++ template | Actually constructs DataCopyParams and calls AscendC (the **implementation body**) | [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) |
| bc | Compiles the stubs into linkable bitcode (the **bridge**) | `meta_op.*.c310.bc` |
| hivmc | Links the bc into the kernel at runtime (the **welding**) | `--link-aicore-bitcode` |

On the MLIR side, it only needs to start a well-defined stub name for each `(route, layout, element type)` (such as `copy_gm_RowMajor_to_l1_zN_float`) and assemble the payload in a fixed layout; whether that stub internally uses Nd2Nz or Fixpipe and how the C0 stride is computed are all left to the C++ template. The stub is declared `always_inline` and is inlined into the call site after linking — **the final `kernel.o` is essentially no different from the DMA intrinsic produced by "pure MLIR lowering"**, but the development cost and correctness guarantees are far better.

This is the fundamental motivation for calling bc to integrate the whole backend: **use the smallest MLIR-side contract to reuse the full move-specialization capability of the C++ template library.** The rest of this document uses `tla.copy` as the thread to unpack this chain step by step.

---

## The Big Picture: Where bc Sits in the Compilation Chain

```
Python kernel fn
   │  (tla_ast_decorators.py AST walk + core_api.py)
   ▼
tla MLIR dialect IR        ── tla.copy / tla.mmad / tla.vec.* …
   │  TlaCompile pass pipeline (PassRegistry.cpp::buildTlaPipeline, 22 passes)
   ▼
HIVM / HIVMAVEIR + func.call @copy_<route>_<dtype>     ◀── the crack is here
   │  (CompilePipeline.cpp: llvmPm: FuncToLLVM + MemRefToLLVM + ArithToLLVM)
   ▼
LLVM IR   (call @_mlir_ciface_copy_<route>_<dtype>)
   │
   │  ┌──────────────────────────────────────────────────────────────┐
   │  │  bc access point: hivmc --link-aicore-bitcode=meta_op.{aic,aiv}.c310.bc │
   │  │  the stub bodies come from precompiled bitcode (csrc/mlir/bc/{Cube,Vector}/*.cpp)  │
   │  └──────────────────────────────────────────────────────────────┘
   ▼
kernel.o   (the stub is inlined as a DMA intrinsic by always_inline)
```

**Core realization**: during TlaCompile, `tla.copy` does **not** lower into a single HIVM datacopy op; it lowers into a `func.call` that invokes a runtime stub named like `copy_gm_RowMajor_to_l1_zN_float`. The stub's body is written in C++ ([csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp)), precompiled into `.bc` by the Bisheng compiler `bisheng`, and then linked into the kernel at compile time by `hivmc`. **bc is the glue that fills that crack.**

---

## What bc Is

"bc" means LLVM **bitcode** (`.bc` files):

| File | Content | Source |
|------|------|------|
| `meta_op.aic.c310.bc` | the bodies of all runtime stubs for the cube core (AIC) | compiled from this project's `csrc/mlir/bc/Cube/*.cpp` + merged with `llvm-link` |
| `meta_op.aiv.c310.bc` | the bodies of all runtime stubs for the vector core (AIV) | compiled from this project's `csrc/mlir/bc/Vector/*.cpp` + merged with `llvm-link` |

These functions include: DMA movement (`dma.cpp`), matrix multiplication (`mmad.cpp`), debug printing (`print_tensor.cpp`), mutexes (`mutex.cpp`), RegStore (`store.cpp`), and so on. They are the "implementation layer" between CATLASS Python and the AscendC hardware API: the CATLASS Python pipeline only generates the IR that calls them, and how the data is actually moved is decided by the C++ templates such as `AscendC::DataCopy` / `AscendC::Fixpipe` inside the stubs (see Section 1); when `bisheng` compiles a stub, it inlines these AscendC calls out into DMA intrinsics.

---

## Following `tla.copy` Through the Whole Flow

### User side: writing one line of `tla.copy`

Taking [tla/kernel/matmul/basic_mmad/basic_matmul.py](../../../../../../examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py) as an example, here are the three typical copies in a cube kernel:

```python
with tla.cube():
    # GM -> L1: move a tile from global memory into an L1 buffer
    tla.copy(l1_a, gm_a_by_l1)
    # L1 -> L0A: feed the L1 tile into the A input of the multiplier
    tla.copy(l0_a, l1_a_by_l0)
    # L0C -> GM: write the accumulator result back to global memory
    tla.copy(gm_c_by_core, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
```

The Python constructor of `tla.copy` is in [core_api.py](../../../../catlass/core_api.py) (the `copy` function, lines 3803–3914). Key design points:

- **The route is implicitly determined by the address space on the tensor type**, not by an independent argument of copy. The `!tla.ptr<dtype, addrspace, align>` of `dst`/`src` carries `gm`/`l1`/`l0a`/`l0b`/`l0c`/`ub`, and the frontend reads the `(src_as, dst_as)` pair to look up the route table ([core_api.py](../../../../catlass/core_api.py) lines 3796–3800).
- **Legal route table** (double-locked by a frontend soft check + a C++ verifier):

  | Route | Owning region | Type |
  |------|------------|------|
  | GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L1->UB | `tla.cube` | cube data path |
  | GM->UB, UB->GM, UB->L1 | `tla.vector` | vector data path |

- **Optional `params`**: only L0C-source routes materialize a third IR operand (`!tla.copy_l0c2dst_params`), carrying `unit_flag` / `quant_mode` / `l0c2ub_mode` and so on; non-L0C sources omit it. `atomic_mode` is always read from `params.atomic_mode` and emitted as an op attribute. See `CopyL0C2DstParams` / `CopyUbToGmParams` in [params.py](../../../../catlass/params.py) for details.

### The tla IR emitted by the frontend

The IR generated by the `tla.copy(l1_a, gm_a_by_l1)` above (GM->L1) looks like:

```
"tla.copy"(%dst_l1_zN, %src_gm_RowMajor) : (!tla.tensor<..., !tla.ptr<f32, l1, ...>>,
                                            !tla.tensor<..., !tla.ptr<f32, gm, 4>>) -> ()
```

The op is defined in [Tla.td](../../../../csrc/mlir/include/Dialect/Tla/IR/Tla.td) (`Tla_CopyOp`, lines 583–601).

### Lowering: `tla.copy` -> `func.call` stub

This is the key step of bc integration. `tla.copy` is eliminated in two passes (one for the cube route and one for the vector route, with symmetric logic):

- **cube side**: `LowerTlaCopyPattern` in [TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) (lines 197–400)
- **vector side**: `LowerCopyPattern` in [TlaVectorRegionCommon.cpp](../../../../csrc/mlir/lib/Passes/TlaVectorRegionCommon.cpp) (lines 110–199)

What they do:

1. Take the src/dst `TensorDescriptor` from `tla.tensor_desc` (materialized by the earlier `tla-lower-tensor-desc`).
2. Read `(srcAddrspace, dstAddrspace, srcLayout, dstLayout, elemType)` and call `getCopyRouteCallee()` ([TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) lines 606–733) to look up the stub name, for example `copy_gm_RowMajor_to_l1_zN_float`, `copy_l0c_to_gm_RowMajor_float`.
3. `buildCopyPayloadForRoute()` (lines 767–782 of the same file) packs shape/stride/coord into a series of `i64` payloads (8 for linear layout, 12 for packed layout, one set each for src and dst).
4. Generate `func.call @copy_<route>_<dtype>(srcMemref, dstMemref, ...payload)` and delete the original `tla.copy`.

The converted MLIR (`tla-compile --emit mlir`) looks like:

```
func.func private @copy_gm_RowMajor_to_l1_zN_float
  {hivm.func_core_type = #hivm.func_core_type<AIC>, ...}
...
call @copy_gm_RowMajor_to_l1_zN_float(...)
```

### bc side: what a stub body looks like

The stubs are implemented in [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) (cube side) and [csrc/mlir/bc/Vector/dma.cpp](../../../../csrc/mlir/bc/Vector/dma.cpp) (vector side). Taking the GM->L1 stub as an example (Cube/dma.cpp:131):

```cpp
void _mlir_ciface_copy_gm_RowMajor_to_l1_zN_float(
    memref_t<__gm__ float, 2> *src, memref_t<__cbuf__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0, int64_t srcStride1,
    int64_t srcCoord0, int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    /* ... dstStride / dstCoord / dstOrgShape ... */) {
    copyGMRowMajorToL1zN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{...}, TensorDesc4D{...});
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyGMRowMajorToL1zN(
    memref_t<__gm__ T, 2> *src, memref_t<__cbuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeGMRowMajorTensor(src, srcDesc);
    auto dstTensor = makeL1zNTensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}
```

Key points:

- **Naming convention**: the stub name is `_mlir_ciface_` + the call target in MLIR. The `_mlir_ciface_` prefix is MLIR's C ABI convention (`func.emit_c_interface`), which expands the memref descriptor into C function parameters field by field.
- **Parameter layout**: the first two are the src/dst memref descriptor pointers (annotated by address space `__gm__` / `__cbuf__` / `__ca__` / `__cb__` / `__ub__`), followed by a series of `int64_t` payloads — exactly the shape/stride/coord assembled by `buildCopyPayloadForRoute` in Section 4.3.
- **Implementation delegation**: the stub reassembles the payload into `TensorDesc2D`/`TensorDesc4D` and calls `Catlass::Gemm::Tile::TileCopyTla` ([copy_gm_to_l1.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)), which inside the template finally reaches `AscendC::DataCopy`. L0C->GM goes through `CopyL0CToGmTla` -> `AscendC::Fixpipe`. **This step "plugs in" the C++ template specializations from Section 1.**
- **The role of `bisheng`**: when compiling this `.cpp`, the Bisheng compiler inlines `AscendC::DataCopy`/`Fixpipe` out into DMA intrinsics (in LLVM IR form). So what is stored in the `.bc` is not the C++ source semantics but IR already expanded down to hardware intrinsics.

### Linking: `hivmc --link-aicore-bitcode` plugs the bc into the kernel

The last step happens in Python. `_build_hivmc_a5_command` in [execution.py](../../../../catlass/execution.py) (line 1832) assembles the hivmc command:

```python
command = [
    str(hivmc), str(mlir_path),
    "--target=Ascend950PR_9589",
    "--disable-ffts",
    f"--link-aicore-bitcode={template_bitcode}",  # ◀── bc is integrated here
    "-o", str(kernel_path),
]
```

- `--link-aicore-bitcode=` accepts one or more `.bc` paths (comma-separated). hivmc links the stub bitcode into the current kernel's LLVM module and then lowers it into `kernel.o`; after linking, the stub body is inlined into the call site — the final `kernel.o` has no stub calls left, only the expanded DMA intrinsics.

---

## The bc Build Flow (`bisheng` + `llvm-link`)

The bc build rules are in [csrc/mlir/bc/CMakeLists.txt](../../../../csrc/mlir/bc/CMakeLists.txt), in two steps:

**Step 1: each `.cpp` -> a single `.bc`** (`compile_single_cpp_to_bc`)

```bash
bisheng -O2 -x cce --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off \
     --cce-aicore-arch=dav-c310-cube \   # cube core; the vec core uses dav-c310-vec
     -DCATLASS_ARCH=3510 \
     -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl \
     -I${CATLASS_DSL_CATLASS_INCLUDE_DIR} \
     dma.cpp -emit-llvm -c -mllvm -disable-llvm-optzns \
     -o dma.aic.c310.bc
```

- `bisheng` is the Bisheng compiler (`${ASCEND_HOME_PATH}/bin/bisheng`), and `-x cce` makes it use the CCE frontend.
- `--cce-aicore-arch=dav-c310-{cube,vec}` specifies the core type and decides which intrinsic set the AscendC API expands into.
- `-mllvm -disable-llvm-optzns` turns off LLVM optimization, keeping "raw" linkable bitcode, and leaves optimization to be done uniformly by hivmc at the end.

**Step 2: `llvm-link` merges into `meta_op.*.bc`** (`add_custom_llvm_link_command` / `link_all_to_meta_op`)

```bash
llvm-link dma.aic.c310.bc mmad.aic.c310.bc mutex.aic.c310.bc ... \
          -o meta_op.aic.c310.bc
```

- Merge all stub bc of the same core type into a single `meta_op.{aic,aiv}.c310.bc`, which is exactly the file that hivmc's `--link-aicore-bitcode` consumes.
- The artifact lands in `csrc/mlir/build/bc/meta_op.{aic,aiv}.c310.bc`.

---

## How to Extend: Adding a New copy Route

Having understood the bc integration mechanism, adding a new route (for example a new layout combination) is a "three-way alignment":

1. **Route table**: add a match in `getCopyRouteCallee` (lines 606–733) of [TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) that returns the new stub name `copy_<src>_<slayout>_to_<dst>_<dlayout>_<elemSuffix>`.
2. **Stub implementation**: add `void _mlir_ciface_copy_<...>(...)` in [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) (or Vector/dma.cpp) that delegates to the corresponding `Catlass::Gemm::Tile::*Tla` template; if the template does not exist, first implement it under [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) — **this is exactly where the reuse value of the C++ library lies: in most cases the template already exists and only a layer of stub wrapping is needed.**
3. **Rebuild bc**: `ninja -C csrc/mlir/build bishengir_template_bitcode` (or a full `./build.sh`) so that `meta_op.*.c310.bc` contains the new stub.

If the route belongs to the existing legal set, the frontend route table in [core_api.py](../../../../catlass/core_api.py) (lines 3796–3800) and `CopyOp::verify` in [TlaOps.cpp](../../../../csrc/mlir/lib/Dialect/Tla/IR/TlaOps.cpp) (lines 594–638) need no change; if it is an entirely new address-space pair, both must be updated in sync.

> **Naming consistency is the crux**: the target symbol of the MLIR `func.call`, the string returned by `getCopyRouteCallee`, and the C function name of the `_mlir_ciface_` stub in the bc must all be exactly aligned, otherwise hivmc reports an undefined symbol at link time. This is also why the stubs all use `always_inline` + `emit_c_interface` — both to keep the symbol visible for linking and to ensure the call can be inlined away after linking to eliminate call overhead.

---

## Key File Index

| Stage | File | Key location |
|------|------|---------|
| op definition | [Tla.td](../../../../csrc/mlir/include/Dialect/Tla/IR/Tla.td) | `Tla_CopyOp` 583–601 |
| frontend construction | [core_api.py](../../../../catlass/core_api.py) | `copy` 3803–3914; route table 3796–3800 |
| parameter definition | [params.py](../../../../catlass/params.py) | `CopyL0C2DstParams` 163–204 |
| address space | [address_space.py](../../../../catlass/address_space.py) | 17–29 |
| C++ verifier | [TlaOps.cpp](../../../../csrc/mlir/lib/Dialect/Tla/IR/TlaOps.cpp) | `CopyOp::verify` 594–638 |
| pass pipeline | [PassRegistry.cpp](../../../../csrc/mlir/lib/Passes/PassRegistry.cpp) | `buildTlaPipeline` 42–82 |
| cube copy lowering | [TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) | `LowerTlaCopyPattern` 197–400 |
| vector copy lowering | [TlaVectorRegionCommon.cpp](../../../../csrc/mlir/lib/Passes/TlaVectorRegionCommon.cpp) | `LowerCopyPattern` 110–199 |
| route selection/payload | [TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) | `getCopyRouteCallee` 606–733 |
| C++ move templates | [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) | `copy_gm_to_l1.hpp`, `copy_l0c_to_gm.hpp`, etc. |
| stub implementation (cube) | [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) | GM->L1 stub from line 131 |
| stub implementation (vector) | [csrc/mlir/bc/Vector/dma.cpp](../../../../csrc/mlir/bc/Vector/dma.cpp) | - |
| bc build rules | [csrc/mlir/bc/CMakeLists.txt](../../../../csrc/mlir/bc/CMakeLists.txt) | `compile_single_cpp_to_bc` / `link_all_to_meta_op` |
| bc runtime linking | [execution.py](../../../../catlass/execution.py) | `_build_hivmc_a5_command` 1832; `_resolve_hivm_template_bitcode` 2003 |
| IR form samples | [tests/lit/copy/copy-l0c-to-gm-atomic-f32.mlir](../../../../tests/lit/copy/copy-l0c-to-gm-atomic-f32.mlir), [tests/lit/tensor/make-tensor-copy-gm-l1-zn.mlir](../../../../tests/lit/tensor/make-tensor-copy-gm-l1-zn.mlir) | `copy-l0c-to-gm-atomic-f32.mlir`, `make-tensor-copy-gm-l1-zn.mlir` |

---

## Appendix: One Diagram Summarizing the Full bc Integration Chain of `tla.copy`

```
tla.copy(gm_a -> l1_a)                        ← user Python
        │ core_api.copy() emit
        ▼
"tla.copy"(%l1, %gm) : <l1 zN>, <gm RowMajor>   ← tla dialect IR
        │ tla-cube-region pass (LowerTlaCopyPattern)
        │   getCopyRouteCallee() -> "copy_gm_RowMajor_to_l1_zN_float"
        │   buildCopyPayloadForRoute() -> 8*i64*2
        ▼
func.call @copy_gm_RowMajor_to_l1_zN_float(src, dst, …payload)   ← HIVM/LLVM IR
        │ CompilePipeline: FuncToLLVM -> call @_mlir_ciface_copy_…
        │ rewriteCifaceCallsWithWrappers -> always_inline wrapper
        ▼
LLVM IR (call @_mlir_ciface_copy_gm_RowMajor_to_l1_zN_float)
        │  ┌─────────────────────────────────────────────────────────┐
        │  │ hivmc --link-aicore-bitcode=meta_op.aic.c310.bc          │
        │  │   the stub body comes from Cube/dma.cpp (precompiled to .bc by bisheng) │
        │  │   -> TileCopyTla -> AscendC::DataCopy (already expanded to intrinsics) │
        │  └─────────────────────────────────────────────────────────┘
        ▼
kernel.o   (DMA intrinsics already inlined, no stub call leftovers)
```

**One-sentence summary**: the CATLASS Python pipeline generates the "call contract" (`func.call` + payload layout), bc provides the "call implementation" (C++ stub -> `TileCopyTla` template -> AscendC -> intrinsic), and `hivmc --link-aicore-bitcode` is the step that welds the two together. `tla.copy` merely uses the most intuitive "move data" operation to show this contract-implementation-welding chain — `tla.mmad`, `tla.store`, and all other ops that go through stubs follow the same bc integration mechanism.
