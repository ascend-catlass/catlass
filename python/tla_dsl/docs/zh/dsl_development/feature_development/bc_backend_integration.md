---
nav_order: 30
---

# 调用 bc 接入整个 Python 后端 —— 以 tla.copy 为例

---

## 背景

### 常规的 lowering 路径

在一个纯 MLIR 编译器里，`tla.copy` 的标准 lowering 路径是在 conversion pass 里用 RewritePattern 把它**直接展开**成 HIVM DMA op：

```
tla.copy  (高层搬运 op)
   │
   │  conversion pass 的 RewritePattern 直接展开
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

**搬运的"怎么做"完全由 MLIR C++ lowering pattern 决定**--每多支持一种 `(源 layout, 目标 layout, 元素类型)` 组合，就要在 pass 里多写一段构造逻辑。

### C++ 版本已实现这些搬运特化

CATLASS C++ 模板库（仓库根目录 `include/catlass`）本就是为高性能 matmul 类算子打造的，数据搬运是其核心能力之一。以 GM->L1 为例，[copy_gm_to_l1.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp) 里的 `TileCopyTla` 是一个按 `(Arch, 源 Tensor, 目标 Tensor)` 多维偏特化的模板：

```cpp
// 偏特化 #1：RowMajor 输入 -> zN 输出，走 Nd2Nz
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, ...>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, TPosition::GM>,
    tla::Tensor<LocalTensor<ElementDst>,  LayoutDst, CoordDst, TPosition::A1>,
    std::enable_if_t<isRowMajor<LayoutSrc> && iszN<ElementDst, LayoutDst>>> {
    void operator()(dstTensor, srcTensor, ...) {
        AscendC::Nd2NzParams intriParams;
        intriParams.nValue = ...;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;  // C0 对齐换算
        ...
        AscendC::DataCopy(dst, src, intriParams);   // Nd2Nz 变体
    }
};
// 偏特化 #2：zN -> zN，走普通 DataCopy(repeatParams)
// 偏特化 #3：ColumnMajor -> nZ
// 偏特化 #4/#5：float4_e2m1x2 / float4_e1m2x2 窄精度打包
// 偏特化 #6+：混合元素类型 ……
```

光这一个文件就有 **8+ 个偏特化**，分别处理 RowMajor->zN、zN->zN、ColumnMajor->nZ、float4 窄精度打包、混合元素类型等组合，每个特化都精确构造 `Nd2NzParams` / `DataCopyParams`，处理 C0 大小、对齐、stride 换算。此外还有一批同类模板：

- [copy_l0c_to_gm.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp)（`CopyL0CToGmTla`，走 `AscendC::Fixpipe`，支持 f32->f16/bf16 量化、`unit_flag` 特性）
- [copy_l0c_to_ub.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp)（`CopyL0CToUBTla`，带 split_m / split_n）
- [copy_l1_to_l0a.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp) / [copy_l1_to_l0b.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp)（L1 -> L0A/L0B）
- vector 侧 [copy_gm_to_ub_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub_tla.hpp) / [copy_ub_to_gm_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm_tla.hpp) / [copy_ub_to_l1_tla.hpp](../../../../../../include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp)

这些模板是 C++ 库调优的成果，编码了大量硬件细节（C0 stride、Nd2Nz vs DataCopy 的选择、Fixpipe 的量化/ReLU/unit_flag）。**若 Python 在 MLIR pass 里重新实现一遍，既是巨大的重复劳动，又容易和 C++ 库的实现产生分歧--同一个搬运算子存在两套逻辑，长期维护成本极高。**

### 复用思路：契约在 MLIR，实现在 C++ 模板，bc 是桥梁

CATLASS Python 的选择是 **不重新实现，而是复用**。复用的机制就是把"调用什么"和"怎么实现"解耦：

| 层 | 职责 | 所在位置 |
|----|------|---------|
| MLIR lowering | 只决定"调用哪个 stub、传什么 payload"（**契约**） | [TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) 等 pass |
| C++ stub | 把 payload 拆包，调对应的 `TileCopyTla` 模板（**实现入口**） | [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) 等 |
| C++ 模板 | 真正构造 DataCopyParams、调 AscendC（**实现本体**） | [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) |
| bc | 把 stub 编译成可链接的 bitcode（**桥梁**） | `meta_op.*.c310.bc` |
| hivmc | 运行时把 bc 链接进 kernel（**焊接**） | `--link-aicore-bitcode` |

MLIR 侧只需为每种 `(路由, layout, 元素类型)` 起一个确定的 stub 名（如 `copy_gm_RowMajor_to_l1_zN_float`），并按固定布局拼好 payload；至于这个 stub 内部走 Nd2Nz 还是 Fixpipe、C0 stride 怎么算，全交给 C++ 模板。stub 被声明为 `always_inline`，链接后会被内联进调用点--**最终 `kernel.o` 里和"纯 MLIR lowering"产出的 DMA intrinsic 没有本质区别**，但开发成本和正确性保障好得多。

这就是 bc 接入整个后端的根本动机：**用最小的 MLIR 侧契约，复用 C++ 模板库全部的搬运特化能力。** 后文以 `tla.copy` 为线索，把这条链路逐步拆开。

---

## 全景图：bc 在编译链路中的位置

```
Python kernel fn
   │  (tla_ast_decorators.py AST walk + core_api.py)
   ▼
tla MLIR dialect IR        ── tla.copy / tla.mmad / tla.vec.* …
   │  TlaCompile pass pipeline (PassRegistry.cpp::buildTlaPipeline, 22 个 pass)
   ▼
HIVM / HIVMAVEIR + func.call @copy_<route>_<dtype>     ◀── 裂缝在这里
   │  (CompilePipeline.cpp: llvmPm: FuncToLLVM + MemRefToLLVM + ArithToLLVM)
   ▼
LLVM IR   (call @_mlir_ciface_copy_<route>_<dtype>)
   │
   │  ┌──────────────────────────────────────────────────────────────┐
   │  │  bc 接入点：hivmc --link-aicore-bitcode=meta_op.{aic,aiv}.c310.bc │
   │  │  stub 函数体来自预编译 bitcode（csrc/mlir/bc/{Cube,Vector}/*.cpp）  │
   │  └──────────────────────────────────────────────────────────────┘
   ▼
kernel.o   (stub 被 always_inline 内联展开为 DMA intrinsic)
```

**核心认知**：`tla.copy` 在 TlaCompile 阶段**不会** lower 成单个 HIVM datacopy op，而是 lower 成一条 `func.call`，去调用一个名字形如 `copy_gm_RowMajor_to_l1_zN_float` 的 runtime stub。这个 stub 的函数体写在 C++ 里（[csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp)），由 Bisheng 编译器 `bisheng` 预编译成 `.bc`，再由 `hivmc` 在编译时链接进 kernel。**bc 就是填上那道裂缝的胶水。**

---

## 什么是 bc

"bc" 即 LLVM **bitcode**（`.bc` 文件）：

| 文件 | 内容 | 来源 |
|------|------|------|
| `meta_op.aic.c310.bc` | cube 核（AIC）所有 runtime stub 的函数体 | 本项目 `csrc/mlir/bc/Cube/*.cpp` 编译 + `llvm-link` 合并 |
| `meta_op.aiv.c310.bc` | vector 核（AIV）所有 runtime stub 的函数体 | 本项目 `csrc/mlir/bc/Vector/*.cpp` 编译 + `llvm-link` 合并 |

这些函数包括：DMA 搬运（`dma.cpp`）、矩阵乘（`mmad.cpp`）、调试打印（`print_tensor.cpp`）、互斥锁（`mutex.cpp`）、RegStore（`store.cpp`）等。它们是 Python 与 AscendC 硬件 API 之间的"实现层"：Python 流水线只负责生成调用它们的 IR，具体怎么搬数据由 stub 内部的 `AscendC::DataCopy` / `AscendC::Fixpipe` 等 C++ 模板决定（见第 1 节）；而 `bisheng` 在编译 stub 时，会把这些 AscendC 调用内联展开成 DMA intrinsic。

---

## 以 `tla.copy` 为线索贯穿全流程

### 用户侧：写一行 `tla.copy`

以 [tla/kernel/matmul/basic_mmad/basic_matmul.py](../../../../../../examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py) 为例，一个 cube kernel 里典型的三段 copy：

```python
with tla.cube():
    # GM -> L1：把全局显存的 tile 搬到 L1 buffer
    tla.copy(l1_a, gm_a_by_l1)
    # L1 -> L0A：把 L1 tile 喂给乘法器 A 输入
    tla.copy(l0_a, l1_a_by_l0)
    # L0C -> GM：把累加器结果写回全局显存
    tla.copy(gm_c_by_core, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
```

`tla.copy` 的 Python 构造函数在 [core_api.py](../../../../catlass/core_api.py)（`copy` 函数，3803–3914 行）。关键设计：

- **路由由 tensor 类型上的地址空间隐式决定**，不是 copy 的独立参数。`dst`/`src` 的 `!tla.ptr<dtype, addrspace, align>` 里携带 `gm`/`l1`/`l0a`/`l0b`/`l0c`/`ub`，前端读出 `(src_as, dst_as)` 二元组查路由表（[core_api.py](../../../../catlass/core_api.py) 3796–3800 行）。
- **合法路由表**（前端软检查 + C++ verifier 双重锁死）：

  | 路由 | 归属 region | 类型 |
  |------|------------|------|
  | GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L1->UB | `tla.cube` | cube 数据通路 |
  | GM->UB, UB->GM, UB->L1 | `tla.vector` | vector 数据通路 |

- **可选 `params`**：仅 L0C 源路由才物化成第三个 IR operand（`!tla.copy_l0c2dst_params`），承载 `unit_flag` / `quant_mode` / `l0c2ub_mode` 等；非 L0C 源省略。`atomic_mode` 一律从 `params.atomic_mode` 读出，作为 op attribute emit。详见 [params.py](../../../../catlass/params.py) 的 `CopyL0C2DstParams` / `CopyUbToGmParams`。

### 前端 emit 的 tla IR

上面 `tla.copy(l1_a, gm_a_by_l1)`（GM->L1）生成的 IR 形如：

```
"tla.copy"(%dst_l1_zN, %src_gm_RowMajor) : (!tla.tensor<..., !tla.ptr<f32, l1, ...>>,
                                            !tla.tensor<..., !tla.ptr<f32, gm, 4>>) -> ()
```

op 定义在 [Tla.td](../../../../csrc/mlir/include/Dialect/Tla/IR/Tla.td)（`Tla_CopyOp`，583–601 行）。

### Lowering：`tla.copy` -> `func.call` stub

这是 bc 接入的关键一步。`tla.copy` 在两个 pass 里被消除（cube 路由 / vector 路由各一个，逻辑对称）：

- **cube 端**：[TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) 的 `LowerTlaCopyPattern`（197–400 行）
- **vector 端**：[TlaVectorRegionCommon.cpp](../../../../csrc/mlir/lib/Passes/TlaVectorRegionCommon.cpp) 的 `LowerCopyPattern`（110–199 行）

它们做的事：

1. 从 `tla.tensor_desc`（由更早的 `tla-lower-tensor-desc` 物化）取 src/dst 的 `TensorDescriptor`。
2. 读出 `(srcAddrspace, dstAddrspace, srcLayout, dstLayout, elemType)`，调 `getCopyRouteCallee()`（[TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) 606–733 行）查表得到 stub 名，例如 `copy_gm_RowMajor_to_l1_zN_float`、`copy_l0c_to_gm_RowMajor_float`。
3. `buildCopyPayloadForRoute()`（同文件 767–782 行）把 shape/stride/coord 拼成一串 `i64` payload（linear layout 8 个、packed layout 12 个，src+dst 各一份）。
4. 生成 `func.call @copy_<route>_<dtype>(srcMemref, dstMemref, ...payload)`，原 `tla.copy` 删除。

转换后的 MLIR（`tla-compile --emit mlir`）形如：

```
func.func private @copy_gm_RowMajor_to_l1_zN_float
  {hivm.func_core_type = #hivm.func_core_type<AIC>, ...}
...
call @copy_gm_RowMajor_to_l1_zN_float(...)
```

### bc 侧：stub 的函数体长什么样

stub 实现在 [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp)（cube 端）和 [csrc/mlir/bc/Vector/dma.cpp](../../../../csrc/mlir/bc/Vector/dma.cpp)（vector 端）。以 GM->L1 的 stub 为例（Cube/dma.cpp:131）：

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

要点：

- **命名约定**：stub 名是 `_mlir_ciface_` + MLIR 里的 call 目标。`_mlir_ciface_` 前缀是 MLIR 的 C ABI 约定（`func.emit_c_interface`），让 memref descriptor 按字段展开成 C 函数参数。
- **参数布局**：前两个是 src/dst 的 memref descriptor 指针（按地址空间 `__gm__` / `__cbuf__` / `__ca__` / `__cb__` / `__ub__` 标注），后面跟一串 `int64_t` payload--正是 4.3 节 `buildCopyPayloadForRoute` 拼出来的那些 shape/stride/coord。
- **实现委托**：stub 把 payload 重新组装成 `TensorDesc2D`/`TensorDesc4D`，调 `Catlass::Gemm::Tile::TileCopyTla`（[copy_gm_to_l1.hpp](../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)），模板内部最终落到 `AscendC::DataCopy`。L0C->GM 走 `CopyL0CToGmTla` -> `AscendC::Fixpipe`。**这一步就是把第 1 节的 C++ 模板特化"接"进来。**
- **`bisheng` 的角色**：Bisheng 编译器在编译这个 `.cpp` 时，会把 `AscendC::DataCopy`/`Fixpipe` 内联展开成 DMA intrinsic（LLVM IR 形式）。所以 `.bc` 里存的不是 C++ 源码语义，而是已经展开到硬件 intrinsic 的 IR。

### 链接：`hivmc --link-aicore-bitcode` 把 bc 接进 kernel

最后一步在 Python 完成。[execution.py](../../../../catlass/execution.py) 的 `_build_hivmc_a5_command`（1832 行）拼出 hivmc 命令：

```python
command = [
    str(hivmc), str(mlir_path),
    "--target=Ascend950PR_9589",
    "--disable-ffts",
    f"--link-aicore-bitcode={template_bitcode}",  # ◀── bc 在此接入
    "-o", str(kernel_path),
]
```

- `--link-aicore-bitcode=` 接收一个或多个 `.bc` 路径（逗号分隔）。hivmc 把 stub 的 bitcode 链接（link）进当前 kernel 的 LLVM module，再下译成 `kernel.o`，链接后 stub 体会被内联进调用点--最终 `kernel.o` 里看不到 stub 调用，只剩展开后的 DMA intrinsic。

---

## bc 的构建流程（`bisheng` + `llvm-link`）

bc 的构建规则在 [csrc/mlir/bc/CMakeLists.txt](../../../../csrc/mlir/bc/CMakeLists.txt)，分两步：

**第一步：每个 `.cpp` -> 单个 `.bc`**（`compile_single_cpp_to_bc`）

```bash
bisheng -O2 -x cce --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off \
     --cce-aicore-arch=dav-c310-cube \   # cube 核；vec 核用 dav-c310-vec
     -DCATLASS_ARCH=3510 \
     -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl \
     -I${CATLASS_DSL_CATLASS_INCLUDE_DIR} \
     dma.cpp -emit-llvm -c -mllvm -disable-llvm-optzns \
     -o dma.aic.c310.bc
```

- `bisheng` 是 Bisheng 编译器（`${ASCEND_HOME_PATH}/bin/bisheng`），`-x cce` 让它走 CCE 前端。
- `--cce-aicore-arch=dav-c310-{cube,vec}` 指定核类型，决定 AscendC API 展开成哪套 intrinsic。
- `-mllvm -disable-llvm-optzns` 关掉 LLVM 优化，保留可链接的"原始" bitcode，优化留给最后 hivmc 统一做。

**第二步：`llvm-link` 合并成 `meta_op.*.bc`**（`add_custom_llvm_link_command` / `link_all_to_meta_op`）

```bash
llvm-link dma.aic.c310.bc mmad.aic.c310.bc mutex.aic.c310.bc ... \
          -o meta_op.aic.c310.bc
```

- 把同核类型所有 stub bc 合并成单个 `meta_op.{aic,aiv}.c310.bc`，这就是 hivmc `--link-aicore-bitcode` 实际吃的文件。
- 产物落在 `csrc/mlir/build/bc/meta_op.{aic,aiv}.c310.bc`。

---

## 如何扩展：新增一条 copy 路由

理解了 bc 接入机制，加一条新路由（比如一个新的 layout 组合）就是"三处对齐"：

1. **路由表**：在 [TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) 的 `getCopyRouteCallee`（606–733 行）加一条匹配，返回新 stub 名 `copy_<src>_<slayout>_to_<dst>_<dlayout>_<elemSuffix>`。
2. **stub 实现**：在 [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp)（或 Vector/dma.cpp）加 `void _mlir_ciface_copy_<...>(...)`，委托给对应的 `Catlass::Gemm::Tile::*Tla` 模板；若模板不存在则先在 [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) 下实现--**这正是 C++ 库复用价值所在：多数情况模板已存在，只需加一层 stub 包装。**
3. **重建 bc**：`ninja -C csrc/mlir/build bishengir_template_bitcode`（或 `./build.sh` 全量），让 `meta_op.*.c310.bc` 包含新 stub。

前端 [core_api.py](../../../../catlass/core_api.py) 的路由表（3796–3800 行）和 [TlaOps.cpp](../../../../csrc/mlir/lib/Dialect/Tla/IR/TlaOps.cpp) 的 `CopyOp::verify`（594–638 行）如果路由属于已有合法集合则无需改动；若是全新地址空间对，需同步更新这两处。

> **命名一致性是命门**：MLIR `func.call` 的目标符号、`getCopyRouteCallee` 返回的字符串、bc 里 `_mlir_ciface_` stub 的 C 函数名，三者必须完全对齐，否则 hivmc 链接时会报 undefined symbol。这也是为什么 stub 都用 `always_inline` + `emit_c_interface`--既保证符号可见用于链接，又保证链接后能内联消除调用开销。

---

## 关键文件索引

| 环节 | 文件 | 关键位置 |
|------|------|---------|
| op 定义 | [Tla.td](../../../../csrc/mlir/include/Dialect/Tla/IR/Tla.td) | `Tla_CopyOp` 583–601 |
| 前端构造 | [core_api.py](../../../../catlass/core_api.py) | `copy` 3803–3914；路由表 3796–3800 |
| 参数定义 | [params.py](../../../../catlass/params.py) | `CopyL0C2DstParams` 163–204 |
| 地址空间 | [address_space.py](../../../../catlass/address_space.py) | 17–29 |
| C++ verifier | [TlaOps.cpp](../../../../csrc/mlir/lib/Dialect/Tla/IR/TlaOps.cpp) | `CopyOp::verify` 594–638 |
| pass pipeline | [PassRegistry.cpp](../../../../csrc/mlir/lib/Passes/PassRegistry.cpp) | `buildTlaPipeline` 42–82 |
| cube copy lowering | [TlaCubeRegionPass.cpp](../../../../csrc/mlir/lib/Passes/TlaCubeRegionPass.cpp) | `LowerTlaCopyPattern` 197–400 |
| vector copy lowering | [TlaVectorRegionCommon.cpp](../../../../csrc/mlir/lib/Passes/TlaVectorRegionCommon.cpp) | `LowerCopyPattern` 110–199 |
| 路由选择/payload | [TlaTensorToMemref.cpp](../../../../csrc/mlir/lib/Passes/TlaTensorToMemref.cpp) | `getCopyRouteCallee` 606–733 |
| C++ 搬运模板 | [include/catlass/gemm/tile/ascend950/](../../../../../../include/catlass/gemm/tile/ascend950/) | `copy_gm_to_l1.hpp`、`copy_l0c_to_gm.hpp` 等 |
| stub 实现（cube） | [csrc/mlir/bc/Cube/dma.cpp](../../../../csrc/mlir/bc/Cube/dma.cpp) | GM->L1 stub 131 行起 |
| stub 实现（vector） | [csrc/mlir/bc/Vector/dma.cpp](../../../../csrc/mlir/bc/Vector/dma.cpp) | - |
| bc 构建规则 | [csrc/mlir/bc/CMakeLists.txt](../../../../csrc/mlir/bc/CMakeLists.txt) | `compile_single_cpp_to_bc` / `link_all_to_meta_op` |
| bc 运行时链接 | [execution.py](../../../../catlass/execution.py) | `_build_hivmc_a5_command` 1832；`_resolve_hivm_template_bitcode` 2003 |
| IR 形态样例 | [tests/lit/copy/copy-l0c-to-gm-atomic-f32.mlir](../../../../tests/lit/copy/copy-l0c-to-gm-atomic-f32.mlir)、[tests/lit/tensor/make-tensor-copy-gm-l1-zn.mlir](../../../../tests/lit/tensor/make-tensor-copy-gm-l1-zn.mlir) | `copy-l0c-to-gm-atomic-f32.mlir`、`make-tensor-copy-gm-l1-zn.mlir` |

---

## 附：一图总结 `tla.copy` 的 bc 接入全链路

```
tla.copy(gm_a -> l1_a)                        ← 用户 Python
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
        │  │   stub 体来自 Cube/dma.cpp (bisheng 预编译成 .bc)            │
        │  │   -> TileCopyTla -> AscendC::DataCopy (已展开为 intrinsic) │
        │  └─────────────────────────────────────────────────────────┘
        ▼
kernel.o   (DMA intrinsic 已内联，无 stub 调用残留)
```

**一句话总结**：Python 流水线生成"调用契约"（`func.call` + payload 布局），bc 提供"调用实现"（C++ stub -> `TileCopyTla` 模板 -> AscendC -> intrinsic），`hivmc --link-aicore-bitcode` 是把两者焊在一起的那道工序。`tla.copy` 只是用最直观的"搬数据"把这条契约-实现-焊接的链路展示出来--`tla.mmad`、`tla.store` 等所有走 stub 的 op 都遵循同一套 bc 接入机制。
