# AtlasA2存量算子向Ascend950代际兼容开发指导

## 写在前面

该文档主要说明，如何基于CATLASS在AtlasA2架构下的存量算子快速迁移到Ascend950架构，包括对原kernel和Block层级组件复用，并无感使用不同架构的Tile组件。

为实现该迁移动作，推荐按照下述三个步骤进行。

## Dispatch组件调整

原AtlasA2平台的存量算子使用A2代际语义的Dispatch组件以路由至所需的Block组件（如[`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md)所使用的Dispatch组件为[`MmadAtlasA2Pingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L32)），我们建议迁移后算子使用公共代际语义的Dispatch组件（除新算子需使用Ascend950代际独有的特性外），例如[`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292)。
如果[`dispatch_policy.hpp`](../../../../include/catlass/gemm/dispatch_policy.hpp)下不包含公共代际语义的Dispatch组件，例如[`MmadAtlasA2Streamk`](../../../../include/catlass/gemm/dispatch_policy.hpp#L237)，由于当前代码仓上不包含组件`MmadStreamk`，需要进行补充：

```cpp
template <class ArchTag, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadStreamk : public MmadBase<ArchTag, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};
```

## Block层修改

在Block层级，我们推荐使用下面的SFINAE方案，以保证其对原A2代际语义和公共代际的Dispatch兼容。在写法上，我们为[`BlockMmad`](../../../../include/catlass/gemm/block/block_mmad.hpp)增加了一个`Enable`模板入参用于候选。
仍以[`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md)为例，其使用的Block组件位于`include/catlass/gemm/block/block_mmad_pingpong.hpp`，为保持对[`MmadAtlasA2Pingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L32)（原代际组件）及[`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292)的特化支持，首先实现下述的`MmadPingpongDispatchChecker`进行判断。

```cpp
/// check if the dispatch is `MmadAtlasA2Pingpong` or `MmadPingpong`
template <class DispatchPolicy>
struct MmadPingpongDispatchChecker {
    static constexpr bool value = false;
};

template <bool ENABLE_UNIT_FLAG>
struct MmadPingpongDispatchChecker<MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>> {
    static constexpr bool value = true;
};

template <class ArchTag, bool ENABLE_UNIT_FLAG, bool USE_HF32_MODE,
    uint32_t L0C_STAGES, bool ENABLE_L1_RESIDENT, uint32_t L1A_STAGES,
    uint32_t L1B_STAGES, uint32_t L0A_STAGES, uint32_t L0B_STAGES>
struct MmadPingpongDispatchChecker<MmadPingpong<ArchTag, ENABLE_UNIT_FLAG, USE_HF32_MODE,
    L0C_STAGES, ENABLE_L1_RESIDENT, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES>> {
    static constexpr bool value = true;
};
```

调整BlockMmad模板写法，通过`std::enable_if_t`结合上述`MmadPingpongDispatchChecker`进行特化候选。

```diff
template <
-    bool ENABLE_UNIT_FLAG_,
+    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
-    MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
+    DispatchPolicy_,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_,
+    std::enable_if_t<MmadPingpongDispatchChecker<DispatchPolicy_>::value>
 > {
public:
    // Type Aliases
-    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
+    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
```

最后，我们建议向修改后的Block组件实现加入代际校验，以便规范对代际范围的约束。如：

```cpp
// Check ArchTag
static_assert(std::is_same_v<ArchTag, Arch::AtlasA2> || std::is_same_v<ArchTag, Arch::Ascend950>,
    "ArchTag can only be AtlasA2 or Ascend950!");
```

需要注意的是，在后续Block层实现中如果利用到与Dispatch组件有关的一些特定内容，也可通过SFINAE实现相应的组件。例如[`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md)中所使用的Block组件需要利用Dispatch中声明的STAGES信息，而在公共组件[`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292)中，其细化了不同区域上的粒度——包括`L1A_STAGES`, `L1B_STAGES`等，这里使用了最小保守策略进行提取（如下），[详见](https://gitcode.com/cann/catlass/blob/8b98e2f4f0822d77cc545a3167d7e150800da596/include/catlass/gemm/block/block_mmad_pingpong.hpp#L58)。

```cpp
// ...

// 针对不同Dispatch特化提取的DispatchStagesGetter
template <class ArchTag, bool ENABLE_UNIT_FLAG, bool USE_HF32_MODE,
    uint32_t L0C_STAGES, bool ENABLE_L1_RESIDENT, uint32_t L1A_STAGES,
    uint32_t L1B_STAGES, uint32_t L0A_STAGES, uint32_t L0B_STAGES>
struct DispatchStagesGetter<MmadPingpong<ArchTag, ENABLE_UNIT_FLAG, USE_HF32_MODE,
    L0C_STAGES, ENABLE_L1_RESIDENT, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES>> {
    // Strategy: select the lowest stage as the common STAGES
    static constexpr uint32_t L1_STAGES = L1A_STAGES > L1B_STAGES ? L1B_STAGES : L1A_STAGES;
    static constexpr uint32_t L0_STAGES = L0A_STAGES > L0B_STAGES ? L0B_STAGES : L0A_STAGES;
    static constexpr uint32_t STAGES = L1_STAGES > L0_STAGES ? L0_STAGES : L1_STAGES;
};
```

注意，如果涉及到`Epilogue`层级的改动，同样需要对其[`block`层组件](../../../../include/catlass/epilogue/block/)进行调整，这里不再赘述。

## 算子修改

完成上述两方面修改后，在算子侧，需要调整`ArchTag`标识和相应的Dispatch组件(**原A2代际下Dispatch特化仍在AtlasA2下兼容**)，例如：

```diff
// examples/00_basic_matmul/basic_matmul.cpp(previous)

// ...
- using ArchTag = Arch::AtlasA2;
+ using ArchTag = Arch::Ascend950;
- using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
+ using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;

using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

// ...
```

此外，与AtlasA2不同的是，`aclrtGetHardwareSyncAddr` 调用及 `hardwareSyncAddr` 变量应从Ascend950相关代码中移除，`RunAdapter` 调用签名同步简化：

```diff
-    // Prepare hardware sync address
-    uint64_t hardwareSyncAddr{0};
-    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

-    RunAdapter(gemm_op, arguments, stream, aicCoreNum, hardwareSyncAddr);
+    RunAdapter(gemm_op, arguments, stream, aicCoreNum);
```

随后即可编译运行适配于Ascend950平台的算子（备注：编译选项请添加`-DCATLASS_ARCH=3510`）

## 完整迁移检查清单

算子从 AtlasA2 迁移到 Ascend950 时，逐项确认以下内容：

- [ ] **ArchTag**：`Arch::AtlasA2` → `Arch::Ascend950`
- [ ] **Gemm Dispatch**：如无公共代际版本则新增（参考 `GemmPingpong`），算子侧替换
- [ ] **Epilogue Dispatch**：如无公共代际版本则新增（参考 `EpilogueGemm`），算子侧替换
- [ ] **Block 特化**：为新的公共 Dispatch 增加 Block 特化（参考 `BlockGemm<GemmPingpong>`）
- [ ] **Epilogue Block 特化**：为新的公共 Epilogue Dispatch 增加 BlockEpilogue 特化
- [ ] **Tile 类型选择器**：检查 `L1AndL0TypeSelector` 是否需 arch-aware 版本，Ascend950 L0 类型可能不同
- [ ] **Arch Guard**：检查是否使用 `__NPU_ARCH__`(Device侧) 或 `CATLASS_ARCH`(Host侧) 守卫，根据需要补充
- [ ] **hardwareSyncAddr 移除**：删除 `aclrtGetHardwareSyncAddr` 调用，`RunAdapter` 去掉第五个参数
- [ ] **编译验证**：编译选项加 `-DCATLASS_ARCH=3510`

---
