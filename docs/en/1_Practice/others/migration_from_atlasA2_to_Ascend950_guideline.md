# Guideline for Migrating Existing Atlas A2 Operators to Ascend 950 Generation Compatibility

## Before You Begin

This document mainly explains how to quickly migrate existing operators based on CATLASS on the Atlas A2 architecture to the Ascend 950 architecture, including reuse of the original kernel- and Block-level components, and transparent use of Tile components across different architectures.

To accomplish this migration, it is recommended to follow the three steps below.

## Dispatch Component Adjustments

Existing operators on the original Atlas A2 platform use Dispatch components with A2-generation semantics to route to the required Block components (for example, the Dispatch component used by [`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md) is [`MmadAtlasA2Pingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L32)); after migration we recommend that operators use Dispatch components with common-generation semantics (unless the new operator needs features unique to the Ascend 950 generation), such as [`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292).
If [`dispatch_policy.hpp`](../../../../include/catlass/gemm/dispatch_policy.hpp) does not contain a Dispatch component with common-generation semantics, such as [`MmadAtlasA2Streamk`](../../../../include/catlass/gemm/dispatch_policy.hpp#L237), it needs to be added, since the current repository does not contain the component `MmadStreamk`:

```cpp
template <class ArchTag, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadStreamk : public MmadBase<ArchTag, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};
```

## Block-Level Modifications

At the Block level, we recommend the following SFINAE approach to ensure compatibility with both the original A2-generation semantics and the common-generation Dispatch. In terms of syntax, we add an `Enable` template parameter to [`BlockMmad`](../../../../include/catlass/gemm/block/block_mmad.hpp) for candidate selection.
Still taking [`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md) as an example, the Block component it uses is located at `include/catlass/gemm/block/block_mmad_pingpong.hpp`. To maintain specialization support for [`MmadAtlasA2Pingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L32) (the original-generation component) and [`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292), first implement the following `MmadPingpongDispatchChecker` for the check.

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

Adjust the BlockMmad template definition to select the specialization candidate via `std::enable_if_t` combined with the above `MmadPingpongDispatchChecker`.

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

Finally, we recommend adding generation validation to the modified Block component implementation, so as to formalize the constraint on the generation range. For example:

```cpp
// Check ArchTag
static_assert(std::is_same_v<ArchTag, Arch::AtlasA2> || std::is_same_v<ArchTag, Arch::Ascend950>,
    "ArchTag can only be AtlasA2 or Ascend950!");
```

Note that if later Block-level implementations leverage certain specifics related to the Dispatch component, the corresponding components can also be implemented via SFINAE. For example, the Block component used in [`00_basic_matmul`](../../../../examples/00_basic_matmul/README.md) needs the STAGES information declared in the Dispatch, whereas in the common component [`MmadPingpong`](../../../../include/catlass/gemm/dispatch_policy.hpp#L292) the granularity over different regions is refined—including `L1A_STAGES`, `L1B_STAGES`, etc. Here a minimal conservative strategy is used for extraction (as shown below), [see details](https://gitcode.com/cann/catlass/blob/8b98e2f4f0822d77cc545a3167d7e150800da596/include/catlass/gemm/block/block_mmad_pingpong.hpp#L58).

```cpp
// ...

// DispatchStagesGetter specialized for different Dispatch policies
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

Note that if changes at the `Epilogue` level are involved, its [`block`-level components](../../../../include/catlass/epilogue/block/) likewise need to be adjusted; this is not elaborated here.

## Operator Modifications

After completing the above two sets of modifications, on the operator side you need to adjust the `ArchTag` identifier and the corresponding Dispatch component (**Dispatch specializations of the original A2 generation remain compatible under Atlas A2**), for example:

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

In addition, unlike Atlas A2, the `aclrtGetHardwareSyncAddr` call and the `hardwareSyncAddr` variable should be removed from Ascend 950-related code, and the `RunAdapter` call signature is simplified accordingly:

```diff
-    // Prepare hardware sync address
-    uint64_t hardwareSyncAddr{0};
-    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

-    RunAdapter(gemm_op, arguments, stream, aicCoreNum, hardwareSyncAddr);
+    RunAdapter(gemm_op, arguments, stream, aicCoreNum);
```

Afterwards, you can compile and run the operator adapted to the Ascend 950 platform (note: add `-DCATLASS_ARCH=3510` to the compilation options).

## Complete Migration Checklist

When migrating an operator from Atlas A2 to Ascend 950, confirm the following items one by one:

- [ ] **ArchTag**: `Arch::AtlasA2` → `Arch::Ascend950`
- [ ] **Gemm Dispatch**: add a common-generation version if none exists (refer to `GemmPingpong`), and replace it on the operator side
- [ ] **Epilogue Dispatch**: add a common-generation version if none exists (refer to `EpilogueGemm`), and replace it on the operator side
- [ ] **Block specialization**: add a Block specialization for the new common Dispatch (refer to `BlockGemm<GemmPingpong>`)
- [ ] **Epilogue Block specialization**: add a BlockEpilogue specialization for the new common Epilogue Dispatch
- [ ] **Tile type selector**: check whether `L1AndL0TypeSelector` needs an arch-aware version; the Ascend 950 L0 type may differ
- [ ] **Arch Guard**: check whether `__NPU_ARCH__` (device side) or `CATLASS_ARCH` (host side) guards are used, and add them as needed
- [ ] **hardwareSyncAddr removal**: delete the `aclrtGetHardwareSyncAddr` call and remove the fifth parameter of `RunAdapter`
- [ ] **Compilation verification**: add `-DCATLASS_ARCH=3510` to the compilation options

---
