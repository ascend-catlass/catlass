/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <array>
#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_mx_preload_tla.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle_grouped_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_aswt_tla.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#ifndef MX_GMM_ENABLE_ASWT
#define MX_GMM_ENABLE_ASWT 0
#endif

#ifndef MX_GMM_ENABLE_BASE_M
#define MX_GMM_ENABLE_BASE_M 0
#endif

#ifndef MX_GMM_ENABLE_PRELOAD
#define MX_GMM_ENABLE_PRELOAD 0
#endif

// MX_GMM_PRELOAD_STAGES is the deferred-MMAD pipeline distance, not a generic
// "prefetch more is better" knob. With MX_GMM_ENABLE_PRELOAD=1, the current
// K-stripe issues GM->L1 loads first, and the MMAD/FIXPIPE for the stripe
// queued MX_GMM_PRELOAD_STAGES steps earlier is executed after the queue is
// filled. The default value 1 overlaps current MTE2 load with the previous
// stripe's compute and is the only safe value with the default L1A/L1B stage
// count of 2. Raising it requires changing the preload BlockMmad stage config
// together, and must still satisfy PRELOAD_STAGES < L1A_STAGES/L1B_STAGES plus
// the L1/event-id budget checks in block_mmad_mx_preload_tla.hpp.
#ifndef MX_GMM_PRELOAD_STAGES
#define MX_GMM_PRELOAD_STAGES 1
#endif

#if MX_GMM_ENABLE_BASE_M && !MX_GMM_ENABLE_ASWT
#error "MX_GMM_ENABLE_BASE_M requires MX_GMM_ENABLE_ASWT"
#endif

// P4 实验开关：是否启用 BlockMmad 的 L1 常驻（lastAddr 命中则跳过 GM->L1 重搬）。
// 默认 1（保持原行为）。在 MoE 小-M 形状下（单核每 group 仅 1 个 tile）lastAddr 几乎不
// 命中，但每次 load 仍要做地址比较，疑似纯 scalar 开销；编译时加 -DMX_GMM_L1_RESIDENT=0
// 关闭后对比 mte2_ratio / 总时间，即可验证它在该场景是否为净负优化。
#ifndef MX_GMM_L1_RESIDENT
#if MX_GMM_ENABLE_PRELOAD
#define MX_GMM_L1_RESIDENT 0
#else
#define MX_GMM_L1_RESIDENT 1
#endif
#endif

using namespace Catlass;
using namespace tla;

struct GroupedGemmOptionsWithMxType : public GroupedGemmOptions {
    const std::string HELPER = "problem_count m n k trans_b quant_type [device_id]";

    uint32_t transB{0};
    std::string quantDataType = "float8_e4m3fn";
    const std::array<std::string, 3> quantDataTypes = {"float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2"};

    GroupedGemmOptionsWithMxType() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            GROUP_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            TRANS_B_INDEX,
            QUANT_TYPE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        transB = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::TRANS_B_INDEX)]);
        quantDataType = std::string(argv[static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)]);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        // valid check
        if (quantDataTypes.end() == std::find(quantDataTypes.begin(), quantDataTypes.end(), quantDataType)) {
            std::cerr << "Invalid quantData type: " << quantDataType << std::endl;
            return -1;
        }

        return 0;
    }
};

using Options = GroupedGemmOptionsWithMxType;

// malloc @npu device and prepare copying data
#define MALLOC_AND_COPY_TO_NPU_DEVICE(descr, hostData, size)                                           \
    uint8_t* device##descr{nullptr};                                                                   \
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device##descr), size, ACL_MEM_MALLOC_HUGE_FIRST)); \
    ACL_CHECK(aclrtMemcpy(device##descr, size, (hostData).data(), size, ACL_MEMCPY_HOST_TO_DEVICE))

template <typename T>
bool SaveResult(const std::string& filename, const std::vector<T>& data, const size_t dataSize)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    std::vector<char> buffer(dataSize);
    std::memcpy(buffer.data(), data.data(), dataSize);

    file.write(buffer.data(), dataSize);
    if (!file) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

template <
    class MatmulKernel, class LayoutA_, class LayoutB_, class LayoutMxScaleA_, class LayoutMxScaleB_, class LayoutC_>
bool LaunchMxGroupedMatmul(
    Options const& options, uint8_t* deviceGroupList, uint8_t* deviceA, LayoutA_ const& layoutA, uint8_t* deviceB,
    LayoutB_ const& layoutB, uint8_t* deviceMxScaleA, LayoutMxScaleA_ const& layoutMxScaleA, uint8_t* deviceMxScaleB,
    LayoutMxScaleB_ const& layoutMxScaleB, uint8_t* deviceC, LayoutC_ const& layoutC, uint8_t*& deviceWorkspace,
    size_t& sizeWorkspace, aclrtStream stream, uint32_t aicCoreNum)
{
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{
        options.problemShape, options.problemCount, deviceGroupList, deviceA,        layoutA, deviceB, layoutB,
        deviceMxScaleA,       layoutMxScaleA,       deviceMxScaleB,  layoutMxScaleB, deviceC, layoutC};

    MatmulAdapter matmul_op;
    if (matmul_op.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "CanImplement check failed." << std::endl;
        return false;
    }
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);
    return true;
}

template <class ElementA, class ElementB, class ElementC, class L1TileShape, class L0TileShape, bool transB = false>
void MxGroupedMatmulSliceM(Options const& options)
{
    // Input shape constraints: problem_count/m/n/k must be greater than 0.
    // GroupedGemmOptionsWithMxType::Parse does not reject zero values, so check them here
    // before any division or device allocation.
    const uint32_t m = options.problemShape.m();
    const uint32_t n = options.problemShape.n();
    const uint32_t k = options.problemShape.k();
    const uint32_t groupCount = options.problemCount;
    if (m == 0 || n == 0 || k == 0 || groupCount == 0) {
        std::cerr << "problem_count, m, n and k must be greater than 0" << std::endl;
        return;
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    // compute mxScale len, k must be multiples of 2
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementMxScale = float8_e8m0_t;
    using ElementGroupList = int64_t;
    static_assert(
        (std::is_same_v<ElementA, float8_e5m2_t> || std::is_same_v<ElementA, float8_e4m3_t> ||
         std::is_same_v<ElementA, float4_e2m1x2_t>) &&
            (std::is_same_v<ElementB, float8_e5m2_t> || std::is_same_v<ElementB, float8_e4m3_t> ||
             std::is_same_v<ElementB, float4_e2m1x2_t>) &&
            std::is_same_v<ElementMxScale, float8_e8m0_t>,
        "ElementA and ElementB must be float8_e5m2_t, float8_e4m3_t, or float4_e2m1x2_t, ElementMxScale must be "
        "float8_e8m0_t");

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::template MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity() * groupCount;
    size_t lenC = tagC.Capacity();
    size_t lenMxScaleA = m * mxScaleAlignedK;
    size_t lenMxScaleB = mxScaleAlignedK * n * groupCount;

    size_t sizeA = lenA * SizeOfBits<ElementA>::value / 8;
    size_t sizeB = lenB * SizeOfBits<ElementB>::value / 8;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeGroupList = groupCount * sizeof(ElementGroupList);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeWorkspace = 0;

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);

    // Read mxfp4 data from bin file(data_gen)
    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile("./data/a_8.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/b_8.bin", hostB.data(), sizeB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/b_scale.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        releaseAclEarly();
        return;
    }

    // Generate average group size
    auto groupList = golden::GenerateAverageGroupList<ElementGroupList, false /* non-cumsum*/>(m, groupCount);

    // Malloc @npu device
    MALLOC_AND_COPY_TO_NPU_DEVICE(GroupList, groupList, sizeGroupList);
    MALLOC_AND_COPY_TO_NPU_DEVICE(A, hostA, sizeA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(B, hostB, sizeB);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleA, hostMxScaleA, sizeMxScaleA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleB, hostMxScaleB, sizeMxScaleB);

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    // ------------------------------------------------------------------
    // Performance tuning knobs (see README "性能调优说明")
    //   * L1_SCALE_FACTOR_K = 16 : 一次 GM->L1 搬 16 个 L1-K 条带的 MX scale，
    //                              减少 scale MTE2 小包头开销（默认 1 = 每个
    //                              L1-K 条带都重搬一次 scale）。
    //                              FP8: L1 占用 = 256 + 8*16 = 384 KB / 512 KB
    //                              FP4: L1 占用 = 256 + 16*16 = 512 KB / 512 KB
    //   * ENABLE_L1_RESIDENT  : 跨 block / 跨 group 调用 BlockMmad 时，若新
    //                          tile 的 GM 地址与上次相同，则跳过 GM->L1 搬运。
    //                          kernel 里 blockMmad 实例只构造一次，
    //                          lastAddr 在地址变化时自然触发重载，安全。
    // ------------------------------------------------------------------
    constexpr uint32_t l1ScaleFactorK = 16;
    constexpr uint32_t l0cStages = 1; // 与 enableUnitFlag=true 配套，只能为 1
    // P4 实验开关，默认 ON；用 -DMX_GMM_L1_RESIDENT=0 关闭以对比 scalar 开销（见文件顶部说明）
    constexpr bool enableL1Resident = (MX_GMM_L1_RESIDENT != 0);
#if MX_GMM_ENABLE_PRELOAD
    using DispatchPolicy = Gemm::MmadMxPreload<
        ArchTag, MX_GMM_PRELOAD_STAGES, enableUnitFlag, l1ScaleFactorK, l0cStages, enableL1Resident>;
#else
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, l1ScaleFactorK, l0cStages, enableL1Resident>;
#endif

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
#if MX_GMM_ENABLE_PRELOAD
    using BlockMmad = Gemm::Block::BlockMmadMxPreloadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
#else
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
#endif
    using BlockEpilogue = void;
    bool launched = false;

#if MX_GMM_ENABLE_ASWT
    using BlockScheduler = typename Gemm::Block::GemmGroupedAswtTailSplitSwizzle<4, false, transB>;
    using MatmulKernel = Gemm::Kernel::GroupedMxMatmulSliceMAswtTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList, (MX_GMM_ENABLE_BASE_M != 0)>;
    launched = LaunchMxGroupedMatmul<MatmulKernel>(
        options, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceMxScaleA, layoutMxScaleA, deviceMxScaleB,
        layoutMxScaleB, deviceC, layoutC, deviceWorkspace, sizeWorkspace, stream, aicCoreNum);
#else

    // ------------------------------------------------------------------
    // 调度器 swizzle 方向按 "每组 M 与 N" 的相对大小动态选择，对齐 example 58/60 的做法：
    //   * 每组 M > N : SwizzleDirection=0（M 优先 Zn），适合 M 较大的场景；
    //   * 否则       : SwizzleDirection=1（N 优先 Nz），grouped GEMM 中典型情况。
    // 这样可以让相邻 block 在 N 或 M 方向共享 B/A 的 GM 数据，配合
    // ENABLE_L1_RESIDENT 命中 L1 缓存，并改善 L2 reuse。
    // ------------------------------------------------------------------
    const uint32_t mPerGroup = (options.problemCount > 0) ? (m / options.problemCount) : m;
    const bool useMPriority = mPerGroup > n;

    if (useMPriority) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel =
            Gemm::Kernel::GroupedMxMatmulSliceMTla<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;
        launched = LaunchMxGroupedMatmul<MatmulKernel>(
            options, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceMxScaleA, layoutMxScaleA,
            deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, deviceWorkspace, sizeWorkspace, stream, aicCoreNum);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel =
            Gemm::Kernel::GroupedMxMatmulSliceMTla<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;
        launched = LaunchMxGroupedMatmul<MatmulKernel>(
            options, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceMxScaleA, layoutMxScaleA,
            deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, deviceWorkspace, sizeWorkspace, stream, aicCoreNum);
    }

#endif

    if (!launched) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceC));
        ACL_CHECK(aclrtFree(deviceGroupList));
        ACL_CHECK(aclrtFree(deviceMxScaleA));
        ACL_CHECK(aclrtFree(deviceMxScaleB));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Save GMM output
    using HostElementC = std::conditional_t<std::is_same_v<ElementC, bfloat16_t>, bfloat16, ElementC>;
    static_assert(sizeof(HostElementC) == sizeof(ElementC), "Host and device C types must have the same size");
    std::vector<HostElementC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostCFp32(lenC);
    for (uint32_t i = 0; i < lenC; ++i) {
        hostCFp32[i] = static_cast<float>(hostC[i]);
    }

    std::string outputFileName = "./data/result.bin";
    SaveResult<float>(outputFileName, hostCFp32, lenC * sizeof(float));

    // Finalize
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

template <bool TB>
void Run(Options const& options)
{
    using ElementC = bfloat16_t;
    if (options.quantDataType == "float8_e4m3fn") {
        MxGroupedMatmulSliceM<
            float8_e4m3_t, float8_e4m3_t, ElementC, Shape<Int<256>, Int<256>, Int<256>>,
            Shape<Int<256>, Int<256>, Int<128>>, TB>(options);
    } else if (options.quantDataType == "float8_e5m2") {
        MxGroupedMatmulSliceM<
            float8_e5m2_t, float8_e5m2_t, ElementC, Shape<Int<256>, Int<256>, Int<256>>,
            Shape<Int<256>, Int<256>, Int<128>>, TB>(options);
    } else if (options.quantDataType == "float4_e2m1fn_x2") {
        MxGroupedMatmulSliceM<
            float4_e2m1x2_t, float4_e2m1x2_t, ElementC, Shape<Int<256>, Int<256>, Int<512>>,
            Shape<Int<256>, Int<256>, Int<256>>, TB>(options);
    } else {
        std::cerr << "Unexpected quant data-type mismatch." << std::endl;
    }
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    // execute the kernel
    options.transB ? Run<true>(options) : Run<false>(options);
    return 0;
}
