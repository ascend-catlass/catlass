/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/broadcast_matmul_perblock_quant_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_perblock_quant.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

struct BroadcastMatmulOptions : public GroupedGemmOptions {
    const std::string HELPER = "batch_count m n k [device_id]";

    BroadcastMatmulOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            BATCH_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) || argc < static_cast<uint32_t>(ArgsIndex::K_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        return 0;
    }
};

using Options = BroadcastMatmulOptions;

template <typename T>
bool ReadFile(const std::string& filename, T* data, size_t size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(data), size);
    if (!file) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        return false;
    }
    file.close();
    return true;
}

template <typename T>
bool SaveResult(const std::string& filename, const T* data, size_t size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for write: " << filename << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), size);
    if (!file) {
        std::cerr << "Failed to write file: " << filename << std::endl;
        return false;
    }
    file.close();
    return true;
}

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t batchCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using ElementA = bfloat16_t;
    using ElementB = bfloat16_t;
    using ElementC = bfloat16_t;
    using ElementDst = float8_e4m3_t;
    using ElementScale = float;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity() * batchCount;
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity() * batchCount;

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeDst = lenC * sizeof(ElementDst);
    size_t sizeScale = batchCount * sizeof(ElementScale);
    size_t sizeWorkspace;

    std::vector<uint16_t> hostA(lenA);
    std::vector<uint16_t> hostB(lenB);

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    if (!ReadFile("./data/a.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/b.bin", hostB.data(), sizeB)) {
        releaseAclEarly();
        return;
    }

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDst{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDst), sizeDst, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 9201
    using ArchTag = Arch::Ascend960DT;
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 9202
    using ArchTag = Arch::Ascend960PR;
#else
    using ArchTag = Arch::Ascend950;
#endif
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    constexpr uint32_t l0CStages = 1;
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
    constexpr bool enableL1Resident = false;
#else
    constexpr bool enableL1Resident = true;
#endif

    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident>;
    using L1TileShape = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<128>, Int<128>, Int<128>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    using TilePerBlockQuant = Epilogue::Tile::TilePerBlockQuant<ArchTag, ElementC, ElementDst, ElementScale>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAscend950PerBlockQuantTla<1>, ElementC, ElementDst, ElementScale, TilePerBlockQuant>;

    // Swizzle offset is 3 and direction is 1.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::BroadcastMatmulPerblockQuantTla<BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{batchCount, options.problemShape, deviceA, layoutA, deviceB, layoutB, layoutC,
                                      deviceDst,  deviceScale};

    MatmulAdapter matmul_op;
    if (matmul_op.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "[ERROR]op cannot be implemented: invalid arguments" << std::endl;
        releaseAclEarly();
        return;
    }
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> hostDst(lenC);
    ACL_CHECK(aclrtMemcpy(hostDst.data(), sizeDst, deviceDst, sizeDst, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostScale(batchCount);
    ACL_CHECK(aclrtMemcpy(hostScale.data(), sizeScale, deviceScale, sizeScale, ACL_MEMCPY_DEVICE_TO_HOST));

    SaveResult("./data/result_dst.bin", hostDst.data(), sizeDst);
    SaveResult("./data/result_scale.bin", hostScale.data(), sizeScale);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceDst));
    ACL_CHECK(aclrtFree(deviceScale));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
