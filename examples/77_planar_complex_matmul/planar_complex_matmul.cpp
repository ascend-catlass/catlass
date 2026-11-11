/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include <iostream>
#include "catlass/gemm/kernel/planar_complex_gemm_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER = "Usage: m n k [device_id] [--datapath DATA_PATH]\n";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};
    std::string dataPath;

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        if (argc < 4) {
            std::cerr << HELPER;
            return -1;
        }
        problemShape.m() = std::atoi(argv[1]);
        problemShape.n() = std::atoi(argv[2]);
        problemShape.k() = std::atoi(argv[3]);
        int argIndex = 4;
        while (argIndex < argc) {
            std::string flag = argv[argIndex++];
            if (flag == "--datapath") {
                if (argIndex >= argc) {
                    std::cerr << "--datapath requires an argument\n";
                    return -1;
                }
                dataPath = argv[argIndex++];
            } else {
                deviceId = std::atoi(flag.c_str());
            }
        }
        return 0;
    }
};

static bool Run(const Options& options)
{
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // ---- Configure types ----
    using ElementA = half;
    using ElementB = half;
    using ElementC = float;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(m, n);

    size_t lenA = layoutA.Capacity();
    size_t lenB = layoutB.Capacity();
    size_t lenC = layoutC.Capacity();
    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);

    // ---- Host data ----
    std::vector<fp16_t> hostAReal(lenA);
    std::vector<fp16_t> hostAImag(lenA);
    std::vector<fp16_t> hostBReal(lenB);
    std::vector<fp16_t> hostBImag(lenB);

    bool hasDataPath = !options.dataPath.empty();
    if (hasDataPath) {
        bool readOk = true;
        readOk &= ReadFile(options.dataPath + "/inputA_real.dat", hostAReal.data(), sizeA);
        readOk &= ReadFile(options.dataPath + "/inputA_imag.dat", hostAImag.data(), sizeA);
        readOk &= ReadFile(options.dataPath + "/inputB_real.dat", hostBReal.data(), sizeB);
        readOk &= ReadFile(options.dataPath + "/inputB_imag.dat", hostBImag.data(), sizeB);
        if (!readOk) {
            std::cerr << "Failed to read input data from " << options.dataPath << std::endl;
            return false;
        }
    } else {
        golden::FillRandomData<fp16_t>(hostAReal, -5.0f, 5.0f);
        golden::FillRandomData<fp16_t>(hostAImag, -5.0f, 5.0f);
        golden::FillRandomData<fp16_t>(hostBReal, -5.0f, 5.0f);
        golden::FillRandomData<fp16_t>(hostBImag, -5.0f, 5.0f);
    }

    // Defer ACL setup until after ReadFile so a read-failure return doesn't leak stream/device.
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));

    aclrtStream stream{nullptr};
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtSetStreamFailureMode(stream, ACL_STOP_ON_FAILURE));

    // NEGATE_A: pick the smaller side to reduce workspace and AIV negate cost.
    bool negateA = (m < n);

    // ---- Device memory ----
    uint8_t* deviceAReal{nullptr};
    uint8_t* deviceAImag{nullptr};
    uint8_t* deviceBReal{nullptr};
    uint8_t* deviceBImag{nullptr};
    uint8_t* deviceCReal{nullptr};
    uint8_t* deviceCImag{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAReal), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAImag), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBReal), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBImag), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCReal), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCImag), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceAReal, sizeA, hostAReal.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceAImag, sizeA, hostAImag.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBReal, sizeB, hostBReal.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBImag, sizeB, hostBImag.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // hardwareSyncAddr is required for AIV<->AIC cross-core sync inside the Mix kernel.
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // ---- Common template config shared by both variants ----
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, /*enableUnitFlag=*/true>;
    using L1TileShapeTla = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
    using L0TileShapeTla = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShapeTla{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShapeTla{});

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;

    // ---- Host-side cost-model: pick kernel variant ----
    // K >= 6000 AND per_core >= 3 tiles -> Four-Pass, otherwise -> Fused.
    uint32_t coreLoops = CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N);
    double perCore = static_cast<double>(coreLoops) / aicCoreNum;
    bool useFourPass = (k >= 6000) && (perCore >= 3.0);

    std::cout << "PlanarComplexGemm dispatch: M=" << m << " N=" << n << " K=" << k << " coreLoops=" << coreLoops
              << " perCore=" << perCore << " -> " << (useFourPass ? "FourPass" : "Fused")
              << " negate=" << (negateA ? "A_imag" : "B_imag") << " swizzleDir=" << (m >= n ? 0 : 1) << std::endl;

    // ---- Launch ----
    bool kernelFailed = false;

    if (useFourPass) {
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShapeTla, L0TileShapeTla, ElementA, ElementB, ElementC, /*ElementBias=*/void,
            TileCopy>;
        if (m >= n) {
            // Swizzle offset is 3 and direction is 0.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using PlanarKernel = Gemm::Kernel::PlanarComplexGemm<
                /*USE_FOUR_PASS=*/true, /*NEGATE_A=*/false, BlockMmad, void, BlockScheduler>;
            using DeviceOp = Gemm::Device::DeviceGemm<PlanarKernel>;

            typename PlanarKernel::Arguments arguments{
                options.problemShape, deviceAReal, deviceAImag, nullptr,     deviceBReal,
                deviceBImag,          nullptr,     deviceCReal, deviceCImag,
            };
            DeviceOp op;
            size_t workspaceSize = op.GetWorkspaceSize(arguments);
            uint8_t* deviceWorkspace{nullptr};
            if (workspaceSize > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            op.Initialize(arguments, deviceWorkspace);
            kernelFailed = (op(stream, aicCoreNum, hardwareSyncAddr) != Status::kSuccess);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            if (deviceWorkspace != nullptr) {
                ACL_CHECK(aclrtFree(deviceWorkspace));
            }
        } else {
            // Swizzle offset is 3 and direction is 1.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using PlanarKernel = Gemm::Kernel::PlanarComplexGemm<
                /*USE_FOUR_PASS=*/true, /*NEGATE_A=*/true, BlockMmad, void, BlockScheduler>;
            using DeviceOp = Gemm::Device::DeviceGemm<PlanarKernel>;

            typename PlanarKernel::Arguments arguments{
                options.problemShape, deviceAReal, deviceAImag, nullptr,     deviceBReal,
                deviceBImag,          nullptr,     deviceCReal, deviceCImag,
            };
            DeviceOp op;
            size_t workspaceSize = op.GetWorkspaceSize(arguments);
            uint8_t* deviceWorkspace{nullptr};
            if (workspaceSize > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            op.Initialize(arguments, deviceWorkspace);
            kernelFailed = (op(stream, aicCoreNum, hardwareSyncAddr) != Status::kSuccess);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            if (deviceWorkspace != nullptr) {
                ACL_CHECK(aclrtFree(deviceWorkspace));
            }
        }
    } else {
        // Fused block is NEGATE_A-agnostic: the kernel aliases the signed GM pointers.
        using BlockMmadPlanar = Gemm::Block::BlockMmadTla<
            Gemm::MmadPlanarComplexFused<ArchTag, /*ENABLE_SHUFFLE_K=*/true>, L1TileShapeTla, L0TileShapeTla, ElementA,
            ElementB, ElementC, /*ElementBias=*/void, TileCopy>;
        if (m >= n) {
            // Swizzle offset is 3 and direction is 0.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using PlanarKernel = Gemm::Kernel::PlanarComplexGemm<
                /*USE_FOUR_PASS=*/false, /*NEGATE_A=*/false, void, BlockMmadPlanar, BlockScheduler>;
            using DeviceOp = Gemm::Device::DeviceGemm<PlanarKernel>;

            typename PlanarKernel::Arguments arguments{
                options.problemShape, deviceAReal, deviceAImag, nullptr,     deviceBReal,
                deviceBImag,          nullptr,     deviceCReal, deviceCImag,
            };
            DeviceOp op;
            size_t workspaceSize = op.GetWorkspaceSize(arguments);
            uint8_t* deviceWorkspace{nullptr};
            if (workspaceSize > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            op.Initialize(arguments, deviceWorkspace);
            kernelFailed = (op(stream, aicCoreNum, hardwareSyncAddr) != Status::kSuccess);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            if (deviceWorkspace != nullptr) {
                ACL_CHECK(aclrtFree(deviceWorkspace));
            }
        } else {
            // Swizzle offset is 3 and direction is 1.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using PlanarKernel = Gemm::Kernel::PlanarComplexGemm<
                /*USE_FOUR_PASS=*/false, /*NEGATE_A=*/true, void, BlockMmadPlanar, BlockScheduler>;
            using DeviceOp = Gemm::Device::DeviceGemm<PlanarKernel>;

            typename PlanarKernel::Arguments arguments{
                options.problemShape, deviceAReal, deviceAImag, nullptr,     deviceBReal,
                deviceBImag,          nullptr,     deviceCReal, deviceCImag,
            };
            DeviceOp op;
            size_t workspaceSize = op.GetWorkspaceSize(arguments);
            uint8_t* deviceWorkspace{nullptr};
            if (workspaceSize > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            op.Initialize(arguments, deviceWorkspace);
            kernelFailed = (op(stream, aicCoreNum, hardwareSyncAddr) != Status::kSuccess);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            if (deviceWorkspace != nullptr) {
                ACL_CHECK(aclrtFree(deviceWorkspace));
            }
        }
    }

    ACL_CHECK(aclrtSynchronizeDevice());
    aclError lastRet = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (lastRet != ACL_RT_SUCCESS) {
        kernelFailed = true;
        const char* errMsg = aclGetRecentErrMsg();
        if (errMsg != nullptr) {
            std::cerr << "Kernel failed: " << errMsg << std::endl;
        }
    }

    std::cout << "PlanarComplexGemm: M=" << m << " N=" << n << " K=" << k
              << " variant=" << (useFourPass ? "FourPass" : "Fused") << (kernelFailed ? " [KERNEL FAILED]" : " done")
              << std::endl;

    // ---- Save device output for NumPy-accelerated comparison ----
    if (kernelFailed) {
        std::cerr << "Kernel failed, skipping device output save." << std::endl;
    } else if (hasDataPath) {
        std::vector<float> hostCReal(lenC);
        std::vector<float> hostCImag(lenC);
        ACL_CHECK(aclrtMemcpy(hostCReal.data(), sizeC, deviceCReal, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostCImag.data(), sizeC, deviceCImag, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

        std::ofstream outReal(options.dataPath + "/outputC_real.dat", std::ios::binary);
        std::ofstream outImag(options.dataPath + "/outputC_imag.dat", std::ios::binary);
        outReal.write(reinterpret_cast<const char*>(hostCReal.data()), sizeC);
        outImag.write(reinterpret_cast<const char*>(hostCImag.data()), sizeC);
        std::cout << "Device output saved to " << options.dataPath << " to validate." << std::endl;
    } else {
        std::cout << "No --datapath provided, skipping validation." << std::endl;
    }

    // Cleanup
    ACL_CHECK(aclrtFree(deviceAReal));
    ACL_CHECK(aclrtFree(deviceAImag));
    ACL_CHECK(aclrtFree(deviceBReal));
    ACL_CHECK(aclrtFree(deviceBImag));
    ACL_CHECK(aclrtFree(deviceCReal));
    ACL_CHECK(aclrtFree(deviceCImag));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return !kernelFailed;
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    return Run(options) ? 0 : 1;
}
