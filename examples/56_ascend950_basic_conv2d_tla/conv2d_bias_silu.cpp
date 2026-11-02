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

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/device/device_conv.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv2d_epilogue.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/detail/kernel_adapter.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace Catlass::Conv::Kernel;
using namespace Catlass::Epilogue;

struct Options {
    const std::string HELPER =
        "56_ascend950_basic_conv2d_tla(conv2d_bias_silu) batch, hi, wi, cin, cout, kh, kw, padLeft, padRight, padTop, "
        "padBottom, strideH, strideW, dilationH, dilationW [device_id]";

    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};               // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {2, 2};                  // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    int32_t deviceId{0};

    Catlass::Conv2dParams problemParams{};

    Options() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            BATCH_INDEX = 1,
            HI_INDEX,
            WI_INDEX,
            CIN_INDEX,
            COUT_INDEX,
            KH_INDEX,
            KW_INDEX,
            PADLEFT_INDEX,
            PADRIGHT_INDEX,
            PADTOP_INDEX,
            PADBOTTOM_INDEX,
            STRIDEH_INDEX,
            STRIDEW_INDEX,
            DILATIONH_INDEX,
            DILATIONW_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc <= static_cast<uint32_t>(ArgsIndex::DILATIONW_INDEX)) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        dataSizes[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_INDEX)]);
        dataSizes[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::HI_INDEX)]);
        dataSizes[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::WI_INDEX)]);
        dataSizes[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::CIN_INDEX)]);
        dataSizes[4] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::COUT_INDEX)]);
        filterSizes[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KH_INDEX)]);
        filterSizes[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KW_INDEX)]);
        pads[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADLEFT_INDEX)]);
        pads[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADRIGHT_INDEX)]);
        pads[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADTOP_INDEX)]);
        pads[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADBOTTOM_INDEX)]);
        strides[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::STRIDEH_INDEX)]);
        strides[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::STRIDEW_INDEX)]);
        dilations[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DILATIONH_INDEX)]);
        dilations[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DILATIONW_INDEX)]);

        problemParams = Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

static void Run(Options const& options)
{
    uint32_t c0 = options.problemParams.C0;
    uint32_t batch = options.problemParams.batch();
    uint32_t hi = options.problemParams.hi();
    uint32_t wi = options.problemParams.wi();
    uint32_t cin1 = options.problemParams.cin1();
    uint32_t ho = options.problemParams.ho();
    uint32_t wo = options.problemParams.wo();
    uint32_t cout1 = options.problemParams.cout1();
    uint32_t cout = options.problemParams.cout();
    uint32_t coutRound = options.problemParams.coutRound();
    uint32_t kh = options.problemParams.kh();
    uint32_t kw = options.problemParams.kw();

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    size_t lenFmap = batch * cin1 * hi * wi * c0;
    size_t lenFilter = cin1 * kh * kw * cout * c0;
    size_t lenBias = coutRound;
    size_t lenOutput = batch * ho * wo * coutRound;

    size_t sizeFmap = lenFmap * sizeof(fp16_t);
    size_t sizeFilter = lenFilter * sizeof(fp16_t);
    size_t sizeBias = lenBias * sizeof(fp16_t);
    size_t sizeOutput = lenOutput * sizeof(fp16_t);

    using LayoutFmap = layout::NC1HWC0;
    using LayoutFilter = layout::CI1KHKWCOCI0;
    using LayoutOutput = layout::NC1HWC0;
    LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
    LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
    LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);
    golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostBias, -5.0f, 5.0f);

    uint8_t* deviceFmap{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceFmap), sizeFmap, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFmap, sizeFmap, hostFmap.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceFilter{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceFilter), sizeFilter, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFilter, sizeFilter, hostFilter.data(), sizeFilter, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceOutput{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceOutput), sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(deviceOutput, sizeOutput, 0, sizeOutput));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr uint32_t L1A_STAGES = 2;
    constexpr uint32_t L1B_STAGES = 2;
    constexpr uint32_t L0A_STAGES = 2;
    constexpr uint32_t L0B_STAGES = 2;
    constexpr uint32_t L0C_STAGES = 1;
    constexpr bool ENABLE_UNIT_FLAG = false;
    using DispatchPolicy =
        Conv::ConvPingpong<ArchTag, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
    using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<4, 8, 4>;
    using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<48, 4>;
    using L0TileShape = Catlass::Conv2dL0Shape<16, 16, 16>;

    using ElementFmap = half;
    using ElementFilter = half;
    using ElementOutput = half;

    using BlockConv2d = Conv::Block::BlockConv2dTla<
        DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape, ElementFmap, ElementFilter, ElementOutput>;
    using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

    using ComputeType = Gemm::GemmType<half, layout::NC1HWC0>;
    using OutputType = Gemm::GemmType<half, layout::NC1HWC0>;
    // 950平台：BiasAdd与SiLU按float计算（内联在BlockEpilogue中），TileElemWiseSilu仅用于提供COMPUTE_LENGTH
    using TileEpilogue = Epilogue::Tile::TileElemWiseSilu<ArchTag, Gemm::GemmType<float, layout::NC1HWC0>, 256>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, ComputeType, OutputType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueAscend950Conv2dElemWise, ComputeType, OutputType, TileEpilogue, TileCopy,
        std::integral_constant<uint32_t, FmapL1TileShape::Ho>, std::integral_constant<uint32_t, FmapL1TileShape::Wo>,
        std::integral_constant<uint32_t, FilterL1TileShape::Cout>>;
    using Conv2dKernel = Conv2dEpilogue<BlockConv2d, BlockEpilogue, BlockScheduler>;

    using Conv2dAdapter = Conv::Device::DeviceConv<Conv2dKernel>;

    Conv2dKernel::Arguments arguments{options.problemParams, deviceFmap, deviceFilter, deviceBias, deviceOutput};
    Conv2dAdapter conv2d_op;
    if (conv2d_op.CanImplement(arguments) == Status::kInvalid) {
        std::cerr << "[ERROR]Conv2d op cannot be implemented." << std::endl;

        ACL_CHECK(aclrtFree(deviceFmap));
        ACL_CHECK(aclrtFree(deviceFilter));
        ACL_CHECK(aclrtFree(deviceBias));
        ACL_CHECK(aclrtFree(deviceOutput));
        releaseAclEarly();
        return;
    }

    size_t sizeWorkspace = conv2d_op.GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(deviceWorkspace, sizeWorkspace, 0, sizeWorkspace));
    }

    conv2d_op.Initialize(arguments, deviceWorkspace);
    conv2d_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostWorkspace(sizeWorkspace / sizeof(fp16_t));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMemcpy(
            hostWorkspace.data(), sizeWorkspace, deviceWorkspace, sizeWorkspace, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    std::vector<fp16_t> hostOutput(lenOutput);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> goldenConv(lenOutput);
    golden::ComputeConv2d(
        options.problemParams, hostFmap, layoutFmap, hostFilter, layoutFilter, goldenConv, layoutOutput);

    std::vector<float> goldenFinal(lenOutput);
    for (uint32_t n = 0; n < batch; n++) {
        for (uint32_t c1 = 0; c1 < cout1; c1++) {
            for (uint32_t h = 0; h < ho; h++) {
                for (uint32_t w = 0; w < wo; w++) {
                    for (uint32_t c0Idx = 0; c0Idx < c0; c0Idx++) {
                        uint32_t c = c1 * c0 + c0Idx;
                        if (c >= cout) {
                            continue;
                        }
                        uint64_t offset = (uint64_t(n) * cout1 + c1) * ho * wo * c0 + h * wo * c0 + w * c0 + c0Idx;
                        goldenFinal[offset] =
                            golden::Silu<float>{}(goldenConv[offset] + static_cast<float>(hostBias[c]));
                    }
                }
            }
        }
    }

    std::vector<fp16_t> hostWsCmp;
    if (sizeWorkspace > 0) {
        size_t wsLen = std::min(hostWorkspace.size(), lenOutput);
        hostWsCmp.assign(hostWorkspace.begin(), hostWorkspace.begin() + wsLen);
        golden::ClearInvalidOutput(hostWsCmp, options.problemParams);
    }
    golden::ClearInvalidOutput(hostOutput, options.problemParams);

    auto printErrors = [&](const std::vector<uint64_t>& errors, const char* tag) {
        for (uint64_t i = 0; i < errors.size() && i < 10; i++) {
            uint64_t off = errors[i];
            uint32_t n = off / (uint64_t(cout1) * ho * wo * c0);
            uint64_t rem = off % (uint64_t(cout1) * ho * wo * c0);
            uint32_t c1 = rem / (uint64_t(ho) * wo * c0);
            rem %= uint64_t(ho) * wo * c0;
            uint32_t h = rem / (uint64_t(wo) * c0);
            rem %= uint64_t(wo) * c0;
            uint32_t w = rem / c0;
            uint32_t c = c1 * c0 + rem % c0;
            std::cout << "  [" << tag << "] n=" << n << " c=" << c << " h=" << h << " w=" << w << std::endl;
        }
    };

    if (sizeWorkspace > 0) {
        std::vector<uint64_t> wsErrors = golden::CompareData(hostWsCmp, goldenConv, cin1 * kh * kw * c0);
        std::cout << "Workspace(AIC conv) vs golden conv comparison: ";
        if (wsErrors.empty()) {
            std::cout << "Compare success." << std::endl;
        } else {
            std::cerr << "Compare failed. Error count: " << wsErrors.size() << std::endl;
        }
    }

    std::vector<uint64_t> outErrors = golden::CompareData(hostOutput, goldenFinal, cin1 * kh * kw * c0);
    std::cout << "Output(AIV) vs golden SiLU(conv+bias) comparison: ";
    if (outErrors.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << outErrors.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceFmap));
    ACL_CHECK(aclrtFree(deviceFilter));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(deviceOutput));

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
