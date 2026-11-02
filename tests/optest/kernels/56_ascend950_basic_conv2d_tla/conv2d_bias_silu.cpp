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

#include <acl/acl.h>
#include <iostream>
#include <stdexcept>

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

#include "catlass_kernel_prebuilt.h"
#include "../common/workspace_alloc.h"

namespace CatlassKernel {
using namespace Catlass;
using namespace Catlass::Conv::Kernel;
using namespace Catlass::Epilogue;

#define ACL_CHECK(status)                                                                   \
    do {                                                                                    \
        aclError error = status;                                                            \
        if (error != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl; \
        }                                                                                   \
    } while (0)

/**
 * @brief Prebuilt entry for example 56_ascend950_basic_conv2d_tla(conv2d_bias_silu).
 *
 * Computes SiLU(Conv2D(fmap, filter) + bias) with fp32 accumulation and fp32 SiLU
 * intermediate computation on Ascend950 (3510). Workspace carries the fp16 conv
 * result from the AIC cores to the AIV cores.
 */
void Ascend950Conv2dBiasSilu(const uint32_t blockNum, aclrtStream stream, const ConvParams& params)
{
    uint32_t batch = params.fmapRelated[0];
    uint32_t hi = params.fmapRelated[1];
    uint32_t wi = params.fmapRelated[2];
    uint32_t cin = params.fmapRelated[3];
    uint32_t cout = params.fmapRelated[4];

    uint8_t kh = params.filterRelated[0];
    uint8_t kw = params.filterRelated[1];

    uint8_t padLeft = params.padList[0];
    uint8_t padRight = params.padList[1];
    uint8_t padTop = params.padList[2];
    uint8_t padBottom = params.padList[3];

    uint8_t strideH = params.strideList[0];
    uint8_t strideW = params.strideList[1];

    uint8_t dilationH = params.dilationList[0];
    uint8_t dilationW = params.dilationList[1];

    Conv2dParams problemParams(
        batch, hi, wi, cin, cout, kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW);

    uint8_t* deviceFmap = params.inputAddr.at(0);
    uint8_t* deviceFilter = params.inputAddr.at(1);
    uint8_t* deviceBias = params.inputAddr.at(2);
    uint8_t* deviceOutput = params.outputAddr.at(0);

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
    // 950平台：SiLU按float计算（内联在BlockEpilogue中），TileElemWiseSilu仅用于提供COMPUTE_LENGTH
    // 全限定 Epilogue::Tile / Epilogue::Block，避免无限定 Tile/Block 与 tla::Tile 等产生歧义
    using TileEpilogue = Epilogue::Tile::TileElemWiseSilu<ArchTag, Gemm::GemmType<float, layout::NC1HWC0>, 256>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, ComputeType, OutputType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueAscend950Conv2dElemWise, ComputeType, OutputType, TileEpilogue, TileCopy,
        std::integral_constant<uint32_t, FmapL1TileShape::Ho>, std::integral_constant<uint32_t, FmapL1TileShape::Wo>,
        std::integral_constant<uint32_t, FilterL1TileShape::Cout>>;
    using Conv2dKernel = Conv::Kernel::Conv2dEpilogue<BlockConv2d, BlockEpilogue, BlockScheduler>;

    using Conv2dAdapter = Conv::Device::DeviceConv<Conv2dKernel>;

    Conv2dKernel::Arguments arguments{problemParams, deviceFmap, deviceFilter, deviceBias, deviceOutput};
    Conv2dAdapter conv2dOp;

    auto status = conv2dOp.CanImplement(arguments);
    if (status == Status::kInvalid) {
        throw std::runtime_error("Conv2d+SiLU op cannot be implemented: L1TileShape/L0TileShape exceeds the L1 space");
    }

    size_t sizeWorkspace = conv2dOp.GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        deviceWorkspace = g_catlassWorkspaceAlloc(sizeWorkspace);
        ACL_CHECK(aclrtMemset(deviceWorkspace, sizeWorkspace, 0, sizeWorkspace));
    }

    conv2dOp.Initialize(arguments, deviceWorkspace);

    conv2dOp(stream, blockNum);
}

} // namespace CatlassKernel
