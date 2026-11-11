/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <iostream>

#include "catlass/gemm/kernel/matrix_inverse.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel_prebuilt.h"
#include "../common/workspace_alloc.h"

namespace CatlassKernel {
using namespace Catlass;
using namespace tla;

#define ACL_CHECK(status)                                                                   \
    do {                                                                                    \
        aclError error = status;                                                            \
        if (error != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl; \
        }                                                                                   \
    } while (0)

/**
 * @brief 模板化的 MatrixInverse kernel 实现。
 *
 * @tparam Element 矩阵元素类型（float / half / bfloat16_t）。
 */
template <class Element>
void MatrixInverseImpl(const uint32_t blockNum, aclrtStream stream, const MatrixInverseParams& params)
{
    uint32_t N = params.N;
    uint8_t* deviceA = params.inputAddr.at(0);

    // Allocate pivot array as internal workspace (kernel writes/reads ipiv internally)
    size_t sizeIpiv = static_cast<size_t>(N) * sizeof(int32_t);
    uint8_t* deviceIpiv = g_catlassWorkspaceAlloc(sizeIpiv);

    using ArchTag = Arch::AtlasA2;
    using LayoutTag = layout::RowMajor;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;

    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _64>;
    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutTag, Element, LayoutTag, Element, LayoutTag>;
    using BlockMmadType =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, Element, Element, Element, void, TileCopy>;
    using BlockSchedulerType = Gemm::Block::GemmIdentityBlockSwizzle<>;

    using InverterKernel = Gemm::Kernel::MatrixInverse<ArchTag, Element, BlockMmadType, BlockSchedulerType>;
    using InverterAdapter = Gemm::Device::DeviceGemm<InverterKernel>;

    auto layoutA = tla::MakeLayout<Element, LayoutTag>(N, N);

    typename InverterKernel::Arguments arguments{N, deviceA, layoutA, deviceIpiv, nullptr};

    InverterAdapter invOp;
    if (invOp.CanImplement(arguments) != Status::kSuccess) {
        return;
    }

    size_t sizeWorkspace = invOp.GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        deviceWorkspace = g_catlassWorkspaceAlloc(sizeWorkspace);
        arguments = typename InverterKernel::Arguments{N, deviceA, layoutA, deviceIpiv, deviceWorkspace};
    }

    invOp.Initialize(arguments, deviceWorkspace);
    invOp(stream, blockNum);
    if (g_catlassWorkspaceFree != nullptr && aclrtSynchronizeStream(stream) == ACL_ERROR_NONE) {
        g_catlassWorkspaceFree(deviceIpiv, sizeIpiv);
        if (deviceWorkspace != nullptr) {
            g_catlassWorkspaceFree(deviceWorkspace, sizeWorkspace);
        }
    }
}

void MatrixInverse(const uint32_t blockNum, aclrtStream stream, const MatrixInverseParams& params)
{
    if (params.dataType == ACL_FLOAT) {
        MatrixInverseImpl<float>(blockNum, stream, params);
    } else {
        std::cerr << "MatrixInverse: unsupported dataType " << params.dataType << " (only ACL_FLOAT is supported)"
                  << std::endl;
    }
}

} // namespace CatlassKernel
