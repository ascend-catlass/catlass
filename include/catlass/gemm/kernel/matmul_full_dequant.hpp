/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class LayoutA_, class LayoutB_, class LayoutC_>
class GemmUniversal {
public:
    CATLASS_DEVICE
    GemmUniversal() = default;
    CATLASS_DEVICE
    ~GemmUniversal() = default;

    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementOut = typename BlockEpilogue::ElementOut;

    static_assert(std::is_same_v<ElementC, typename BlockEpilogue::ElementIn>);
    static_assert(std::is_same_v<ElementBias, typename BlockEpilogue::ElementBias>);
    static_assert(std::is_same_v<LayoutA_, layout::RowMajor>, "LayoutA must be RowMajor");
    static_assert(std::is_same_v<LayoutB_, layout::RowMajor>, "LayoutB must be RowMajor");
    static_assert(std::is_same_v<LayoutC_, layout::RowMajor>, "LayoutC must be RowMajor");

    using LayoutA = Catlass::detail::TagToLayout_t<ElementA, LayoutA_>;
    using LayoutB = Catlass::detail::TagToLayout_t<ElementB, LayoutB_>;
    using LayoutC = Catlass::detail::TagToLayout_t<ElementOut, LayoutC_>;

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA{nullptr};
        GM_ADDR ptrB{nullptr};
        GM_ADDR ptrD{nullptr};
        GM_ADDR ptrX2Scale{nullptr};
        GM_ADDR ptrX1Scale{nullptr};
        GM_ADDR ptrBias{nullptr};
        Arguments() = default;
    };
    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA{nullptr};
        LayoutA layoutA{};
        GM_ADDR ptrB{nullptr};
        LayoutB layoutB{};
        GM_ADDR ptrD{nullptr};
        LayoutC layoutD{};
        GM_ADDR ptrX2Scale{nullptr};
        GM_ADDR ptrX1Scale{nullptr};
        GM_ADDR ptrBias{nullptr};
        Params() = default;
    };

private:
    using AGlobalTensorType = AscendC::GlobalTensor<ElementA>;
    using BGlobalTensorType = AscendC::GlobalTensor<ElementB>;
    using CLocalTensorType = AscendC::LocalTensor<ElementC>;

    using AGmTensor = tla::Tensor<AGlobalTensorType, LayoutA, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using BGmTensor = tla::Tensor<BGlobalTensorType, LayoutB, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    AGlobalTensorType aGm_;
    BGlobalTensorType bGm_;
    CLocalTensorType cUb_;
    AGmTensor aGmTensor_;
    BGmTensor bGmTensor_;

    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static int16_t AIV_SYNC_AIC_FLAG = 5;
    constexpr static int16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static int16_t FLAG_ID_MAX = 16;
    constexpr static int16_t COUNT_ID_MAX = 15;
    constexpr static int16_t COUNT_FLAG = 3;

    CATLASS_DEVICE
    void Init(Params const& params)
    {
        int64_t m = params.problemShape.m();
        int64_t n = params.problemShape.n();
        int64_t k = params.problemShape.k();

        AscendC::GlobalTensor<ElementA> tempA;
        tempA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA), m * k);
        aGm_.address_ = tempA.address_;
        AscendC::GlobalTensor<ElementB> tempB;
        tempB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB), k * n);
        bGm_.address_ = tempB.address_;

        aGmTensor_ = tla::MakeTensor(aGm_, params.layoutA, Arch::PositionGM{});
        bGmTensor_ = tla::MakeTensor(bGm_, params.layoutB, Arch::PositionGM{});
    }

public:
    CATLASS_HOST_DEVICE
    static bool CanImplement(Arguments const& args)
    {
        if (args.problemShape.m() == 0 || args.problemShape.n() == 0 || args.problemShape.k() == 0 ||
            args.ptrA == nullptr || args.ptrB == nullptr || args.ptrD == nullptr || args.ptrX2Scale == nullptr) {
            return false;
        }
        if constexpr (BlockEpilogue::x1QuantMode != Catlass::Epilogue::QuantMode::NONE) {
            if (args.ptrX1Scale == nullptr) {
                return false;
            }
        }
        if constexpr (BlockEpilogue::hasBias) {
            if (args.ptrBias == nullptr) {
                return false;
            }
        }
        return BlockEpilogue::CanImplement() == Status::kSuccess;
    }

    CATLASS_HOST_DEVICE
    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return BlockEpilogue::GetWorkspaceSize();
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(Arguments const& args, GM_ADDR workspace)
    {
        LayoutA layoutA = tla::MakeLayout<ElementA, LayoutA_>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = tla::MakeLayout<ElementB, LayoutB_>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutD = tla::MakeLayout<ElementOut, LayoutC_>(args.problemShape.m(), args.problemShape.n());
        Params params{};
        params.problemShape = args.problemShape;
        params.ptrA = args.ptrA;
        params.layoutA = layoutA;
        params.ptrB = args.ptrB;
        params.layoutB = layoutB;
        params.ptrD = args.ptrD;
        params.layoutD = layoutD;
        params.ptrX2Scale = args.ptrX2Scale;
        params.ptrX1Scale = args.ptrX1Scale;
        params.ptrBias = args.ptrBias;
        return params;
    }

    CATLASS_DEVICE void operator()(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        constexpr bool enable2UB = (AscendC::IsSameType<ElementC, ElementAccumulator>::value);
        if ASCEND_IS_AIV {
            if constexpr (!enable2UB && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        if (curBlockIdx >= blockNum) {
            return;
        }

        Init(params);
        BlockEpilogue epilogue(
            resource, params.ptrD, params.ptrX2Scale, params.ptrX1Scale, params.ptrBias, params.problemShape);
        BlockScheduler bs(curBlockIdx, blockNum, params.problemShape);

        if (bs.endBlockIdx_ + 1 <= blockNum / 2) {
            bs.UpdateTailTile();
        }
        bool enableCVSync = false;
        int count = 0;
        int64_t count_id = 0;
        uint32_t coreLoops = bs.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= bs.endBlockIdx_);
            bs.UpdateMNTileIdx(loopIdx, isLastLoop);
            bs.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = bs.GetBlockShape();
            auto blkElemCoord = bs.GetBlockCoordByElement();

            uint32_t mCoord = blkElemCoord.m();
            uint32_t nCoord = blkElemCoord.n();
            uint32_t kCoord = blkElemCoord.k();

            uint32_t blockM = blockShape.m();
            uint32_t blockN = blockShape.n();
            uint32_t blockK = blockShape.k();

            GemmCoord actualBlockShape = GemmCoord{blockM, blockN, blockK};

            if (blockM <= 0 || blockN <= 0) {
                break;
            }

            auto ubLocalTemp = resource.ubBuf.template GetBufferByByte<ElementC>(0);

            auto tensorA = tla::MakeTensor(aGm_, aGmTensor_.layout(), Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(bGm_, bGmTensor_.layout(), Arch::PositionGM{});

            cUb_.SetAddr(ubLocalTemp.address_);
            int64_t alignN = RoundUp(blockN, static_cast<int64_t>(Catlass::BYTE_PER_BLK / sizeof(ElementC)));
            auto cUbLayout = tla::MakeLayout<ElementC, layout::RowMajor>(blockM, alignN);
            auto tensorC = tla::MakeTensor(cUb_, cUbLayout, Arch::PositionUB{});

            if ASCEND_IS_AIC {
                if (enableCVSync) {
                    count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + count_id);
                    if constexpr (enable2UB) {
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                            AIV_SYNC_AIC_FLAG + count_id + FLAG_ID_MAX);
                    }
                }

                auto aInputTlaTensor =
                    GetTile(aGmTensor_, tla::MakeCoord(mCoord, kCoord), tla::MakeShape(blockM, blockK));
                auto bInputTlaTensor =
                    GetTile(bGmTensor_, tla::MakeCoord(kCoord, nCoord), tla::MakeShape(blockK, blockN));
                auto cOutputTlaTensor = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(blockM, alignN));

                blockMmad(aInputTlaTensor, bInputTlaTensor, cOutputTlaTensor, actualBlockShape);
                enableCVSync = true;
                count++;
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + count_id);
                if constexpr (enable2UB) {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIC_SYNC_AIV_FLAG + count_id + FLAG_ID_MAX);
                }
            }
            // AIV Process
            if ASCEND_IS_AIV {
                // Synchronize with aic
                count++;
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + count_id);
                // Calulate epilogue
                auto cUbTlaTensor = GetTile(tensorC, tla::MakeCoord(mCoord, nCoord), tla::MakeShape(blockM, blockN));
                epilogue(cUbTlaTensor);
                // Notify aic
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + count_id);
            }
        }
        // Match extra event after aic process finished
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + count_id);
                if constexpr (enable2UB) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + count_id + FLAG_ID_MAX);
                }
            }
        }
    }
};

} // namespace Catlass::Gemm::Kernel
#endif // CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP
