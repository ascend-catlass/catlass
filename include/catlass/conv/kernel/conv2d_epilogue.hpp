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

#ifndef CATLASS_CONV_KERNEL_CONV2D_EPILOGUE_HPP
#define CATLASS_CONV_KERNEL_CONV2D_EPILOGUE_HPP

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv_coord.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Conv::Kernel {

template <class BlockConv2d_, class BlockEpilogue_, class BlockScheduler_>
class Conv2dEpilogue {
public:
    using BlockConv2d = BlockConv2d_;
    using ArchTag = typename BlockConv2d::ArchTag;
    using FmapL1TileShape = typename BlockConv2d::FmapL1TileShape;
    using FilterL1TileShape = typename BlockConv2d::FilterL1TileShape;
    using ElementFmap = typename BlockConv2d::ElementFmap;
    using LayoutFmap = typename BlockConv2d::LayoutFmap;
    using ElementFilter = typename BlockConv2d::ElementFilter;
    using LayoutFilter = typename BlockConv2d::LayoutFilter;
    using ElementOutput = typename BlockConv2d::ElementOutput;
    using LayoutOutput = typename BlockConv2d::LayoutOutput;
    using ElementAccumulator = typename BlockConv2d::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    using ElementD = typename BlockEpilogue::ElementD;
    using ElementC = typename BlockEpilogue::ElementC;

    using BlockScheduler = BlockScheduler_;

    // workspace类型（AIC写入/AIV读取），区别于BlockEpilogue中用于SiLU计算的ElementCompute
    using ElementWorkspace = ElementOutput;

    static constexpr uint16_t C0_WORKSPACE = BYTE_PER_C0 / sizeof(ElementWorkspace);

    struct Params {
        Catlass::Conv2dParams problemShape;
        GM_ADDR ptrFmap;
        LayoutFmap layoutFmap;
        GM_ADDR ptrFilter;
        LayoutFilter layoutFilter;
        GM_ADDR ptrBias;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrOutput;
        LayoutOutput layoutOutput;
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
        layout::NC1HWC0 layoutWorkspace;
        layout::NC1HWC0 layoutOutputEpilogue;
#endif
        EpilogueParams epilogueParams;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            Catlass::Conv2dParams const& problemShape_, GM_ADDR ptrFmap_, LayoutFmap layoutFmap_, GM_ADDR ptrFilter_,
            LayoutFilter layoutFilter_, GM_ADDR ptrBias_, GM_ADDR ptrWorkspace_, GM_ADDR ptrOutput_,
            LayoutOutput layoutOutput_, EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_),
              ptrFmap(ptrFmap_),
              layoutFmap(layoutFmap_),
              ptrFilter(ptrFilter_),
              layoutFilter(layoutFilter_),
              ptrBias(ptrBias_),
              ptrWorkspace(ptrWorkspace_),
              ptrOutput(ptrOutput_),
              layoutOutput(layoutOutput_),
              epilogueParams(epilogueParams_)
        {}

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
        CATLASS_HOST_DEVICE
        Params(
            Catlass::Conv2dParams const& problemShape_, GM_ADDR ptrFmap_, LayoutFmap layoutFmap_, GM_ADDR ptrFilter_,
            LayoutFilter layoutFilter_, GM_ADDR ptrBias_, GM_ADDR ptrWorkspace_, GM_ADDR ptrOutput_,
            LayoutOutput layoutOutput_, layout::NC1HWC0 layoutWorkspace_, layout::NC1HWC0 layoutOutputEpilogue_,
            EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_),
              ptrFmap(ptrFmap_),
              layoutFmap(layoutFmap_),
              ptrFilter(ptrFilter_),
              layoutFilter(layoutFilter_),
              ptrBias(ptrBias_),
              ptrWorkspace(ptrWorkspace_),
              ptrOutput(ptrOutput_),
              layoutOutput(layoutOutput_),
              layoutWorkspace(layoutWorkspace_),
              layoutOutputEpilogue(layoutOutputEpilogue_),
              epilogueParams(epilogueParams_)
        {}
#endif
    };

    struct Arguments {
        Catlass::Conv2dParams problemShape;
        GM_ADDR ptrFmap;
        GM_ADDR ptrFilter;
        GM_ADDR ptrBias;
        GM_ADDR ptrOutput;
    };

    static bool CanImplement(const Arguments& args)
    {
        const Catlass::Conv2dParams& problemShape = args.problemShape;
        if (problemShape.strideH() == 0 || problemShape.strideW() == 0 || problemShape.dilationH() == 0 ||
            problemShape.dilationW() == 0) {
            return false;
        }
        // The dilated kernel must fit into the padded input feature map, otherwise no output position exists
        if (problemShape.hi() + problemShape.padTop() + problemShape.padBottom() <
                problemShape.dilationH() * (problemShape.kh() - 1) + 1 ||
            problemShape.wi() + problemShape.padLeft() + problemShape.padRight() <
                problemShape.dilationW() * (problemShape.kw() - 1) + 1) {
            return false;
        }
        return BlockConv2d::CanImplement(problemShape.getFilterParams());
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        uint32_t batch = args.problemShape.batch();
        uint32_t ho = args.problemShape.ho();
        uint32_t wo = args.problemShape.wo();
        uint32_t cout1 = args.problemShape.cout1();
        size_t lenWorkspace = static_cast<size_t>(batch) * cout1 * ho * wo * C0_WORKSPACE;
        size_t result = lenWorkspace * sizeof(ElementWorkspace);
        return result;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
        auto layoutFmap = tla::MakeLayoutFmap<ElementFmap>(
            args.problemShape.batch(), args.problemShape.cin1(), args.problemShape.hi(), args.problemShape.wi());
        auto layoutFilter = tla::MakeLayoutFilter<ElementFilter, Arch::PositionGM>(
            args.problemShape.cin1(), args.problemShape.kh(), args.problemShape.kw(), args.problemShape.cout());
        auto layoutOutput = tla::MakeLayoutFmap<ElementOutput>(
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo());

        layout::NC1HWC0 layoutWorkspace{
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo(),
            C0_WORKSPACE};
        layout::NC1HWC0 layoutOutputEpilogue{
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo(),
            C0_WORKSPACE};

        typename BlockEpilogue::Params epilogueParams(
            (GM_ADDR)workspace, layoutWorkspace, args.ptrBias, args.problemShape.cout(), args.ptrOutput,
            layoutOutputEpilogue);

        Params params{args.problemShape, args.ptrFmap,    layoutFmap,           args.ptrFilter,
                      layoutFilter,      args.ptrBias,    (GM_ADDR)workspace,   args.ptrOutput,
                      layoutOutput,      layoutWorkspace, layoutOutputEpilogue, epilogueParams};
        return params;
#else
        auto layoutFmap = tla::MakeLayoutFmap<ElementFmap>(
            args.problemShape.batch(), args.problemShape.cin1(), args.problemShape.hi(), args.problemShape.wi());
        auto layoutFilter = tla::MakeLayoutFilter<ElementFilter, Arch::PositionGM>(
            args.problemShape.cin1(), args.problemShape.kh(), args.problemShape.kw(), args.problemShape.cout());
        auto layoutOutput = tla::MakeLayoutFmap<ElementOutput>(
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo());

        layout::NC1HWC0 layoutWorkspaceEpilogue{
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo(),
            C0_WORKSPACE};

        typename BlockEpilogue::Params epilogueParams(
            (GM_ADDR)workspace, layoutWorkspaceEpilogue, args.ptrBias, args.problemShape.cout(), args.ptrOutput,
            layoutWorkspaceEpilogue);

        Params params{args.problemShape, args.ptrFmap,       layoutFmap,     args.ptrFilter, layoutFilter,
                      args.ptrBias,      (GM_ADDR)workspace, args.ptrOutput, layoutOutput,   epilogueParams};
        return params;
#endif
    }

    CATLASS_DEVICE
    Conv2dEpilogue()
    {}

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    // 950平台：AIC/AIV统一入口，运行时按核类型分支
    // 使用TLA tensor方式访问GM，bulk同步模型（AIC全部完成后通知AIV）
    CATLASS_DEVICE void operator()(Params const& params)
    {
        using namespace AscendC;

        BlockScheduler conv2dBlockScheduler(
            params.problemShape.getPostIm2colShape(),
            MakeCoord(FmapL1TileShape::Ho, FmapL1TileShape::Wo, FilterL1TileShape::Cout));
        uint32_t loops = conv2dBlockScheduler.GetLoops();
        Catlass::Arch::CrossCoreFlag flagSync{0};

        if ASCEND_IS_AIC {
            using ElementFmapLocal = typename BlockConv2d_::ElementFmap;
            using ElementFilterLocal = typename BlockConv2d_::ElementFilter;
            using ElementOutputLocal = typename BlockConv2d_::ElementOutput;

            uint32_t aicoreIndex = AscendC::GetBlockIdx();
            uint32_t loopStep = AscendC::GetBlockNum();

            BlockConv2d_ blockConv2d(resource, params.problemShape.getFilterParams());

            AscendC::GlobalTensor<ElementFmapLocal> gmFmap;
            gmFmap.SetGlobalBuffer((__gm__ ElementFmapLocal*)params.ptrFmap);
            AscendC::GlobalTensor<ElementFilterLocal> gmFilter;
            gmFilter.SetGlobalBuffer((__gm__ ElementFilterLocal*)params.ptrFilter);
            AscendC::GlobalTensor<ElementOutputLocal> gmWorkspace;
            gmWorkspace.SetGlobalBuffer((__gm__ ElementOutputLocal*)params.ptrWorkspace);

            auto tensorFmap = tla::MakeTensor(gmFmap, params.layoutFmap, Catlass::Arch::PositionGM{});
            auto tensorFilter = tla::MakeTensor(gmFilter, params.layoutFilter, Catlass::Arch::PositionGM{});
            auto tensorOutput = tla::MakeTensor(gmWorkspace, params.layoutOutput, Catlass::Arch::PositionGM{});

            for (uint32_t loopIdx = aicoreIndex; loopIdx < loops; loopIdx += loopStep) {
                ProcessConv2dBlock(
                    params, conv2dBlockScheduler, blockConv2d, tensorFmap, tensorFilter, tensorOutput, loopIdx);
            }

            // AIC barrier: wait for all AIC cores to finish before setting flag
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();
            // AIC core finishes all loops, set flag for corresponding AIV core
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagSync);
        } else if ASCEND_IS_AIV {
            using ElementD = typename BlockEpilogue_::ElementD;

            // AIV core waits for corresponding AIC core to finish
            Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagSync);
            // AIV barrier: wait for all AIV cores to start processing together
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

            uint32_t taskRation = static_cast<uint32_t>(AscendC::GetTaskRation());
            uint32_t aivCoreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx()) / taskRation;
            uint32_t aivSubIdx = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            uint32_t aivLoopStep = static_cast<uint32_t>(AscendC::GetBlockNum()) * taskRation;
            uint32_t aivLoopStart = aivCoreIndex + aivSubIdx * static_cast<uint32_t>(AscendC::GetBlockNum());

            BlockEpilogue_ blockEpilogue(resource, params.epilogueParams);

            // workspace类型由ElementWorkspace决定（AIC写入/AIV读取）
            AscendC::GlobalTensor<ElementWorkspace> gmWorkspace;
            gmWorkspace.SetGlobalBuffer((__gm__ ElementWorkspace*)params.ptrWorkspace);
            AscendC::GlobalTensor<ElementD> gmOutput;
            gmOutput.SetGlobalBuffer((__gm__ ElementD*)params.ptrOutput);

            for (uint32_t loopIdx = aivLoopStart; loopIdx < loops; loopIdx += aivLoopStep) {
                Catlass::Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
                Catlass::Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);

                blockEpilogue(blockCoord, actualBlockShape, gmWorkspace, gmOutput);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
#else
    // AtlasA2平台：AIC/AIV模板特化，TLA tensor方式访问GM，逐block同步模型
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        using namespace AscendC;

        BlockScheduler conv2dBlockScheduler(
            params.problemShape.getPostIm2colShape(),
            MakeCoord(FmapL1TileShape::Ho, FmapL1TileShape::Wo, FilterL1TileShape::Cout));
        uint32_t loops = conv2dBlockScheduler.GetLoops();

        BlockEpilogue_ blockEpilogue(resource, params.epilogueParams);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementD> gmOutput;
        gmOutput.SetGlobalBuffer((__gm__ ElementD*)params.ptrOutput);

        for (uint32_t loopIdx = coreIdx; loopIdx < loops; loopIdx += coreNum) {
            Catlass::Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);

            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            blockEpilogue(blockCoord, actualBlockShape, gmWorkspace, gmOutput);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        using namespace AscendC;

        BlockScheduler conv2dBlockScheduler(
            params.problemShape.getPostIm2colShape(),
            MakeCoord(FmapL1TileShape::Ho, FmapL1TileShape::Wo, FilterL1TileShape::Cout));
        uint32_t loops = conv2dBlockScheduler.GetLoops();

        BlockConv2d_ blockConv2d(resource, params.problemShape.getFilterParams());

        AscendC::GlobalTensor<ElementFmap> gmFmap;
        gmFmap.SetGlobalBuffer((__gm__ ElementFmap*)params.ptrFmap);
        AscendC::GlobalTensor<ElementFilter> gmFilter;
        gmFilter.SetGlobalBuffer((__gm__ ElementFilter*)params.ptrFilter);
        AscendC::GlobalTensor<ElementOutput> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementOutput*)params.ptrWorkspace);

        auto tensorFmap = tla::MakeTensor(gmFmap, params.layoutFmap, Catlass::Arch::PositionGM{});
        auto tensorFilter = tla::MakeTensor(gmFilter, params.layoutFilter, Catlass::Arch::PositionGM{});
        auto tensorOutput = tla::MakeTensor(gmWorkspace, params.layoutOutput, Catlass::Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < loops; loopIdx += AscendC::GetBlockNum()) {
            ProcessConv2dBlock(
                params, conv2dBlockScheduler, blockConv2d, tensorFmap, tensorFilter, tensorOutput, loopIdx);
            // 完成信号必须与AIV侧每个输出块的WaitFlag一一配对：即使当前块完全位于padding区域被
            // 跳过（ProcessConv2dBlock返回false），也必须发送信号，否则AIV会永久等待导致kernel挂死。
            // 被跳过块的workspace保持为0（host侧分配workspace后memset清零），AIV epilogue据此
            // 输出SiLU(0 + bias)，与golden参考及950平台bulk同步路径的行为一致。
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif

private:
    // AIC 循环体公共逻辑：padding 计算 + GetTile + blockConv2d 调用
    // 返回 true 表示已处理，false 表示 hiActual==0||wiActual==0 跳过
    template <class TensorFmap, class TensorFilter, class TensorOutput, class BlockConv2dType>
    CATLASS_DEVICE bool ProcessConv2dBlock(
        Params const& params, BlockScheduler& conv2dBlockScheduler, BlockConv2dType& blockConv2d,
        TensorFmap const& tensorFmap, TensorFilter const& tensorFilter, TensorOutput const& tensorOutput,
        uint32_t loopIdx)
    {
        using FmapL1TileShapeLocal = typename BlockConv2dType::FmapL1TileShape;
        using FilterL1TileShapeLocal = typename BlockConv2dType::FilterL1TileShape;
        static constexpr uint16_t C0_FMAP = BYTE_PER_C0 / sizeof(typename BlockConv2dType::ElementFmap);
        static constexpr uint16_t C0_FILTER = BYTE_PER_C0 / sizeof(typename BlockConv2dType::ElementFilter);
        static constexpr uint16_t C0_OUT = BYTE_PER_C0 / sizeof(typename BlockConv2dType::ElementOutput);

        Catlass::Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
        Catlass::Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);

        uint8_t blockPadLeft = 0, blockPadRight = 0, blockPadTop = 0, blockPadBottom = 0;

        // Compute indices of hi
        uint32_t hoStart = blockCoord.h() * FmapL1TileShapeLocal::Ho;
        int32_t hiStart = hoStart * params.problemShape.strideH() - params.problemShape.padTop();
        int32_t hiEnd = hiStart + (actualBlockShape.h() - 1) * params.problemShape.strideH() +
                        (params.problemShape.kh() - 1) * params.problemShape.dilationH();
        int32_t hiLast = params.problemShape.hi() - 1;
        if (hiStart > hiLast || hiEnd < 0) {
            return false;
        }
        if (hiStart < 0) {
            blockPadTop = 0 - hiStart;
            hiStart = 0;
        }
        if (hiEnd > hiLast) {
            blockPadBottom = hiEnd - hiLast;
            hiEnd = hiLast;
        }
        uint32_t hiActual = hiEnd - hiStart + 1;

        // Compute indices of wi
        uint32_t woStart = blockCoord.w() * FmapL1TileShapeLocal::Wo;
        int32_t wiStart = woStart * params.problemShape.strideW() - params.problemShape.padLeft();
        int32_t wiEnd = wiStart + (actualBlockShape.w() - 1) * params.problemShape.strideW() +
                        (params.problemShape.kw() - 1) * params.problemShape.dilationW();
        int32_t wiLast = params.problemShape.wi() - 1;
        if (wiStart > wiLast || wiEnd < 0) {
            return false;
        }
        if (wiStart < 0) {
            blockPadLeft = 0 - wiStart;
            wiStart = 0;
        }
        if (wiEnd > wiLast) {
            blockPadRight = wiEnd - wiLast;
            wiEnd = wiLast;
        }
        uint32_t wiActual = wiEnd - wiStart + 1;

        Catlass::Conv2dCoord actualConv2dBlockShape(
            1, hiActual, wiActual, actualBlockShape.cout(), actualBlockShape.cin1());
        uint8_t blockPadList[4] = {blockPadLeft, blockPadRight, blockPadTop, blockPadBottom};

        auto tensorBlockFmap = tla::GetTile(
            tensorFmap, tla::MakeCoord(blockCoord.batch(), 0, (uint32_t)hiStart, (uint32_t)wiStart, 0),
            tla::MakeShape(actualBlockShape.batch(), actualBlockShape.cin1(), hiActual, wiActual, C0_FMAP));

        auto tensorBlockFilter = tla::GetTile(
            tensorFilter, tla::MakeCoord(0, 0, 0, blockCoord.cout() * FilterL1TileShapeLocal::Cout, 0),
            tla::MakeShape(
                actualBlockShape.cin1(), params.problemShape.kh(), params.problemShape.kw(), actualBlockShape.cout(),
                C0_FILTER));

        auto tensorBlockOutput = tla::GetTile(
            tensorOutput,
            tla::MakeCoord(
                blockCoord.batch(), blockCoord.cout() * FilterL1TileShapeLocal::Cout / C0_OUT, hoStart, woStart, 0),
            tla::MakeShape(
                actualBlockShape.batch(), CeilDiv(actualBlockShape.cout(), C0_OUT), actualBlockShape.h(),
                actualBlockShape.w(), C0_OUT));

        blockConv2d(tensorBlockFmap, tensorBlockFilter, tensorBlockOutput, actualConv2dBlockShape, blockPadList);
        return true;
    }

    static constexpr Catlass::Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Conv::Kernel

#endif // CATLASS_CONV_KERNEL_CONV2D_EPILOGUE_HPP
