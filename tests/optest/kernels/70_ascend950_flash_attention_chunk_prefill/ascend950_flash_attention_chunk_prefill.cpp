
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

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "fai_tilingdata.h"
#include "fai_tiling.cpp"

#include "catlass_kernel_prebuilt.h"
#include "../common/workspace_alloc.h"

// #include <cstddef>
// using std::size_t;
// using namespace tla;
// using namespace Catlass;
namespace CatlassKernel {
using namespace Catlass;
using namespace tla;
using namespace std;
using namespace optiling;

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        (void)error;                                                                         \
    } while (0)

static constexpr uint32_t PRE_LAUNCH = 2;
static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
static constexpr uint32_t UB_S_BUF_STAGES = 2;

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    Format qFormat,
    Format kvFormat,
    CacheMode kvcacheType,
    PageShape kvcacheShape,
    MaskCategory maskCategory,
    CacheLayout cacheLayout,
    bool enableDN>
class FAIKernel950 {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = std::conditional_t<enableDN, typename BlockMmadQK::ElementB, typename BlockMmadQK::ElementA>;
    using ElementK = std::conditional_t<enableDN, typename BlockMmadQK::ElementA, typename BlockMmadQK::ElementB>;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;
    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;

    using LayoutQ = std::conditional_t<enableDN, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<enableDN, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutMask = layout::RowMajor;

    __aicore__ inline
    FAIKernel950() {}

    __aicore__ inline
    void operator()(PFAKernelParams const &params)
    {
        __gm__ FAInferTilingData *faiTilingData =
            reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        //tiling data

        batch_ = faiTilingData->batch;
        qHeads_ = faiTilingData->numHeads;
        kvHeads_ = faiTilingData->kvHeads;
        embed_ = faiTilingData->embeddingSize;
        embedV_ = faiTilingData->embeddingSizeV;
        firstBatchTaskNum_ = faiTilingData->firstBatchTaskNum;
        totalTaskNum_ = faiTilingData->totalTaskNum;
        scaleValue_ = faiTilingData->scaleValue;
        // base tile info
        qBaseTile_ = faiTilingData->qBaseTile;
        kvBaseTile_ = faiTilingData->kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = faiTilingData->actSeqAval;
        // aligned seqlen q & kv
        qSeqlenAligned_ = faiTilingData->qSeqlenAligned;
        kvSeqlenAligned_ = faiTilingData->kvSeqlenAligned;
        maxNumBlocksPerBatch_ = faiTilingData->maxNumBlocksPerBatch;
        blockSize_ = faiTilingData->blockSize;
        numBlocks_ = faiTilingData->numBlocks;
        qkL1TileM_ = faiTilingData->qkL1TileM;
        qkL1TileN_ = faiTilingData->qkL1TileN;
        qkL1TileKLeft_ = faiTilingData->qkL1TileKLeft;
        qkL1TileKRight_ = faiTilingData->qkL1TileKRight;
        pvL1TileM_ = faiTilingData->pvL1TileM;
        pvL1TileN_ = faiTilingData->pvL1TileN;
        pvL1TileKLeft_ = faiTilingData->pvL1TileKLeft;
        pvL1TileKRight_ = faiTilingData->pvL1TileKRight;
        qL1BufNum_ = faiTilingData->qL1BufNum;
        kL1BufNum_ = faiTilingData->kL1BufNum;
        vL1BufNum_ = faiTilingData->vL1BufNum;
        pL1BufNum_ = faiTilingData->pL1BufNum;

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        auto pvL1AddrStart_ = qkL1TileM_ * qkL1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
            qkL1TileKRight_ * qkL1TileN_ * kL1BufNum_ * sizeof(ElementK);
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * i);
        }
        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;

        uint32_t ubSRoundTile = 0;
        if constexpr (std::is_same_v<ElementS, half>) {
            for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
                ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
                ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES +
                    rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
            }
            ubSRoundTile = 16;
        } else {
            for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
                ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                    32768 * i);
            }

            for (uint32_t i = 0; i < UB_S_BUF_STAGES; i++) {
                ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                    4 * 32768 +
                    rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
            }
            ubSRoundTile = 128;
        }

        Gemm::Block::BlockMmadQKTileHelper qkL1TileHelper(qkL1TileM_, qkL1TileN_, qkL1TileKLeft_, qkL1TileKRight_,
            qL1BufNum_, kL1BufNum_);
        Gemm::Block::BlockMmadPVTileHelper pvL1TileHelper(pvL1TileM_, pvL1TileN_, pvL1TileKLeft_, pvL1TileKRight_,
            pL1BufNum_, vL1BufNum_);

        qkL0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
        qkL0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        pvL0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        pvL0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, qkL1TileHelper);
        BlockMmadPV blockMmadPV(resource, pvL1AddrStart_, pvL1TileHelper);
#endif

#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        int64_t strideQ = 0;
        int64_t strideO = 0;
        int64_t strideK = 0;
        int64_t strideV = 0;
        int64_t kvNumTokens = gActualKvseqlen.GetValue(batch_); // used for TND_NZ

        strideQ = qHeads_ * embed_;
        if constexpr (cacheLayout == CacheLayout::nd) {
            strideK = kvHeads_ * embed_;
            strideV = kvHeads_ * embedV_;
        } else {
            strideK = 16;
            strideV = 16;
        }
        strideO = qHeads_ * embedV_;

        uint32_t embedVRound = RoundUp(embedV_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t blockBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        int64_t qSeqlen = gActualQseqlen.GetValue(curBatch);
        int64_t kvSeqlen = gActualKvseqlen.GetValue(curBatch);
        qSeqlen = gActualQseqlen.GetValue(curBatch + 1);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                qBOffset += qSeqlen * strideQ;
                blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch_); 
                oBOffset += qSeqlen * strideO;
                
                qSeqlen = gActualQseqlen.GetValue(curBatch);
                kvSeqlen = gActualKvseqlen.GetValue(curBatch);
                qSeqlen = gActualQseqlen.GetValue(curBatch + 1) - static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                curTotalTaskNum += (qSeqlen + qBaseTile_ - 1) / qBaseTile_ * qHeads_;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSTileIdx = taskIdxCurBatch / qHeads_;
            uint32_t qHeadIdx = taskIdxCurBatch - qSTileIdx * qHeads_;
            uint32_t kvHeadIdx = qHeadIdx / groupSize;
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = qSTileIdx * 128;
            gmOffsetQ = qBOffset + qSOffset * strideQ + qHeadIdx * embed_;
            gmOffsetO = oBOffset + qSOffset * strideO + qHeadIdx * embedV_;
            strideK = embed_;
            strideV = embedV_;
            gmOffsetK = static_cast<uint64_t>(kvHeadIdx) * embed_ * blockSize_;
            gmOffsetV = static_cast<uint64_t>(kvHeadIdx) * embedV_ * blockSize_;
           
            uint32_t qsBlockNum =  (qSeqlen + qBaseTile_ - 1) / qBaseTile_;
            uint32_t rowNum = qSTileIdx == qsBlockNum - 1 ? qSeqlen - (qsBlockNum - 1) * qBaseTile_ : qBaseTile_;
            uint32_t rowNumRound = 0;
            if constexpr (enableDN) {
                rowNumRound = RoundUp(rowNum, 32);
            } else {
                rowNumRound = RoundUp(rowNum, 16);
            }
            uint32_t kvSTileSizeAct = kvBaseTile_;

            uint32_t noSkipKvS = kvSeqlen;
            uint32_t diffS = kvSeqlen - qSeqlen;
            noSkipKvS = (qSTileIdx + 1U) * qBaseTile_ + diffS;
            noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);
            

            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
#ifdef __DAV_CUBE__
            uint32_t qShapeCol = strideQ;
            uint32_t kShapeCol = strideK;
            uint32_t vShapeCol = strideV;
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
            auto gmQLayoutTlaDN = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            auto gmQTensorTlaDN = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTlaDN, Arch::PositionGM{});
            
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            if constexpr (enableDN) {
                blockMmadQK.loadQGM(gmQTensorTlaDN, actualBlockShapeQ);
            } else {
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
            }
            uint32_t kShapeRow = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache) {
                kShapeRow = numBlocks_ * blockSize_ * kvHeads_;
            } else {
                kShapeRow = kvSeqlen;
            }
            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeCol, kShapeRow);
            auto gmKLayoutTlaDN = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmKTensorTlaDN = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTlaDN, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = strideO;
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t kvSTileIdx = 0; kvSTileIdx < kvSLoopNum + PRE_LAUNCH; kvSTileIdx++) {
                if (kvSTileIdx < kvSLoopNum) {
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, ubSRoundTile));
                    auto ubSLayoutTlaDN = tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, 16), 64);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    auto ubSTensorTlaDN = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTlaDN, Arch::PositionUB{});
                    uint32_t qkReadyFlagId = ubSBufId;
                    Arch::CrossCoreFlag qkReadyFlag(qkReadyFlagId);
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdx, qkL0ATotalStages_, pvL0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdx, qkL0BTotalStages_, pvL0BTotalStages_, kvSLoopNum, true);
                    if constexpr (enableDN) {
                        blockMmadQK(
                            gmKTensorTlaDN, ubSTensorTlaDN,
                            gBlockTable[blockBOffset],
                            actualBlockShapeQK,
                            blockSize_,
                            kvSTileIdx, 0, kvHeads_,
                            kvNumTokens,
                            kvBaseTile_, 0, 0, 0,
                            qkReadyFlag,
                            prefixSumL0AStages, 
                            prefixSumL0BStages);
                    } else {
                        blockMmadQK(
                            gmKTensorTla, ubSTensorTla,
                            gBlockTable[blockBOffset],
                            actualBlockShapeQK,
                            blockSize_,
                            kvSTileIdx, 0, kvHeads_,
                            kvNumTokens,
                            kvBaseTile_, 0, 0, 0,
                            qkReadyFlag,
                            prefixSumL0AStages, 
                            prefixSumL0BStages);
                    }
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    uint32_t l1PBufId = kvSTileIdx % pL1BufNum_;
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, Catlass::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId],
                    l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    auto gmMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(2048, 2048);
                    auto gmMaskTensorTla = tla::MakeTensor(gMask, gmMaskLayoutTla, Arch::PositionGM{});
                    uint32_t triUp = noSkipKvS - rowNum;
                    uint32_t triDown = noSkipKvS;
                    uint32_t kvSStartIdx = kvSTileIdx * kvBaseTile_;
                    uint32_t kvSEndIdx = kvSStartIdx + kvSTileSizeAct;
                    bool doTriUMask = triUp < kvSEndIdx - 1;
                    if (doTriUMask) {
                        epilogueOnlineSoftmax(
                            l1PTensorTla,
                            gmMaskTensorTla,
                            actualBlockShapeQK,
                            (kvSTileIdx == 0),
                            ubSBufId,
                            l1PBufId,
                            qkReadyFlag,
                            softmaxReadyFlag,
                            triUp,
                            triDown,
                            0, 0,
                            kvSStartIdx,
                            kvSEndIdx,
                            1);
                    } else {
                        epilogueOnlineSoftmax(
                            l1PTensorTla,
                            actualBlockShapeQK,
                            (kvSTileIdx == 0),
                            ubSBufId,
                            l1PBufId,
                            qkReadyFlag,
                            softmaxReadyFlag);
                    }
#endif
                }
                if (kvSTileIdx >= PRE_LAUNCH) {
                    uint32_t kvSTileIdxNow = kvSTileIdx - PRE_LAUNCH;
                    if (kvSTileIdxNow == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdxNow * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapePV{rowNum, embedV_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvSTileIdxNow % UB_S_OTMP_BUF_STAGES;
                    uint32_t pvReadyFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvSTileIdxNow % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedVRound);
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor[ubOTmpBufId],
                        ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint64_t prefixSumL0AStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdxNow, qkL0ATotalStages_, pvL0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdxNow, qkL0BTotalStages_, pvL0BTotalStages_, kvSLoopNum, false);
                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gBlockTable[blockBOffset],
                        actualBlockShapePV,
                        blockSize_,
                        kvSTileIdxNow, 0, kvHeads_,
                        kvNumTokens,
                        kvBaseTile_, 0, 0, 0,
                        softmaxReadyFlag, pvReadyFlag,
                        prefixSumL0AStages, 
                        prefixSumL0BStages);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint32_t curTileMod = kvSTileIdxNow % (PRE_LAUNCH + 1);
                    if constexpr (enableDN) {
                        epilogueRescaleO(
                            gmOTensorTla, actualBlockShapePV,
                            curTileMod, kvSTileIdxNow,
                            (kvSTileIdxNow == 0),
                            (kvSTileIdxNow == kvSLoopNum - 1),
                            pvReadyFlag, 1);
                    } else {
                        epilogueRescaleO(
                            gmOTensorTla, actualBlockShapePV,
                            curTileMod, kvSTileIdxNow,
                            (kvSTileIdxNow == 0),
                            (kvSTileIdxNow == kvSLoopNum - 1),
                            pvReadyFlag, 0);
                    }
#endif
                }
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // cross core sync
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // mask
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(1);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline
    uint64_t CalcCrossqkpvPrefixSumL0ABStages(
        uint32_t KvSTileIdx, uint32_t singleqkL0Stages,
        uint32_t singlepvL0Stages, uint32_t kvSLoopNum,
        bool isCurPhaseqk)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseqk) {
            prefixSumStages = (KvSTileIdx <= PRE_LAUNCH) ?
                KvSTileIdx * singleqkL0Stages :
                KvSTileIdx * singleqkL0Stages + (KvSTileIdx - PRE_LAUNCH) * singlepvL0Stages;
        } else {
            prefixSumStages = (KvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                (KvSTileIdx + 1 + PRE_LAUNCH) * singleqkL0Stages + KvSTileIdx * singlepvL0Stages:
                kvSLoopNum * singleqkL0Stages + KvSTileIdx * singlepvL0Stages;
        }
        return prefixSumStages;
    }

private:
    Arch::Resource<ArchTag> resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t embedV_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t maxNumBlocksPerBatch_;
    uint32_t blockSize_;
    uint32_t numBlocks_;

    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint32_t actSeqAval_;

    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;

    uint32_t qkL1TileM_;
    uint32_t qkL1TileN_;
    uint32_t qkL1TileKLeft_;
    uint32_t qkL1TileKRight_;
    uint32_t pvL1TileM_;
    uint32_t pvL1TileN_;
    uint32_t pvL1TileKLeft_;
    uint32_t pvL1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;

    uint32_t qkL0ATotalStages_;
    uint32_t qkL0BTotalStages_;
    uint32_t pvL0ATotalStages_;
    uint32_t pvL0BTotalStages_;
};

template <class InDtype, class SMDtype, 
        Format qFormat, Format kvFormat, 
        CacheMode kvcacheType, PageShape kvcacheShape, 
        MaskCategory maskCategory, CacheLayout cacheLayout>
CATLASS_GLOBAL void FAInfer(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<Q_BLK>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyQK = std::conditional_t<std::is_same_v<ElementS, half>, 
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>,
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;

   // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmax;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using FAIKernel950 = FAIKernel950<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout, false>;
    PFAKernelParams params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAIKernel950 faInfer;
    faInfer(params);
}

template <class InDtype, class SMDtype, 
        Format qFormat, Format kvFormat, 
        CacheMode kvcacheType, PageShape kvcacheShape, 
        MaskCategory maskCategory, CacheLayout cacheLayout>
CATLASS_GLOBAL void FAInferDn(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutK = layout::RowMajor;
    using LayoutQ = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<Q_BLK>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyQK = std::conditional_t<std::is_same_v<ElementS, half>, 
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>,
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ, ElementS, void, TileCopyQK>;

   // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmax;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using FAIKernel950 = FAIKernel950<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout, true>;
    PFAKernelParams params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAIKernel950 faInfer;
    faInfer(params);
}

void PFAmpl(const uint32_t blockNum, aclrtStream stream, const FlashAttentionChunkPrefillParams &params)
{   
    // Get the number of cube cores of the current hardware
    auto aicCoreNum = blockNum;
    // Parameters initialization.
    int32_t batch = params.batch;
    int32_t qSeqlen = params.qSeqlen;
    int32_t kvSeqlen = params.kvSeqlen;
    int32_t numHeads = params.numHeads;
    int32_t kvHeads = params.kvHeads;
    int32_t qkHeadSize = params.qkembeddingSize;
    int32_t vHeadSize = params.vembeddingSize;
    int32_t blockSize = params.blockSize;
    int32_t numBlocks = params.numBlocks;
    int32_t maskType = 1;
    int32_t cacheMode = 1;
    int32_t pageShape = 1;
    int32_t layout = 1;
    int32_t innerPrec = 0;
    int32_t lseFlag = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    string cacheLayout = params.cacheLayout;
    aclDataType dataType = params.dataType;
    // std::string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;

    // // read qNtokens num
    // void *qNtokens = nullptr;
    // ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    // ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = params.qNtokens;

    // void *kvNtokens = nullptr;
    // ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    // ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
    // int32_t kvNumTokens = static_cast<int32_t *>(kvNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t seqArraySizeTND = (batch + 1) * sizeof(int64_t);
    uint64_t qSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)qkHeadSize * sizeof(half); 
    uint64_t oSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)vHeadSize * sizeof(half); 
    uint64_t lseSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(int32_t);
    uint64_t kSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)qkHeadSize * sizeof(half);
    uint64_t vSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)vHeadSize * sizeof(half);
    uint64_t maskSize = 2048 * 2048 * sizeof(int8_t);
    uint64_t blockTableSize = static_cast<uint64_t>(
        batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    uint32_t tilingSize = sizeof(FAInferTilingData);

    // Allocate matrices in host and device memory.
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice = params.inputAddr.at(0);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&qSeqHost), seqArraySizeTND));
    ACL_CHECK(aclrtMemcpy(qSeqHost, seqArraySizeTND, qSeqDevice, seqArraySizeTND, ACL_MEMCPY_DEVICE_TO_HOST));
    // Allocate matrices in host and device memory.
    uint8_t *kvSeqHost;
    uint8_t *kvSeqDevice = params.inputAddr.at(1);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&kvSeqHost), seqArraySize));
    ACL_CHECK(aclrtMemcpy(kvSeqHost, seqArraySize, kvSeqDevice, seqArraySize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice = params.inputAddr.at(2);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&qHost), qSize));
    ACL_CHECK(aclrtMemcpy(qHost, qSize, qDevice, qSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice = params.inputAddr.at(3);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&kHost), kSize));
    ACL_CHECK(aclrtMemcpy(kHost, kSize, kDevice, kSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHost;
    uint8_t *vDevice = params.inputAddr.at(4);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&vHost), vSize));
    ACL_CHECK(aclrtMemcpy(vHost, vSize, vDevice, vSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Allocate matrices in host and device memory and load Matrix mask.
    uint8_t *maskHost;
    uint8_t *maskDevice = params.inputAddr.at(5);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&maskHost), maskSize));
    ACL_CHECK(aclrtMemcpy(maskHost, maskSize, maskDevice, maskSize, ACL_MEMCPY_DEVICE_TO_HOST));
    
    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t *blockTableHost;
    uint8_t *blockTableDevice = params.inputAddr.at(6);
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&blockTableHost), blockTableSize));
    ACL_CHECK(aclrtMemcpy(blockTableHost, blockTableSize, blockTableDevice, blockTableSize, ACL_MEMCPY_DEVICE_TO_HOST));
    
    bool enableDN = false;
    // 当前逻辑只支持half
    if (maskType == 0 && qkHeadSize <= 128 && vHeadSize <= 128 && innerPrec == 0) {
        enableDN = true;
    }

    FAInferTilingData faiTilingData;
    FAInferContext faiContext;


    faiContext.pagedCacheFlag = (cacheMode == 1);
    faiContext.pagedShapeFlag = (pageShape == 1);
    faiContext.kvcacheNzFlag = (cacheLayout == "nz");
    faiContext.numHeads = numHeads;
    faiContext.numBlocks = numBlocks;
    faiContext.blockSize = blockSize;
    faiContext.kvHeads = kvHeads;
    faiContext.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * qkHeadSize));
    faiContext.layout = "TND";
    faiContext.lseFlag = lseFlag;
    if (faiContext.pagedCacheFlag) {
        faiContext.maxNumBlocksPerBatch = (maxKvSeqlen + blockSize - 1) / blockSize;
    }
    faiContext.embeddingSize = qkHeadSize;
    faiContext.embeddingSizeV = vHeadSize;
    
    faiContext.maskType = static_cast<optiling::MaskType>(maskType);
    faiContext.dataType = static_cast<optiling::DataType>(dataType == ACL_BF16);
    faiContext.batch = batch;
    faiContext.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faiContext.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);
    faiContext.preToken = preToken;
    faiContext.nextToken = nextToken;

    // flashDecodeFlag determination
    int64_t maxQSeqlenCalc = 0;
    int64_t minQSeqlenCalc = INT64_MAX;
    int64_t minKVSeqlenCalc = INT64_MAX;
    for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int64_t qSeqlenVal = *(faiContext.qSeqlenList + batchIdx);
        int64_t kvSeqlenVal = *(faiContext.kvSeqlenList + batchIdx);
        if (faiContext.layout == "TND") {
            if (batchIdx > 0) {
                int64_t prevQSeqlenSum = *(faiContext.qSeqlenList + batchIdx - 1);
                qSeqlenVal = qSeqlenVal - prevQSeqlenSum;
            }
        }
        if (qSeqlenVal > maxQSeqlenCalc) {
            maxQSeqlenCalc = qSeqlenVal;
        }
        if (qSeqlenVal < minQSeqlenCalc) {
            minQSeqlenCalc = qSeqlenVal;
        }
        if (kvSeqlenVal < minKVSeqlenCalc) {
            minKVSeqlenCalc = kvSeqlenVal;
        }
    }
    faiContext.maxQSeqlen = maxQSeqlenCalc;
    uint32_t numTasks = faiContext.batch * faiContext.kvHeads;
    bool isLongSeq = (numTasks <= 0.8 * aicCoreNum) && (minKVSeqlenCalc >= aicCoreNum * 512);
    bool isShortSeq = (numTasks <= 0.4 * aicCoreNum) && (minKVSeqlenCalc >= 1024);
    faiContext.flashDecodeFlag = false;

    FAInferTiling fai_tiling(faiContext);
    fai_tiling.SetCoreNum(aicCoreNum);
    fai_tiling.DoTiling(faiTilingData);
    uint64_t tilingKey = fai_tiling.GetTilingKey();

    uint8_t *workspaceDevice{nullptr};
    faiTilingData.workSpaceSize = 1024 * 1024 * 32 * 4;
    ACL_CHECK(aclrtMalloc((void**)(&workspaceDevice), faiTilingData.workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *oDevice = params.outputAddr.at(0);
    uint8_t *lseDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void**)(&lseDevice), lseSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t blockDim = aicCoreNum;
    if (faiContext.flashDecodeFlag) {
        auto needCoreNum = faiTilingData.get_needCoreNum();
        if (needCoreNum != 0) {
            blockDim = needCoreNum;
        }
    }

    // tiling output
    uint8_t* tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &faiTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    if (cacheLayout == "nd") {
        if (dataType == ACL_FLOAT16) {
            FAInfer<half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd><<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (dataType == ACL_BF16) {
            FAInfer<bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd><<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
    } else { // cacheLayout == "nz"
        if (dataType == ACL_FLOAT16) {
            FAInfer<half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz><<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (dataType == ACL_BF16) {
            FAInfer<bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz><<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclrtFree(tilingDevice);
    aclrtFree(lseDevice);
    aclrtFree(workspaceDevice);
    aclrtFreeHost(qSeqHost);
    aclrtFreeHost(kvSeqHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(maskHost);
    aclrtFreeHost(blockTableHost);
    // // Destroy specified Stream and reset device.
    // ACL_CHECK(aclrtDestroyStream(stream));
    // ACL_CHECK(aclFinalize());
}

void FlashAttentionChunkPrefill(uint32_t blockNum, aclrtStream stream, const FlashAttentionChunkPrefillParams& params)
{
    PFAmpl(blockNum, stream, params);
}
} // namespace CatlassKernel