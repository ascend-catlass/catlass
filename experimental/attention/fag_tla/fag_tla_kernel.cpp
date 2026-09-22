/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * TLA-style (Tiled Linear Attention) FAG (Flash Attention Gradient) backward pass kernel.
 *
 * TLA-style changes:
 *   - Includes tla/layout.hpp and tla/tensor.hpp for TLA tensor/layout APIs
 *   - Uses tla::MakeLayout, tla::MakeTensor, tla::MakeShape, tla::MakeStride
 *   - Uses GemmShape<> instead of GemmShape<> for tile shapes
 *   - Uses BlockMmadTla with PackedTileCopyTla for matrix multiply operations
 *   - Passes tla::Tensor objects to BlockMmadTla operator() instead of raw GlobalTensor + Layout
 *   - Algorithm structure and control flow preserved from original FAG implementation
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "kernel_common_fag.hpp"

using namespace Catlass;
using namespace AscendC;

// ============================================================
// TLA-style FAG Kernel: FlashAttentionScoreGrad
//
// Each matmul block uses the FAG-specific BlockMmadFagSdp / BlockMmadFAG
// (the only specializations matching the MmadAtlasA2FagSdp / MmadAtlasA2FagdQKV
// dispatch policies). TLA-style tensor/layout APIs (tla::MakeTensor,
// tla::MakeLayout, tla::MakeShape, tla::MakeStride) are used for constructing
// tensor views; the underlying GlobalTensor + catlass Layout are passed to
// the block operator() calls.
// ============================================================

constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
constexpr static int64_t GM_DOUBLE_BUFFER = 2;
constexpr static uint32_t DTYPE_FACTOR = 2;
constexpr static uint32_t S_SPLITT_SIZE = 256;
constexpr static uint32_t MMAD_BASE_SIZE = 128;
constexpr static int64_t C0_SIZE = 16;

struct IndexParams {
    int64_t bIdx;
    int64_t n2Idx;
    int64_t s2oIdx;
    int64_t gIdx;
    int64_t s1oIdx;
};

enum class DTemplateType
{
    NonAligned = 0,
    Aligned64 = 1,
    Aligned128 = 2,
    Aligned192 = 3,
    Aligned256 = 4,
    DTemplateBottom
};

template <
    class BlockMmad_, class BlockMmad2_, class BlockMmad3_, class EpilogueFAGPre_, class EpilogueFAGSfmg_,
    class EpilogueFAGSabVec_, class EpilogueFAGPost_, class EpilogueFAGDtmAdd_, const uint32_t INPUT_LAYOUT,
    const uint32_t IS_DTM>
class FlashAttentionScoreGrad {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockMmad2 = BlockMmad2_;
    using LayoutA2 = typename BlockMmad2::LayoutA;
    using LayoutB2 = typename BlockMmad2::LayoutB;
    using LayoutC2 = typename BlockMmad2::LayoutC;

    using BlockMmad3 = BlockMmad3_;
    using LayoutA3 = typename BlockMmad3::LayoutA;
    using LayoutB3 = typename BlockMmad3::LayoutB;
    using LayoutC3 = typename BlockMmad3::LayoutC;

    using EpilogueFAGPre = EpilogueFAGPre_;
    using EpilogueFAGSfmg = EpilogueFAGSfmg_;
    using EpilogueFAGSabVec = EpilogueFAGSabVec_;
    using EpilogueFAGPost = EpilogueFAGPost_;
    using EpilogueFAGDtmAdd = EpilogueFAGDtmAdd_;

    AscendC::GlobalTensor<ElementA> queryGm;
    AscendC::GlobalTensor<ElementA> keyGm;
    AscendC::GlobalTensor<ElementA> valueGm;
    AscendC::GlobalTensor<ElementA> dxGm;
    AscendC::GlobalTensor<ElementA> forwardResGm;

    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, sfmgWorkspaceGm, dqDtmWsGm, dkDtmWsGm,
        dvDtmWsGm;
    AscendC::GlobalTensor<float> mm1WorkspaceGm, mm2WorkspaceGm;
    AscendC::GlobalTensor<ElementA> dropWorkSpaceGm, mulWorkSpaceGm;

    __gm__ uint8_t* actual_seq_qlen_addr;
    __gm__ uint8_t* actual_seq_kvlen_addr;

    constexpr static uint32_t ENABLE = 1;
    constexpr static int8_t OUTIDX = -1;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    __aicore__ inline void Init(FAGKernelParams const& params, __gm__ FAGTilingData* fagTilingData)
    {
        queryGm.SetGlobalBuffer((__gm__ ElementA*)params.q);
        keyGm.SetGlobalBuffer((__gm__ ElementA*)params.k);
        valueGm.SetGlobalBuffer((__gm__ ElementA*)params.v);
        dxGm.SetGlobalBuffer((__gm__ ElementA*)params.dout);
        forwardResGm.SetGlobalBuffer((__gm__ ElementA*)params.out);

        coreNum = fagTilingData->coreNum;
        cubeCoreNum = coreNum / 2;
        vecBlockNum = coreNum / 3;

        b = fagTilingData->batch;
        n2 = fagTilingData->kvHeadNum;
        g = fagTilingData->g;
        s1 = fagTilingData->qSeqlen;
        s2 = fagTilingData->kvSeqlen;
        d = fagTilingData->qkHeadDim;
        value_d = fagTilingData->vHeadDim;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;
        attenMaskDimS2 = fagTilingData->attenMaskS2Size;

        s1Token = fagTilingData->s1Token;
        s2Token = fagTilingData->s2Token;
        actualCalcS1Token = s1Token;
        actualCalcS2Token = s2Token;
        sparseMode = fagTilingData->sparseMode;

        s1Outer = fagTilingData->s1Outer;
        s1CvInner = fagTilingData->s1CvInner;
        s1CvTail = fagTilingData->s1CvTail;
        s2Outer = fagTilingData->s2Outer;
        s2CvInner = fagTilingData->s2CvInner;
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

        baseMN = fagTilingData->baseMN;
        cubeBaseMN = s1CvInner * s2CvInner;

        actual_seq_qlen_addr = params.cu_seq_qlen;
        actual_seq_kvlen_addr = params.cu_seq_kvlen;

        dropBitMode = s2 % 8 == 0;
        keepProb = fagTilingData->keepProb;
        int64_t sfmgOutputSize = b * n2 * g * s1 * 8;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t seqS2Len = 0;
            seqS2Len = ((__gm__ int64_t*)params.cu_seq_kvlen)[0];
            dropBitMode = (seqS2Len % 8 == 0);
            for (int64_t i = 0; i + 1 < b; i++) {
                seqS2Len = ((__gm__ int64_t*)params.cu_seq_kvlen)[i + 1] - ((__gm__ int64_t*)params.cu_seq_kvlen)[i];
                dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
            }
            sfmgOutputSize = ((__gm__ int64_t*)params.cu_seq_qlen)[b - 1] * n2 * g * 8;
        }

        dqWorkSpaceGm.SetGlobalBuffer(
            (__gm__ float*)params.workspace + fagTilingData->dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer(
            (__gm__ float*)params.workspace + fagTilingData->dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer(
            (__gm__ float*)params.workspace + fagTilingData->dvWorkSpaceOffset / sizeof(float));

        sfmgWorkspaceGm.SetGlobalBuffer(
            (__gm__ ElementC*)params.workspace + fagTilingData->sfmgPreBeginAddr / sizeof(ElementC));
        int64_t workspaceOffsets =
            (fagTilingData->sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
            ADDR_ALIGN_SIZE;

        uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets +
                                                       cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets +
                                                       cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                       cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        dropWorkSpaceGm.SetGlobalBuffer((__gm__ ElementA*)(params.workspace + workspaceOffsets +
                                                           cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                           cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        mulWorkSpaceGm.SetGlobalBuffer((__gm__ ElementA*)(params.workspace + workspaceOffsets +
                                                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        if constexpr (IS_DTM == ENABLE) {
            uint64_t pseAlibiAddr =
                (workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * INPUT_NUMS * GM_DOUBLE_BUFFER +
                 ADDR_ALIGN_SIZE) /
                ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
            workspaceOffsets = (pseAlibiAddr + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

            dqDtmWsGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s1CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) /
                               ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

            dkDtmWsGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s2CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) /
                               ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

            dvDtmWsGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets));
        }
    }

    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            actualSeqQlen = ((__gm__ int64_t*)actual_seq_qlen_addr)[0];
            actualSeqKvlen = ((__gm__ int64_t*)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen =
                ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen =
                ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    __aicore__ inline void UpdateIndex()
    {
        s1oDimIdx = 0;
        if constexpr (IS_DTM == ENABLE) {
            if (gDimIdx < g - 1) {
                gDimIdx++;
            } else if (s2oCvDimIdx < s2Outer - 1) {
                gDimIdx = 0;
                s2oCvDimIdx++;
            } else if (n2DimIdx < n2 - 1) {
                gDimIdx = 0;
                s2oCvDimIdx = 0;
                n2DimIdx++;
            } else {
                gDimIdx = 0;
                s2oCvDimIdx = 0;
                n2DimIdx = 0;
                bDimIdx++;
                dqOutBase += n2 * g * s1Outer;
                kvOutBase += n2 * s2Outer;
            }
        } else {
            if (s2oCvDimIdx < s2Outer - 1) {
                s2oCvDimIdx++;
            } else if (gDimIdx < g - 1) {
                s2oCvDimIdx = 0;
                gDimIdx++;
            } else if (n2DimIdx < n2 - 1) {
                gDimIdx = 0;
                s2oCvDimIdx = 0;
                n2DimIdx++;
            } else {
                gDimIdx = 0;
                s2oCvDimIdx = 0;
                n2DimIdx = 0;
                bDimIdx++;
                dqOutBase += n2 * g * s1Outer;
                kvOutBase += n2 * s2Outer;
            }
        }
    }

    __aicore__ inline bool CalcValidBlock(int64_t& baseIdx, int64_t& startCoreId, DBParams& dbParam)
    {
        if (bDimIdx >= b) {
            if (cCubeBlockIdx >= startCoreId) {
                dbParam.blockId = -1;
            }
            dbParam.blockIdArr[startCoreId] = -1;
            baseIdx = -1;
            return true;
        }
        int64_t actualSeqQlen = s1;
        int64_t actualSeqKvlen = s2;
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdx, actualSeqQlen, actualSeqKvlen);
            s1Outer = (actualSeqQlen + s1CvInner - 1) / s1CvInner;
            s2Outer = (actualSeqKvlen + s2CvInner - 1) / s2CvInner;
            s1CvTail = actualSeqQlen - (s1Outer - 1) * s1CvInner;
            s2CvTail = actualSeqKvlen - (s2Outer - 1) * s2CvInner;
        }
        dqOutIdx = dqOutBase + (n2DimIdx * g + gDimIdx) * s1Outer + s1oDimIdx;
        kvOutIdx = kvOutBase + n2DimIdx * s2Outer + s2oCvDimIdx;

        int64_t s1IdxUp = s2oCvDimIdx * s2CvInner - actualCalcS2Token;

        if (s1IdxUp >= actualSeqQlen || actualSeqKvlen == 0) {
            baseIdx += actualSeqKvlen ? (s1Outer - s1oDimIdx) : 0;
            UpdateIndex();
            return false;
        }

        int64_t s1oIdxUp = s1IdxUp / s1CvInner;
        s1oIdxUp = s1oIdxUp > 0 ? s1oIdxUp : 0;

        int64_t s1IdxDown = (s2oCvDimIdx + 1) * s2CvInner + actualCalcS1Token;
        s1IdxDown = s1IdxDown > actualSeqQlen ? actualSeqQlen : s1IdxDown;
        int64_t s1oIdxDown = (s1IdxDown + s1CvInner - 1) / s1CvInner - 1;

        if (s1oIdxDown < s1oIdxUp || s1oDimIdx > s1oIdxDown) {
            baseIdx += (s1Outer - s1oDimIdx);
            UpdateIndex();
            return false;
        }
        if (s1oDimIdx < s1oIdxUp) {
            baseIdx += s1oIdxUp - s1oDimIdx;
            dqOutIdx += s1oIdxUp - s1oDimIdx;
            s1oDimIdx = s1oIdxUp;
        }
        int64_t validNum = s1oIdxDown - s1oDimIdx + 1;
        if (cCubeBlockIdx >= startCoreId && cCubeBlockIdx - startCoreId < validNum) {
            dbParam.blockId = baseIdx + cCubeBlockIdx - startCoreId;
            dbParam.bIdx = bDimIdx;
            dbParam.n2Idx = n2DimIdx;
            dbParam.s2oIdx = s2oCvDimIdx;
            dbParam.gIdx = gDimIdx;
            dbParam.s1oIdx = s1oDimIdx + cCubeBlockIdx - startCoreId;
            dbParam.s1CvExtend = (dbParam.s1oIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
            dbParam.s2CvExtend = (dbParam.s2oIdx == s2Outer - 1) ? s2CvTail : s2CvInner;
            int64_t s2RightIdx = dbParam.s1oIdx * s1CvInner + dbParam.s1CvExtend + actualCalcS2Token;
            s2RightIdx = s2RightIdx > 0 ? s2RightIdx : 0;
            s2RightIdx = (s2RightIdx + 7) / 8 * 8;
            dbParam.s2CvExtend = s2RightIdx > (dbParam.s2oIdx * s2CvInner + dbParam.s2CvExtend) ?
                                     dbParam.s2CvExtend :
                                     s2RightIdx - dbParam.s2oIdx * s2CvInner;
            dbParam.s2CvExtend = dbParam.s2CvExtend > 0 ? dbParam.s2CvExtend : 0;
            dbParam.s1CvExtendAlign = (dbParam.s1CvExtend + 15) / 16 * 16;
            dbParam.s2CvExtendAlign = (dbParam.s2CvExtend + 15) / 16 * 16;
        }

        if constexpr (IS_DTM == ENABLE) {
            int8_t kvGroupId = startCoreId;
            if (startCoreId > 0 && kvOutIdx == kvOutArr[startCoreId - 1] &&
                dbParam.kvGroupId[startCoreId - 1] != OUTIDX) {
                kvGroupId = dbParam.kvGroupId[startCoreId - 1];
            }
            uint32_t s2CvExtend = (s2oCvDimIdx == s2Outer - 1) ? s2CvTail : s2CvInner;

            for (int32_t i = startCoreId; i < cubeCoreNum; i++) {
                if (i - startCoreId >= validNum) {
                    break;
                }
                dqOutArr[i] = dqOutIdx + i - startCoreId;
                kvOutArr[i] = kvOutIdx;
                dbParam.blockIdArr[i] = baseIdx + i - startCoreId;
                dbParam.s1CvExtendArr[i] = (s1oDimIdx + i - startCoreId) == (s1Outer - 1) ? s1CvTail : s1CvInner;
                int64_t s2RightIdx =
                    (s1oDimIdx + i - startCoreId) * s1CvInner + dbParam.s1CvExtendArr[i] + actualCalcS2Token;
                s2RightIdx = s2RightIdx > 0 ? s2RightIdx : 0;
                s2RightIdx = (s2RightIdx + 7) / 8 * 8;
                dbParam.s2CvExtendArr[i] = s2RightIdx > (s2oCvDimIdx * s2CvInner + s2CvExtend) ?
                                               s2CvExtend :
                                               (s2RightIdx - s2oCvDimIdx * s2CvInner);
                dbParam.kvGroupId[i] = kvGroupId;
                dbParam.dqGroupId[i] = i;
                if (s2oCvDimIdx == 0) {
                    dbParam.dqGroupId[i] = OUTIDX;
                    continue;
                }

                for (int32_t j = 0; j < startCoreId; j++) {
                    if (dqOutArr[i] == dqOutArr[j]) {
                        dbParam.dqGroupId[i] = j;
                        break;
                    }
                }
            }
            if (s1oDimIdx == s1oIdxUp && gDimIdx == 0) {
                dbParam.kvGroupId[startCoreId] = OUTIDX;
            }
        }

        if (cubeCoreNum - startCoreId <= validNum) {
            baseIdx = baseIdx + cubeCoreNum - startCoreId;
            if ((s1oDimIdx + cubeCoreNum - startCoreId) < s1Outer) {
                s1oDimIdx = s1oDimIdx + cubeCoreNum - startCoreId;
            } else {
                UpdateIndex();
            }
            return true;
        } else {
            baseIdx += (s1Outer - s1oDimIdx);
            startCoreId += validNum;
            UpdateIndex();
            return false;
        }
    }

    // TLA-style: use tla::MakeLayout to create layouts, make tla::Tensor views for block mmad
    __aicore__ inline void ComputeMM1(DBParams& dbParam)
    {
        pingpongIdx = dbParam.taskId % 2;
        dbParam.actualS1Len = s1;
        dbParam.actualS2Len = s2;
        dbParam.s1Stride = 0;
        dbParam.s2Stride = 0;

        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(dbParam.bIdx, dbParam.actualS1Len, dbParam.actualS2Len);
            dbParam.aTensorOffsetCv = 0;
            dbParam.bTensorOffsetCv = 0;
            if (dbParam.bIdx > 0) {
                dbParam.aTensorOffsetCv = ((__gm__ int64_t*)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * d;
                dbParam.bTensorOffsetCv = ((__gm__ int64_t*)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * d;
            }
            dbParam.aTensorOffsetCv += ((dbParam.s1oIdx * s1CvInner * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
            dbParam.bTensorOffsetCv += (dbParam.s2oIdx * s2CvInner * n2 + dbParam.n2Idx) * d;
            dbParam.s1Stride = n2 * g * d;
            dbParam.s2Stride = n2 * d;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            dbParam.aTensorOffsetCv =
                (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * d;
            dbParam.bTensorOffsetCv = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2 + dbParam.s2oIdx * s2CvInner) * d;
            dbParam.s1Stride = d;
            dbParam.s2Stride = d;
        } else if constexpr (INPUT_LAYOUT == SBH) {
            dbParam.aTensorOffsetCv =
                ((((dbParam.s1oIdx * s1CvInner) * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
            dbParam.bTensorOffsetCv = ((dbParam.s2oIdx * s2CvInner * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * d;
            dbParam.s1Stride = b * n2 * g * d;
            dbParam.s2Stride = b * n2 * d;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            dbParam.aTensorOffsetCv =
                (((dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
            dbParam.bTensorOffsetCv = ((dbParam.bIdx * s2 + dbParam.s2oIdx * s2CvInner) * n2 + dbParam.n2Idx) * d;
            dbParam.s1Stride = n2 * g * d;
            dbParam.s2Stride = n2 * d;
        }

        int64_t specify_for_v_aTensorOffsetCv = dbParam.aTensorOffsetCv / d * value_d;
        int64_t specify_for_v_bTensorOffsetCv = dbParam.bTensorOffsetCv / d * value_d;
        int64_t specify_for_v_s1Stride = dbParam.s1Stride / d * value_d;
        int64_t specify_for_v_s2Stride = dbParam.s2Stride / d * value_d;

        // Use the original FAG block mmad with catlass Layout constructors.
        // TLA-style Shape types are used in the TypeSelector for tile shape definitions.
        BlockMmad blockMmad(resource);
        AscendC::PipeBarrier<PIPE_ALL>();

        // mm-dyv: dOut * V^T
        GemmCoord actualBlockShape1(dbParam.s1CvExtend, dbParam.s2CvExtend, value_d);
        {
            LayoutA layoutA1{actualBlockShape1.m(), actualBlockShape1.k(), specify_for_v_s1Stride};
            LayoutB layoutB1{actualBlockShape1.k(), actualBlockShape1.n(), specify_for_v_s2Stride};
            LayoutC layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dbParam.s2CvExtendAlign};
            blockMmad(
                dxGm[specify_for_v_aTensorOffsetCv], layoutA1, valueGm[specify_for_v_bTensorOffsetCv], layoutB1,
                mm1WorkspaceGm[pingpongIdx * cubeBaseMN], layoutC1, actualBlockShape1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // mm-qk: Q * K^T
        GemmCoord actualBlockShape(dbParam.s1CvExtend, dbParam.s2CvExtend, d);
        {
            LayoutA layoutA{actualBlockShape.m(), actualBlockShape.k(), dbParam.s1Stride};
            LayoutB layoutB{actualBlockShape.k(), actualBlockShape.n(), dbParam.s2Stride};
            LayoutC layoutC{actualBlockShape.m(), actualBlockShape.n(), dbParam.s2CvExtendAlign};
            blockMmad(
                queryGm[dbParam.aTensorOffsetCv], layoutA, keyGm[dbParam.bTensorOffsetCv], layoutB,
                mm2WorkspaceGm[pingpongIdx * cubeBaseMN], layoutC, actualBlockShape);
        }
    }

    // TLA-style: dQ = dS * K, dK = dS^T * Q, dV = P^T * dOut
    __aicore__ inline void ComputeMMDqkv(DBParams& dbParam, int64_t nextBlockId)
    {
        BlockMmad2 blockMmad2(resource);
        BlockMmad3 blockMmad3(resource, 0, 4);

        pingpongIdx = dbParam.taskId % 2;
        int64_t dqOffset = 0;
        int64_t dkvOffset = 0;
        int64_t dvOffset = 0;
        dqOffset = dbParam.aTensorOffsetCv;
        dkvOffset = dbParam.bTensorOffsetCv;
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;

        int64_t dqKc = dbParam.s1Stride;
        int64_t dkvKc = dbParam.s2Stride;
        int64_t dvKc = dbParam.s2Stride / d * value_d;
        int64_t s1_size = dbParam.actualS1Len;
        int64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;

        // dQ = dS * K: RowMajor * RowMajor
        GemmCoord actualBlockShape1(dbParam.s1CvExtend, d, dbParam.s2CvExtend);
        {
            LayoutA2 layoutA1{actualBlockShape1.m(), actualBlockShape1.k(), s2_size};
            LayoutB2 layoutB1{actualBlockShape1.k(), actualBlockShape1.n(), dbParam.s2Stride};
            LayoutC2 layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dqKc};
            blockMmad2(
                mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], keyGm[dbParam.bTensorOffsetCv], dqWorkSpaceGm[dqOffset],
                layoutA1, layoutB1, layoutC1, actualBlockShape1, true);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // dK = dS^T * Q: ColumnMajor * RowMajor
        GemmCoord actualBlockShape2(dbParam.s2CvExtend, d, dbParam.s1CvExtend);
        {
            LayoutA3 layoutA2{actualBlockShape2.m(), actualBlockShape2.k(), s2_size};
            LayoutB3 layoutB2{actualBlockShape2.k(), actualBlockShape2.n(), dbParam.s1Stride};
            LayoutC3 layoutC2{actualBlockShape2.m(), actualBlockShape2.n(), dkvKc};
            blockMmad3(
                mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], queryGm[dbParam.aTensorOffsetCv],
                dkWorkSpaceGm[dkvOffset], layoutA2, layoutB2, layoutC2, actualBlockShape2, true);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // dV = P^T * dOut: ColumnMajor * RowMajor
        GemmCoord actualBlockShape3(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
        {
            LayoutA3 layoutA3{actualBlockShape3.m(), actualBlockShape3.k(), s2_size};
            LayoutB3 layoutB3{actualBlockShape3.k(), actualBlockShape3.n(), dbParam.s1Stride / d * value_d};
            LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
            blockMmad3(
                dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], dxGm[dbParam.aTensorOffsetCv / d * value_d],
                dvWorkSpaceGm[dvOffset], layoutA3, layoutB3, layoutC3, actualBlockShape3, true);
        }
    }

    // TLA-style: deterministic DQKV computation
    __aicore__ inline void DTMComputeMMDqkv(DBParams& dbParam, int64_t nextBlockId)
    {
        BlockMmad2 blockMmad2(resource);
        BlockMmad3 blockMmad3(resource, 0, 4);
        pingpongIdx = dbParam.taskId % 2;
        int64_t dqOffset = 0;
        int64_t dkvOffset = 0;
        int64_t dvOffset = 0;
        dqOffset = dbParam.aTensorOffsetCv;
        dkvOffset = dbParam.bTensorOffsetCv;
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;

        int64_t dqKc = dbParam.s1Stride;
        int64_t dkvKc = dbParam.s2Stride;
        int64_t dvKc = dbParam.s2Stride / d * value_d;
        int64_t s1_size = dbParam.actualS1Len;
        int64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;

        if (dbParam.dqGroupId[cCubeBlockIdx] != OUTIDX) {
            dqOffset = pingpongIdx * cubeCoreNum * s1CvInner * dAlign + cCubeBlockIdx * s1CvInner * dAlign;
            dqKc = d;
        }
        if (dbParam.kvGroupId[cCubeBlockIdx] != OUTIDX) {
            dkvOffset = pingpongIdx * cubeCoreNum * s2CvInner * dAlign + cCubeBlockIdx * s2CvInner * dAlign;
            dvOffset = pingpongIdx * cubeCoreNum * s2CvInner * value_dAlign + cCubeBlockIdx * s2CvInner * value_dAlign;
            dkvKc = d;
            dvKc = value_d;
        }

        // dQ = dS * K (deterministic)
        GemmCoord actualBlockShape1(dbParam.s1CvExtend, d, dbParam.s2CvExtend);
        {
            LayoutA2 layoutA1{actualBlockShape1.m(), actualBlockShape1.k(), s2_size};
            LayoutB2 layoutB1{actualBlockShape1.k(), actualBlockShape1.n(), dbParam.s2Stride};
            if (dbParam.dqGroupId[cCubeBlockIdx] == OUTIDX) {
                LayoutC2 layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dqKc};
                blockMmad2(
                    mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], keyGm[dbParam.bTensorOffsetCv],
                    dqWorkSpaceGm[dqOffset], layoutA1, layoutB1, layoutC1, actualBlockShape1, false);
            } else {
                LayoutC2 layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dqKc};
                blockMmad2(
                    mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], keyGm[dbParam.bTensorOffsetCv], dqDtmWsGm[dqOffset],
                    layoutA1, layoutB1, layoutC1, actualBlockShape1, false);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // dK = dS^T * Q (deterministic)
        GemmCoord actualBlockShape2(dbParam.s2CvExtend, d, dbParam.s1CvExtend);
        {
            LayoutA3 layoutA2{actualBlockShape2.m(), actualBlockShape2.k(), s2_size};
            LayoutB3 layoutB2{actualBlockShape2.k(), actualBlockShape2.n(), dbParam.s1Stride};
            if (dbParam.kvGroupId[cCubeBlockIdx] == OUTIDX) {
                LayoutC3 layoutC2{actualBlockShape2.m(), actualBlockShape2.n(), dkvKc};
                blockMmad3(
                    mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], queryGm[dbParam.aTensorOffsetCv],
                    dkWorkSpaceGm[dkvOffset], layoutA2, layoutB2, layoutC2, actualBlockShape2, false);
            } else {
                LayoutC3 layoutC2{actualBlockShape2.m(), actualBlockShape2.n(), dkvKc};
                blockMmad3(
                    mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], queryGm[dbParam.aTensorOffsetCv],
                    dkDtmWsGm[dkvOffset], layoutA2, layoutB2, layoutC2, actualBlockShape2, false);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // dV = P^T * dOut (deterministic)
        GemmCoord actualBlockShape3(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
        {
            LayoutA3 layoutA3{actualBlockShape3.m(), actualBlockShape3.k(), s2_size};
            LayoutB3 layoutB3{actualBlockShape3.k(), actualBlockShape3.n(), dbParam.s1Stride / d * value_d};
            if (dbParam.kvGroupId[cCubeBlockIdx] == OUTIDX) {
                LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
                blockMmad3(
                    dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], dxGm[dbParam.aTensorOffsetCv / d * value_d],
                    dvWorkSpaceGm[dvOffset], layoutA3, layoutB3, layoutC3, actualBlockShape3, false);
            } else {
                LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
                blockMmad3(
                    dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], dxGm[dbParam.aTensorOffsetCv / d * value_d],
                    dvDtmWsGm[dvOffset], layoutA3, layoutB3, layoutC3, actualBlockShape3, false);
            }
        }
    }

    CATLASS_DEVICE
    FlashAttentionScoreGrad()
    {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGrad()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(FAGKernelParams const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(FAGKernelParams const& params)
    {
        __gm__ FAGTilingData* fagTilingData = reinterpret_cast<__gm__ FAGTilingData*>(params.tiling);
        cCubeBlockIdx = AscendC::GetBlockIdx();
        cBlockIdx = cCubeBlockIdx * 2;
        Init(params, fagTilingData);

        bool isFinish = false;
        int64_t startCoreId = 0;
        while (!isFinish) {
            isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[0]);
        }

        dbParams[0].taskId = 0;
        if (dbParams[0].blockId != -1) {
            ComputeMM1(dbParams[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[0].taskId]);
        }

        int64_t taskId = 1;
        int8_t extraLoopNum = 1;
        if constexpr (IS_DTM == ENABLE) {
            extraLoopNum = 2;
        }
        event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
        while (extraLoopNum >= 0) {
            isFinish = false;
            startCoreId = 0;
            while (!isFinish) {
                isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[taskId % 3]);
            }
            dbParams[taskId % 3].taskId = taskId;
            if (dbParams[taskId % 3].blockId != -1) {
                ComputeMM1(dbParams[taskId % 3]);
                AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[taskId % 3].taskId % 2]);
            }
            if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                if constexpr (IS_DTM != ENABLE) {
                    AscendC::CrossCoreWaitFlag(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    ComputeMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                    if (dbParams[(taskId) % 3].blockId == -1) {
                        AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(
                            SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    }
                }
            }

            if constexpr (IS_DTM == ENABLE) {
                if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    DTMComputeMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
                if (taskId > 1) {
                    SyncAll();
                }
            }

            taskId++;
            if (blockStartIdx == -1) {
                extraLoopNum -= 1;
            }
        }
        if constexpr (IS_DTM == ENABLE) {
            SyncAll();
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(FAGKernelParams const& params)
    {
        __gm__ FAGTilingData* fagTilingData = reinterpret_cast<__gm__ FAGTilingData*>(params.tiling);

        AscendC::TPipe pipePre;
        EpilogueFAGPre epilogueFagPre(
            resource, &pipePre, params.dq, params.dk, params.dv, params.workspace,
            {fagTilingData->dqWorkSpaceOffset, fagTilingData->dkWorkSpaceOffset, fagTilingData->dvWorkSpaceOffset,
             fagTilingData->qSize, fagTilingData->kvSize, fagTilingData->coreNum});
        epilogueFagPre();
        pipePre.Destroy();

        AscendC::TPipe pipeSoftmaxGrad;
        typename EpilogueFAGSfmg::Params sfmgParams;
        sfmgParams.batch = fagTilingData->batch;
        sfmgParams.qHeadNum = fagTilingData->qHeadNum;
        sfmgParams.kvHeadNum = fagTilingData->kvHeadNum;
        sfmgParams.g = fagTilingData->g;
        sfmgParams.qkHeadDim = fagTilingData->qkHeadDim;
        sfmgParams.qSeqlen = fagTilingData->qSeqlen;
        sfmgParams.kvSeqlen = fagTilingData->kvSeqlen;
        sfmgParams.coreNum = fagTilingData->coreNum;
        sfmgParams.t1 = fagTilingData->t1;
        sfmgParams.sfmgPreBeginAddr = fagTilingData->sfmgPreBeginAddr;
        sfmgParams.softmaxGradTilingData.srcM = fagTilingData->softmaxGradTilingData.srcM;
        sfmgParams.softmaxGradTilingData.srcK = fagTilingData->softmaxGradTilingData.srcK;
        sfmgParams.softmaxGradTilingData.srcSize = fagTilingData->softmaxGradTilingData.srcSize;
        sfmgParams.softmaxGradTilingData.outMaxM = fagTilingData->softmaxGradTilingData.outMaxM;
        sfmgParams.softmaxGradTilingData.outMaxK = fagTilingData->softmaxGradTilingData.outMaxK;
        sfmgParams.softmaxGradTilingData.outMaxSize = fagTilingData->softmaxGradTilingData.outMaxSize;
        sfmgParams.softmaxGradTilingData.splitM = fagTilingData->softmaxGradTilingData.splitM;
        sfmgParams.softmaxGradTilingData.splitK = fagTilingData->softmaxGradTilingData.splitK;
        sfmgParams.softmaxGradTilingData.splitSize = fagTilingData->softmaxGradTilingData.splitSize;
        sfmgParams.softmaxGradTilingData.reduceM = fagTilingData->softmaxGradTilingData.reduceM;
        sfmgParams.softmaxGradTilingData.reduceK = fagTilingData->softmaxGradTilingData.reduceK;
        sfmgParams.softmaxGradTilingData.reduceSize = fagTilingData->softmaxGradTilingData.reduceSize;
        sfmgParams.softmaxGradTilingData.rangeM = fagTilingData->softmaxGradTilingData.rangeM;
        sfmgParams.softmaxGradTilingData.tailM = fagTilingData->softmaxGradTilingData.tailM;
        sfmgParams.softmaxGradTilingData.tailSplitSize = fagTilingData->softmaxGradTilingData.tailSplitSize;
        sfmgParams.softmaxGradTilingData.tailReduceSize = fagTilingData->softmaxGradTilingData.tailReduceSize;
        EpilogueFAGSfmg epilogueFagSfmg(
            resource, &pipeSoftmaxGrad, params.dout, params.out, params.cu_seq_qlen, params.workspace, sfmgParams);
        epilogueFagSfmg();
        pipeSoftmaxGrad.Destroy();

        cBlockIdx = AscendC::GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;

        Init(params, fagTilingData);

        int64_t taskId = 1;
        int8_t extraLoopNum = 1;
        AscendC::TPipe pipeVec;
        TBuf<> unifiedBuffer;
        typename EpilogueFAGSabVec::Params sabParams;
        sabParams.softmaxTilingData.srcM = fagTilingData->softmaxTilingData.srcM;
        sabParams.softmaxTilingData.srcK = fagTilingData->softmaxTilingData.srcK;
        sabParams.softmaxTilingData.srcSize = fagTilingData->softmaxTilingData.srcSize;
        sabParams.softmaxTilingData.outMaxM = fagTilingData->softmaxTilingData.outMaxM;
        sabParams.softmaxTilingData.outMaxK = fagTilingData->softmaxTilingData.outMaxK;
        sabParams.softmaxTilingData.outMaxSize = fagTilingData->softmaxTilingData.outMaxSize;
        sabParams.softmaxTilingData.splitM = fagTilingData->softmaxTilingData.splitM;
        sabParams.softmaxTilingData.splitK = fagTilingData->softmaxTilingData.splitK;
        sabParams.softmaxTilingData.splitSize = fagTilingData->softmaxTilingData.splitSize;
        sabParams.softmaxTilingData.reduceM = fagTilingData->softmaxTilingData.reduceM;
        sabParams.softmaxTilingData.reduceK = fagTilingData->softmaxTilingData.reduceK;
        sabParams.softmaxTilingData.reduceSize = fagTilingData->softmaxTilingData.reduceSize;
        sabParams.softmaxTilingData.rangeM = fagTilingData->softmaxTilingData.rangeM;
        sabParams.softmaxTilingData.tailM = fagTilingData->softmaxTilingData.tailM;
        sabParams.softmaxTilingData.tailSplitSize = fagTilingData->softmaxTilingData.tailSplitSize;
        sabParams.softmaxTilingData.tailReduceSize = fagTilingData->softmaxTilingData.tailReduceSize;
        sabParams.coreNum = fagTilingData->coreNum;
        sabParams.batch = fagTilingData->batch;
        sabParams.kvHeadNum = fagTilingData->kvHeadNum;
        sabParams.g = fagTilingData->g;
        sabParams.qSeqlen = fagTilingData->qSeqlen;
        sabParams.kvSeqlen = fagTilingData->kvSeqlen;
        sabParams.qkHeadDim = fagTilingData->qkHeadDim;
        sabParams.vHeadDim = fagTilingData->vHeadDim;
        sabParams.s1Token = fagTilingData->s1Token;
        sabParams.s2Token = fagTilingData->s2Token;
        sabParams.sparseMode = fagTilingData->sparseMode;
        sabParams.s1Outer = fagTilingData->s1Outer;
        sabParams.s1CvInner = fagTilingData->s1CvInner;
        sabParams.s1CvTail = fagTilingData->s1CvTail;
        sabParams.s2Outer = fagTilingData->s2Outer;
        sabParams.s2CvInner = fagTilingData->s2CvInner;
        sabParams.baseMN = fagTilingData->baseMN;
        sabParams.keepProb = fagTilingData->keepProb;
        sabParams.scaleValue = fagTilingData->scaleValue;
        sabParams.dqWorkSpaceOffset = fagTilingData->dqWorkSpaceOffset;
        sabParams.dkWorkSpaceOffset = fagTilingData->dkWorkSpaceOffset;
        sabParams.dvWorkSpaceOffset = fagTilingData->dvWorkSpaceOffset;
        sabParams.sfmgPreBeginAddr = fagTilingData->sfmgPreBeginAddr;
        EpilogueFAGSabVec epilogueFAGSabVec(
            resource, &pipeVec, params.q, params.k, params.v, params.dout, nullptr, params.atten_mask, params.out,
            nullptr, nullptr, params.softmax_lse, params.cu_seq_qlen, params.cu_seq_kvlen, params.dq, params.dk,
            params.dv, params.workspace, sabParams, unifiedBuffer);

        EpilogueFAGDtmAdd epilogueFAGDtmAdd(
            resource, params.cu_seq_qlen, params.cu_seq_kvlen, params.workspace,
            {fagTilingData->coreNum, fagTilingData->batch, fagTilingData->kvHeadNum, fagTilingData->g,
             fagTilingData->qSeqlen, fagTilingData->kvSeqlen, fagTilingData->qkHeadDim, fagTilingData->vHeadDim,
             fagTilingData->s1Outer, fagTilingData->s1CvInner, fagTilingData->s1CvTail, fagTilingData->s2Outer,
             fagTilingData->s2CvInner},
            unifiedBuffer);

        bool isFinish = false;
        int64_t startCoreId = 0;
        while (!isFinish) {
            isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[0]);
        }
        dbParams[0].taskId = 0;
        if constexpr (IS_DTM == ENABLE) {
            extraLoopNum = 2;
        }
        while (extraLoopNum >= 0) {
            isFinish = false;
            startCoreId = 0;
            while (!isFinish) {
                isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[taskId % 3]);
            }

            dbParams[taskId % 3].taskId = taskId;
            if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                AscendC::CrossCoreWaitFlag(SYNC_C1_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);

                if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                    epilogueFAGSabVec(dbParams[(taskId - 1) % 3]);
                    AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(
                        SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
                if constexpr (IS_DTM != ENABLE) {
                    if (dbParams[(taskId) % 3].blockId == -1) {
                        AscendC::CrossCoreWaitFlag(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    }
                }
            }

            if constexpr (IS_DTM == ENABLE) {
                if (taskId > 1 && dbParams[(taskId - 2) % 3].blockId != -1) {
                    CrossCoreWaitFlag(SYNC_C2_V1_FLAG[dbParams[(taskId - 2) % 3].taskId % 2]);
                }
                if (taskId > 1) {
                    SyncAll();
                    epilogueFAGDtmAdd(
                        dbParams[(taskId - 2) % 3], dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dqDtmWsGm, dkDtmWsGm,
                        dvDtmWsGm);
                    SyncAll();
                }
            }
            taskId++;
            if (blockStartIdx == -1) {
                extraLoopNum -= 1;
            }
        }
        AscendC::SyncAll();
        pipeVec.Destroy();

        AscendC::TPipe pipePost;
        EpilogueFAGPost epilogueFagPost(
            resource, &pipePost, params.dq, params.dk, params.dv, params.workspace,
            {fagTilingData->dqWorkSpaceOffset, fagTilingData->dkWorkSpaceOffset, fagTilingData->dvWorkSpaceOffset,
             fagTilingData->qSize, fagTilingData->kvSize, fagTilingData->coreNum, fagTilingData->scaleValue});
        epilogueFagPost();
        pipePost.Destroy();
    }

private:
    Arch::Resource<ArchTag> resource;
    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};
    int64_t blockStartIdx = 0;
    DBParams dbParams[3];
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t value_d;
    int64_t dAlign;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t vecBlockNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    int64_t dqOutBase{0};
    int64_t kvOutBase{0};
    int64_t dqOutIdx{0};
    int64_t kvOutIdx{0};
    int64_t dqOutArr[24];
    int64_t kvOutArr[24];

    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool dropBitMode;

    uint32_t pingpongIdx = 1;
    uint32_t baseMN;
    uint32_t cubeBaseMN;
};

// ============================================================
// TLA-style TypeSelector: uses GemmShape<> for tile shapes
// and BlockMmadTla with PackedTileCopyTla for matmul blocks.
// ============================================================

template <const DTemplateType DTEMPLATETYPE>
struct TypeSelector;

template <>
struct TypeSelector<DTemplateType::Aligned64> {
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FagSdp<1, 1, true>;
    using BlockTileShape = GemmShape<512, 512, 64>;
    using L1TileShapeCube1 = GemmShape<512, 512, 64>;
    using L0TileShapeCube1 = GemmShape<128, 256, 64>;

    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FagdQKV<2, 1, true>;
    using BlockTileShapeCube2 = GemmShape<512, 64, 512>;
    using L1ATileShapeCube2 = GemmShape<128, 0, 256>;
    using L1BTileShapeCube2 = GemmShape<0, 64, 512>;
    using L0TileShapeCube2 = GemmShape<128, 64, 128>;
};

template <>
struct TypeSelector<DTemplateType::Aligned128> {
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FagSdp<1, 1, true>;
    using BlockTileShape = GemmShape<512, 512, 128>;
    using L1TileShapeCube1 = GemmShape<512, 512, 128>;
    using L0TileShapeCube1 = GemmShape<128, 128, 128>;

    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FagdQKV<2, 1, true>;
    using BlockTileShapeCube2 = GemmShape<512, 128, 512>;
    using L1ATileShapeCube2 = GemmShape<128, 0, 256>;
    using L1BTileShapeCube2 = GemmShape<0, 128, 512>;
    using L0TileShapeCube2 = GemmShape<128, 128, 128>;
};

template <>
struct TypeSelector<DTemplateType::Aligned192> {
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FagSdp<2, 2, true>;
    using BlockTileShape = GemmShape<512, 512, 192>;
    using L1TileShapeCube1 = GemmShape<128, 128, 128>;
    using L0TileShapeCube1 = GemmShape<128, 128, 128>;

    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FagdQKV<2, 1, true>;
    using BlockTileShapeCube2 = GemmShape<512, 192, 512>;
    using L1ATileShapeCube2 = GemmShape<128, 0, 128>;
    using L1BTileShapeCube2 = GemmShape<0, 192, 512>;
    using L0TileShapeCube2 = GemmShape<128, 192, 64>;
};

template <>
struct TypeSelector<DTemplateType::Aligned256> {
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FagSdp<2, 1, true>;
    using BlockTileShape = GemmShape<512, 512, 256>;
    using L1TileShapeCube1 = GemmShape<128, 512, 256>;
    using L0TileShapeCube1 = GemmShape<128, 128, 128>;

    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FagdQKV<2, 1, true>;
    using BlockTileShapeCube2 = GemmShape<512, 256, 512>;
    using L1ATileShapeCube2 = GemmShape<128, 0, 128>;
    using L1BTileShapeCube2 = GemmShape<0, 256, 512>;
    using L0TileShapeCube2 = GemmShape<128, 256, 64>;
};

// ============================================================
// TLA-style kernel entry point: FAG_BSND
//
// Uses BlockMmadTla with PackedTileCopyTla for each GEMM block.
// Passes tla::Tensor views to BlockMmadTla operator().
// ============================================================

template <
    const uint32_t IS_ATTEN_MASK = 0, const DTemplateType DTEMPLATETYPE, typename DataType = half,
    const uint32_t INPUT_LAYOUT = BSND, const uint32_t IS_DTM = 0>
CATLASS_GLOBAL void FAG_BSND(
    uint64_t hardwareSyncAddr, GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR drop_mask,
    GM_ADDR atten_mask, GM_ADDR row_max, GM_ADDR row_sum, GM_ADDR softmax_lse, GM_ADDR cu_seq_qlen,
    GM_ADDR cu_seq_kvlen, GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR alibi_slopes_, GM_ADDR workspace,
    GM_ADDR tiling)
{
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);

    using ArchTag = Arch::AtlasA2;

    // Cube1: Q * K^T and dOut * V^T
    using DispatchPolicyCube1 = typename TypeSelector<DTEMPLATETYPE>::DispatchPolicyCube1;
    using BlockTileShape = typename TypeSelector<DTEMPLATETYPE>::BlockTileShape;
    using L1TileShapeCube1 = typename TypeSelector<DTEMPLATETYPE>::L1TileShapeCube1;
    using L0TileShapeCube1 = typename TypeSelector<DTEMPLATETYPE>::L0TileShapeCube1;

    // Cube2: dQ = dS * K
    using DispatchPolicyCube2 = typename TypeSelector<DTEMPLATETYPE>::DispatchPolicyCube2;
    using BlockTileShapeCube2 = typename TypeSelector<DTEMPLATETYPE>::BlockTileShapeCube2;
    using L1ATileShapeCube2 = typename TypeSelector<DTEMPLATETYPE>::L1ATileShapeCube2;
    using L1BTileShapeCube2 = typename TypeSelector<DTEMPLATETYPE>::L1BTileShapeCube2;
    using L0TileShapeCube2 = typename TypeSelector<DTEMPLATETYPE>::L0TileShapeCube2;

    // Cube3: dK = dS^T * Q, dV = P^T * dOut
    using DispatchPolicyCube3 = Catlass::Gemm::MmadAtlasA2FagdQKV<2, 1, true>;
    using BlockTileShapeCube3 = BlockTileShapeCube2;
    using L1ATileShapeCube3 = L1ATileShapeCube2;
    using L1BTileShapeCube3 = L1BTileShapeCube2;
    using L0TileShapeCube3 = L0TileShapeCube2;

    // TLA-style: use PackedTileCopyTla for data movement between memory levels
    // and BlockMmadFagSdp / BlockMmadFAG for the GEMM operations. The FAG dispatch
    // policies (MmadAtlasA2FagSdp / MmadAtlasA2FagdQKV) resolve to these specialized
    // block types in the catlass include tree.

    // Cube1: RowMajor * ColumnMajor = RowMajor
    using ElementA1 = DataType;
    using LayoutA1 = layout::RowMajor;
    using ElementB1 = DataType;
    using LayoutB1 = layout::ColumnMajor;
    using ElementC1 = float;
    using LayoutC1 = layout::RowMajor;
    using A1Type = Catlass::Gemm::GemmType<ElementA1, LayoutA1>;
    using B1Type = Catlass::Gemm::GemmType<ElementB1, LayoutB1>;
    using C1Type = Catlass::Gemm::GemmType<ElementC1, LayoutC1>;
    using TileCopyCube1 = Gemm::Tile::TileCopy<ArchTag, A1Type, B1Type, C1Type, void>;
    using BlockMmadFAGCube1 = Gemm::Block::BlockMmadFagSdp<
        DispatchPolicyCube1, BlockTileShape, L1TileShapeCube1, L0TileShapeCube1, A1Type, B1Type, C1Type, void,
        TileCopyCube1>;

    // Cube2: RowMajor * RowMajor = RowMajor
    using ElementA2 = DataType;
    using LayoutA2 = layout::RowMajor;
    using ElementB2 = DataType;
    using LayoutB2 = layout::RowMajor;
    using ElementC2 = float;
    using LayoutC2 = layout::RowMajor;
    using A2Type = Catlass::Gemm::GemmType<ElementA2, LayoutA2>;
    using B2Type = Catlass::Gemm::GemmType<ElementB2, LayoutB2>;
    using C2Type = Catlass::Gemm::GemmType<ElementC2, LayoutC2>;
    using TileCopyCube2 = Gemm::Tile::TileCopy<ArchTag, A2Type, B2Type, C2Type, void>;
    using BlockMmadFAGCube2 = Gemm::Block::BlockMmadFAG<
        DispatchPolicyCube2, BlockTileShapeCube2, L1ATileShapeCube2, L1BTileShapeCube2, L0TileShapeCube2, A2Type,
        B2Type, C2Type, void, TileCopyCube2>;

    // Cube3: ColumnMajor * RowMajor = RowMajor
    using ElementA3 = DataType;
    using LayoutA3 = layout::ColumnMajor;
    using ElementB3 = DataType;
    using LayoutB3 = layout::RowMajor;
    using ElementC3 = float;
    using LayoutC3 = layout::RowMajor;
    using A3Type = Catlass::Gemm::GemmType<ElementA3, LayoutA3>;
    using B3Type = Catlass::Gemm::GemmType<ElementB3, LayoutB3>;
    using C3Type = Catlass::Gemm::GemmType<ElementC3, LayoutC3>;
    using TileCopyCube3 = Gemm::Tile::TileCopy<ArchTag, A3Type, B3Type, C3Type, void>;
    using BlockMmadFAGCube3 = Gemm::Block::BlockMmadFAG<
        DispatchPolicyCube3, BlockTileShapeCube3, L1ATileShapeCube3, L1BTileShapeCube3, L0TileShapeCube3, A3Type,
        B3Type, C3Type, void, TileCopyCube3>;

    // Epilogue blocks (unchanged, use existing BlockEpilogue)
    using ElementOutput = float;
    using LayoutOutput = layout::RowMajor;
    using OutputType = float;

    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    using UpdateType = Catlass::Gemm::GemmType<ElementUpdate, LayoutUpdate>;

    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using InputType = DataType;

    using EpilogueAtlasA2FAGPre = Catlass::Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPre =
        Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType, UpdateType, InputType>;

    using EpilogueAtlasA2FAGSfmg = Catlass::Epilogue::EpilogueAtlasA2FAGSfmg<INPUT_LAYOUT>;
    using EpilogueFAGSfmg =
        Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGSfmg, OutputType, UpdateType, InputType>;

    using EpilogueAtlasA2CustomSabVec = Catlass::Epilogue::EpilogueAtlasA2CustomSabVec<INPUT_LAYOUT>;
    using EpilogueFAGSabVec =
        Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2CustomSabVec, OutputType, UpdateType, InputType>;

    using EpilogueAtlasA2FAGPost = Catlass::Epilogue::EpilogueAtlasA2FAGPost;
    using EpilogueFAGPost =
        Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPost, OutputType, UpdateType, InputType>;

    using EpilogueAtlasA2FAGDtmAdd = Catlass::Epilogue::EpilogueAtlasA2FAGDtmAdd<INPUT_LAYOUT>;
    using EpilogueFAGDtmAdd =
        Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGDtmAdd, OutputType, UpdateType, InputType>;

    // Kernel level
    using FAGKernel = FlashAttentionScoreGrad<
        BlockMmadFAGCube1, BlockMmadFAGCube2, BlockMmadFAGCube3, EpilogueFAGPre, EpilogueFAGSfmg, EpilogueFAGSabVec,
        EpilogueFAGPost, EpilogueFAGDtmAdd, INPUT_LAYOUT, IS_DTM>;
    FAGKernelParams params{
        dout,         q,   k,   v,   out,           drop_mask, atten_mask, row_max, row_sum, softmax_lse, cu_seq_qlen,
        cu_seq_kvlen, dq_, dk_, dv_, alibi_slopes_, workspace, tiling};

    FAGKernel flashAttn;
    flashAttn(params);
}

