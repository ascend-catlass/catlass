/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATRIX_INVERSE_HPP
#define CATLASS_GEMM_KERNEL_MATRIX_INVERSE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// 矩阵求逆算子：使用分块 LU 分解计算 A 的逆矩阵 A^(-1)。
//
// 并行化策略：
//   - Panel LU、行交换、TRSM      → AIV core 0 串行
//   - Schur 余项 GEMM             → AIC 多核并行
//   - InvertUpperTri 分块 TRTRI   → DGEMM 剥离到 AIC
//   - ApplyLInverse 分块求解      → DGEMM 剥离到 AIC
//   - 列交换                      → AIV core 0 串行
//
// Workspace 布局：
//   Offset 0:            gmInvLDense  (N*N)        GETRF Schur GEMM / L 因子 (ApplyLInverse 用)
//   Offset N*N:          panel        (N*NB)       预留
//   Offset N*N+N*NB:     gmInvUDense  (N*N)        稠密 invU / U 因子源 (TRTRI 用)
//   Offset 2*N*N+N*NB:   gmGemmTemp   (N*NB)       TRTRI + ApplyLInverse GEMM 暂存
template <class ArchTag_, class Element_, class BlockMmad_, class BlockScheduler_>
class MatrixInverse {
public:
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    using LayoutA = typename BlockMmad::LayoutA;
    using L1TileShape = typename BlockMmad::L1TileShape;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t NB = 64;

    // Epilogue: D = alpha*C + beta*X
    using EpiDispatch = Epilogue::EpilogueAtlasA2Gemm;
    using EpiFloat = Gemm::GemmType<float, Layout>;
    static constexpr uint32_t kEpiLen = (L1_TILE_M * L1_TILE_N) / 2;
    using EpiAdd = Epilogue::Tile::TileElemWiseAdd<ArchTag, EpiFloat, kEpiLen>;
    using EpiMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, EpiFloat, kEpiLen>;
    using EpiShape = MatrixShape<L1_TILE_M / 2, L1_TILE_N>;
    using EpiCast = Epilogue::Tile::TileCast<ArchTag, EpiFloat, EpiFloat, EpiShape>;
    using EpiCopy = Epilogue::Tile::TileCopy<ArchTag, EpiFloat, EpiFloat, EpiFloat>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpiDispatch, EpiFloat, EpiFloat, EpiFloat, EpiAdd, EpiMuls, EpiCast, EpiCopy>;

    struct Params {
        uint32_t N;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrIpiv;
        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE Params()
        {}
        CATLASS_HOST_DEVICE
        Params(uint32_t N_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrIpiv_, GM_ADDR ptrWorkspace_)
            : N(N_), ptrA(ptrA_), layoutA(layoutA_), ptrIpiv(ptrIpiv_), ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        uint32_t N;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrIpiv;
        uint8_t* ptrWorkspace{nullptr};

        CATLASS_HOST_DEVICE Arguments()
        {}
        CATLASS_HOST_DEVICE
        Arguments(uint32_t N_, uint8_t* ptrA_, LayoutA layoutA_, uint8_t* ptrIpiv_, uint8_t* ptrWorkspace_)
            : N(N_), ptrA(ptrA_), layoutA(layoutA_), ptrIpiv(ptrIpiv_), ptrWorkspace(ptrWorkspace_)
        {}
    };

    static bool CanImplement(const Arguments& args)
    {
        // PanelGetrf 在 UB 上依次叠加 ubU/ubT/ubTmp 三段长度为 bufSize 的缓冲，
        // bufSize 最大为 RoundUp8(N)，需保证 3*bufSize*sizeof(Element) 不超过 UB 容量。
        constexpr uint32_t MAX_N = ArchTag::UB_SIZE / (3 * sizeof(Element));
        return args.N > 0 && args.N <= MAX_N;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        size_t matSize = static_cast<size_t>(args.N) * args.N * sizeof(Element);
        size_t panelSize = static_cast<size_t>(args.N) * NB * sizeof(Element);
        size_t invUDense = static_cast<size_t>(args.N) * args.N * sizeof(Element);
        size_t gemmTemp = static_cast<size_t>(args.N) * NB * sizeof(Element);
        return matSize + panelSize + invUDense + gemmTemp;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        GM_ADDR ptrWorkspace = workspace;
        return Params(args.N, args.ptrA, args.layoutA, args.ptrIpiv, ptrWorkspace);
    }

    CATLASS_DEVICE MatrixInverse()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    // ==================================================================
    // AIC: GETRF Schur GEMM + TRTRI GEMM + ApplyLInverse GEMM
    // ==================================================================
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t N = params.N;
        uint32_t ldA = N;
        size_t wsElemOff = static_cast<size_t>(N) * N;
        size_t invUDenseOff = wsElemOff + static_cast<size_t>(N) * NB;
        size_t gemmTempOff = invUDenseOff + static_cast<size_t>(N) * N;

        AscendC::GlobalTensor<Element> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrA));
        AscendC::GlobalTensor<Element> gmWork;
        gmWork.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace));
        AscendC::GlobalTensor<Element> gmInvUDense;
        gmInvUDense.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + invUDenseOff);
        AscendC::GlobalTensor<Element> gmGemmTemp;
        gmGemmTemp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + gemmTempOff);

        Arch::Resource<ArchTag> resource;

        // ======== GETRF: TrsmLeftLowerUnit + Schur complement GEMM ========
        for (uint32_t k = 0; k < N; k += NB) {
            uint32_t actualNb = (k + NB <= N) ? NB : (N - k);
            bool hasTrailing = (k + actualNb < N);

            // B1: wait for AIV Panel done, do Trsm
            AscendC::SyncAll<false>();

            if (hasTrailing) {
                TrsmLeftLowerUnitGemm(resource, gmA, gmGemmTemp, k, actualNb, N - k - actualNb, ldA);
            }

            // B2: Trsm done, do full Schur
            AscendC::SyncAll<false>();

            if (hasTrailing) {
                uint32_t M = N - k - actualNb;
                SchurGemmToWorkspace(resource, gmA, gmWork, k, actualNb, M, actualNb, ldA);
            }

            // B3: Schur done
            AscendC::SyncAll<false>();

            // B4: wait for AIV epilogue done before next iteration
            AscendC::SyncAll<false>();
        }

        // ======== TRTRI: InvertUpperTri GEMM (j 从左向右, k 从 j+NB 向右) ========
        for (uint32_t j = 0; j < N; j += NB) {
            uint32_t actNb = (j + NB <= N) ? NB : (N - j);
            uint32_t prevRows = j; // 当前对角块上方的行数

            // TRSMtemp GEMM: gmInvUDense(0:j, j:j+actNb) = U × invU_diag (AIC direct)
            if (j > 0) {
                AscendC::SyncAll<false>();

                TrsmTempGemmToWorkspace(resource, gmInvUDense, gmInvUDense, j, actNb, N, ldA);

                AscendC::SyncAll<false>();
            }

            for (uint32_t k = j + actNb; k < N; k += NB) {
                uint32_t kActNb = (k + NB <= N) ? NB : (N - k);

                AscendC::SyncAll<false>();

                InvertUpperTriGemmToWorkspace(
                    resource, gmInvUDense, gmInvUDense, gmGemmTemp, j, actNb, prevRows, k, kActNb, N, ldA);

                AscendC::SyncAll<false>();
            }

            // DTRMM: gmGemmTemp = invU(0:j,0:j) × temp(0:j, j:j+actNb) (AIC)
            //   invU_top 为稠密上三角（下三角=0），~50% Cube 利用率，但消除串行 O(N³/6)
            if (j > 0) {
                AscendC::SyncAll<false>();

                DtrmmGemmToWorkspace(resource, gmInvUDense, gmGemmTemp, j, actNb, N, ldA);

                AscendC::SyncAll<false>();
            }
        }

        // ======== ApplyLInverse GEMM ========
        int lastBlock = (static_cast<int>(N) - 1) / static_cast<int>(NB) * static_cast<int>(NB);
        for (int jj = lastBlock; jj >= 0; jj -= static_cast<int>(NB)) {
            uint32_t colStart = static_cast<uint32_t>(jj);
            uint32_t actualNb = (colStart + NB <= N) ? NB : (N - colStart);
            uint32_t M = N - colStart - actualNb;

            // B1: wait for AIV, do ApplyLInverse GEMM
            AscendC::SyncAll<false>();

            if (M > 0) {
                ApplyLInverseGemmToWorkspace(resource, gmA, gmWork, gmGemmTemp, colStart, actualNb, M, N, ldA);
            }

            // B2: ApplyLInverse GEMM done
            AscendC::SyncAll<false>();

            // B3: wait for AIV invL flush, do ApplyLIntraBlock GEMM
            AscendC::SyncAll<false>();

            ApplyLIntraBlockGemm(resource, gmA, gmGemmTemp, colStart, actualNb, N, ldA, NB);

            // B4: ApplyLIntraBlock GEMM done
            AscendC::SyncAll<false>();
        }
    }

    // ==================================================================
    // AIV: GETRF + Blocked TRTRI + ApplyLInverse + SwapColumns
    // ==================================================================
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        uint32_t N = params.N;
        uint32_t ldA = N;
        size_t wsElemOff = static_cast<size_t>(N) * N;
        size_t invUDenseOff = wsElemOff + static_cast<size_t>(N) * NB;
        size_t gemmTempOff = invUDenseOff + static_cast<size_t>(N) * N;

        AscendC::GlobalTensor<Element> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrA));
        AscendC::GlobalTensor<int32_t> gmIpiv;
        gmIpiv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrIpiv));
        AscendC::GlobalTensor<Element> gmInvLDense;
        gmInvLDense.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace));
        AscendC::GlobalTensor<Element> gmInvUDense;
        gmInvUDense.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + invUDenseOff);
        AscendC::GlobalTensor<Element> gmGemmTemp;
        gmGemmTemp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + gemmTempOff);

        // ======== GETRF loop ========
        for (uint32_t k = 0; k < N; k += NB) {
            uint32_t actualNb = (k + NB <= N) ? NB : (N - k);
            bool hasTrailing = (k + actualNb < N);

            // ================================================================
            // Core 0: Panel(k)
            // ================================================================
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::DataCacheCleanAndInvalid<
                    Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
                PanelGetrf(gmA, gmIpiv, k, N - k, actualNb, ldA);
                if (k > 0) {
                    ApplyRowSwaps(gmA, gmIpiv, k, actualNb, 0, k, ldA);
                }
                if (hasTrailing) {
                    ApplyRowSwaps(gmA, gmIpiv, k, actualNb, k + actualNb, N, ldA);
                    ComputeInvLDiagGETRF(gmA, gmGemmTemp, k, actualNb, ldA);
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmGemmTemp);
                }
                AscendC::DataCacheCleanAndInvalid<
                    Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
            }

            // B1 + B2: sync with AIC Trsm
            AscendC::SyncAll<false>();
            AscendC::SyncAll<false>();

            // Invalidate gmA after Trsm (AIC DMA writes to gmA)
            if (hasTrailing) {
                if (AscendC::GetBlockIdx() == 0) {
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
                }
            }

            // B3: sync with AIC Schur
            AscendC::SyncAll<false>();

            // ================================================================
            // Full Schur epilogue (ALL AIV cores): gmA -= gmInvLDense
            // ================================================================
            if (hasTrailing) {
                uint32_t M = N - k - actualNb;
                GM_ADDR ptrSchur = reinterpret_cast<GM_ADDR>(
                    reinterpret_cast<__gm__ Element*>(params.ptrA) + (k + actualNb) * ldA + (k + actualNb));
                Layout layoutSchur(M, M, ldA);
                typename BlockEpilogue::Params epiParams{-1.0f, 1.0f, ptrSchur, layoutSchur, ptrSchur, layoutSchur};
                Arch::Resource<ArchTag> resource;
                BlockEpilogue epilogue(resource, GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K}, epiParams);
                Layout layoutWork(M, M);
                BlockScheduler scheduler(GemmCoord{M, M, 1}, MatrixCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = scheduler.GetCoreLoops();
                uint32_t blkIdx = AscendC::GetBlockIdx();
                uint32_t blkNum = AscendC::GetBlockNum();
                for (uint32_t t = blkIdx; t < coreLoops; t += blkNum) {
                    GemmCoord bc = scheduler.GetBlockCoord(t);
                    GemmCoord as = scheduler.GetActualBlockShape(bc);
                    epilogue(as, bc, gmInvLDense, layoutWork, 0);
                }
            }

            // Barrier: wait for all cores to finish epilogue before next Panel
            AscendC::SyncAll<false>();
        }

        // ======== Phase 2: GETRI ========

        // Step 1: copy LU to workspace, build dense U in gmInvUDense
        if (AscendC::GetBlockIdx() == 0) {
            CopyStrictLowerToDense(gmA, gmInvLDense, N, ldA); // L (strict lower)
            CopyUpperToDense(gmA, gmInvUDense, N, ldA);       // U (dense)
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        // Clean gmInvLDense on all cores (multi-core scalar writes → AIC DMA reads)
        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmInvLDense);
        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmInvUDense);

        // Step 2: Blocked TRTRI — invU in gmInvUDense
        //   外层 j 从左向右遍历对角块
        for (uint32_t j = 0; j < N; j += NB) {
            uint32_t actNb = (j + NB <= N) ? NB : (N - j);

            if (AscendC::GetBlockIdx() == 0) {
                TRTRIdiag(gmInvUDense, j, actNb, N); // ① invU_diag

                if (j > 0) {
                    // Flush diag block (written by TRTRIdiag) → AIC DMA read
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmInvUDense);
                }
            }

            // Sync with AIC TRSMtemp GEMM (writes directly to gmInvUDense)
            if (j > 0) {
                AscendC::SyncAll<false>();
                AscendC::SyncAll<false>();
            }

            // 内层 k 遍历右侧块（③ DGEMM 剥离到 AIC）
            for (uint32_t k = j + actNb; k < N; k += NB) {
                uint32_t kActNb = (k + NB <= N) ? NB : (N - k);

                AscendC::SyncAll<false>();
                AscendC::SyncAll<false>();

                if (j > 0) {
                    // Invalidate after AIC DMA W → Scalar R
                    if (AscendC::GetBlockIdx() == 0) {
                        AscendC::DataCacheCleanAndInvalid<
                            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
                            gmInvUDense);
                    }

                    // Apply GEMM result (multi-core epilogue):
                    //   gmInvUDense(0:j, k:k+kActNb) -= gmGemmTemp
                    GM_ADDR ptrGemmTemp =
                        reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + gemmTempOff);
                    Layout layoutGemmTemp(j, kActNb);
                    GM_ADDR ptrInvUBlk = reinterpret_cast<GM_ADDR>(
                        reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + invUDenseOff + k);
                    Layout layoutInvUBlk(j, kActNb, N);
                    typename BlockEpilogue::Params epiParams{-1.0f,          1.0f,       ptrGemmTemp,
                                                             layoutGemmTemp, ptrInvUBlk, layoutInvUBlk};
                    Arch::Resource<ArchTag> resource;
                    BlockEpilogue epilogue(resource, GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K}, epiParams);
                    BlockScheduler scheduler(GemmCoord{j, kActNb, 1}, MatrixCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t coreLoops = scheduler.GetCoreLoops();
                    uint32_t blkIdx = AscendC::GetBlockIdx();
                    uint32_t blkNum = AscendC::GetBlockNum();
                    for (uint32_t t = blkIdx; t < coreLoops; t += blkNum) {
                        GemmCoord bc = scheduler.GetBlockCoord(t);
                        GemmCoord as = scheduler.GetActualBlockShape(bc);
                        epilogue(as, bc, gmGemmTemp, layoutGemmTemp, 0);
                    }

                    // Flush for next k iteration's AIC DMA R
                    if (AscendC::GetBlockIdx() == 0) {
                        AscendC::DataCacheCleanAndInvalid<
                            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
                            gmInvUDense);
                    }
                }
            }

            // DTRMM: AIC GEMM → gmGemmTemp → AIV 取反拷贝到 gmInvUDense
            if (j > 0) {
                AscendC::SyncAll<false>();
                AscendC::SyncAll<false>();

                if (AscendC::GetBlockIdx() == 0) {
                    // Invalidate after AIC DMA W → Scalar R
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmInvUDense);
                    NegateCopyDtrmm(gmInvUDense, gmGemmTemp, j, actNb, N);
                    // Flush for subsequent ApplyLInverse phase AIC DMA read
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmInvUDense);
                }
            }
        }

        // Step 3: copy invU from gmInvUDense back to gmA (上三角) + 下三角清零
        if (AscendC::GetBlockIdx() == 0) {
            CopyUpperToDense(gmInvUDense, gmA, N, N);
            // Flush before AIC reads
            AscendC::DataCacheCleanAndInvalid<
                Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
        }

        // Step 4: Blocked ApplyLInverse
        int lastBlock = (static_cast<int>(N) - 1) / static_cast<int>(NB) * static_cast<int>(NB);
        for (int jj = lastBlock; jj >= 0; jj -= static_cast<int>(NB)) {
            uint32_t colStart = static_cast<uint32_t>(jj);
            uint32_t actualNb = (colStart + NB <= N) ? NB : (N - colStart);
            uint32_t M = N - colStart - actualNb;

            AscendC::SyncAll<false>();
            AscendC::SyncAll<false>();

            if (M > 0) {
                GM_ADDR ptrInvUBlk = reinterpret_cast<GM_ADDR>(
                    reinterpret_cast<__gm__ Element*>(params.ptrWorkspace) + invUDenseOff + colStart);
                Layout layoutInvU(N, actualNb, N);
                GM_ADDR ptrDst = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ Element*>(params.ptrA) + colStart);
                Layout layoutDst(N, actualNb, N);
                typename BlockEpilogue::Params epiParams{-1.0f, 1.0f, ptrInvUBlk, layoutInvU, ptrDst, layoutDst};
                Arch::Resource<ArchTag> resource;
                BlockEpilogue epilogue(resource, GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K}, epiParams);
                Layout layoutGemmTemp(N, actualNb);
                BlockScheduler scheduler(GemmCoord{N, actualNb, 1}, MatrixCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = scheduler.GetCoreLoops();
                uint32_t blkIdx = AscendC::GetBlockIdx();
                uint32_t blkNum = AscendC::GetBlockNum();
                for (uint32_t t = blkIdx; t < coreLoops; t += blkNum) {
                    GemmCoord bc = scheduler.GetBlockCoord(t);
                    GemmCoord as = scheduler.GetActualBlockShape(bc);
                    epilogue(as, bc, gmGemmTemp, layoutGemmTemp, 0);
                }
            }

            if (AscendC::GetBlockIdx() == 0) {
                if (M > 0) {
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
                }
                // Compute invL: read L_diag from gmWork, write to gmGemmTemp (O(NB³))
                ComputeInvLDiag(gmInvLDense, gmGemmTemp, colStart, actualNb, N, ldA, NB);
                // Flush invL → AIC DMA read for ApplyLIntraBlockGemm
                AscendC::DataCacheCleanAndInvalid<
                    Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmGemmTemp);
            }

            // Wait for AIC ApplyLIntraBlockGemm (direct write to gmA)
            AscendC::SyncAll<false>();
            AscendC::SyncAll<false>();

            if (AscendC::GetBlockIdx() == 0) {
                // Invalidate gmA after AIC DMA write
                AscendC::DataCacheCleanAndInvalid<
                    Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
                if (colStart > 0) {
                    AscendC::DataCacheCleanAndInvalid<
                        Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
                }
            }
        }

        // Step 5: SwapColumns
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::DataCacheCleanAndInvalid<
                Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
            for (int jj = static_cast<int>(N) - 1; jj >= 0; --jj) {
                uint32_t j = static_cast<uint32_t>(jj);
                uint32_t pivotRow = static_cast<uint32_t>(gmIpiv.GetValue(j));
                if (pivotRow != j) {
                    SwapColumns(gmA, j, pivotRow, N, ldA);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // 构造 RowMajor tla layout（rows×cols，显式 leading dim ld），用于把每个 GEMM helper
    // 原有的 `Layout(rows, cols, ld)`（catlass RowMajor + leading dim）1:1 翻译成 tla::Layout。
    static CATLASS_HOST_DEVICE auto RmLayout(uint32_t rows, uint32_t cols, uint32_t ld)
    {
        return tla::MakeLayout(
            tla::MakeShape(rows, cols), tla::MakeStride(static_cast<int64_t>(ld), tla::Int<1>{}),
            tla::MakeShape(rows, cols));
    }

    // ============================================
    // Panel LU 分解
    // ============================================
    CATLASS_DEVICE
    void PanelGetrf(
        AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<int32_t>& gmIpiv, uint32_t colStart, uint32_t M,
        uint32_t K, uint32_t ldA)
    {
        // Pre-allocate UB: row-wise needs maxNCols, col-wise needs M. Align to 8.
        uint32_t maxNCols = (K / 8) * 8;
        uint32_t bufSize = ((maxNCols > M ? maxNCols : M) + 7) / 8 * 8;
        Arch::Resource<ArchTag> resource;
        auto ubU = resource.ubBuf.template GetBufferByByte<Element>(0);
        auto ubT = resource.ubBuf.template GetBufferByByte<Element>(bufSize * sizeof(Element));
        auto ubTmp = resource.ubBuf.template GetBufferByByte<Element>(2 * bufSize * sizeof(Element));

        for (uint32_t kk = 0; kk < K; ++kk) {
            uint32_t globalK = colStart + kk;
            uint32_t pivotRow = kk;
            Element maxAbsVal = static_cast<Element>(0);
            for (uint32_t i = kk; i < M; ++i) {
                Element val = gmA.GetValue((colStart + i) * ldA + globalK);
                Element absVal = (val < 0) ? -val : val;
                if (absVal > maxAbsVal) {
                    maxAbsVal = absVal;
                    pivotRow = i;
                }
            }
            gmIpiv.SetValue(globalK, static_cast<int32_t>(colStart + pivotRow));
            if (pivotRow != kk) {
                if (ldA % 8 == 0) {
                    uint32_t rOffKk = (colStart + kk) * ldA + colStart;
                    uint32_t rOffPivot = (colStart + pivotRow) * ldA + colStart;
                    AscendC::DataCopy(ubU, gmA[rOffKk], K);
                    AscendC::DataCopy(ubT, gmA[rOffPivot], K);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(gmA[rOffKk], ubT, K);
                    AscendC::DataCopy(gmA[rOffPivot], ubU, K);
                    AscendC::PipeBarrier<PIPE_MTE3>();
                } else {
                    for (uint32_t j = 0; j < K; ++j) {
                        uint32_t globalJ = colStart + j;
                        Element tmp = gmA.GetValue((colStart + kk) * ldA + globalJ);
                        gmA.SetValue(
                            (colStart + kk) * ldA + globalJ, gmA.GetValue((colStart + pivotRow) * ldA + globalJ));
                        gmA.SetValue((colStart + pivotRow) * ldA + globalJ, tmp);
                    }
                }
            }
            Element diag = gmA.GetValue((colStart + kk) * ldA + globalK);
            if (diag != static_cast<Element>(0)) {
                for (uint32_t i = kk + 1; i < M; ++i) {
                    Element scaled = gmA.GetValue((colStart + i) * ldA + globalK) / diag;
                    gmA.SetValue((colStart + i) * ldA + globalK, scaled);
                }
                uint32_t nCols = K - kk - 1;
                uint32_t gmOff = (colStart + kk) * ldA + (colStart + kk + 1);
                if (nCols >= 4 && ldA % 8 == 0) {
                    // Row-wise DataCopy with alignment skip:
                    //   skip = leading cols to reach 32B boundary
                    //   DataCopy bulk from aligned address, scalar prefix+suffix
                    uint32_t skip = (8 - gmOff % 8) % 8;
                    uint32_t dataCols = (nCols > skip) ? (nCols - skip) : 0;
                    uint32_t nColsAligned = (dataCols / 8) * 8;

                    // Scalar prefix: columns before 32B boundary
                    for (uint32_t j = kk + 1; j < kk + 1 + skip && j < K; ++j) {
                        uint32_t globalJ = colStart + j;
                        Element u = gmA.GetValue((colStart + kk) * ldA + globalJ);
                        for (uint32_t i = kk + 1; i < M; ++i) {
                            Element l = gmA.GetValue((colStart + i) * ldA + globalK);
                            Element cur = gmA.GetValue((colStart + i) * ldA + globalJ);
                            gmA.SetValue((colStart + i) * ldA + globalJ, cur - l * u);
                        }
                    }

                    if (nColsAligned >= 8) {
                        uint32_t uOff = gmOff + skip;
                        uint32_t colOff = colStart + kk + 1 + skip;
                        AscendC::DataCacheCleanAndInvalid<
                            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);

                        AscendC::DataCopy(ubU, gmA[uOff], nColsAligned);
                        for (uint32_t i = kk + 1; i < M; ++i) {
                            Element l = gmA.GetValue((colStart + i) * ldA + globalK);
                            AscendC::DataCopy(ubT, gmA[(colStart + i) * ldA + colOff], nColsAligned);
                            AscendC::PipeBarrier<PIPE_MTE2>();
                            AscendC::Muls(ubTmp, ubU, l, nColsAligned);
                            AscendC::Sub(ubT, ubT, ubTmp, nColsAligned);
                            AscendC::DataCopy(gmA[(colStart + i) * ldA + colOff], ubT, nColsAligned);
                            AscendC::PipeBarrier<PIPE_MTE3>();
                        }
                    }
                    // Scalar suffix
                    uint32_t tailStart = kk + 1 + skip + nColsAligned;
                    for (uint32_t j = tailStart; j < K; ++j) {
                        uint32_t globalJ = colStart + j;
                        Element u = gmA.GetValue((colStart + kk) * ldA + globalJ);
                        for (uint32_t i = kk + 1; i < M; ++i) {
                            Element l = gmA.GetValue((colStart + i) * ldA + globalK);
                            Element cur = gmA.GetValue((colStart + i) * ldA + globalJ);
                            gmA.SetValue((colStart + i) * ldA + globalJ, cur - l * u);
                        }
                    }
                } else if (nCols >= 4) {
                    // Col-wise vector: L col→UB(scalar), per-col: target→UB→Muls+Sub→GM
                    uint32_t sl = M - kk - 1;
                    for (uint32_t ii = 0; ii < sl; ++ii)
                        ubU.SetValue(ii, gmA.GetValue((colStart + kk + 1 + ii) * ldA + globalK));
                    for (uint32_t j = kk + 1; j < K; ++j) {
                        uint32_t globalJ = colStart + j;
                        Element u = gmA.GetValue((colStart + kk) * ldA + globalJ);
                        for (uint32_t ii = 0; ii < sl; ++ii)
                            ubT.SetValue(ii, gmA.GetValue((colStart + kk + 1 + ii) * ldA + globalJ));
                        AscendC::Muls(ubTmp, ubU, u, sl);
                        AscendC::Sub(ubT, ubT, ubTmp, sl);
                        for (uint32_t ii = 0; ii < sl; ++ii)
                            gmA.SetValue((colStart + kk + 1 + ii) * ldA + globalJ, ubT.GetValue(ii));
                    }
                } else {
                    for (uint32_t j = kk + 1; j < K; ++j) {
                        uint32_t globalJ = colStart + j;
                        Element u = gmA.GetValue((colStart + kk) * ldA + globalJ);
                        for (uint32_t i = kk + 1; i < M; ++i) {
                            Element l = gmA.GetValue((colStart + i) * ldA + globalK);
                            Element cur = gmA.GetValue((colStart + i) * ldA + globalJ);
                            gmA.SetValue((colStart + i) * ldA + globalJ, cur - l * u);
                        }
                    }
                }
                AscendC::DataCacheCleanAndInvalid<
                    Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);
            }
        }
    }

    // ============================================
    // 行交换（laswp）—— DataCopy 向量化版本
    //
    // 使用 UB 缓冲区进行整行 DataCopy 交换，比逐元素 GetValue/SetValue
    // 效率高很多。要求：UB 地址 32B 对齐、搬运长度 32B 对齐。
    // 尾部不足 32B 的部分回退到标量交换。
    // ============================================
    CATLASS_DEVICE
    void ApplyRowSwaps(
        AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<int32_t>& gmIpiv, uint32_t pivotStart,
        uint32_t numPivots, uint32_t colStart, uint32_t colEnd, uint32_t ldA)
    {
        constexpr uint32_t kAlignBytes = 32;
        constexpr uint32_t kAlignElems = kAlignBytes / sizeof(Element);

        uint32_t nCols = colEnd - colStart;
        uint32_t alignedCols = nCols / kAlignElems * kAlignElems;

        // 分配两个 32B 对齐的 UB 缓冲区用于行交换
        // bufLen * sizeof(Element) 作为 GetBufferByByte 的偏移量，
        // bufLen 对齐到 kAlignElems 保证偏移量是 32B 的倍数
        Arch::Resource<ArchTag> resource;
        uint32_t bufLen = (alignedCols > 0) ? alignedCols : kAlignElems;
        auto ubRow1 = resource.ubBuf.template GetBufferByByte<Element>(0);
        auto ubRow2 = resource.ubBuf.template GetBufferByByte<Element>(bufLen * sizeof(Element));

        // 刷新 gmA 缓存，避免读到其他 core 写入的脏数据
        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);

        for (uint32_t kk = 0; kk < numPivots; ++kk) {
            uint32_t pivotIdx = static_cast<uint32_t>(gmIpiv.GetValue(pivotStart + kk));
            uint32_t rowK = pivotStart + kk;
            if (pivotIdx == rowK) {
                continue;
            }

            uint32_t offK = rowK * ldA + colStart;
            uint32_t offP = pivotIdx * ldA + colStart;

            // 对齐部分：DataCopy 整行搬运
            if (alignedCols > 0) {
                AscendC::DataCopy(ubRow1, gmA[offK], alignedCols);
                AscendC::DataCopy(ubRow2, gmA[offP], alignedCols);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(gmA[offK], ubRow2, alignedCols);
                AscendC::DataCopy(gmA[offP], ubRow1, alignedCols);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // 尾部不足 32B 的部分回退到标量交换
            for (uint32_t j = alignedCols; j < nCols; ++j) {
                Element tmp = gmA.GetValue(offK + j);
                gmA.SetValue(offK + j, gmA.GetValue(offP + j));
                gmA.SetValue(offP + j, tmp);
            }
        }
    }

    // ============================================
    // ComputeInvLDiagGETRF: 求面板 L_diag (nb×nb 单位下三角) 的逆 → gmGemmTemp
    //   invL 紧凑存储在 gmGemmTemp(0:nb, 0:nb), stride=nb
    // ============================================
    CATLASS_DEVICE
    void ComputeInvLDiagGETRF(
        AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<Element>& gmInvLBuf, uint32_t k, uint32_t nb,
        uint32_t ldA)
    {
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < nb; ++j)
                gmInvLBuf.SetValue(i * nb + j, static_cast<Element>(0));

        for (uint32_t i = 0; i < nb; ++i)
            gmInvLBuf.SetValue(i * nb + i, static_cast<Element>(1));

        for (uint32_t j = 0; j < nb; ++j) {
            for (uint32_t i = j + 1; i < nb; ++i) {
                Element sum = -gmA.GetValue((k + i) * ldA + (k + j));
                for (uint32_t m = j + 1; m < i; ++m) {
                    sum -= gmA.GetValue((k + i) * ldA + (k + m)) * gmInvLBuf.GetValue(m * nb + j);
                }
                gmInvLBuf.SetValue(i * nb + j, sum);
            }
        }
    }

    // ============================================
    // TrsmLeftLowerUnit GEMM: gmA(k:k+nb, k+nb:) = invL × gmA_right (AIC)
    // ============================================
    CATLASS_DEVICE
    void TrsmLeftLowerUnitGemm(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmA,
        AscendC::GlobalTensor<Element>& gmInvLBuf, uint32_t k, uint32_t nb, uint32_t nRhs, uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{nb, nRhs, nb}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            int64_t offA = static_cast<int64_t>(mStart) * nb;
            auto tensorA = tla::MakeTensor(gmInvLBuf[offA], RmLayout(mSize, nb, nb), Arch::PositionGM{});
            int64_t offB = static_cast<int64_t>(k) * ldA + (k + nb + nStart);
            auto tensorB = tla::MakeTensor(gmA[offB], RmLayout(nb, nSize, ldA), Arch::PositionGM{});
            int64_t offD = static_cast<int64_t>(k + mStart) * ldA + (k + nb + nStart);
            auto tensorD = tla::MakeTensor(gmA[offD], RmLayout(mSize, nSize, ldA), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, nb});
        }
    }

    // ============================================
    // Schur GEMM: D = L21 × U12 → workspace (AIC)
    // ============================================
    CATLASS_DEVICE
    void SchurGemmToWorkspace(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<Element>& gmWork,
        uint32_t k, uint32_t nb, uint32_t M, uint32_t K, uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        Layout layoutFull(ldA, ldA);
        BlockScheduler scheduler(GemmCoord{M, M, K}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            int64_t offA = layoutFull.GetOffset(MatrixCoord(k + nb + mStart, k));
            auto tensorA = tla::MakeTensor(gmA[offA], RmLayout(mSize, K, ldA), Arch::PositionGM{});
            int64_t offB = layoutFull.GetOffset(MatrixCoord(k, k + nb + nStart));
            auto tensorB = tla::MakeTensor(gmA[offB], RmLayout(K, nSize, ldA), Arch::PositionGM{});
            int64_t offD = static_cast<int64_t>(mStart) * M + nStart;
            auto tensorD = tla::MakeTensor(gmWork[offD], RmLayout(mSize, nSize, M), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, K});
        }
    }

    // ============================================
    // TRSMtemp GEMM: gmInvUDense(0:j, j:j+actNb) = U × invU_diag (AIC, direct write)
    //   U       : gmWork(0:j, j:j+actNb), stride N
    //   invU_diag: gmInvUDense(j:j+actNb, j:j+actNb), stride N, 下三角=0
    //   D        : gmInvUDense(0:j, j:j+actNb), stride N
    // ============================================
    CATLASS_DEVICE
    void TrsmTempGemmToWorkspace(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmWork,
        AscendC::GlobalTensor<Element>& gmInvUDense, uint32_t j, uint32_t actNb, uint32_t N, uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{j, actNb, actNb}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            // A = U tile: gmWork rows mStart.., cols j..
            int64_t offA = static_cast<int64_t>(mStart) * ldA + j;
            auto tensorA = tla::MakeTensor(gmWork[offA], RmLayout(mSize, actNb, ldA), Arch::PositionGM{});
            // B = invU_diag tile: gmInvUDense rows j.., cols j+nStart..
            int64_t offB = static_cast<int64_t>(j) * ldA + (j + nStart);
            auto tensorB = tla::MakeTensor(gmInvUDense[offB], RmLayout(actNb, nSize, ldA), Arch::PositionGM{});
            // D = gmInvUDense rows mStart.., cols j+nStart..
            int64_t offD = static_cast<int64_t>(mStart) * ldA + (j + nStart);
            auto tensorD = tla::MakeTensor(gmInvUDense[offD], RmLayout(mSize, nSize, ldA), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, actNb});
        }
    }

    // ============================================
    // TRTRI GEMM: D = temp × U_block → gmGemmTemp (AIC)
    //   temp   : gmInvUDense(0:prevRows, j:j+actNb), stride N
    //   U_block: gmWork(j:j+actNb, k:k+kActNb), stride N
    //   D      : gmGemmTemp, stride kActNb
    // ============================================
    CATLASS_DEVICE
    void InvertUpperTriGemmToWorkspace(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmInvUDense,
        AscendC::GlobalTensor<Element>& gmWork, AscendC::GlobalTensor<Element>& gmGemmTemp, uint32_t j, uint32_t actNb,
        uint32_t prevRows, uint32_t k, uint32_t kActNb, uint32_t N, uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{prevRows, kActNb, actNb}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            // A = temp tile: gmInvUDense rows mStart.., cols j..
            int64_t offA = static_cast<int64_t>(mStart) * ldA + j;
            auto tensorA = tla::MakeTensor(gmInvUDense[offA], RmLayout(mSize, actNb, ldA), Arch::PositionGM{});
            // B = U_block tile: gmWork rows j.., cols k+nStart..
            int64_t offB = static_cast<int64_t>(j) * ldA + (k + nStart);
            auto tensorB = tla::MakeTensor(gmWork[offB], RmLayout(actNb, nSize, ldA), Arch::PositionGM{});
            // D tile: gmGemmTemp[mStart*kActNb + nStart], stride kActNb
            int64_t offD = static_cast<int64_t>(mStart) * kActNb + nStart;
            auto tensorD = tla::MakeTensor(gmGemmTemp[offD], RmLayout(mSize, nSize, kActNb), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, actNb});
        }
    }

    // ============================================
    // ApplyLInverse GEMM: D = X_right × L_block → gmGemmTemp (AIC)
    // ============================================
    CATLASS_DEVICE
    void ApplyLInverseGemmToWorkspace(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<Element>& gmLU,
        AscendC::GlobalTensor<Element>& gmGemmTemp, uint32_t colStart, uint32_t actualNb, uint32_t K, uint32_t N,
        uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{N, actualNb, K}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            int64_t offA = static_cast<int64_t>(mStart) * ldA + (colStart + actualNb);
            auto tensorA = tla::MakeTensor(gmA[offA], RmLayout(mSize, K, ldA), Arch::PositionGM{});
            int64_t offB = static_cast<int64_t>(colStart + actualNb) * ldA + (colStart + nStart);
            auto tensorB = tla::MakeTensor(gmLU[offB], RmLayout(K, nSize, ldA), Arch::PositionGM{});
            int64_t offD = static_cast<int64_t>(mStart) * actualNb + nStart;
            auto tensorD = tla::MakeTensor(gmGemmTemp[offD], RmLayout(mSize, nSize, actualNb), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, K});
        }
    }

    // ============================================
    // 拷贝严格下三角: gmA(下三角) → gmDst(稠密), 对角和上三角 = 0
    // ============================================
    CATLASS_DEVICE
    void CopyStrictLowerToDense(
        AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<Element>& gmDst, uint32_t N, uint32_t ldA)
    {
        constexpr uint32_t kAlignBytes = 32;
        constexpr uint32_t kAlignElems = kAlignBytes / sizeof(Element);

        // N 不对齐 → gmA[i*N] 和 gmDst[i*N] 均无法保证 32B，回退标量
        if (N % kAlignElems != 0) {
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < i; ++j) {
                    gmDst.SetValue(i * N + j, gmA.GetValue(i * ldA + j));
                }
                for (uint32_t j = i; j < N; ++j) {
                    gmDst.SetValue(i * N + j, static_cast<Element>(0));
                }
            }
            return;
        }

        // 对齐快速路径：读端 gmA[i*N]、写端 gmDst[i*N]、搬运长度均 32B 对齐
        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);

        Arch::Resource<ArchTag> resource;
        auto ubRow = resource.ubBuf.template GetBufferByByte<Element>(0);

        for (uint32_t i = 0; i < N; ++i) {
            // Step 1: UB 行清零（向量广播指令）
            AscendC::Duplicate(ubRow, static_cast<Element>(0), static_cast<int32_t>(N));

            // Step 2: DataCopy 搬运 gmA 第 i 行前 i_aligned 个元素
            //         gmA[i*N] 对齐、i_aligned 对齐，三端满足
            uint32_t iAligned = i / kAlignElems * kAlignElems;
            if (iAligned > 0) {
                AscendC::DataCopy(ubRow, gmA[i * N], iAligned);
            }
            for (uint32_t j = iAligned; j < i; ++j) {
                ubRow.SetValue(j, gmA.GetValue(i * N + j));
            }

            // Step 3: 整行写回 gmDst（gmDst[i*N] 对齐、N 对齐）
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gmDst[i * N], ubRow, N);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ============================================
    // Copy U upper triangle from gmA → dense gmInvUDense (lower = 0)
    // ============================================
    CATLASS_DEVICE
    void CopyUpperToDense(
        AscendC::GlobalTensor<Element>& gmA, AscendC::GlobalTensor<Element>& gmDst, uint32_t N, uint32_t ldA)
    {
        constexpr uint32_t kAlignBytes = 32;
        constexpr uint32_t kAlignElems = kAlignBytes / sizeof(Element);

        if (N % kAlignElems != 0) {
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < i; ++j) {
                    gmDst.SetValue(i * N + j, static_cast<Element>(0));
                }
                for (uint32_t j = i; j < N; ++j) {
                    gmDst.SetValue(i * N + j, gmA.GetValue(i * ldA + j));
                }
            }
            return;
        }

        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmA);

        Arch::Resource<ArchTag> resource;
        auto ubRow = resource.ubBuf.template GetBufferByByte<Element>(0);

        for (uint32_t i = 0; i < N; ++i) {
            // Step 1: UB 行清零（向量广播指令）
            AscendC::Duplicate(ubRow, static_cast<Element>(0), static_cast<int32_t>(N));

            // Step 2: 拷贝上三角 [i, N) — 起始列 i 可能不对齐
            //         标量前缀补齐到 kAlignElems 边界，主体 DataCopy，尾部标量
            uint32_t colStart = i;
            uint32_t firstAligned = (colStart + kAlignElems - 1) / kAlignElems * kAlignElems;

            // 标量前缀：补齐到 kAlignElems 边界，使读端 GM 地址和 UB 地址均 32B 对齐
            uint32_t leadEnd = (firstAligned < N) ? firstAligned : N;
            for (uint32_t j = colStart; j < leadEnd; ++j) {
                ubRow.SetValue(j, gmA.GetValue(i * N + j));
            }

            // DataCopy 主体：N 和 firstAligned 均为 kAlignElems 的倍数，
            // N - firstAligned 也是倍数，不存在尾部
            if (firstAligned < N) {
                AscendC::DataCopy(ubRow[firstAligned], gmA[i * N + firstAligned], N - firstAligned);
            }

            // Step 3: 整行写回 gmDst
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gmDst[i * N], ubRow, N);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ============================================
    // ① TRTRI diag: 原地求 NB×NB 上三角矩阵的逆
    // ============================================
    CATLASS_DEVICE
    void TRTRIdiag(AscendC::GlobalTensor<Element>& gmDense, uint32_t j, uint32_t actNb, uint32_t N)
    {
        for (int jj = static_cast<int>(j + actNb) - 1; jj >= static_cast<int>(j); --jj) {
            uint32_t col = static_cast<uint32_t>(jj);
            Element diag = gmDense.GetValue(col * N + col);
            if (diag != static_cast<Element>(0)) {
                gmDense.SetValue(col * N + col, static_cast<Element>(1) / diag);
            }
            for (int ii = jj - 1; ii >= static_cast<int>(j); --ii) {
                uint32_t row = static_cast<uint32_t>(ii);
                Element sum = static_cast<Element>(0);
                for (uint32_t kk = row + 1; kk <= col; ++kk) {
                    sum += gmDense.GetValue(row * N + kk) * gmDense.GetValue(kk * N + col);
                }
                Element u_ii = gmDense.GetValue(row * N + row);
                if (u_ii != static_cast<Element>(0)) {
                    gmDense.SetValue(row * N + col, -sum / u_ii);
                }
            }
        }
    }

    // ============================================
    // DTRMM GEMM: gmGemmTemp = invU(0:j,0:j) × temp(0:j, j:j+actNb) (AIC)
    //   invU_top 在 gmInvUDense 中为稠密格式（下三角=0），
    //   当稠密 GEMM 算，~50% Cube 利用率，但消除串行 O(N³/6)
    //   输出到 gmGemmTemp，由 AIV core 0 取反拷贝到 gmInvUDense
    // ============================================
    CATLASS_DEVICE
    void DtrmmGemmToWorkspace(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmInvUDense,
        AscendC::GlobalTensor<Element>& gmGemmTemp, uint32_t j, uint32_t actNb, uint32_t N, uint32_t ldA)
    {
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{j, actNb, j}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            // A = invU(0:j, 0:j): j×j dense, stride = ldA
            int64_t offA = static_cast<int64_t>(mStart) * ldA;
            auto tensorA = tla::MakeTensor(gmInvUDense[offA], RmLayout(mSize, j, ldA), Arch::PositionGM{});
            // B = temp(0:j, j:j+actNb): j×actNb dense, stride = ldA
            int64_t offB = static_cast<int64_t>(0) * ldA + (j + nStart);
            auto tensorB = tla::MakeTensor(gmInvUDense[offB], RmLayout(j, nSize, ldA), Arch::PositionGM{});
            // D: gmGemmTemp, stride = actNb
            int64_t offD = static_cast<int64_t>(mStart) * actNb + nStart;
            auto tensorD = tla::MakeTensor(gmGemmTemp[offD], RmLayout(mSize, nSize, actNb), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, j});
        }
    }

    // ============================================
    // NegateCopy: gmDense(0:j, j:j+actNb) = -gmGemmTemp
    // ============================================
    CATLASS_DEVICE
    void NegateCopyDtrmm(
        AscendC::GlobalTensor<Element>& gmDense, AscendC::GlobalTensor<Element>& gmGemmTemp, uint32_t j, uint32_t actNb,
        uint32_t N)
    {
        constexpr uint32_t kAlignBytes = 32;
        constexpr uint32_t kAlignElems = kAlignBytes / sizeof(Element);

        if (N % kAlignElems != 0) {
            for (uint32_t i = 0; i < j; ++i)
                for (uint32_t kk = 0; kk < actNb; ++kk)
                    gmDense.SetValue(i * N + (j + kk), -gmGemmTemp.GetValue(i * actNb + kk));
            return;
        }

        AscendC::DataCacheCleanAndInvalid<
            Element, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmGemmTemp);

        // 预计算写端对齐参数（j 对所有行相同）
        uint32_t const jEnd = j + actNb;
        uint32_t const jAlignedStart = (j + kAlignElems - 1) / kAlignElems * kAlignElems;
        uint32_t const jAlignedEnd = jEnd / kAlignElems * kAlignElems;
        uint32_t const leadCnt = (jAlignedStart < jEnd) ? (jAlignedStart - j) : actNb;
        uint32_t const bulkCnt = (jAlignedEnd > jAlignedStart) ? (jAlignedEnd - jAlignedStart) : 0;
        uint32_t const tailCnt = jEnd - jAlignedEnd;

        Arch::Resource<ArchTag> resource;
        auto ubRow = resource.ubBuf.template GetBufferByByte<Element>(0);

        for (uint32_t i = 0; i < j; ++i) {
            // 前缀：GM 标量读 → 取反 → GM 标量写（最多 kAlignElems-1 个）
            for (uint32_t k = 0; k < leadCnt; ++k) {
                gmDense.SetValue(i * N + j + k, -gmGemmTemp.GetValue(i * actNb + k));
            }

            // 主体：DataCopy 读 → UB 内 Muls 取反 → DataCopy 写（两端对齐）
            if (bulkCnt > 0) {
                AscendC::DataCopy(ubRow, gmGemmTemp[i * actNb + leadCnt], bulkCnt);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Muls(ubRow, ubRow, static_cast<Element>(-1), bulkCnt);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(gmDense[i * N + jAlignedStart], ubRow, bulkCnt);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // 尾部：GM 标量读 → 取反 → GM 标量写（最多 kAlignElems-1 个）
            for (uint32_t k = 0; k < tailCnt; ++k) {
                gmDense.SetValue(i * N + jAlignedEnd + k, -gmGemmTemp.GetValue(i * actNb + leadCnt + bulkCnt + k));
            }
        }
    }

    // ============================================
    // ComputeInvLDiag: 求 NB×NB 单位下三角 L_diag 的逆 → gmGemmTemp
    //   invL 存放在 gmGemmTemp 右侧: cols NB-nb..NB-1, 避开 GEMM 输出区 (0:nb)
    // ============================================
    CATLASS_DEVICE
    void ComputeInvLDiag(
        AscendC::GlobalTensor<Element>& gmLU, AscendC::GlobalTensor<Element>& gmInvLBuf, uint32_t colStart, uint32_t nb,
        uint32_t N, uint32_t ldA, uint32_t NB)
    {
        uint32_t invLColOff = NB - nb;

        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < nb; ++j)
                gmInvLBuf.SetValue(i * NB + invLColOff + j, static_cast<Element>(0));

        for (uint32_t i = 0; i < nb; ++i)
            gmInvLBuf.SetValue(i * NB + invLColOff + i, static_cast<Element>(1));

        for (uint32_t j = 0; j < nb; ++j) {
            for (uint32_t i = j + 1; i < nb; ++i) {
                Element sum = -gmLU.GetValue((colStart + i) * ldA + (colStart + j));
                for (uint32_t k = j + 1; k < i; ++k) {
                    sum -= gmLU.GetValue((colStart + i) * ldA + (colStart + k)) *
                           gmInvLBuf.GetValue(k * NB + invLColOff + j);
                }
                gmInvLBuf.SetValue(i * NB + invLColOff + j, sum);
            }
        }
    }

    // ============================================
    // ApplyLIntraBlock GEMM: gmA(:, colStart:colStart+nb) = gmA × invL (AIC)
    // ============================================
    CATLASS_DEVICE
    void ApplyLIntraBlockGemm(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<Element>& gmA,
        AscendC::GlobalTensor<Element>& gmInvLBuf, uint32_t colStart, uint32_t nb, uint32_t N, uint32_t ldA,
        uint32_t NB)
    {
        uint32_t invLColOff = NB - nb;
        BlockMmad blockMmad(resource);
        BlockScheduler scheduler(GemmCoord{N, nb, nb}, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t tileIdx = blockIdx; tileIdx < coreLoops; tileIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(tileIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mSize = actualShape.m();
            uint32_t nSize = actualShape.n();
            int64_t offA = static_cast<int64_t>(mStart) * ldA + colStart;
            auto tensorA = tla::MakeTensor(gmA[offA], RmLayout(mSize, nb, ldA), Arch::PositionGM{});
            int64_t offB = static_cast<int64_t>(0) * NB + (invLColOff + nStart);
            auto tensorB = tla::MakeTensor(gmInvLBuf[offB], RmLayout(nb, nSize, NB), Arch::PositionGM{});
            int64_t offD = static_cast<int64_t>(mStart) * ldA + (colStart + nStart);
            auto tensorD = tla::MakeTensor(gmA[offD], RmLayout(mSize, nSize, ldA), Arch::PositionGM{});
            blockMmad(tensorA, tensorB, tensorD, GemmCoord{mSize, nSize, nb});
        }
    }

    // ============================================
    // SwapColumns
    // ============================================
    CATLASS_DEVICE
    void SwapColumns(AscendC::GlobalTensor<Element>& gmA, uint32_t col1, uint32_t col2, uint32_t rows, uint32_t ldA)
    {
        for (uint32_t i = 0; i < rows; ++i) {
            Element tmp = gmA.GetValue(i * ldA + col1);
            gmA.SetValue(i * ldA + col1, gmA.GetValue(i * ldA + col2));
            gmA.SetValue(i * ldA + col2, tmp);
        }
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATRIX_INVERSE_HPP
