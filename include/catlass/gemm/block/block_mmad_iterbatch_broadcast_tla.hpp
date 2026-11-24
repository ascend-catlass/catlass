/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_ITERBATCH_BROADCAST_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_ITERBATCH_BROADCAST_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Block-level IterBatch broadcast batched matmul (phase-1 small-matrix path).
//
// Semantics: C[b] = cast(op(A[ba]) * op(B[bb]) + Bias(b)), where ba = (batchA == 1) ? 0 : b and
// bb = (batchB == 1) ? 0 : b. A broadcast operand (batch stride 0) is copied from GM exactly once
// per L1 stage and its L0 bricks are shared by all batches of an L0 sub-group.
//
// Two-level batch organization:
//   L1 group  : iterBatchL1 (B1) batches of full A/B matrices (+ bias rows) resident in one L1 stage.
//   L0 group  : iterBatchL0 (B0) batches sharing one L0C stage, each with an independent accumulator.
// Constraint: 1 <= B0 <= B1 (checked by ResourcePlan), B0 does not need to divide B1.
//
// Event protocol (mirrors block_mmad_pingpong_tla.hpp):
//   id space  : event ids live in one 8-entry register file PER WAITER PIPE, not per HardEvent:
//                each HardEvent::X_Y pair indexes the file of the waiting pipe Y. M_MTE1 and
//                MTE2_MTE1 share the MTE1 file; MTE1_M and FIX_M share the M file; MTE1_MTE2
//                (MTE2 file) and M_FIX (FIX file) are exclusive. Numeric overlap is only a
//                hazard inside one file: the same value on different waiter pipes is safe
//                (MMAD_READY_EVENT == btEvent and l0CEventList[i] == l0AEventList[i] below
//                rely on this).
//   L1 stage   : ONE stage-level event shared by the A/B/bias regions. Wait<MTE1_MTE2> before
//                the GM->L1 reload of the stage, Set<MTE2_MTE1> after the whole group load,
//                Wait<MTE2_MTE1> before the first L1->L0 read, Set<MTE1_MTE2> after the last
//                read. Its id sits right after the L0-side M_MTE1 ids: M_MTE1(id) and
//                MTE2_MTE1(id) index the same MTE1-pipe event registers, so any numeric overlap
//                between the compute-side and prefetch-side families leaks set state across
//                grid-stride group iterations once the pipeline is saturated (B1=256 fills
//                100% of L1 and the MTE2 prefetch runs parallel with MTE1 compute).
//   L0A/L0B   : one event per stage covering all batch slots of the stage,
//                Wait<M_MTE1> before L1->L0 reload, Set<M_MTE1> after the consuming mmads.
//   L0C       : Wait<FIX_M> before the first mmad of a sub-group, MTE1_M set/wait around the
//                mmads on MMAD_READY_EVENT (M file; numerically equal to btEvent, which lives
//                in the separate MTE1 file; the l0C ids share the M file and stay below it),
//                Set<M_FIX>/Wait<M_FIX> before fixpipe, Set<FIX_M> after fixpipe.
//   BiasTable : Wait<M_MTE1> before copy L1->BT, Set<M_MTE1> after the consuming mmad.
template <
    class ArchTag_, bool USE_HF32_MODE_, uint32_t L0C_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadIterBatchBroadcast<ArchTag_, USE_HF32_MODE_, L0C_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_>, L1TileShape_,
    L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy =
        MmadIterBatchBroadcast<ArchTag_, USE_HF32_MODE_, L0C_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L0 tile bytes. M/N of the L0 tile equal the L1 tile, only K is bricked on L0.
    static constexpr uint32_t L0A_TILE_BYTES = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_BYTES = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_BYTES = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // One bias row in L1, aligned to 32B so per-row offsets stay aligned. Shared by the
    // stage layout below and ResourcePlan's bias budget.
    static constexpr uint32_t BIAS_ROW_BYTES =
        HAS_BIAS ? static_cast<uint32_t>(RoundUp(L1_TILE_N * sizeof(ElementBias), BYTE_PER_C0)) : 0;

    static constexpr int32_t MMAD_READY_EVENT = static_cast<int32_t>(L0A_STAGES + L0B_STAGES);

    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // Check HF32_MODE
    static_assert(
        !USE_HF32_MODE || (USE_HF32_MODE && std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>),
        "HF32 MODE only supports in float!");

    // Check LayoutC
    static_assert(
        tla::detail::isRowMajor<LayoutC>::value ||
            ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
              std::is_same_v<ElementC, float>) &&
             tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // The M/N axes are not tiled between L1 and L0, only K is bricked.
    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    // A runtime L0 tile layout (m, kActual) must fit into one L0A/L0B slot of L0*_TILE_BYTES.
    static_assert(L0_TILE_M % C0_NUM_PER_FRACTAL == 0, "L0TileShape::M must be fractal(16) aligned.");
    static_assert(L0_TILE_N % C0_NUM_PER_FRACTAL == 0, "L0TileShape::N must be fractal(16) aligned.");
    static_assert(L1_TILE_M % C0_NUM_PER_FRACTAL == 0, "L1TileShape::M must be fractal(16) aligned.");
    static_assert(L1_TILE_N % C0_NUM_PER_FRACTAL == 0, "L1TileShape::N must be fractal(16) aligned.");
    static_assert(L0_TILE_K * sizeof(ElementA) % BYTE_PER_C0 == 0, "L0TileShape::K of A must be 32B aligned.");
    static_assert(L0_TILE_K * sizeof(ElementB) % BYTE_PER_C0 == 0, "L0TileShape::K of B must be 32B aligned.");

    // Single L0 tiles must fit; the multi-batch budget is checked at runtime by the resource plan.
    static_assert(L0A_STAGES * L0A_TILE_BYTES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_STAGES * L0B_TILE_BYTES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_STAGES * L0C_TILE_BYTES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(L0C_TILE_BYTES <= ArchTag::L0C_SIZE, "One L0C tile exceeding the L0C space!");

    // Cross-group prefetch needs a second L1 stage to overlap with the compute of the current group.
    static_assert(L1_STAGES >= 2, "MmadIterBatchBroadcast requires L1_STAGES >= 2 for group prefetch!");

    // Event id budgets. On the MTE1 pipe the compute-side M_MTE1 family (l0A/l0B/bt, ids
    // 0 .. L0A+L0B) and the prefetch-side MTE2_MTE1 family (one id per L1 stage, ids
    // L0A+L0B+1 .. L0A+L0B+L1_STAGES) index the SAME hardware event registers and must stay
    // disjoint: any numeric overlap lets a stale set from one family (e.g. the never-consumed
    // btEvent set of the no-bias path) satisfy a wait of the other once the MTE2 prefetch runs
    // parallel with MTE1 compute (B1=256 fills 100% of L1 from grid-stride iteration 2
    // onwards). On the M pipe the MTE1_M mmad-ready id and the FIX_M l0C ids index the same
    // registers, so the l0C ids must stay below MMAD_READY_EVENT; M_FIX targets the exclusive
    // FIX-pipe file and adds no constraint.
    static_assert(
        L0A_STAGES + L0B_STAGES + 1 + L1_STAGES <= 8, "L0+L1 stages exceed the supported range of EVENT(0~7)");
    static_assert(L0A_STAGES + L0B_STAGES + 2 <= 8, "L0 stages exceed the supported range of EVENT_ID(0~7)");
    static_assert(L0C_STAGES <= 8, "L0C stages exceed the supported range of EVENT_ID(0~7)");
    static_assert(L0C_STAGES <= L0A_STAGES + L0B_STAGES, "L0C ids would alias the mmad-ready id on the M pipe");

    // Runtime capacity planner for the two-level batch split above. Every input derives
    // from this block's own tile / stage / layout configuration, so it lives here; the
    // kernel only uses it on the host side as a factory of Result.
    //
    // Budget formulas (a broadcast side physically degrades to one copy):
    //   L1  = L1_STAGES  * (nA1 * slotA + nB1 * slotB + slotBias)
    //   L0A = L0A_STAGES * nA0 * l0ATileBytes
    //   L0B = L0B_STAGES * nB0 * l0BTileBytes
    //   L0C = L0C_STAGES * B0 * l0CTileBytes    // output never shares, always B0
    // where
    //   nA1 = broadcastA ? 1 : B1,  nB1 = broadcastB ? 1 : B1
    //   nA0 = broadcastA ? 1 : B0,  nB0 = broadcastB ? 1 : B0
    //
    // iterBatchL1 == 0 means auto (derive the maximum B1), iterBatchL0 == 0 means auto
    // (derive the maximum B0 bounded by B1). An explicit request that does not fit is
    // rejected with the failing budget instead of being clamped.
    struct ResourcePlan {
        // Status of a plan. When the user explicitly requests iterBatchL1/iterBatchL0 and
        // the plan fails, the status plus the budget in Result must be reported to the
        // caller (reject, never silently fall back or clamp).
        enum class Status : uint32_t
        {
            OK = 0,
            // Phase-1 small-matrix path: the full A/B matrices of one group must reside in L1,
            // i.e. m <= L1_TILE_M, n <= L1_TILE_N, k <= L1_TILE_K.
            SHAPE_EXCEEDS_L1_TILE = 1,
            // Runtime bias mode is not None but ElementBias is void.
            INVALID_BIAS_MODE = 2,
            // Explicit spec violates 1 <= iterBatchL0 <= iterBatchL1.
            INVALID_BATCH_SPEC = 3,
            // Explicit iterBatchL1 does not fit into L1 (budget details are filled).
            L1_OVERFLOW = 4,
            L0A_OVERFLOW = 5,
            L0B_OVERFLOW = 6,
            L0C_OVERFLOW = 7,
            // Even one batch of A/B(+bias) can not reside in L1. The maximum batch count
            // must NOT be clamped to 1 to mask this failure.
            SINGLE_MATRIX_L1_TOO_LARGE = 8,
            // A single L0 tile already exceeds L0A/L0B/L0C.
            L0_TILE_TOO_LARGE = 9,
        };

        // Result of ResourcePlan::Make(). All byte fields are accumulated from individually
        // aligned offsets (aligned A/B/bias regions are summed separately).
        struct Result {
            bool valid{false};
            Status status{Status::OK};
            uint32_t iterBatchL1{0}; // B1: batches of one L1 resident group
            uint32_t iterBatchL0{0}; // B0: batches of one L0 sub-group
            // Budget details in bytes.
            uint64_t slotA{0};    // aligned bytes of one full A matrix stored in L1
            uint64_t slotB{0};    // aligned bytes of one full B matrix stored in L1
            uint64_t slotBias{0}; // aligned bytes of the bias region of one L1 stage
            uint64_t l1Used{0};   // L1_STAGES * (nA1 * slotA + nB1 * slotB + slotBias)
            uint64_t l0aUsed{0};  // L0A_STAGES * nA0 * l0ATileBytes
            uint64_t l0bUsed{0};  // L0B_STAGES * nB0 * l0BTileBytes
            uint64_t l0cUsed{0};  // L0C_STAGES * B0 * l0CTileBytes, output never shares
        };

        // Tail guard reserved at the very end of the L1 budget.
        static constexpr uint64_t L1_TAIL_GUARD_BYTES = BYTE_PER_FRACTAL;

        // Bias bytes of one L1 stage for the given group batch count.
        CATLASS_HOST_DEVICE
        static constexpr uint64_t BiasSlotBytes(uint32_t iterBatchL1, BiasMode biasMode)
        {
            if constexpr (HAS_BIAS) {
                switch (biasMode) {
                    case BiasMode::PerBatchN:
                        return static_cast<uint64_t>(iterBatchL1) * BIAS_ROW_BYTES;
                    case BiasMode::SharedN:
                        return BIAS_ROW_BYTES;
                    case BiasMode::None:
                    default:
                        return 0;
                }
            } else {
                return 0;
            }
        }

        // Aligned bytes of a matrix (rows x cols) stored in L1 with the given fractal tag.
        // The two dimensions are aligned independently (zN: rows->16/fractal, cols->C0;
        // nZ: rows->C0, cols->16) and then multiplied, so the result is the sum of the
        // aligned fractal blocks, not the raw row*col bytes.
        template <class Element, class FractalTag>
        CATLASS_HOST_DEVICE static constexpr uint64_t AlignedMatrixBytes(uint32_t rows, uint32_t cols)
        {
            constexpr uint32_t eleNumPerC0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
            if constexpr (std::is_same_v<FractalTag, layout::zN>) {
                constexpr uint32_t rowsAlign = C0_NUM_PER_FRACTAL;
                constexpr uint32_t colsAlign = eleNumPerC0;
                return static_cast<uint64_t>(RoundUp(rows, rowsAlign)) * RoundUp(cols, colsAlign) * sizeof(Element);
            } else {
                constexpr uint32_t rowsAlign = eleNumPerC0;
                constexpr uint32_t colsAlign = C0_NUM_PER_FRACTAL;
                return static_cast<uint64_t>(RoundUp(rows, rowsAlign)) * RoundUp(cols, colsAlign) * sizeof(Element);
            }
        }

        // Aligned bytes of the full A / B matrix of one batch in L1 (fractal tag per
        // helper::L1FractalLayout, the rule shared with helper::L1ATypeSelector/L1BTypeSelector).
        CATLASS_HOST_DEVICE
        static uint64_t SlotABytes(uint32_t m, uint32_t k)
        {
            using FractalTagA = typename helper::L1FractalLayout<typename TileCopy::LayoutTagA>::type;
            return AlignedMatrixBytes<ElementA, FractalTagA>(m, k);
        }

        CATLASS_HOST_DEVICE
        static uint64_t SlotBBytes(uint32_t k, uint32_t n)
        {
            using FractalTagB = typename helper::L1FractalLayout<typename TileCopy::LayoutTagB>::type;
            return AlignedMatrixBytes<ElementB, FractalTagB>(k, n);
        }

        CATLASS_HOST_DEVICE
        static uint64_t L1Budget(
            uint32_t iterBatchL1, uint64_t slotA, uint64_t slotB, uint64_t slotBias, bool broadcastA, bool broadcastB)
        {
            uint64_t nA1 = broadcastA ? 1 : iterBatchL1;
            uint64_t nB1 = broadcastB ? 1 : iterBatchL1;
            return L1_STAGES * (nA1 * slotA + nB1 * slotB + slotBias);
        }

        CATLASS_HOST_DEVICE
        static uint64_t L0ABudget(uint32_t iterBatchL0, bool broadcastA)
        {
            uint64_t nA0 = broadcastA ? 1 : iterBatchL0;
            return L0A_STAGES * nA0 * L0A_TILE_BYTES;
        }

        CATLASS_HOST_DEVICE
        static uint64_t L0BBudget(uint32_t iterBatchL0, bool broadcastB)
        {
            uint64_t nB0 = broadcastB ? 1 : iterBatchL0;
            return L0B_STAGES * nB0 * L0B_TILE_BYTES;
        }

        CATLASS_HOST_DEVICE
        static uint64_t L0CBudget(uint32_t iterBatchL0)
        {
            // Output batches never share an L0C accumulator.
            return L0C_STAGES * static_cast<uint64_t>(iterBatchL0) * L0C_TILE_BYTES;
        }

        // Derive the maximum iterBatchL1 (B1) under the L1 budget.
        CATLASS_HOST_DEVICE
        static int64_t MaxL1Batch(
            uint32_t batchCount, uint64_t slotA, uint64_t slotB, BiasMode biasMode, bool broadcastA, bool broadcastB)
        {
            const uint64_t perStage = (ArchTag::L1_SIZE - L1_TAIL_GUARD_BYTES) / L1_STAGES;
            const bool perBatchBias = (biasMode == BiasMode::PerBatchN);
            const uint64_t sharedBias = (biasMode == BiasMode::SharedN) ? BIAS_ROW_BYTES : 0;
            const uint64_t biasPerBatch = perBatchBias ? BIAS_ROW_BYTES : 0;

            if (broadcastA && broadcastB) {
                if (slotA + slotB + sharedBias > perStage) {
                    return -1;
                }
                return perBatchBias ? static_cast<int64_t>((perStage - slotA - slotB) / biasPerBatch) :
                                      static_cast<int64_t>(batchCount);
            }
            if (broadcastA) {
                if (slotA + sharedBias > perStage) {
                    return -1;
                }
                return static_cast<int64_t>((perStage - slotA - sharedBias) / (slotB + biasPerBatch));
            }
            if (broadcastB) {
                if (slotB + sharedBias > perStage) {
                    return -1;
                }
                return static_cast<int64_t>((perStage - slotB - sharedBias) / (slotA + biasPerBatch));
            }
            if (sharedBias > perStage) {
                return -1;
            }
            return static_cast<int64_t>((perStage - sharedBias) / (slotA + slotB + biasPerBatch));
        }

        // Derive the maximum iterBatchL0 (B0) under the L0A/L0B/L0C budgets. Returns -1
        // when even a single L0 tile does not fit (the block's static_asserts already
        // reject that at compile time).
        CATLASS_HOST_DEVICE
        static int64_t MaxL0Batch(bool broadcastA, bool broadcastB)
        {
            if (L0A_STAGES * L0A_TILE_BYTES > ArchTag::L0A_SIZE) {
                return -1;
            }
            if (L0B_STAGES * L0B_TILE_BYTES > ArchTag::L0B_SIZE) {
                return -1;
            }
            if (L0C_STAGES * L0C_TILE_BYTES > ArchTag::L0C_SIZE) {
                return -1;
            }

            int64_t cap = INT64_MAX;
            if (!broadcastA) {
                cap = Min(cap, static_cast<int64_t>(ArchTag::L0A_SIZE / (L0A_STAGES * L0A_TILE_BYTES)));
            }
            if (!broadcastB) {
                cap = Min(cap, static_cast<int64_t>(ArchTag::L0B_SIZE / (L0B_STAGES * L0B_TILE_BYTES)));
            }
            cap = Min(cap, static_cast<int64_t>(ArchTag::L0C_SIZE / (L0C_STAGES * L0C_TILE_BYTES)));
            return cap;
        }

        // Build the plan.
        //   batchCount: number of output batches.
        //   m, n, k:    problem shape (must fit into L1TileShape, phase-1 small-matrix path).
        //   broadcastA / broadcastB: whether A / B is broadcast on the batch dim (batchA==1 / batchB==1).
        //   iterBatchL1 / iterBatchL0: 0 means auto.
        //   coreNum: number of cores the kernel is launched with. When iterBatchL1 is auto, B1 is
        //            additionally bounded by ceil(batchCount / coreNum) so that the grid-stride
        //            loop yields at least coreNum L1 groups (avoid a single-core collapse when
        //            the capacity bound is vacuous, e.g. both operands broadcast without bias).
        //            0 disables this bound (capacity-only). An explicit iterBatchL1 is never overridden.
        CATLASS_HOST_DEVICE
        static Result Make(
            uint32_t batchCount, uint32_t m, uint32_t n, uint32_t k, bool broadcastA, bool broadcastB,
            BiasMode biasMode, uint32_t iterBatchL1, uint32_t iterBatchL0, uint32_t coreNum = 0)
        {
            Result result;
            result.status = Status::OK;

            // 1. Bias mode must be supported by the compile-time ElementBias.
            if (!HAS_BIAS && biasMode != BiasMode::None) {
                result.status = Status::INVALID_BIAS_MODE;
                return result;
            }

            // 2. Phase-1 small-matrix path: full A/B of one group must reside in L1.
            if (m > L1_TILE_M || n > L1_TILE_N || k > L1_TILE_K) {
                result.status = Status::SHAPE_EXCEEDS_L1_TILE;
                return result;
            }

            // 3. Aligned slot sizes.
            const uint64_t slotA = SlotABytes(m, k);
            const uint64_t slotB = SlotBBytes(k, n);
            result.slotA = slotA;
            result.slotB = slotB;

            // 4. Resolve iterBatchL1 (B1).
            uint32_t b1 = iterBatchL1;
            if (b1 == 0) {
                int64_t cap = MaxL1Batch(batchCount, slotA, slotB, biasMode, broadcastA, broadcastB);
                if (cap < 1) {
                    result.status = Status::SINGLE_MATRIX_L1_TOO_LARGE;
                    result.slotBias = BiasSlotBytes(1, biasMode);
                    result.l1Used = L1Budget(1, slotA, slotB, result.slotBias, broadcastA, broadcastB);
                    return result;
                }
                // Core-count bound: with auto B1, keep enough L1 groups for all cores of a
                // grid-stride launch (groupCount >= coreNum). Without it, a vacuous capacity
                // bound (both operands broadcast, no per-batch bias) collapses B1 to batchCount
                // and only one core works.
                int64_t parallelCap = (coreNum > 0) ? static_cast<int64_t>((batchCount + coreNum - 1) / coreNum) :
                                                      static_cast<int64_t>(batchCount);
                b1 = static_cast<uint32_t>(Min(cap, Min(static_cast<int64_t>(batchCount), parallelCap)));
                if (b1 == 0) {
                    // batchCount == 0, nothing to do but keep the spec valid.
                    b1 = 1;
                }
            }

            result.slotBias = BiasSlotBytes(b1, biasMode);
            result.l1Used = L1Budget(b1, slotA, slotB, result.slotBias, broadcastA, broadcastB);
            if (result.l1Used > ArchTag::L1_SIZE - L1_TAIL_GUARD_BYTES) {
                result.status = Status::L1_OVERFLOW;
                result.iterBatchL1 = b1;
                return result;
            }

            // 5. Resolve iterBatchL0 (B0), 1 <= B0 <= B1.
            int64_t l0Cap = MaxL0Batch(broadcastA, broadcastB);
            if (l0Cap < 1) {
                result.status = Status::L0_TILE_TOO_LARGE;
                result.iterBatchL1 = b1;
                result.iterBatchL0 = 1;
                result.l0aUsed = L0ABudget(1, broadcastA);
                result.l0bUsed = L0BBudget(1, broadcastB);
                result.l0cUsed = L0CBudget(1);
                return result;
            }

            uint32_t b0 = iterBatchL0;
            if (b0 == 0) {
                b0 = static_cast<uint32_t>(Min(l0Cap, static_cast<int64_t>(b1)));
            } else if (b0 > b1) {
                result.status = Status::INVALID_BATCH_SPEC;
                result.iterBatchL1 = b1;
                result.iterBatchL0 = b0;
                return result;
            }

            result.l0aUsed = L0ABudget(b0, broadcastA);
            result.l0bUsed = L0BBudget(b0, broadcastB);
            result.l0cUsed = L0CBudget(b0);
            if (result.l0aUsed > ArchTag::L0A_SIZE) {
                result.status = Status::L0A_OVERFLOW;
                result.iterBatchL1 = b1;
                result.iterBatchL0 = b0;
                return result;
            }
            if (result.l0bUsed > ArchTag::L0B_SIZE) {
                result.status = Status::L0B_OVERFLOW;
                result.iterBatchL1 = b1;
                result.iterBatchL0 = b0;
                return result;
            }
            if (result.l0cUsed > ArchTag::L0C_SIZE) {
                result.status = Status::L0C_OVERFLOW;
                result.iterBatchL1 = b1;
                result.iterBatchL0 = b0;
                return result;
            }

            result.iterBatchL1 = b1;
            result.iterBatchL0 = b0;
            result.valid = true;
            return result;
        }
    };

    /// Construct. The plan must come from ResourcePlan::Make with valid == true.
    CATLASS_DEVICE
    BlockMmadTla(
        Arch::Resource<ArchTag>& resource, typename ResourcePlan::Result const& plan, bool broadcastA, bool broadcastB,
        BiasMode biasMode)
    {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(true);
            } else {
                AscendC::SetHF32Mode(false);
            }
            resource_ = &resource;
            b1Max_ = plan.iterBatchL1;
            // The host ResourcePlan::Make guarantees iterBatchL0 >= 1 for a valid plan.
            // Clamp here so a launch with a tampered plan cannot divide by zero on
            // device (the L0 sub-group loop computes (actualBatch + b0 - 1) / b0).
            b0_ = (plan.iterBatchL0 != 0) ? plan.iterBatchL0 : 1;
            broadcastA_ = broadcastA;
            broadcastB_ = broadcastB;
            biasMode_ = biasMode;
            biasActive_ = HAS_BIAS && (biasMode != BiasMode::None);
            nA1Max_ = broadcastA_ ? 1 : b1Max_;
            nB1Max_ = broadcastB_ ? 1 : b1Max_;
            biasRowsMax_ = biasActive_ ? ((biasMode_ == BiasMode::PerBatchN) ? b1Max_ : 1) : 0;
            slotA_ = static_cast<uint32_t>(plan.slotA);
            slotB_ = static_cast<uint32_t>(plan.slotB);
            stageBytes_ = nA1Max_ * slotA_ + nB1Max_ * slotB_ + biasRowsMax_ * BIAS_ROW_BYTES;

            for (uint32_t s = 0; s < L1_STAGES; ++s) {
                // One stage-level event per L1 stage (the A/B/bias regions share it), placed
                // right after the L0-side M_MTE1 ids: M_MTE1(id) and MTE2_MTE1(id) index the
                // same MTE1-pipe event registers, so the compute-side and prefetch-side event
                // families must never overlap numerically.
                l1EventList[s] = static_cast<int32_t>(L0A_STAGES + L0B_STAGES + 1 + s);
                // L1 stages are initially free for GM->L1 loads.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[s]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; ++i) {
                l0AEventList[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; ++i) {
                l0BEventList[i] = static_cast<int32_t>(L0A_STAGES + i);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            btEvent = static_cast<int32_t>(L0A_STAGES + L0B_STAGES);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(btEvent);
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                l0CEventList[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
            if constexpr (HAS_BIAS) {
                l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(false);
            }
            for (uint32_t s = 0; s < L1_STAGES; ++s) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[s]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(btEvent);
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    /// Perform one L1 group of the IterBatch broadcast batched matmul.
    ///   tensorA/tensorB/tensorC: GM tensors of the FIRST batch of this group (kernel-side group offset).
    ///   tensorBias:             GM tensor of the first bias row of this group (vector layout), or EmptyClass.
    ///   nextTensorA/nextTensorB/nextTensorBias: GM tensors of the FIRST batch of the NEXT group on this
    ///                           core, built by the kernel with the correct grid-stride batch base.
    ///                           Read only when hasNextBlock, so placeholders are otherwise accepted.
    ///   actualShape:            problem shape (m, n, k), must fit L1TileShape (phase-1 path).
    ///   actualL1Batch:          batch count of this group (<= plan.iterBatchL1).
    ///   nextActualL1Batch:      batch count of the next group on this core (used when hasNextBlock).
    ///   isFirstBlock:           true when this is the first group of the core (no prefetch available).
    ///   hasNextBlock:           true when this core has another group after the current one.
    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(
        TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape, uint32_t actualL1Batch,
        uint32_t nextActualL1Batch, bool isFirstBlock, bool hasNextBlock, TensorA& nextTensorA, TensorB& nextTensorB,
        TensorBias const& nextTensorBias = {}, TensorBias const& tensorBias = {})
    {
        const uint32_t m = actualShape.m();
        const uint32_t n = actualShape.n();
        const uint32_t k = actualShape.k();

        const uint32_t nextStage = (l1StageId + 1 < L1_STAGES) ? (l1StageId + 1) : 0;
        if (isFirstBlock) {
            // No prefetch happened before the first group, load it now.
            LoadGroup(l1StageId, tensorA, tensorB, tensorBias, actualL1Batch, m, n, k);
        }
        if (hasNextBlock) {
            // Prefetch the next group into the other L1 stage, overlapping with the compute below.
            // The next-group GM tensors come from the kernel layer: under grid-stride the next
            // group on this core is scheduled at the scheduler's (loopIdx + blockNum), whose
            // batch base is coreNum groups away, NOT actualL1Batch batches away from the current one.
            if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                LoadGroup(nextStage, nextTensorA, nextTensorB, nextTensorBias, nextActualL1Batch, m, n, k);
            } else {
                EmptyClass noBias;
                LoadGroup(nextStage, nextTensorA, nextTensorB, noBias, nextActualL1Batch, m, n, k);
            }
        }
        ComputeGroup(l1StageId, tensorA, tensorB, tensorC, tensorBias, actualL1Batch, m, n, k);
        l1StageId = nextStage;
    }

protected:
    /// Batch GM->L1 copy helpers (private to this block).
    ///
    /// Phase 1 implementation: a device loop of the single-matrix copy functors of
    /// TileCopy. The loop can later be replaced by a hardware batch copy without
    /// changing the call sites. The helpers NEVER re-read the same matrix from GM:
    /// a broadcast operand (batchStride == 0) is copied exactly once, a non-broadcast
    /// operand is copied once per resident batch. Destination slots are addressed by
    /// byte offsets derived from the aligned per-slot size of the capacity plan.

    /// Copy `batchCount` rank-2 matrices (rows x cols) from GM into consecutive L1
    /// slots of one L1 stage.
    ///   tensorSrc:   GM tensor of the first matrix of the group (already batch-offset).
    ///   batchStride: batch stride in elements; 0 means broadcast (copy only the first).
    ///   l1Offset/slotBytes: byte offset of the first slot and the aligned byte size
    ///                of one slot.
    ///   l1Layout:    compile-time L1 layout of one slot.
    ///   copy:        single-matrix copy functor, e.g. CopyGmToL1A (dst, src).
    template <class TensorSrc, class L1Layout, class Copy>
    CATLASS_DEVICE void CopyBatchToL1(
        TensorSrc const& tensorSrc, int64_t batchStride, uint32_t batchCount, uint32_t rows, uint32_t cols,
        uint32_t l1Offset, uint32_t slotBytes, L1Layout const& l1Layout, Copy& copy)
    {
        static_assert(TensorSrc::rank == 2, "CopyBatchToL1 expects a rank-2 source tensor.");
        using ElementDst = typename TensorSrc::Element; // GM->L1 copies keep the element type
        for (uint32_t i = 0; i < batchCount; ++i) {
            auto tensorSrcI = tla::MakeTensor(
                tensorSrc.data()[static_cast<uint64_t>(i) * static_cast<uint64_t>(batchStride)], tensorSrc.layout(),
                Arch::PositionGM{});
            auto tensorTileI = GetTile(tensorSrcI, tla::MakeCoord(0U, 0U), tla::MakeShape(rows, cols));
            auto l1Slot = resource_->l1Buf.template GetBufferByByte<ElementDst>(l1Offset + slotBytes * i);
            auto tensorL1I = tla::MakeTensor(l1Slot, l1Layout, Arch::PositionL1{});
            copy(tensorL1I, tensorTileI);
        }
    }

    /// Copy `rowCount` rank-1 rows (bias vectors) from GM into consecutive L1 rows.
    ///   tensorSrc:   GM tensor of the first row (shape must be the full row length).
    ///   batchStride: row stride in elements; 0 means a single shared row.
    ///   rowBytes:    aligned byte size of one L1 row.
    template <class TensorSrc, class L1Layout, class Copy>
    CATLASS_DEVICE void CopyBatchRowsToL1(
        TensorSrc const& tensorSrc, int64_t batchStride, uint32_t rowCount, uint32_t rowElems, uint32_t l1Offset,
        uint32_t rowBytes, L1Layout const& l1Layout, Copy& copy)
    {
        static_assert(TensorSrc::rank == 1, "CopyBatchRowsToL1 expects a rank-1 source tensor.");
        using ElementDst = typename TensorSrc::Element; // GM->L1 copies keep the element type
        for (uint32_t i = 0; i < rowCount; ++i) {
            auto tensorSrcI = tla::MakeTensor(
                tensorSrc.data()[static_cast<uint64_t>(i) * static_cast<uint64_t>(batchStride)], tensorSrc.layout(),
                Arch::PositionGM{});
            auto tensorTileI = GetTile(tensorSrcI, tla::MakeCoord(0U), tla::MakeShape(rowElems));
            auto l1Slot = resource_->l1Buf.template GetBufferByByte<ElementDst>(l1Offset + rowBytes * i);
            auto tensorL1I = tla::MakeTensor(l1Slot, l1Layout, Arch::PositionL1{});
            copy(tensorL1I, tensorTileI);
        }
    }

    /// Load one whole L1 group (A slots + B slots + bias rows) from GM into the given L1 stage.
    /// A broadcast operand is copied exactly once (batchStride == 0).
    template <class TensorA, class TensorB, class TensorBias>
    CATLASS_DEVICE void LoadGroup(
        uint32_t stage, TensorA const& tensorA, TensorB const& tensorB, TensorBias const& tensorBias,
        uint32_t actualBatch, uint32_t m, uint32_t n, uint32_t k)
    {
        const uint32_t nA1 = broadcastA_ ? 1 : actualBatch;
        const uint32_t nB1 = broadcastB_ ? 1 : actualBatch;
        const uint32_t base = stage * stageBytes_;
        const int64_t strideA = broadcastA_ ? 0 : static_cast<int64_t>(m) * k;
        const int64_t strideB = broadcastB_ ? 0 : static_cast<int64_t>(k) * n;

        // The whole stage (A slots + B slots + bias rows) is released as one unit, wait once.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[stage]);
        {
            using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
            CopyGmToL1A copyGmToL1A;
            auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
            CopyBatchToL1(tensorA, strideA, nA1, m, k, base, slotA_, layoutL1A, copyGmToL1A);
        }
        {
            using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
            CopyGmToL1B copyGmToL1B;
            auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
            CopyBatchToL1(tensorB, strideB, nB1, k, n, base + nA1Max_ * slotA_, slotB_, layoutL1B, copyGmToL1B);
        }
        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
            if (biasActive_) {
                using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
                CopyGmToL1Bias copyGmToL1Bias;
                const uint32_t biasRows = (biasMode_ == BiasMode::PerBatchN) ? actualBatch : 1;
                const int64_t biasStride = (biasMode_ == BiasMode::PerBatchN) ? static_cast<int64_t>(n) : 0;
                CopyBatchRowsToL1(
                    tensorBias, biasStride, biasRows, n, base + nA1Max_ * slotA_ + nB1Max_ * slotB_, BIAS_ROW_BYTES,
                    L1BIAS_LAYOUT, copyGmToL1Bias);
            }
        }
        // The whole group load of this stage is queued on the MTE2 pipe (ordered after all the
        // copies above), the stage content is ready for the L1->L0 reads of ComputeGroup.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[stage]);
    }

    /// Compute one L1 group: iterate L0 sub-groups, brick K on L0, mmad per batch, fixpipe per batch.
    template <class TensorA, class TensorB, class TensorC, class TensorBias>
    CATLASS_DEVICE void ComputeGroup(
        uint32_t stage, TensorA const& tensorA, TensorB const& tensorB, TensorC& tensorC, TensorBias const& tensorBias,
        uint32_t actualBatch, uint32_t m, uint32_t n, uint32_t k)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        const int64_t strideC = static_cast<int64_t>(m) * n;
        const uint32_t nA0Max = broadcastA_ ? 1 : b0_;
        const uint32_t nB0Max = broadcastB_ ? 1 : b0_;
        const uint32_t base = stage * stageBytes_;
        const uint32_t biasBase = base + nA1Max_ * slotA_ + nB1Max_ * slotB_;
        const uint32_t kBricks = CeilDiv<L0_TILE_K>(k);
        const uint32_t subGroupCount = (actualBatch + b0_ - 1) / b0_;
        const uint32_t biasSlot = (biasMode_ == BiasMode::PerBatchN) ? 1 : 0; // 1: per batch, 0: shared

        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        for (uint32_t g = 0; g < subGroupCount; ++g) {
            const uint32_t batchBase = g * b0_;
            const uint32_t actualB0 = (batchBase + b0_ <= actualBatch) ? b0_ : (actualBatch - batchBase);
            const uint32_t sC = l0CListId;
            const bool firstSub = (g == 0);
            const bool lastSub = (g == subGroupCount - 1);
            const uint32_t nA0 = broadcastA_ ? 1 : actualB0;
            const uint32_t nB0 = broadcastB_ ? 1 : actualB0;

            // L0C slots of this stage must be free (previous fixpipe finished).
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[sC]);

            for (uint32_t kBrick = 0; kBrick < kBricks; ++kBrick) {
                const uint32_t kActual = (kBrick < kBricks - 1) ? L0_TILE_K : (k - kBrick * L0_TILE_K);
                const uint32_t sA = l0AListId;
                const uint32_t sB = l0BListId;
                const bool firstBrick = (kBrick == 0);
                const bool lastBrick = (kBrick == kBricks - 1);
                const bool initC = firstBrick;

                // ---- Load L0A bricks of this k brick for all batch slots of the sub-group ----
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[sA]);
                if (firstSub && firstBrick) {
                    // First L1 read of this stage (the A/B bricks and the bias rows below share
                    // one stage-level event), wait for the whole GM->L1 group load.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[stage]);
                }
                for (uint32_t j = 0; j < nA0; ++j) {
                    auto l0ASlot =
                        resource_->l0ABuf.template GetBufferByByte<ElementA>((sA * nA0Max + j) * L0A_TILE_BYTES);
                    auto tensorL0A = tla::MakeTensor(
                        l0ASlot, tla::MakeLayout<ElementA, LayoutTagL0A>(m, kActual), Arch::PositionL0A{});
                    const uint32_t l1SlotIdx = broadcastA_ ? 0 : (batchBase + j);
                    auto l1ASlot = resource_->l1Buf.template GetBufferByByte<ElementA>(base + l1SlotIdx * slotA_);
                    auto tensorL1A = tla::MakeTensor(l1ASlot, layoutL1A, Arch::PositionL1{});
                    auto tensorTileL1A = GetTileA(tensorL1A, 0, kBrick * L0_TILE_K, m, kActual);
                    copyL1ToL0A(tensorL0A, tensorTileL1A);
                }

                // ---- Load L0B bricks of this k brick for all batch slots of the sub-group ----
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[sB]);
                for (uint32_t j = 0; j < nB0; ++j) {
                    auto l0BSlot =
                        resource_->l0BBuf.template GetBufferByByte<ElementB>((sB * nB0Max + j) * L0B_TILE_BYTES);
                    auto tensorL0B = tla::MakeTensor(
                        l0BSlot, tla::MakeLayout<ElementB, LayoutTagL0B>(kActual, n), Arch::PositionL0B{});
                    const uint32_t l1SlotIdx = broadcastB_ ? 0 : (batchBase + j);
                    auto l1BSlot = resource_->l1Buf.template GetBufferByByte<ElementB>(
                        base + nA1Max_ * slotA_ + l1SlotIdx * slotB_);
                    auto tensorL1B = tla::MakeTensor(l1BSlot, layoutL1B, Arch::PositionL1{});
                    auto tensorTileL1B =
                        GetTile(tensorL1B, tla::MakeCoord(kBrick * L0_TILE_K, 0U), tla::MakeShape(kActual, n));
                    copyL1ToL0B(tensorL0B, tensorTileL1B);
                }

                // ---- mmad per batch of the sub-group ----
                // Inputs of this k brick are ready (queued on MTE1 after the L0A/L0B loads above).
                // MTE1_M indexes the M-pipe register file, so this set/wait pair cannot alias the
                // MTE1-file l0A/l0B/bt state even though the id numerically equals btEvent; the
                // only other M-file user is the l0C id family, which stays below this id.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_READY_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_READY_EVENT);
                for (uint32_t b = 0; b < actualB0; ++b) {
                    const uint32_t slotAIdx = broadcastA_ ? 0 : b;
                    const uint32_t slotBIdx = broadcastB_ ? 0 : b;
                    auto l0ASlot =
                        resource_->l0ABuf.template GetBufferByByte<ElementA>((sA * nA0Max + slotAIdx) * L0A_TILE_BYTES);
                    auto tensorL0A = tla::MakeTensor(
                        l0ASlot, tla::MakeLayout<ElementA, LayoutTagL0A>(m, kActual), Arch::PositionL0A{});
                    auto l0BSlot =
                        resource_->l0BBuf.template GetBufferByByte<ElementB>((sB * nB0Max + slotBIdx) * L0B_TILE_BYTES);
                    auto tensorL0B = tla::MakeTensor(
                        l0BSlot, tla::MakeLayout<ElementB, LayoutTagL0B>(kActual, n), Arch::PositionL0B{});
                    auto l0CSlot =
                        resource_->l0CBuf.template GetBufferByByte<ElementAccumulator>((sC * b0_ + b) * L0C_TILE_BYTES);
                    auto tensorL0C = tla::MakeTensor(l0CSlot, layoutL0C, Arch::PositionL0C{});

                    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                        if (biasActive_ && initC) {
                            // Load the bias row of this batch from L1 into the bias table, then mmad with bias.
                            // (The GM->L1 bias load is already covered by the stage-level
                            // l1EventList wait at the first L1 read of this stage.)
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(btEvent);
                            const uint32_t groupBatch = batchBase + b;
                            const uint32_t biasRowIdx = biasSlot * groupBatch;
                            auto l1BiasRow = resource_->l1Buf.template GetBufferByByte<ElementBias>(
                                biasBase + biasRowIdx * BIAS_ROW_BYTES);
                            auto tensorL1BiasRow = tla::MakeTensor(l1BiasRow, L1BIAS_LAYOUT, Arch::PositionL1{});
                            auto tensorTileL1Bias = GetTile(tensorL1BiasRow, tla::MakeCoord(0U), tla::MakeShape(n));
                            copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                            // Bias table content is ready (queued on MTE1 after copyL1ToBT).
                            // Same dedicated MMAD_READY_EVENT id as the L0A/L0B readiness pair:
                            // set immediately precedes wait within the same instruction stream,
                            // so a single id stays balanced.
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_READY_EVENT);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_READY_EVENT);
                            tileMmad(tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, m, n, kActual, initC);
                            // Bias table consumed by the mmad above.
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(btEvent);
                        } else {
                            tileMmad(tensorL0C, tensorL0A, tensorL0B, m, n, kActual, initC);
                        }
                    } else {
                        tileMmad(tensorL0C, tensorL0A, tensorL0B, m, n, kActual, initC);
                    }
                }
                // All mmads consuming the L0A/L0B bricks of this k brick are queued before this point.
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[sA]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[sB]);
                if (lastSub && lastBrick) {
                    // Last L1 read of this stage is queued (the A/B brick copies above plus the
                    // bias-row BT reads of the b-loop), the whole stage can be reloaded from GM.
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[stage]);
                }
                l0AListId = (sA + 1 < L0A_STAGES) ? (sA + 1) : 0;
                l0BListId = (sB + 1 < L0B_STAGES) ? (sB + 1) : 0;
            }

            // ---- fixpipe per batch of the sub-group ----
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[sC]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[sC]);
            for (uint32_t b = 0; b < actualB0; ++b) {
                const uint32_t groupBatch = batchBase + b;
                auto tensorCb =
                    tla::MakeTensor(tensorC.data()[groupBatch * strideC], tensorC.layout(), Arch::PositionGM{});
                auto l0CSlot =
                    resource_->l0CBuf.template GetBufferByByte<ElementAccumulator>((sC * b0_ + b) * L0C_TILE_BYTES);
                auto tensorL0C = tla::MakeTensor(l0CSlot, layoutL0C, Arch::PositionL0C{});
                copyL0CToDst(tensorCb, tensorL0C);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[sC]);
            l0CListId = (sC + 1 < L0C_STAGES) ? (sC + 1) : 0;
        }
    }

    template <class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA& tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
    }

    // Runtime resource plan (all derived from ResourcePlan::Result).
    Arch::Resource<ArchTag>* resource_{nullptr};
    uint32_t b1Max_{0}; // B1: batches of one L1 group
    uint32_t b0_{0};    // B0: batches of one L0 sub-group
    bool broadcastA_{false};
    bool broadcastB_{false};
    BiasMode biasMode_{BiasMode::None};
    bool biasActive_{false};
    uint32_t nA1Max_{1}; // A slots per L1 stage
    uint32_t nB1Max_{1}; // B slots per L1 stage
    uint32_t biasRowsMax_{0};
    uint32_t slotA_{0};      // aligned bytes of one A slot in L1
    uint32_t slotB_{0};      // aligned bytes of one B slot in L1
    uint32_t stageBytes_{0}; // bytes of one L1 stage

    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1EventList[L1_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t btEvent{0};

    // The id of current stage
    uint32_t l1StageId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_ITERBATCH_BROADCAST_TLA_HPP
