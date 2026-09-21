/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_CATLASS_KERNEL_JIT_H
#define OPTEST_CATLASS_KERNEL_JIT_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <acl/acl.h>

#include "catlass/gemm_coord.hpp"

namespace CatlassKernel {

struct TParamsBase {
    Catlass::GemmCoord l1TileShape;
    Catlass::GemmCoord l0TileShape;
    Catlass::GemmCoord swizzle;
};

/**
 * @brief Compile-time JIT parameters shared by numbered matmul-family examples.
 *        Uses map-based storage for extensibility.
 */
struct TParams : TParamsBase {
    std::unordered_map<std::string, aclDataType> element;
    std::unordered_map<std::string, bool> transpose;
    std::unordered_map<std::string, bool> useNz;
    std::unordered_map<std::string, bool> flag; ///< Generic compile-time boolean options

    aclDataType elem(const std::string& k, aclDataType def = ACL_FLOAT16) const
    {
        auto it = element.find(k);
        return it != element.end() ? it->second : def;
    }
    bool trans(const std::string& k, bool def = false) const
    {
        auto it = transpose.find(k);
        return it != transpose.end() ? it->second : def;
    }
    bool nz(const std::string& k, bool def = false) const
    {
        auto it = useNz.find(k);
        return it != useNz.end() ? it->second : def;
    }
    bool flagOn(const std::string& k, bool def = false) const
    {
        auto it = flag.find(k);
        return it != flag.end() ? it->second : def;
    }
};

/**
 * @brief Runtime matrix parameters shared by numbered matmul-family examples.
 */
struct MatmulParams {
    uint32_t m = 1;                   ///< M dimension.
    uint32_t n = 1;                   ///< N dimension.
    uint32_t k = 1;                   ///< K dimension.
    uint32_t batch = 1;               ///< Batch dimension for batched variants.
    std::vector<uint8_t*> inputAddr;  ///< Input buffer addresses.
    std::vector<uint8_t*> outputAddr; ///< Output buffer addresses.
    uint32_t x1QuantMode = 0;         ///< Example 57 x1 quant mode (QuantMode enum value).
    uint32_t x2QuantMode = 0;         ///< Example 57 x2 quant mode (QuantMode enum value).
    bool hasQuantBias = false;        ///< Example 57 optional bias flag.
};

/**
 * @brief Runtime parameters for experimental/matmul/ascend950_matmul_gather_scatter.
 */
struct MatmulGatherScatterParams : public MatmulParams {
    uint32_t j = 1; ///< Number of selected rows and logical matmul M dimension.
};

struct MatmulEvgParams : public MatmulParams {
    std::string evgType; ///< EVG postprocess mode (example 64 matmul_evg).
    float negativeSlope = 1;
};

/**
 * @brief Runtime parameters for grouped matmul examples.
 */
struct GroupedMatmulParams : public MatmulParams {
    enum class SliceMode : uint32_t
    {
        M = 0,
        K = 1,
        N = 2
    };

    SliceMode sliceMode = SliceMode::M; ///< Grouped matmul slice dimension.
};

/**
 * @brief Runtime parameters for strided batched matmul examples.
 */
struct StridedBatchedMatmulParams : public MatmulParams {
    int64_t strideA = -1; ///< Stride between batches for A (elements).
    int64_t strideB = -1; ///< Stride between batches for B (elements).
    int64_t strideC = -1; ///< Stride between batches for C (elements).
    int64_t lda = -1;     ///< Leading dimension of A.
    int64_t ldb = -1;     ///< Leading dimension of B.
    int64_t ldc = -1;     ///< Leading dimension of C.
};

/**
 * @brief Runtime parameters for GEMM examples with alpha and beta scaling.
 */
struct GemmParams : public MatmulParams {
    float alpha = 1.0f; ///< Alpha scale in D = alpha * A * B + beta * C.
    float beta = 0.0f;  ///< Beta scale in D = alpha * A * B + beta * C.
};

/**
 * @brief Runtime parameters for Ascend950 SYRK: D = alpha * X @ X^T + beta * Y.
 *
 * ``inputAddr[0]`` = X, ``inputAddr[1]`` = Y, ``outputAddr[0]`` = D; ``m`` must equal ``n``.
 * For batched inputs use ``batch > 1`` with contiguous ``(B, M, K)`` / ``(B, M, M)`` tensors.
 */
struct SyrkParams : public MatmulParams {
    float alpha = 1.0f; ///< Alpha scale in D = alpha * X @ X^T + beta * Y.
    float beta = 0.0f;  ///< Beta scale in D = alpha * X @ X^T + beta * Y.
};

/**
 * @brief Runtime parameters for example 76_trmm.
 */
struct TrmmParams : public MatmulParams {
    uint32_t side = 0;  ///< 0: left triangular input, 1: right triangular input.
    uint32_t uplo = 0;  ///< 0: lower triangular, 1: upper triangular.
    uint32_t trans = 0; ///< 0: no transpose, 1: transpose triangular input.
    uint32_t diag = 0;  ///< 0: non-unit diagonal. Unit diagonal is not supported by the kernel.
    float alpha = 1.0f; ///< Output scaling factor.
};

/**
 * @brief Runtime parameters for example 75_symm.
 */
struct SymmParams : public MatmulParams {
    uint32_t side = 0;  ///< 0: left symmetric input, 1: right symmetric input.
    uint32_t uplo = 0;  ///< 0: lower triangle is stored, 1: upper triangle is stored.
    float alpha = 1.0f; ///< Reserved for API parity. The current kernel supports only alpha=1.
};

/**
 * @brief Runtime parameters for example 61_ascend950_svd_quant_matmul.
 */
struct SvdQuantMatmulParams : public MatmulParams {
    uint32_t r = 32;   ///< SVD rank.
    float qmax = 8.0f; ///< SmoothQuant qmax.
};

extern "C" {

/**
 * @brief JIT interface for example 00_basic_matmul.
 */
void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 01_batched_matmul.
 */
void BatchedMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 02_grouped_matmul_slice_m.
 */
void GroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 03_matmul_add.
 */
void MatmulAdd(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 04_padding_matmul.
 */
void PaddingMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 05_grouped_matmul_slice_k.
 */
void GroupedMatmulSliceK(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 06_optimized_matmul.
 */
void OptimizedMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 07_grouped_matmul_slice_m_per_token_dequant_moe.
 */
void GroupedMatmulSliceMPerTokenDequantMoe(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 08_grouped_matmul.
 */
void GroupedMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 09_splitk_matmul.
 */
void SplitkMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 10_grouped_matmul_slice_m_per_token_dequant.
 */
void GroupedMatmulSliceMPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 11_grouped_matmul_slice_k_per_token_dequant.
 */
void GroupedMatmulSliceKPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 12_quant_matmul.
 */
void QuantMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 13_basic_matmul_tla.
 */
void BasicMatmulTLA(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 14_optimized_matmul_tla.
 */
void OptimizedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 15_gemm.
 */
void Gemm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params);

/**
 * @brief Reserved JIT interface for example 16_group_gemm.
 */
void GroupGemm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params);

/**
 * @brief Reserved JIT interface for example 17_gemv_aiv.
 */
void GemvAIV(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params);

/**
 * @brief Reserved JIT interface for example 18_gemv_aic.
 */
void GemvAIC(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params);

/**
 * @brief JIT interface for example 20_matmul_bias.
 */
void MatmulBias(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 21_basic_matmul_preload_zN.
 */
void BasicMatmulPreloadZN(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 22_padding_splitk_matmul.
 */
void PaddingSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 25_matmul_full_loadA.
 */
void MatmulFullLoadA(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 26_matmul_relu.
 */
void MatmulRelu(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 27_matmul_gelu.
 */
void MatmulGelu(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 28_matmul_silu.
 */
void MatmulSilu(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 30_w8a16_matmul.
 */
void W8A16Matmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 31_small_matmul.
 */
void SmallMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 32_w4a8_matmul.
 */
void W4A8Matmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 34_single_core_splitk_matmul.
 */
void SingleCoreSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 35_w4a8_grouped_matmul_msd.
 */
void W4A8GroupedMatmulMSD(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 36_w4a8_matmul_msd.
 */
void W4A8MatmulMSD(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 37_streamk_matmul.
 */
void StreamkMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 38_w4a4_matmul_per_token_per_channel_dequant.
 */
void W4A4MatmulPerTokenPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 41_sparse_matmul_tla.
 */
void SparseMatmulTLA(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 42_quant_optimized_matmul_tla.
 */
void QuantOptimizedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 43_ascend950_basic_matmul.
 */
void Ascend950BasicMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 44_quant_matmul_full_loadA_tla.
 */
void QuantMatmulFullLoadATLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 45_strided_batched_matmul_tla.
 */
void StridedBatchedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const StridedBatchedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 46_ascend950_matmul_fixpipe_opti.
 */
void Ascend950MatmulFixpipeOpti(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 47_ascend950_grouped_matmul_slice_m_per_token_dequant.
 */
void Ascend950GroupedMatmulSliceMPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant.
 */
void Ascend950GroupedMatmulSliceMPerTensorPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 50_ascend950_basic_matmul_gemv.
 */
void Ascend950BasicMatmulGemv(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 51_ascend950_quant_matmul_per_group_per_block_tla.
 */
void Ascend950QuantMatmulPerGroupPerBlockTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 52_quant_multi_core_splitk_matmul_tla.
 */
void QuantMultiCoreSplitkMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 53_ascend950_fp8_mx_matmul.
 */
void Ascend950Fp8MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 53_ascend950_fp8_mx_matmul_aswt.
 */
void Ascend950Fp8MxMatmulAswt(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 54_ascend950_fp4_mx_matmul.
 */
void Ascend950Fp4MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 54_ascend950_fp4_mx_matmul_aswt.
 */
void Ascend950Fp4MxMatmulAswt(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 55_ascend950_mx_grouped_matmul_slice_m.
 */
void Ascend950MxGroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 57_ascend950_matmul_full_dequant.
 */
void Ascend950MatmulFullDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 58_ascend950_fp8_mx_batch_matmul.
 */
void Ascend950Fp8MxBatchMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 63_ascend950_dual_level_quant_mx_batch_matmul.
 */
void Ascend950DualLevelQuantMxBatchMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 59_ascend950_a8w4_mx_matmul.
 */
void Ascend950A8W4MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 61_ascend950_svd_quant_matmul.
 */
void Ascend950SvdQuantMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SvdQuantMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 60_ascend950_grouped_matmul_slice_m.
 */
void Ascend950GroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 64_ascend950_matmul_evg (unified EVG matmul entry).
 *
 * Selects the JIT template via ``params.evgType`` (e.g. add, add_ub, bias, leaky_relu, ...).
 */
void MatmulEvg(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulEvgParams& params);

/**
 * @brief JIT interface for example 67_ascend950_batched_matmul.
 */
void Ascend950BatchedMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.
 */
void Ascend950Fp8MxGroupedMatmulSliceMSwigluMxQuant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 66_ascend950_streamk_matmul.
 */
void Ascend950StreamkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Runtime parameters for example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing.
 */
struct GroupedMxFinalizeRoutingParams : public MatmulParams {
    uint32_t problemCount = 1;
    uint32_t groupListType = 0;
    float sharedInputWeight = 0.0f;
    uint32_t sharedInputOffset = 0;
    uint32_t bsdp = 1;
};

/**
 * @brief Reserved JIT interface for example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing.
 */
void Ascend950Fp8MxGroupedMatmulFinalizeRouting(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMxFinalizeRoutingParams& params);

/**
 * @brief Reserved JIT interface for example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing (no_deter variant).
 */
void Ascend950Fp8MxGroupedMatmulFinalizeRoutingNoDeter(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMxFinalizeRoutingParams& params);

/**
 * @brief JIT interface for example 73_ascend950_matmul_full_loadA.
 */
void Ascend950MatmulFullLoadA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);
/**
 * @brief JIT interface for example 75_symm.
 */
void Symm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SymmParams& params);

/**
 * @brief JIT interface for example 82_ascend950_basic_syrk.
 *
 * Y = X * X^T. ``params.m`` must equal ``params.n``; only ``inputAddr[0]`` (X) and
 * ``outputAddr[0]`` (Y) are consumed.
 */
void Ascend950BasicSyrk(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for 84_ascend950_syrk (MIX L0C→workspace + AIV AXPBY).
 *
 * D = alpha * X @ X^T + beta * Y. ``params.m`` must equal ``params.n``.
 */
void Ascend950Syrk(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SyrkParams& params);

/**
 * @brief Reserved JIT interface for example 74_ascend950_weight_quant_a8w4_grouped_mx_matmul.
 *
 * Grouped MX A8W4 matmul: C = (MxScaleA * A_fp8) @ (MxScaleB * B_fp4) per group.
 * B is the packed FP4 prologue (int8 bytes, Weight4BitnZ layout). Output is FP32.
 * ``params.batch`` carries the group (expert) count; ``inputAddr[2]`` is the group list.
 */
void Ascend950A8W4GroupedMxMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief JIT interface for example 78_ascend950_dual_matmul_silu_mul.
 */
void Ascend950DualMatmulSiluMul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for experimental/matmul/ascend950_fp4_mx_quant_matmul.
 */
void Ascend950Fp4MxQuantMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for experimental/matmul/ascend950_fp8_e4m3_quant_matmul.
 */
void Ascend950Fp8E4M3QuantMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for experimental/matmul/ascend950_matmul_gather_scatter.
 */
void Ascend950MatmulGatherScatter(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulGatherScatterParams& params);

/**
 * @brief Reserved JIT interface for example 102_dynamic_optimized_matmul.
 */
void DynamicOptimizedMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 103_dynamic_optimized_quant_matmul_per_token_basic.
 */
void DynamicOptimizedQuantMatmulPerTokenBasic(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 68_ascend950_multi_core_splitk_matmul.
 */
void Ascend950MultiCoreSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 69_ascend950_tail_multi_core_splitk_matmul.
 */
void Ascend950TailMultiCoreSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 76_trmm.
 */
void Trmm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const TrmmParams& params);

/**
 * @brief JIT interface for example 77_planar_complex_matmul.
 */
void PlanarComplexMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief JIT interface for example 80_grouped_matmul_slice_m_gelu.
 */
void GroupedMatmulSliceMGelu(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params);

} // extern "C"

} // namespace CatlassKernel

#endif // OPTEST_CATLASS_KERNEL_JIT_H
