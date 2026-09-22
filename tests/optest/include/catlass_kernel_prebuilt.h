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

#ifndef OPTEST_CATLASS_KERNEL_PREBUILT_H
#define OPTEST_CATLASS_KERNEL_PREBUILT_H

#include <string>

#include <cstdint>
#include <vector>

#include <acl/acl.h>

#include "catlass_kernel_jit.h"

namespace CatlassKernel {

/**
 * @brief Runtime parameters shared by prebuilt numbered examples.
 */
struct PrebuiltParams {
    std::vector<uint8_t*> inputAddr;  ///< Input buffer addresses.
    std::vector<uint8_t*> outputAddr; ///< Output buffer addresses.
};

/**
 * @brief Runtime parameters for convolution examples.
 */
struct ConvParams : public PrebuiltParams {
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;  ///< Input tensor element type.
    aclDataType biasDataType = aclDataType::ACL_FLOAT;     ///< Bias tensor element type.
    aclDataType outputDataType = aclDataType::ACL_FLOAT16; ///< Output tensor element type.
    std::vector<uint32_t> fmapRelated;                     ///< Feature-map dimensions.
    std::vector<uint32_t> filterRelated;                   ///< Filter dimensions.
    std::vector<uint32_t> strideList;                      ///< Convolution strides.
    std::vector<uint32_t> padList;                         ///< Padding values.
    std::vector<uint32_t> dilationList;                    ///< Dilation values.
};

/**
 * @brief Runtime parameters for flash-attention style examples.
 */
struct FlashAttentionParams : public PrebuiltParams {
    uint32_t qNtokens = 0;              ///< Total Q tokens for variable-length input.
    uint32_t batch = 0;                 ///< Batch size.
    uint32_t qSeqlen = 0;               ///< Q sequence length.
    uint32_t kvSeqlen = 0;              ///< KV sequence length.
    uint32_t numHeads = 0;              ///< Number of Q heads.
    uint32_t kvHeads = 0;               ///< Number of KV heads.
    uint32_t embeddingSize = 0;         ///< Per-head embedding dimension.
    uint32_t isVariedLen = 0;           ///< Whether variable-length input is used.
    uint32_t maskType = 0;              ///< Mask mode.
    uint32_t blockSize = 128;           ///< Tile block size.
    aclDataType dataType = ACL_FLOAT16; ///< Input/output element type.
};

/**
 * @brief Runtime parameters for example 78_x_attention.
 */
struct XAttentionParams : public PrebuiltParams {
    uint32_t numTokens = 0;             ///< Total query tokens (batch * beam size).
    uint32_t batch = 0;                 ///< Batch size.
    uint32_t beamSize = 0;              ///< Number of beams per batch.
    uint32_t sharedKvSeqLen = 0;        ///< Shared KV sequence length used for host tiling.
    uint32_t numHeads = 0;              ///< Number of query heads.
    uint32_t kvHeads = 0;               ///< Number of key/value heads.
    uint32_t embeddingSize = 0;         ///< Per-head embedding dimension.
    uint32_t maxDecodeStep = 0;         ///< Capacity of each unshared decode cache.
    bool sharedPaged = true;            ///< True for shared paged + unshared continuous cache.
    float scaleValue = 0.0f;            ///< Attention scale; zero selects 1/sqrt(embeddingSize).
    aclDataType dataType = ACL_FLOAT16; ///< Input/output element type.
};

/**
 * @brief Runtime parameters for MLA examples.
 */
struct MlaParams : public FlashAttentionParams {
    uint32_t qRopeHeadDim = 0;               ///< Q rope head dimension.
    uint32_t kvRopeHeadDim = 0;              ///< KV rope head dimension.
    uint32_t numBlocks = 0;                  ///< Total paged KV cache blocks.
    std::vector<int32_t> qSeqHost;           ///< Host-side Q sequence lengths for tiling.
    std::vector<int32_t> kvSeqHost;          ///< Host-side KV sequence lengths for tiling.
    mutable std::vector<uint8_t> outputHost; ///< Host-side output staging buffer.
};

/**
 * @brief Runtime parameters for flash-attention style examples.
 */
struct FlashAttentionChunkPrefillParams : public PrebuiltParams {
    uint32_t qNtokens = 0;        ///< Total Q tokens for variable-length input.
    uint32_t batch = 0;           ///< Batch size.
    uint32_t qSeqlen = 0;         ///< Q sequence length.
    uint32_t kvSeqlen = 0;        ///< KV sequence length.
    uint32_t numHeads = 0;        ///< Number of Q heads.
    uint32_t kvHeads = 0;         ///< Number of KV heads.
    uint32_t qkembeddingSize = 0; ///< Per-head embedding dimension.
    uint32_t vembeddingSize = 0;  ///< Per-head embedding dimension.
    uint32_t isVariedLen = 0;     ///< Whether variable-length input is used.
    uint32_t maskType = 0;        ///< Mask mode.
    uint32_t blockSize = 128;     ///< Tile block size.
    uint32_t numBlocks = 2048;    ///< Tile block num.
    std::string cacheLayout = "nd";
    aclDataType dataType = ACL_FLOAT16; ///< Input/output element type.
};

/**
 * @brief Runtime parameters for example 81_ascend950_rain_fusion_attention.
 */
struct RainFusionAttentionParams : public PrebuiltParams {
    uint32_t batch = 0;
    uint32_t numHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t blockShapeX = 128;
    uint32_t blockShapeY = 128;
    uint32_t maxKvBlockNum = 0;
    uint32_t totalQsBlockNum = 0;
    uint32_t maxQSeqlen = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t maskType = 0;
    float scaleValue = 0.0f;
    uint32_t qInputLayout = 0;
    uint32_t kvInputLayout = 0;
    uint32_t isVariedLen = 0;
    aclDataType dataType = ACL_FLOAT16;
    std::vector<int64_t> qSeqHost;
    std::vector<int64_t> kvSeqHost;
};

/**
 * @brief Reserved prebuilt interface for example 19_mla.
 */
__attribute__((weak)) void Mla(const uint32_t blockNum, aclrtStream stream, const MlaParams& params);

/**
 * @brief Reserved prebuilt interface for example 23_flash_attention_infer.
 */
__attribute__((weak)) void FlashAttentionInfer(
    const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 78_x_attention.
 */
__attribute__((weak)) void XAttention(
    const uint32_t blockNum, aclrtStream stream, const XAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 24_conv_bias.
 */
__attribute__((weak)) void ConvBias(const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

/**
 * @brief Reserved prebuilt interface for example 33_basic_conv2d.
 */
__attribute__((weak)) void BasicConv2d(const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

/**
 * @brief Reserved prebuilt interface for example 40_flash_attention_infer_tla.
 */
__attribute__((weak)) void FlashAttentionInferTLA(
    const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 49_ascend950_flash_attention_infer.
 */
__attribute__((weak)) void Ascend950FlashAttentionInfer(
    const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 56_ascend950_basic_conv2d_tla.
 */
__attribute__((weak)) void Ascend950BasicConv2dTLA(
    const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

/**
 * @brief Runtime parameters for Ascend950 MXFP8 flash attention examples.
 */
struct Ascend950MxFp8FlashAttentionParams : public FlashAttentionParams {
    uint32_t usePscale = 0; ///< Whether to use P matrix quantization scale.
};

/**
 * @brief Prebuilt interface for example 72_ascend950_fp8_mx_flash_attention_infer.
 */
__attribute__((weak)) void Ascend950MxFp8FlashAttentionInfer(
    const uint32_t blockNum, aclrtStream stream, const Ascend950MxFp8FlashAttentionParams& params);

/**
 * @brief Prebuilt interface for example 29_a2_fp8_e4m3_matmul.
 */
extern "C" __attribute__((weak)) void A2Fp8E4M3Matmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved prebuilt interface for example 70_ascend950_flash_attention_chunk_prefill.
 */
__attribute__((weak)) void FlashAttentionChunkPrefill(
    const uint32_t blockNum, aclrtStream stream, const FlashAttentionChunkPrefillParams& params);

/**
 * @brief Reserved prebuilt interface for example 81_ascend950_rain_fusion_attention.
 */
__attribute__((weak)) void RainFusionAttention(
    const uint32_t blockNum, aclrtStream stream, const RainFusionAttentionParams& params);

/**
 * @brief Runtime parameters for example 83_ascend950_hstu_infer.
 *
 * HSTU inference attention: replaces softmax with scaled SiLU activation
 * (P = siluScale * S * sigmoid(S)). Supports NTD/TND layouts and paged KV
 * cache (NHD layout with block table indirection).
 *
 * inputAddr[0]: per-batch cumulative Q sequence lengths (int64, batch+1).
 * inputAddr[1]: per-batch cumulative KV sequence lengths (int64, batch+1).
 * inputAddr[2]: query tensor.
 * inputAddr[3]: key tensor.
 * inputAddr[4]: value tensor.
 * inputAddr[5]: paged KV block table (uint32 [batch, maxNumBlocks]).
 * outputAddr[0]: output tensor (same layout as query).
 */
struct HstuInferParams : public PrebuiltParams {
    uint32_t qNtokens = 0;             ///< Total Q tokens across batches (from cumulative lengths).
    uint32_t batch = 0;                ///< Batch size.
    uint32_t numHeads = 0;             ///< Number of Q heads.
    uint32_t kvHeads = 0;              ///< Number of KV heads (must equal numHeads).
    uint32_t embeddingSize = 0;        ///< Per-head embedding dimension (256 for best performance).
    uint32_t maxKvSeqlen = 0;          ///< Max KV sequence length across batches (paged mode only).
    uint32_t numPagedBlocks = 0;       ///< Total paged KV cache blocks (paged mode only).
    uint32_t pagedBlockSize = 0;       ///< Paged block size, 0 disables paged KV cache.
    uint32_t maskType = 0;             ///< 0: no mask; 1: causal multiplicative mask.
    float siluScale = 0.1f;            ///< SiLU activation scale.
    std::string layout = "TND";        ///< Q/O layout: "TND" or "NTD".
    aclDataType dataType = ACL_FLOAT16; ///< Input/output element type (fp16 only).
};

/**
 * @brief Reserved prebuilt interface for example 83_ascend950_hstu_infer.
 */
__attribute__((weak)) void Ascend950HstuInfer(
    const uint32_t blockNum, aclrtStream stream, const HstuInferParams& params);

/**
 * @brief Broadcast MatMul with Per-Block Quantization（Ascend 950 TLA）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param params    运行期参数（M/N/K/batch、地址）。
 */
__attribute__((weak)) void BroadcastMatmulPerblockQuant(
    const uint32_t blockNum, aclrtStream stream, const MatmulParams& params);

/**
 * @brief Runtime parameters for matrix inverse example 78_matrix_inverse.
 *
 * Computes the inverse of a square matrix A (N×N) in-place using LU decomposition
 * with partial pivoting. The input matrix A is overwritten with its inverse A^{-1}.
 * The pivot array is allocated internally as workspace.
 *
 * inputAddr[0]: pointer to input matrix A (N×N, row-major)
 */
struct MatrixInverseParams : public PrebuiltParams {
    uint32_t N = 0;                   ///< Matrix dimension (N for N×N matrix).
    aclDataType dataType = ACL_FLOAT; ///< Element type of matrix A.
};

/**
 * @brief Reserved prebuilt interface for example 78_matrix_inverse.
 */
__attribute__((weak)) void MatrixInverse(
    const uint32_t blockNum, aclrtStream stream, const MatrixInverseParams& params);

/**
 * @brief Runtime parameters for flash-attention gradient (FAG) examples.
 *
 * FlashAttentionScoreGrad computes dQ/dK/dV given the forward cached results
 * (out and softmax log-sum-exp) plus the output gradient dout.
 *
 * inputAddr layout (TND):
 *   [0] cu_seq_qlen  (int64, prefix-sum of per-batch Q lengths)
 *   [1] cu_seq_kvlen (int64, prefix-sum of per-batch KV lengths)
 *   [2] dout         (half/bf16, (totalQ, numHeads, qkHeadDim))
 *   [3] q            (half/bf16, (totalQ, numHeads, qkHeadDim))
 *   [4] k            (half/bf16, (totalKV, kvHeads, qkHeadDim))
 *   [5] v            (half/bf16, (totalKV, kvHeads, vHeadDim))
 *   [6] out          (forward output, same shape as q)
 *   [7] softmax_lse  (float32, (totalQ, numHeads))
 * outputAddr layout:
 *   [0] dq  (same shape as q)
 *   [1] dk  (same shape as k)
 *   [2] dv  (same shape as v)
 */
struct FlashAttentionGradParams : public PrebuiltParams {
    uint32_t batch = 0;        ///< Batch size (number of variable-length sequences).
    uint32_t numHeads = 0;     ///< Number of Q heads.
    uint32_t kvHeads = 0;      ///< Number of KV heads.
    uint32_t qkHeadDim = 0;    ///< Per-head Q/K embedding dimension.
    uint32_t vHeadDim = 0;     ///< Per-head V embedding dimension.
    bool isDeterministic = false; ///< Whether to use deterministic DQKV path.
    aclDataType dataType = ACL_FLOAT16; ///< Input/output element type (half/bf16).
};

/**
 * @brief Reserved prebuilt interface for example 87_fag_tla.
 */
__attribute__((weak)) void FlashAttentionGradTLA(
    const uint32_t blockNum, aclrtStream stream, const FlashAttentionGradParams& params);

} // namespace CatlassKernel

#endif // OPTEST_CATLASS_KERNEL_PREBUILT_H
