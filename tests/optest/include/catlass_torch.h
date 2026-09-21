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

#ifndef OPTEST_CATLASS_TORCH_H
#define OPTEST_CATLASS_TORCH_H

#include <tuple>

#include <torch/extension.h>

/**
 * @brief PyTorch extension entry for CATLASS basic matrix multiplication.
 *
 * This declaration mirrors the schema registered in ``src/catlass_torch.cpp``.
 * The implementation is provided by ``MatmulLike<BasicMatmul>::Run``.
 *
 * @param mat1 Left input matrix.
 * @param mat2 Right input matrix.
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor basic_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS quantized optimized matmul (TLA).
 *
 * Source: example 42_quant_optimized_matmul_tla.
 * Implementation provided by ``QuantMatmulLike<QuantOptimizedMatmulTLA>::Run``.
 *
 * @param mat1 Left input matrix (int8).
 * @param mat2 Right input matrix (int8).
 * @param scale Per-column scale tensor (float).
 * @param perTokenScale Per-row scale tensor (float).
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor quant_optimized_matmul_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale, const at::Tensor& perTokenScale,
    const c10::ScalarType& outDType, const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS quantized matmul with full loadA (TLA).
 *
 * Source: example 44_quant_matmul_full_loadA_tla.
 * Implementation provided by ``QuantMatmulLike<QuantMatmulFullLoadATLA>::Run``.
 *
 * @param mat1 Left input matrix (int8).
 * @param mat2 Right input matrix (int8).
 * @param scale Per-column scale tensor (float).
 * @param perTokenScale Per-row scale tensor (float).
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor quant_matmul_full_loadA_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale, const at::Tensor& perTokenScale,
    const c10::ScalarType& outDType, const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS quantized multi-core splitK matmul (TLA).
 *
 * Source: example 52_quant_multi_core_splitk_matmul_tla.
 * Implementation provided by ``QuantMatmulLike<QuantMultiCoreSplitkMatmulTLA>::Run``.
 *
 * @param mat1 Left input matrix (int8).
 * @param mat2 Right input matrix (int8).
 * @param scale Per-column scale tensor (float).
 * @param perTokenScale Per-row scale tensor (float).
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor quant_multi_core_splitk_matmul_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale, const at::Tensor& perTokenScale,
    const c10::ScalarType& outDType, const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP8 matmul ASWT (TLA).
 *
 * Source: example 53_ascend950_fp8_mx_matmul_aswt.
 */
at::Tensor ascend950_fp8_mx_matmul_aswt(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const bool transA, const bool transB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP4 matmul ASWT (TLA).
 *
 * Source: example 54_ascend950_fp4_mx_matmul_aswt.
 */
at::Tensor ascend950_fp4_mx_matmul_aswt(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const bool transA, const bool transB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 EVG matmul (example 64_ascend950_matmul_evg).
 *
 * Post-processing mode is selected via ``evgType`` (add, add_ub, bias, leaky_relu, sigmoid, silu, tanh).
 * ``extra`` is required for add/add_ub (2-D) and bias (1-D); ignored for unary activation modes.
 */
at::Tensor matmul_evg(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& extra, const c10::ScalarType& outDType,
    const std::string& evgType, const double negativeSlope, const bool transA, const bool transB, const bool formatA,
    const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS FP8 E4M3 matmul (prologue dequant).
 *
 * Source: example 29_a2_fp8_e4m3_matmul.
 */
at::Tensor a2_fp8_e4m3_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS W8A16 matmul (int8 weight dequant).
 *
 * Source: example 30_w8a16_matmul.
 */
at::Tensor w8a16_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS W4A8 matmul (int4 weight dequant).
 *
 * Source: example 32_w4a8_matmul.
 */
at::Tensor w4a8_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS 4:2 sparse matmul (TLA).
 *
 * Source: example 41_sparse_matmul_tla.
 * Implementation provided by ``SparseMatmulLike<SparseMatmulTLA>::Run``.
 */
at::Tensor sparse_matmul_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& idx, const c10::ScalarType& outDType,
    const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS strided batched matmul (TLA).
 *
 * Source: example 45_strided_batched_matmul_tla.
 * Implementation provided by ``StridedBatchedMatmulLike<StridedBatchedMatmulTLA>::Run``.
 */
at::Tensor strided_batched_matmul_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB, const bool batchTransA, const bool batchTransB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 matmul fixpipe optimization.
 *
 * Source: example 46_ascend950_matmul_fixpipe_opti.
 * Implementation provided by ``MatmulLike<Ascend950MatmulFixpipeOpti>::Run``.
 */
at::Tensor ascend950_matmul_fixpipe_opti(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 basic matmul GEMV.
 *
 * Source: example 50_ascend950_basic_matmul_gemv.
 * Implementation provided by ``MatmulLike<Ascend950BasicMatmulGemv>::Run``.
 */
at::Tensor ascend950_basic_matmul_gemv(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 quant matmul per-group per-block (TLA).
 *
 * Source: example 51_ascend950_quant_matmul_per_group_per_block_tla.
 * Implementation provided by ``QuantPerGroupPerBlockMatmulLike<Ascend950QuantMatmulPerGroupPerBlockTLA>::Run``.
 */
at::Tensor ascend950_quant_matmul_per_group_per_block_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& x1Scale, const at::Tensor& x2Scale,
    const c10::ScalarType& outDType, const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 matmul full dequant.
 *
 * Source: example 57_ascend950_matmul_full_dequant.
 * Implementation provided by ``MatmulFullDequantLike<Ascend950MatmulFullDequant>::Run``.
 */
at::Tensor ascend950_matmul_full_dequant(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::optional<at::Tensor>& x1Scale,
    const c10::optional<at::Tensor>& x2Scale, const c10::optional<at::Tensor>& bias, const c10::ScalarType& outDType,
    const bool transA, const bool transB, const bool formatA, const bool formatB, const std::string& x1QuantMode,
    const std::string& x2QuantMode);

/**
 * @brief PyTorch extension entry for Ascend950 fused dual matmul + SiLU + multiply.
 *
 * Computes ``SiLU(x @ b0.T) * (x @ b1.T)`` where b0/b1 are physical (N, K)
 * tensors interpreted as ColumnMajor logical (K, N).
 */
at::Tensor ascend950_dual_matmul_silu_mul(
    const at::Tensor& x, const at::Tensor& b0, const at::Tensor& b1, const c10::ScalarType& outDType);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP8 batch matmul (TLA).
 *
 * Source: example 58_ascend950_fp8_mx_batch_matmul.
 */
at::Tensor ascend950_fp8_mx_batch_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const bool transA, const bool transB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 dual-level quant MX batch matmul.
 *
 * Source: example 63_ascend950_dual_level_quant_mx_batch_matmul.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> ascend950_dual_level_quant_mx_batch_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP8 grouped matmul + finalize routing.
 *
 * Source: example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing.
 */
at::Tensor ascend950_fp8_mx_grouped_matmul_finalize_routing(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const at::Tensor& group_list, const at::Tensor& logit, const at::Tensor& row_index, const at::Tensor& bias,
    const at::Tensor& shared_input, bool transA, bool transB, int64_t batch, int64_t data_parallel_size,
    double shared_input_weight, int64_t shared_input_offset, int64_t group_list_type);

#endif
