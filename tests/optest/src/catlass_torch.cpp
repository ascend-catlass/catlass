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

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel_jit.h"
#include "catlass_kernel_prebuilt.h"
#include "common/register.h"
#include "common/workspace.h"
#include "template/basic_conv2d.h"
#include "template/batched_matmul.h"
#include "template/dual_matmul.h"
#include "template/flash_attention.h"
#include "template/flash_attention_chunk_prefill.h"
#include "template/hstu_infer.h"
#include "template/rain_fusion_attention.h"
#include "template/grouped_matmul.h"
#include "template/grouped_quant_matmul.h"
#include "template/matmul.h"
#include "template/matmul_evg.h"
#include "template/matmul_extra.h"
#include "template/matmul_full_dequant.h"
#include "template/matmul_gather_scatter.h"
#include "template/mla.h"
#include "template/mx_matmul.h"
#include "template/mx_grouped_matmul.h"
#include "template/grouped_fixpipe_dequant_matmul.h"
#include "template/quant_matmul.h"
#include "template/quant_per_group_per_block_matmul.h"
#include "template/sparse_matmul.h"
#include "template/strided_batched_matmul.h"
#include "template/w4a4_matmul_per_token_per_channel_dequant.h"
#include "template/w4a8_matmul.h"
#include "template/ascend950_basic_conv2d_tla.h"
#include "template/ascend950_mxfp8_flash_attention.h"
#include "template/broadcast_matmul_perblock_quant.h"
#include "template/gemm.h"
#include "template/group_gemm.h"
#include "template/mx_grouped_matmul_swiglu_mx_quant.h"
#include "template/a8w4_mx_matmul.h"
#include "template/a8w4_grouped_mx_matmul.h"
#include "template/svd_quant_matmul.h"
#include "template/ascend950_quant_matmul.h"
#include "template/basic_syrk.h"
#include "template/syrk.h"
#include "template/trmm.h"
#include "template/conv_bias.h"
#include "template/symm.h"
#include "template/planar_complex_matmul.h"
#include "template/matrix_inverse.h"
#include "template/x_attention.h"
// ── Workspace allocator bridge ──
// 通过 dlsym 注入到 g_catlassWorkspaceAlloc，使 JIT 模板分配 NPU tensor
// 而非裸 aclrtMalloc。tensor 保存在静态池中，kernel 执行期间有效。
#include <dlfcn.h>
#include <vector>

namespace {
static std::vector<at::Tensor> g_wsPool;

uint8_t* wsAlloc(size_t size)
{
    auto opts = at::TensorOptions().dtype(torch::kInt8).device(torch_npu::utils::get_npu_device_type());
    auto t = at::empty({static_cast<int64_t>(size)}, opts);
    auto* p = static_cast<uint8_t*>(const_cast<void*>(t.storage().data()));
    g_wsPool.push_back(std::move(t));
    return p;
}

uint8_t* wsAllocFromHost(const void* hostData, size_t size)
{
    auto opts = at::TensorOptions().dtype(torch::kInt8).device(torch_npu::utils::get_npu_device_type());
    auto dst = at::empty({static_cast<int64_t>(size)}, opts);
    auto* p = static_cast<uint8_t*>(const_cast<void*>(dst.storage().data()));
    aclrtMemcpy(p, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    g_wsPool.push_back(std::move(dst));
    return p;
}

void wsFree(uint8_t* p, size_t)
{
    for (auto it = g_wsPool.begin(); it != g_wsPool.end(); ++it)
        if (it->storage().data() == p) {
            g_wsPool.erase(it);
            break;
        }
}

struct WsInit {
    WsInit()
    {
        auto sa = (void (*)(decltype(wsAlloc)*))dlsym(RTLD_DEFAULT, "CatlassSetWorkspaceAlloc");
        auto sf = (void (*)(decltype(wsFree)*))dlsym(RTLD_DEFAULT, "CatlassSetWorkspaceFree");
        auto sc = (void (*)(decltype(wsAllocFromHost)*))dlsym(RTLD_DEFAULT, "CatlassSetWorkspaceAllocFromHost");
        if (sa)
            sa(wsAlloc);
        if (sf)
            sf(wsFree);
        if (sc)
            sc(wsAllocFromHost);
    }
} wsInit;
} // namespace

namespace CatlassKernelWrapper {

using BasicMatmulOp = MatmulLike<CatlassKernel::BasicMatmul>;
static auto& basic_matmul = BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(basic_matmul);

using BatchedMatmulOp = BatchedMatmulLike<CatlassKernel::BatchedMatmul>;
static auto& batched_matmul = BatchedMatmulOp::Run;
REGISTER_TORCH_FUNC(batched_matmul);

using GroupedMatmulSliceMOp = GroupedMatmulLike<CatlassKernel::GroupedMatmulSliceM, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m = GroupedMatmulSliceMOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m);

using GroupedMatmulSliceKOp = GroupedMatmulLike<CatlassKernel::GroupedMatmulSliceK, GmmSliceDir::K>;
static auto& grouped_matmul_slice_k = GroupedMatmulSliceKOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_k);

using GroupedMatmulOp = GroupedMatmulLike<CatlassKernel::GroupedMatmul, GmmSliceDir::K>;
static auto& grouped_matmul = GroupedMatmulOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul);

using MatmulAddOp = MatmulExtraLike<CatlassKernel::MatmulAdd, false>;
static auto& matmul_add = MatmulAddOp::Run;
REGISTER_TORCH_FUNC(matmul_add);

using PaddingMatmulOp = MatmulLike<CatlassKernel::PaddingMatmul>;
static auto& padding_matmul = PaddingMatmulOp::Run;
REGISTER_TORCH_FUNC(padding_matmul);

using OptimizedMatmulOp = MatmulLike<CatlassKernel::OptimizedMatmul>;
static auto& optimized_matmul = OptimizedMatmulOp::Run;
REGISTER_TORCH_FUNC(optimized_matmul);

using MatmulBiasOp = MatmulExtraLike<CatlassKernel::MatmulBias, true>;
static auto& matmul_bias = MatmulBiasOp::Run;
REGISTER_TORCH_FUNC(matmul_bias);

using GroupedMatmulSliceMPerTokenDequantMoeOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceMPerTokenDequantMoe, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m_per_token_dequant_moe = GroupedMatmulSliceMPerTokenDequantMoeOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m_per_token_dequant_moe);

using GroupedMatmulSliceMPerTokenDequantOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceMPerTokenDequant, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m_per_token_dequant = GroupedMatmulSliceMPerTokenDequantOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m_per_token_dequant);

using GroupedMatmulSliceKPerTokenDequantOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceKPerTokenDequant, GmmSliceDir::K>;
static auto& grouped_matmul_slice_k_per_token_dequant = GroupedMatmulSliceKPerTokenDequantOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_k_per_token_dequant);

using SplitkMatmulOp = MatmulLike<CatlassKernel::SplitkMatmul>;
static auto& splitk_matmul = SplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(splitk_matmul);

using QuantMatmulOp = QuantMatmulLike<CatlassKernel::QuantMatmul>;
static auto& quant_matmul = QuantMatmulOp::Run;
REGISTER_TORCH_FUNC(quant_matmul);

using PaddingSplitkMatmulOp = MatmulLike<CatlassKernel::PaddingSplitkMatmul>;
static auto& padding_splitk_matmul = PaddingSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(padding_splitk_matmul);

using BasicMatmulTLAOp = MatmulLike<CatlassKernel::BasicMatmulTLA>;
static auto& basic_matmul_tla = BasicMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(basic_matmul_tla);

using MatmulReluOp = MatmulLike<CatlassKernel::MatmulRelu>;
static auto& matmul_relu = MatmulReluOp::Run;
REGISTER_TORCH_FUNC(matmul_relu);

using MatmulGeluOp = MatmulLike<CatlassKernel::MatmulGelu>;
static auto& matmul_gelu = MatmulGeluOp::Run;
REGISTER_TORCH_FUNC(matmul_gelu);

using MatmulSiluOp = MatmulLike<CatlassKernel::MatmulSilu>;
static auto& matmul_silu = MatmulSiluOp::Run;
REGISTER_TORCH_FUNC(matmul_silu);

using OptimizedMatmulTLAOp = MatmulLike<CatlassKernel::OptimizedMatmulTLA>;
static auto& optimized_matmul_tla = OptimizedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(optimized_matmul_tla);

using BasicMatmulPreloadZNOp = MatmulLike<CatlassKernel::BasicMatmulPreloadZN>;
static auto& basic_matmul_preload_zN = BasicMatmulPreloadZNOp::Run;
REGISTER_TORCH_FUNC(basic_matmul_preload_zN);

using MatmulFullLoadAOp = MatmulLike<CatlassKernel::MatmulFullLoadA>;
static auto& matmul_full_loadA = MatmulFullLoadAOp::Run;
REGISTER_TORCH_FUNC(matmul_full_loadA);

using SmallMatmulOp = MatmulLike<CatlassKernel::SmallMatmul>;
static auto& small_matmul = SmallMatmulOp::Run;
REGISTER_TORCH_FUNC(small_matmul);

using SingleCoreSplitkMatmulOp = MatmulLike<CatlassKernel::SingleCoreSplitkMatmul>;
static auto& single_core_splitk_matmul = SingleCoreSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(single_core_splitk_matmul);

using StreamkMatmulOp = MatmulLike<CatlassKernel::StreamkMatmul>;
static auto& streamk_matmul = StreamkMatmulOp::Run;
REGISTER_TORCH_FUNC(streamk_matmul);

using QuantOptimizedMatmulTLAOp = QuantMatmulLike<CatlassKernel::QuantOptimizedMatmulTLA>;
static auto& quant_optimized_matmul_tla = QuantOptimizedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(quant_optimized_matmul_tla);

using Ascend950BasicMatmulOp = MatmulLike<CatlassKernel::Ascend950BasicMatmul>;
static auto& ascend950_basic_matmul = Ascend950BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_basic_matmul);

using Ascend950BatchedMatmulOp = BatchedMatmulLike<CatlassKernel::Ascend950BatchedMatmul>;
static auto& ascend950_batched_matmul = Ascend950BatchedMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_batched_matmul);

using Ascend950StreamkMatmulOp = MatmulLike<CatlassKernel::Ascend950StreamkMatmul>;
static auto& ascend950_streamk_matmul = Ascend950StreamkMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_streamk_matmul);

using QuantMatmulFullLoadATLAOp = QuantMatmulLike<CatlassKernel::QuantMatmulFullLoadATLA>;
static auto& quant_matmul_full_loadA_tla = QuantMatmulFullLoadATLAOp::Run;
REGISTER_TORCH_FUNC(quant_matmul_full_loadA_tla);

using QuantMultiCoreSplitkMatmulTLAOp = QuantMatmulLike<CatlassKernel::QuantMultiCoreSplitkMatmulTLA>;
static auto& quant_multi_core_splitk_matmul_tla = QuantMultiCoreSplitkMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(quant_multi_core_splitk_matmul_tla);

using Ascend950Fp8MxMatmulAswtOp = MxMatmulLike<CatlassKernel::Ascend950Fp8MxMatmulAswt>;
static auto& ascend950_fp8_mx_matmul_aswt = Ascend950Fp8MxMatmulAswtOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_matmul_aswt);

using Ascend950Fp4MxMatmulAswtOp = MxMatmulLike<CatlassKernel::Ascend950Fp4MxMatmulAswt>;
static auto& ascend950_fp4_mx_matmul_aswt = Ascend950Fp4MxMatmulAswtOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp4_mx_matmul_aswt);

using MatmulEvgOp = MatmulEvgLike<CatlassKernel::MatmulEvg>;
static auto& matmul_evg = MatmulEvgOp::Run;
REGISTER_TORCH_FUNC(matmul_evg);

using Ascend950MatmulFullLoadAOp = MatmulLike<CatlassKernel::Ascend950MatmulFullLoadA>;
static auto& ascend950_matmul_full_loadA = Ascend950MatmulFullLoadAOp::Run;
REGISTER_TORCH_FUNC(ascend950_matmul_full_loadA);

using Ascend950DualMatmulSiluMulOp = DualMatmulSiluMulLike<CatlassKernel::Ascend950DualMatmulSiluMul>;
static auto& ascend950_dual_matmul_silu_mul = Ascend950DualMatmulSiluMulOp::Run;
REGISTER_TORCH_FUNC(ascend950_dual_matmul_silu_mul);

using Ascend950MatmulGatherScatterOp = MatmulGatherScatterLike<CatlassKernel::Ascend950MatmulGatherScatter>;
static auto& ascend950_matmul_gather_scatter = Ascend950MatmulGatherScatterOp::Run;
REGISTER_TORCH_FUNC(ascend950_matmul_gather_scatter);
using A2Fp8E4M3MatmulOp = MatmulLike<CatlassKernel::A2Fp8E4M3Matmul>;
static auto& a2_fp8_e4m3_matmul = A2Fp8E4M3MatmulOp::Run;
REGISTER_TORCH_FUNC(a2_fp8_e4m3_matmul);

using TrmmOp = TrmmLike<CatlassKernel::Trmm>;
static auto& trmm = TrmmOp::Run;
REGISTER_TORCH_FUNC(trmm);
static auto& basic_conv2d = BasicConv2dOp::Run;
REGISTER_TORCH_FUNC(basic_conv2d);

static auto& mla = MlaOp::Run;
REGISTER_TORCH_FUNC(mla);

static auto& flash_attention_infer = FlashAttentionInferOp::Run;
REGISTER_TORCH_FUNC(flash_attention_infer);

static auto& flash_attention_infer_tla = FlashAttentionInferTLAOp::Run;
REGISTER_TORCH_FUNC(flash_attention_infer_tla);

static auto& ascend950_flash_attention_infer = Ascend950FlashAttentionInferOp::Run;
REGISTER_TORCH_FUNC(ascend950_flash_attention_infer);

static auto& ascend950_fp8_mx_flash_attention_infer = Ascend950MxFp8FlashAttentionInferOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_flash_attention_infer);

static auto& ascend950_hstu_infer = Ascend950HstuInferOp::Run;
REGISTER_TORCH_FUNC(ascend950_hstu_infer);

using W8A16MatmulOp = MatmulLike<CatlassKernel::W8A16Matmul>;
static auto& w8a16_matmul = W8A16MatmulOp::Run;
REGISTER_TORCH_FUNC(w8a16_matmul);

using W4A8MatmulOp = W4A8MatmulLike<CatlassKernel::W4A8Matmul>;
static auto& w4a8_matmul = W4A8MatmulOp::Run;
REGISTER_TORCH_FUNC(w4a8_matmul);

using W4A4MatmulPerTokenPerChannelDequantOp =
    W4A4MatmulPerTokenPerChannelDequantLike<CatlassKernel::W4A4MatmulPerTokenPerChannelDequant>;
static auto& w4a4_matmul_per_token_per_channel_dequant = W4A4MatmulPerTokenPerChannelDequantOp::Run;
REGISTER_TORCH_FUNC(w4a4_matmul_per_token_per_channel_dequant);

using SparseMatmulTLAOp = SparseMatmulLike<CatlassKernel::SparseMatmulTLA>;
static auto& sparse_matmul_tla = SparseMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(sparse_matmul_tla);

using StridedBatchedMatmulTLAOp = StridedBatchedMatmulLike<CatlassKernel::StridedBatchedMatmulTLA>;
static auto& strided_batched_matmul_tla = StridedBatchedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(strided_batched_matmul_tla);

using Ascend950MatmulFixpipeOptiOp = MatmulLike<CatlassKernel::Ascend950MatmulFixpipeOpti>;
static auto& ascend950_matmul_fixpipe_opti = Ascend950MatmulFixpipeOptiOp::Run;
REGISTER_TORCH_FUNC(ascend950_matmul_fixpipe_opti);

using Ascend950BasicMatmulGemvOp = MatmulLike<CatlassKernel::Ascend950BasicMatmulGemv>;
static auto& ascend950_basic_matmul_gemv = Ascend950BasicMatmulGemvOp::Run;
REGISTER_TORCH_FUNC(ascend950_basic_matmul_gemv);

using Ascend950QuantMatmulPerGroupPerBlockTLAOp =
    QuantPerGroupPerBlockMatmulLike<CatlassKernel::Ascend950QuantMatmulPerGroupPerBlockTLA>;
static auto& ascend950_quant_matmul_per_group_per_block_tla = Ascend950QuantMatmulPerGroupPerBlockTLAOp::Run;
REGISTER_TORCH_FUNC(ascend950_quant_matmul_per_group_per_block_tla);

using Ascend950MatmulFullDequantOp = MatmulFullDequantLike<CatlassKernel::Ascend950MatmulFullDequant>;
static auto& ascend950_matmul_full_dequant = Ascend950MatmulFullDequantOp::Run;
REGISTER_TORCH_FUNC(ascend950_matmul_full_dequant);

using Ascend950Fp8MxBatchMatmulOp = MxBatchedMatmulLike<CatlassKernel::Ascend950Fp8MxBatchMatmul>;
static auto& ascend950_fp8_mx_batch_matmul = Ascend950Fp8MxBatchMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_batch_matmul);

using BroadcastMatmulPerblockQuantOp = BroadcastMatmulPerblockQuantLike<CatlassKernel::BroadcastMatmulPerblockQuant>;
static auto& broadcast_matmul_perblock_quant = BroadcastMatmulPerblockQuantOp::Run;
REGISTER_TORCH_FUNC(broadcast_matmul_perblock_quant);

using Ascend950DualLevelQuantMxBatchMatmulOp =
    DualLevelQuantMxBatchedMatmulLike<CatlassKernel::Ascend950DualLevelQuantMxBatchMatmul>;
static auto& ascend950_dual_level_quant_mx_batch_matmul = Ascend950DualLevelQuantMxBatchMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_dual_level_quant_mx_batch_matmul);

// ── example 15_gemm ──
using GemmOp = GemmLike<CatlassKernel::Gemm>;
static auto& gemm = GemmOp::Run;
REGISTER_TORCH_FUNC(gemm);

// ── example 16_group_gemm ──
using GroupGemmOp = GroupGemmLike<CatlassKernel::GroupGemm>;
static auto& group_gemm = GroupGemmOp::Run;
REGISTER_TORCH_FUNC(group_gemm);

// ── example 17_gemv_aiv ──
using GemvAIVOp = GemmLike<CatlassKernel::GemvAIV, true>;
static auto& gemv_aiv = GemvAIVOp::Run;
REGISTER_TORCH_FUNC(gemv_aiv);

// ── example 18_gemv_aic ──
using GemvAICOp = GemmLike<CatlassKernel::GemvAIC, true>;
static auto& gemv_aic = GemvAICOp::Run;
REGISTER_TORCH_FUNC(gemv_aic);
using Ascend950SvdQuantMatmulOp = SvdQuantMatmulLike<CatlassKernel::Ascend950SvdQuantMatmul>;
static auto& ascend950_svd_quant_matmul = Ascend950SvdQuantMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_svd_quant_matmul);

using Ascend950MultiCoreSplitkMatmulOp = MatmulLike<CatlassKernel::Ascend950MultiCoreSplitkMatmul>;
static auto& ascend950_multi_core_splitk_matmul = Ascend950MultiCoreSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_multi_core_splitk_matmul);

using Ascend950A8W4MxMatmulOp = A8W4MxMatmulLike<CatlassKernel::Ascend950A8W4MxMatmul>;
static auto& ascend950_a8w4_mx_matmul = Ascend950A8W4MxMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_a8w4_mx_matmul);

using Ascend950A8W4GroupedMxMatmulOp = A8W4GroupedMxMatmulLike<CatlassKernel::Ascend950A8W4GroupedMxMatmul>;
static auto& ascend950_a8w4_grouped_mx_matmul = Ascend950A8W4GroupedMxMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_a8w4_grouped_mx_matmul);

using Ascend950Fp8MxGroupedMatmulSliceMSwigluMxQuantOp =
    GroupedMxSwigluMxQuantMatmulLike<CatlassKernel::Ascend950Fp8MxGroupedMatmulSliceMSwigluMxQuant>;
static auto& ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant =
    Ascend950Fp8MxGroupedMatmulSliceMSwigluMxQuantOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant);

using Ascend950TailMultiCoreSplitkMatmulOp = MatmulLike<CatlassKernel::Ascend950TailMultiCoreSplitkMatmul>;
static auto& ascend950_tail_multi_core_splitk_matmul = Ascend950TailMultiCoreSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_tail_multi_core_splitk_matmul);

using Ascend950Fp4MxQuantMatmulOp = Fp4MxQuantMatmulLike<CatlassKernel::Ascend950Fp4MxQuantMatmul>;
static auto& ascend950_fp4_mx_quant_matmul = Ascend950Fp4MxQuantMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp4_mx_quant_matmul);

using Ascend950Fp8E4M3QuantMatmulOp = Fp8E4M3QuantMatmulLike<CatlassKernel::Ascend950Fp8E4M3QuantMatmul>;
static auto& ascend950_fp8_e4m3_quant_matmul = Ascend950Fp8E4M3QuantMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_e4m3_quant_matmul);
using Ascend950GroupedMatmulSliceMPerTokenDequantOp =
    GroupedQuantMatmulLike<CatlassKernel::Ascend950GroupedMatmulSliceMPerTokenDequant, GmmSliceDir::M>;
static auto& ascend950_grouped_matmul_slice_m_per_token_dequant = Ascend950GroupedMatmulSliceMPerTokenDequantOp::Run;
REGISTER_TORCH_FUNC(ascend950_grouped_matmul_slice_m_per_token_dequant);

using Ascend950GroupedMatmulSliceMPerTensorPerChannelDequantOp =
    GroupedFixpipeDequantMatmulLike<CatlassKernel::Ascend950GroupedMatmulSliceMPerTensorPerChannelDequant>;
static auto& ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant =
    Ascend950GroupedMatmulSliceMPerTensorPerChannelDequantOp::Run;
REGISTER_TORCH_FUNC(ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant);

using Ascend950MxGroupedMatmulSliceMOp = MxGroupedMatmulLike<CatlassKernel::Ascend950MxGroupedMatmulSliceM>;
static auto& ascend950_mx_grouped_matmul_slice_m = Ascend950MxGroupedMatmulSliceMOp::Run;
REGISTER_TORCH_FUNC(ascend950_mx_grouped_matmul_slice_m);

using Ascend950GroupedMatmulSliceMOp = GroupedMatmulLike<CatlassKernel::Ascend950GroupedMatmulSliceM, GmmSliceDir::M>;
static auto& ascend950_grouped_matmul_slice_m = Ascend950GroupedMatmulSliceMOp::Run;
REGISTER_TORCH_FUNC(ascend950_grouped_matmul_slice_m);

using Ascend950Fp8MxGroupedMatmulFinalizeRoutingOp =
    MxGroupedMatmulFinalizeRoutingLike<CatlassKernel::Ascend950Fp8MxGroupedMatmulFinalizeRouting>;
static auto& ascend950_fp8_mx_grouped_matmul_finalize_routing = Ascend950Fp8MxGroupedMatmulFinalizeRoutingOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_grouped_matmul_finalize_routing);

using Ascend950Fp8MxGroupedMatmulFinalizeRoutingNoDeterOp =
    MxGroupedMatmulFinalizeRoutingLike<CatlassKernel::Ascend950Fp8MxGroupedMatmulFinalizeRoutingNoDeter>;
static auto& ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter =
    Ascend950Fp8MxGroupedMatmulFinalizeRoutingNoDeterOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter);

static auto& ascend950_flash_attention_chunk_prefill = Ascend950FlashAttentionChunkPrefillOp::Run;
REGISTER_TORCH_FUNC(ascend950_flash_attention_chunk_prefill);

static auto& ascend950_rain_fusion_attention = Ascend950RainFusionAttentionOp::Run;
REGISTER_TORCH_FUNC(ascend950_rain_fusion_attention);

static auto& ascend950_basic_conv2d_tla = Ascend950BasicConv2dTLAOp::Run;
REGISTER_TORCH_FUNC(ascend950_basic_conv2d_tla);

static auto& conv_bias = ConvBiasOp::Run;
REGISTER_TORCH_FUNC(conv_bias);

using SymmOp = SymmLike<CatlassKernel::Symm>;
static auto& symm = SymmOp::Run;
REGISTER_TORCH_FUNC(symm);

using GroupedMatmulSliceMGeluOp = GroupedMatmulLike<CatlassKernel::GroupedMatmulSliceMGelu, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m_gelu = GroupedMatmulSliceMGeluOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m_gelu);

using PlanarComplexMatmulOp = PlanarComplexMatmulLike<CatlassKernel::PlanarComplexMatmul>;
static auto& planar_complex_matmul = PlanarComplexMatmulOp::Run;
REGISTER_TORCH_FUNC(planar_complex_matmul);

// ── example 78_matrix_inverse ──
static auto& matrix_inverse = MatrixInverseOp::Run;
REGISTER_TORCH_FUNC(matrix_inverse);

// ── example 78_x_attention ──
static auto& x_attention = XAttentionOp::Run;
REGISTER_TORCH_FUNC(x_attention);

// ── example 82_ascend950_basic_syrk ──
using Ascend950BasicSyrkOp = BasicSyrkLike<CatlassKernel::Ascend950BasicSyrk>;
static auto& ascend950_basic_syrk = Ascend950BasicSyrkOp::Run;
REGISTER_TORCH_FUNC(ascend950_basic_syrk);

// ── 84_ascend950_syrk ──
using Ascend950SyrkOp = SyrkLike<CatlassKernel::Ascend950Syrk>;
static auto& ascend950_syrk = Ascend950SyrkOp::Run;
REGISTER_TORCH_FUNC(ascend950_syrk);

} // namespace CatlassKernelWrapper
