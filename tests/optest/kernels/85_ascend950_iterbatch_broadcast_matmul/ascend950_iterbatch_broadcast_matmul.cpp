/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See the License
 * root for the full text of the License.
 */

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

/**
 * @brief example 85_ascend950_iterbatch_broadcast_matmul: Resolve and launch the JIT-specialized
 *        IterBatchBroadcastMatmulTla implementation.
 */
extern "C" void Ascend950IterBatchBroadcastMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams,
    const IterBatchBroadcastMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_iterbatch_broadcast_matmul", tParams);
    auto* entry = JitCompiler::instance().getKernel(
        "ascend950_iterbatch_broadcast_matmul_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
