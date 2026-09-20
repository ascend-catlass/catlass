/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"
#include "jit_macros.h"

namespace CatlassKernel {

extern "C" void Ascend950MatmulGatherScatter(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulGatherScatterParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_matmul_gather_scatter", tParams);

    auto* entry =
        JitCompiler::instance().getKernel("ascend950_matmul_gather_scatter_impl.cpp", macros, JitKernelType::MIX);
    JIT_CHECK(entry != nullptr, "JIT load failed for ascend950_matmul_gather_scatter_impl.cpp");
    entry(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
