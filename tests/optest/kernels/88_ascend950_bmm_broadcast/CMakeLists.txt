# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_kernel(NAME ascend950_bmm_broadcast
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_bmm_broadcast.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_bmm_broadcast_impl.cpp)

# Verify the strict FP32 full-K path with Bias at build time.
# Runtime JIT compilation supplies its own parameters through JitMacroGenerator.
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_bmm_broadcast_impl.cpp
    APPEND PROPERTY COMPILE_DEFINITIONS
    CATLASS_JIT_ELEMENT_A=float
    CATLASS_JIT_LAYOUT_A=RowMajor
    CATLASS_JIT_LAYOUT_B=RowMajor
    CATLASS_BMM_HAS_BIAS=1
    CATLASS_BMM_HF32=0)
