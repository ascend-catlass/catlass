# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_kernel(NAME ascend950_multi_core_splitk_matmul
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_core_splitk_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/multi_core_splitk_matmul_impl.cpp)

set(_MULTI_CORE_VERIFY_BASE jit_verify_ascend950_multi_core_splitk_matmul_multi_core_splitk_matmul_impl_3510)
set(_MULTI_CORE_SCHEDULER 30)
if(TARGET ${_MULTI_CORE_VERIFY_BASE})
    target_compile_definitions(${_MULTI_CORE_VERIFY_BASE} PRIVATE
        CATLASS_JIT_ELEMENT_A=float
        CATLASS_JIT_ELEMENT_B=float
        CATLASS_JIT_ELEMENT_C=float
        CATLASS_JIT_LAYOUT_A=RowMajor
        CATLASS_JIT_LAYOUT_B=RowMajor
        CATLASS_JIT_LAYOUT_C=RowMajor
        CATLASS_JIT_BLOCK_SCHEDULER=${_MULTI_CORE_SCHEDULER})
endif()
