# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(basic_conv2d.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(33_basic_conv2d cube basic_conv2d.cpp)

set_source_files_properties(conv2d_bias_silu.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(33_conv2d_bias_silu mix conv2d_bias_silu.cpp)
target_link_libraries(33_conv2d_bias_silu PRIVATE m)

# AtlasA2平台SiLU fp16计算特化验证目标（默认float卷积累加）：
# - ENABLE_FP16_SILU=1 启用 SiLU 按 fp16 计算的特化路径
if(CATLASS_ARCH STREQUAL "2201")
    catlass_example_add_executable(33_conv2d_bias_silu_fp16_silu mix conv2d_bias_silu.cpp)
    target_compile_definitions(33_conv2d_bias_silu_fp16_silu PRIVATE ENABLE_FP16_SILU=1)
    target_link_libraries(33_conv2d_bias_silu_fp16_silu PRIVATE m)
endif()
