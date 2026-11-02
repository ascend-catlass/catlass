# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(basic_conv2d_tla.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(56_ascend950_basic_conv2d_tla cube basic_conv2d_tla.cpp)

set_source_files_properties(conv2d_bias_silu.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(56_ascend950_conv2d_bias_silu mix conv2d_bias_silu.cpp)
target_link_libraries(56_ascend950_conv2d_bias_silu PRIVATE m)
