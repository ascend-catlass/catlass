# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(basic_syrk_tla.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(82_ascend950_basic_syrk cube basic_syrk_tla.cpp)
target_compile_definitions(82_ascend950_basic_syrk PRIVATE L2_CACHE_HINT)
target_link_libraries(82_ascend950_basic_syrk PRIVATE m)
