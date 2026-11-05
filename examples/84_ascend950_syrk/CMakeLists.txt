# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(syrk_tla.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(84_ascend950_syrk mix syrk_tla.cpp)
target_compile_definitions(84_ascend950_syrk PRIVATE L2_CACHE_HINT)
target_link_libraries(84_ascend950_syrk PRIVATE m)

set_source_files_properties(batched_basic_syrk.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(84_ascend950_batched_basic_syrk mix batched_basic_syrk.cpp)
target_link_libraries(84_ascend950_batched_basic_syrk PRIVATE m)

set_source_files_properties(syrk_polynomial.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(84_ascend950_syrk_polynomial mix syrk_polynomial.cpp)
target_link_libraries(84_ascend950_syrk_polynomial PRIVATE m)
