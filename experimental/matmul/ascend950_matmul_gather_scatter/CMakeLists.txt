# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

set_source_files_properties(matmul_gather_scatter.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(ascend950_matmul_gather_scatter mix matmul_gather_scatter.cpp)
target_compile_definitions(ascend950_matmul_gather_scatter PRIVATE L2_CACHE_HINT)
# The fallback launches 32 x 32 SIMT threads. Keep its per-thread register budget explicit
# so future tuning does not silently change the validated launch configuration.
set(MATMUL_GATHER_SCATTER_SIMT_MAX_REG_COUNT 128)
target_compile_options(
    ascend950_matmul_gather_scatter
    PRIVATE --cce-simt-maxrregcount=${MATMUL_GATHER_SCATTER_SIMT_MAX_REG_COUNT})
