add_kernel(NAME ascend950_matmul_gather_scatter
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_matmul_gather_scatter.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_matmul_gather_scatter_impl.cpp)
