add_kernel(NAME ascend950_89_ascend950_hif8_quant_batch_matmul
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/89_ascend950_hif8_quant_batch_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/89_ascend950_hif8_quant_batch_matmul_impl.cpp)
