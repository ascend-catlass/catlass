#include "../common.h"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
#include "../print_scalar.h"
extern "C" {
#define TLA_AIV_SCALAR_ALIAS(NAME, TYPE)                                                    \
    [aicore] __attribute__((always_inline)) void NAME##_aiv(TYPE value, uint64_t workspace) \
    {                                                                                       \
        NAME(value, workspace);                                                             \
    }
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_i8, int8_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_i16, int16_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_i32, int32_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_u8, uint8_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_u16, uint16_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_x_u32, uint32_t)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_v_f16, half)
TLA_AIV_SCALAR_ALIAS(_mlir_ciface_tla_printf_v_f32, float)
#undef TLA_AIV_SCALAR_ALIAS
[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_string_aiv(
    const char* format, uint64_t length, uint64_t workspace)
{
    _mlir_ciface_tla_printf_format_string(format, length, workspace);
}
[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_values_aiv(
    const char* format, uint64_t length, uint64_t count, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
    uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7, uint64_t workspace)
{
    _mlir_ciface_tla_printf_format_values(format, length, count, a0, a1, a2, a3, a4, a5, a6, a7, workspace);
}
}
#endif
