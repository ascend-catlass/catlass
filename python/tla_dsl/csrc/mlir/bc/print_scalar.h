#pragma once

#include "print_fifo.h"

extern "C" {
#if defined(__DAV_VEC__)
#define TLA_SCALAR_PREFIX "[AIV Block %u/%u] "
#else
#define TLA_SCALAR_PREFIX "[AIC Block %u/%u] "
#endif
#define TLA_PRINT_INTEGER(SUFFIX, TYPE, FORMAT, WIDE)                                                               \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_##SUFFIX(TYPE value, uint64_t workspace) \
    {                                                                                                               \
        uint64_t arguments[3] = {                                                                                   \
            __asc_aicore::asc_debug_get_block_idx(), __asc_aicore::asc_debug_get_block_total_num(),                 \
            static_cast<uint64_t>(static_cast<WIDE>(value))};                                                       \
        tla::debug::WriteFormat(                                                                                    \
            TLA_SCALAR_PREFIX FORMAT, sizeof(TLA_SCALAR_PREFIX FORMAT) - 1, 3, arguments, workspace);               \
    }
TLA_PRINT_INTEGER(i8, int8_t, "x=%d", int64_t)
TLA_PRINT_INTEGER(i16, int16_t, "x=%d", int64_t)
TLA_PRINT_INTEGER(i32, int32_t, "x=%d", int64_t)
TLA_PRINT_INTEGER(u8, uint8_t, "x=%u", uint64_t)
TLA_PRINT_INTEGER(u16, uint16_t, "x=%u", uint64_t)
TLA_PRINT_INTEGER(u32, uint32_t, "x=%u", uint64_t)
#undef TLA_PRINT_INTEGER

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_v_f32(float value, uint64_t workspace)
{
    union {
        float value;
        uint32_t bits;
    } encoded{value};
    uint64_t arguments[3] = {
        __asc_aicore::asc_debug_get_block_idx(), __asc_aicore::asc_debug_get_block_total_num(), encoded.bits};
    tla::debug::WriteFormat(TLA_SCALAR_PREFIX "v=%f", sizeof(TLA_SCALAR_PREFIX "v=%f") - 1, 3, arguments, workspace);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_v_f16(half value, uint64_t workspace)
{
    _mlir_ciface_tla_printf_v_f32(static_cast<float>(value), workspace);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_string(
    const char* format, uint64_t length, uint64_t workspace)
{
    tla::debug::WriteFormat(format, length, 0, nullptr, workspace);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_values(
    const char* format, uint64_t length, uint64_t count, uint64_t value0, uint64_t value1, uint64_t value2,
    uint64_t value3, uint64_t value4, uint64_t value5, uint64_t value6, uint64_t value7, uint64_t workspace)
{
    uint64_t arguments[8] = {value0, value1, value2, value3, value4, value5, value6, value7};
    tla::debug::WriteFormat(format, length, count, arguments, workspace);
}
}

#undef TLA_SCALAR_PREFIX
