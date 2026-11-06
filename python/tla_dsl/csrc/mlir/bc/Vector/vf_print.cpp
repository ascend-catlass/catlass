#include "../common.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && __has_include("utils/debug/asc_printf.h")
#include "c_api/asc_simd.h"
#include "utils/base/helpers.h"
#include "utils/debug/asc_dump.h"
#include "utils/debug/asc_printf.h"
#include "vector_reg_utils.h"
#include "../print_fifo.h"

namespace {
template <typename... Args>
__simd_callee__ inline void printScalars(__ubuf__ const char* format, Args... args)
{
    using namespace __asc_simd_vf;
    enable_asc_diagnostics();
    // Reuse the reserved format's terminator as the empty prefix. A separate
    // C++ "" literal would allocate UB outside TLA's static/dynamic plan.
    auto* empty = format + get_cstring_len_vf(format) - 1;
    scalar_printf_impl(DumpType::DUMP_SCALAR, format, empty, args...);
}

template <typename T, bool IsRegister, typename V>
__simd_callee__ inline void dumpPrefix(
    V& value, uint32_t count, uint32_t shape0, uint32_t shape1, uint32_t rank, uint32_t callId)
{
    using namespace __asc_simd_vf;
    // Printing observes preceding vector stores before scalar UB reads.
    asc_mem_bar(VS_ALL);
    constexpr auto position = IsRegister ? DumpTensorPosition::REG : DumpTensorPosition::UB;
    auto* info = get_printf_ubuf_addr(0);
    uint32_t length = reserve_dump_tlv<T>(info, count);
    if (length == 0)
        return;
    auto* record = reinterpret_cast<__ubuf__ DumpTensorTlv*>(info->buffer + info->writeLen);
    uint32_t alignedBytes = length - sizeof(DumpTensorTlv);
    set_dump_tlv_info_vf<position, T>(value, record, alignedBytes, 0x54540000U | callId, count, info->blockIdx);
    // Supply the same descriptor namespace and embedded shape as the existing
    // TLA FIFO decoder. Complete metadata before publishing writeLen to the
    // scalar-side drainer, which runs concurrently with the vector function.
    record->dim = rank;
    record->shape[0] = shape0;
    record->shape[1] = rank == 2 ? shape1 : 0;
    if constexpr (IsRegister) {
        asc_mem_bar(SV_ALL);
        set_dump_tlv_data_reg<T>(value, record, alignedBytes, count);
        // The scalar drainer may consume a record as soon as it is published.
        asc_mem_bar(VS_ALL);
    } else {
        set_dump_tlv_data_ubuf<T>(value, record, alignedBytes, count);
    }
    info->writeLen += length;
    asc_mem_bar(SV_ALL);
}
// The host reads logs after stream completion. CANN's default drainer retries
// when the GM ring is full, expecting a concurrent RTS consumer. Keep its UB
// handshake but publish whole TLVs without waiting for such a consumer.
__aicore__ inline void drainVf()
{
    auto* info = get_printf_ubuf_addr_aicore(0);
    __ubuf__ volatile BlockVFBufInfo* state = info;
    for (;;) {
        uint32_t read = state->readLen;
        uint32_t written = state->writeLen;
        if (state->magic != ASCENDC_SIMD_VF_MAGIC_NUMBER || state->length > ASCENDC_SIMD_VF_PRINTF_UBUF_MAX_SIZE ||
            written > state->length) {
            info->flag = 1;
        } else if (read > written) {
            // The VF resets writeLen before readLen after a drained batch.
            continue;
        } else if (read < written) {
            auto* fifo = __asc_aicore::get_block_info();
            while (read < written) {
                auto* source = reinterpret_cast<__ubuf__ uint8_t*>(info->buffer) + read;
                if (written - read < 8) {
                    info->flag = 1;
                    break;
                }
                uint32_t length = *reinterpret_cast<__ubuf__ uint32_t*>(source + 4);
                if (!length || length > written - read - 8) {
                    info->flag = 1;
                    break;
                }
                uint32_t total = 8 + length;
                if (!tla::debug::CanWrite(fifo, total)) {
                    info->flag = 1;
                    break;
                }
                auto* destination = __asc_aicore::get_ringbuf_tlv_addr(fifo);
                for (uint32_t index = 0; index < total; ++index)
                    destination[index] = source[index];
                __asc_aicore::asc_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(destination));
                __asc_aicore::update_write_info(__asc_aicore::get_block_write_info(fifo), total);
                read += total;
            }
            info->readLen = written;
            if (state->flag == 0)
                continue;
        } else if (state->finish != 1) {
            continue;
        }
        // Release a producer that observes flag, and wait for the finish VF.
        pipe_barrier(PIPE_ALL);
        if (state->flag == 0 && state->readLen < state->writeLen)
            continue;
        break;
    }
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void _mlir_ciface_tla_vf_debug_begin(uint64_t workspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(workspace);
    AscVFDebugInitUb();
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_tla_vf_debug_end()
{
    AscVFDebugFinish();
    drainVf();
}

// Slots use the existing debug FIFO ABI: sign/zero-extended integers and
// float32 bits in the low word. Emit exactly the live slots so the host can
// validate the format/argument count without accepting corrupt trailing slots.
__aicore__ __attribute__((always_inline)) void _mlir_ciface_tla_vf_printf(
    __ubuf__ const char* format, uint32_t count, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
    uint64_t a5, uint64_t a6, uint64_t a7)
{
    switch (count) {
        case 0:
            printScalars(format);
            break;
        case 1:
            printScalars(format, a0);
            break;
        case 2:
            printScalars(format, a0, a1);
            break;
        case 3:
            printScalars(format, a0, a1, a2);
            break;
        case 4:
            printScalars(format, a0, a1, a2, a3);
            break;
        case 5:
            printScalars(format, a0, a1, a2, a3, a4);
            break;
        case 6:
            printScalars(format, a0, a1, a2, a3, a4, a5);
            break;
        case 7:
            printScalars(format, a0, a1, a2, a3, a4, a5, a6);
            break;
        case 8:
            printScalars(format, a0, a1, a2, a3, a4, a5, a6, a7);
            break;
    }
}

#define REGISTER_VF_DUMP(T, suffix)                                                                                   \
    __aicore__                                                                                                        \
        __attribute__((always_inline)) void tla_vf_print_reg_##suffix(VectorReg<T> value, int32_t n, uint32_t callId) \
    {                                                                                                                 \
        if (n > 0 && n <= 256 / sizeof(T))                                                                            \
            dumpPrefix<T, true>(value, static_cast<uint32_t>(n), n, 0, 1, callId);                                    \
    }                                                                                                                 \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_tla_vf_print_ub_##suffix(                             \
        memref_t<__ubuf__ T, 1>* value, int32_t n, int32_t shape0, int32_t shape1, int32_t rank, uint32_t callId)     \
    {                                                                                                                 \
        auto* address = value->aligned + value->offset;                                                               \
        if (n > 0)                                                                                                    \
            dumpPrefix<T, false>(address, static_cast<uint32_t>(n), shape0, shape1, rank, callId);                    \
    }

REGISTER_VF_DUMP(float, f32)
REGISTER_VF_DUMP(half, f16)
REGISTER_VF_DUMP(int8_t, i8)
REGISTER_VF_DUMP(uint8_t, u8)
REGISTER_VF_DUMP(int16_t, i16)
REGISTER_VF_DUMP(uint16_t, u16)
REGISTER_VF_DUMP(int32_t, i32)
REGISTER_VF_DUMP(uint32_t, u32)
#undef REGISTER_VF_DUMP
} // extern "C"
#endif
