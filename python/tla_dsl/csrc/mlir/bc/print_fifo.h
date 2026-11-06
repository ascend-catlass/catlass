#pragma once

#include <cstdint>

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

// CANN TLVs and per-physical-core rings, with private overflow state after the
// complete CANN allocation. No CANN reserved header field is repurposed.
namespace tla::debug {
constexpr uint32_t kCoreRecords = 108;
// Isolate each scalar core's cache writeback from its neighbours' flags.
constexpr uint32_t kControlBytes = 128;

[aicore] __attribute__((always_inline)) inline __gm__ __asc_aicore::DebugBlockHeadInfo* GetFifo(uint64_t workspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(workspace);
    return __asc_aicore::get_block_info();
}

[aicore] __attribute__((always_inline)) inline bool CanWrite(
    __gm__ __asc_aicore::DebugBlockHeadInfo* info, uint64_t bytes)
{
    if (!info)
        return false;
    auto* control = reinterpret_cast<__gm__ uint32_t*>(
        g_sysPrintFifoSpace + static_cast<uint64_t>(info->length) * kCoreRecords +
        __asc_aicore::asc_debug_get_core_idx() * kControlBytes);
    auto* write = __asc_aicore::get_block_write_info(info);
    if (control[0] || write->bufOffset > info->ringBufLen || bytes > info->ringBufLen - write->bufOffset) {
        control[0] = 1;
        __asc_aicore::asc_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(control));
        return false;
    }
    return true;
}

template <typename FormatPointer>
[aicore] __attribute__((always_inline)) inline void WriteFormat(
    FormatPointer format, uint64_t formatLen, uint64_t count, const uint64_t* arguments, uint64_t workspace)
{
    constexpr uint64_t headerBytes = 24;
    if (!format || count > 8 || (count && !arguments) || formatLen > 1024 * 1024)
        return;
    uint64_t rawBytes = headerBytes + count * 8 + formatLen + 1;
    uint64_t bytes = (rawBytes + 7) & ~uint64_t{7};
    auto* info = GetFifo(workspace);
    if (!CanWrite(info, bytes))
        return;
    auto* record = __asc_aicore::get_ringbuf_tlv_addr(info);
    *reinterpret_cast<__gm__ uint32_t*>(record) = 1;
    *reinterpret_cast<__gm__ uint32_t*>(record + 4) = bytes - 8;
    *reinterpret_cast<__gm__ uint32_t*>(record + 8) = get_block_idx();
    *reinterpret_cast<__gm__ uint32_t*>(record + 12) = 0;
    *reinterpret_cast<__gm__ uint64_t*>(record + 16) = 8 + count * 8;
    for (uint64_t index = 0; index < count; ++index)
        *reinterpret_cast<__gm__ uint64_t*>(record + headerBytes + index * 8) = arguments[index];
    auto* text = record + headerBytes + count * 8;
    for (uint64_t index = 0; index < formatLen; ++index)
        text[index] = static_cast<uint8_t>(format[index]);
    text[formatLen] = 0;
    for (uint64_t index = rawBytes; index < bytes; ++index)
        record[index] = 0;
    __asc_aicore::asc_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(record));
    __asc_aicore::update_write_info(__asc_aicore::get_block_write_info(info), static_cast<uint32_t>(bytes));
}
} // namespace tla::debug
#endif
