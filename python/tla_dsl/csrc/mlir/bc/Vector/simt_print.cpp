#include <cstdint>
#include "simt_api/asc_simt.h"

// Private protocol shared with ascend_simt_print.py. All sizes are bytes.
// A closed cursor retains the used-byte count in its low 31 bits. Closing and
// reserving use the same CAS, so no reservation can pass a failed allocation.
namespace {
constexpr uint32_t kClosed = 0x80000000U;
constexpr uint32_t kComplete = 0x544c4150U;
constexpr uint32_t kHeaderBytes = 64;

__simt_callee__ inline __gm__ uint32_t* reserve(uint64_t workspace, uint32_t bytes)
{
    auto control = reinterpret_cast<__gm__ uint32_t*>(workspace);
    uint32_t cursor = atomicCAS(control, 0U, 0U);
    for (;;) {
        if (cursor & kClosed) {
            atomicAdd(reinterpret_cast<__gm__ uint64_t*>(control + 2), uint64_t{1});
            return nullptr;
        }
        const bool fits = bytes <= control[1] - cursor;
        uint32_t next = fits ? cursor + bytes : cursor | kClosed;
        uint32_t observed = atomicCAS(control, cursor, next);
        if (observed != cursor) {
            cursor = observed;
            continue;
        }
        if (!fits) {
            atomicAdd(reinterpret_cast<__gm__ uint64_t*>(control + 2), uint64_t{1});
            return nullptr;
        }
        return reinterpret_cast<__gm__ uint32_t*>(workspace + kHeaderBytes + cursor);
    }
}

__simt_callee__ inline void header(
    __gm__ uint32_t* record, uint32_t bytes, uint32_t kind, uint32_t metadata, uint32_t count, uint32_t rows = 1,
    uint32_t columns = 1, uint32_t rank = 0)
{
    record[1] = bytes;
    record[2] = kind;
    record[3] = metadata;
    record[4] = blockIdx.x;
    record[5] = blockIdx.y;
    record[6] = blockIdx.z;
    record[7] = threadIdx.x;
    record[8] = threadIdx.y;
    record[9] = threadIdx.z;
    record[10] = count;
    record[11] = rows;
    record[12] = columns;
    record[13] = rank;
}

__simt_callee__ inline void publish(__gm__ uint32_t* record)
{
    __threadfence();
    atomicExch(record, kComplete);
}

template <typename Pointer>
__simt_callee__ inline void tensor(
    uint64_t workspace, Pointer source, int64_t requested, uint32_t extent, uint32_t dtype, uint32_t rows,
    uint32_t columns, uint32_t rank)
{
    if (requested <= 0)
        return;
    uint32_t count = requested > extent ? extent : static_cast<uint32_t>(requested);
    uint32_t width = (dtype == 0 || dtype == 3 || dtype == 8) ? 4 : (dtype == 1 || dtype == 6 || dtype == 7) ? 2 : 1;
    uint32_t bytes = kHeaderBytes + ((count * width + 7) & ~7U);
    auto record = reserve(workspace, bytes);
    if (!record)
        return;
    header(record, bytes, 2, dtype, count, rows, columns, rank);
    auto output = reinterpret_cast<__gm__ uint8_t*>(record + 16);
    for (uint32_t index = 0; index < count * width; ++index)
        output[index] = source[index];
    publish(record);
}
} // namespace

extern "C" __simt_callee__ __attribute__((always_inline)) void _mlir_ciface_tla_simt_printf(
    uint64_t workspace, uint32_t format, uint32_t count, uint64_t value0, uint64_t value1, uint64_t value2,
    uint64_t value3, uint64_t value4, uint64_t value5, uint64_t value6, uint64_t value7)
{
    auto record = reserve(workspace, 128);
    if (!record)
        return;
    header(record, 128, 1, format, count);
    auto values = reinterpret_cast<__gm__ uint64_t*>(record + 16);
    values[0] = value0;
    values[1] = value1;
    values[2] = value2;
    values[3] = value3;
    values[4] = value4;
    values[5] = value5;
    values[6] = value6;
    values[7] = value7;
    publish(record);
}

extern "C" __simt_callee__ __attribute__((always_inline)) void _mlir_ciface_tla_simt_print_gm(
    uint64_t workspace, __gm__ uint8_t* source, int64_t count, uint32_t extent, uint32_t dtype, uint32_t rows,
    uint32_t columns, uint32_t rank)
{
    tensor(workspace, source, count, extent, dtype, rows, columns, rank);
}

extern "C" __simt_callee__ __attribute__((always_inline)) void _mlir_ciface_tla_simt_print_ub(
    uint64_t workspace, __ubuf__ uint8_t* source, int64_t count, uint32_t extent, uint32_t dtype, uint32_t rows,
    uint32_t columns, uint32_t rank)
{
    tensor(workspace, source, count, extent, dtype, rows, columns, rank);
}
