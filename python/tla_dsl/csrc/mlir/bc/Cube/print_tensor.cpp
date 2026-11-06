#include "../common.h"
#include "../dma_common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"
#include "../print_tensor_workspace.h"
#include "../print_fifo.h"

namespace {
// CANN's 1 MiB debug FIFO reserves 48 bytes for the shape TLV and 72 bytes
// for the tensor TLV. Its 32-byte payload alignment leaves 262112 f32 values.
constexpr uint64_t kMaxFloat32Elements = 262112;
constexpr uint32_t kL1PrintTensorMaxElements = 8;
constexpr uint32_t kL0CPrintTensorElements = 256;

__aicore__ inline AscendC::ShapeInfo MakeShapeInfo(uint32_t shape0, uint32_t shape1)
{
    uint64_t rank = shape1 == 0U ? 1 : 2;
    uint32_t shape[2] = {shape0, shape1};
    AscendC::ShapeInfo shapeInfo(static_cast<uint8_t>(rank), shape);
    for (uint64_t index = rank; index < K_MAX_SHAPE_DIM; ++index) {
        shapeInfo.shape[index] = 0U;
        shapeInfo.originalShape[index] = 0U;
    }
    return shapeInfo;
}

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
static_assert(
    sizeof(decltype(__asc_aicore::DebugBlockHeadInfo::debugBusAddr)) == sizeof(uint64_t),
    "L1 tensor printing requires the C310 debug-bus FIFO ABI");
template <typename ElementType>
[aicore] __attribute__((always_inline)) void printTensor(
    uint64_t print_workspace, uint64_t tensor_address, uint64_t count, uint64_t packed_shape, uint64_t call_id)
{
    int32_t signedShape0 = static_cast<int32_t>(packed_shape);
    int32_t signedShape1 = static_cast<int32_t>(packed_shape >> 32);
    if (signedShape0 <= 0 || signedShape1 < 0 || count == 0 || count > kMaxFloat32Elements ||
        (tensor_address & (sizeof(ElementType) - 1U)) != 0U ||
        count > static_cast<uint64_t>(signedShape0) * (signedShape1 == 0 ? 1U : static_cast<uint32_t>(signedShape1)))
        return;

    tla::print_tensor::InitializeWorkspace(print_workspace);
    // Preflight before CANN's waiting ring-buffer path. Each physical core has
    // a single scalar writer, including the VF drainer.
    auto* fifo = tla::debug::GetFifo(print_workspace);
    uint64_t recordBytes = 72 + ((count * sizeof(ElementType) + 31) & ~uint64_t{31});
    if (!tla::debug::CanWrite(fifo, recordBytes))
        return;
    AscendC::ShapeInfo shapeInfo =
        MakeShapeInfo(static_cast<uint32_t>(signedShape0), static_cast<uint32_t>(signedShape1));
    AscendC::GlobalTensor<ElementType> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementType*>(tensor_address), count);
    AscendC::DumpTensor(
        tensor[0], tla::print_tensor::EncodeDescriptor(call_id), static_cast<uint32_t>(count), shapeInfo);

    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
}

template <typename ElementType>
[aicore] __attribute__((always_inline)) void printTensorL1(
    uint64_t print_workspace, memref_t<__cbuf__ ElementType, 1>* tensor_memref, uint64_t byte_offset, uint64_t count,
    uint64_t packed_shape, uint64_t call_id)
{
    int32_t signedShape0 = static_cast<int32_t>(packed_shape);
    int32_t signedShape1 = static_cast<int32_t>(packed_shape >> 32);
    if (signedShape0 <= 0 || signedShape1 <= 0 || count == 0 || count > kL1PrintTensorMaxElements ||
        count > static_cast<uint64_t>(signedShape0) * static_cast<uint32_t>(signedShape1))
        return;

    uint32_t base_address = localAddr(tensor_memref);
    if (byte_offset > UINT32_MAX || byte_offset > static_cast<uint64_t>(UINT32_MAX - base_address))
        return;
    uint32_t tensor_address = base_address + static_cast<uint32_t>(byte_offset);
    if ((tensor_address & 31U) != 0U)
        return;

    tla::print_tensor::InitializeWorkspace(print_workspace);
    // Preflight before CANN's waiting ring-buffer path. Each physical core has
    // a single scalar writer, including the VF drainer.
    auto* fifo = tla::debug::GetFifo(print_workspace);
    uint64_t recordBytes = 72 + ((count * sizeof(ElementType) + 31) & ~uint64_t{31});
    if (!tla::debug::CanWrite(fifo, recordBytes))
        return;
    auto* print_address = reinterpret_cast<__gm__ uint8_t*>(print_workspace);
    g_sysPrintFifoSpace = print_address;
    uint32_t shape[2] = {static_cast<uint32_t>(signedShape0), static_cast<uint32_t>(signedShape1)};
    AscendC::ShapeInfo shapeInfo(2, shape);
    AscendC::LocalTensor<ElementType> tensor(AscendC::TPosition::A1, tensor_address, kL1PrintTensorMaxElements);
    AscendC::DumpTensor(
        tensor[0], tla::print_tensor::EncodeDescriptor(call_id), static_cast<uint32_t>(count), shapeInfo);

    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
}

template <typename ElementType>
[aicore] __attribute__((always_inline)) void printTensorL0C(
    uint64_t print_workspace, memref_t<__cc__ ElementType, 1>* tensor_memref, uint64_t byte_offset, uint64_t count,
    uint64_t packed_shape, uint64_t call_id)
{
    int32_t signedShape0 = static_cast<int32_t>(packed_shape);
    int32_t signedShape1 = static_cast<int32_t>(packed_shape >> 32);
    if (signedShape0 != 16 || signedShape1 != 16 || count != kL0CPrintTensorElements || byte_offset > UINT32_MAX)
        return;

    uint32_t base_address = localAddr(tensor_memref);
    if (byte_offset > static_cast<uint64_t>(UINT32_MAX - base_address))
        return;
    uint32_t tensor_address = base_address + static_cast<uint32_t>(byte_offset);
    if ((tensor_address & 31U) != 0U)
        return;

    tla::print_tensor::InitializeWorkspace(print_workspace);
    // Preflight before CANN's waiting ring-buffer path. Each physical core has
    // a single scalar writer, including the VF drainer.
    auto* fifo = tla::debug::GetFifo(print_workspace);
    // C310 Fixpipe copies a complete 16x16 f32 tile even for a smaller typed view.
    uint64_t recordBytes = 72 + 256 * sizeof(float);
    if (!tla::debug::CanWrite(fifo, recordBytes))
        return;
    auto* print_address = reinterpret_cast<__gm__ uint8_t*>(print_workspace);
    g_sysPrintFifoSpace = print_address;
    uint32_t shape[2] = {16U, 16U};
    AscendC::ShapeInfo shapeInfo(2, shape);
    AscendC::LocalTensor<ElementType> tensor(AscendC::TPosition::CO1, tensor_address, kL0CPrintTensorElements);
    AscendC::DumpTensor(tensor[0], tla::print_tensor::EncodeDescriptor(call_id), kL0CPrintTensorElements, shapeInfo);

    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
}

#endif
} // namespace

extern "C" {

__attribute__((used, section(".tla_print_tensor_abi"))) const char tla_print_tensor_abi_shared_v2[] =
    "tla_print_tensor_abi_shared_v2";

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
#define TLA_PRINT_TENSOR_WRAPPER(SUFFIX, TYPE)                                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_tla_print_tensor_gm_##SUFFIX(            \
        uint64_t workspace, uint64_t address, uint64_t count, uint64_t packed_shape, uint64_t call_id) \
    {                                                                                                  \
        printTensor<TYPE>(workspace, address, count, packed_shape, call_id);                           \
    }
TLA_PRINT_TENSOR_WRAPPER(f16, half)
TLA_PRINT_TENSOR_WRAPPER(f32, float)
TLA_PRINT_TENSOR_WRAPPER(i8, int8_t)
TLA_PRINT_TENSOR_WRAPPER(i16, int16_t)
TLA_PRINT_TENSOR_WRAPPER(i32, int32_t)
TLA_PRINT_TENSOR_WRAPPER(u8, uint8_t)
TLA_PRINT_TENSOR_WRAPPER(u16, uint16_t)
TLA_PRINT_TENSOR_WRAPPER(u32, uint32_t)
#undef TLA_PRINT_TENSOR_WRAPPER

#define TLA_PRINT_TENSOR_L1_WRAPPER(SUFFIX, TYPE)                                                            \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_tla_print_tensor_l1_##SUFFIX(                  \
        uint64_t workspace, memref_t<__cbuf__ TYPE, 1>* tensor_memref, uint64_t byte_offset, uint64_t count, \
        uint64_t packed_shape, uint64_t call_id)                                                             \
    {                                                                                                        \
        printTensorL1<TYPE>(workspace, tensor_memref, byte_offset, count, packed_shape, call_id);            \
    }
TLA_PRINT_TENSOR_L1_WRAPPER(f16, half)
TLA_PRINT_TENSOR_L1_WRAPPER(f32, float)
TLA_PRINT_TENSOR_L1_WRAPPER(i8, int8_t)
TLA_PRINT_TENSOR_L1_WRAPPER(i16, int16_t)
TLA_PRINT_TENSOR_L1_WRAPPER(i32, int32_t)
TLA_PRINT_TENSOR_L1_WRAPPER(u8, uint8_t)
TLA_PRINT_TENSOR_L1_WRAPPER(u16, uint16_t)
TLA_PRINT_TENSOR_L1_WRAPPER(u32, uint32_t)
#undef TLA_PRINT_TENSOR_L1_WRAPPER

#define TLA_PRINT_TENSOR_L0C_WRAPPER(SUFFIX, TYPE)                                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_tla_print_tensor_l0c_##SUFFIX(               \
        uint64_t workspace, memref_t<__cc__ TYPE, 1>* tensor_memref, uint64_t byte_offset, uint64_t count, \
        uint64_t packed_shape, uint64_t call_id)                                                           \
    {                                                                                                      \
        printTensorL0C<TYPE>(workspace, tensor_memref, byte_offset, count, packed_shape, call_id);         \
    }
TLA_PRINT_TENSOR_L0C_WRAPPER(f32, float)
TLA_PRINT_TENSOR_L0C_WRAPPER(f16, half)
TLA_PRINT_TENSOR_L0C_WRAPPER(i8, int8_t)
TLA_PRINT_TENSOR_L0C_WRAPPER(i16, int16_t)
TLA_PRINT_TENSOR_L0C_WRAPPER(i32, int32_t)
TLA_PRINT_TENSOR_L0C_WRAPPER(u8, uint8_t)
TLA_PRINT_TENSOR_L0C_WRAPPER(u16, uint16_t)
TLA_PRINT_TENSOR_L0C_WRAPPER(u32, uint32_t)
#undef TLA_PRINT_TENSOR_L0C_WRAPPER

#endif

} // extern "C"
