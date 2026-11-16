#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// Keep the Ascend C register contract, including b8 even slots and the b64
// two-store implementation. The memref offset is measured in data elements.
#define REGISTER_SCATTER(Data, Index, Name)                                             \
    extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_scatter_##Name( \
        memref_t<__ubuf__ Data, 1>* dst, AscendC::Reg::RegTensor<Data>::RegType src,    \
        AscendC::Reg::RegTensor<Index>::RegType indices, ave_preg preg)                 \
    {                                                                                   \
        AscendC::Reg::RegTensor<Data> srcReg;                                           \
        AscendC::Reg::RegTensor<Index> indexReg;                                        \
        srcReg.reg[0] = src;                                                            \
        indexReg.reg[0] = indices;                                                      \
        auto mask = convertAVEPregToVecBool(preg);                                      \
        AscendC::Reg::Scatter(dst->aligned + dst->offset, srcReg, indexReg, mask);      \
    }

REGISTER_SCATTER(int8_t, uint16_t, int8_t_u16)
REGISTER_SCATTER(uint8_t, uint16_t, uint8_t_u16)
REGISTER_SCATTER(int16_t, uint16_t, int16_t_u16)
REGISTER_SCATTER(uint16_t, uint16_t, uint16_t_u16)
REGISTER_SCATTER(half, uint16_t, half_u16)
REGISTER_SCATTER(bfloat16_t, uint16_t, bfloat16_t_u16)
REGISTER_SCATTER(int32_t, uint32_t, int32_t_u32)
REGISTER_SCATTER(uint32_t, uint32_t, uint32_t_u32)
REGISTER_SCATTER(float, uint32_t, float_u32)
REGISTER_SCATTER(int64_t, uint32_t, int64_t_u32)
REGISTER_SCATTER(int64_t, uint64_t, int64_t_u64)
REGISTER_SCATTER(uint64_t, uint32_t, uint64_t_u32)
REGISTER_SCATTER(uint64_t, uint64_t, uint64_t_u64)

#undef REGISTER_SCATTER

#endif
