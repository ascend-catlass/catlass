// Thin register-only bridges for encodings absent from MLIR's builtin types.
// Attributes select an entry point at lowering time; only data and mask are
// runtime arguments. Layout and scale remain explicit in the DSL.
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "vector_reg_utils.h"

#define REGISTER_HIF8_NARROW(NAME, ROUND, SAT, SLOT)                                                                 \
    extern "C" __aiv__                                                                                               \
        __attribute__((always_inline)) vector_u8 lowp_cast_f32_to_hif8_##NAME(vector_f32 source, ave_preg predicate) \
    {                                                                                                                \
        vector_hif8 result;                                                                                          \
        vcvt(result, source, convertAVEPregToVecBool(predicate), ROUND, SAT, SLOT, MODE_ZEROING);                    \
        return reinterpret_cast<vector_u8&>(result);                                                                 \
    }

#define REGISTER_HIF8_NARROW_SLOTS(ROUND_NAME, ROUND, SAT_NAME, SAT)           \
    REGISTER_HIF8_NARROW(ROUND_NAME##_##SAT_NAME##_slot0, ROUND, SAT, PART_P0) \
    REGISTER_HIF8_NARROW(ROUND_NAME##_##SAT_NAME##_slot1, ROUND, SAT, PART_P1) \
    REGISTER_HIF8_NARROW(ROUND_NAME##_##SAT_NAME##_slot2, ROUND, SAT, PART_P2) \
    REGISTER_HIF8_NARROW(ROUND_NAME##_##SAT_NAME##_slot3, ROUND, SAT, PART_P3)

#define REGISTER_HIF8_WIDEN(NAME, SLOT)                                                                               \
    extern "C" __aiv__                                                                                                \
        __attribute__((always_inline)) vector_f32 lowp_cast_hif8_to_f32_##NAME(vector_u8 source, ave_preg predicate)  \
    {                                                                                                                 \
        vector_f32 result;                                                                                            \
        vcvt(result, reinterpret_cast<vector_hif8&>(source), convertAVEPregToVecBool(predicate), SLOT, MODE_ZEROING); \
        return result;                                                                                                \
    }

REGISTER_HIF8_NARROW_SLOTS(round, ROUND_A, sat, RS_ENABLE)
REGISTER_HIF8_NARROW_SLOTS(round, ROUND_A, nosat, RS_DISABLE)
REGISTER_HIF8_NARROW_SLOTS(hybrid, ROUND_H, sat, RS_ENABLE)
REGISTER_HIF8_NARROW_SLOTS(hybrid, ROUND_H, nosat, RS_DISABLE)
REGISTER_HIF8_WIDEN(slot0, PART_P0)
REGISTER_HIF8_WIDEN(slot1, PART_P1)
REGISTER_HIF8_WIDEN(slot2, PART_P2)
REGISTER_HIF8_WIDEN(slot3, PART_P3)

#undef REGISTER_HIF8_WIDEN
#undef REGISTER_HIF8_NARROW_SLOTS
#undef REGISTER_HIF8_NARROW
#endif
