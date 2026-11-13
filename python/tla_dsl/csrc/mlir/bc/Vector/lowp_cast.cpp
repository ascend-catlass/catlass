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

#define REGISTER_FP4_NARROW(FORMAT, NAME, ROUND, SLOT)                                               \
    extern "C" __aiv__ __attribute__((always_inline)) vector_u8 lowp_cast_bf16_to_##FORMAT##_##NAME( \
        vector_bf16 source, ave_preg predicate)                                                      \
    {                                                                                                \
        vector_##FORMAT result;                                                                      \
        vcvt(result, source, convertAVEPregToVecBool(predicate), ROUND, SLOT, MODE_ZEROING);         \
        return reinterpret_cast<vector_u8&>(result);                                                 \
    }

#define REGISTER_FP4_NARROW_SLOTS(FORMAT, NAME, ROUND)            \
    REGISTER_FP4_NARROW(FORMAT, NAME##_sat_slot0, ROUND, PART_P0) \
    REGISTER_FP4_NARROW(FORMAT, NAME##_sat_slot1, ROUND, PART_P1) \
    REGISTER_FP4_NARROW(FORMAT, NAME##_sat_slot2, ROUND, PART_P2) \
    REGISTER_FP4_NARROW(FORMAT, NAME##_sat_slot3, ROUND, PART_P3)

#define REGISTER_FP4_WIDEN(FORMAT, NAME, SLOT)                                                            \
    extern "C" __aiv__ __attribute__((always_inline)) vector_bf16 lowp_cast_##FORMAT##_to_bf16_##NAME(    \
        vector_u8 source, ave_preg predicate)                                                             \
    {                                                                                                     \
        vector_bf16 result;                                                                               \
        vcvt(                                                                                             \
            result, reinterpret_cast<vector_##FORMAT&>(source), convertAVEPregToVecBool(predicate), SLOT, \
            MODE_ZEROING);                                                                                \
        return result;                                                                                    \
    }

#define REGISTER_FP4_FORMAT(FORMAT)                   \
    REGISTER_FP4_NARROW_SLOTS(FORMAT, round, ROUND_A) \
    REGISTER_FP4_NARROW_SLOTS(FORMAT, floor, ROUND_F) \
    REGISTER_FP4_NARROW_SLOTS(FORMAT, ceil, ROUND_C)  \
    REGISTER_FP4_NARROW_SLOTS(FORMAT, trunc, ROUND_Z) \
    REGISTER_FP4_NARROW_SLOTS(FORMAT, rint, ROUND_R)  \
    REGISTER_FP4_WIDEN(FORMAT, slot0, PART_P0)        \
    REGISTER_FP4_WIDEN(FORMAT, slot1, PART_P1)        \
    REGISTER_FP4_WIDEN(FORMAT, slot2, PART_P2)        \
    REGISTER_FP4_WIDEN(FORMAT, slot3, PART_P3)

REGISTER_FP4_FORMAT(f4e2m1x2)
REGISTER_FP4_FORMAT(f4e1m2x2)

#undef REGISTER_FP4_FORMAT
#undef REGISTER_FP4_WIDEN
#undef REGISTER_FP4_NARROW_SLOTS
#undef REGISTER_FP4_NARROW
#endif
