// Thin register-only bridges for encodings absent from MLIR's builtin types.
// Attributes select an entry point at lowering time; only data and mask are
// runtime arguments. Layout and scale remain explicit in the DSL.
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "vector_reg_utils.h"

#define REGISTER_HIF8_NARROW(FROM, NAME, ROUND, SAT, SLOT)                                         \
    extern "C" __aiv__ __attribute__((always_inline)) vector_u8 lowp_cast_##FROM##_to_hif8_##NAME( \
        vector_##FROM source, ave_preg predicate)                                                  \
    {                                                                                              \
        vector_hif8 result;                                                                        \
        vcvt(result, source, convertAVEPregToVecBool(predicate), ROUND, SAT, SLOT, MODE_ZEROING);  \
        return reinterpret_cast<vector_u8&>(result);                                               \
    }

#define REGISTER_HIF8_NARROW_SLOTS(ROUND_NAME, ROUND, SAT_NAME, SAT)                  \
    REGISTER_HIF8_NARROW(f32, ROUND_NAME##_##SAT_NAME##_slot0, ROUND, SAT, PART_P0)   \
    REGISTER_HIF8_NARROW(f32, ROUND_NAME##_##SAT_NAME##_slot1, ROUND, SAT, PART_P1)   \
    REGISTER_HIF8_NARROW(f32, ROUND_NAME##_##SAT_NAME##_slot2, ROUND, SAT, PART_P2)   \
    REGISTER_HIF8_NARROW(f32, ROUND_NAME##_##SAT_NAME##_slot3, ROUND, SAT, PART_P3)   \
    REGISTER_HIF8_NARROW(f16, ROUND_NAME##_##SAT_NAME##_slot0, ROUND, SAT, PART_EVEN) \
    REGISTER_HIF8_NARROW(f16, ROUND_NAME##_##SAT_NAME##_slot1, ROUND, SAT, PART_ODD)

#define REGISTER_HIF8_WIDEN(TO, NAME, SLOT)                                                                           \
    extern "C" __aiv__ __attribute__((always_inline)) vector_##TO lowp_cast_hif8_to_##TO##_##NAME(                    \
        vector_u8 source, ave_preg predicate)                                                                         \
    {                                                                                                                 \
        vector_##TO result;                                                                                           \
        vcvt(result, reinterpret_cast<vector_hif8&>(source), convertAVEPregToVecBool(predicate), SLOT, MODE_ZEROING); \
        return result;                                                                                                \
    }

REGISTER_HIF8_NARROW_SLOTS(round, ROUND_A, sat, RS_ENABLE)
REGISTER_HIF8_NARROW_SLOTS(round, ROUND_A, nosat, RS_DISABLE)
REGISTER_HIF8_NARROW_SLOTS(hybrid, ROUND_H, sat, RS_ENABLE)
REGISTER_HIF8_NARROW_SLOTS(hybrid, ROUND_H, nosat, RS_DISABLE)
REGISTER_HIF8_WIDEN(f32, slot0, PART_P0)
REGISTER_HIF8_WIDEN(f32, slot1, PART_P1)
REGISTER_HIF8_WIDEN(f32, slot2, PART_P2)
REGISTER_HIF8_WIDEN(f32, slot3, PART_P3)
REGISTER_HIF8_WIDEN(f16, slot0, PART_EVEN)
REGISTER_HIF8_WIDEN(f16, slot1, PART_ODD)

#undef REGISTER_HIF8_WIDEN
#undef REGISTER_HIF8_NARROW_SLOTS
#undef REGISTER_HIF8_NARROW

// The packed result uses the same byte carrier as the Int4x2 load/store path.
#define REGISTER_I16_TO_I4X2(NAME, SAT, SLOT)                                                                        \
    extern "C" __aiv__                                                                                               \
        __attribute__((always_inline)) vector_u8 lowp_cast_i16_to_i4x2_##NAME(vector_s16 source, ave_preg predicate) \
    {                                                                                                                \
        vector_s4x2 result;                                                                                          \
        vcvt_s162s4(result, source, convertAVEPregToVecBool(predicate), SAT, SLOT, MODE_ZEROING);                    \
        return reinterpret_cast<vector_u8&>(result);                                                                 \
    }

#define REGISTER_I16_TO_I4X2_SLOTS(NAME, SAT)        \
    REGISTER_I16_TO_I4X2(NAME##_slot0, SAT, PART_P0) \
    REGISTER_I16_TO_I4X2(NAME##_slot1, SAT, PART_P1) \
    REGISTER_I16_TO_I4X2(NAME##_slot2, SAT, PART_P2) \
    REGISTER_I16_TO_I4X2(NAME##_slot3, SAT, PART_P3)

REGISTER_I16_TO_I4X2_SLOTS(sat, RS_ENABLE)
REGISTER_I16_TO_I4X2_SLOTS(nosat, RS_DISABLE)

#undef REGISTER_I16_TO_I4X2_SLOTS
#undef REGISTER_I16_TO_I4X2

#define REGISTER_F16_TO_I4X2(NAME, ROUND, SAT, SLOT)                                                                 \
    extern "C" __aiv__                                                                                               \
        __attribute__((always_inline)) vector_u8 lowp_cast_f16_to_i4x2_##NAME(vector_f16 source, ave_preg predicate) \
    {                                                                                                                \
        vector_s4x2 result;                                                                                          \
        vcvt_f162s4(result, source, convertAVEPregToVecBool(predicate), ROUND, SAT, SLOT, MODE_ZEROING);             \
        return reinterpret_cast<vector_u8&>(result);                                                                 \
    }

#define REGISTER_F16_TO_I4X2_SLOTS(NAME, ROUND, SAT_NAME, SAT)           \
    REGISTER_F16_TO_I4X2(NAME##_##SAT_NAME##_slot0, ROUND, SAT, PART_P0) \
    REGISTER_F16_TO_I4X2(NAME##_##SAT_NAME##_slot1, ROUND, SAT, PART_P1) \
    REGISTER_F16_TO_I4X2(NAME##_##SAT_NAME##_slot2, ROUND, SAT, PART_P2) \
    REGISTER_F16_TO_I4X2(NAME##_##SAT_NAME##_slot3, ROUND, SAT, PART_P3)

#define REGISTER_F16_TO_I4X2_ROUND(NAME, ROUND)             \
    REGISTER_F16_TO_I4X2_SLOTS(NAME, ROUND, sat, RS_ENABLE) \
    REGISTER_F16_TO_I4X2_SLOTS(NAME, ROUND, nosat, RS_DISABLE)

REGISTER_F16_TO_I4X2_ROUND(round, ROUND_A)
REGISTER_F16_TO_I4X2_ROUND(floor, ROUND_F)
REGISTER_F16_TO_I4X2_ROUND(ceil, ROUND_C)
REGISTER_F16_TO_I4X2_ROUND(trunc, ROUND_Z)
REGISTER_F16_TO_I4X2_ROUND(rint, ROUND_R)

#undef REGISTER_F16_TO_I4X2_ROUND
#undef REGISTER_F16_TO_I4X2_SLOTS
#undef REGISTER_F16_TO_I4X2

template <typename Slot>
__aiv__ __attribute__((always_inline)) vector_bf16 widenE8M0(vector_u8 source, ave_preg predicate, Slot slot)
{
    const vector_bool sourceMask = convertAVEPregToVecBool(predicate);
    vector_u16 exponent;
    vcvt(exponent, source, sourceMask, slot, MODE_ZEROING);

    // Mark selected 0x00 bytes before widening so masked-off bytes stay zero.
    vector_bool minScaleMask;
    vcmps_eq(minScaleMask, source, uint8_t{0}, sourceMask);
    vector_u8 minScaleBytes;
    vdup(minScaleBytes, uint8_t{0x40}, minScaleMask, MODE_ZEROING);
    vector_u16 minScaleBits;
    vcvt(minScaleBits, minScaleBytes, sourceMask, slot, MODE_ZEROING);

    // The width-changing cast already zeroed inactive lanes. Use a B16
    // predicate for the remaining steps so an odd source slot stays active.
    const vector_bool allB16 = pge_b16(PAT_ALL);
    vector_u16 bits;
    vshls(bits, exponent, int16_t{7}, allB16, MODE_ZEROING);
    vor(bits, bits, minScaleBits, allB16, MODE_ZEROING);
    vector_bool nanMask;
    vcmps_eq(nanMask, bits, uint16_t{0x7f80}, allB16);
    vdup(bits, uint16_t{0x7fc0}, nanMask, MODE_MERGING);
    return reinterpret_cast<vector_bf16&>(bits);
}

extern "C" __aiv__ __attribute__((always_inline)) vector_bf16
lowp_cast_e8m0_to_bf16_slot0(vector_u8 source, ave_preg predicate)
{
    return widenE8M0(source, predicate, PART_EVEN);
}

extern "C" __aiv__ __attribute__((always_inline)) vector_bf16
lowp_cast_e8m0_to_bf16_slot1(vector_u8 source, ave_preg predicate)
{
    return widenE8M0(source, predicate, PART_ODD);
}

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
