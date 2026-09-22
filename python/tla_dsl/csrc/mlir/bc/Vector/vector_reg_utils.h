#pragma once

// Minimal vector/predicate register shims for tla_dsl bitcode ops.
// Mirrors the small subset of AscendNPU-IR RegBase/VecUtils.h used by squeeze.

#ifndef __aiv__
#define __aiv__ [aicore]
#endif

#define AVE_PREDICATE_LENGTH 256

#define DEFINE_AVE_PREG_TYPE(BASE_TYPE, NAME, SIZE) typedef BASE_TYPE NAME __attribute__((ext_vector_type(SIZE)))

DEFINE_AVE_PREG_TYPE(bool, ave_preg, AVE_PREDICATE_LENGTH);

template <typename T>
struct VectorTy;

template <>
struct VectorTy<float> {
    using type = vector_f32;
};
template <>
struct VectorTy<half> {
    using type = vector_f16;
};
template <>
struct VectorTy<bfloat16_t> {
    using type = vector_bf16;
};
template <>
struct VectorTy<int32_t> {
    using type = vector_s32;
};
template <>
struct VectorTy<uint32_t> {
    using type = vector_u32;
};
template <>
struct VectorTy<int16_t> {
    using type = vector_s16;
};
template <>
struct VectorTy<uint16_t> {
    using type = vector_u16;
};
// 8-bit registers. Needed by the strided block load/store stubs: fp8 has no
// scalar semantics on AIV, so those move their payload as int8_t.
template <>
struct VectorTy<int8_t> {
    using type = vector_s8;
};
template <>
struct VectorTy<uint8_t> {
    using type = vector_u8;
};

template <typename T>
using VectorReg = typename VectorTy<T>::type;

#define convertAVEPregToVecBool(p) (*reinterpret_cast<vector_bool*>(&(p)))
