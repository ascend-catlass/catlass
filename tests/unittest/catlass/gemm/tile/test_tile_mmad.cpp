/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"
#include "stub/ascendc_logger.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

#include "common/helper.hpp"
#include "common/shape.hpp"

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// 定义TileMmadTest测试类，继承自AscendCTest测试框架基类
template <class ArchTag>
class TypedTileMmadTest : public TileMmadTest, public testing::WithParamInterface<TestCubeMatrixShapeWithUnitflag> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        TileMmadTest::_setShape(GetParam().m, GetParam().n, GetParam().k);
    }

    void setUnitFlag()
    {
        unitFlag = GetParam().unitFlag;
        initC = GetParam().initC;
    }

    const bool isSmallCube()
    {
        return (_m / C0_NUM_PER_FRACTAL) *
            (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;
    }

    template <class ElementMmad, bool withBias = false>
    void BaseCheck(AscendCCallLog const &logTileMmad)
    {
        ASSERT_EQ(logTileMmad.name, "Mmad");
        if constexpr (withBias) {
            ASSERT_EQ(logTileMmad.args.size(), 5); // C, A, B, Bias, MmadParams
        } else {
            ASSERT_EQ(logTileMmad.args.size(), 4); // C, A, B, MmadParams
        }

        const std::type_index& T0 = logTileMmad.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(ElementMmad));
    }

    // ===================== Non-TLA TileMmad callers =====================

    /// Base version, without bias, direct call to TileMmad
    template <class AType, class BType, class ElementMmad = typename AType::Element>
    auto MakeCall(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor)
    {
        using ElementA = typename AType::Element;
        using ElementB = typename BType::Element;
        static_assert(std::is_same_v<ElementA, ElementB> && std::is_same_v<ElementA, ElementMmad>,
            "ElementA and ElementB should be the same");

        TileMmad<ArchTag, AType, BType, void> tileMmad;

        setShape();
        setUnitFlag();
        tileMmad(l0CTensor, l0ATensor, l0BTensor, _m, _n, _k, initC, unitFlag);

        return AscendCCallLogger::Instance().GetLogs();
    }

    /// With bias passed to TileMmad
    template <class AType, class BType, class BiasType, class ElementMmad = typename AType::Element>
    auto MakeCall(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor,
        const AscendC::LocalTensor<BiasType>& l0BiasTensor)
    {
        using ElementA = typename AType::Element;
        using ElementB = typename BType::Element;
        static_assert(std::is_same_v<ElementA, ElementB> && std::is_same_v<ElementA, ElementMmad>,
            "ElementA and ElementB should be the same");

        TileMmad<ArchTag, AType, BType, BiasType> tileMmad;

        setShape();
        setUnitFlag();
        tileMmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, _m, _n, _k, initC, unitFlag);

        return AscendCCallLogger::Instance().GetLogs();
    }

    // ===================== TLA TileMmadTla callers =====================
    // L0 tla tensors are built with the L0C nested layout (only data()/shape()/
    // originShape() are inspected by the stub), so MakeLayoutL0C suffices for C/A/B.
    using CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;

    template <class ElementMmad>
    auto MakeTensorC(const AscendC::LocalTensor<ElementMmad>& t, uint32_t m, uint32_t n)
    {
        return tla::Tensor<AscendC::LocalTensor<ElementMmad>, decltype(tla::MakeLayoutL0C(m, n)),
            CoordZero, AscendC::TPosition::CO1>(t, tla::MakeLayoutL0C(m, n));
    }
    template <class ElementMmad>
    auto MakeTensorA(const AscendC::LocalTensor<ElementMmad>& t, uint32_t m, uint32_t k)
    {
        return tla::Tensor<AscendC::LocalTensor<ElementMmad>, decltype(tla::MakeLayoutL0C(m, k)),
            CoordZero, AscendC::TPosition::A2>(t, tla::MakeLayoutL0C(m, k));
    }
    template <class ElementMmad>
    auto MakeTensorB(const AscendC::LocalTensor<ElementMmad>& t, uint32_t k, uint32_t n)
    {
        return tla::Tensor<AscendC::LocalTensor<ElementMmad>, decltype(tla::MakeLayoutL0C(k, n)),
            CoordZero, AscendC::TPosition::B2>(t, tla::MakeLayoutL0C(k, n));
    }

    /// TLA without bias, explicit dims
    template <class ElementMmad>
    auto MakeCallTla(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor)
    {
        setShape();
        setUnitFlag();
        TileMmadTla<ArchTag, ElementMmad, layout::zN> tileMmadTla;
        tileMmadTla(MakeTensorC(l0CTensor, _m, _n), MakeTensorA(l0ATensor, _m, _k),
            MakeTensorB(l0BTensor, _k, _n), _m, _n, _k, initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

    /// TLA with bias
    template <class ElementMmad>
    auto MakeCallTlaBias(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor,
        const AscendC::LocalTensor<ElementMmad>& l0BiasTensor)
    {
        setShape();
        setUnitFlag();
        TileMmadTla<ArchTag, ElementMmad, layout::zN> tileMmadTla;
        tileMmadTla(MakeTensorC(l0CTensor, _m, _n), MakeTensorA(l0ATensor, _m, _k),
            MakeTensorB(l0BTensor, _k, _n), MakeTensorC(l0BiasTensor, _m, _n), _m, _n, _k, initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

    /// TLA auto dims from originShape, shape auto derived from tensor in
    template <class ElementMmad>
    auto MakeCallTlaAuto(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor)
    {
        setShape();
        setUnitFlag();
        TileMmadTla<ArchTag, ElementMmad, layout::zN> tileMmadTla;
        tileMmadTla(MakeTensorC(l0CTensor, _m, _n), MakeTensorA(l0ATensor, _m, _k),
            MakeTensorB(l0BTensor, _k, _n), initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

    // TLA batched
    template <class ElementMmad>
    auto MakeCallTlaBatch(
        const AscendC::LocalTensor<ElementMmad>& l0CTensor,
        const AscendC::LocalTensor<ElementMmad>& l0ATensor,
        const AscendC::LocalTensor<ElementMmad>& l0BTensor,
        uint32_t l0Batch)
    {
        setShape();
        setUnitFlag();
        TileMmadTla<ArchTag, ElementMmad, layout::zN> tileMmadTla;
        tileMmadTla(MakeTensorC(l0CTensor, _m, _n), MakeTensorA(l0ATensor, _m, _k),
            MakeTensorB(l0BTensor, _k, _n), _m, _n, _k, l0Batch);
        return AscendCCallLogger::Instance().GetLogs();
    }

protected:
    static const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    bool initC = false;
    uint8_t unitFlag = 0b00;
};

// ============================================================================
// Common test bodies shared by AtlasA2 / Ascend950 suites.
// ============================================================================
#define DEFINE_TILE_MMAD_TESTS(SuiteName)                                                                              \
                                                                                                                       \
    /* 基础Mmad功能测试（无偏置, non-TLA） */                                                                          \
    /* Data-path: A·B → C (Mmad) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: basic non-TLA Mmad, explicit m/n/k (optional PIPE_M barrier) */ \
    TEST_P(SuiteName, MmadTestBasic)                                                                                       \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;                                \
        using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;                                \
                                                                                                                       \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
                                                                                                                       \
        auto logs = MakeCall<AType, BType>(l0CTensor, l0ATensor, l0BTensor);                                          \
        const bool hasPipeM = (_m / C0_NUM_PER_FRACTAL) * (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;       \
                                                                                                                       \
        ASSERT_EQ(logs.size(), hasPipeM ? 2 : 1);                                                                     \
        AscendCCallLog logMmad = logs[0];                                                                             \
        BaseCheck<ElementMmad>(logMmad);                                                                              \
        ASSERT_EQ(logMmad.GetArgsAt(0).RawValue(), &l0CTensor);                                                       \
        ASSERT_EQ(logMmad.GetArgsAt(1).RawValue(), &l0ATensor);                                                       \
        ASSERT_EQ(logMmad.GetArgsAt(2).RawValue(), &l0BTensor);                                                       \
                                                                                                                       \
        const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();                       \
        ASSERT_EQ(mmadArg->m, _m);                                                                                    \
        ASSERT_EQ(mmadArg->n, _n);                                                                                    \
        ASSERT_EQ(mmadArg->k, _k);                                                                                    \
        ASSERT_EQ(mmadArg->cmatrixInitVal, initC);                                                                    \
        ASSERT_EQ(mmadArg->unitFlag, unitFlag);                                                                       \
                                                                                                                       \
        if (hasPipeM) {                                                                                               \
            AscendCCallLog logPipeBarrier = logs[1];                                                                  \
            ASSERT_EQ(logPipeBarrier.name, "PipeBarrier");                                                            \
            ASSERT_EQ(*logPipeBarrier.GetArgsTAt(0).Value<pipe_t>(), pipe_t::PIPE_M);                                 \
        }                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* 带偏置的Mmad功能测试（non-TLA） */                                                                              \
    /* Data-path: A·B + Bias → C (Mmad) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: non-TLA Mmad with bias */ \
    /* (cmatrixInitVal forced false -- initial c-matrix comes from bias table) */ \
    TEST_P(SuiteName, MmadTestWithBias)                                                                                   \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;                                \
        using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;                                \
        using BiasType = ElementMmad;                                                                                  \
                                                                                                                       \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BiasTensor;                                                               \
                                                                                                                       \
        auto logs = MakeCall<AType, BType, BiasType, ElementMmad>(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor);     \
        const bool hasPipeM = (_m / C0_NUM_PER_FRACTAL) * (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;       \
                                                                                                                       \
        ASSERT_EQ(logs.size(), hasPipeM ? 2 : 1);                                                                     \
        AscendCCallLog logMmad = logs[0];                                                                             \
        BaseCheck<ElementMmad, true>(logMmad);                                                                        \
        ASSERT_EQ(logMmad.GetArgsAt(0).RawValue(), &l0CTensor);                                                       \
        ASSERT_EQ(logMmad.GetArgsAt(1).RawValue(), &l0ATensor);                                                       \
        ASSERT_EQ(logMmad.GetArgsAt(2).RawValue(), &l0BTensor);                                                       \
        ASSERT_EQ(logMmad.GetArgsAt(3).RawValue(), &l0BiasTensor);                                                    \
                                                                                                                       \
        const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(4).Value<AscendC::MmadParams>();                       \
        ASSERT_EQ(mmadArg->m, _m);                                                                                    \
        ASSERT_EQ(mmadArg->n, _n);                                                                                    \
        ASSERT_EQ(mmadArg->k, _k);                                                                                    \
        ASSERT_EQ(mmadArg->cmatrixInitVal, false);                                                                    \
        ASSERT_EQ(mmadArg->unitFlag, unitFlag);                                                                       \
                                                                                                                       \
        if (hasPipeM) {                                                                                               \
            AscendCCallLog logPipeBarrier = logs[1];                                                                  \
            ASSERT_EQ(logPipeBarrier.name, "PipeBarrier");                                                            \
            ASSERT_EQ(*logPipeBarrier.GetArgsTAt(0).Value<pipe_t>(), pipe_t::PIPE_M);                                 \
        }                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* TileMmadTla 无偏置, 显式 m/n/k (overload 1) */                                                                  \
    /* Data-path: A·B → C (Mmad) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: TLA Mmad with explicit m/n/k (overload 1) */ \
    TEST_P(SuiteName, MmadTestTlaBasic)                                                                                   \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
                                                                                                                       \
        auto logs = MakeCallTla<ElementMmad>(l0CTensor, l0ATensor, l0BTensor);                                        \
        const bool hasPipeM = (_m / C0_NUM_PER_FRACTAL) * (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;       \
                                                                                                                       \
        ASSERT_EQ(logs.size(), hasPipeM ? 2 : 1);                                                                     \
        AscendCCallLog logMmad = logs[0];                                                                             \
        BaseCheck<ElementMmad>(logMmad);                                                                              \
        ASSERT_EQ(logMmad.GetArgsAt(0).GetInstAddr(), 0);                                                             \
        ASSERT_EQ(logMmad.GetArgsAt(1).GetInstAddr(), 0);                                                             \
        ASSERT_EQ(logMmad.GetArgsAt(2).GetInstAddr(), 0);                                                             \
                                                                                                                       \
        const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();                       \
        ASSERT_EQ(mmadArg->m, _m);                                                                                    \
        ASSERT_EQ(mmadArg->n, _n);                                                                                    \
        ASSERT_EQ(mmadArg->k, _k);                                                                                    \
        ASSERT_EQ(mmadArg->cmatrixInitVal, initC);                                                                    \
        ASSERT_EQ(mmadArg->unitFlag, unitFlag);                                                                       \
                                                                                                                       \
        if (hasPipeM) {                                                                                               \
            AscendCCallLog logPipeBarrier = logs[1];                                                                  \
            ASSERT_EQ(logPipeBarrier.name, "PipeBarrier");                                                            \
            ASSERT_EQ(*logPipeBarrier.GetArgsTAt(0).Value<pipe_t>(), pipe_t::PIPE_M);                                 \
        }                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* TileMmadTla 带偏置 (overload 2) */                                                                              \
    /* Data-path: A·B + Bias → C (Mmad) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: TLA Mmad with bias (overload 2, cmatrixInitVal false) */ \
    TEST_P(SuiteName, MmadTestTlaWithBias)                                                                                \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BiasTensor;                                                               \
                                                                                                                       \
        auto logs = MakeCallTlaBias<ElementMmad>(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor);                      \
        const bool hasPipeM = (_m / C0_NUM_PER_FRACTAL) * (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;       \
                                                                                                                       \
        ASSERT_EQ(logs.size(), hasPipeM ? 2 : 1);                                                                     \
        AscendCCallLog logMmad = logs[0];                                                                             \
        BaseCheck<ElementMmad, true>(logMmad);                                                                        \
                                                                                                                       \
        const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(4).Value<AscendC::MmadParams>();                       \
        ASSERT_EQ(mmadArg->m, _m);                                                                                    \
        ASSERT_EQ(mmadArg->n, _n);                                                                                    \
        ASSERT_EQ(mmadArg->k, _k);                                                                                    \
        ASSERT_EQ(mmadArg->cmatrixInitVal, false);                                                                    \
        ASSERT_EQ(mmadArg->unitFlag, unitFlag);                                                                       \
                                                                                                                       \
        if (hasPipeM) {                                                                                               \
            AscendCCallLog logPipeBarrier = logs[1];                                                                  \
            ASSERT_EQ(logPipeBarrier.name, "PipeBarrier");                                                            \
            ASSERT_EQ(*logPipeBarrier.GetArgsTAt(0).Value<pipe_t>(), pipe_t::PIPE_M);                                 \
        }                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* TileMmadTla 自动推导 m/n/k (overload 4) */                                                                      \
    /* Data-path: A·B → C (Mmad) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: TLA Mmad, m/n/k auto-deduced from originShape (overload 4) */ \
    TEST_P(SuiteName, MmadTestTlaAutoShape)                                                                               \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
                                                                                                                       \
        auto logs = MakeCallTlaAuto<ElementMmad>(l0CTensor, l0ATensor, l0BTensor);                                    \
        ASSERT_GE(logs.size(), 1U);                                                                                   \
        AscendCCallLog logMmad = logs[0];                                                                             \
        BaseCheck<ElementMmad>(logMmad);                                                                              \
                                                                                                                       \
        const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();                       \
        ASSERT_EQ(mmadArg->m, _m);                                                                                    \
        ASSERT_EQ(mmadArg->n, _n);                                                                                    \
        ASSERT_EQ(mmadArg->k, _k);                                                                                    \
        ASSERT_EQ(mmadArg->cmatrixInitVal, initC);                                                                    \
        ASSERT_EQ(mmadArg->unitFlag, unitFlag);                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    /* TileMmadTla 批量 (overload 3) */                                                                               \
    /* Data-path: A·B → C (Mmad, batched) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: TLA batched Mmad (overload 3, l0Batch repeats, initC=true) */ \
    TEST_P(SuiteName, MmadTestTlaBatch)                                                                                   \
    {                                                                                                                  \
        using ElementMmad = float;                                                                                    \
        AscendC::LocalTensor<ElementMmad> l0CTensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0ATensor;                                                                  \
        AscendC::LocalTensor<ElementMmad> l0BTensor;                                                                  \
                                                                                                                       \
        const uint32_t l0Batch = 3;                                                                                   \
        auto logs = MakeCallTlaBatch<ElementMmad>(l0CTensor, l0ATensor, l0BTensor, l0Batch);                          \
        ASSERT_EQ(logs.size(), l0Batch);                                                                              \
        for (uint32_t i = 0; i < l0Batch; ++i) {                                                                      \
            BaseCheck<ElementMmad>(logs[i]);                                                                          \
            const AscendC::MmadParams* mmadArg = logs[i].GetArgsAt(3).Value<AscendC::MmadParams>();                   \
            ASSERT_EQ(mmadArg->m, _m);                                                                                \
            ASSERT_EQ(mmadArg->n, _n);                                                                                \
            ASSERT_EQ(mmadArg->k, _k);                                                                                \
            ASSERT_EQ(mmadArg->cmatrixInitVal, true);                                                                 \
            ASSERT_EQ(mmadArg->unitFlag, _0);                                                                         \
        }                                                                                                            \
    }

#define INSTANTIATE_TILE_MMAD_SUITE(Prefix, SuiteName)                                                                \
    INSTANTIATE_TEST_SUITE_P(Prefix, SuiteName,                                                                       \
        ::testing::Values(                                                                                            \
            TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, true, 0b00},                                           \
            TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, false, 0b00},                                          \
            TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, true, 0b10},                                           \
            TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, false, 0b10},                                          \
            TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, false, 0b11},                                          \
            TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, true, 0b00},                                              \
            TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, false, 0b00},                                             \
            TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, true, 0b10},                                              \
            TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, false, 0b10},                                             \
            TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, false, 0b11},                                             \
            TestCubeMatrixShapeWithUnitflag{33U, 18U, 256U, true, 0b00},                                             \
            TestCubeMatrixShapeWithUnitflag{33U, 18U, 256U, false, 0b00},                                            \
            TestCubeMatrixShapeWithUnitflag{33U, 18U, 256U, true, 0b10},                                             \
            TestCubeMatrixShapeWithUnitflag{33U, 18U, 256U, false, 0b10},                                            \
            TestCubeMatrixShapeWithUnitflag{33U, 18U, 256U, false, 0b11}))

// ============================================================================
// Instantiate for Catlass::Arch::AtlasA2
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
using TypedTileMmadA2Suite = TypedTileMmadTest<Catlass::Arch::AtlasA2>;

DEFINE_TILE_MMAD_TESTS(TypedTileMmadA2Suite)

INSTANTIATE_TILE_MMAD_SUITE(AtlasA2, TypedTileMmadA2Suite);
#endif // CATLASS_ARCH == 2201

// ============================================================================
// Instantiate for Catlass::Arch::Ascend950
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using TypedTileMmadA950Suite = TypedTileMmadTest<Catlass::Arch::Ascend950>;

DEFINE_TILE_MMAD_TESTS(TypedTileMmadA950Suite)

// Ascend950-only: VectorLayout A enables gemv mode (disableGemv == false)
// Data-path: A(Vector)·B → C (Mmad)
// Element-type: no-except (float)
// Speciality: Ascend950 gemv mode (VectorLayout A → disableGemv=false)
TEST_P(TypedTileMmadA950Suite, MmadTestGemv)
{
    using ElementMmad = float;
    using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::VectorLayout>;
    using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;

    AscendC::LocalTensor<ElementMmad> l0CTensor;
    AscendC::LocalTensor<ElementMmad> l0ATensor;
    AscendC::LocalTensor<ElementMmad> l0BTensor;

    auto logs = MakeCall<AType, BType>(l0CTensor, l0ATensor, l0BTensor);
    const bool hasPipeM = (_m / C0_NUM_PER_FRACTAL) * (_n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD;

    ASSERT_EQ(logs.size(), hasPipeM ? 2 : 1);
    AscendCCallLog logMmad = logs[0];
    BaseCheck<ElementMmad>(logMmad);

    const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();
    ASSERT_EQ(mmadArg->m, _m);
    ASSERT_EQ(mmadArg->n, _n);
    ASSERT_EQ(mmadArg->k, _k);
    ASSERT_EQ(mmadArg->disableGemv, false);
    ASSERT_EQ(mmadArg->cmatrixInitVal, initC);
    ASSERT_EQ(mmadArg->unitFlag, unitFlag);

    if (hasPipeM) {
        AscendCCallLog logPipeBarrier = logs[1];
        ASSERT_EQ(logPipeBarrier.name, "PipeBarrier");
        ASSERT_EQ(*logPipeBarrier.GetArgsTAt(0).Value<pipe_t>(), pipe_t::PIPE_M);
    }
}

// Ascend950-only: non-vector layout keeps gemv disabled (disableGemv == true)
// Data-path: A·B → C (Mmad)
// Element-type: no-except (float)
// Speciality: Ascend950 non-vector A keeps gemv disabled (disableGemv=true)
TEST_P(TypedTileMmadA950Suite, MmadTestDisableGemv)
{
    using ElementMmad = float;
    using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;

    AscendC::LocalTensor<ElementMmad> l0CTensor;
    AscendC::LocalTensor<ElementMmad> l0ATensor;
    AscendC::LocalTensor<ElementMmad> l0BTensor;

    auto logs = MakeCall<AType, BType>(l0CTensor, l0ATensor, l0BTensor);
    AscendCCallLog logMmad = logs[0];
    BaseCheck<ElementMmad>(logMmad);

    const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();
    ASSERT_EQ(mmadArg->disableGemv, true);
}


template <class ArchTag>
class Hif8TileMmadTest : public TileMmadTest, public testing::WithParamInterface<TestCubeMatrixShapeWithUnitflag> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        TileMmadTest::_setShape(GetParam().m, GetParam().n, GetParam().k);
    }

    void setUnitFlag()
    {
        unitFlag = GetParam().unitFlag;
        initC = GetParam().initC;
    }

    static int FindLog(const std::vector<AscendCCallLog>& logs, const std::string& name, int from = 0)
    {
        for (int i = from; i < (int)logs.size(); ++i) {
            if (logs[i].name == name) {
                return i;
            }
        }
        return -1;
    }

    static void CheckHif8CtrlSequence(const std::vector<AscendCCallLog>& logs, int mmadIdx)
    {
        ASSERT_GE(mmadIdx, 0);
        int getIdx = FindLog(logs, "GetCtrlSpr");
        ASSERT_GE(getIdx, 0);
        ASSERT_LT(getIdx, mmadIdx);
        ASSERT_EQ(*logs[getIdx].GetArgsTAt(0).template Value<int8_t>(), 45);
        ASSERT_EQ(*logs[getIdx].GetArgsTAt(1).template Value<int8_t>(), 45);
        int setIdx = FindLog(logs, "SetCtrlSpr");
        ASSERT_GE(setIdx, 0);
        ASSERT_LT(setIdx, mmadIdx);
        ASSERT_EQ(*logs[setIdx].GetArgsTAt(0).template Value<int8_t>(), 45);
        ASSERT_EQ(*logs[setIdx].GetArgsTAt(1).template Value<int8_t>(), 45);
        ASSERT_EQ(*logs[setIdx].GetArgsTAt(2).template Value<int64_t>(), 1);
        int rstIdx = FindLog(logs, "SetCtrlSpr", mmadIdx + 1);
        ASSERT_GT(rstIdx, mmadIdx);
        ASSERT_EQ(*logs[rstIdx].GetArgsTAt(0).template Value<int8_t>(), 45);
        ASSERT_EQ(*logs[rstIdx].GetArgsTAt(1).template Value<int8_t>(), 45);
        ASSERT_EQ(*logs[rstIdx].GetArgsTAt(2).template Value<int64_t>(), 0); // stub GetCtrlSpr returns 0
    }

    static void CheckHif8Alias(const std::vector<AscendCCallLog>& logs, uint32_t addrA, uint32_t addrB)
    {
        int first = FindLog(logs, "ReinterpretCast");
        ASSERT_GE(first, 0);
        int second = FindLog(logs, "ReinterpretCast", first + 1);
        ASSERT_GT(second, first);
        ASSERT_EQ(logs[first].GetArgsTAt(0).Type(), typeid(fp8_e4m3fn_t));
        ASSERT_EQ(logs[second].GetArgsTAt(0).Type(), typeid(fp8_e4m3fn_t));
        ASSERT_EQ(*logs[first].GetArgsAt(0).template Value<uint32_t>(), addrA);
        ASSERT_EQ(*logs[second].GetArgsAt(0).template Value<uint32_t>(), addrB);
    }

    template <class AType, class BType>
    auto MakeCall(
        const AscendC::LocalTensor<float>& l0CTensor, const AscendC::LocalTensor<hifloat8_t>& l0ATensor,
        const AscendC::LocalTensor<hifloat8_t>& l0BTensor)
    {
        TileMmad<ArchTag, AType, BType, void> tileMmad;
        setShape();
        setUnitFlag();
        tileMmad(l0CTensor, l0ATensor, l0BTensor, _m, _n, _k, initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

    template <class AType, class BType>
    auto MakeCallBias(
        const AscendC::LocalTensor<float>& l0CTensor, const AscendC::LocalTensor<hifloat8_t>& l0ATensor,
        const AscendC::LocalTensor<hifloat8_t>& l0BTensor, const AscendC::LocalTensor<float>& l0BiasTensor)
    {
        TileMmad<ArchTag, AType, BType, float> tileMmad;
        setShape();
        setUnitFlag();
        tileMmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, _m, _n, _k, initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

    using CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;

    auto MakeTensorC(const AscendC::LocalTensor<float>& t, uint32_t m, uint32_t n)
    {
        return tla::Tensor<AscendC::LocalTensor<float>, decltype(tla::MakeLayoutL0C(m, n)), CoordZero,
            AscendC::TPosition::CO1>(t, tla::MakeLayoutL0C(m, n));
    }
    auto MakeTensorAB(const AscendC::LocalTensor<hifloat8_t>& t, uint32_t m, uint32_t n)
    {
        return tla::Tensor<AscendC::LocalTensor<hifloat8_t>, decltype(tla::MakeLayoutL0C(m, n)), CoordZero,
            AscendC::TPosition::A2>(t, tla::MakeLayoutL0C(m, n));
    }

    auto MakeCallTla(
        const AscendC::LocalTensor<float>& l0CTensor, const AscendC::LocalTensor<hifloat8_t>& l0ATensor,
        const AscendC::LocalTensor<hifloat8_t>& l0BTensor)
    {
        setShape();
        setUnitFlag();
        TileMmadTla<ArchTag, hifloat8_t, layout::zN> tileMmadTla;
        tileMmadTla(
            MakeTensorC(l0CTensor, _m, _n), MakeTensorAB(l0ATensor, _m, _k), MakeTensorAB(l0BTensor, _k, _n), _m,
            _n, _k, initC, unitFlag);
        return AscendCCallLogger::Instance().GetLogs();
    }

protected:
    bool initC = false;
    uint8_t unitFlag = 0b00;
};

using Hif8TileMmadA950Suite = Hif8TileMmadTest<Catlass::Arch::Ascend950>;

TEST_P(Hif8TileMmadA950Suite, MmadTestHif8Basic)
{
    using AType = Catlass::Gemm::GemmType<hifloat8_t, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<hifloat8_t, Catlass::layout::RowMajor>;

    AscendC::LocalTensor<float> l0CTensor;
    AscendC::LocalTensor<hifloat8_t> l0ATensor(0x100, 64);
    AscendC::LocalTensor<hifloat8_t> l0BTensor(0x200, 64);

    auto logs = MakeCall<AType, BType>(l0CTensor, l0ATensor, l0BTensor);
    int mmadIdx = FindLog(logs, "Mmad");
    ASSERT_GE(mmadIdx, 0);
    CheckHif8Alias(logs, 0x100, 0x200);
    CheckHif8CtrlSequence(logs, mmadIdx);

    AscendCCallLog logMmad = logs[mmadIdx];
    ASSERT_EQ(logMmad.args.size(), 4); // C, A, B, MmadParams
    ASSERT_EQ(logMmad.GetArgsTAt(0).Type(), typeid(float));
    ASSERT_EQ(logMmad.GetArgsTAt(1).Type(), typeid(fp8_e4m3fn_t));
    ASSERT_EQ(logMmad.GetArgsTAt(2).Type(), typeid(fp8_e4m3fn_t));

    const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();
    ASSERT_EQ(mmadArg->m, _m);
    ASSERT_EQ(mmadArg->n, _n);
    ASSERT_EQ(mmadArg->k, _k);
    ASSERT_EQ(mmadArg->cmatrixInitVal, initC);
    ASSERT_EQ(mmadArg->unitFlag, unitFlag);
}

TEST_P(Hif8TileMmadA950Suite, MmadTestHif8Bias)
{
    using AType = Catlass::Gemm::GemmType<hifloat8_t, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<hifloat8_t, Catlass::layout::RowMajor>;

    AscendC::LocalTensor<float> l0CTensor;
    AscendC::LocalTensor<hifloat8_t> l0ATensor(0x100, 64);
    AscendC::LocalTensor<hifloat8_t> l0BTensor(0x200, 64);
    AscendC::LocalTensor<float> l0BiasTensor;

    auto logs = MakeCallBias<AType, BType>(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor);
    int mmadIdx = FindLog(logs, "Mmad");
    ASSERT_GE(mmadIdx, 0);
    CheckHif8Alias(logs, 0x100, 0x200);
    CheckHif8CtrlSequence(logs, mmadIdx);

    AscendCCallLog logMmad = logs[mmadIdx];
    ASSERT_EQ(logMmad.args.size(), 5); // C, A, B, Bias, MmadParams
    ASSERT_EQ(logMmad.GetArgsTAt(3).Type(), typeid(float));

    const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(4).Value<AscendC::MmadParams>();
    ASSERT_EQ(mmadArg->cmatrixInitVal, false);
    ASSERT_EQ(mmadArg->unitFlag, unitFlag);
}

TEST_P(Hif8TileMmadA950Suite, MmadTestHif8TlaBasic)
{
    AscendC::LocalTensor<float> l0CTensor;
    AscendC::LocalTensor<hifloat8_t> l0ATensor(0x100, 64);
    AscendC::LocalTensor<hifloat8_t> l0BTensor(0x200, 64);

    auto logs = MakeCallTla(l0CTensor, l0ATensor, l0BTensor);
    int mmadIdx = FindLog(logs, "Mmad");
    ASSERT_GE(mmadIdx, 0);
    CheckHif8CtrlSequence(logs, mmadIdx);

    AscendCCallLog logMmad = logs[mmadIdx];
    ASSERT_EQ(logMmad.GetArgsTAt(1).Type(), typeid(fp8_e4m3fn_t));
    ASSERT_EQ(logMmad.GetArgsTAt(2).Type(), typeid(fp8_e4m3fn_t));

    const AscendC::MmadParams* mmadArg = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();
    ASSERT_EQ(mmadArg->m, _m);
    ASSERT_EQ(mmadArg->n, _n);
    ASSERT_EQ(mmadArg->k, _k);
    ASSERT_EQ(mmadArg->cmatrixInitVal, initC);
}

INSTANTIATE_TEST_SUITE_P(Hif8Ascend950, Hif8TileMmadA950Suite,
    ::testing::Values(
        TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, true, 0b00},
        TestCubeMatrixShapeWithUnitflag{128U, 128U, 128U, false, 0b10},
        TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, true, 0b00},
        TestCubeMatrixShapeWithUnitflag{32U, 32U, 32U, false, 0b00}));

INSTANTIATE_TILE_MMAD_SUITE(Ascend950, TypedTileMmadA950Suite);
#endif // CATLASS_ARCH == 3510
