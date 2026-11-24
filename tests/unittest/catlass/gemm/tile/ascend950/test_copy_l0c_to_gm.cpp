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

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;
using namespace AscendC;

using ArchTag = Arch::Ascend950;

// Testcase for L0C->GM TileCopy Utilities
class TileCopyL0CToGmTestAscend950 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShapeWithUnitflag> {
protected:
    // 重写SetUp方法，在每个测试用例执行前进行初始化操作
    void SetUp() override
    {
        // 调用父类的SetUp方法完成基础初始化
        AscendCTest::SetUp();
    }

    template <class Element>
    void setShape() {
        _setShape<Element, false>(GetParam().row, GetParam().col);
        _m = GetParam().row;
        _n = GetParam().col;
        _unitFlag = GetParam().unitFlag;
        _channelSplit = GetParam().channelSplit;
        _m_round = _row_round;
        _n_round = _col_round;
    }

    template <class ElementDst, class ElementAccu, bool isFixpipeAPI = true>
    void BaseCheck(AscendCCallLog const &logTileCopy, AscendC::GlobalTensor<ElementDst>& gmTensor, AscendC::LocalTensor<ElementAccu>& l0cTensor){
        ASSERT_EQ(logTileCopy.name, isFixpipeAPI ? "Fixpipe" : "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        auto logTileCopyGmTensor = logTileCopy.GetArgsAt(0).RawValue();
        auto logTileCopyL0cTensor = logTileCopy.GetArgsAt(1).RawValue();
        ASSERT_EQ(logTileCopyGmTensor, &gmTensor);  // 验证GM张量地址一致
        ASSERT_EQ(logTileCopyL0cTensor, &l0cTensor);  // 验证L0C张量地址一致
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(ElementDst));

        const std::type_index& T1 = logTileCopy.GetArgsTAt(1).Type();
        ASSERT_EQ(T1, typeid(ElementAccu));
    }

protected:
    uint16_t _m = 32;
    uint16_t _n = 32;
    uint16_t _m_round = 0;
    uint16_t _n_round = 0;
    uint8_t _unitFlag = 0;
    bool _channelSplit = false;
};

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN (L0C) → RowMajor (GM)
// Element-type: no-except (float → float)
// Speciality: NoQuant (no-quant DataCopy path, nz2nd enabled)
TEST_P(TileCopyL0CToGmTestAscend950, zNToRowMajorTestNoQuant)
{
    using ElementAccumulator = float;
    using ElementDst = float;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    constexpr auto quantPre = QuantMode_t::NoQuant;

    CopyL0CToGm<ArchTag, ElementAccumulator,Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;
    
    // prepare tensor
    AscendC::GlobalTensor<ElementAccumulator> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementAccumulator>();
    setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);
    ASSERT_TRUE(typeid(ElementDst) == typeid(ElementAccumulator));
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    
    // call copyL0CToGm
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, _unitFlag);

    // Get logs
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 2); 
    // 1. setFixpipe config; 2. execute DataCopy 

    AscendCCallLog logfixpipeCfg = logs[0];
    ASSERT_EQ(logfixpipeCfg.name, "SetFixpipeNz2ndFlag");
    ASSERT_EQ(logfixpipeCfg.args.size(), 3);
    ASSERT_EQ(*logfixpipeCfg.GetArgsAt(0).Value<uint16_t>(), _1); // ndNum = 1

    AscendCCallLog logTileCopy = logs[1];
    BaseCheck<ElementDst, ElementAccumulator, false/*no quant mode -- DataCopy*/>(logTileCopy, gmTensor, l0cTensor);

    const AscendC::DataCopyCO12DstParams* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
    ASSERT_EQ(dataCopyArg->mSize, _m);
    ASSERT_EQ(dataCopyArg->nSize, _n);
    ASSERT_EQ(dataCopyArg->srcStride, _m_round);   // 源相邻Z排布的偏移，单位:C0_SIZE(16B)
    ASSERT_EQ(dataCopyArg->dstStride, _n);         // 目标相邻N排布的偏移，单位:Element
    ASSERT_EQ(dataCopyArg->unitFlag, _unitFlag);
    ASSERT_EQ(dataCopyArg->nz2ndEn, true);
    ASSERT_EQ(dataCopyArg->quantPre, quantPre);
}

// Data-path: zN (L0C) → zN (GM)
// Element-type: no-except (float → float)
// Speciality: NoQuant (no-quant DataCopy, float→float enables channelSplit)
TEST_P(TileCopyL0CToGmTestAscend950, zNTozNTestNoQuant)
{
    using ElementAccumulator = float;
    using ElementDst = float;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;
    constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementAccumulator>();
    layoutSrc = LayoutSrc::template MakeLayout<ElementAccumulator>(_m, _n);
    layoutDst = LayoutDst::template MakeLayout<ElementDst>(_m, _n);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, _unitFlag);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator, false>(logTileCopy, gmTensor, l0cTensor);

    uint32_t expectedMSize = layoutDst.shape(0) * layoutDst.shape(1);
    uint32_t expectedNSize = layoutDst.shape(2) * layoutDst.shape(3);

    const AscendC::DataCopyCO12DstParams* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
    ASSERT_EQ(dataCopyArg->mSize, expectedMSize);
    ASSERT_EQ(dataCopyArg->nSize, expectedNSize);
    ASSERT_EQ(dataCopyArg->srcStride, _m_round);
    ASSERT_EQ(dataCopyArg->dstStride, _m_round);       // zN dstStride = stride(3)/(32/sizeof)=m_round
    ASSERT_EQ(dataCopyArg->unitFlag, _unitFlag);
    ASSERT_EQ(dataCopyArg->nz2ndEn, false);
    ASSERT_EQ(dataCopyArg->channelSplit, true);         // float→float channelSplit=true
    ASSERT_EQ(dataCopyArg->quantPre, quantPre);
}

// Data-path: zN (L0C) → zN (GM)
// Element-type: float → half
// Speciality: NoQuantFloatToHalf (no-quant DataCopy, float→half disables channelSplit)
TEST_P(TileCopyL0CToGmTestAscend950, zNTozNTestNoQuantFloatToHalf)
{
    using ElementAccumulator = float;
    using ElementDst = half;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;
    constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementAccumulator>();
    layoutSrc = LayoutSrc::template MakeLayout<ElementAccumulator>(_m, _n);
    layoutDst = LayoutDst::template MakeLayout<ElementDst>(_m, _n);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, _unitFlag);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator, false>(logTileCopy, gmTensor, l0cTensor);

    uint32_t expectedMSize = layoutDst.shape(0) * layoutDst.shape(1);
    uint32_t expectedNSize = layoutDst.shape(2) * layoutDst.shape(3);

    const AscendC::DataCopyCO12DstParams* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
    ASSERT_EQ(dataCopyArg->mSize, expectedMSize);
    ASSERT_EQ(dataCopyArg->nSize, expectedNSize);
    ASSERT_EQ(dataCopyArg->srcStride, _m_round);
    ASSERT_EQ(dataCopyArg->unitFlag, _unitFlag);
    ASSERT_EQ(dataCopyArg->nz2ndEn, false);
    ASSERT_EQ(dataCopyArg->channelSplit, false);        // float→half no channelSplit
    ASSERT_EQ(dataCopyArg->quantPre, quantPre);
}

// Data-path: zN (L0C) → RowMajor (GM)
// Element-type: float → half
// Speciality: PerTensor (per-tensor quant via Fixpipe, deqScalar applied)
TEST_P(TileCopyL0CToGmTestAscend950, zNToRowMajorTestPerTensor) 
{
    using ElementAccumulator = float;
    using ElementDst = half;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    constexpr CO2Layout fixpipeFmt = CO2Layout::ROW_MAJOR;
    constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::PER_TENSOR>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>, 
        ScaleGranularity::PER_TENSOR> copyL0CToGm;
    
    // prepare tensor
    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementAccumulator>();
    setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);

    // execute
    float _scale = 2.0;
    uint64_t _scale_uint64 = static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(&_scale));
    copyL0CToGm.params.scale = _scale;
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, _unitFlag);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logfixpipe = logs[0];
    BaseCheck<ElementDst, ElementAccumulator, true>(logfixpipe, gmTensor, l0cTensor);
    
    const AscendC::FixpipeParamsC310<fixpipeFmt>* fixpipeArg = logfixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<fixpipeFmt>>();
    ASSERT_EQ(fixpipeArg->nSize, _n);
    ASSERT_EQ(fixpipeArg->mSize, _m);
    ASSERT_EQ(fixpipeArg->srcStride, _m_round);
    ASSERT_EQ(fixpipeArg->dstStride, _n);
    ASSERT_EQ(fixpipeArg->quantPre, quantPre);
    ASSERT_EQ(fixpipeArg->deqScalar, _scale_uint64);
    ASSERT_EQ(fixpipeArg->unitFlag, _unitFlag);

    auto *fixpipeCfg = logfixpipe.GetArgsTAt(2).Value<AscendC::FixpipeConfig>();
    ASSERT_EQ(fixpipeCfg->format, fixpipeFmt);
}

// Data-path: zN (L0C) → RowMajor (GM)
// Element-type: float → half
// Speciality: PerChannel (per-channel quant via Fixpipe with scale tensor arg)
TEST_P(TileCopyL0CToGmTestAscend950, zNToRowMajorTestPerChannel)
{
    using ElementAccumulator = float;
    using ElementDst = half;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    constexpr CO2Layout fixpipeFmt = CO2Layout::ROW_MAJOR;
    constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::PER_CHANNEL>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>,
        ScaleGranularity::PER_CHANNEL> copyL0CToGm;

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;
    AscendC::LocalTensor<uint64_t> scaleTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementAccumulator>();
    setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL0CToGm(gmTensor, l0cTensor, scaleTensor, layoutDst, layoutSrc, _unitFlag);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logfixpipe = logs[0];
    ASSERT_EQ(logfixpipe.name, "Fixpipe");
    ASSERT_EQ(logfixpipe.args.size(), 4);  // dst, src, scale, params

    auto logGmTensor = logfixpipe.GetArgsAt(0).RawValue();
    auto logL0cTensor = logfixpipe.GetArgsAt(1).RawValue();
    auto logScaleTensor = logfixpipe.GetArgsAt(2).RawValue();
    ASSERT_EQ(logGmTensor, &gmTensor);
    ASSERT_EQ(logL0cTensor, &l0cTensor);
    ASSERT_EQ(logScaleTensor, &scaleTensor);
    ASSERT_EQ(logfixpipe.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logfixpipe.GetArgsAt(1).GetInstAddr(), 0);
    ASSERT_EQ(logfixpipe.GetArgsAt(2).GetInstAddr(), 0);

    const std::type_index& T0 = logfixpipe.GetArgsTAt(0).Type();
    ASSERT_EQ(T0, typeid(ElementDst));

    const AscendC::FixpipeParamsC310<fixpipeFmt>* fixpipeArg = logfixpipe.GetArgsAt(3).Value<AscendC::FixpipeParamsC310<fixpipeFmt>>();
    ASSERT_EQ(fixpipeArg->nSize, _n);
    ASSERT_EQ(fixpipeArg->mSize, _m);
    ASSERT_EQ(fixpipeArg->srcStride, _m_round);
    ASSERT_EQ(fixpipeArg->dstStride, _n);
    ASSERT_EQ(fixpipeArg->quantPre, quantPre);
    ASSERT_EQ(fixpipeArg->unitFlag, _unitFlag);

    auto *fixpipeCfg = logfixpipe.GetArgsTAt(2).Value<AscendC::FixpipeConfig>();
    ASSERT_EQ(fixpipeCfg->format, fixpipeFmt);
}

// Data-path: zN (L0C) → RowMajor (GM)
// Element-type: float → fp8_e4m3fn_t / hifloat8_t
// Speciality: PerTensor QuantMode mapping (QF322FP8_PRE / QF322HIF8_PRE)
TEST_P(TileCopyL0CToGmTestAscend950, zNToRowMajorTestPerTensorFp8Hif8)
{
    using ElementAccumulator = float;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    constexpr CO2Layout fixpipeFmt = CO2Layout::ROW_MAJOR;

    auto runOne = [&](auto dstTag, QuantMode_t expectedQuant) {
        using ElementDst = decltype(dstTag);
        constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
            ScaleGranularity::PER_TENSOR>::VALUE;
        ASSERT_EQ(quantPre, expectedQuant);

        CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>,
            ScaleGranularity::PER_TENSOR> copyL0CToGm;

        AscendC::GlobalTensor<ElementDst> gmTensor;
        AscendC::LocalTensor<ElementAccumulator> l0cTensor;
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setShape<ElementAccumulator>();
        setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);

        float _scale = 2.0;
        uint64_t _scale_uint64 = static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(&_scale));
        copyL0CToGm.params.scale = _scale;
        copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, _unitFlag);

        AscendCCallLogger& logger = AscendCCallLogger::Instance();
        auto logs = logger.GetLogs();
        ASSERT_EQ(logs.size(), 1);
        AscendCCallLog logFixpipe = logs[0];
        BaseCheck<ElementDst, ElementAccumulator, true>(logFixpipe, gmTensor, l0cTensor);

        const AscendC::FixpipeParamsC310<fixpipeFmt>* fixpipeArg =
            logFixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<fixpipeFmt>>();
        ASSERT_EQ(fixpipeArg->quantPre, quantPre);
        ASSERT_EQ(fixpipeArg->deqScalar, _scale_uint64);
        ASSERT_EQ(fixpipeArg->unitFlag, _unitFlag);
        logger.Clear();
    };

    runOne(fp8_e4m3fn_t{}, QuantMode_t::QF322FP8_PRE);
    runOne(hifloat8_t{}, QuantMode_t::QF322HIF8_PRE);
}

// Data-path: zN (L0C) → RowMajor (GM)
// Element-type: float → fp8_e4m3fn_t / hifloat8_t
// Speciality: PerChannel QuantMode mapping (VQF322FP8_PRE / VQF322HIF8_PRE)
TEST_P(TileCopyL0CToGmTestAscend950, zNToRowMajorTestPerChannelFp8Hif8)
{
    using ElementAccumulator = float;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    constexpr CO2Layout fixpipeFmt = CO2Layout::ROW_MAJOR;

    auto runOne = [&](auto dstTag, QuantMode_t expectedQuant) {
        using ElementDst = decltype(dstTag);
        constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementAccumulator, ElementDst,
            ScaleGranularity::PER_CHANNEL>::VALUE;
        ASSERT_EQ(quantPre, expectedQuant);

        CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>,
            ScaleGranularity::PER_CHANNEL> copyL0CToGm;

        AscendC::GlobalTensor<ElementDst> gmTensor;
        AscendC::LocalTensor<ElementAccumulator> l0cTensor;
        AscendC::LocalTensor<uint64_t> scaleTensor;
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setShape<ElementAccumulator>();
        setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);
        ASSERT_TRUE(isContiguous(layoutSrc));
        ASSERT_TRUE(isContiguous(layoutDst));

        copyL0CToGm(gmTensor, l0cTensor, scaleTensor, layoutDst, layoutSrc, _unitFlag);

        AscendCCallLogger& logger = AscendCCallLogger::Instance();
        auto logs = logger.GetLogs();
        ASSERT_EQ(logs.size(), 1);
        AscendCCallLog logFixpipe = logs[0];
        ASSERT_EQ(logFixpipe.name, "Fixpipe");
        ASSERT_EQ(logFixpipe.args.size(), 4);

        auto logGmTensor = logFixpipe.GetArgsAt(0).RawValue();
        auto logL0cTensor = logFixpipe.GetArgsAt(1).RawValue();
        auto logScaleTensor = logFixpipe.GetArgsAt(2).RawValue();
        ASSERT_EQ(logGmTensor, &gmTensor);
        ASSERT_EQ(logL0cTensor, &l0cTensor);
        ASSERT_EQ(logScaleTensor, &scaleTensor);

        const AscendC::FixpipeParamsC310<fixpipeFmt>* fixpipeArg =
            logFixpipe.GetArgsAt(3).Value<AscendC::FixpipeParamsC310<fixpipeFmt>>();
        ASSERT_EQ(fixpipeArg->quantPre, quantPre);
        ASSERT_EQ(fixpipeArg->unitFlag, _unitFlag);
        logger.Clear();
    };

    runOne(fp8_e4m3fn_t{}, QuantMode_t::VQF322FP8_PRE);
    runOne(hifloat8_t{}, QuantMode_t::VQF322HIF8_PRE);
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyL0CToGm,
    TileCopyL0CToGmTestAscend950,
    ::testing::Values(
        TestMatrixShapeWithUnitflag{32U, 32U, false},   // aligned
        TestMatrixShapeWithUnitflag{32U, 32U, true},    // aligned
        TestMatrixShapeWithUnitflag{17U, 32U, false},   // not aligned
        TestMatrixShapeWithUnitflag{17U, 32U, true},    // not aligned
        TestMatrixShapeWithUnitflag{128U, 33U, true},    // larger with unaligned
        TestMatrixShapeWithUnitflag{128U, 33U, false}   // larger with unaligned
    )
);

#endif // CATLASS_ARCH == 3510
