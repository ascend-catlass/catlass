/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE,
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_COMMON_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_COMMON_HPP

#include <cstdint>

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Block {

constexpr uint32_t BNSD = 0;
constexpr uint32_t SBH = 1;
constexpr uint32_t BSND = 2;
constexpr uint32_t TND = 3;

// Softmax 通用定义。
constexpr uint8_t SOFTMAX_BASIC_TILE_NUM = 8;
constexpr uint8_t SOFTMAX_COMPUTE_DIM = 2;
constexpr uint8_t SOFTMAXGRAD_COMPUTE_DIM = 3;
constexpr uint32_t SCALAR_STACK_DEPTH = 8;
constexpr uint32_t FLOAT_NUM_PER_BLK = AscendC::ONE_BLK_SIZE / AscendC::B32_BYTE_SIZE;
constexpr uint32_t HALF_REPEAT_STRIDE = AscendC::DEFAULT_REPEAT_STRIDE / AscendC::B16_BYTE_SIZE;

struct LastAxisShapeND {
    uint32_t m;
    uint32_t k;
};

struct SoftMaxTiling {
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;
};

__aicore__ inline LastAxisShapeND FagGetLastAxisShapeND(const AscendC::ShapeInfo& shapeInfo)
{
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < shapeInfo.shapeDim; i++) {
        calculateSize *= shapeInfo.shape[i];
    }
    LastAxisShapeND ndinfo;
    ndinfo.k = shapeInfo.shape[shapeInfo.shapeDim - 1];
    ndinfo.m = calculateSize / ndinfo.k;
    return ndinfo;
}

__aicore__ inline LastAxisShapeND FagGetLastAxisOriginShapeND(const AscendC::ShapeInfo& srcShapeInfo)
{
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < srcShapeInfo.originalShapeDim; i++) {
        calculateSize *= srcShapeInfo.originalShape[i];
    }
    LastAxisShapeND ndinfo;
    ndinfo.k = srcShapeInfo.originalShape[srcShapeInfo.originalShapeDim - 1];
    ndinfo.m = calculateSize / ndinfo.k;
    return ndinfo;
}

struct DBParams {
    int64_t blockId;
    int64_t taskId;
    int64_t bIdx;
    int64_t n2Idx;
    int64_t s2oIdx;
    int64_t gIdx;
    int64_t s1oIdx;
    int32_t s1CvExtend;
    int32_t s2CvExtend;
    int32_t s1CvExtendAlign;
    int32_t s2CvExtendAlign;
    int64_t aTensorOffsetCv{0};
    int64_t bTensorOffsetCv{0};
    int64_t actualS1Len{0};
    int64_t actualS2Len{0};
    int64_t s1Stride;
    int64_t s2Stride;
    int64_t blockIdArr[24];
    int32_t s1CvExtendArr[24];
    int32_t s2CvExtendArr[24];
    int8_t dqGroupId[24];
    int8_t kvGroupId[24];
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_COMMON_HPP
