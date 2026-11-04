/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
#define EXAMPLES_COMMON_GOLDEN_MATMUL_HPP

#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"

namespace Catlass::golden {

// simple matmul
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmul(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = 0;
                if constexpr (std::is_same_v<LayoutA, layout::VectorLayout>) {
                    offsetA = layoutA.GetOffset(MakeCoord(k));
                } else {
                    offsetA = layoutA.GetOffset(MakeCoord(i, k));
                }
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
        }
    }
}

////////////////////////////////////
// new add
// simple gemm
template <
    typename Element, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    class ElementGolden, class LayoutGolden>
void ComputeGemm(
    const GemmCoord& problemShape, Element alpha, Element beta, const std::vector<ElementA>& dataA,
    const LayoutA& layoutA, const std::vector<ElementB>& dataB, const LayoutB& layoutB,
    const std::vector<ElementC>& dataC, const LayoutC& layoutC, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(alpha) * static_cast<ElementGolden>(dataA[offsetA]) *
                               static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] =
                static_cast<ElementGolden>(beta) * static_cast<ElementGolden>(dataC[offsetGolden]) +
                static_cast<ElementGolden>(accumulator);
        }
    }
}

template <
    typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY,
    class ElementGolden, class LayoutGolden>
void ComputeGemv(
    const Catlass::GemvCoord& problemShape, Element alpha, Element beta, const std::vector<ElementA>& dataA,
    const LayoutA& layoutA, const std::vector<ElementX>& dataX, const LayoutX& layoutX,
    const std::vector<ElementY>& dataY, const LayoutY& layoutY, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i));
        ElementGolden accumulator = 0;
        for (uint32_t k = 0; k < problemShape.n(); ++k) {
            size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
            size_t offsetX = layoutX.GetOffset(MakeCoord(k));
            accumulator += static_cast<ElementGolden>(alpha) * static_cast<ElementGolden>(dataA[offsetA]) *
                           static_cast<ElementGolden>(dataX[offsetX]);
        }
        size_t offsetY = layoutY.GetOffset(MakeCoord(i));
        dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) * static_cast<ElementGolden>(dataY[offsetY]) +
                                   static_cast<ElementGolden>(accumulator);
    }
}

// simple grouped gemm
template <
    typename Element, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    class ElementGolden, class LayoutGolden>
void ComputeGroupGemm(
    uint32_t problemCount, const std::vector<GemmCoord>& problemShapeList, const std::vector<Element>& alphaList,
    const std::vector<Element>& betaList, const std::vector<ElementA>& dataA, const std::vector<LayoutA>& layoutAList,
    const std::vector<ElementB>& dataB, const std::vector<LayoutB>& layoutBList, const std::vector<ElementC>& dataC,
    const std::vector<LayoutC>& layoutCList, std::vector<ElementGolden>& dataGolden,
    const std::vector<LayoutGolden>& layoutGoldenList)
{
    size_t inGroupOffsetA = 0;
    size_t inGroupOffsetB = 0;
    size_t inGroupOffsetC = 0;
    size_t inGroupOffsetGolden = 0;

    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        GemmCoord problemShape = problemShapeList[inGroupId];
        Element alpha = alphaList[inGroupId];
        Element beta = betaList[inGroupId];

        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = inGroupOffsetGolden + layoutGoldenList[inGroupId].GetOffset(MakeCoord(i, j));
                ElementGolden accumulator = 0;

                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = inGroupOffsetA + layoutAList[inGroupId].GetOffset(MakeCoord(i, k));
                    size_t offsetB = inGroupOffsetB + layoutBList[inGroupId].GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<ElementGolden>(alpha) * static_cast<ElementGolden>(dataA[offsetA]) *
                                   static_cast<ElementGolden>(dataB[offsetB]);
                }

                size_t offsetC = inGroupOffsetC + layoutCList[inGroupId].GetOffset(MakeCoord(i, j));
                dataGolden[offsetGolden] =
                    static_cast<ElementGolden>(beta) * static_cast<ElementGolden>(dataC[offsetC]) +
                    static_cast<ElementGolden>(accumulator);
            }
        }

        inGroupOffsetA += static_cast<size_t>(problemShape.m()) * problemShape.k();
        inGroupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        inGroupOffsetC += static_cast<size_t>(problemShape.m()) * problemShape.n();
        inGroupOffsetGolden += static_cast<size_t>(problemShape.m()) * problemShape.n();
    }
}
////////////////////////////////////

// simple batched matmul
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeBatchedMatmul(
    const uint32_t batchedCount, const GemmCoord& problemShape, const std::vector<ElementA>& dataA,
    const LayoutA& layoutA, const std::vector<ElementB>& dataB, const LayoutB& layoutB,
    std::vector<ElementGolden>& dataC, const LayoutGolden& layoutGolden)
{
    for (uint32_t batchId = 0; batchId < batchedCount; ++batchId) {
        size_t batchOffsetA = static_cast<size_t>(problemShape.m()) * problemShape.k() * batchId;
        size_t batchOffsetB = static_cast<size_t>(problemShape.k()) * problemShape.n() * batchId;
        size_t batchoffsetGolden = static_cast<size_t>(problemShape.m()) * problemShape.n() * batchId;
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j)) + batchoffsetGolden;
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k)) + batchOffsetA;
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j)) + batchOffsetB;
                    accumulator +=
                        static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataC[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
    }
}

// simple grouped matmul
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeGroupedMatmul(
    uint32_t problemCount, const std::vector<GemmCoord>& problemShapeList, const std::vector<ElementA>& dataA,
    const std::vector<LayoutA>& layoutAList, const std::vector<ElementB>& dataB,
    const std::vector<LayoutB>& layoutBList, std::vector<ElementGolden>& dataGolden,
    const std::vector<LayoutGolden>& layoutGoldenList)
{
    size_t inGroupOffsetA = 0;
    size_t inGroupOffsetB = 0;
    size_t inGroupOffsetGolden = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        GemmCoord problemShape = problemShapeList[inGroupId];
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = inGroupOffsetGolden + layoutGoldenList[inGroupId].GetOffset(MakeCoord(i, j));
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = inGroupOffsetA + layoutAList[inGroupId].GetOffset(MakeCoord(i, k));
                    size_t offsetB = inGroupOffsetB + layoutBList[inGroupId].GetOffset(MakeCoord(k, j));
                    accumulator +=
                        static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
        inGroupOffsetA += static_cast<size_t>(problemShape.m()) * problemShape.k();
        inGroupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        inGroupOffsetGolden += static_cast<size_t>(problemShape.m()) * problemShape.n();
    }
}

// matmul add
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB,
    class ElementX, // Layout X must be same as LayoutGolden
    class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseAdd(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB,
    const std::vector<ElementX>& dataX, // layoutX must be same as layoutGolden
    std::vector<ElementGolden>& dataGolden, const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = accumulator + static_cast<ElementGolden>(dataX[offsetGolden]);
        }
    }
}

template <class ElementGroupList, class ElementScale, class LayoutB>
void ComputeGroupedMatmulFixpipeDequant(
    const GemmCoord& problemShape, uint32_t problemCount, const std::vector<ElementGroupList>& groupList,
    const std::vector<int8_t>& dataA, const layout::RowMajor& layoutA, const std::vector<int8_t>& dataB,
    const LayoutB& layoutB, std::vector<float>& dataGolden, const layout::RowMajor& layoutGolden,
    const int32_t quantMode, const ElementScale& dataScale = 1.0,
    const std::vector<ElementScale>& dataPerChannelScale = {})
{
    size_t groupOffsetB = 0;
    size_t groupOffsetScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = startRow; i < groupList[inGroupId]; ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = groupOffsetB + layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                if (quantMode == 0) { // perTensor
                    dataGolden[offsetGolden] = static_cast<float>(accumulator) * static_cast<float>(dataScale);
                } else if (quantMode == 1) { // perChannel
                    dataGolden[offsetGolden] =
                        static_cast<float>(accumulator) * static_cast<float>(dataPerChannelScale[j]);
                } else {
                    std::cerr << "[ERROR] quantMode must be '0' or '1'." << std::endl;
                    return;
                }
            }
        }
        groupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        startRow = groupList[inGroupId];
    }
}

template <class ElementGroupList, class ElementScale, class LayoutB, class LayoutScale, class LayoutPerTokenScale>
void ComputeGroupedMatmulPerTokenDequant(
    const GemmCoord& problemShape, uint32_t problemCount, const std::vector<ElementGroupList>& groupList,
    const std::vector<int8_t>& dataA, const layout::RowMajor& layoutA, const std::vector<int8_t>& dataB,
    const LayoutB& layoutB, const std::vector<ElementScale>& dataScale, const LayoutScale&,
    const std::vector<ElementScale>& dataPerTokenScale, const LayoutPerTokenScale&, std::vector<float>& dataGolden,
    const layout::RowMajor& layoutGolden)
{
    size_t groupOffsetB = 0;
    size_t groupOffsetScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = startRow; i < groupList[inGroupId]; ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = groupOffsetB + layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<float>(accumulator) *
                                           static_cast<float>(dataScale[groupOffsetScale + j]) *
                                           static_cast<float>(dataPerTokenScale[i]);
            }
        }
        groupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        startRow = groupList[inGroupId];
    }
}

template <class LayoutA, class LayoutB, class LayoutScale, class LayoutPerTokenScale, class ElementScale>
void QuantMatmul(
    const GemmCoord& problemShape, const std::vector<int8_t>& dataA, const LayoutA& layoutA,
    const std::vector<int8_t>& dataB, const LayoutB& layoutB, const std::vector<ElementScale>& dataScale,
    const LayoutScale& layoutScale, const std::vector<ElementScale>& dataPerTokenScale,
    const LayoutPerTokenScale& layoutPerTokenScale, std::vector<float>& dataGolden,
    const layout::RowMajor& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            int32_t accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = static_cast<float>(accumulator) * static_cast<float>(dataScale[j]) *
                                       static_cast<float>(dataPerTokenScale[i]);
        }
    }
}

template <class ElementGroupList, class ElementScale, class LayoutScale, class LayoutPerTokenScale>
void ComputeGroupedMatmulSliceKPerTokenDequant(
    const GemmCoord& problemShape, uint32_t problemCount, const std::vector<ElementGroupList>& groupList,
    const std::vector<int8_t>& dataA, const layout::ColumnMajor& layoutA, const std::vector<int8_t>& dataB,
    const layout::RowMajor& layoutB, const std::vector<ElementScale>& dataScale, const LayoutScale&,
    const std::vector<ElementScale>& dataPerTokenScale, const LayoutPerTokenScale&, std::vector<float>& dataGolden,
    const layout::RowMajor& layoutGolden)
{
    size_t groupOffsetD = 0;
    size_t groupOffsetScale = 0;
    size_t groupOffsetPerTokenScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = startRow; k < groupList[inGroupId]; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[groupOffsetD + offsetGolden] =
                    static_cast<float>(accumulator) * static_cast<float>(dataScale[groupOffsetScale + j]) *
                    static_cast<float>(dataPerTokenScale[groupOffsetPerTokenScale + i]);
            }
        }

        groupOffsetD += static_cast<size_t>(problemShape.m()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        groupOffsetPerTokenScale += static_cast<size_t>(problemShape.m());
        startRow = groupList[inGroupId];
    }
}

template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden,
    class ElementBias>
void ComputeMatmulBias(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, const std::vector<ElementBias>& dataBias,
    std::vector<ElementGolden>& dataGolden, const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = static_cast<ElementGolden>(dataBias[j]);
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = accumulator;
        }
    }
}

// matmul_activation
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden,
    class Activation>
void ComputeMatmulElemWiseActivation(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden, Activation activation)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = activation(accumulator);
        }
    }
}

template <class ElementT>
struct Relu {
    ElementT operator()(ElementT val) const
    {
        return val < ElementT(0) ? ElementT(0) : val;
    }
};

template <class ElementT>
struct Sigmoid {
    ElementT operator()(ElementT val) const
    {
        // 使用类型安全的数学函数调用
        using std::exp; // 启用ADL查找
        const ElementT one = static_cast<ElementT>(1);
        return one / (one + exp(-val)); // 全部使用ElementT类型
    }
};

template <class ElementT>
struct Silu {
    ElementT operator()(ElementT val) const
    {
        return val * Sigmoid<ElementT>{}(val);
    }
};

template <class ElementT>
struct Gelu {
    ElementT operator()(ElementT val) const
    {
        static constexpr ElementT kScale = 1.59576912; // 2 * sqrt(2/π)
        static constexpr ElementT kCubicCoef = 0.044715;

        // 更稳定的计算方式
        const ElementT x_cubed = val * val * val;
        const ElementT inner = val + kCubicCoef * x_cubed;

        return val * Sigmoid<ElementT>{}(kScale * inner);
    }
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseRelu(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    ComputeMatmulElemWiseActivation(
        problemShape, dataA, layoutA, dataB, layoutB, dataGolden, layoutGolden, Relu<ElementGolden>{});
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseGelu(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    ComputeMatmulElemWiseActivation(
        problemShape, dataA, layoutA, dataB, layoutB, dataGolden, layoutGolden, Gelu<ElementGolden>{});
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseSilu(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    ComputeMatmulElemWiseActivation(
        problemShape, dataA, layoutA, dataB, layoutB, dataGolden, layoutGolden, Silu<ElementGolden>{});
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseLeakyRelu(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden, ElementGolden scalar)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = accumulator < ElementGolden(0) ? scalar * accumulator : accumulator;
        }
    }
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseTanh(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = tanh(accumulator);
        }
    }
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmulElemWiseSigmoid(
    const GemmCoord& problemShape, const std::vector<ElementA>& dataA, const LayoutA& layoutA,
    const std::vector<ElementB>& dataB, const LayoutB& layoutB, std::vector<ElementGolden>& dataGolden,
    const LayoutGolden& layoutGolden)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = 1 / (1 + exp(-accumulator));
        }
    }
}

template <class LayoutA, class LayoutB, class LayoutScalex1, class LayoutScalex2>
void QuantMatmulPergroupPerBlockDequant(
    const GemmCoord& problemShape, const std::vector<int8_t>& dataA, const LayoutA& layoutA,
    const std::vector<int8_t>& dataB, const LayoutB& layoutB, const std::vector<float>& dataScalex1,
    const LayoutScalex1& layoutScalex1, const std::vector<float>& dataScalex2, const LayoutScalex2& layoutScalex2,
    std::vector<float>& dataGolden, const layout::RowMajor& layoutGolden, uint32_t group_size, uint32_t block_size)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            float accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                size_t offset_x1 = layoutScalex1.GetOffset(MakeCoord(i, k / group_size));
                size_t offset_x2 = layoutScalex2.GetOffset(MakeCoord(k / block_size, j / block_size));
                accumulator += static_cast<float>(dataA[offsetA]) * static_cast<float>(dataB[offsetB]) *
                               static_cast<float>(dataScalex1[offset_x1]) * static_cast<float>(dataScalex2[offset_x2]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = static_cast<float>(accumulator);
        }
    }
}

// Decode each output coordinate once. Broadcast matrices have a zero batch stride.
template <class Element, class LayoutA, class LayoutB>
void ComputeBatchedMatmulBroadcast(
    const GemmCoord& problemShape, uint32_t batchA, uint32_t batchB, uint32_t biasMode,
    const std::vector<Element>& hostA, const std::vector<Element>& hostB, const std::vector<Element>& hostBias,
    std::vector<float>& hostGolden)
{
    size_t m = problemShape.m();
    size_t n = problemShape.n();
    size_t k = problemShape.k();
    size_t batchStrideA = 0;
    size_t batchStrideB = 0;
    size_t batchStrideBias = 0;
    if (batchA != 1) {
        batchStrideA = m * k;
    }
    if (batchB != 1) {
        batchStrideB = k * n;
    }
    if (biasMode == 2) {
        batchStrideBias = n;
    }
    LayoutA layoutA(problemShape.m(), problemShape.k());
    LayoutB layoutB(problemShape.k(), problemShape.n());
    for (size_t outputOffset = 0; outputOffset < hostGolden.size(); ++outputOffset) {
        size_t batchIndex = outputOffset / (m * n);
        uint32_t row = (outputOffset / n) % m;
        uint32_t column = outputOffset % n;
        float accumulator = 0.0f;
        if (biasMode != 0) {
            accumulator = static_cast<float>(hostBias[batchIndex * batchStrideBias + column]);
        }
        for (uint32_t inner = 0; inner < k; ++inner) {
            size_t offsetA = batchIndex * batchStrideA + layoutA.GetOffset(MakeCoord(row, inner));
            size_t offsetB = batchIndex * batchStrideB + layoutB.GetOffset(MakeCoord(inner, column));
            accumulator += static_cast<float>(hostA[offsetA]) * static_cast<float>(hostB[offsetB]);
        }
        hostGolden[outputOffset] = static_cast<float>(static_cast<Element>(accumulator));
    }
}

// Preserve the example's dtype-specific tolerances and output guard checks.
template <class Element, bool IsBfloat16, bool HF32>
size_t CompareBatchedMatmulBroadcast(
    const std::vector<Element>& hostC, const std::vector<float>& hostGolden, size_t guardElements, Element guardValue)
{
    double tolerance = 0.0001;
    if constexpr (IsBfloat16) {
        tolerance = 0.016;
    } else if constexpr (HF32) {
        tolerance = 0.005;
    } else if constexpr (sizeof(Element) == 2) {
        tolerance = 0.002;
    }
    size_t errors = 0;
    for (size_t index = 0; index < hostGolden.size(); ++index) {
        float actual = static_cast<float>(hostC[guardElements + index]);
        float expected = hostGolden[index];
        double difference = std::abs(static_cast<double>(actual) - expected);
        if (!std::isfinite(actual) || difference > tolerance + tolerance * std::abs(expected)) {
            ++errors;
        }
    }
    for (size_t index = 0; index < guardElements; ++index) {
        if (static_cast<float>(hostC[index]) != static_cast<float>(guardValue) ||
            static_cast<float>(hostC[guardElements + hostGolden.size() + index]) != static_cast<float>(guardValue)) {
            ++errors;
        }
    }
    return errors;
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
