/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_UTILS_TYPE_TRAITS_H
#define TEST_UTILS_TYPE_TRAITS_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace test_utils {

constexpr aclDataType AclDataTypeValue(int value)
{
    return static_cast<aclDataType>(value);
}

inline const std::vector<std::string> typeStrVec = {
    "float",        "float16",          "int8",         "int32",         "uint8",          "int16",
    "uint16",       "uint32",           "int64",        "uint64",        "double",         "bool",
    "string",       "complex64",        "complex128",   "bfloat16",      "int4",           "uint1",
    "complex32",    "_hifloat8",        "float8_e5m2",  "float8_e4m3fn", "float8_e8m0fnu", "_float6_e3m2",
    "_float6_e2m3", "float4_e2m1fn_x2", "_float4_e1m2",
};

inline const std::vector<torch::Dtype> typeTorchVec = {
    torch::kFloat32,        torch::kFloat16,
    torch::kInt8,           torch::kInt32,
    torch::kUInt8,          torch::kInt16,
    torch::kUInt16,         torch::kUInt32,
    torch::kInt64,          torch::kUInt64,
    torch::kDouble,         torch::kBool,
    torch::kFloat16,        torch::kComplexFloat,
    torch::kComplexDouble,  torch::kBFloat16,
    torch::kInt4,           torch::kUInt1,
    torch::kComplexHalf,    torch::kFloat16,
    torch::kFloat8_e5m2,    torch::kFloat8_e4m3fn,
    torch::kFloat8_e8m0fnu, torch::kFloat16,
    torch::kFloat16,
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
    torch::kFloat4_e2m1fn_x2,
#else
    torch::kFloat16, // float4_e2m1fn_x2 not available before torch 2.9
#endif
    torch::kFloat16,
};

inline const std::vector<aclDataType> typeAclVec = {
    ACL_FLOAT,       ACL_FLOAT16,     ACL_INT8,        ACL_INT32,         ACL_UINT8,       ACL_INT16,
    ACL_UINT16,      ACL_UINT32,      ACL_INT64,       ACL_UINT64,        ACL_DOUBLE,      ACL_BOOL,
    ACL_STRING,      ACL_COMPLEX64,   ACL_COMPLEX128,  ACL_BF16,          ACL_INT4,        ACL_UINT1,
    ACL_COMPLEX32,   AclDataTypeValue(34), AclDataTypeValue(35), AclDataTypeValue(36), AclDataTypeValue(37),
    AclDataTypeValue(38), AclDataTypeValue(39), AclDataTypeValue(40), AclDataTypeValue(41),
};

inline const std::vector<const char*> typeBishengStrVec = {
    "float",           "half",          "int8_t",        "int32_t",   "uint8_t",
    "int16_t",         "uint16_t",      "uint32_t",      "int64_t",   "uint64_t",
    "double",          "bool",          "std::string",   "complex64", nullptr,
    "bfloat16_t",      "int4x2_t",      nullptr,         "complex32", nullptr,
    "float8_e5m2_t",   "float8_e4m3_t", "float8_e8m0_t", nullptr,     nullptr,
    "float4_e2m1x2_t", "fp4x2_e1m2_t",
};

/**
 * @brief Return the index of a value inside a type mapping table.
 *
 * @tparam Vec Vector-like container type.
 * @tparam Val Value type to compare with the container entries.
 * @param vec Mapping table to scan.
 * @param val Value to locate.
 * @return Zero-based index on success, or -1 when the value is absent.
 */
template <typename Vec, typename Val>
inline int64_t FindTypeIndex(const Vec& vec, const Val& val)
{
    for (size_t i = 0; i < vec.size(); ++i)
        if (vec[i] == val)
            return static_cast<int64_t>(i);
    return -1;
}

/**
 * @brief Convert between CATLASS test dtype representations.
 *
 * Supported source/target representations are torch dtype, ACL dtype,
 * canonical string name, and bisheng C++ type string. The mapping tables above
 * intentionally share indexes, so conversion stays data-driven and central.
 *
 * @tparam S Source type.
 * @tparam T Target type.
 * @param src Source dtype value.
 * @return Converted dtype value.
 * @throws std::runtime_error when either representation is unsupported.
 */
template <typename S, typename T>
T TypeCast(const S& src)
{
    if constexpr (std::is_same_v<S, T>) {
        return src;
    }
    int64_t idx = -1;
    if constexpr (std::is_same_v<S, torch::Dtype>) {
        idx = FindTypeIndex(typeTorchVec, src);
    } else if constexpr (std::is_same_v<S, aclDataType>) {
        idx = FindTypeIndex(typeAclVec, src);
    } else if constexpr (std::is_same_v<S, std::string>) {
        idx = FindTypeIndex(typeStrVec, src);
    }
    if (idx == -1) {
        throw std::runtime_error("Invalid type conversion");
    }
    if constexpr (std::is_same_v<T, torch::Dtype>) {
        return typeTorchVec[idx];
    }
    if constexpr (std::is_same_v<T, aclDataType>) {
        return typeAclVec[idx];
    }
    if constexpr (std::is_same_v<T, const char*>) {
        const char* result = typeBishengStrVec[idx];
        if (result == nullptr) {
            throw std::runtime_error("No bisheng type mapping for requested dtype");
        }
        return result;
    }
    throw std::runtime_error("Invalid type conversion");
}

} // namespace test_utils

#endif
