/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_TENSOR_H
#define ASCENDC_STUB_KERNEL_TENSOR_H

#include <cstdio>
#include <string>
#include <typeinfo>
#include <cstdint>
#include "arg.h"
#include "ascendc_logger.h"

namespace AscendC {

template <typename T>
class LocalTensor {
    template <typename> friend class LocalTensor;
public:
    using PrimType = T;

    LocalTensor() : position_(0), size_(0), addr_(0)
    {}

    LocalTensor(uint32_t addr, uint32_t size) : position_(0), size_(size), addr_(addr)
    {}

    uint32_t GetSize() const
    {
        return size_;
    }
    void SetSize(uint32_t size)
    {
        size_ = size;
    }
    int32_t GetPosition() const
    {
        return position_;
    }

    uint32_t GetAddr() const
    {
        return addr_;
    }

    void SetAddr(uint32_t addr)
    {
        addr_ = addr;
    }

    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        LocalTensor<U> result;
        result.position_ = position_;
        result.size_ = size_;
        result.addr_ = addr_;
        const std::vector<Arg> argsT = {Arg::MakeArg<U>()};
        const std::vector<Arg> args = {Arg::MakeArgWithValue(addr_)};
        ASCENDC_LOG_CALL_T(ReinterpretCast, argsT, args);
        return result;
    }

    LocalTensor operator[](uint32_t offset) const
    {
        LocalTensor result = *this;
        result.addr_ += offset * sizeof(T);
        result.size_ -= offset;
        return result;
    }

    std::string toString() const
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LocalTensor<%s>(addr=%u, size=%u)", typeid(T).name(), addr_, size_);
        return std::string(buffer);
    }

private:
    int32_t position_;
    uint32_t size_;
    uint32_t addr_;
};

template <typename T>
class GlobalTensor {
    template <typename> friend class GlobalTensor;
public:
    using PrimType = T;

    GlobalTensor() : bufferSize_(0), addr_(nullptr)
    {}

    template <typename U>
    void SetGlobalBuffer(U* buffer, uint64_t bufferSize)
    {
        addr_ = reinterpret_cast<T*>(buffer);
        bufferSize_ = bufferSize;
    }

    uint64_t GetSize() const
    {
        return bufferSize_;
    }

    uintptr_t GetAddr() const
    {
        return reinterpret_cast<uintptr_t>(addr_);
    }

    GlobalTensor operator[](uint64_t offset) const
    {
        GlobalTensor result = *this;
        result.addr_ = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(addr_) + offset * sizeof(T));
        result.bufferSize_ -= offset;
        return result;
    }

    template <typename U>
    GlobalTensor<U> ReinterpretCast() const
    {
        GlobalTensor<U> result;
        result.addr_ = reinterpret_cast<U*>(addr_);
        result.bufferSize_ = bufferSize_;
        return result;
    }

    std::string toString() const
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "GlobalTensor<%s>(size=%lu)", typeid(T).name(), bufferSize_);
        return std::string(buffer);
    }

private:
    uint64_t bufferSize_;
    T* addr_;
};

} // namespace AscendC

template <typename T>
struct ArgUseRef<AscendC::LocalTensor<T>> : std::true_type {};

template <typename T>
struct ArgUseRef<AscendC::GlobalTensor<T>> : std::true_type {};
#endif // ASCENDC_STUB_KERNEL_TENSOR_H
