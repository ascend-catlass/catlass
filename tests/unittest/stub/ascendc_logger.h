/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_ASCENDC_LOGGER_H
#define ASCENDC_STUB_ASCENDC_LOGGER_H

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <any>
#include <memory>
#include <typeindex>
#include "arg.h"

namespace Catlass {
namespace Test {

struct AscendCCallLog {
    std::string name;
    std::vector<Arg> argsT;
    std::vector<Arg> args;

    explicit AscendCCallLog(const std::string& name) : name(name)
    {}
    /**
     * @brief 获取指定索引位置的参数
     * @param index 参数索引
     * @return 参数值，若索引超出范围则返回默认值
     */
    const Arg GetArgsAt(size_t index) const
    {
        if (index >= args.size())
            return Arg::MakeArg<void>();
        return args[index];
    }
    /**
     * @brief 获取指定索引位置的 `模板` 参数
     * @param index 参数索引
     * @return 参数类型，若索引超出范围则返回默认类型
     */
    const Arg GetArgsTAt(size_t index) const
    {
        if (index >= argsT.size())
            return Arg::MakeArg<void>();
        return argsT[index];
    }
};

class AscendCCallLogger {
public:
    static AscendCCallLogger& Instance()
    {
        thread_local static AscendCCallLogger instance;
        return instance;
    }
    /**
     * @brief 记录一次AscendC函数调用
     * @param name 函数名
     */
    void Log(const std::string& name)
    {
        Log(name, {}, {});
    }
    /**
     * @brief 记录一次AscendC函数调用，带参数
     * @param name 函数名
     * @param args 参数列表
     */
    void Log(const std::string& name, const std::vector<Arg>& args)
    {
        Log(name, {}, args);
    }
    /**
     * @brief 记录一次AscendC函数调用，带参数和模板参数
     * @param name 函数名
     * @param argsT 参数类型列表
     * @param args 参数列表
     */
    void Log(const std::string& name, const std::vector<Arg>& argsT, const std::vector<Arg>& args)
    {
        AscendCCallLog logEntry(name);
        logEntry.argsT = argsT;
        for (size_t i = 0; i < args.size(); ++i) {
            logEntry.args.push_back(args[i]);
        }
        logs_.push_back(std::move(logEntry));
    }
    /**
     * @brief 清空记录的AscendC函数调用日志
     */
    void Clear()
    {
        logs_.clear();
    }
    /**
     * @brief 获取记录的AscendC函数调用日志
     * @return 日志列表
     */
    const std::vector<AscendCCallLog>& GetLogs() const
    {
        return logs_;
    }
    /**
     * @brief 获取数据搬运日志，排除只做类型视图转换的 ReinterpretCast
     * @return 日志列表
     */
    std::vector<AscendCCallLog> GetCopyLogs() const
    {
        std::vector<AscendCCallLog> copyLogs;
        copyLogs.reserve(logs_.size());
        for (const auto& log : logs_) {
            if (log.name != "ReinterpretCast") {
                copyLogs.push_back(log);
            }
        }
        return copyLogs;
    }

private:
    AscendCCallLogger() = default;
    std::vector<AscendCCallLog> logs_;
};

#define ASCENDC_LOG_CALL(FUNC_NAME, ARGS) ::Catlass::Test::AscendCCallLogger::Instance().Log(#FUNC_NAME, ARGS)

#define ASCENDC_LOG_CALL_T(FUNC_NAME, ARGST, ARGS) \
    ::Catlass::Test::AscendCCallLogger::Instance().Log(#FUNC_NAME, ARGST, ARGS)

#define ASCENDC_CLEAR_LOGS() ::Catlass::Test::AscendCCallLogger::Instance().Clear()

#define ASCENDC_GET_LOGS() ::Catlass::Test::AscendCCallLogger::Instance().GetLogs()

} // namespace Test
} // namespace Catlass

#endif // ASCENDC_STUB_ASCENDC_LOGGER_H
