/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is part of the CATLASS examples.
 *
 * Redirects an example's ACL lifecycle calls to the example driver, so that a
 * batch of examples sharing one process acquires the device once.
 *
 * This header is force-included (-include) only when an example is compiled
 * into the driver's shared-library form; the standalone executables never see
 * it and keep calling ACL directly. Example sources are therefore unchanged,
 * and remain the reference for correct ACL usage.
 *
 * aclrtSetDevice costs ~1.5 s and aclrtResetDevice ~5 s on Ascend950PR, against
 * ~1.3 s of actual work in a typical example, so a suite of 50 examples spends
 * roughly 325 s acquiring and releasing a device it could hold throughout.
 *
 * The hooks are declared but NOT defined here. Each example library is opened
 * with RTLD_LOCAL, so a definition in this header would give every library its
 * own copy of the lifecycle state and the second example would call the real
 * aclInit again -- which returns 100002 and, because ACL_CHECK only prints,
 * surfaces as an "aclError" line that the test harness treats as a failure.
 * Leaving them undefined makes each library resolve them against the driver
 * executable instead, so one piece of state governs them all.
 */
#ifndef EXAMPLES_COMMON_EXAMPLE_DRIVER_SHIM_H
#define EXAMPLES_COMMON_EXAMPLE_DRIVER_SHIM_H

#include <acl/acl.h>

extern "C" {
aclError CatlassDriverInitOnce(const char* config);
aclError CatlassDriverSetDeviceOnce(int32_t deviceId);
aclError CatlassDriverDeferReset(int32_t deviceId);
aclError CatlassDriverDeferFinalize();
}

#define aclInit(config) CatlassDriverInitOnce(config)
#define aclrtSetDevice(deviceId) CatlassDriverSetDeviceOnce(deviceId)
#define aclrtResetDevice(deviceId) CatlassDriverDeferReset(deviceId)
#define aclFinalize() CatlassDriverDeferFinalize()

#endif // EXAMPLES_COMMON_EXAMPLE_DRIVER_SHIM_H
