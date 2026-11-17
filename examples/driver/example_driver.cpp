/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is part of the CATLASS examples.
 *
 * Runs example programs one after another inside a single process, so the
 * device is acquired once for a whole batch instead of once per example.
 *
 * Each example is built a second time as a shared library whose main() is
 * renamed (see examples/CMakeLists.txt) and whose ACL lifecycle calls are
 * redirected here (see common/example_driver_shim.h). The examples' own
 * sources are not modified and their standalone executables are unaffected.
 *
 * The examples still run strictly one at a time, in the order given: this
 * removes repeated device acquisition, not the serialisation of kernels on the
 * card.
 *
 * Protocol, so the caller can interleave other work between cases (a test
 * generates a case's input immediately before running it): read one case per
 * line from stdin as tab-separated fields -- the library path followed by the
 * arguments the standalone executable would receive -- run it, then write
 *
 *     ##CATLASS_CASE_DONE rc=<returncode>
 *
 * to stdout once the example's own output has been flushed, on a line of its
 * own. A case the driver cannot run reports
 *
 *     ##CATLASS_CASE_UNSUPPORTED
 *
 * instead, which asks the caller to run that example as its standalone binary.
 * An empty stdin ends the session.
 */
#include <acl/acl.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kDoneSentinel = "##CATLASS_CASE_DONE rc=";
// The driver cannot give this example the run it asked for -- the library will
// not load, or the example wants a device other than the one the batch holds.
// The caller falls back to the standalone binary rather than failing the case.
constexpr const char* kUnsupportedSentinel = "##CATLASS_CASE_UNSUPPORTED";

int g_deviceId = -1;
bool g_aclInitialised = false;
// Set when the running example asks for a device other than the batch's. The
// batch holds one card, so that run is not the one the example asked for and
// its result is discarded in favour of the standalone binary's.
bool g_deviceMismatch = false;

/// Release the device once the batch is finished, rather than after every
/// example. main() calls this on every exit path: atexit() is not used, since a
/// handler that fails to run would leave the device held.
void DriverTeardown()
{
    if (g_deviceId >= 0) {
        aclrtResetDevice(g_deviceId);
        g_deviceId = -1;
    }
    if (g_aclInitialised) {
        aclFinalize();
        g_aclInitialised = false;
    }
}

/// The example's entry point, after main() was renamed at compile time. Every
/// example declares argv as const char**, which is what this mangling is; one
/// that did not would simply be reported unsupported and run standalone.
void* ResolveEntry(void* handle)
{
    return dlsym(handle, "_Z21catlass_example_entryiPPKc");
}

/// One example's outcome. `supported` is kept apart from `rc` because -1 is an
/// ordinary return code here -- examples/common/options.hpp returns it for bad
/// arguments -- so a sentinel value could not tell "the driver cannot run this"
/// from "the example rejected its input", and the caller must distinguish them:
/// the first is re-run standalone, the second is the case's answer.
struct CaseResult {
    bool supported;
    int rc;
};

constexpr CaseResult kUnsupportedCase{false, 0};

/// Run one example. Reports it unsupported when the library cannot be loaded or
/// the example asked for another device, which is the caller's signal to run it
/// standalone.
CaseResult RunCase(const std::vector<std::string>& fields)
{
    // RTLD_LOCAL keeps each example's symbols to itself: every example defines
    // Run() and its own helpers at namespace scope, so a global load would
    // collide across the suite.
    void* handle = dlopen(fields[0].c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "driver: cannot open " << fields[0] << ": " << dlerror() << std::endl;
        return kUnsupportedCase;
    }
    void* entry = ResolveEntry(handle);
    if (entry == nullptr) {
        std::cerr << "driver: no example entry point in " << fields[0] << std::endl;
        dlclose(handle);
        return kUnsupportedCase;
    }
    std::vector<const char*> argv;
    argv.reserve(fields.size());
    for (const std::string& field : fields) {
        argv.push_back(field.c_str());
    }
    auto example = reinterpret_cast<int (*)(int, const char**)>(entry);
    g_deviceMismatch = false;
    int rc = example(static_cast<int>(argv.size()), argv.data());
    if (g_deviceMismatch) {
        return kUnsupportedCase;
    }
    return CaseResult{true, rc};
}

/// Flush every stream the example may have written before the sentinel, so the
/// caller never reads the marker ahead of the output it delimits.
void FlushAll()
{
    std::cout.flush();
    std::cerr.flush();
    fflush(nullptr);
}

std::vector<std::string> SplitTabs(const std::string& line)
{
    std::vector<std::string> fields;
    std::string::size_type start = 0;
    while (start <= line.size()) {
        std::string::size_type tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

} // namespace

extern "C" aclError CatlassDriverInitOnce(const char* config)
{
    if (g_aclInitialised) {
        return ACL_SUCCESS;
    }
    aclError status = aclInit(config);
    if (status == ACL_SUCCESS) {
        g_aclInitialised = true;
    }
    return status;
}

extern "C" aclError CatlassDriverSetDeviceOnce(int32_t deviceId)
{
    if (g_deviceId == static_cast<int>(deviceId)) {
        return ACL_SUCCESS;
    }
    if (g_deviceId >= 0) {
        // One batch, one card, so this example cannot have the device it named.
        // Returning an error here would not stop it: ACL_CHECK prints and
        // carries on, and the printed "aclError" alone fails the case even
        // though the kernel went on to compute the right answer. Record the
        // mismatch instead and let RunCase report the case unsupported, so the
        // standalone binary produces the result on the device the example asked
        // for. A driver limitation costs a case its speed, never its verdict.
        g_deviceMismatch = true;
        return ACL_SUCCESS;
    }
    aclError status = aclrtSetDevice(deviceId);
    if (status == ACL_SUCCESS) {
        g_deviceId = static_cast<int>(deviceId);
    }
    return status;
}

extern "C" aclError CatlassDriverDeferReset(int32_t)
{
    return ACL_SUCCESS;
}

extern "C" aclError CatlassDriverDeferFinalize()
{
    return ACL_SUCCESS;
}

int main()
{
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<std::string> fields = SplitTabs(line);
        CaseResult result = fields.empty() ? CaseResult{true, 0} : RunCase(fields);
        FlushAll();
        // The leading newline terminates whatever the example wrote last. An
        // example whose final write has no newline of its own would otherwise
        // run into the sentinel and form a line that is neither the output nor
        // the marker, and the reader -- which waits for a line of its own --
        // would wait for the rest of the batch.
        if (result.supported) {
            std::cout << "\n" << kDoneSentinel << result.rc << std::endl;
        } else {
            std::cout << "\n" << kUnsupportedSentinel << std::endl;
        }
        std::cout.flush();
    }
    DriverTeardown();
    return 0;
}
