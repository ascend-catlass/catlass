#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CATLASS example tests (pytest).

A session fixture initializes the build dir (cmake configure once); each case
incrementally builds its target bin before running, avoiding repeated cmake args.

Run:  python3 -m pytest tests/test_example.py -v
"""

import itertools
import atexit
import contextlib
import functools
import os
import random
import runpy
import re
import shutil
import subprocess
import tempfile
import traceback
import threading
import sys

import pytest

CMAKE_SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Dedicated build dir: not shared with test_compile.sh, avoids CMakeCache pollution
CMAKE_BUILD_DIR = os.path.join(CMAKE_SOURCE_DIR, "build", "test_example")
CMAKE_BINARY_PATH = os.path.join(CMAKE_SOURCE_DIR, "output", "bin")
CMAKE_EXAMPLES_PATH = os.path.join(CMAKE_SOURCE_DIR, "examples")
# Driver-loadable copies of the examples, and the driver that runs them. Each
# example acquires and releases the device itself, which on Ascend950PR costs
# ~6.5 s against ~1.3 s of actual work, so a suite of 50 spends most of its time
# taking and returning a device it could simply hold.
EXAMPLE_LIBS_TARGET = "catlass_example_libs"
EXAMPLE_LIBS_PATH = os.path.join(CMAKE_SOURCE_DIR, "output", "example_libs")
DRIVER_PATH = os.path.join(CMAKE_BINARY_PATH, "catlass_example_driver")
MERGED_DRIVER = os.environ.get("TEST_EXAMPLE_MERGED", "1") != "0"
# Generous enough never to fire on a healthy case, which run in seconds, while
# still ending a wedged one rather than letting it hang the suite.
RUN_TIMEOUT = 300
# Each gen_data helper imports torch, which costs 4.3 s of the 4.6 s a real
# helper takes; 33 of the cases call one. Running them here pays that once.
INPROC_HELPERS = os.environ.get("TEST_EXAMPLE_INPROC_HELPERS", "1") != "0"
# 102/103 dynamic matmul shared libs live in shared_lib (102) and shared_lib/lib (103)
CMAKE_SHARED_LIB_PATH = os.path.join(CMAKE_BINARY_PATH, "..", "shared_lib")
os.environ["LD_LIBRARY_PATH"] = (
    os.path.join(CMAKE_SHARED_LIB_PATH, "lib")
    + os.pathsep
    + CMAKE_SHARED_LIB_PATH
    + os.pathsep
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# NPU device id: from env DEVICE_ID (default 0), passed as the example deviceid arg
DEVICE_ID = os.environ.get("DEVICE_ID", "0")


def get_npu_arch():
    device_name = acl.get_soc_name()

    if re.match(r"Ascend910B.+", device_name, re.IGNORECASE) or re.search(r"Ascend910_93", device_name, re.IGNORECASE):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.IGNORECASE):
        return 3510
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


try:
    import numpy as np
except ImportError:  # a checkout with no runtime environment (lint, collect-only)
    np = None

try:
    import acl

    NPU_ARCH = get_npu_arch()
except Exception:  # noqa: BLE001 - no NPU environment (e.g. local lint / CI collect)
    acl = None
    NPU_ARCH = None

only_on_2201 = pytest.mark.skipif(NPU_ARCH != 2201, reason="This case only runs on 2201")
only_on_3510 = pytest.mark.skipif(NPU_ARCH != 3510, reason="This case only runs on 3510")


# ---------------------------------------------------------------------------
# build fixture: initialize the build dir (cmake configure once)
# ---------------------------------------------------------------------------
def _configure_build():
    """Clean the dedicated build dir and configure once.

    Always reconfigure from scratch so the 102/103 COMPILE_DYNAMIC_* options
    are guaranteed ON; the dedicated path never collides with test_compile.sh.
    """
    shutil.rmtree(CMAKE_BUILD_DIR, ignore_errors=True)
    # The driver's artifacts are installed into output/, outside the build dir,
    # and a case loads a library on existence alone. Remove them with the build
    # that produced them, so a library from an earlier session can never stand
    # in for one the current source fails to produce.
    shutil.rmtree(EXAMPLE_LIBS_PATH, ignore_errors=True)
    with contextlib.suppress(OSError):
        os.remove(DRIVER_PATH)
    os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)
    cmake_cmd = [
        "cmake",
        "-S",
        CMAKE_SOURCE_DIR,
        "-B",
        CMAKE_BUILD_DIR,
        f"-DCATLASS_ARCH={NPU_ARCH}",
        f"-DCMAKE_INSTALL_PREFIX={os.path.dirname(CMAKE_BINARY_PATH)}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if MERGED_DRIVER:
        cmake_cmd.append("-DCATLASS_BUILD_EXAMPLE_DRIVER=ON")
    # 102/103 dynamic matmul targets only exist when these options are ON at configure time
    if NPU_ARCH == 2201:
        cmake_cmd += [
            "-DCOMPILE_DYNAMIC_OPTIMIZED_MATMUL=ON",
            "-DCOMPILE_DYNAMIC_QBMM_OPTIMIZED_MATMUL=ON",
        ]
    _ = _run(cmake_cmd, check=True)


@pytest.fixture(scope="session")
def build_env():
    """Initialize the build dir (cmake configure once), return its path."""
    if NPU_ARCH is None:
        pytest.skip("No Ascend NPU environment")
    _configure_build()
    if MERGED_DRIVER:
        # Built as one target, not per case: run_case needs the library and the
        # driver before the first example. Neither step is fatal -- the examples
        # run standalone without them -- but both must succeed before any case
        # may use the driver, or a failed build would be served from whatever
        # happens to be on disk.
        global _DRIVER_ARTIFACTS_OK
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(os.cpu_count() or 1)
        built = _run(["cmake", "--build", CMAKE_BUILD_DIR, "--target", EXAMPLE_LIBS_TARGET,
              "-j", jobs], check=False)
        if built.returncode == 0:
            installed = _run(
                ["cmake", "--install", CMAKE_BUILD_DIR, "--component", EXAMPLE_LIBS_TARGET],
                check=False,
            )
            _DRIVER_ARTIFACTS_OK = installed.returncode == 0
    return CMAKE_BUILD_DIR


# ---------------------------------------------------------------------------
# run helper
# ---------------------------------------------------------------------------
def ret_check(ret: subprocess.CompletedProcess):
    """Fail unless stderr has no acl/rt/compare errors and returncode is 0."""
    for error_log_line in ret.stderr.decode().splitlines():
        acl_match = re.search(r"aclError:\s*([1-9][0-9]{5})", error_log_line)
        compare_match = re.search(r"Compare failed\. Error count:\s*([1-9][0-9]*)", error_log_line)
        acl_code = 0 if acl_match is None else int(acl_match.group(1))
        compare_code = 0 if compare_match is None else int(compare_match.group(1))
        assert acl_code == 0, f"There is an ACL error: {acl_code}"
        assert compare_code == 0, f"There is a compare error: {compare_code}"
    assert ret.returncode == 0, f"Return code is not zero: {ret.returncode}"


def _run(cmd, **kwargs):
    """``subprocess.run``, except a data-generation helper runs in this process."""
    if INPROC_HELPERS and _is_python_helper(cmd):
        return _run_helper_inproc(cmd, **kwargs)
    return subprocess.run(cmd, **kwargs)


# A helper that opens a device must keep its own process. The session's device
# belongs to the driver, and a helper sharing this interpreter would hold a
# second ACL context for as long as the suite runs -- a device the harness never
# releases, because nothing here ever exits. Matched on the source rather than a
# list of paths, so a helper that starts using a device is covered the day it
# does.
# Anchored patterns, not a substring search: a bare "npu" also appears inside
# "input", which every one of these helpers names.
_DEVICE_MARKERS = re.compile(
    r"""torch_npu | torch\.npu | \.npu\( | ["']npu["'] | \baclrt""", re.VERBOSE
)


@functools.lru_cache(maxsize=None)
def _helper_is_device_free(script: str) -> bool:
    try:
        with open(script, "r", errors="replace") as handle:
            source = handle.read()
    except OSError:
        # Unreadable here means unreadable in a child too, but a subprocess is
        # the conservative answer: it reports the failure without taking the
        # device decision on trust.
        return False
    return _DEVICE_MARKERS.search(source) is None


def _is_python_helper(cmd) -> bool:
    if len(cmd) < 2:
        return False
    exe = os.path.basename(str(cmd[0]))
    if not (exe.startswith("python") and str(cmd[1]).endswith(".py")):
        return False
    return _helper_is_device_free(str(cmd[1]))


@contextlib.contextmanager
def _captured_fds():
    """Capture file descriptors 1 and 2, the way a child process is captured.

    ``redirect_stdout`` rebinds ``sys.stdout`` only. A helper's numpy and torch
    layers write to the descriptors from C++, so that output would bypass the
    capture and never reach ``ret_check``'s scan -- which reads these streams to
    decide whether a case passed.
    """
    with tempfile.TemporaryFile() as raw_out, tempfile.TemporaryFile() as raw_err:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_fds = (os.dup(1), os.dup(2))
        saved_streams = (sys.stdout, sys.stderr)
        os.dup2(raw_out.fileno(), 1)
        os.dup2(raw_err.fileno(), 2)
        # Python's own writes go to the same place, through descriptors of their
        # own so closing these leaves 1 and 2 alone.
        sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        sys.stderr = os.fdopen(os.dup(2), "w", buffering=1)
        try:
            yield raw_out, raw_err
        finally:
            for stream in (sys.stdout, sys.stderr):
                with contextlib.suppress(ValueError, OSError):
                    stream.flush()
                    stream.close()
            sys.stdout, sys.stderr = saved_streams
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            os.close(saved_fds[0])
            os.close(saved_fds[1])


@contextlib.contextmanager
def _independent_rng():
    """Give the helper the independent generators a fresh interpreter gives it.

    Several helpers seed the process-global generators while being imported --
    ``np.random.seed(42)`` in example 63's, ``seed(1)`` in example 70's. Run as
    their own process that seeding was private to them. Shared here, it would
    carry into every helper that runs afterwards and seeds nothing of its own,
    so those inputs would depend on which cases ran before them and a case
    would generate different data under ``-k`` than in a full run.
    """
    saved_random = random.getstate()
    saved_numpy = None if np is None else np.random.get_state()
    random.seed(None)
    if np is not None:
        np.random.seed(None)
    try:
        yield
    finally:
        random.setstate(saved_random)
        if np is not None:
            np.random.set_state(saved_numpy)


def _run_helper_inproc(cmd, *, check=False, capture_output=False, cwd=None, **_):
    """Execute cmd as a script in this interpreter, mimicking subprocess.run.

    Returns a CompletedProcess with bytes streams, because callers decode them.
    A SystemExit becomes the return code and any other exception becomes a
    non-zero one, so ``check`` behaves exactly as it did with a real child.
    """
    script, args = str(cmd[1]), [str(a) for a in cmd[2:]]
    old_argv, old_path, old_cwd, rc = sys.argv, sys.path, os.getcwd(), 0
    # Several helpers set WORKSPACE and the ACL log variables at import time.
    # Those belong to the helper's own run: left in place they would reach every
    # example the suite runs afterwards, and one of them turns off the ACL log
    # output that a case is checked against.
    old_environ = dict(os.environ)
    out = err = b""
    # Outside the try below: a cwd that cannot be entered is the caller's error
    # to see, exactly as subprocess.run raises it rather than reporting a child
    # that failed.
    if cwd:
        os.chdir(cwd)
    try:
        sys.argv = [script] + args
        # The interpreter puts a script's own directory first on sys.path, and a
        # helper may import a module beside it. runpy does not, and the import
        # error would arrive as a non-zero return code that most callers ignore
        # -- leaving the example to read data files an earlier session wrote.
        sys.path = [os.path.dirname(os.path.abspath(script))] + sys.path
        with contextlib.ExitStack() as stack:
            stack.enter_context(_independent_rng())
            raw = stack.enter_context(_captured_fds()) if capture_output else None
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit as exc:
                rc = 0 if exc.code in (None, 0) else (exc.code if isinstance(exc.code, int) else 1)
            except Exception:  # noqa: BLE001 - a crash here is a non-zero exit, not ours to raise
                rc = 1
                traceback.print_exc(file=sys.stderr)
            if raw is not None:
                sys.stdout.flush()
                sys.stderr.flush()
                for handle in raw:
                    handle.seek(0)
                out, err = raw[0].read(), raw[1].read()
    finally:
        sys.argv = old_argv
        sys.path = old_path
        os.chdir(old_cwd)
        if os.environ != old_environ:
            os.environ.clear()
            os.environ.update(old_environ)
    done = subprocess.CompletedProcess([script] + args, rc, out, err)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, done.args, done.stdout, done.stderr)
    return done


class _ExampleDriver:
    """A long-lived driver process that runs examples on one open device.

    One case is sent per line and the reply is delimited by a sentinel, so the
    caller keeps full control of ordering -- a case's input is still generated
    immediately before that case runs, which matters because several cases share
    an example's data files with different shapes.

    Any failure demotes the driver permanently: ``available`` goes false and
    callers fall back to executing the standalone binary, so a driver problem
    costs speed rather than correctness.
    """

    SENTINEL = b"##CATLASS_CASE_DONE rc="
    # The driver cannot give that example the run it asked for -- the library
    # will not load, or the example wants a device other than the one the
    # session holds. Run it standalone instead of reporting a failure it never
    # actually observed.
    UNSUPPORTED = b"##CATLASS_CASE_UNSUPPORTED"

    def __init__(self, driver_path: str):
        self._path = driver_path
        self._proc = None
        self._timed_out = False
        self._answered = True
        # Counts cases, so a watchdog that has already begun running can tell
        # whether the case it was armed for is still the one in flight.
        self._generation = 0
        # Held across the watchdog's decision to kill and the reply's decision
        # that the case is over, which race each other by construction.
        self._lock = threading.Lock()
        self.available = os.path.isfile(driver_path) and os.access(driver_path, os.X_OK)

    def _start(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        try:
            # stderr joins stdout: the harness checks both for acl/compare
            # errors, and one stream keeps the sentinel ordered against the
            # output it terminates.
            self._proc = subprocess.Popen(
                [self._path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError:
            self.available = False
            return False
        return True

    def library_for(self, executable_name: str):
        path = os.path.join(EXAMPLE_LIBS_PATH, f"lib{executable_name}.so")
        return path if os.path.isfile(path) else None

    def run(self, library: str, args, timeout: float) -> subprocess.CompletedProcess:
        if not self._start():
            raise RuntimeError("example driver could not be started")
        request = "\t".join([library] + [str(a) for a in args]) + "\n"
        assert self._proc is not None
        self._proc.stdin.write(request.encode())
        self._proc.stdin.flush()
        # A watchdog, not select(): readline() on a buffered pipe pulls every
        # available byte into Python's buffer and hands back one line, so the
        # sentinel can already be in memory while select() on the descriptor
        # reports nothing new -- which hangs the case until its timeout. Killing
        # the driver is what ends a blocking read, exactly as subprocess does it.
        with self._lock:
            self._timed_out = False
            self._answered = False
            self._generation += 1
            generation = self._generation
        watchdog = threading.Timer(timeout, self._expire, args=(generation,))
        watchdog.daemon = True
        watchdog.start()
        try:
            return self._read_reply(library, args, timeout)
        finally:
            with self._lock:
                self._answered = True
            watchdog.cancel()

    def _expire(self, generation: int) -> None:
        # cancel() does nothing to a timer that has already started running, so
        # a case that replied just as its timeout came up would otherwise have a
        # healthy driver killed under it -- and the next case would pay for the
        # device again. The generation says which case armed this watchdog: one
        # left over from an earlier case must not kill the driver running this
        # one, and the boolean alone cannot tell them apart.
        with self._lock:
            if generation != self._generation or self._answered:
                return
            self._timed_out = True
            proc = self._proc
            # The signal goes out under the same lock that decided to send it.
            # Deciding and signalling have to be indivisible: between the two,
            # the case could reply and the next one start on this very process,
            # and the signal would then land on a case whose deadline has not
            # even been set. kill() only delivers a signal and never blocks, so
            # holding the lock across it costs nothing.
            if proc is not None:
                proc.kill()

    def _read_reply(self, library, args, timeout) -> subprocess.CompletedProcess:
        chunks = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                timed_out = self._timed_out
                self.shutdown(kill=True)
                if timed_out:
                    raise subprocess.TimeoutExpired([library] + list(args), timeout)
                raise RuntimeError("example driver exited mid-case")
            if line.startswith(self.UNSUPPORTED):
                raise LookupError(library)
            if line.startswith(self.SENTINEL):
                rc = int(line[len(self.SENTINEL):].strip() or 1)
                output = b"".join(chunks)
                # Both streams carry the merged output: ret_check reads stderr,
                # while a few cases read stdout.
                return subprocess.CompletedProcess(
                    [library] + [str(a) for a in args], rc, output, output
                )
            chunks.append(line)

    def shutdown(self, kill: bool = False) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if kill:
            self.available = False
            proc.kill()
        else:
            # Draining and reaping are one bounded operation. Releasing the
            # device is where the runtime reports anything it was still
            # holding, so a driver blocked on a full pipe would never reach the
            # end of its teardown and would be killed in the middle of it --
            # leaving the device to the kernel rather than to ACL. An unbounded
            # read is no better: a driver that stalls in aclrtResetDevice while
            # holding stdout open would hang the interpreter's exit, where
            # nothing is left to time it out.
            # communicate() closes stdin itself, which is what tells the
            # driver to finish; closing it here first makes communicate() raise
            # on the flush and skip the bound entirely.
            try:
                _ = proc.communicate(timeout=30)
                return
            except subprocess.TimeoutExpired:
                proc.kill()
            except (OSError, ValueError):
                pass
        # Reap whatever was killed, so the driver does not outlive us as a
        # zombie.
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


_DRIVER = None
# True once the driver and the example libraries have both been built and
# installed by this session. A case may use the driver only then: libraries are
# loaded on existence alone, so an artifact left by an earlier build would
# otherwise be run in place of one the current source never produced -- and the
# case would pass without the current source ever having been compiled.
_DRIVER_ARTIFACTS_OK = False


def _driver():
    """The session's driver, or None when merged running is off/unavailable."""
    global _DRIVER
    if not (MERGED_DRIVER and _DRIVER_ARTIFACTS_OK):
        return None
    if _DRIVER is None:
        _DRIVER = _ExampleDriver(DRIVER_PATH)
    return _DRIVER if _DRIVER.available else None


def build_case(build_env, executable_name: str):
    """Incrementally build and install the requested example target."""
    _ = _run(
        ["cmake", "--build", build_env, "--target", executable_name, "-j"],
        check=True,
    )
    _ = _run(
        ["cmake", "--install", build_env, "--component", executable_name],
        check=True,
    )


@atexit.register
def _close_driver() -> None:
    """Stop the driver process; it is a child and would otherwise outlive us."""
    if _DRIVER is not None:
        _DRIVER.shutdown()


def run_case(build_env, executable_name: str, args: list, *, cwd=None):
    """Run the example, on the session's open device when the driver has it."""
    args = [str(arg) for arg in args]
    driver = _driver() if cwd is None else None  # the driver has no per-case cwd
    if driver is not None:
        library = driver.library_for(executable_name)
        if library is not None:
            try:
                ret = driver.run(library, args, RUN_TIMEOUT)
            except LookupError:
                pass  # this example only; the driver stays in use for the rest
            except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError):
                pass  # driver demoted inside; fall through to the standalone binary
            else:
                ret_check(ret)
                return ret
    build_case(build_env, executable_name)
    ret = _run(
        [os.path.join(CMAKE_BINARY_PATH, executable_name)] + args,
        capture_output=True,
        check=False,
        cwd=cwd,
    )
    ret_check(ret)
    return ret


# ---------------------------------------------------------------------------
# handwritten cases (with gen_data or special args)
# ---------------------------------------------------------------------------
@only_on_2201
def test_19_mla(build_env):
    batch = 1
    q_seqlen_list = "1"
    kv_seqlen_list = "128"
    num_heads = 16
    num_blocks = 16
    block_size = 128
    dtype = "half"

    case_py = [
        str(batch),
        q_seqlen_list,
        kv_seqlen_list,
        str(num_heads),
        str(num_blocks),
        str(block_size),
        dtype,
    ]
    _ = _run(
        ["python", os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "gen_data.py")] + case_py,
        check=False,
    )
    # example args: batch, qSeqlenList, kvSeqlenList, numHeads, numBlocks, blockSize,
    #               --dtype half --datapath ... --device DEVICE_ID
    case_cpp = [
        str(batch),
        q_seqlen_list,
        kv_seqlen_list,
        str(num_heads),
        str(num_blocks),
        str(block_size),
        "--dtype",
        dtype,
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "data"),
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "19_mla", case_cpp)


@only_on_2201
def test_24_conv_bias(build_env):
    # gen_data args: batch, di, cin1, hi, wi, cout, ... conv sizes..., float16(dtype)
    case_base = [str(i) for i in [32, 64, 1, 32, 48, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
    case_py = case_base + ["float16"]
    _ = _run(
        ["python", os.path.join(CMAKE_EXAMPLES_PATH, "24_conv_bias", "gen_data.py")] + case_py,
        check=False,
    )
    # example args: conv size params..., device_id
    case_cpp = [str(i) for i in [32, 1, 4, 32, 48, 16, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    case_cpp[-1] = DEVICE_ID
    run_case(build_env, "24_conv_bias", case_cpp)


@only_on_2201
def test_29_a2_fp8_e4m3_matmul(build_env):
    # gen_data args: m, n, k, trans_a, trans_b
    case_py = [str(i) for i in [256, 512, 1024, 0, 0]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "29_a2_fp8_e4m3_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 512, 1024, 0]]
    case_cpp[-1] = DEVICE_ID
    run_case(build_env, "29_a2_fp8_e4m3_matmul", case_cpp)


@only_on_2201
def test_32_w4a8_matmul(build_env):
    # gen_data args: m, n, k, device_id
    case_py = [str(i) for i in [860, 5712, 4535, 0]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "32_w4a8_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [860, 5712, 4535, 0]]
    case_cpp[-1] = DEVICE_ID
    run_case(build_env, "32_w4a8_matmul", case_cpp, cwd=CMAKE_BINARY_PATH)


@only_on_2201
def test_38_w4a4_matmul(build_env):
    # gen_data args: m, n, k
    case_py = [str(i) for i in [96, 4096, 1280]]  # (M, N, K)
    _ = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "38_w4a4_matmul_per_token_per_channel_dequant",
                "gen_data.py",
            ),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = case_py + [DEVICE_ID]
    run_case(build_env, "38_w4a4_matmul_per_token_per_channel_dequant", case_cpp)


@only_on_2201
def test_41_sparse_matmul_tla(build_env):
    # gen_data args: m, n, k
    case_py = [str(i) for i in [160, 320, 64]]  # (M, N, K)
    # sparse_gen_data.py writes <example_dir>/input|output, while the example reads "./input|output",
    # so both steps must run with the example directory as CWD.
    case_dir = os.path.join(CMAKE_EXAMPLES_PATH, "41_sparse_matmul_tla")
    _ = _run(
        [
            "python",
            os.path.join(case_dir, "sparse_gen_data.py"),
        ]
        + case_py,
        check=False,
        cwd=case_dir,
    )
    # example args: m, n, k, device_id
    case_cpp = case_py + [DEVICE_ID]
    ret = run_case(build_env, "41_sparse_matmul_tla", case_cpp, cwd=case_dir)
    assert ret.returncode == 0


@only_on_2201
def test_23_flash_attention_infer(build_env):
    # README example: gen_data.py 1 177 512 1 1 128 0 1 bf16 1 (Paged KV cache, causal)
    case_py = [str(i) for i in [1, 177, 512, 1, 1, 128, 0, 1]] + ["bf16", "1"]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "23_flash_attention_infer", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, qSeqlen, kvSeqlen, numHeads, kvHeads, headSize, isVariedLen, maskType,
    #               --dtype bf16 --datapath ... --device DEVICE_ID
    case_cpp = [str(i) for i in [1, 177, 512, 1, 1, 128, 0, 1]] + [
        "--dtype",
        "bf16",
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "23_flash_attention_infer", "data"),
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "23_flash_attention_infer", case_cpp)


@only_on_2201
def test_40_flash_attention_infer_tla(build_env):
    # README example: gen_data.py 1 177 512 1 1 128 0 1 bf16 1 (Paged KV cache, causal)
    case_py = [str(i) for i in [1, 177, 512, 1, 1, 128, 0, 1]] + ["bf16", "1"]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "40_flash_attention_infer_tla", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, qSeqlen, kvSeqlen, numHeads, kvHeads, headSize, isVariedLen, maskType,
    #               --dtype bf16 --datapath ... --device DEVICE_ID
    case_cpp = [str(i) for i in [1, 177, 512, 1, 1, 128, 0, 1]] + [
        "--dtype",
        "bf16",
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "40_flash_attention_infer_tla", "data"),
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "40_flash_attention_infer_tla", case_cpp)


@only_on_3510
def test_49_ascend950_flash_attention_infer(build_env):
    case_py = [str(i) for i in [1, 138, 100, 4, 2, 128, 0, 0, 0]] + ["half"]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "49_ascend950_flash_attention_infer", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, qSeqlenList, kvSeqlenList, numHeads, numBlocks, blockSize,
    #               --dtype half --device DEVICE_ID --datapath ...
    case_cpp = [str(i) for i in [1, 138, 100, 4, 2, 128, 0, 0, 0]] + [
        "--dtype",
        "half",
        "--device",
        DEVICE_ID,
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "49_ascend950_flash_attention_infer", "data"),
    ]
    run_case(build_env, "49_ascend950_flash_attention_infer", case_cpp)


@only_on_3510
def test_53_ascend950_fp8_mx_matmul(build_env):
    # gen_data: trans_a=0, trans_b=1 -> RowMajor A + ColumnMajor B (matches example layout)
    case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 512, 1024]] + [DEVICE_ID]
    run_case(build_env, "53_ascend950_fp8_mx_matmul", case_cpp)


@only_on_3510
def test_54_ascend950_fp4_mx_matmul(build_env):
    # same as 53: trans_b=1 -> ColumnMajor B (matches fp4_mx_matmul.cpp layout)
    case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 512, 1024]] + [DEVICE_ID]
    run_case(build_env, "54_ascend950_fp4_mx_matmul", case_cpp)


@only_on_3510
def test_53_ascend950_fp8_mx_matmul_aswt(build_env):
    case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 512, 1024]] + [DEVICE_ID]
    run_case(build_env, "53_ascend950_fp8_mx_matmul_aswt", case_cpp)


@only_on_3510
def test_54_ascend950_fp4_mx_matmul_aswt(build_env):
    case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 512, 1024]] + [DEVICE_ID]
    run_case(build_env, "54_ascend950_fp4_mx_matmul_aswt", case_cpp)


@only_on_3510
def test_53_ascend950_fp8_mx_matmul_small_shape(build_env):
    # smaller shape than default, still trans_a=0, trans_b=1 (RowMajor A + ColumnMajor B)
    case_py = [str(i) for i in [128, 256, 512, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [128, 256, 512]] + [DEVICE_ID]
    run_case(build_env, "53_ascend950_fp8_mx_matmul", case_cpp)


@only_on_3510
def test_54_ascend950_fp4_mx_matmul_small_shape(build_env):
    case_py = [str(i) for i in [128, 256, 512, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [128, 256, 512]] + [DEVICE_ID]
    run_case(build_env, "54_ascend950_fp4_mx_matmul", case_cpp)


@only_on_3510
def test_53_ascend950_fp8_mx_matmul_cube_1024(build_env):
    # big cube shape, covers a different tile split than 256x512x1024
    case_py = [str(i) for i in [1024, 1024, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [1024, 1024, 1024]] + [DEVICE_ID]
    run_case(build_env, "53_ascend950_fp8_mx_matmul", case_cpp)


@only_on_3510
def test_54_ascend950_fp4_mx_matmul_cube_1024(build_env):
    case_py = [str(i) for i in [1024, 1024, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [1024, 1024, 1024]] + [DEVICE_ID]
    run_case(build_env, "54_ascend950_fp4_mx_matmul", case_cpp)


@only_on_3510
def test_53_ascend950_fp8_mx_matmul_aswt_shape_512_1024_256(build_env):
    # different tile split than default, covers ASWT scheduling (L1 M/N=256)
    case_py = [str(i) for i in [512, 1024, 256, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [512, 1024, 256]] + [DEVICE_ID]
    run_case(build_env, "53_ascend950_fp8_mx_matmul_aswt", case_cpp)


@only_on_3510
def test_54_ascend950_fp4_mx_matmul_aswt_shape_512_1024_256(build_env):
    case_py = [str(i) for i in [512, 1024, 256, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [512, 1024, 256]] + [DEVICE_ID]
    run_case(build_env, "54_ascend950_fp4_mx_matmul_aswt", case_cpp)


@only_on_3510
def test_58_ascend950_fp8_mx_batch_matmul(build_env):
    # gen_data args: batch, m, n, k, trans_a, trans_b (trans_b=1 -> ColumnMajor B)
    # k <= 512: bfloat16 output stays within the 1/256 compare tolerance
    case_py = [str(i) for i in [2, 512, 1024, 512, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "58_ascend950_mx_batch_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, m, n, k, device_id
    case_cpp = [str(i) for i in [2, 512, 1024, 512]] + [DEVICE_ID]
    run_case(build_env, "58_ascend950_fp8_mx_batch_matmul", case_cpp)


@only_on_3510
def test_58_ascend950_fp8_mx_batch_matmul_small_shape(build_env):
    case_py = [str(i) for i in [1, 256, 256, 512, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "58_ascend950_mx_batch_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, m, n, k, device_id
    case_cpp = [str(i) for i in [1, 256, 256, 512]] + [DEVICE_ID]
    run_case(build_env, "58_ascend950_fp8_mx_batch_matmul", case_cpp)


@only_on_3510
def test_58_ascend950_fp4_mx_batch_matmul(build_env):
    # gen_data_fp4 writes to data_fp4/ so it never clashes with the fp8 data dir
    case_py = [str(i) for i in [5, 256, 512, 1024, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "58_ascend950_mx_batch_matmul", "gen_data_fp4.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, m, n, k, device_id
    case_cpp = [str(i) for i in [5, 256, 512, 1024]] + [DEVICE_ID]
    run_case(build_env, "58_ascend950_fp4_mx_batch_matmul", case_cpp)


@only_on_3510
def test_58_ascend950_fp4_mx_batch_matmul_large_k(build_env):
    # fp4 outputs float, so a large-k shape (2048) stays accurate
    case_py = [str(i) for i in [4, 128, 256, 2048, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "58_ascend950_mx_batch_matmul", "gen_data_fp4.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, m, n, k, device_id
    case_cpp = [str(i) for i in [4, 128, 256, 2048]] + [DEVICE_ID]
    run_case(build_env, "58_ascend950_fp4_mx_batch_matmul", case_cpp)


@only_on_3510
def test_55_ascend950_mx_grouped_matmul_slice_m(build_env):
    build_case(build_env, "55_ascend950_mx_grouped_matmul_slice_m")
    # gen_data_compare args: G, M, N, K, trans_b, quant_type, device_id
    for trans, quant_type in itertools.product(("0", "1"), ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2")):
        case_py = [str(i) for i in [2, 588, 988, 1030]] + [
            trans,
            quant_type,
            DEVICE_ID,
        ]
        ret = _run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH,
                    "55_ascend950_mx_grouped_matmul_slice_m",
                    "gen_data_compare.py",
                ),
            ]
            + case_py,
            capture_output=True,
            check=False,
        )
        ret_check(ret)


@only_on_3510
def test_55_ascend950_mx_grouped_matmul_slice_m_aswt_fp4_odd_n_transb(build_env):
    build_case(build_env, "55_ascend950_mx_grouped_matmul_slice_m_aswt")
    # gen_data_compare args: G, M, N, K, trans_b, quant_type, device_id, --bin ...
    case_py = [str(i) for i in [2, 588, 989, 1030, 1]] + [
        "float4_e2m1fn_x2",
        DEVICE_ID,
        "--bin",
        "55_ascend950_mx_grouped_matmul_slice_m_aswt",
    ]
    ret = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "55_ascend950_mx_grouped_matmul_slice_m",
                "gen_data_compare.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    ret_check(ret)
    assert "Compare success" in ret.stdout.decode()


@only_on_3510
def test_57_ascend950_matmul_full_dequant(build_env):
    case_py = (
        ["--shape"] + ["513 513 513"] + ["--x1_quant_mode"] + ["per_token"] + ["--x2_quant_mode"] + ["per_channel"]
    )
    _ = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "57_ascend950_matmul_full_dequant",
                "scripts",
                "gen_data.py",
            ),
        ]
        + case_py,
        check=False,
    )
    _ = _run(
        [
            "cp",
            "-r",
            "input",
            CMAKE_BINARY_PATH,
        ],
        check=False,
    )
    _ = _run(
        [
            "cp",
            "-r",
            "output",
            CMAKE_BINARY_PATH,
        ],
        check=False,
    )
    case_cpp = [str(i) for i in [513, 513, 513]] + ["per_token"] + ["per_channel"]
    run_case(build_env, "57_ascend950_matmul_full_dequant", case_cpp)


@only_on_3510
def test_59_ascend950_a8w4_mx_matmul(build_env):
    case_py = [str(i) for i in [256, 128, 128, 0, 1]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "59_ascend950_a8w4_mx_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, device_id
    case_cpp = [str(i) for i in [256, 128, 128]] + [DEVICE_ID]
    run_case(build_env, "59_ascend950_a8w4_mx_matmul", case_cpp)


@only_on_3510
def test_61_ascend950_svd_quant_matmul(build_env):
    # README example: gen_data.py 256 256 512 32, then <exe> 256 256 512 32 0
    case_py = [str(i) for i in [256, 256, 512, 32]]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "61_ascend950_svd_quant_matmul", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: m, n, k, r, device_id; it reads "./examples/61_.../data", so run from repo root
    case_cpp = case_py + [DEVICE_ID]
    run_case(build_env, "61_ascend950_svd_quant_matmul", case_cpp, cwd=CMAKE_SOURCE_DIR)


@only_on_3510
def test_62_ascend950_broadcast_matmul_perblock_quant(build_env):
    build_case(build_env, "62_ascend950_broadcast_matmul_perblock_quant")
    # gen_data_compare args: batch_count, m, n, k, device_id
    case_py = [str(i) for i in [5920, 128, 128, 128, 0]]
    case_py[-1] = DEVICE_ID
    ret = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "62_ascend950_broadcast_matmul_perblock_quant",
                "gen_data_compare.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    ret_check(ret)


@only_on_3510
def test_63_ascend950_dual_level_quant_mx_batch_matmul(build_env):
    # gen_data args: batch_count, m, n, k
    case_py = [str(i) for i in [1, 128, 128, 128]]  # batch_count, m, n, k
    _ = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "63_ascend950_dual_level_quant_mx_batch_matmul",
                "gen_data.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    # example args: batch_count, m, n, k, device_id
    case_cpp = [str(i) for i in [1, 128, 128, 128, 0]]
    case_cpp[-1] = DEVICE_ID
    run_case(build_env, "63_ascend950_dual_level_quant_mx_batch_matmul", case_cpp)


@only_on_3510
def test_65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant(build_env):
    # gen_data args: batch_count, m, n, k
    case_py = [str(i) for i in [2, 128, 128, 128]]  # batch_count, m, n, k
    _ = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant",
                "gen_data.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    # example args: batch_count, m, n, k, device_id
    case_cpp = [str(i) for i in [2, 128, 128, 128, 0]]
    case_cpp[-1] = DEVICE_ID
    run_case(
        build_env,
        "65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant",
        case_cpp,
    )


@only_on_3510
def test_70_ascend950_flash_attention_chunk_prefill(build_env):
    case_py = [str(i) for i in [1, 100, 138, 8, 1, 128, 128, 0]] + ["half"] + [str(i) for i in [2, 0, 128]] + ["nd"]
    _ = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "70_ascend950_flash_attention_chunk_prefill",
                "gen_data.py",
            ),
        ]
        + case_py,
        check=False,
    )
    # example args: batch, qSeqlenList, kvSeqlenList, headNum, blockSize...,
    #               --dtype half --cache_layout nd --device DEVICE_ID --datapath ...
    case_cpp = [str(i) for i in [1, 100, 138, 8, 1, 128, 128, 0, 2, 128]] + [
        "--dtype",
        "half",
        "--cache_layout",
        "nd",
        "--device",
        DEVICE_ID,
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "70_ascend950_flash_attention_chunk_prefill", "data"),
    ]
    run_case(build_env, "70_ascend950_flash_attention_chunk_prefill", case_cpp)


@only_on_3510
def test_71_ascend950_fp8_mx_grouped_matmul_finalize_routing(build_env):
    build_case(build_env, "71_ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter")
    # gen_data_compare args: batch_count, m, n, k, ..., quant_type, device_id
    case_py = [str(i) for i in [4, 128, 128, 128, 0, 0, 0, 16, 2, 0, 0.0, 0]] + [
        "float8_e5m2",
        DEVICE_ID,
    ]
    ret = _run(
        [
            "python",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "71_ascend950_fp8_mx_grouped_matmul_finalize_routing",
                "gen_data_compare.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    ret_check(ret)


@only_on_3510
def test_74_ascend950_weight_quant_a8w4_grouped_mx_matmul(build_env):
    # gen_data args:
    # (Explicit) group_list group_m_list m n k isNz
    # (average mode) expect_m_per_group group_num expect_m_per_group m n k isNz
    case_groups = (
        (["expect_m_per_group", "48", "64", "3072", "2048", "4096", "1"], ["48", "3072", "2048", "4096"]),
        (["group_list", "33,64,128", "256", "2048", "4096", "1"], ["3", "256", "2048", "4096"]),
    )
    for case_py, case_cpp in case_groups:
        ret = _run(
            [
                "python3",
                os.path.join(
                    CMAKE_EXAMPLES_PATH,
                    "74_ascend950_weight_quant_a8w4_grouped_mx_matmul",
                    "gen_data.py",
                ),
            ]
            + case_py,
            capture_output=True,
            check=False,
        )

        ret_check(ret)

        # example args: group_num, m, n, k, device_id
        run_case(build_env, "74_ascend950_weight_quant_a8w4_grouped_mx_matmul", case_cpp + [DEVICE_ID])


@only_on_2201
def test_78_x_attention(build_env):
    case_base = [str(i) for i in [1, 4, 512, 8, 2, 128, 32, 16, 0]]
    dtype = "half"
    _ = _run(
        [sys.executable, os.path.join(CMAKE_EXAMPLES_PATH, "78_x_attention", "gen_data.py")]
        + case_base
        + [dtype],
        check=True,
    )
    case_cpp = case_base + [
        "--dtype",
        dtype,
        "--datapath",
        os.path.join(CMAKE_EXAMPLES_PATH, "78_x_attention", "data"),
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "78_x_attention", case_cpp)


@only_on_3510
def test_80_ascned950_grouped_matmul_slice_m_gelu(build_env):
    case_py = [
        "4",  # group_num
        "2048",  # m
        "256",  # n
        "256",  # k
        DEVICE_ID,  # device_id
    ]
    ret = _run(
        [
            "python3",
            os.path.join(
                CMAKE_EXAMPLES_PATH,
                "80_ascend950_grouped_matmul_slice_m_gelu",
                "gen_data.py",
            ),
        ]
        + case_py,
        capture_output=True,
        check=False,
    )
    ret_check(ret)

    # example args: group_num, m, n, k, device_id
    case_cpp = ["4", "2048", "256", "256", DEVICE_ID]
    run_case(build_env, "80_ascend950_grouped_matmul_slice_m_gelu", case_cpp)


@only_on_3510
def test_83_ascend950_hstu_infer(build_env):
    # gen_data args: batch, qSeqlen, kvSeqlen, numHeads, kvHeads, headSize, isVariedLen,
    #                siluScale, layout, dtype, maskType, pagedBlockSize, dataPath
    data_path = os.path.join(CMAKE_EXAMPLES_PATH, "83_ascend950_hstu_infer", "data")
    case_py = [str(i) for i in [2, 256, 256, 8, 8, 256, 0, 0.1]] + [
        "TND",
        "half",
        "0",
        "0",
        data_path,
    ]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "83_ascend950_hstu_infer", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale
    #               --dtype half --layout TND --mask 0 --paged_block_size 0
    #               --datapath ... --device ...
    case_cpp = [str(i) for i in [2, 256, 256, 8, 8, 256, 0, 0.1]] + [
        "--dtype",
        "half",
        "--layout",
        "TND",
        "--mask",
        "0",
        "--paged_block_size",
        "0",
        "--datapath",
        data_path,
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "83_ascend950_hstu_infer", case_cpp)


@only_on_3510
def test_83_ascend950_hstu_infer_paged(build_env):
    # paged KV cache: causal mask, block size 128
    data_path = os.path.join(CMAKE_EXAMPLES_PATH, "83_ascend950_hstu_infer", "data")
    case_py = [str(i) for i in [2, 256, 256, 8, 8, 256, 0, 0.1]] + [
        "TND",
        "half",
        "1",
        "128",
        data_path,
    ]
    _ = _run(
        [
            "python",
            os.path.join(CMAKE_EXAMPLES_PATH, "83_ascend950_hstu_infer", "gen_data.py"),
        ]
        + case_py,
        check=False,
    )
    # example args: batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale
    #               --dtype half --layout TND --mask 1 --paged_block_size 128
    #               --datapath ... --device ...
    case_cpp = [str(i) for i in [2, 256, 256, 8, 8, 256, 0, 0.1]] + [
        "--dtype",
        "half",
        "--layout",
        "TND",
        "--mask",
        "1",
        "--paged_block_size",
        "128",
        "--datapath",
        data_path,
        "--device",
        DEVICE_ID,
    ]
    run_case(build_env, "83_ascend950_hstu_infer", case_cpp)


@only_on_3510
def test_56_ascend950_conv2d_bias_silu(build_env):
    # Default shape from example Options: stride=2, 3x3 kernel, pad=2
    case_cpp = [str(i) for i in [2, 33, 43, 112, 80, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1]]
    case_cpp.append(DEVICE_ID)
    run_case(build_env, "56_ascend950_conv2d_bias_silu", case_cpp)


@only_on_2201
def test_33_conv2d_bias_silu(build_env):
    # Default shape from example Options: stride=1, 3x3 kernel, pad=2 (stride=2 with a
    # 3x3 kernel exceeds the AtlasA2 512K L1 and is rejected by CanImplement).
    case_cpp = [str(i) for i in [2, 33, 43, 112, 80, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]]
    case_cpp.append(DEVICE_ID)
    run_case(build_env, "33_conv2d_bias_silu", case_cpp)


# ---------------------------------------------------------------------------
# normal cases: generated via parametrize
# ---------------------------------------------------------------------------
normal_cases_2201 = [
    "00_basic_matmul 256 512 1024 0",
    "01_batched_matmul 5 256 512 1024 0",
    "02_grouped_matmul_slice_m 128 512 1024 2048 0",
    "03_matmul_add 256 512 1024 0",
    "04_padding_matmul 256 512 1024 0",
    "05_grouped_matmul_slice_k 128 512 1024 32 0",
    "06_optimized_matmul 256 512 1024 0",
    "07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0",
    "08_grouped_matmul 128 512 1024 2048 0",
    "09_splitk_matmul 256 512 1024 0",
    "10_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
    "11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0",
    "12_quant_matmul 256 512 1024 0",
    "13_basic_matmul_tla 256 512 1024 0",
    "14_optimized_matmul_tla 256 512 1024 0",
    "15_gemm 256 512 1024 0",
    "16_group_gemm 3 '128,256,512' '256,512,128' '512,256,128' 0",
    "17_gemv_aiv 256 512 0",
    "18_gemv_aic 256 512 0",
    "20_matmul_bias 256 512 1024 0",
    "21_basic_matmul_preload_zN 256 512 1024 0",
    "22_padding_splitk_matmul 256 512 1024 0",
    "25_matmul_full_loadA 256 512 1024 0",
    "26_matmul_relu 256 512 1024 0",
    "27_matmul_gelu 256 512 1024 0",
    "28_matmul_silu 256 512 1024 0",
    "30_w8a16_matmul 256 512 1024 0",
    "31_small_matmul 256 1024 256 0",
    "33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "33_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "37_streamk_matmul 256 512 1024 0",
    "34_single_core_splitk_matmul 256 512 1024 0",
    "42_quant_optimized_matmul_tla 256 512 1024 0",
    "44_quant_matmul_full_loadA_tla 256 512 1024 0",
    "45_strided_batched_matmul_tla 5 256 512 1024 0",
    "52_quant_multi_core_splitk_matmul_tla 256 512 1024 0",
    "75_symm 256 512 256 0 0 0",
    "76_trmm 512 256 0 0 0 0 1.0 0",
    "77_planar_complex_matmul 256 512 1024 0",
    "102_dynamic_optimized_matmul 256 512 1024 0 0 0",
    "103_dynamic_optimized_quant_matmul_per_token_basic 256 512 1024 0 0 0",
]

normal_cases_3510 = [
    "43_ascend950_basic_matmul 256 512 1024 0",
    "46_ascend950_matmul_fixpipe_opti 256 512 1024 0",
    "47_ascend950_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
    "48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant 128 512 1024 2048 0 0",
    "50_ascend950_basic_matmul_gemv 1 128 127 0",
    "51_ascend950_quant_matmul_per_group_per_block_tla 256 512 1024 0",
    "56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "56_ascend950_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "60_ascend950_grouped_matmul_slice_m 128 512 1024 2048 0",
    "64_ascend950_matmul_evg_add 256 512 1024 0",
    "64_ascend950_matmul_evg_leaky_relu 256 512 1024 0",
    "64_ascend950_matmul_evg_sigmoid 256 512 1024 0",
    "64_ascend950_matmul_evg_silu 256 512 1024 0",
    "64_ascend950_matmul_evg_tanh 256 512 1024 0",
    "64_ascend950_matmul_evg_bias 256 512 1024 0",
    "64_ascend950_matmul_evg_add_ub 256 512 1024 0",
    "66_ascend950_streamk_matmul 256 512 1024 0",
    "67_ascend950_batched_matmul 5 256 512 1024 0",
    "68_ascend950_multi_core_splitk_matmul 256 512 2048 0",
    "69_ascend950_tail_multi_core_splitk_matmul 1280 2048 2048 0",
    "73_ascend950_matmul_full_loadA 256 512 256 0",
    "82_ascend950_basic_syrk 1024 1024 0",
    "84_ascend950_syrk 1024 1024 1 1.0 1.0 0",
    "88_ascend950_bmm_broadcast 3 3 1 67 131 145 0",
]


@only_on_2201
@pytest.mark.parametrize("case", normal_cases_2201, ids=lambda c: c.split()[0])
def test_normal_2201(build_env, case):
    parts = case.split()
    parts[-1] = DEVICE_ID  # deviceid arg from env DEVICE_ID (default 0)
    run_case(build_env, parts[0], parts[1:])


@only_on_3510
def test_89_ascend950_hif8_quant_batch_matmul(build_env):
    # gen_data args: batch M K N bias x1quantmode x2quantmode
    # CLI: batch M K N bias x1quantmode x2quantmode [device_id]
    cases = [
        "1 128 256 128 0 1 1",
        "1 128 256 128 128 0 2",
    ]
    for case in cases:
        batch, m, k, n, bias, x1, x2 = case.split()
        _ = _run(
            ["python", os.path.join(CMAKE_EXAMPLES_PATH, "89_ascend950_hif8_quant_batch_matmul", "gen_data", "gen_data.py"),
             batch, m, k, n, bias, x1, x2],
            check=True,
        )
        ret = run_case(build_env, "89_ascend950_hif8_quant_batch_matmul",
                       [batch, m, k, n, bias, x1, x2, DEVICE_ID])
        assert "Compare success" in ret.stdout.decode()


@only_on_3510
@pytest.mark.parametrize("case", normal_cases_3510, ids=lambda c: c.split()[0])
def test_normal_3510(build_env, case):
    parts = case.split()
    parts[-1] = DEVICE_ID  # deviceid arg from env DEVICE_ID (default 0)
    run_case(build_env, parts[0], parts[1:])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
