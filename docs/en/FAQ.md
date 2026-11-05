# FAQ

## How do I specify the target hardware architecture for CATLASS?

At the first community meeting in March 2026, we officially confirmed that the CATLASS community mainline would add support for the next-generation Ascend hardware, the Ascend 950PR&950DT series products. Official support starts with CATLASS 1.5.0. To distinguish the underlying interface implementations across platforms, this support introduces a compilation macro that must be set in the corresponding build commands.

- New macro: `CATLASS_ARCH`, which specifies the target architecture. Its value is listed in the `__NPU_ARCH__` column of [SIMD BuiltIn Keywords](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/en/guide/programming_guide/language_extension/SIMD-BuiltIn_keyword.md).
  - `Atlas A2 and Atlas A3 series products`: `2201`
  - `Ascend 950PR&950DT series products`: `3510`

- Usage by scenario:
  - `bisheng` command line: `bisheng ... -DCATLASS_ARCH=2201 ...`
  - `cmake`: `add_compile_definitions(CATLASS_ARCH=2201)`
  - `msopgen/aclnn` projects:
    - Old syntax (CANN < 9.0.0): `add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 ...)`
    - New syntax (CANN >= 9.0.0): `npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCATLASS_ARCH=2201)` (in an msopgen project, the first parameter defaults to `ascendc_kernels` and can be adjusted as needed)
  - CATLASS source repository: `bash scripts/build.sh -DCATLASS_ARCH=2201 ...`
  - Code reference: [examples/CMakeLists.txt](https://gitcode.com/cann/catlass/blob/master/examples/CMakeLists.txt)

## A build fails with a missing C++ standard header (`'cstdint' file not found`) or a link error about a missing standard library (`unable to find library -lstdc++`). What should I do?

**Symptoms**

On some distributions (for example, Debian 13), after installing a gcc version that satisfies the constraints in [Software and Hardware Requirements](../../README_en.md#-software-and-hardware-requirements), the first build command in the quick start still fails:

- Compile stage: `fatal error: 'cstdint' file not found`, usually at `#include <cstdint>` in `include/catlass/catlass.hpp`.
- Link stage: `ld.lld: error: unable to find library -lstdc++`, together with `ccecld: Linker ReturnCode: 1`.

**Cause**

The CANN operator compiler (`bisheng`/`ccec`) still relies on the C++ standard headers and standard libraries provided by the host system gcc. If the system has not exposed the gcc C++ header directories (`/usr/include/c++/<gcc-version>`, `/usr/include/<triple>/c++/<gcc-version>`) and library directory (`/usr/lib/gcc/<triple>/<gcc-version>`) in the search path, or if that gcc is not the system default and is installed under a non-standard prefix, the compiler cannot find these files. The error only reports a missing header or library and never mentions that the search paths were not injected.

**Workaround**

Export the environment variables based on the actual gcc version and target triple, then rebuild. The following example uses aarch64 with gcc 12:

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/12:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/12:$LD_LIBRARY_PATH
```

- Query `<gcc-version>` with `gcc -dumpversion` and `<triple>` with `gcc -dumpmachine` (for x86_64 it is `x86_64-linux-gnu`). Replace `12` and `aarch64-linux-gnu` in the example with your actual values.
- If gcc is installed under a non-standard prefix (such as `/opt/gcc-12`), replace `/usr` in the paths above with that prefix, and make sure the `gcc` used during the build is that version.
- Consider adding these variables to `~/.bashrc` or after CANN's `set_env.sh` so that you do not need to export them in every new terminal.

**Verification**

```bash
# Show the gcc version and target triple.
gcc -dumpversion
gcc -dumpmachine

# Show the C++ header search path and confirm it contains /usr/include/c++/<gcc-version>.
echo | gcc -x c++ -E -v - 2>&1 | sed -n '/#include <...> search starts here:/,/End of search list./p'
```

Then run `bash scripts/build.sh 00_basic_matmul` again. The output `[INFO] Target '00_basic_matmul' built successfully` indicates that the environment is ready.

For such environment issues, `cmake/funcs.cmake` includes automatic detection: when the C++ standard library cannot be found during compilation or linking, it attempts to inject the C++ header and library search paths derived from the local gcc implicit search paths. This logic depends on the local gcc search paths and may fail on unusual distributions or non-standard install prefixes. If the issue persists, export the environment variables manually as described above, and include the details requested in [how to report environment issues](#how-should-i-report-other-build-or-link-environment-issues).

## How should I report other build or link environment issues?

Environment issues depend heavily on the distribution, gcc version, and CANN version. When filing an issue, include the following so that it can be reproduced elsewhere:

- System information: `uname -m`, `cat /etc/os-release`
- Dependency versions: `gcc --version`, `gcc -dumpversion`, `gcc -dumpmachine`, `cmake --version`, `python3 --version`
- CANN version: `cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info` (`${arch}` comes from `uname -m`)
- The complete build command, plus the first line and the end of the error output (keep file paths and line numbers)
- What you have already tried (for example, whether `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH` were exported)

New reproducible environment issues will keep being added to this section.
