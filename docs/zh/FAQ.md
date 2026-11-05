# 常见问题

## 如何指定CATLASS的目标硬件架构？

我们于2026年3月第一次社区会议正式确定CATLASS社区主线将开始新增对下一代昇腾硬件Ascend 950PR&950DT系列产品的支持，对应的正式版本从1.5.0开始。为在不同平台区分底层接口的实现，该新增支持将引入新的编译宏，用户需要注意在对应编译命令中进行相应适配。

- 新增宏：`CATLASS_ARCH`，用于指定目标架构。其取值可在[SIMD BuiltIn关键字](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/language_extension/simd_builtin_keywords.md)中查询（`__NPU_ARCH__`列）。
  - `Atlas A2系列产品、Atlas A3系列产品`：`2201`
  - `Ascend 950PR&950DT系列产品`：`3510`

- 相关场景说明：
  - `bisheng`命令行场景：`bisheng ... -DCATLASS_ARCH=2201 ...`
  - `cmake`场景：`add_compile_definitions(CATLASS_ARCH=2201)`
  - `msopgen/aclnn`工程场景：
    - 旧写法（CANN < 9.0.0）：`add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 ...)`
    - 新写法（CANN >= 9.0.0）：`npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCATLASS_ARCH=2201)`（msopgen工程中，第一个参数默认为`ascendc_kernels`，可根据实际情况进行调整）
  - CATLASS源码仓：`bash scripts/build.sh -DCATLASS_ARCH=2201 ...`
  - 库上代码参考：[examples/CMakeLists.txt](https://gitcode.com/cann/catlass/blob/master/examples/CMakeLists.txt)

## 编译报错找不到C++标准头文件（`'cstdint' file not found`）或链接报错找不到标准库（`unable to find library -lstdc++`），如何处理？

**现象**

在部分发行版（如 Debian 13）上，按[软硬件配套说明](../../README.md#-软硬件配套说明)安装了满足版本约束的gcc后，首次执行快速入门的编译命令仍会失败：

- 编译期：`fatal error: 'cstdint' file not found`，报错位置通常在`include/catlass/catlass.hpp`的`#include <cstdint>`。
- 链接期：`ld.lld: error: unable to find library -lstdc++`，并伴随`ccecld: Linker ReturnCode: 1`。

**原因**

CANN的算子编译器（`bisheng`/`ccec`）在编译链路中仍依赖宿主机系统gcc提供的C++标准头文件和标准库。若系统未将gcc的C++头文件目录（`/usr/include/c++/<gcc版本>`、`/usr/include/<triple>/c++/<gcc版本>`）和库目录（`/usr/lib/gcc/<triple>/<gcc版本>`）加入搜索路径，或该gcc不是系统默认版本、安装在非标准路径下，编译器就无法找到这些文件。报错只停留在"找不到头文件/找不到库"的表层信息上，不会提示搜索路径未注入。

**处理**

按实际gcc版本与目标三元组导出环境变量后再编译，以下以aarch64 + gcc 12为例：

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/12:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/12:$LD_LIBRARY_PATH
```

- `<gcc版本>`通过`gcc -dumpversion`查询，`<triple>`通过`gcc -dumpmachine`查询（x86_64环境为`x86_64-linux-gnu`），请按实际取值替换示例中的`12`和`aarch64-linux-gnu`。
- 若gcc安装在非标准前缀（如`/opt/gcc-12`），需将上述路径中的`/usr`替换为对应前缀，并确保编译时调用的`gcc`就是该版本。
- 建议将上述变量写入`~/.bashrc`或CANN的`set_env.sh`之后，避免每个新终端都要重复导出。

**验证**

```bash
# 查看gcc版本与目标三元组
gcc -dumpversion
gcc -dumpmachine

# 查看C++头文件搜索路径，确认其中包含/usr/include/c++/<gcc版本>
echo | gcc -x c++ -E -v - 2>&1 | sed -n '/#include <...> search starts here:/,/End of search list./p'
```

确认后重新执行`bash scripts/build.sh 00_basic_matmul`，输出`[INFO] Target '00_basic_matmul' built successfully`即表示环境已就绪。

针对这类环境问题，`cmake/funcs.cmake` 中已集成自动检测：当编译或链接找不到 C++ 标准库时，会尝试按本机 gcc 的隐式搜索路径注入 C++ 头文件与库搜索路径。该逻辑依赖本机 gcc 的搜索路径推导，在特殊发行版或非标准安装前缀下可能失效；若仍出现该问题，请按上文步骤手动导出环境变量，并参考[环境问题反馈](#遇到其他编译链接类环境问题如何反馈)提供信息。

## 遇到其他编译/链接类环境问题，如何反馈？

环境类问题与发行版、gcc版本、CANN版本强相关，反馈issue时请附带以下信息，以便在其他环境中复现：

- 系统信息：`uname -m`、`cat /etc/os-release`
- 依赖版本：`gcc --version`、`gcc -dumpversion`、`gcc -dumpmachine`、`cmake --version`、`python3 --version`
- CANN版本：`cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info`（`${arch}`由`uname -m`查询）
- 完整编译命令，以及报错信息的首行与结尾（保留文件路径与行号）
- 已尝试过的处理方式（如是否导出过`CPLUS_INCLUDE_PATH`、`LIBRARY_PATH`）

后续新出现且能复现的环境类问题，将持续补充到本节。
