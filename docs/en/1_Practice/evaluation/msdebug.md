# Using msDebug in a CATLASS Sample Project

[msDebug](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0062.html) is a tool for debugging operator programs running on NPUs. This tool provides operator developers with a mechanism for debugging operators on Ascend devices. Debugging methods include reading device memory and registers, as well as pausing and resuming program execution.

- ⚠️ **Note**: If you are developing and debugging using containers, ensure that `/dev/drv_debug` is mapped into containers (refer to the [driver check guide](https://www.hiascend.com/document/caselibrary/detail/atlasopdev_0006)).

## Examples

The following uses `00_basic_matmul` as an example to describe how to use msDebug.

### Enabling the Driver's Debugging Function

Refer to the [msDebug overview](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0062.html) to install the driver in `debug` mode, or execute `echo 1 > /proc/debug_switch` to enable the debug channel on a driver installed in `full` mode.

To avoid security issues, do not enable the debug channel in production environments.

- If the following issue occurs, it indicates that the driver version is too low and needs to be updated.

```bash
msdebug failed to initialize. please install HDK.
[ERROR] error code: 0x20102
terminate called after throwing an instance of 'MSDEBUG_ERROR_CODE'
```

### Compilation and Execution

1. Following [Quick Start](../01_quick_start.md), enable the compilation switch `--debug --msdebug` of the tool to enable the `debug` and `msdebug` compilation of the operator sample.

    ```bash
    bash scripts/build.sh --debug --msdebug 00_basic_matmul
    ```

    - `--debug` controls the debug switch for both host and device code, while `--msdebug` controls the debug switch for device code only.
    - If only `--debug` is added, only host debugging is enabled, and only host code can be debugged with gdb/lldb.

2. Switch to the `output/bin` directory where the executable file is compiled, and run the operator sample program using `msdebug`.

    ```bash
    cd output/bin
    # Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional)
    msdebug ./00_basic_matmul 256 512 1024 0
    ```

    ```bash
    msdebug ./00_basic_matmul 256 512 1024 0
    msdebug(MindStudio Debugger) is part of MindStudio Operator-dev Tools.
    The tool provides developers with a mechanism for debugging Ascend kernels running on actual hardware.
    This enables developers to debug Ascend kernels without being affected by potential changes brought by simulation and emulation environments.
    (msdebug) target create "./00_basic_matmul"
    Current executable set to '/home/catlass/output/bin/00_basic_matmul' (aarch64).
    (msdebug) settings set -- target.run-args  "256" "512" "1024" "0"
    (msdebug)
    ```

### Command Line Debugging

#### Setting Breakpoints and Running the Program

Set two breakpoints using the commands `b basic_matmul.cpp:45` and `b basic_matmul.cpp:90` (lines 90-101 in [`00_basic_matmul.cpp`](../../../../examples/00_basic_matmul/basic_matmul.cpp) are type alias definitions, not runtime machine code). Then use `breakpoint list` to view existing breakpoints.

```bash
(msdebug) b basic_matmul.cpp:45
Breakpoint 1: where = 00_basic_matmul`Run(GemmOptions const&) + 460 at basic_matmul.cpp:45:18, address = 0x000000000016df8c
(msdebug) b basic_matmul.cpp:90
Breakpoint 2: where = 00_basic_matmul`Run(GemmOptions const&) + 2816 at basic_matmul.cpp:101:39, address = 0x000000000016e8c0
(msdebug) breakpoint list
Current breakpoints:
1: file = 'basic_matmul.cpp', line = 45, exact_match = 0, locations = 1
  1.1: where = 00_basic_matmul`Run(GemmOptions const&) + 460 at basic_matmul.cpp:45:18, address = 00_basic_matmul[0x000000000016df8c], unresolved, hit count = 0

2: file = 'basic_matmul.cpp', line = 90, exact_match = 0, locations = 1
  2.1: where = 00_basic_matmul`Run(GemmOptions const&) + 2816 at basic_matmul.cpp:101:39, address = 00_basic_matmul[0x000000000016e8c0], unresolved, hit count = 0

(msdebug)
```

Execute the command `r`. The program will run until the first breakpoint. Then execute `c` to proceed to the next breakpoint. Note that for multi-core programs, the operator program is typically dispatched to multiple accelerator cores for concurrent execution. Once one accelerator core hits a breakpoint, it will interrupt and notify the other accelerator cores to stop immediately. Therefore, other accelerator cores are not guaranteed to also stop at the same breakpoint simultaneously. The same breakpoint may also be hit again by other accelerator cores. Developers can use breakpoint disable/delete commands to prevent cores from repeatedly hitting the same breakpoint.

```bash
(msdebug) r
Process 813993 launched: '/home/catlass/output/bin/00_basic_matmul' (aarch64)
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 1.1
    frame #0: 0x0000aaaaaac0df8c 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:45:18
   42
   43       uint32_t m = options.problemShape.m();
   44       uint32_t n = options.problemShape.n();
-> 45       uint32_t k = options.problemShape.k();
   46
   47       size_t lenA = static_cast<size_t>(m) * k;
   48       size_t lenB = static_cast<size_t>(k) * n;
(msdebug) c
Process 813993 resuming
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 2.1
    frame #0: 0x0000aaaaaac0e8c0 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:101:39
   98      using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
   99
   100      using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
-> 101      MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
   102      MatmulAdapter matmulOp;
   103      matmulOp.CanImplement(arguments);
   104      size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
(msdebug) c
Process 813993 resuming
[Launch of Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel11BasicMatmulINS1_5Blo on Device 0]
Compare success.
Process 813993 exited with status = 0 (0x00000000)
(msdebug)
```

#### Viewing Variables and Memory

To view a scalar, run the `p` command to view the value of the current n variable.

```bash
Process 813993 launched: '/home/catlass/output/bin/00_basic_matmul' (aarch64)
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 1.1
    frame #0: 0x0000aaaaaac0df8c 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:45:18
   42
   43       uint32_t m = options.problemShape.m();
   44       uint32_t n = options.problemShape.n();
-> 45       uint32_t k = options.problemShape.k();
   46
   47       size_t lenA = static_cast<size_t>(m) * k;
   48       size_t lenB = static_cast<size_t>(k) * n;
(msdebug) p n
(uint32_t) $0 = 512
```

To view the memory, run the `p` command.

You can run the `x -m UB -f float16[] 65536 -c 4 -s 4` command to print the value in the accumulatorBuffer memory. A maximum of 1024 bytes can be printed at once.

```bash
(msdebug) c
Process 814339 resuming
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = breakpoint 2.1
    frame #0: 0x000000000000bf98 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:136:19
   133
   134              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   135              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
-> 136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
   137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138
   139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
(msdebug) p outputBuffer
(AscendC::LocalTensor<__fp16>[2]) $2 = {
  [0] = {
    AscendC::BaseLocalTensor<__fp16> = { # Memory and data type
      address_ = (dataLen = 131072, bufferAddr = 65536, bufferHandle = "", logicPos = '\v') # Start address and data length
    }
    shapeInfo_ = {
      shapeDim = '\x88'
      originalShapeDim = '\xf8'
      shape = {}
      originalShape = {}
      dataFormat = ND
    }
  }
  [1] = {
    AscendC::BaseLocalTensor<__fp16> = {
      address_ = (dataLen = 49152, bufferAddr = 147456, bufferHandle = "", logicPos = '\v')
    }
    shapeInfo_ = {
      shapeDim = '\x88'
      originalShapeDim = '\xf8'
      shape = {}
      originalShape = {}
      dataFormat = ND
    }
  }
}
(msdebug) x -m UB -f float16[] 65536 -c 4 -s 4 # Print four lines of 4-byte FP16 data from address 65536 in the UB memory.
0x00010000: {355.5 188.75}
0x00010004: {244.125 -364.75}
0x00010008: {-104.875 -156}
0x0001000c: {232 -100.75}
(msdebug) x -m UB -f float16[] 65536 -c 4 -s 8 # Print four lines of 8-byte FP16 data from address 65536 in the UB memory.
0x00010000: {355.5 188.75 244.125 -364.75}
0x00010008: {-104.875 -156 232 -100.75}
0x00010010: {-47.4062 105.875 -322.5 -265.75}
0x00010018: {260 200.125 -139.25 -190.625}
(msdebug)
```

To debug line by line, run the `n` command to advance the program to the next line.

```bash
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000bfe4 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:137:73
   134              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   135              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
-> 137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138
   139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
   140          }
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000c000 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:139:28
   136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
   137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138
-> 139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
   140          }
   141
   142          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000c014 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:96:68
   93           AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
   94
   95           uint32_t loops = (elementCount + tileLen - 1) / tileLen;
-> 96           for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
   97               uint32_t actualTileLen = tileLen;
   98               if (loopIdx == loops - 1) {
   99                   actualTileLen = elementCount - loopIdx * tileLen;
(msdebug)
```

To check all variables of the current stack frame, run the `var` command.

```bash
(msdebug) var
(Catlass::Gemm::Kernel::ReduceAdd<Catlass::Arch::AtlasA2, float, __fp16, 8192> *__stack__) this = 0x00000000001cf838
(const AscendC::GlobalTensor<__fp16> &__stack__) dst = 0x00000000001cf930: {
  AscendC::BaseGlobalTensor<__fp16> = {
    address_ = 0x000012c0c0094000
    oriAddress_ = 0x000012c0c0094000
  }
  bufferSize_ = 1898896
  shapeInfo_ = {
    shapeDim = 'h'
    originalShapeDim = '\xf9'
    shape = {}
    originalShape = {}
    dataFormat = ND
  }
  cacheMode_ = CACHE_MODE_NORMAL
}
(const AscendC::GlobalTensor<float> &__stack__) src = 0x00000000001cf908: {
  AscendC::BaseGlobalTensor<float> = {
    address_ = 0x000012c041400000
    oriAddress_ = 0x000012c041400000
  }
  bufferSize_ = 1898904
  shapeInfo_ = {
    shapeDim = 'H'
    originalShapeDim = '\xf9'
    shape = {}
    originalShape = {}
    dataFormat = ND
  }
  cacheMode_ = CACHE_MODE_NORMAL
}
(uint64_t) elementCount = 131072
(uint32_t) splitkFactor = 2
(const uint32_t) ELE_PER_VECTOR_BLOCK = 64
(uint32_t) aivNum = 48
(uint32_t) aivId = 26
(uint64_t) taskPerAiv = 2752
(uint32_t) tileLen = 2752
(uint32_t) loops = 48
(uint32_t) loopIdx = 26
(msdebug)
```

#### Exiting Debugging

After debugging is complete, use the `q` command to exit `msdebug`. If you force exit using `Ctrl+C` or other means, the `msdebug` process will not terminate and will continue running in the background. In this case, you can run `ps -ef | grep msdebug` to find the corresponding process PID, then run `kill -9 PID` to terminate the process. Multiple `msdebug` processes cannot be started simultaneously for debugging.

```bash
(msdebug) q
Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] y
```

#### Common Commands

| Command                                    | Abbreviation                      | Purpose                                                                                                                                                                                                                                                                                   | Example                                |
| ------------------------------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| breakpoint filename:lineNo                 | b                                 | Set a breakpoint.                                                                                                                                                                                                                                                                         | b add\_custom.cpp:85<br>b my\_function |
| run                                        | r                                 | Perform running again.                                                                                                                                                                                                                                                                    | r                                      |
| continue                                   | c                                 | Resume running.                                                                                                                                                                                                                                                                           | c                                      |
| print                                      | p                                 | Print variables.                                                                                                                                                                                                                                                                          | p zLocal                               |
| frame variable                             | var                               | Print all variables in the current frame.                                                                                                                                                                                                                                                 | var                                    |
| memory read                                | x                                 | Read memory.<br>`-m` specifies the memory location. GM, UB, L0A, L0B, and L0C are supported.<br>`-f` specifies the [byte conversion format](#appendix).<br>`-s` specifies the number of bytes to be printed in each line.<br>`-c` specifies the number of lines to be printed.            | x -m GM -f float16[] 1000 -c 2 -s 128  |
| register read                              | re r                              | Read register values.<br/>`-a` reads all register values.<br/>`\$REG_NAME` reads the value of the register with the specified name.                                                                                                                                                         | register read -are r \$PC              |
| thread step-over                           | next<br>n                         | Move to the next executable line of code in the same call stack.                                                                                                                                                                                                                          | n                                      |
| ascend info devices                        | /                                 | Query device information.                                                                                                                                                                                                                                                                 | ascend info devices                    |
| ascend info cores                          | /                                 | Query AI Core information for an operator.                                                                                                                                                                                                                                                | ascend info cores                      |
| ascend info tasks                          | /                                 | Query task information for an operator.                                                                                                                                                                                                                                                   | ascend info tasks                      |
| ascend info stream                         | /                                 | Query stream information for an operator.                                                                                                                                                                                                                                                 | ascend info stream                     |
| ascend info blocks                         | /                                 | Query block information for an operator.<br>Optional parameter: `-d/--details` displays the code of all blocks at the current breakpoint.                                                                                                                                                  | ascend info blocks                     |
| ascend aic core                            | /                                 | Switch the target cube core of the debugger.                                                                                                                                                                                                                                              | ascend aic 1                           |
| ascend aiv core                            | /                                 | Switch the target vector core of the debugger.                                                                                                                                                                                                                                            | ascend aiv 5                           |
| target modules addkernel.o                 | image addkernel.o                 | Import operator debugging information when the Pybind framework starts operators.<br>(Note: If this command is executed after the program has already been run with the `run` command,<br>an additional `image load` command is required to make the debugging information take effect.) | image addAddCustom\_xxx.o              |
| target modules load -f kernel.o -s address | image load -f kernel.o -s address | Make the imported debugging information take effect after the program has run.                                                                                                                                                                                                            | image load -f AddCustom\_xxx.o -s 0    |

## Appendix

### Data Formats Supported by msDebug

```bash
Valid values are:
"default"
'B' or "boolean"
'b' or "binary"
'y' or "bytes"
'Y' or "bytes with ASCII"
'c' or "character"
'C' or "printable character"
'F' or "complex float"
's' or "c-string"
'd' or "decimal"
'E' or "enumeration"
'x' or "hex"
'X' or "uppercase hex"
'f' or "float"
"brain float16"
'o' or "octal"
'O' or "OSType"
'U' or "unicode16"
"unicode32"
'u' or "unsigned decimal"
'p' or "pointer"
"char[]"
"int8_t[]"
"uint8_t[]"
"int16_t[]"
"uint16_t[]"
"int32_t[]"
"uint32_t[]"
"int64_t[]"
"uint64_t[]"
"bfloat16[]"
"float16[]"
"float32[]"
"float64[]"
"uint128_t[]"
'I' or "complex integer"
'a' or "character array"
'A' or "address"
"hex float"
'i' or "instruction"
'v' or "void"
'u' or "unicode8"
```

### Specifying NPUs for Debugging

Set the environment variable `ASCEND_RT_VISIBLE_DEVICES` to the ID of the NPU to be used. For example:

```bash
# Specify that the current process only uses the device with Device ID 2.
export ASCEND_RT_VISIBLE_DEVICES=2
```
