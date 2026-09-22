# basic_matmul_aclnn example

aclnn is the standard calling interface used throughout the CANN software stack. The msOpGen tool provided by CANN generates the template project framework for these interfaces, enabling users to quickly develop custom operators with aclnn APIs and leverage the full feature suite of the CANN stack. This sample demonstrates how to integrate a CATLASS operator template into an msOpGen project, complete with example code and key implementation notes following the CATLASS example style.

The following guide uses `basic_matmul` to demonstrate the end-to-end integration workflow with msOpGen.

## Creating an Operator Project

Refer to the [Operator Project Creation Guide](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0060.html) to define your operator prototype JSON file and generate the base engineering code.

### Defining JSON

Sample code: [catlass_basic_matmul.json](./catlass_basic_matmul.json)

### Generating a Project

Run the following script to run `msOpGen` to generate an operator project:

```bash
msopgen gen -i catlass_basic_matmul.json -c ai_core-<soc_version> -lan cpp -out catlass_basic_matmul
```

- The `soc_version` string can be retrieved with `npu-smi info`, for example, `Ascendxxxyyy`.
- Ensure that the input JSON configuration file (`catlass_basic_matmul.json` in the preceding example) is configured with 644 file permissions.
- Ensure that the target output directory (`catlass_basic_matmul` in the preceding example) is created with 755 permissions.

## Implementing Host-side Code

Refer to [Host-Side Tiling Implementation - Basic Process](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html) to implement the TilingFunc logic.

To support **operator graph fusion**, refer to [Integrating Operators into a GE Graph](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0078.html) to implement `InferShape` and `InferDataType`.

Sample code:
[op_host/catlass_basic_matmul.cpp](./op_host/catlass_basic_matmul.cpp)
[op_host/catlass_basic_matmul_tiling.h](./op_host/catlass_basic_matmul_tiling.h)

## Implementing Device-side Code

Refer to [Operator Implementation on the Kernel](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0063.html) to write your device kernel code.

Sample code:
[op_kernel/catlass_basic_matmul.cpp](./op_kernel/catlass_basic_matmul.cpp)

**Note**:

- You must update the compilation configurations to include the CATLASS header search paths. Modify `op_kernel/CMakeLists.txt` to append the **include path** and **architecture macro** options.
  - Add the include path option `-I${CATLASS_INCLUDE_PATH}`. Replace `${CATLASS_INCLUDE_PATH}` with the actual absolute path to the `include` folder within your local CATLASS repository.
  - Add the architecture macro option `-DCATLASS_ARCH=${ARCH}`. `${ARCH}` indicates the ID of the corresponding architecture.
- The default value varies according to the CANN version.
  - CANN version >= `9.0.0.beta2`

    ```diff
    # ...
    + npu_op_kernel_options(ascendc_kernels ALL OPTIONS -I${CATLASS_INCLUDE_PATH})
    # ...
    ```

  - CANN version < `9.0.0.beta2`

    ```diff
    # set custom compile options
    if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
        add_ops_compile_options(ALL OPTIONS -g -O0)
    endif()
    + add_ops_compile_options(ALL OPTIONS -I${CATLASS_INCLUDE_PATH})
    add_kernels_compile()
    ```

- The split compilation toolchain used by `msOpGen` projects does not support passing high-level C++ structures (such as `Catlass::GemmCoord`) directly as kernel entry-point parameters. To pass structured configuration parameters, serialize the field members through the `tiling` structural data buffer on the host, and reconstruct the concrete instance manually within the device kernel code.

  ```cpp
  // Correct
  extern "C" __global__ __aicore__ void
  catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
  {
      GET_TILING_DATA(tiling_data, tiling);
      Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, tiling_data.k};
      // ...
  }
  // Unsupported currently
  extern "C" __global__ __aicore__ void
  catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace,  Catlass::GemmCoord problemShape)
  {
      // ...
  }
  ```

## Compilation and Deployment

Refer to [Operator Project Building](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html) and [OPP Deployment](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0069.html) to build and install your custom operator implementation, then set the necessary environment variables.

Generally, callers must include the header file `aclnn_catlass_basic_matmul.h` and link against the compiled `dynamic library libcust_opapi.so`. Assuming standard deployment paths, these artifacts are placed in the following locations:

```bash
$ASCEND_HOME_PATH/opp/vendors/customize/op_api/include/aclnn_catlass_basic_matmul.h
$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib/libcust_opapi.so
```

Utilize these locations when configuring target dependencies in your custom `Makefile` or `CMakeLists.txt`. You can view the CMake compilation example in [5. Invocation](#invocation).

## Invocation

Refer to the [API Overview](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/operatorlist_00001.html) to familiarize yourself with foundational aclnn calling paradigms, then reference [basic_matmul_aclnn.cpp](./basic_matmul_aclnn.cpp) for invocation.

You can compile `CMakeLists.txt` as follows:

```cmake
project(basic_matmul_aclnn)
cmake_minimum_required(VERSION 3.22)
set (CATLASS_REPO_DIR <Set_To_Actual_Local_CATLASS_Repository_Path>)
add_executable(basic_matmul_aclnn basic_matmul_aclnn.cpp)
target_include_directories(basic_matmul_aclnn PRIVATE
    ${CATLASS_REPO_DIR}/examples/common
    ${CATLASS_REPO_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/include/experiment/runtime
    $ENV{ASCEND_HOME_PATH}/include/experiment/msprof
    # Directory of the custom operator header file
    $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include
)
target_link_directories(basic_matmul_aclnn PRIVATE
    $ENV{ASCEND_HOME_PATH}/lib64
    # Directory of the custom operator library file
    $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/
)
target_link_libraries(basic_matmul_aclnn PRIVATE
    ascendcl
    nnopbase
    # Name of the custom operator library file
    cust_opapi
)
```

## Preset Example

We provide an integrated build script pipeline to automate these deployment procedures and quickly verify the aclnn operator interface.

### Running and Compiling the Test Suite

```bash
bash scripts/build.sh basic_matmul_aclnn
cd output/run
chmod +x ./custom_opp_*.run
./custom_opp_*.run
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./basic_matmul_aclnn 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```sh
Compare success.
```

## Precautions

- This example is used only as a reference for integrating the CATLASS operator into msopgen. To ensure code simplicity, generalization is not supported, such as multiple operators or platforms.
- Currently, only the example of integrating the `basic_matmul` operator is provided.
- The example supports only the following products:
  - `Atlas A2 training products/Atlas A2 inference products` (`2201` architecture)
  - `Atlas A3 training products/Atlas A3 inference products` (`2201` architecture)
