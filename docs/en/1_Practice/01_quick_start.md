# Template Library Quick Start

Before learning about template library development, complete the basic environment setup and source code download by referring to the following steps. Ensure that the NPU driver and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) are installed, and then try to build and run a basic template library sample.

## Installing CANN

See [CANN Quick Installation](https://www.hiascend.com/cann/download), select the stable version (only CANN 8.5.0 and later are supported), download the corresponding package based on your product series, CPU architecture, and OS, and complete the installation by following the commands provided on the page.

## Installing Python Dependencies

Some complex samples use Python to generate and process input and output data, so Python and its dependencies must be installed. Install a Python version available for your operating system, and ensure that pip or another package manager is installed. The following uses pip as an example.

### Installing PyTorch and TorchNPU

Some complex samples use PyTorch and TorchNPU. See [PyTorch Ascend Installation and Deployment](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download), and select and install the TorchNPU version that matches your CANN version.

The following example installs `torch==2.9.0+cpu` and `torch-npu==2.9.0.post8`:

```bash
pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.9.0.post8 --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/
```

## Installing Other Basic Dependencies

Install the basic dependencies one by one by referring to [Required Software and Hardware](../../../README_en.md#-software-and-hardware-requirements) according to your actual OS environment.

### Environment Verification

After installing the CANN package, verify that the environment and driver work properly.

- **Check the NPU device**

    ```bash
    # Run npu-smi. If the device information is displayed properly, the driver is working.
    npu-smi info
    ```

- **Check the CANN version**

    ```bash
    # View the CANN toolkit package version (default installation path).
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # View the CANN ops package version (default installation path).
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    `${arch}` can be queried using `uname -m`, for example, aarch64 or x86_64.

- **Check PyTorch and TorchNPU**

    ```bash
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    Output similar to the following indicates a successful installation.

    ```text
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

- **Check other basic dependencies**

    ```bash
    gcc --version
    cmake --version
    ```

### Environment Variable Configuration

Run the appropriate command to make the environment variables take effect.

```bash
# Default installation path (using the root user as an example; for a non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Custom installation path
# source ${install_path}/cann/set_env.sh
```

### Downloading the Source Code

Download the CATLASS code repository to the local PC.

```bash
# Download the project source code, using the master branch as an example.
git clone https://gitcode.com/cann/catlass.git
```

Go to the repository root and install the common Python packages:

```bash
pip install -r requirements.txt
```

## Build and Execution

> The template library provides a set of reusable templates and basic components for matrix multiplication operator development. Operator samples can be found [here](../../../examples).

- Samples whose names contain `ascend950` are supported only on Ascend 950PR&950DT series products.
- Other samples are supported only on Atlas A2 series products and Atlas A3 series products, and the architecture tag `ArchTag` in the code is uniformly `Arch::AtlasA2`.

### Building the Sample

Go to the root directory of the project and run the following build command:

```bash
bash scripts/build.sh [options] <target>
```

- `options`: Optional build options. Currently supported options include:
  - `--clean`: Removes the `build` and `output` directories under the repository root.
  - `--debug`: Compiles in debug mode.
  - `--msdebug`: Enables [msDebug](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsdebug). For details, see [Using msDebug in a CATLASS Sample Project](./evaluation/msdebug.md).
  - `--simulator`: Enables the simulator mode. After this option is enabled, the code will not run on an actual NPU. For details, see [CATLASS Sample Simulation](./evaluation/performance_tools.md#example-with-msprof-op-simulator).
  - `-DCATLASS_ARCH`: Specifies the NPU architecture. Currently, `2201` (Atlas A2 series products and Atlas A3 series products) and `3510` (Ascend 950PR&950DT series products) are supported. If not specified, the default is `2201`.
  - `-D<option>`: Passes other build options to CMake.

- `target`: Specifies the operator sample to build. You can specify a specific sample name or one of the following:
  - `catlass_examples`: Builds all samples in the repository.
  - `python_extension`: Builds the Pybind extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `torch_library`: Builds the Torch extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `mstuner_catlass`: Builds msTuner_CATLASS. For details, see [`mstuner_catlass` Usage Guide](../../../tools/tuner/README.md).

Take the [basic_matmul](../../../examples/00_basic_matmul/README.md) sample as an example. Run the following command (for other samples that support the Ascend 950PR&950DT series products, add `-DCATLASS_ARCH=3510`):

```bash
# Build the operator component.
bash scripts/build.sh 00_basic_matmul
```

If the following prompt appears, the build succeeded.

```text
[INFO] Target '00_basic_matmul' built successfully
```

When the build fails, `build.sh` exits with a non-zero status and prints error messages. Common outputs include:

- `[ERROR]No target specified`: no sample name was specified.
- `[ERROR]ASCEND_HOME_PATH environment variable is not set!`: the CANN environment script has not been sourced (see [Environment Variable Configuration](#environment-variable-configuration)).
- `[ERROR]Multiple targets specified`: more than one sample name was specified.

Other failures come from CMake configuration or the compiler. The output contains the file and line number of the error, which can be used to locate issues such as missing headers or mismatched architecture tags.

If the build fails with a missing C++ standard header (for example, `fatal error: 'cstdint' file not found`) or a missing standard library at link time (for example, `unable to find library -lstdc++`), the usual cause is that the system C++ header/library search paths were not passed to the CANN compiler. See the [FAQ](../FAQ.md) for the workaround.

### Executing Operators

The operator build output is located in the `output/bin` directory. Switch to this directory to run the operator sample program.
Take the [basic_matmul] sample as an example. You can run the operator by executing the following commands:

```bash
# Switch to the build output directory.
cd output/bin
# ./00_basic_matmul m n k [deviceId]
./00_basic_matmul 256 512 1024 0
```

- `256`, `512`, and `1024` are the dimensions of the matrix multiplication along the M-axis, N-axis, and K-axis, respectively (left/right matrix data is randomly generated).
- `deviceId` (optional, default value: 0) specifies the NPU ID.

After executing the operator sample, the following result indicates that the computation meets the precision expectations (in this sample, the left and right matrices of the matmul are filled with random numbers, and the ground truth is based on CPU computation).

```text
Compare success.
```

When the precision comparison fails, the program prints `Compare failed. Error count: <N>`, where `N` is the number of elements outside the error tolerance. In that case, first verify that the input data generation matches the golden computation, then check the data copy ranges and the accumulation process.

To start complete development of your first innovative operator, see the [Innovative Example Development Guide](./10_innovative_example_development_guide.md). To try host-side operator template assembly, see the [Host-Side Code Assembly Guide](./02_host_example_assembly.md).
