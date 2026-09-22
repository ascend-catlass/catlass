# Template Library Quick Start

## Environment Setup

> **Note**: Ensure that [basic dependencies](../../../README_en.md#-software-and-hardware-requirements), [NPU driver](https://www.hiascend.com/hardware/firmware-drivers/community), and firmware have been installed before you start.

### **Installing the Community Edition CANN Toolkit**

Download the CANN development kit `Ascend-cann-toolkit_{version}_linux-{arch}.run` based on the type of your [Ascend product](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html). For details about the download link, see [CANN toolkit](https://www.hiascend.com/en/developer/download/community/result?module=cann). (For details about the CATLASS version support, see [Required Software and Hardware](../../../README_en.md#-software-and-hardware-requirements).)

Then, install the CANN development kit. (For details, see [CANN Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit).)

```bash
# Ensure that the installation package is executable.
chmod +x Ascend-cann-toolkit_{version}_linux-{arch}.run
# Install the CANN toolkit.
./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install_path}
```

- `{version}`: CANN package version
- `{arch}`: system architecture
- `{install_path}`: installation path, which defaults to `/usr/local/Ascend`

For details about other online installation methods, see [CANN Quick Installation](https://www.hiascend.com/cann/download).

### **Enabling the CANN Environment**

After installation, execute the following command to enable the CANN environment.

```bash
# Default installation path (using the root user as an example; for a non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Custom installation path
# source ${install_path}/set_env.sh
```

### **Downloading the Source Code**

Download the CATLASS code repository to the local PC.

```bash
# Download the project source code, using the master branch as an example.
git clone https://gitcode.com/cann/catlass.git
```

## Build and Execution

> The template library provides a set of reusable templates and basic components to empower matrix multiplication operator development. Operator samples can be found [here](../../../examples).

### **Building the Sample**

Go to the root directory of the project and run the following build command:

```bash
bash scripts/build.sh [options] <target>
```

- `options`: Optional build options. Currently supported options include:
  - `--clean`: Clears the previous build and output directories (the default paths are `/build` and `/output`, respectively).
  - `--debug`: Compiles in debug mode.
  - `--msdebug`: Enables [msDebug](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0062.html). For details, see [Using msDebug in a CATLASS Sample Project](./evaluation/msdebug.md).
  - `--simulator`: Enables the simulator mode. After this option is enabled, the code will not run on an actual NPU. For details, see [CATLASS Sample Simulation](./evaluation/performance_tools.md#example-with-msprof-op-simulator).
  - `--enable_profiling`: Enables the profiling tool. For details, see [Tuning Performance in a CATLASS Sample Project](./evaluation/performance_tools.md#profiling-overview).
  - `--enable_print`: Enables the compiler's print functionality. For details, see [Device-Side Printing Based on `cce::printf`](./evaluation/print.md).
  - `--enable_ascendc_dump`: Enables `Ascend C` operator debugging APIs. For details, see [Using Ascend C Operator Debugging APIs in a CATLASS Sample Project](./evaluation/ascendc_dump.md).
  - `-DCATLASS_ARCH`: Specifies the NPU architecture. Currently, `2201` and `3510` are supported.
  - `-D<option>`: Passes other build options to CMake.

- `target`: Specifies the operator sample to build. You can specify a specific sample name or one of the following:
  - `catlass_examples`: Builds all samples in the repository.
  - `python_extension`: Builds the Pybind extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `torch_library`: Builds the Torch extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `mstuner_catlass`: Builds msTuner_CATLASS. For details, see [`mstuner_catlass` Usage Guide](../../../tools/tuner/README.md).

Take the [basic_matmul](../../../examples/00_basic_matmul/README.md) sample as an example. Run the following command:

```bash
# Build the operator component.
bash scripts/build.sh 00_basic_matmul
```

If the following prompt appears, the build succeeded.

```bash
"[INFO] Target "{target}" built successfully."
```

### **Executing Operators**

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

For details about how to start developing your first operator, see [Host-Side Code Assembly Guide](./02_host_example_assembly.md).
