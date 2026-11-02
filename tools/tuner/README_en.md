# msTuner_CATLASS (MindStudio Tuner for CATLASS) - Automatic Tiling Optimization Tool

msTuner_CATLASS is a tool for optimizing Tiling parameters of operators in the CATLASS template library. It supports user-defined search spaces, can instantiate all operators within the search space, and complete on-board performance testing in batches, providing a reference for the optimization of operator Tiling parameters.

## Quick Start

Take the tiling parameter optimization of **00_basic_matmul** with m=256, n=512, k=1024 as an example. Use the default search space configuration and execute the following command to compile the tool.

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass
```

Run the `mstuner_catlass` command to start the performance test.

```bash
export LD_LIBRARY_PATH=$PWD/output/lib64/:$LD_LIBRARY_PATH
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

After the command is successfully executed, the following information is displayed. (The actual running result may vary depending on the hardware differences and hardware performance fluctuation.)

```bash
$ ./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
[INFO ] Set profile output file /path_to_my_repo/catlass/output/results.csv
[INFO ] Start to initialize device 0
[INFO ] Initializing device 0 success
[INFO ] Initializing 1701 operations
[WARN ] Current freq 800 is lower than rated freq 1800, run warm up
[INFO ] Warm up finished, rated freq 1800, current freq 1800
================================

             case_id : 1
   task_duration(us) : 19.380
           device_id : 0
           operation : Gemm
         description : catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x0
       l0_tile_shape : 32x128x32
       l1_tile_shape : 32x128x128
             swizzle : swizzle3x0
                   m : 256
                   n : 512
                   k : 1024
                   A : fp16:row
                   B : fp16:row
                   C : fp16:row

================================

...

================================
Top 10:
case_id,task_duration(us),device_id,operation,description,m,n,k,A,B,C
489,12.740,7,Gemm,catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_64x128x128_64x128x64_swizzle3x1,256,512,1024,fp16:row,fp16:row,fp16:row
...
[INFO ] Save profile data to /path_to_my_repo/catlass/output/results.csv success
```

## Compilation

You can filter operators using the `-DCATLASS_LIBRARY_KERNELS=<kernel_name>` command. If the `description` of an operator contains `kernel_name`, the test case code of the operator will be generated and compiled. For example, run the following command to compile the `00_basic_matmul` operator:

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass
```

You can directly specify the description information of a specific single operator instance. For example, use the following command to compile only the operator with case_id=1 shown in the Quick Start.

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x1 mstuner_catlass
```

Currently, the following operator types are supported:

- 00_basic_matmul
- 02_grouped_matmul_slice_m
- 06_optimized_matmul
- 08_grouped_matmul
- 12_quant_matmul
- 27_matmul_gelu

Note:

- When the 06_optimized_matmul operator has different m/n/k inputs, different kernels are enabled, including 06_optimized_matmul_padding_ab, 06_optimized_matmul_padding_a_only, 06_optimized_matmul_padding_b_only, 06_optimized_matmul_without_padding. You can directly specify the expected kernel to accelerate the optimization process. For details about the kernel matching logic, see [optimized_matmul.cpp](../../examples/06_optimized_matmul/optimized_matmul.cpp).

In addition to the preceding commands, you can also use the CMake command to complete the compilation.

```bash
mkdir build
cd build
cmake .. -DCATLASS_LIBRARY_KERNELS=00_basic_matmul
make mstuner_catlass -j
cmake --install . --component catlass_kernels
cmake --install . --component mstuner_catlass
```

## Tool Running Commands

The `mstuner_catlass` tool supports the following commands.

| Command         | Example                         | Default Value| Description           |
| ------------- | ------------------- |-----------| ------------------ |
| --help, -h    | --help                        | / | Displays the commands supported by the tool.                                          |
| --kernels     | --kernels=00_basic_matmul        | / | Filters the types of operators for optimization. It performs substring matching with the string in the description column of the operator, and the operator will be skipped if there is no match.|
| --output      | --output=./profile_result.csv | / | Specifies the file path for writing operator performance data to disk.                                |
| --device      | --device=0                    | 0 | Specifies the single card ID to run.                                            |
| --m           | --m=256                       | 256 | Specifies the dimension m of the input matrix.                                         |
| --n           | --n=512                       | 512 | Specifies the dimension m of the input matrix.                                         |
| --k           | --k=1024                      | 1024 | Specifies the dimension m of the input matrix.                                         |
| --A           | --A=fp16:row                  | / | Filters operators by the data type and memory layout of matrix A.                     |
| --B           | --B=fp16:column               | / | Filters operators by the data type and memory layout of matrix B.                   |
| --C           | --C=fp16:row                  | / | Filters operators by the data type and memory layout of matrix C.                   |
| --group_count | --group_count=128             | 128 | Specifies the number of groups for grouped_matmul operators.                         |

When multiple data types and memory layouts are configured and generated for A, B, and C in the search space, you can use the `--A/--B/--C=<data type>:<memory layout>` command to filter operators.

- The data type can be `u8, int8, int32, fp16, bf16, fp32`.
- The memory layout can be `row, column, nZ, zN, zZ, padding_row_major, padding_column_major, nN`.
- The input must be in the format of `<data:layout>`, for example, `<data:layout>` or `fp32:zZ`.
Note: If `--output` is not specified, operator profile data will not be written to disks.

## Search Space Configuration

msTuner_CATLASS allows you to customize the search space of operator tiling parameters, including layouts, data types, L1/L0 TileShape, and Swizzle policies. It automatically generates a full search space through orthogonalization and allows you to customize pruning functions to filter the search space. Each orthogonal configuration combination is instantiated into an independent operator, and the generated operator instantiation code is stored in the `build/tools/library/generated` directory.

When the search space is configured to a large range, tens of thousands of operators may be instantiated. Code expansion causes long compilation time. In addition, too many operators may exceed the hardware limit, and the compilation may fail. In addition, when there are a large number of operators, the registration before operator delivery takes a long time. Therefore, it is recommended that the search space be limited to 5000 to ensure smooth tool running and optimal experience.

You can view the number of operators in the log file `build/tools/library/catlass_library_code_generation.log`. As shown in the following figure, the search space of 00_basic_matmul contains 1701 operators.

```txt
INFO:search_space:00_basic_matmul tile_shapes size=1701
INFO:search_space:08_grouped_matmul tile_shapes size=576
INFO:manifest:operations that will be generated in total: 1701
...
```

Search space configuration supports entry-level configuration and advanced configuration.

### Entry-level Configuration

msTuner_CATLASS supports entry-level simplified configuration of the search space for operator tiling parameters (see the `tools/library/scripts/search_space_config.py` file). Developers can adjust the following parameters to set the search range.

- kernel_type: operator type
- data_type_a/data_type_b/data_type_c: element type of the A/B/C input matrix
- layout_a/layout_b/layout_c: memory layout of the A/B/C input matrix
- l1_tile_m_range: search range of the m-axis value of the L1 tile shape
- L1_tile_n_range: value search range of the n-axis of the L1 tile shape
- L1_tile_k_range: value search range of the k-axis of the L1 tile shape
- block_swizzle: Swizzle policy

The search space configuration of the 00_basic_matmul operator is as follows:

```python
@OperationRegistry.register_high_priority('00_basic_matmul')
def register(manifest):
    config = search_space.SearchSpaceConfiguration(
        kernel_type='00_basic_matmul',

        data_type_a=library.DataType.fp16,
        data_type_b=library.DataType.fp16,
        data_type_c=library.DataType.fp16,

        layout_a=library.LayoutType.RowMajor,
        layout_b=library.LayoutType.RowMajor,
        layout_c=library.LayoutType.RowMajor,

        l1_tile_m_range=(32, 128),  # min and max of a range are set here
        l1_tile_n_range=(128, 256),
        l1_tile_k_range=(128, 256),

        block_swizzle='Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    )

    search_space.register_custom_kernel(config, manifest)
```

The operator search space of the entry-level configuration will overwrite the configuration of the same type of operator in the advanced configuration. If the entry-level configuration is not required and the advanced configuration is used, you can comment out this line of code.

```python
# @OperationRegistry.register_high_priority('00_basic_matmul')
```

### Advanced Settings

msTuner_CATLASS supports more flexible custom configuration of the search space for operator tiling parameters in the `tools/library/scripts/search_space.py` file. It supports custom configuration of the orthogonal combination mode of layouts, data types, L1/L0 Tile Shapes, Swizzle strategies and other parameters, and custom pruning functions to filter and traverse the search space.

Taking the search space of the `00_basic_matmul` operator as an example, its configuration is located in the function `register_gemm_00_basic_matmul_operation`.

- Layout configuration

  ```python
  layouts = [
    [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
  ]

  ```

- Data type configuration

  ```python
  data_types = [
      [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16]
  ]
  ```

- L1/L0 tile shape configuration and custom pruning function `tile_shape_constraint_for_pingpong`

  ```python
  tile_shapes = list(generate_tile_shapes(
      tile_shape_constraint_for_pingpong, # set constraint function based on dispatch policy
      # below are arguments for constraint function
      element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
      stages=2, # stages of dispatch policy for estimating boundary conditions, e.g. 2 for UB stages
      step=16, # step size for iterating the next tile shape on each dimension of L1/L0 tile shape
      tile_shape_range=TileShapeRange(
          l1_tile_m_range=(32, 128),  # range of L1TileShape::M/N/K
          l1_tile_n_range=(128, 256),
          l1_tile_k_range=(128, 256),
          l0_tile_m_range=(32, 128),  # range of L0TileShape::M/N/K
          l0_tile_n_range=(128, 256),
          l0_tile_k_range=(32, 64)
      )
  ))
  ```

- Swizzle policy configuration

  ```python
      block_swizzle_descriptions = [
          'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
      ]
  ```

Similarly, the search space configuration of the `08_grouped_matmul` operator is located in the function `register_gemm_08_grouped_matmul_operation` and supports custom configuration.
