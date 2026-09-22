# Sample Contribution Process

This document details the process of contributing complete operator samples to the CATLASS template library. The process includes four stages: design, development, test, and merging. It aims to help developers contribute high-quality operator samples in a standardized manner.

## Design

### Requirement Analysis

Before starting the implementation, the following points must be clarified:

- **Operator functionality**: Define the mathematical definition of the operator, input and output formats, supported data types, etc.
- **Use cases**: Specify the typical use and performance requirements of the operator.
- **Compatibility**: Confirm the operator's compatibility with the existing template library.
- **New features**: Identify the unique features of the new operator to ensure it does not completely duplicate existing features in the repository.

### Solution Design

Design a specific implementation solution based on the requirement analysis:

- **Template selection**: Select an appropriate base GEMM template (e.g., BasicMatmul, SplitkMatmul, etc.).
- **Parameter design**: Design tiling parameters, dispatch policies, etc.
- **Optimization strategy**: Develop performance optimization strategies, such as Preload and SplitK.
- **New components**: Design new components for new features to ensure complete functionality and good integration with existing components. For code that does not involve new features, reuse existing component templates in the repository whenever possible.

### Documentation Design

Design the document structure for the operator sample:

- **README.md**: Contains brief operator description, usage examples, etc.
- **Design document**: Explains the sample's prototype design, solution design, sample implementation, and performance testing. When evaluating performance, if a benchmark exists, describe the scenarios where the sample has advantages over the benchmark.

## Development

### Environment Setup

Set up the development environment by referring to [Quick Start](./01_quick_start.md).

1. Install the CANN toolkit.
2. Clone the code repository.
3. Configure the build environment.

### Code Implementation

Implement the operator sample based on the following structure:

```bash
examples/${id}_${op_name}/
├── CMakeLists.txt        # CMake build file
├── README.md             # Operator sample usage guide
├── ${id}_${op_name}.md   # Operator sample design document
└── ${op_name}.cpp        # Main implementation file
```

### Core Implementation

- **Host**: Assemble the operator on a host by referring to [Host Example Assembly](./02_host_example_assembly.md).
- **Kernel**: Implement the operator logic at the kernel by referring to [Kernel Development](./03_kernel_development.md).
- **Block**: Implement the block-layer logic by referring to [BlockMmad Development](./04_block_mmad_development.md) and [Block Scheduler Development](./05_block_scheduler_development.md).
- **Tile**: Implement the tile-layer logic by referring to [Tile Development](./06_tile_development.md).
- **Epilogue**: Implement the epilogue logic by referring to [Epilogue Adaptation](./07_epilogue_adaptation.md).

### Build and Testing

Run the following commands to build and test (using A2 as an example):

```bash
# Build the operator sample.
bash scripts/build.sh ${id}_${op_name}

# Run the test.
./output/${id}_${op_name}
```

## Test

### Precision Test

Perform generalization precision testing on at least 200 cases, covering different input shapes and data types. If the test benchmark cannot reuse an existing benchmark from `examples/common/golden`, supplement it with a custom benchmark. **The precision test results must be described in the pull request (PR).**

### Performance Test

Conduct comparative testing against performance benchmarks to demonstrate performance advantages. Add detailed information such as performance results, test environment, and benchmark to the operator design document.

## Merge

### Code Preparation

- **Code standards**: Ensure that the code complies with the project's coding style and standards.
- **Documentation improvement**: Complete the README.md and design document.
- **Test coverage**: Ensure that the test covers sufficient scenarios.
- **Test cases**: Add test cases for the new sample in `tests/test_example.py`.

### Submission Process

Refer to [Contributing](../../../CONTRIBUTING.md) and follow the process below:

1. **Create issue**: Create a `Requirement` issue on GitCode to describe the design solution of the operator sample.
2. **Assign task**: Enter `/assign` or `/assign @yourself` in the issue comment to assign the issue to yourself.
3. **Commit code**: Commit the code to your personal branch.
4. **Create PR**: Create a PR and fill in the PR template in detail.
5. **Review code**: Wait for the project maintainer to review the code.
6. **Merge code**: After the review is passed, the code will be merged into master.

### Subsequent Maintenance

- **Issue response**: Respond promptly to issues raised by the community.
- **Documentation updates**: Update the documentation based on feedback.
