# Introduction to CATLASS

In the `Transformer` model, General Matrix Multiplication (GEMM) handles much of the computation. Optimizing GEMM boosts overall efficiency. Different scenarios require various GEMM operator implementations, and evolving algorithms create new customization needs. This makes it hard to predict all variants in advance. Directly developing GEMM operators based on hardware is challenging and time-intensive. To address this issue, the CANN team has launched CATLASS, which uses a layered, modular design to decouple GEMM computation into flexible components such as data tiling policies and compute unit settings.

This speeds up development tailored to Atlas hardware. The library offers reusable templates, basic components, and example cases, helping developers quickly set up the computation pipeline. They can customize compute kernels based on hardware and needs, enhancing efficiency and maintaining high performance.

CATLASS uses a layered design. It divides the implementation into multiple levels by analyzing hardware architectures and GEMM requirements. Each level's common logic is abstracted into templates, with room for custom extensions. This ensures software matches specific hardware and computation stages. Specific steps in algorithms are left to subclasses, allowing the subclasses to redefine these steps without altering the overall structure.

## Layered Modular Design of Operator Implementation

![api_level_en](../../assets/images/api_level_en.png)

## Flexible Configuration of Operator Pipeline

CATLASS allows for flexible development. Developers can reuse preset patterns to quickly set up basic functions, modify modules for custom needs, or swap components to create custom pipelines. This design ensures high performance and gives developers enough flexibility and scalability.

![api_custom](../../assets/images/api_custom.png)

## Project Directory

The full project directory structure is as follows:

```bash
catlass
├── 3rdparty                     # Third-party dependencies
|   └── googletest               # GoogleTest framework
├── cmake                        # CMake common functions
├── docs                         # Documentation directory
├── examples                     # Directory for all kernel operator examples
|   ├── 00_basic_matmul          # Single-operator example
|   |   ├── basic_matmul.cpp     # Operator calling on the host
|   |   ├── CMakeLists.txt
|   |   └── README.md            # Operator sample description
|   ├── ...
|   ├── advanced                 # Advanced examples
|   └── common                   # Common components (golden, helper, etc.)
├── include                      # Template header files
|   ├── catlass                  # Implementation logic of operators at different levels
|   |   ├── arch                 # Hardware architecture abstraction
|   |   ├── conv                 # Convolution operator templates
|   |   ├── epilogue             # Epilogue templates
|   |   ├── gemm                 # GEMM operator templates
|   |   ├── gemv                 # GEMV operator templates
|   |   └── layout               # Data layout definitions
|   └── tla                      # TLA framework
├── python                       # Python code generation framework
|   └── catlass_cppgen           # Python framework for generating C++ kernel code
├── scripts                      # Scripts
├── tests                        # Test cases
|   ├── optest                   # Python operator functional tests
|   ├── unittest                 # C++ unit tests
|   └── self_contained_includes  # Header self-containment check
└── tools                        # Tuning tools
```
