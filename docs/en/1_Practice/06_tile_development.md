# Tile Component Code Explained

## Tile Component Overview

Tile components are the lowest-level computation and data operation units in the CATLASS template library. They reside at the end of the entire hierarchy and interact directly with the hardware. Their primary responsibility is to implement basic operations such as matrix multiplication (TileMmad) and data copy (TileCopy), serving as the foundation for building high-performance operators.

Tile components adopt highly optimized implementations that fully exploit hardware features, including:

- Direct calls to Ascend C underlying APIs
- Support for multiple data types and layouts
- Optimization for different architectures
- Efficient memory access patterns

This document uses `TileMmad` as an example to dive into the code structure, main interfaces, and design ideas of tile components. It also covers commonalities of other tile components such as TileCopy.

## Template Assembly Mechanism

Tile components adopt a template-based design that supports flexible configuration and extension. Using TileMmad as an example, its basic template structure is as follows:

```cpp
template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_
>
struct TileMmad {
    // Core implementation
};
```

### Core Template Parameters

| Parameter | Description                                                                         |
| --------- | ----------------------------------------------------------------------------------- |
| ArchTag_  | Architecture tag used to distinguish different NPU architectures (e.g., 2201, 3510) |
| AType_    | Type of matrix A, containing element type and layout information                    |
| BType_    | Type of matrix B, containing element type and layout information                    |
| BiasType_ | Type of bias, which defaults to an empty type                                       |

These template parameters allow tile components to flexibly adapt to different hardware architectures and computation requirements.

### Type Export

Tile components define a series of exported types using the `using` keyword, ensuring type consistency and maintainability:

```cpp
using ElementA = typename AType_::Element;
using ElementB = typename BType_::Element;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
```

ElementAccumulatorSelector is a helper tool that automatically selects an appropriate accumulator type based on the input ElementA and ElementB types. This ensures a balance between computation precision and performance.

## Core Data Structure

Tile components typically contain few data members, relying mainly on template parameters and input parameters to perform computations. Take TileMmad as an example. It has no additional data members. All information required for computation is passed through template parameters and the parameters of the operator() method.

This design makes tile components highly lightweight and flexible for frequent calls.

## Main Interfaces

### Constructor

```cpp
CATLASS_DEVICE
TileMmad() {}
```

TileMmad provides a default constructor for creating TileMmad objects. Since TileMmad has no data members, the constructor is null.

### operator() Method

The operator() method is the core interface of tile components, executing actual computation or data operations. TileMmad provides multiple overload versions of the operator() method to support different combinations of input parameters.

#### Basic Matrix Multiplication

```cpp
CATLASS_DEVICE
void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
     AscendC::LocalTensor<ElementA> const &l0ATensor,
     AscendC::LocalTensor<ElementB> const &l0BTensor,
     uint32_t m, uint32_t n, uint32_t k,
     bool initC = true, uint8_t unitFlag = 0)
{
    // Implementation of matrix multiplication
}
```

This method performs a basic matrix multiplication operation. It stores the result of multiplying l0ATensor and l0BTensor into l0CTensor. Parameter description:

- l0CTensor: L0 cache tensor for the result matrix C
- l0ATensor: L0 cache tensor for the input matrix A
- l0BTensor: L0 cache tensor for the input matrix B
- m: Number of rows in matrix A and matrix C
- n: Number of columns in matrix B and matrix C
- k: Number of columns in matrix A and number of rows in matrix B
- initC: Whether to initialize the result matrix C. Defaults to true.
- unitFlag: Compute unit flag. Defaults to 0.

#### Matrix Multiplication with Bias

```cpp
CATLASS_DEVICE
void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
     AscendC::LocalTensor<ElementA> const &l0ATensor,
     AscendC::LocalTensor<ElementB> const &l0BTensor,
     AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,
     uint32_t m, uint32_t n, uint32_t k,
     bool initC = true, uint8_t unitFlag = 0)
{
    // Implementation of matrix multiplication with bias
}
```

This method adds support for the bias matrix l0BiasTensor on top of the basic matrix multiplication.

## Implementation Details and Optimization Strategies

### Architecture-specific Optimization

Tile components implement specific optimizations for different NPU architectures:

```cpp
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 2201)
    // Implementation specific to the 2201 architecture
#elif (defined (__NPU_ARCH__) && __NPU_ARCH__ == 3510)
    // Implementation specific to the 3510 architecture
#endif
```

This conditional compilation approach ensures that tile components can fully utilize different architectures while maintaining code uniformity.

### MmadParams Configuration

MmadParams is a parameter structure in Ascend C, which configures matrix multiplication operations:

```cpp
AscendC::MmadParams mmadParams;
mmadParams.m = m;
mmadParams.n = n;
mmadParams.k = k;
mmadParams.unitFlag = unitFlag;
mmadParams.cmatrixInitVal = initC;
```

TileMmad configures appropriate MmadParams based on different input parameters and architectures to achieve optimal performance.

### Pipe Barrier Optimization

To ensure computation correctness and performance, TileMmad inserts pipe barriers where appropriate:

```cpp
const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
    AscendC::PipeBarrier<PIPE_M>();
}
```

Pipe barriers are inserted based on the matrix size and only when the matrix is relatively small to avoid unnecessary performance overhead.

## TileCopy Overview

TileCopy is another important tile component responsible for copying data among cache levels. Similar to TileMmad, it adopts a template-based design and supports multiple data types and layouts.

```cpp
template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    // ...
};
```

TileCopy internally combines multiple specific copy components, including but not limited to:

- CopyGmToL1A/B: copy from global memory to L1 cache
- CopyL1ToL0A/B: copy from L1 cache to L0 cache
- CopyL0CToGm: copy from L0 cache to global memory

This design allows TileCopy to flexibly handle data transfers between different levels of the memory hierarchy.

## Execution Flow Analysis

The typical execution flow of tile components is as follows:

### TileMmad Execution Flow

1. **Initialization**: Create a TileMmad object.
2. **Parameter configuration**: Prepare the input tensor and computation parameters.
3. **Computation**: Call the operator() method to perform matrix multiplication.
4. **Result return**: Store the computation result in the output tensor.

### TileCopy Execution Flow

1. **Initialization**: Create a TileCopy object.
2. **Parameter configuration**: Prepare the source and destination tensors.
3. **Copy**: Call the appropriate copy method to perform data transfer.
4. **Transfer**: Transfer data from the source to the destination location.

## Relationship with Other Components

The relationship between tile components and other components is as follows:

- **Upper-layer components**: Block MMAD components (e.g., BlockMmad) call TileMmad and TileCopy to perform specific computations and data operations.
- **Underlying APIs**: Tile components directly call Ascend C underlying APIs to fully exploit hardware.
- **Helper tools**: Tile components use helper tools in the `helper` namespace (e.g., ElementAccumulatorSelector) to perform operations like type selection.

This layered design gives the CATLASS template library good modularity and maintainability.

## Summary

Tile components are the underlying computation and data operation units in the CATLASS template library. They interact directly with hardware. They adopt a template-based design, support multiple data types and layouts, and implement specific optimizations for different architectures.

TileMmad, as one of the important components, implements efficient matrix multiplication operations. By properly configuring MmadParams and inserting pipe barriers, it ensures computation correctness and performance. TileCopy is responsible for copying data between different levels of cache. It adopts a compositional design to flexibly handle various data transfer requirements.

The design of tile components fully reflects the core ideas of the CATLASS template library: achieving highly flexible and efficient operator development through template-based and layered design. Understanding how tile components work is crucial for deeply grasping the CATLASS template library and developing high-performance operators.
