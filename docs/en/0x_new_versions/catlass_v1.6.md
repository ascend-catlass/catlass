# Overview

Ascend 950 has introduced multiple hardware features. The ones related to CATLASS include:

- Added support for MX scaling
- Added L0C → UB data channel with support for multiple movement modes
- Added support for UB → L1 data channel
- Added support for passing Coord for on-chip data movement
- Added support for Regbase and SIMT computation

To support the new platform while maintaining compatibility with older platforms and further improving ease of use, CATLASS has released a new version. The new capabilities focus on two aspects:

- Adapting to the new hardware capabilities of the 950 series, such as adding new data channels, extending BlockEpilogue, supporting CopyL0CToDst, and adding BlockPrologue.
- Extending the interfaces of TLA (Tensor Layout Abstraction) and adding support for EVG (Epilogue Visitor Graph).

# CATLASS Adaptation to New Capabilities for 950 Series

## MX Scaling Support

**MX Quantization Computation Logic**

`mxScaling` is a per-group quantization design, where each group shares a quantization scale along the reduction axis (K axis). On the 950 platform, every 32 elements share one scale. The specific instruction logic is that ScaleA is multiplied by A via broadcasting multiplication, and ScaleB is similarly multiplied by B. The results are then multiplied together and accumulated onto C:

$$C=(\mathrm{ScaleA}\otimes A)*(\mathrm{ScaleB}\otimes B)+C$$

As shown in the left figure below, the two blue elements within a group correspond to one blue scale input for the quantization parameter. Taking the blue block as an example, a (16, 32) block in L0A and a (32, 16) block in L0B are multiplied by scale A of shape (16, 1) and scale B of shape (1, 16), respectively, resulting in a (16, 16) computation result.

![](<../../assets/images/CATLASS new version capability introduction-image-10.png>)

**Data Movement**

For mxFP8/FP4, hardware and instruction interfaces have two constraints:

1. The layout of scale in L0 is fixed: scale A is zZ, and scale B is nN. This applies to both transportation and non-transportation.

2. Scale does not support transposition during the L1-to-L0 movement.

Therefore, the design for data movement is determined:

- For the L1-to-L0 movement, the corresponding fractal requirements must be met. Therefore, the layout on L1 must also be zZ/nN.
- Because two consecutive elements are fixed in the scale layout on L0, the two elements in the K direction on the GM and L1 must be consecutive to complete the movement from GM to L1. Therefore, when moving data from GM to L1, two fp8 elements need to be packed as one fp16 element (for the mx data type, the scale is fp8e8m0), using the DN2NZ movement interface.

For data movement, GMToL1 packs two fp4 elements into one int8 element for movement, and the corresponding encapsulation is completed by the instruction interface.

![mxFP8 layout in L0A and L0B](<../../assets/images/CATLASS new version capability introduction-image-9.png>)

![Scale layout in GM and L1 (RowMajor)](<../../assets/images/CATLASS new version capability introduction-image-8.png>)

For this, the new interface designs in CATLASS include:

1. `MakeMxScaleLayout`: Constructs the layout for MxScale input.

```c++
// Make a MxScale layout with Rows and Cols.
template <class Element,   // Input data type
          class LayoutTag, // Input layout type: row/col/zZ/nN
          bool isMxScaleB, // Whether it is the left or right matrix
          class T,         // rows/cols data type, both dynamic and static types allowed
          class U>
CATLASS_HOST_DEVICE constexpr
auto MakeMxScaleLayout(T const& rows, U const& cols)
```

2. `TileCopyTla`: Specialized for GMToL1 and L1ToL0 MxScale-related types.

```c++
// GMToL1
/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, B ColumnMajor in and nN out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleBTrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalenN<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor){
        // Implementation
        // ....
    }
}

// L1ToL0
// Partial specialization for CopyL1ToL0A, Ascend950, B8 or B4, nZ in and zN out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;
    template <class TensorDst, class TensorSrc, class TensorMxScale>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorMxScale const &scaleTensor)
    {
    // ... Implementation
    }
}
```

3.  `PackedMxTileCopyTla`: TileCopy encapsulation for MxScale movement

```c++
template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementMxScaleA_,      // Describe the type and layout of MxScale A/B
    class LayoutMxScaleA_,
    class ElementMxScaleB_,
    class LayoutMxScaleB_,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    class L0CCopyMode = CopyToGM
>
struct PackedMxTileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
    ElementC_, LayoutTagC, ElementBias, ReluEnable_, DEQUANT_GRANULARITY, L0CCopyMode> {
    // Implementation
    }
```

Data flows supported by the MXFP feature

| Input/Output        | Support Data Type                                       |
| ------------- | --------------------------------------------- |
| A/B           | fp8\_e5m2/fp8\_e4m3 or fp4x2\_e1m2/fp4x2\_e2m1|
| scaleA/scaleB | fp8\_e8m0                                     |
| L0C           | fp32                                          |
| C             | fp32/fp16/bf16                                |

**Computation**

The instruction interface for mxmmad has a requirement for the size of the reduction axis: K must be a multiple of 64. The actualK needs to be rounded up, and the rounded-up portion must be zeroed when moving from GM to L1. The zeroing behavior differs for RowMajor and ColumnMajor.

Take FP8 as an example. During movement, the 32-byte aligned positions along the inner axis are automatically zeroed. The following uses the movement of (32, 30) as an example.

1. Under RowMajor, K is the inner axis. The last two data elements of (32, 30) will be automatically padded with zero. Only the tail needs padding with zero.

2. Under ColumnMajor, M is the inner axis. When (32, 30) is moved to L1, the K direction will not be automatically zeroed. Therefore, 34 columns of data in the K direction need to be zeroed.
![Left: RowMajor, where the K axis is automatically padded with zero. Right: ColumnMajor, where the K axis needs to be explicitly padded with 0|697](<../../assets/images/CATLASS%20new%20version%20capability%20introduction-zeropadding.png>)
```c++
// Init Zero for k axis
InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, kL1ActualNext));
```

## Data Path Adaptations

CATLASS adopts a hierarchical design. Therefore, new features are introduced based on the CATLASS hierarchy: Tile -> Block -> Kernel -> Device.

![](<../../assets/images/CATLASS new version capability introduction-image.png>)

### New Features at the Tile Layer

#### GMToL1

1. Data movement: DN2NZ movement

![](<../../assets/images//CATLASS new version capability introduction-image-1.png>)

```c++
/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, MxScaleA RowMajor in and zZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleANoTrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalezZ<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
      // Other logic
      // ... ...
        AscendC::Dn2NzParams intriParams;
        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;
      // Other logic
      // ... ...
      AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
}
```

2. Extended support for FP4 data movement

```c++
/// Partial specialization for CopyGmToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods
    // ...Other logic
    AscendC::Nd2NzParams intriParams;

    intriParams.ndNum = ndNum;
    intriParams.nValue = nValue;
    intriParams.dValue = dValue;
    // Two fp4 elements packed as one of b8 type for movement
    if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
        intriParams.dValue = CeilDiv(intriParams.dValue, 2);
    }
    intriParams.srcNdMatrixStride = srcNdMatrixStride;
    intriParams.srcDValue = srcDValue;
    if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
        intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
    }
    intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
    intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
    intriParams.dstNzMatrixStride = dstNzMatrixStride;
    // ...Other logic
}
```

3. Specialized TileCopyTla for MxScale movement (see the MxScale section above)

##### L1ToL0

1. Added Coord descriptions: At the instruction level, L1 now supports the ability to describe a tensor using coordinates. Taking L1ToL0A as an example: With Coord, the actual memory address is represented by BuiltinTensor + Coord. As shown below, the BuiltinTensor (dataptr) of both the large and small matrices points to the same address, with the offset determined by the difference in their Coord values.

![](<../../assets/images/CATLASS new version capability introduction-image-2.png>)

| Parameter          | Description (Using an M × K Matrix as an Example)                                                                         |
| -------------- | -------------------------------------------------------------------------------------- |
| mStartPosition | Start position along the M axis of the source matrix, in units of 16 elements                                                            |
| kStartPosition | Start position along the K axis of the source matrix, in units of 32 bytes                                                                   |
| mStep          | Movement length along the M axis of the source matrix, in units of 16 elements. Value range: mStep ∈ [0, 255]                                        |
| kStep          | Movement length along the K axis of the source matrix, in units of 32 bytes. Value range: nStep ∈ [0, 255]                                               |
| srcStride      | Interval between the start address of the current fractal and the next fractal in the K direction of the source matrix, in units of 512 bytes                                                  |
| dstStride      | Interval between the start address of the current fractal and the next fractal in the K direction of the destination matrix, in units of 512 bytes                                                 |
| ifTranspose    | Whether to enable transposition for each fractal matrix. Defaults to `false`. When enabled, the source and destination operands support the b4, b8, b16, and b32 data types.|

CATLASS adds Coord to the existing Tensor representation to indicate the relationship between the Tensor to be moved and the BuiltinTensor.

```c++
/// Partial specialization for CopyL1ToL0A, Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0A and zN"
        );

        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();  // tla::Coord

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
        loadDataParams.mStep = dstOuterShapeRow;
        loadDataParams.kStep = dstOuterShapeCol;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = false;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());  // offset
        AscendC::LoadData(dstTensor.data()[dstOffset],           // built-in tensor
                          srcTensor.data(),                      // built-in tensor
                          loadDataParams);                       // srcTensor.coord represented in params
    }
}
```

2. Specialized TileCopyTla for MxScale movement (see the MxScale section above)

##### L0CToUB

1. Added support for L0CToUB data path: The instruction supports three modes for L0CToUB movement: single-destination, dual-destination M split, and dual-destination N split.

| dualDstCtrl | Input| Dual-destination control<br>`2'b00`: single-destination mode. The entire matrix is written to the target UB configured by using the `subBlockId` parameter.<br>2'b01: dual-destination mode. The matrix is split along the M dimension, and `M / 2 * N` is written to AIV. M must be a multiple of 2. On-the-fly quantization is not supported.<br>`2'b10`: dual-destination mode. The matrix is split along the N dimension, and `M * N / 2` is written to AIV. N must be a multiple of 2. On-the-fly quantization is not supported.<br>`2'b11`: Reserved.|
| ----------- | -- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| subBlockId  | Input| ID of the destination UB when the single-destination mode is enabled.                                                                                                                                                                      |

CATLASS provides a corresponding adaptation:

```java
enum class L0CCopyToUbMode {
    NO_SPLIT = 0,
    SPLIT_M,
    SPLIT_N,
    RESERVED
};
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class L0CCopyMode, // New template parameter
    class Enable = void>
struct CopyL0CToUBTla {
    // ....
    };
```

2. Added CopyL0CToDst (optional): On the 950 platform, L0C can be moved to either GM or UB. To keep the TileCopy interface consistent, the corresponding tile encapsulation is designed as CopyL0CToDst. When calling a sample, the specific TileCopy implementation (CopyL0CToGm or CopyL0CToUB) is declared. The destination address space (GM or UB) is allocated at the kernel layer and passed to the corresponding BlockMmad.

```c++
using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
```

##### UBToL1

The 950 platform added the UBToL1 data path, with corresponding movement adaptations:

```c++
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct CopyUb2L1Tla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyUb2L1Tla, can not find the specialization.");
};

/// Partial specialization for Atlas950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value &&
                     tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        // Implementation
    };
};
```

##### Mmad

The instruction interface has changed: the L0A input format changed from zZ to zN. CATLASS has been adapted accordingly.

![](<../../assets/images/CATLASS new version capability introduction-image-3.png>)

Macros are used in the code to isolate minor architectural differences:

```c++
template <
    /// Tag indicating architecture
    class ArchTag,
    /// Element for A matrix operand
    class ElementA,
    /// LayoutTag for A matrix operand in L1
    class LayoutTagL1A
>
struct TileMmadTla {
    // Methods

    CATLASS_DEVICE
    TileMmadTla() {}

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr(std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif

        AscendC::Mmad(l0CTensor.data(),
                      l0ATensor.data(),
                      l0BTensor.data(),
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB, class TensorBias>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         TensorBias const &l0BiasTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CTensor.data(),
                      l0ATensor.data(),
                      l0BTensor.data(),
                      l0BiasTensor.data(),
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         uint32_t l0Batch)
    {
        const uint32_t L0AM = tla::get<0, 0>(l0ATensor.shape()) * tla::get<0, 1>(l0ATensor.shape());
        const uint32_t L0AK = tla::get<1, 0>(l0ATensor.shape()) * tla::get<1, 1>(l0ATensor.shape());
        const uint32_t L0BK = tla::get<0, 0>(l0BTensor.shape()) * tla::get<0, 1>(l0BTensor.shape());
        const uint32_t L0BN = tla::get<1, 0>(l0BTensor.shape()) * tla::get<1, 1>(l0BTensor.shape());
        const uint32_t L0CM = tla::get<0, 0>(l0CTensor.shape()) * tla::get<0, 1>(l0CTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(l0CTensor.shape()) * tla::get<1, 1>(l0CTensor.shape());

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if (defined (__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif
        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendC::Mmad(l0CTensor.data()[l0BatchIdx * L0CM * L0CN],
                l0ATensor.data()[l0BatchIdx * L0AM * L0AK],
                l0BTensor.data()[l0BatchIdx * L0BK * L0BN],
                mmadParams);
        }
    }
};
```

### New Features at the Block Layer

#### Added BlockPrologue

A new prologue module supports the UB → L1 data path and operations such as dequantization.

```c++
template <
    class SrcType_,
    class DstType_,
    class TileElemWisePrologue_,
    class TileCopy_>
class BlockPrologue <
    PrologueElemWiseOneSource,
    SrcType_,
    DstType_,
    TileElemWisePrologue_,
    TileCopy_> {
//...
};
```

#### New Features in BlockMmad

With the new UB → L1 data path on the platform, the source for BlockMmad can be either GM or UB. Its input can be GlobalTensor or LocalTensor. Whether A/B/C are GlobalTensor or LocalTensor is determined at the kernel layer when constructing the tensor.

#### New Features in BlockEpilogue

##### New Data Path Adaptations

On the 950 platform, the destination of L0C movement can be GM, UB, or L1, and the corresponding address space can be **allocated as needed**. It can be allocated at the kernel layer and passed in, or allocated in a stage such as the epilogue and then returned. The current implementation primarily allocates it at the kernel layer and passes it in.

##### Regbase and SIMT

Epilogue template parameters still use the original design, and the internal operator implementation uses the corresponding micro-instruction programming. This is largely transparent at the CATLASS design level. Simply add `__simd_vf__` or `__simt_vf__` before the corresponding function calls and use the corresponding instruction interfaces.

```c++
__simd_vf__ inline void FlashUpdate(__ubuf__ T *updateUb,  __ubuf__ T *curUb, __ubuf__ T *expMaxUb,
 uint16_t m, uint16_t nLoops, uint32_t tailN)
{
    RegTensor<float> expMaxVreg;
    RegTensor<float> preSrcVreg;
    RegTensor<float> curSrcVreg;
    RegTensor<float> mulVreg;
    RegTensor<float> addVreg;

    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTailN = UpdateMask<float>(tailN);

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_BRC_B32>(expMaxVreg, expMaxUb + i);
        for (uint16_t j = 0; j < nLoops; ++j) {
            AscendC::MicroAPI::LoadAlign(preSrcVreg, updateUb + i * DBaseSize + j * FLOAT_REP_SIZE);
            AscendC::MicroAPI::LoadAlign(curSrcVreg, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
            AscendC::MicroAPI::Mul(mulVreg, expMaxVreg, preSrcVreg, pregFull);
            AscendC::MicroAPI::Add(addVreg, mulVreg, curSrcVreg, pregFull);
            AscendC::MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(
                updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, addVreg, pregFull);
        }
        AscendC::MicroAPI::LoadAlign(preSrcVreg, updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
        AscendC::MicroAPI::LoadAlign(curSrcVreg, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
        AscendC::MicroAPI::Mul(mulVreg, expMaxVreg, preSrcVreg, pregTailN);
        AscendC::MicroAPI::Add(addVreg, mulVreg, curSrcVreg, pregTailN);
        AscendC::MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(
            updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, addVreg, pregTailN);
    }
}
```

### New Features at the Kernel Layer

Kernel interfaces now accept prologue template parameters.

```c++
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class BlockPrologueA_,
    class BlockPrologueB_
>
class MatmulKernel {
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    BlockMmad blockMmad(resource);
    BlockPrologueA prologueA(resource);
    // Used at the kernel layer
    auto tensorA = prologueA.GetMmadSrcTensor(actualBlockShape);
    blockmmad(tensorA,x,x);
}
```

#### Inter-Core Synchronization

A new synchronization instruction, `mode 4`, has been added to the instruction interface. It supports synchronization between AIC and a specified AIV. CATLASS also supports this feature.

### New Features at the Device Layer

The 950 platform has no `ffts_address` parameter. The device layer no longer includes this input argument at this generation, and the existing design already supports it.

# Extended Interface Capabilities

## New TLA Interfaces

### TLA Tensor Definition

TLA (Tensor Layout Abstraction), as a data structure in CATLASS, provides an encapsulation of basic data structures to offer more convenient access for matrix computation-related interfaces. In essence, a tensor represents a multi-dimensional array, abstracting the details of how array elements are organized and stored in memory. This allows users to write generic algorithms for accessing multi-dimensional arrays and to specialize those algorithms based on tensor traits, such as depth, rank, layout, data type, and position.

- A tensor contains four template parameters: `BuiltinTensor`, `Layout`, `Coord` and `Position`.
  - `BuiltinTensor` is the underlying storage object itself, which is `GlobalTensor` or `LocalTensor` in AscendC.
  - `Layout` describes how logical coordinates map to memory and how the logical valid range is expressed. It includes `shape`, `stride`, and `origin_shape`.
  - `Position` is a position tag in Ascend C, such as `Arch::PositionGM{}` and `Arch::PositionL1{}`. It distinguishes which level of storage (GM, L1, L0, etc.) the data resides in.
  - `Coord` represents the address offset of the tensor relative to the original representation, with the offset in units of elements. The newly added Coord adapts to the on-chip movement interface of the 950 platform, supports different computation levels, and can be adapted to previous generations of implementations.

Starting from the 950 platform, **all new samples in CATLASS are implemented based on TLA**.

![Differences between shape and originShape in Layout. shape represents the layout semantics, including alignment information. originShape represents the original logical semantics of the matrix, indicating the valid data range. In the figure, each small grid represents four elements.](<../../assets/images/CATLASS new version capability introduction-image-4.png>)

### New TLA Interfaces and Their Design Objectives

These new interfaces provide tile-granularity partitioning semantics, hide tail tile processing, and simplify programming. The semantics is similar to `local_tile`. Major interfaces and their functions are as follows:

- `MakeTensor`: Accepts information such as `shape` and `coord` to construct a `tla::Tensor`. Creates a logical view and does not perform data movement itself.
- `MakeTensorLike`: Accepts `like_tensor` as input to facilitate the construction of `tla::Tensor` across different storage levels. This interface only binds an existing storage to a new view whose logical size is the same as that of the reference tensor. It does not perform data movement.
- `GetTile`: Obtains information about the current tile, with the offset in the unit of elements.
- `TileView`: Obtains information about the current tile, with the offset in the unit of tiles.

### Major Interfaces

#### MakeTensor

Currently, the `MakeTensor` interface is provided to construct `Tensor`. You can specify Coord or use the default Coord of (0, 0).

```c++
using namespace tla;

GlobalTensor<float> A = ...;
auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);

auto tensorA = MakeTensor(A, layout, Arch::PositionGM{});
// tensorA.coord() == (0, 0)

auto tensorA_sub = MakeTensor(A, layout, MakeCoord(1, 5), Arch::PositionGM{});
// tensorA_sub.coord() == (1, 5)

auto tileA = GetTile(tensorA_sub, MakeCoord(2, 4), MakeShape(4, 8));
// tileA.coord() == (3, 9)
```

It can also be constructed using VectorLayout.

```c++
// rank-1 VectorLayout example (1D vector)
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(A, v1024, Arch::PositionGM{});
```

![](<../../assets/images/CATLASS new version capability introduction-image-5.png>)

#### Underscore Semantics (_)

TLA `Tensor` supports indexing using `operator()` and supports using `tla::_` to express a full slice, thereby returning a **sub-tensor view** (without copying data). The basic rules are as follows:

- **No underscore**: `tensor(i, j, ...)` returns a **BuiltinTensor** (more accurately, the result of `tensor.data()[offset]`), whose base address is the start address corresponding to the element coordinates, instead of directly returning the element value.
- **With underscore**: `tensor(..., tla::_, ...)` returns a sub-tensor, whose dimensions are determined by the dimensions where the underscores are located. Coordinates are limited to one level: Each dimension of coord must be a scalar (or `tla::_`). Nested tuples cannot be used as coord elements.

For the dimensions of the output tensor, suppose the input tensor has rank R, and the set of dimension indices where underscores appear in coord is `{d_0, d_1, ..., d_{k-1}}` (preserving the original order). The output tensor has rank k. The `layout.shape()`, `stride()`, and `origin_shape()` of the output tensor are the projections of the input layout onto these dimensions (taken in the order of {d_0..d_{k-1}}). The `coord()` of the output tensor is initialized to all zeros.

For example, for a 3D tensor `A(B, M, K)`:

```c++
auto A2 = A3(b, tla::_, tla::_);  // 3D -> 2D. The (M, K) view is obtained.
auto A1 = A2(r, tla::_) // 2D -> 1D. The (K) view is obtained.
```

![](<../../assets/images/CATLASS new version capability introduction-image-6.png>)

#### MakeTensorLike

`MakeTensorLike` is used to create a new tensor view with the same logical dimensions as `likeTensor`. `MakeTensorLike` points to a pre-allocated built-in tensor. It reads `likeTensor.layout().origin_shape()` to obtain the logical dimensions:

- If the rank of `likeTensor` is 2, `(rows, cols)` is obtained.
- If the rank of `likeTensor` is 1, `(len)` is obtained.

**Example**

```c++
// Scenario 1: Source and destination element types are the same
// This is the most common scenario. For example, when creating a corresponding L1 tensor from a half tile in GM, the element type remains unchanged, but the storage level changes.
auto tensorTileA = tla::TileView(
      tensorA,
      tla::MakeCoord(blockM, kTile),
      tla::MakeShape(L1_TILE_M, L1_TILE_K)
);
auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensorList[l1ListId],
      tensorTileA,
      Arch::PositionL1{}
);
// Results:
// 1. tensorL1A uses the L1 target layout.
// 2. The originShape of tensorL1A is the same as that of tensorTileA.
// 3. The element type is automatically inferred from likeTensor.


// Scenario 2: Target element type is different
// If the element type of the target tensor is different from that of the source tensor, you need to explicitly specify ElementDst.
auto tensorL0C = tla::MakeTensorLike<LayoutTagL0C, float>(
      l0cTensor,
      tensorTileC,
      Arch::PositionL0C{}
);

// Results:
// 1. The logical size of tensorL0C is inherited from tensorTileC.
// 2. The target element type is explicitly float.
// 3. This applies to the accumulator or type promotion scenarios.


// Scenario 3: Additional control over the target layout required
auto layoutBaseL1A = tla::MakeLayout<half, LayoutTagL1A>(L1_TILE_M, L1_TILE_K);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensor,
      tensorTileA,
      Arch::PositionL1A{},
      layoutBaseL1A
);

// Results:
// 1. The shape/stride of tensorL1A is from layoutBaseL1A.
// 2. The originShape of tensorL1A is inherited from tensorTileA in GM.
// 3. Even if the current tile is a tail tile, the logical valid range is not lost.
```

#### GetTile

The `GetTile` interface is used to obtain a TileTensor. `GetTile` helps obtain a tile view from the parent tensor without copying data. Here, coord is an element coordinate: the resulting tensor's `coord()` will be the parent tensor's coordinate plus this offset. The resulting `layout()` specifies the expected tile dimensions (rows/cols) via `tileShape` and, if necessary, converts it into a corresponding `shape()` according to the parent layout's structure. At the same time, based on the parent tensor's `origin_shape()`, a new `origin_shape()` is automatically clipped to express the tail tile (the actual logical dimensions at the boundary).

Function signature:

```c++
template <class Tensor, class Coord, class Shape>
auto GetTile(Tensor const& tensor,
             Coord const& coord,   // Element coordinates (not tile coordinates). The rank must be the same as that of tensor.rank.
             Shape const& shape);  // tileShape: the expected size for memory layout calculation. The rank must be the same as that of tensor.rank.
// Currently, Tensor::rank == 1 or Tensor::rank == 2 is supported.
// Currently, both coord and shape can be one-layer tuples (depth == 1).
```

Sample code:

```c++
AscendC::GlobalTensor<float> gmA;
auto w8xh16 = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);
Tensor tensor_8x16 = MakeTensor(gmA, w8xh16, Arch::PositionGM{});

// coord indicates the element coordinates. GetTile automatically handles boundary conditions.
auto tensor_tile = GetTile(tensor_8x16, tla::MakeCoord(2, 4), MakeShape(4, 8));

// tensor_tile.layout().shape() returns the dimensions (4, 8) used for memory layout.
// tensor_tile.layout().origin_shape() returns the actual logical dimensions (automatically calculated based on the tail tile).
```

Rank-1 VectorLayout also supports `GetTile`:

```c++
AscendC::GlobalTensor<float> gmA;
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(gmA, v1024, Arch::PositionGM{});

// coord is an element coordinate (1D).
auto vec_tile = GetTile(vec, tla::MakeCoord(100u), tla::MakeShape(256u));
// vec_tile.layout().shape() == (256)
// vec_tile.layout().origin_shape() == (min(256, 1024-100))
```

#### TileView

Similar to `GetTile`, `TileView` obtains information about TileTensor. Unlike GetTile, the input coordinate of TileView is in **tile coordinate** instead of element coordinates.

**Function signature and construction example:**

```c++
template <class TensorT, class TileCoord, class TileShape>
auto TileView(TensorT const& tensor,
               TileCoord const& tileCoord,  // Tile Unit coordinates (not element coordinates)
               TileShape const& tileShape); // Tile size used for memory layout
```

**Example**

```c++

template <class TensorA, class TensorB, class TensorC>
CATLASS_DEVICE
void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC)
{
    //... Pre-loop logic
    // Main loop
    // Get the number of iterations in the K direction
    uint32_t kTileCount = CeilDiv<L1_TILE_K>(tla::get<1>(tensorA.origin_shape()));  // dim 1 = K
    for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
        uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
        uint32_t kLoopIdxNext = kLoopIdx + 1;
        // Get L1 tensor for next stage
        auto l1ATensor = l1ATensorList[l1ListIdNext];
        auto l1BTensor = l1BTensorList[l1ListIdNext];
        // Get GM tile for next stage
        auto tensorTileA = tla::TileView(tensorA,
                                           tla::MakeCoord(0, kLoopIdxNext),  // (m_tile, k_tile)
                                           tla::MakeShape(Int<L1_TILE_M>{}, Int<L1_TILE_K>{}));
        auto tensorTileB = tla::TileView(tensorB,
                                           tla::MakeCoord(kLoopIdxNext, 0),  // (k_tile, n_tile)
                                           tla::MakeShape(Int<L1_TILE_K>{}, Int<L1_TILE_N>{}));
        auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(l1ATensor, tensorTileA, Arch::PositionL1{}, L1A_LAYOUT);
        auto tensorL1B = tla::MakeTensorLike<LayoutTagL1B>(l1BTensor, tensorTileB, Arch::PositionL1{}, L1B_LAYOUT);

        // Load next matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
        copyGmToL1A(tensorL1A, tensorTileA);
        // ... Post-loop logic
    }
}
```

`TileView` is also supported for rank-1 VectorLayout:

```c++
AscendC::GlobalTensor<float> gmA;
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(gmA, v1024, Arch::PositionGM{});

// tileCoord is a tile-unit coordinate; tileShape is the tile dimensions (number of elements)
auto vec_tile2 = tla::LocalTile(vec, tla::MakeCoord(3), tla::MakeShape(256));
// Equivalent to GetTile(vec, MakeCoord(3*256), MakeShape(256))
```

**Design pattern**: The following patterns are recommended at the block and kernel layers:

1. Use `TileView` to create logical tiles and automatically handle boundary conditions.

2. Use `MakeTensorLike` to create tensors at different positions, automatically inheriting `origin_shape`.

## EVG

Epilogue Visitor Graph (EVG) is a declarative framework for GEMM epilogues. It abstracts epilogue operations (such as addition, type conversion, broadcasting, and reduction) into composable template nodes, assembling them into a computation graph using a tree or topological structure.

Developers only need to declare the computation logic using "expressions" (e.g., `D = C + X`). The framework automatically handles data movement, UB space allocation, event synchronization, and pipeline scheduling.

Compared to manually organizing GM/UB copies and event synchronization, EVG significantly reduces development complexity while aiming to maintain comparable performance. It also supports graph and node reuse and flexible extension. Developers can use the predefined data and computation nodes under `Epilogue::Fusion` to implement complex, nested epilogue logic.

### Supported Data Structures

EVG supports two types of data structures: TreeVisitor and TopologicalVisitor, supporting scenarios with or without shared subexpressions.

- **TreeVisitor**: Supports tree-structured node composition. Easier to write and maintain. Suitable for scenarios without shared subexpressions.
- **TopologicalVisitor**: Supports Directed Acyclic Graph (DAG) topology. Uses DAG representation when an intermediate result is used by multiple subsequent nodes. Suitable for scenarios with shared subexpressions.

![](<../../assets/images/CATLASS new version capability introduction-image-7.png>)

EVG provides nodes with more abstract semantics that operate on UB. The nodes supported at different stages are as follows:

1. Load: Data is loaded from GM to UB, including operations such as `AccLoad` and `AuxLoad`.

2. Compute: Computations are performed in UB, including operations such as `Compute` and `Cast`.

3. Store: Results are written back to GM, including operations such as `AuxStore`.

### Sample Code

The addition operation `Epilogue::Fusion::Add` encapsulates `AscendC::Add`. When using EVG, you only need to declare the computation logic without worrying about the details of movement, events, or layout.

Sample code for TreeVisitor:

```c++
// ...
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/kernel/matmul_visitor.hpp"
#include "catlass/epilogue/fusion/fusion.hpp"

// Define EVG: D = C + X
// C is the workspace (result of A * B), and D is the final output (result of C + X)
// Allocate space. 3 indicates the number of nodes requesting space, and 2 indicates the number of buffers

using LayoutC = decltype(layoutC);
using EVG = Epilogue::Fusion::TreeVisitor<
    Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>,
    Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Add, ElementC>,  // Intermediate child node for computation
        Epilogue::Fusion::VisitorAccLoad<ElementC>,  // Left child node for loading C (workspace)
        Epilogue::Fusion::VisitorAuxLoad<ElementC, LayoutC>   // Right child node for loading X
    >
>;
```

Sample code for TopoligicalVisitor:

```c++
// Node order
// 0-AccLoad, 1-Compute1(2X), 2-Compute2(Exp(2X)),
// 3-Compute3(Exp(2X) + 1), 4-Compute4(Exp(2X) - 1), 5-Compute5(Compute3 / Compute4), 6-Store
using Edges = tla::tuple<
    tla::seq<>,         // 0: AccLoad has no child nodes
    tla::seq<0>,        // 1: Depends on AccLoad-->2X
    tla::seq<1>,        // 2: Depends on Compute1-->Exp(2X)
    tla::seq<2>,        // 3: Depends on Compute2-->(Exp(2X) - 1)
    tla::seq<2>,        // 4: Depends on Compute2-->(Exp(2X) + 1)
    tla::seq<3, 4>,     // 5: Depends on Compute3 and Compute4-->(Compute3/Compute4)
    tla::seq<5>         // 6: Store depends on Compute5
>;

using EVG = Epilogue::Fusion::TopologicalVisitor<
    Edges,
    Epilogue::Fusion::VisitorAccLoad<ElementC>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Muls, ElementC, ElementC>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Exp, ElementC>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Adds, ElementC, ElementC>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Adds, ElementC, ElementC>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Div, ElementC>,
    Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>
>;

```
