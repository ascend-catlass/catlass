# 概述

Ascend950平台在硬件特性上，新增了多项内容。其中与CATLASS关联的特性主要有

- 新增支持MX Scaling
- 新增L0C→UB通路，并支持多搬运模式
- 新增支持UB→L1通路
- 片上数据搬运支持传入Coord
- 支持Regbase与SIMT计算

为扩展支持新平台兼容旧平台并进一步提升易用性，CATLASS发布新版，相关新增能力主要为两方面

- 适配950对应的硬件新增能力，如支持新增通路，扩展BlockEpilogue，支持CopyL0CToDst，新增BlockPrologue等
- 扩展TLA(Tensor Layout Abstraction)的相关接口，新增支持EVG(Epilogue Visitor Graph)功能

# CATLASS针对950新增能力适配

## MxScaling支持

**mx量化计算逻辑**

mxScaling是一种pergroup量化的设计，在Reduce轴(K轴)上每个group组共享一个量化系数。在950平台每32个element对应一个scale。具体指令如下，ScaleA通过broadcasting multiplication与A相乘，ScaleB同理与B相乘，其结果相乘然后累加到C上

$$C=(\mathrm{ScaleA}\otimes A)*(\mathrm{ScaleB}\otimes B)+C$$

如下左图所示，一个group内的两个蓝色的element对应量化参数的一个蓝色的scale输入。以蓝色块为例，一个L0A的(16,32)和L0B的(32, 16)分别叠加上scaleA的(16,1)和scaleB的(1, 16)，相乘计算得到(16, 16)的计算结果

![](<../../assets/images/CATLASS 新版本能力介绍-image-10.png>)

**搬运部分：**

对于mxFP8/FP4来说，硬件和对应的指令接口有两个约束

1. scale在L0上的排布是固定的scaleA是zZ，scaleB是nN，对于trans/notrans都是如此

2. scale在L1->L0的搬运过程中不支持转置

所以这就决定了在数据搬运的时候的设计

- 对L1->L0的搬运，需要满足对应的分形要求，所以L1上也需要是对应的zZ/nN的排布
- 因为scale在L0上排布固定两个element连续，所以在GM和L1上K方向上两个元素必须连续，才能完成从GM到L1的搬运。所以在GM向L1搬运的时候，需要将两个fp8打包作为一个fp16进行搬运(对于mx数据类型scale都是fp8e8m0)，并且使用DN2NZ的搬运接口

对于数据搬运来说，GMToL1是两个fp4打包成一个int8进行搬运，对应封装由指令接口完成

![mxFP8在L0A和L0B的排布](<../../assets/images/CATLASS 新版本能力介绍-image-9.png>)

![scale在GM和L1上的对应排布(RowMajor情况)](<../../assets/images/CATLASS 新版本能力介绍-image-8.png>)

针对这部分，CATLASS的新增接口设计主要包括

1. MakeMxScaleLayout：用于构造MxScale输入的排布

    ```c++
    // Make a MxScale layout with Rows and Cols.
    template <class Element,   // 输入数据类型
            class LayoutTag, // 输入排布类型，支持row/col/zZ/nN
            bool isMxScaleB, // 说明是左矩阵或右矩阵
            class T,         // rows/cols数据类型，同时支持动静态数据类型
            class U>
    CATLASS_HOST_DEVICE constexpr
    auto MakeMxScaleLayout(T const& rows, U const& cols)
    ```

2. GMToL1和L12L0对MxScale相关类型进行了特化TileCopyTla

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
            // 具体实现
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
        // ... 具体实现
        }
    }
    ```

3. PackedMxTileCopyTla：用于mxscale搬运的TileCopy封装

    ```c++
    template <
        /// Tag indicating architecture
        class ArchTag,
        class ElementA_,
        class LayoutTagA,
        class ElementB_,
        class LayoutTagB,
        class ElementMxScaleA_,      //描述MxScaleA/B的类型和layout
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
        //具体实现
        }
    ```

    MXFP特性支持的数据流

    | 输入/输出     | 支持数据类型                                   |
    | ------------- | ---------------------------------------------- |
    | A/B           | fp8\_e5m2/fp8\_e4m3 或 fp4x2\_e1m2/fp4x2\_e2m1 |
    | scaleA/scaleB | fp8\_e8m0                                      |
    | L0C           | fp32                                           |
    | C             | fp32/fp16/bf16                                 |

**计算部分：**

mxmmad的指令接口对累加轴的大小有要求，需要是K是64的倍数，实际的actualK需要向上取整，取整的部分要在GM->L1的时候要置零，对于RowMajor和ColumnMajor置零的情况有所不同

以fp8为例，搬运的时候会对内轴方向32Byte对齐的位置自动置零，以(32,30)的搬入为例

1. rowmajor下，K方向是内轴，(32, 30)下的最后两个数据会自动补0，只需对后半段补0

2. columnmajor下，M方向是内轴，(32, 30)搬到L1的时候，K方向就不会自动置0了，所以K方向有34列数据要置零
   ![左图为RowMajor示意图，K轴自动补0; 右图为ColumnMajor示意图，K轴需显式补0|697](<../../assets/images/CATLASS 新版本能力介绍-zeropadding.png>)

```c++
// Init Zero for k axis
InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, kL1ActualNext));
```

## 通路适配情况

因为CATLASS是分层设计，故按照CATLASS分层的Tile->Block->Kernel->Device介绍新增特性

![](<../../assets/images/CATLASS 新版本能力介绍-image.png>)

### Tile层新增特性

#### GMToL1

1. 数据搬运：新增支持DN2NZ的搬运

    ![](<../../assets/images/CATLASS 新版本能力介绍-image-1.png>)

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
        // 其他逻辑
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
        // 其他逻辑
        // ... ...
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
        }
    }
    ```

2. 扩展支持对fp4数据的搬运

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
        // ...其他逻辑
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        //两个fp4作为一个b8数据类型进行搬运
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
        //...其他逻辑
    }
    ```

3. 对MxScale的搬运TileCopyTla进行了特化（见上述MxScale部分）

##### L1ToL0

1. 增加Coord描述：指令能力上，L1上新增支持按Coord坐标描述tensor的能力。以L12L0A为例。在Coord方式下，通过BuiltinTensor+Coord来表示实际的内存地址。如下图所示，左边Tensor的大矩阵和小矩阵的BuiltinTensor(dataptr)指向同一个地址，通过Coord的差别获取偏移

    ![](<../../assets/images/CATLASS 新版本能力介绍-image-2.png>)

    | 参数名称       | 含义（以M\*K矩阵为例）                                                                                                                         |
    | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
    | mStartPosition | 源矩阵M轴方向的起始位置，单位为16 element                                                                                                      |
    | kStartPosition | 源矩阵K轴方向的起始位置，单位为32B                                                                                                             |
    | mStep          | 源矩阵M轴方向搬运长度，单位为16 element。取值范围：mStep∈\[0, 255]                                                                             |
    | kStep          | 源矩阵K轴方向搬运长度，单位为32B，取值范围：nStep∈\[0, 255]                                                                                    |
    | srcStride      | 源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位为512B                                                                            |
    | dstStride      | 目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位为512B                                                                          |
    | ifTranspose    | 是否启用转置功能，对每个分形矩阵进行转置，默认为false;true启动, false不启用。使用转置功能时，源操作数、目的操作数支持b4/b8/b16/b32的数据类型。 |

    CATLASS在原有Tensor表示上增加Coord，表示待搬运Tensor与BuiltinTensor的关系

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
                            loadDataParams);                       // srcTensor的coord在params体现
        }
    }
    ```

2. 对MxScale的搬运TileCopyTla进行了特化（见上述MxScale部分）

##### L0CToUB

1. 新增L0CToUB数据通路支持：指令能力上支持L0CToUB的3种模式，支持单目标模式/双目标M模式/双目标N模式

    | dualDstCtrl | 输入 | 双目标控制模式<br />2'b00：单目标模式，将整个矩阵写入通过subBlockId参数配置的目标UB<br />2'b01：双目标模式，按M维度进行拆分，M / 2 \* N写入AIV，M必须是2的倍数。不支持随路量化<br />2'b10：双目标模式，按N维度进行拆分，M \* N / 2写入AIV，N需为2的倍数。不支持随路量化<br />2'b11：保留 |
    | ----------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | subBlockId  | 输入 | 在启用单目标模式时指示目标UB的编号                                                                                                                                                                                                                                                       |

    故CATLASS对此做了相应适配设计

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
        class L0CCopyMode // 新增的模板参数
        class Enable = void>
    struct CopyL0CToUBTla {
        // ....
        };
    ```

2. 新增CopyL0C2Dst(可选)：因为在950平台，L0C可搬运至GM或者UB，为保持TileCopy接口一致，设计对应的Tile封装为CopyL0CToDst，并在example调用指定TileCopy时声明具体实现为CopyL0CToGm还是CopyL0CToUB。搬运的目的地址空间（GM or UB）在kernel层做对应申请，并传入对应的BlockMmad

    ```c++
    using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
    ```

##### UBToL1

950新增UBToL1通路，增加对应搬运适配

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
        // 具体实现
    };
};
```

##### Mmad

指令接口上L0A输入格式由zZ转变为zN，CATLASS进行了相关适配

![](<../../assets/images/CATLASS 新版本能力介绍-image-3.png>)

在代码中对少量代际差异使用宏进行隔离

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

### Block层新增特性

#### 新增BlockPrologue

新增prologue模块，支持UB->L1通路，支持添加反量化等操作

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
}；
```

#### BlockMmad新增特性

平台新增UB->L1的数据通路，BlockMmad传入源可以在GM或UB，其入参支持为GlobalTensor或LocalTensor，A/B/C是GlobalTensor或LocalTensor在Kernel层构建Tensor时确定

#### BlockEpilogue新增特性

##### 新增数据通路适配

在950平台，L0C搬出的目的地址可以为GM/UB/L1，对应地址空间**可按需申请**。可在Kernel申请后传入，也可以在对应阶段比如Epilogue申请然后返回。当前实现主要为Kernel层申请后传入

##### Regbase与SIMT

epilogue部分模板参数仍使用原有设计，内部operator实现部分使用对应微指令编程。CATLASS设计层面基本无感知，可在对应函数调用前增加\_\_simd\_vf\_\_和\_\_simt\_vf\_\_并使用对应指令接口即可

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

### Kernel层新增特性

Kernel接口新增Prologue模板参数传入实现

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
    // 在kernel层使用
    auto tensorA = prologueA.GetMmadSrcTensor(actualBlockShape);
    blockmmad(tensorA,x,x);
}
```

#### 核间同步

指令接口上新增同步指令mode4，支持AIC与指定AIV进行同步，CATLASS也支持此功能

### Device层新特性

950平台无`hardwareSyncAddress`参数，Device层在这一代际无此入参，已有设计已可支持

# 新增接口能力扩展

## TLA新增接口

### TLA Tensors定义

TLA(Tensor Layout Abstraction)作为CATLASS的一种数据结构，提供了对基础数据结构的封装，以提供更便捷的矩阵计算相关计算接口访问。本质上，`Tensor` （张量）表示一个多维数组，抽象了数组元素在内存中的组织方式与存储方式的细节。这使得用户能够编写通用的访问多维数组的算法，并可根据张量的特性（traits）对算法进行特化。如张量的depth、rank、layout、数据的类型、位置等。

- `Tensor` 包含4个模板参数: `BuiltinTensor`、 `Layout`、 `Coord`、 `Position`
  - `BuiltinTensor` 底层存储对象本身，为AscendC内的 `GlobalTensor` 或者 `LocalTensor`
  - `Layout`描述逻辑坐标如何映射到内存，以及逻辑有效范围如何表达。包含shape，stride，origin\_shape
  - `Position` 为AscendC 中的位置标签，例如 `Arch::PositionGM{}`、`Arch::PositionL1{}`。它用于区分数据位于 GM、L1、L0 等哪一层存储
  - `Coord`表示Tensor相对原始表示的地址偏移，偏移量以元素个数为单位。新增Coord为适配950平台片上搬运接口，可支持不同计算层级，可适配以往代际实现

自950平台起，CATLASS**所有新增样例基于TLA实现**

![Layout中的shape和originShape的区别。shape表示布局语义，包含对齐信息。originShape表示矩阵原始的逻辑语义，指有效数据范围。图中一个小格子是4个element](<../../assets/images/CATLASS 新版本能力介绍-image-4.png>)

### TLA新增接口及其设计目标

提供Tile块粒度的切分语义，隐藏尾块处理，提升编程易用性，语义类似local\_tile。主要接口及其功能如下：

- MakeTensor：接受shape/coord等信息，构造tla::Tensor。创建的是逻辑视图，本身不执行数据搬运
- MakeTensorLike：接受like\_tensor作为输入，方便不同储存层级间构造tla::Tensor。该接口是把一块已有存储绑定成“与参考 Tensor 逻辑尺寸一致”的新视图，本身不执行数据搬运
- GetTile：获取当前tile块信息，以元素个数为单位偏移
- TileView：获取当前tile块信息，以tile块个数为单位偏移

### 主要接口

#### MakeTensor

当前提供 `MakeTensor` 接口构造`Tensor`，支持指定Coord传入或默认以Coord(0, 0)方式构造：

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

同时也支持以VectorLayout进行构造

```c++
// rank-1 VectorLayout 示例（一维向量）
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(A, v1024, Arch::PositionGM{});
```

![](<../../assets/images/CATLASS 新版本能力介绍-image-5.png>)

#### 下划线语义(\_)

TLA `Tensor`支持使用 `operator()` 进行索引，并支持用 `tla::_` 表达“整维切片”（full slice），从而返回一个 **子 Tensor 视图**（不拷贝数据）。使用的基本规则如下

- **无下划线**：`tensor(i, j, ...)` 返回一个 **BuiltinTensor**（更准确地说是 `tensor.data()[offset]` 的结果），其基址为该元素坐标对应的起始地址，而不是直接返回元素值。
- **带下划线**：`tensor(..., tla::_, ...)` 返回一个子 Tensor，子 Tensor 的维度由下划线所在的维度决定。这里坐标仅支持“一层”：coord 的每个维度必须是标量（或 `tla::_`），不支持嵌套 tuple 作为 coord 元素。

在输出 Tensor 的维度上，设输入张量 rank 为 (R)，在 coord 中出现的下划线维度索引集合为 ({d\_0, d\_1, ..., d\_{k-1}})（保持原顺序），则输出 Tensor 的 rank 为 (k)，输出 Tensor 的 `layout.shape()` `stride()` `origin_shape()`为输入 layout 在这些维度上的投影（按 ({d\_0..d\_{k-1}}) 依次取出），输出 Tensor 的 `coord()` 初始化为全 0

例如，对 3D 张量 `A(B,M,K)`：

```c++
auto A2 = A3(b, tla::_, tla::_);  // 3D -> 2D，得到 (M, K) 视图
auto A1 = A2(r, tla::_) // 2D -> 1D，得到 (K)视图
```

![](<../../assets/images/CATLASS 新版本能力介绍-image-6.png>)

#### MakeTensorLike

`MakeTensorLike` 用于创建一个与 `likeTensor` 逻辑尺寸一致的新 tensor 视图，MakeTensorLike会指向预先申请的built-in Tensor。它读取 `likeTensor.layout().origin_shape()` 得到逻辑尺寸：

- 若 likeTensor rank=2：得到 `(rows, cols)`
- 若 likeTensor rank=1：得到 `(len)`

**使用示例**：

```c++
//场景一：源和目标元素类型相同
//最常见的场景。例如从 GM 中的一个 half tile 创建对应的 L1 Tensor，元素类型不变，只是存储层级改变
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
// 结果：
// 1. tensorL1A 使用 L1 目标布局
// 2. tensorL1A 的 originShape 与 tensorTileA 相同
// 3. 元素类型从 likeTensor 自动推断


//场景二：目标元素类型不同
//当目标 Tensor 的元素类型与源 Tensor 不一致时，需要显式指定 ElementDst
auto tensorL0C = tla::MakeTensorLike<LayoutTagL0C, float>(
      l0cTensor,
      tensorTileC,
      Arch::PositionL0C{}
);

// 结果：
// 1. tensorL0C 的逻辑尺寸继承自 tensorTileC
// 2. 目标元素类型显式为 float
// 3. 适用于 accumulator 或类型提升场景


//场景三：需要额外控制目标布局
auto layoutBaseL1A = tla::MakeLayout<half, LayoutTagL1A>(L1_TILE_M, L1_TILE_K);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensor,
      tensorTileA,
      Arch::PositionL1A{},
      layoutBaseL1A
);

// 结果：
// 1. tensorL1A 的 shape/stride 来自 layoutBaseL1A
// 2. tensorL1A 的 originShape 继承自GM上的 tensorTileA
// 3. 即使当前 tile 是尾块，逻辑有效范围也不会丢失
```

#### GetTile

`GetTile` 接口用来获取TileTensor。`GetTile` 用于从父 tensor 上切出一个 tile **视图**（不拷贝数据）。其中 `coord` 是**元素坐标**：返回 tensor 的 `coord()` 会在父 tensor 的基础上加上该偏移；返回的 `layout()` 以 `tileShape` 指定 tile 的期望尺寸（rows/cols），并在需要时按父 layout 的结构转换成对应的 `shape()`；同时根据父 tensor 的 `origin_shape()` 自动裁剪新的 `origin_shape()` 来表达 tail tile（边界处的实际逻辑尺寸）。

函数签名

```c++
template <class Tensor, class Coord, class Shape>
auto GetTile(Tensor const& tensor,
             Coord const& coord,   // 元素坐标（不是 tile 坐标），rank 与 tensor.rank 一致
             Shape const& shape);  // tileShape：用于内存布局计算的“期望尺寸”，rank 与 tensor.rank 一致
//当前支持Tensor::rank == 1 或 Tensor::rank == 2
//当前支持coord 与 shape 都为一层 tuple(depth==1)
```

使用代码示例

```c++
AscendC::GlobalTensor<float> gmA;
auto w8xh16 = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);
Tensor tensor_8x16 = MakeTensor(gmA, w8xh16, Arch::PositionGM{});

// coord 是元素坐标，GetTile 会自动处理边界情况
auto tensor_tile = GetTile(tensor_8x16, tla::MakeCoord(2, 4), MakeShape(4, 8));

// tensor_tile.layout().shape() 返回用于内存布局的尺寸 (4, 8)
// tensor_tile.layout().origin_shape() 返回实际逻辑尺寸（自动根据尾块情况进行计算）
```

同时对 rank-1 VectorLayout 也支持 `GetTile`：

```c++
AscendC::GlobalTensor<float> gmA;
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(gmA, v1024, Arch::PositionGM{});

// coord 是元素坐标（一维）
auto vec_tile = GetTile(vec, tla::MakeCoord(100u), tla::MakeShape(256u));
// vec_tile.layout().shape() == (256)
// vec_tile.layout().origin_shape() == (min(256, 1024-100))
```

#### TileView

`TileView`与 `GetTile` 类似，提供了获取TileTensor的功能。与GetTile不同的是，TileView输入坐标语义是 **tile 坐标**，而非元素坐标

**函数签名与构造示例**

```c++
template <class TensorT, class TileCoord, class TileShape>
auto TileView(TensorT const& tensor,
               TileCoord const& tileCoord,  // tile 单位坐标（不是元素坐标）
               TileShape const& tileShape); // 用于内存布局的 tile 尺寸
```

**使用示例**

```c++

template <class TensorA, class TensorB, class TensorC>
CATLASS_DEVICE
void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC)
{
    // ... 前置逻辑
    // main loop
    // 获取K方向循环次数
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

        // load next matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
        copyGmToL1A(tensorL1A, tensorTileA);
        // ....后续逻辑
    }
}
```

对 rank-1 VectorLayout 也支持 `TileView`：

```c++
AscendC::GlobalTensor<float> gmA;
auto v1024 = tla::MakeLayout<float, Catlass::layout::VectorLayout>(1024);
Tensor vec = MakeTensor(gmA, v1024, Arch::PositionGM{});

// tileCoord 是 tile 单位坐标；tileShape 是 tile 尺寸（元素数）
auto vec_tile2 = tla::LocalTile(vec, tla::MakeCoord(3), tla::MakeShape(256));
// 等价于 GetTile(vec, MakeCoord(3*256), MakeShape(256))
```

**设计模式**：在 block 层和 kernel 层，推荐使用以下模式：

1. 使用 `TileView` 创建 tile的逻辑切分，自动处理边界情况

2. 使用 `MakeTensorLike` 在不同位置创建 tensor，自动继承 `origin_shape`

## EVG

EVG（Epilogue Visitor Graph） 是用于 GEMM 后处理（Epilogue）的声明式框架。它将后处理操作（如加法、类型转换、广播、归约等）抽象为可组合的模板节点，通过树形或拓扑结构拼接，形成计算图

开发者只需用"表达式"声明计算逻辑（如 `D = C + X`），框架自动处理数据搬运、UB 空间分配、事件同步和流水调度

相比手工组织 GM/UB 拷贝和事件同步，EVG 显著降低开发复杂度，同时以尽量保持相近性能为目标，并支持图和节点复用与灵活扩展。具体的开发者可利用`Epilogue::Fusion` 下已经提供的预定义的数据和计算节点来实现嵌套的复杂的后处理逻辑

### 支持数据结构

EVG主要支持的数据结构有两类，包括TreeVisitor，TopologicalVisitor，针对是否存在共享子表达式的场景进行支持

- **TreeVisitor**：支持树形结构的节点组合，编写和维护更简单，适合没有共享子表达式的场景
- **TopologicalVisitor**：支持DAG拓扑结构，当一个中间结果被后续多个节点使用时使用DAG表达，适合有共享子表达式的场景

![](<../../assets/images/CATLASS 新版本能力介绍-image-7.png>)

EVG在UB支持的语义上更抽象的节点，在不同阶段支持的节点如下

1. Load阶段：数据从GM加载到UB，包括AccLoad、AuxLoad等

2. Compute阶段：在UB中进行计算，包括Compute、Cast等操作

3. Store阶段：将结果写回GM，包括AuxStore等

### 代码示例

加法操作`Epilogue::Fusion::Add`是对`AscendC::Add`的封装，EVG在使用时只需声明计算逻辑，无需关注搬运/事件/布局细节

TreeVisitor示例代码如下

```c++
// ...
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/kernel/matmul_visitor.hpp"
#include "catlass/epilogue/fusion/fusion.hpp"

// 定义 EVG: D = C + X
// C 是 workspace（A*B 的结果），D 是最终输出（C+X 的结果）
// 申请空间, 3为申请空间的节点数量，2代表缓冲区数量

using LayoutC = decltype(layoutC);
using EVG = Epilogue::Fusion::TreeVisitor<
    Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>,
    Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Add, ElementC>,  //中间子节点，计算
        Epilogue::Fusion::VisitorAccLoad<ElementC>,  // 左子节点，加载 C (workspace)
        Epilogue::Fusion::VisitorAuxLoad<ElementC, LayoutC>   // 右子节点，加载 X
    >
>;
```

TopoligicalVisitor示例代码如下

```c++
// 节点顺序：
// 0-AccLoad, 1-Compute1(2X), 2-Compute2(Exp(2X)),
// 3-Compute3(Exp(2X) + 1), 4-Compute4(Exp(2X) - 1), 5-Compute5(Compute3 / Compute4), 6-Store
using Edges = tla::tuple<
    tla::seq<>,         // 0: AccLoad 无子节点
    tla::seq<0>,        // 1: 依赖 AccLoad-->2X
    tla::seq<1>,        // 2: 依赖 Compute1-->Exp(2X)
    tla::seq<2>,        // 3: 依赖 Compute2-->(Exp(2X) - 1)
    tla::seq<2>,        // 4: 依赖 Compute2-->(Exp(2X) + 1)
    tla::seq<3, 4>,     // 5: 依赖 Compute3 与 Compute4-->(Compute3 / Compute4)
    tla::seq<5>         // 6: Store 依赖 Compute5
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
