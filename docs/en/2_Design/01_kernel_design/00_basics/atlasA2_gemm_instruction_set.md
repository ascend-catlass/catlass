# Introduction to the Atlas A2 GEMM Hardware Instruction Set

## API

### Data Movement

Data movement from GM to L1 uses the DataCopy interface, and from L1 to L0 uses the LoadData family of interfaces. For different input data types (int8/fp16/fp32) and data layouts (A-RowMajor/A-ColumnMajor/B-RowMajor/B-ColumnMajor, which can also be understood as non-transposed and transposed), the usage and input parameters of the related interfaces differ, giving **3\*4=12 combinations**. The following introduces the interface usage and parameter configuration for the 12 combinations.

First, the relevant definitions:

1. Large fractal and small fractal: both L1 and L0 use NZ-related formats. Each internal fractal arrangement is called a small fractal, and the overall arrangement is called a large fractal. For example, the zN layout means the small fractal is z and the large fractal is N.

2. FractalShape: the shape of the small fractal, which is (16, 32/sizeof(T)) on both L1 and L0.

3. FractalNum: when a small fractal is moved from L1 to L0 and the LoadDataWithTranspose interface is called, that interface can transpose only one merged square matrix at a time. For the int8 and F32 data types, FractalShape is (16,32) and (16,8) respectively, and two consecutive fractals must be merged into one square matrix before transposing; therefore this parameter indicates how many fractals a square matrix contains. For int8 and f32 data types, FractalNum is 2. For the f16 data type, this parameter has no effect.

4. FractalSize: the number of elements contained in a small fractal, which is 16\*(32/sizeof(T)); when the input data type is int8/f16/f32, this variable takes the value 512/256/128.

5. Usage of the DataCopy family of interfaces: see [details](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md).

6. Usage of the LoadData family of interfaces: see [details](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadData_2D.md).

7. The specific usage is as follows (here "transpose" refers to the change from the L1 small fractal to the L0 small fractal):

   1. LoadData: non-transposed, or transposed and a single-fractal square matrix

   2. LoadDataWithTranspose: transposed and the L1 layout is a contiguous multi-fractal square matrix

   3. LoadData3D: transposed and the L1 layout is a non-contiguous multi-fractal square matrix

   The usage of the LoadData family of instructions from L1 to L0 is summarized as follows.

   | Input matrix (GM)  | fp16     | int8                  | fp32         |
   | ------------- | -------- | --------------------- | ------------ |
   | A-RowMajor    | LoadData | LoadData              | LoadData     |
   | A-ColumnMajor | LoadData | LoadDataWithTranspose | LoadData3Dv2 |
   | B-RowMajor    | LoadData | LoadDataWithTranspose | LoadData3Dv2 |
   | B-ColumnMajor | LoadData | LoadData              | LoadData     |

#### A-RowMajor-Half

![Figure 1](https://raw.gitcode.com/user-images/assets/9091846/0c69a5ae-56c1-4662-b99d-fa7f4d44afe3/AscendC-basic-knowledge-image.png)

1. GM->L1

Use DataCopy for data movement; the specific parameter configuration is as follows. For the specific meaning of the parameters, see the related links.

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = m;   //40
nd2nzA1Params.dValue = k;   //56
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = k;

// The following parameter takes the aligned length of the A matrix along the height direction on L1.
// Since A is not transposed, this parameter is the same for all three data types.
nd2nzA1Params.dstNzC0Stride = CeilAlign(m, fractalShape[0]);

nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
```

- L1->L0

Because the small fractal of A-RowMajor is also RowMajor, the LoadData interface is used.

```cpp
__aicore__ inline void SplitA()
{
    AscendC::LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
    AscendC::LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
    uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize;
    uint32_t srcOffset = fractalSize;
    // Nz -> Zz
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilDivision(k, fractalShape[1]);
    loadDataParams.srcStride = CeilDivision(m, fractalShape[0]);
    // Gap between the end address of the previous fractal and the start address of the next fractal of the destination operand between adjacent iterations along the K axis
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    for (int i = 0; i < CeilDivision(m, fractalShape[0]); ++i) {
        AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
    }
    inQueueA2.EnQue<T>(a2Local);
    inQueueA1.FreeTensor(a1Local);
}
```

#### A-RowMajor-Int8

Configuration is the same as above.

#### A-RowMajor-Fp32

Configuration is the same as above.

#### A-ColumnMajor-Half

![Figure 2: Schematic of the data layout of the A matrix from GM-->L1-->L0A for ColumnMajor and the half data type](https://raw.gitcode.com/user-images/assets/9091846/83119a38-4518-42b6-b29e-497641cc6c87/AscendC-basic-knowledge-image-1.png)

1. GM->L1

When configuring the members of the Nd2NzParams struct, note that the shape of the source operand is (K, M) and the unit of dstNzC0Stride is 32B; this parameter takes the aligned row count of the zN matrix on L1, that is, the length of the K axis after aligning to FractalShape\[0].

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = k;
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = m;
// The following parameter takes the aligned length of the A matrix along the height direction on L1.
// Since A is transposed, the configuration of this parameter differs between the three data types.
if constexpr (AscendC::IsSameType<T, half>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzA1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
```

- L1->L0A

When the A matrix layout is ColumnMajor, the small fractal is n on L1 and z on L0A. In this case, use the LoadData instruction to complete the data movement and set ifTranspose=True.

As shown in Figure 2, loop over the M axis as the outer axis and configure loadDataParams.repeatTimes with the K axis as the inner axis. Here the values of srcoffset and dstoffset are CeilDivision(k, FractalShape\[0]) \* fractalSize.

```cpp
uint32_t dstOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
uint32_t srcOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
// Nz -> Zz
AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0]);
//source operand, inner axis, between adjacent iterations
loadDataParams.srcStride = 1;
loadDataParams.dstGap = 0;
loadDataParams.ifTranspose = true;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

#### A-ColumnMajor-Int8

![Figure 3: Schematic of the data layout of the A matrix from GM-->L1-->L0A for ColumnMajor and the int8 data type](https://raw.gitcode.com/user-images/assets/9091846/311fc412-cbf4-40e9-a636-4f13ec1d272e/AscendC-basic-knowledge-image-2.png)

1. GM->L1

When configuring the members of the Nd2NzParams struct, note that the shape of the source operand is (K, M) and the unit of dstNzC0Stride is 32B; this parameter takes the aligned row count of the zN matrix on L1.

- L1->L0A

The data layouts of the matrix on GM, L1, and L0A are RowMajor, nZ, and zZ respectively. The small fractal from nZ to zZ is transposed, so the LoadDataWithTranspose interface must be called. As shown in Figure 3, loop over the M axis as the outer axis and configure loadDataParams.repeatTimes with the K axis as the inner axis. Note that, because two consecutive fractals are merged into one square matrix during transposition, loadDataParams.repeatTimes=CeilDivision(k, fractalShape\[0] \* fractalNum). In addition, as shown in Figure 2, the two fractals that were in the same square matrix before transposition in L0A are contiguous in the physical layout of L1; after transposition, the gap between the end address of the previous fractal and the start address of the next is CeilDivision(k, fractalShape\[1]) - 1, in units of 512B (16\*32).

```cpp
// dstoffset must be derived from the alignment of the A matrix along the width direction on L0
uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize * fractalNum;
// srcoffset must be derived from the alignment of the A matrix along the height direction on L1
uint32_t srcOffset = CeilDivision(k, fractalShape[0] * fractalNum) * fractalSize * fractalNum;

// Nz -> Zz
AscendC::LoadData2dTransposeParams loadDataParams;
loadDataParams.startIndex = 0;
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0] * fractalNum);
loadDataParams.srcStride = 1;
loadDataParams.dstGap = 0;
// Gap between the end address of the previous fractal and the start address of the next fractal of the destination operand before transposition within each iteration, in units of 512B
loadDataParams.dstFracGap = CeilDivision(k, fractalShape[1]) - 1;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadDataWithTranspose(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

#### A-ColumnMajor-Fp32

![Figure 4: Schematic showing that the A matrix laid out on L1 cannot call the LoadDataWithTranspose instruction](https://raw.gitcode.com/user-images/assets/9091846/27c54a9f-f0ea-49bf-92cb-ae68dfe74d44/AscendC-basic-knowledge-image-3.png)

Data on L1 cannot directly call the LoadDataWithTranspose instruction for transposition, because two consecutive fractals along the K axis cannot be merged into one 16\*16 square matrix.

1. GM->L1

When configuring the members of the Nd2NzParams struct, recall that the shape of the source operand is (K, M) and the unit of dstNzC0Stride is 32B; this parameter takes the aligned row count of the matrix on L1 after transposition, that is, the length of the K axis after aligning to FractalShape\[0].

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = k;
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = m;

// The following parameter takes the aligned length of the A matrix along the height direction on L1.
// Since A is transposed, the configuration of this parameter differs between the three data types.
        if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzA1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}

nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
```

- L1->L0

Because the fractals cannot be merged into a square matrix for the transfer, the LoadData3DV2 interface must be called. Before writing to L0A, the height and width axes of the A matrix are first aligned to 16 and 8 respectively; then the instruction transposes both the large and small fractals of the A matrix and finally writes it to L0A in zZ layout.

The following describes how to configure the members of the LoadData3DParamsV2 struct of the Load3Dv2 instruction.

According to the img2col process performed by the Load3Dv2 instruction, the height of the A matrix after img2col is ho \* wo. Substituting the convolution kernel width, the convolution kernel sliding step, and the convolution kernel dilation factor into the formulas for ho and wo, the height of the A matrix is CeilAlign(k, fractalShape\[0]); the width of the A matrix after img2col is ci \* kh \* kw; substituting kh=1 and kw=1 gives the width of the A matrix as CeilAlign(m, fractalShape\[1]). Finally, set loadDataParams.enTranspose = true to transpose the whole A matrix and transpose each of its fractals.

```cpp
// source operand height
loadDataParams.l1H = CeilAlign(k, fractalShape[0]);
// source operand width
loadDataParams.l1W = 1;
// number of channels of the source operand,
// the height of the img2col result matrix is ho * wo; substituting the convolution kernel width, sliding step, and dilation factor into the formulas for ho and wo gives ho * wo = loadDataParams.l1H * loadDataParams.l1w
// the width of the img2col result matrix is ci * kh * kw; substituting kh=1, kw=1 gives the result matrix width ci = loadDataParams.channelSize = m
loadDataParams.channelSize = CeilAlign(m, fractalShape[1]);
// transfer length of this instruction along the width dimension of the destination operand
loadDataParams.kExtension = CeilAlign(m, fractalShape[1]);
// transfer length of this instruction along the height dimension of the destination operand
loadDataParams.mExtension = CeilAlign(k, fractalShape[1] * fractalNum);
// sliding step of the convolution kernel along the width dimension of the source operand
loadDataParams.strideW = 1;
// sliding step of the convolution kernel along the height dimension of the source operand
loadDataParams.strideH = 1;
// convolution kernel width
loadDataParams.filterW = 1;
// convolution kernel height
loadDataParams.filterH = 1;
// convolution kernel width dilation factor
loadDataParams.dilationFilterW = 1;
// convolution kernel height dilation factor
loadDataParams.dilationFilterH = 1;
loadDataParams.enTranspose = true;
```

#### B-RowMajor-Half

![Figure 5: Schematic of data movement of the B matrix from GM-->L1-->L0B without transposition and with the half data type](https://raw.gitcode.com/user-images/assets/9091846/227a5505-1fcd-49fa-afd6-601d8543b281/AscendC-basic-knowledge-image-4.png)

1. GM->L1

When configuring the members of the Nd2NzParams struct, note that the shape of the source operand is (K, N) and the unit of dstNzC0Stride is 32B; this parameter takes the row count of the fractal-aligned matrix on L1.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// The following parameter takes the aligned length of the B matrix along the height direction on L1.
// Since A is transposed, the configuration of this parameter differs between the three data types.
     if constexpr (AscendC::IsSameType<T, half>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

The fractal on L1 is transposed into the fractal on L0 and is an fp16 square matrix, so the LoadData instruction is used for data movement.

As shown in Figure 5, loop over the K axis as the outer axis and configure loadDataParams.repeatTimes with the N axis as the inner axis.

```cpp
uint32_t dstOffset = CeilDivision(n, fractalShape[0] * fractalNum) * fractalSize * fractalNum;
uint32_t srcOffset = fractalSize * fractalNum;
// Nz -> Zn
AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = CeilDivision(n, fractalShape[0] * fractalNum);
loadDataParams.srcStride = CeilDivision(k, fractalShape[0] * fractalNum);
loadDataParams.dstGap = 0;
loadDataParams.ifTranspose = true;
for (int i = 0; i < CeilDivision(k, fractalShape[0] * fractalNum); ++i) {
    AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
}
```

#### B-RowMajor-Int

![Figure 6: Schematic of the data layout of the B matrix from GM-->L1-->L0B for RowMajor and the int8 data type](https://raw.gitcode.com/user-images/assets/9091846/7d56129c-58ee-4a8b-bc9c-1a0c9fe67e1f/AscendC-basic-knowledge-image-5.png)

1. GM->L1

When configuring the members of the Nd2NzParams struct, note that the shape of the source operand is (K, N) and the unit of dstNzC0Stride is 32B; this parameter takes the row count of the aligned matrix on L1.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// The following parameter takes the aligned length of the B matrix along the height direction on L1.
// Since A is transposed, the configuration of this parameter differs between the three data types.
if constexpr (AscendC::IsSameType<T, int8_t>::value && AscendC::IsSameType<U, int32_t>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0] * fractalNum);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

Because the fractal on L1 is transposed into the fractal on L0B and this is not the FP16 square matrix scenario, the LoadDataWithTranspose interface is called.

As shown in Figure 6, loop over the K axis as the outer axis and configure loadDataParams.repeatTimes with the N axis. Note that, because two consecutive fractals are merged into one square matrix during transposition, loadDataParams.repeatTimes=CeilDivision(k, fractalShape\[0]\*fractalNum). In addition, as shown in Figure 6, the two fractals that were in the same square matrix before transposition in L0A are contiguous on L1 and remain contiguous after transposition, so the gap between the address of the previous fractal and that of the next is 0.

```cpp
uint32_t dstOffset = CeilDivision(n, fractalShape[0] * fractalNum) * fractalSize * fractalNum;
uint32_t srcOffset = fractalSize * fractalNum;
AscendC::LoadData2dTransposeParams loadDataParams;
loadDataParams.startIndex = 0;
loadDataParams.repeatTimes = CeilDivision(n, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(k, fractalShape[0] * fractalNum);
loadDataParams.dstGap = 1;
loadDataParams.dstFracGap = 0;
for (int i = 0; i < CeilDivision(k, fractalShape[0] * fractalNum); ++i) {
    AscendC::LoadDataWithTranspose(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
}
```

#### B-RowMajor-Fp32

For Fp32 input, the LoadData3DV2 interface is used for data movement. When the LoadData3DV2 instruction is called, before writing to L0B it first aligns the K and N axes of the B matrix to 16 and 8 respectively, transposes and moves the B matrix as a whole, and finally writes an nZ layout to L0B.

For alignment, the K axis of the B matrix on L1 is aligned to FractalShape\[0]\*fractalNum and the N axis to FractalShape\[1]. On L0B, the K axis is aligned to FractalShape\[0]\*FractalNum and the N axis to FractalShape\[1].

1. GM->L1

When configuring the members of the Nd2NzParams struct, note that the shape of the source operand is (K, N) and the unit of dstNzC0Stride is 32B; this parameter is the aligned row count of the matrix on L1.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// The following parameter takes the aligned length of the B matrix along the height direction on L1.
// Since A is transposed, the configuration of this parameter differs between the three data types.
if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

Movement of the fp32 data type requires the corresponding configuration for the Load3Dv2 instruction interface.

According to the img2col process performed by the Load3Dv2 instruction, the height of the B matrix after img2col is ho \* wo. Substituting the convolution kernel width, sliding step, and dilation factor into the formulas for ho and wo, the height of the B matrix is CeilAlign(k, fractalShape\[0]); the width of the B matrix after img2col is ci \* kh \* kw; substituting kh=1 and kw=1 gives the width of the B matrix as CeilAlign(n, fractalShape\[1]). Note that the loadDataParams.enTranspose setting is effective only for the A matrix; for the B matrix, taking the value true or false does not affect functionality.

```cpp
loadDataParams.l1H = CeilAlign(k, fractalShape[0]);
loadDataParams.l1W = 1;
loadDataParams.channelSize = CeilAlign(n, fractalShape[1]);
loadDataParams.kExtension = CeilAlign(n, fractalShape[1]);
loadDataParams.mExtension = CeilAlign(k, fractalShape[0]);
loadDataParams.strideW = 1;
loadDataParams.strideH = 1;
loadDataParams.filterW = 1;
loadDataParams.filterH = 1;
loadDataParams.dilationFilterW = 1;
loadDataParams.dilationFilterH = 1;
loadDataParams.filterSizeW = false;
loadDataParams.filterSizeH = false;
loadDataParams.enTranspose = true;
loadDataParams.fMatrixCtrl = false;
```

#### B-ColumnMajor-Half

![Figure 7: Schematic of the data layout of the B matrix from GM-->L1-->L0B with transposition and the half data type](https://raw.gitcode.com/user-images/assets/9091846/47b2e408-f9c9-4668-9496-65287a9a3d22/AscendC-basic-knowledge-image-6.png)

1. GM->L1

Configure DataParams according to the specific data layout, and use DataCopy for data movement.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = n;
nd2nzB1Params.dValue = k;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = k;

// The following parameter takes the aligned length of the B matrix along the height direction on L1.
// Since B is transposed, the configuration of this parameter is the same for all three data types.
nd2nzB1Params.dstNzC0Stride = CeilAlign(n, fractalShape[0]);
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

With the B matrix ColumnMajor input, L1 uses nZ representation and L0 uses nZ representation, so Load2D can be called directly for data movement.

As shown in Figure 7, loop over the K axis as the outer axis and configure loadDataParams.repeatTimes with the N axis as the inner axis. As shown in the figure, srcoffset is the address offset of the LocalTensor on L1 each time the B matrix loops along the K axis, and dstoffset is the address offset of the LocalTensor on L0B each time the B matrix loops along the K axis. Because the B matrix on L1 is equivalent to the B matrix on L0B, srcOffset and dstOffset take the same value.

```cpp
__aicore__ inline void SplitB()
{
    AscendC::LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
    AscendC::LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
    // srcOffset and dstOffset are the same
    // the n axis is aligned to fractalShape[0]
    uint32_t dstOffset = CeilDivision(n, fractalShape[0]) * fractalSize;
    uint32_t srcOffset = CeilDivision(n, fractalShape[0]) * fractalSize;
    // Nz -> Zz
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilDivision(n, fractalShape[0]);
    loadDataParams.srcStride = 1;
    // Gap between the end address of the previous fractal and the start address of the next fractal of the destination operand between adjacent iterations along the N axis
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    // the k axis is aligned to fractalShape[1]
    for (int i = 0; i < CeilDivision(k, fractalShape[1]); ++i) {
        AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
    }
    inQueueB1.FreeTensor(b1Local);
    inQueueB2.EnQue<T>(b2Local);
}
```

#### B-ColumnMajor-Int

Same as above.

#### B-ColumnMajor-Fp32

Same as above.

#### Fixpipe

When moving data through the Fixpipe instruction, the members of the MmadParams struct must be configured; for their specific meaning, see [Fixpipe - Data Movement](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.md). Here the unit of fixpipeParams.srcStride is 32/sizeof(T) elements; it means the start address offset of adjacent small fractals in the source NZ matrix. The elements of the same row in the (RowMajor) matrix are in adjacent Z arrangement in the source NZ matrix, and the value of this parameter is the length of the C matrix on L0C after aligning the M axis to 16.

```cpp
AscendC::FixpipeParamsV220 fixpipeParams;
fixpipeParams.nSize = n;
fixpipeParams.mSize = m;

// The source operand comes from L0c, so m only needs to be aligned to 16, independent of the data type.
//start address offset of adjacent Z arrangements in the source NZ matrix
fixpipeParams.srcStride = CeilAlign(m, fractalShape[0]);
fixpipeParams.dstStride = n;

fixpipeParams.ndNum = 1;
fixpipeParams.srcNdStride = 0;
fixpipeParams.dstNdStride = 0;
```

### Matrix Computation

The following describes how to configure the members of the MmadParams struct of the Mmad instruction; for the specific meaning of each variable, see [Mmad - Matrix Computation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.md).

Note that when the Mmad instruction executes, the matrix computation unit reads multiple fractals consecutively from L0A/L0B to participate in the matrix multiply; the number of fractals read depends on the values of the members m, n, and k of the MmadParams struct and on the alignment requirements of the Mmad instruction for each axis of the A matrix/B matrix on L0A/L0B.

Because the Mmad instruction reads fractals consecutively with an A matrix fractal of (16,32) and a B matrix fractal of (32,16), that is, the total number of fractals the matrix computation unit reads consecutively from L0A/L0B is CeilDivision(m, 16)\*CeilDivision(k,32) for L0A and CeilDivision(k,32)\*CeilDivision(n,16) for L0B.

Therefore, when the actual alignment requirements for the A and B matrices on each axis on L0A/L0B are inconsistent with the default alignment requirements of the Mmad instruction, reading consecutive fractals may erroneously read fractals filled entirely with invalid data and ignore valid data.

As shown in Figure 8, taking int8 input data and RowMajor A/B matrices as an example, with A shape(30, 60), B shape(60, 70), L0A aligned to 16 and 32 along the M and K axes respectively, and L0B aligned to 32 and 16\*2 along the K and N axes respectively, while the default alignment requirements of the Mmad instruction along the M/K/N axes are 16/32/16 respectively, the actual alignment requirement of the N axis here is inconsistent with the default requirement of the Mmad instruction.

As shown in the left figure of Figure 8, if mmadParams.n = 70 is set, fractal number 5 will be read while the valid data of fractal number 10 will not be included.

As shown in the right figure of Figure 8, if mmadParams.n=CeilAlign(n, FractalShape\[0]\*FractalNum)=96 is set, all fractals will be read. Although the matrix computation result then contains invalid data, setting fixpipeParams.nSize when the Fixpipe instruction moves the data out ensures that results computed from invalid data are not moved out.

![Figure 8: Actual alignment requirement of the N axis for B-RowMajor and the int8 data type](https://raw.gitcode.com/user-images/assets/9091846/70f3155f-4db7-4c4c-a6a4-95a42c1aa6c1/AscendC-basic-knowledge-image-7.png)

Similar to the above scenario, when the input data type is float and the A matrix is ColumnMajor, the actual alignment requirement of the K axis is also inconsistent with the default alignment requirement of the MMad instruction, but the solution in this scenario differs from the above: the mmadParams.kDirectionAlign parameter must be introduced separately.

According to the formula of matrix multiplication, the K axis is the common dimension of the A/B matrices. In this case, if mmadParams.k=CeilAlign(k, fractalShape\[1]\*fractalNum) is set as in the above scenario, the value of every element of the C matrix will be affected by the extra invalid data read in, and it is also not possible to discard the invalid data at the move-out stage by setting fixpipeParams parameters.

As shown in Figure 9, mmadParams.kDirectionAlign takes effect only when the input data type is float. When the A matrix is ColumnMajor, kDirectionAlign is set to true; then the A matrix on L0A is 16-aligned along K, and the matrix computation unit skips the padded invalid data when reading from L0A. In other scenarios this parameter takes its default value false, and the A matrix on L0A is 8-aligned along K.

That is, for reading an L0A matrix of type float, the K direction is aligned to 8 elements by default. But if the A matrix is transposed and FP32, L1 has an nZ 8\*16 layout, and since it must be merged into a 16\*16 transfer, the K direction is 16-aligned.

If kDirectionAlign=false, the K direction is 8-aligned and fractals 0-9 are read. Fractal 5 is read but fractal 10 is not, causing a computation error.

If kDirectionAlign=true, the K direction is aligned to 16 elements, and fractal number 5 is skipped when reading consecutive physical addresses, correctly reading the data into the cube.

So the purpose of setting kDirectionAlign=true is to align to 16 on L0A and skip the part that should not participate in the computation.

![Figure 9: Actual alignment of the K axis for a transposed A matrix and the float data type](https://raw.gitcode.com/user-images/assets/9091846/717fe7f3-2267-4603-8e45-a894ad6f5569/AscendC-basic-knowledge-image-8.png)

```cpp
AscendC::MmadParams mmadParams;
// left matrix Height
mmadParams.m = m;
// right matrix width
mmadParams.n = n;
if constexpr (AscendC::IsSameType<T, int8_t>::value && AscendC::IsSameType<U, int32_t>::value) {
    if constexpr (!isBtranspose) {
        // mmad aligns the n axis to 16 by default, but because the n axis is aligned to 2 * 16 during the b transposition, a 32 * 16 fractal filled entirely with invalid data is padded.
        // If mmadParams.n = n is still set, the cube unit will read extra fractals of invalid data while the fractal of valid data is not read.
        // In this case the n axis can be aligned to 32 so that this fractal participates in the computation, and the fractal obtained from the computation involving the invalid fractal is skipped when moving out.
        mmadParams.n = CeilAlign(n, fractalShape[0] * fractalNum);
    }
}
// left matrix Width, right matrix Height
mmadParams.k = k;
if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    if constexpr (isAtranspose) {
        mmadParams.kDirectionAlign = true;
    }
}
```

## Synchronization Instructions

The execution units inside an AIC core/AIV core (such as the MTE2 movement unit and the Vector computation unit) run asynchronously in parallel, and data dependencies may exist when they read and write the same storage resource. To ensure data consistency and computation correctness, the operation timing must be coordinated through synchronization control.

### Intra-Core Synchronization

There are 8 parallel instruction pipelines inside AIC/AIV.

| Pipeline type   | Meaning                                                     |
| ---------- | -------------------------------------------------------- |
| PIPE\_S    | Scalar pipeline; used when the Tensor GetValue function is called |
| PIPE\_V    | Vector pipeline                                             |
| PIPE\_M    | Cube pipeline                                               |
| PIPE\_MTE1 | Pipeline originating from the L1 buffer, including L1->L0A, L1->L0B, and L1->BT |
| PIPE\_MTE2 | Pipeline originating from GM, including GM->L1, GM->L0A, GM->L0B, and GM->UB |
| PIPE\_MTE3 | Pipeline from UB/L1 back to GM, including UB->GM and L1->GM |
| PIPE\_FIX  | Fixpipe-related pipeline, including L0C->GM, L0C->L1, and L1->FP buffer |
| PIPE\_ALL  | all pipelines                                                 |

![AscendC-basic-knowledge-image-9](https://raw.gitcode.com/user-images/assets/9091846/718dc2fa-e681-453f-9b6f-0a6da9313fd6/AscendC-basic-knowledge-image-9.png)

### Inter-Core Synchronization

The inter-core synchronization mechanism applies only to AIC/AIV Mix operators, not to pure AIC or pure AIV operators. (AIC means Cube; AIV means Vector.)

Commonly used inter-core synchronization instruction sets:

- [SyncAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)
- [CrossCoreSetFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md) with [CrossCoreWaitFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)

The synchronization control implemented by `CrossCoreSetFlag` together with `CrossCoreWaitFlag` is divided into the following modes:

- `Mode 0`: synchronization control between AI Cores. In the AIC scenario, all AIC cores are synchronized: the instructions after CrossCoreWaitFlag are executed only after all AIC cores have executed CrossCoreSetFlag. In the AIV scenario, all AIV cores are synchronized: the instructions after CrossCoreWaitFlag are executed only after all AIV cores have executed CrossCoreSetFlag.
- `Mode 1`: synchronization control between AIV cores inside an AI Core. The instructions after CrossCoreWaitFlag are executed only after both AIV cores have run CrossCoreSetFlag.
- `Mode 2`: synchronization control between AIC and AIV inside an AI Core. After the AIC core executes CrossCoreSetFlag, the instructions after CrossCoreWaitFlag on both AIV cores continue to execute; after both AIV cores execute CrossCoreSetFlag, the instructions after CrossCoreWaitFlag on the AIC core can execute.

![zh-cn_image_0000002562462709](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/figure/zh-cn_image_0000002562462709.png)
