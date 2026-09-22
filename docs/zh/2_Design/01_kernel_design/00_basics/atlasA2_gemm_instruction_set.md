# AtlasA2 GEMM类硬件指令集介绍

## API

### 数据搬运

数据搬运从GM到L1使用DataCopy接口，从L1到L0使用LoadData系列接口，相关接口在不同的数据输入类型（int8/fp16/fp32）和数据排布(A-RowMajor/A-ColumnMajor/B-RowMajor/B-ColumnMajor，也可理解为非转置和转置)下接口的使用方式和入参不同，共有**3\*4=12种组合**, 下面将针对12种组合下的接口使用方式和参数配置进行介绍

首先给出相关的定义：

1. 大分形和小分形：在L1和L0上都是NZ相关格式排布，其中每个内部分形排布称为小分形，整体的排布称为大分形，例如zN排布就是小分形为z，大分形为N的排布

2. FractalShape：小分形的形状，在L1和L0上都是(16, 32/sizeof(T))

3. FractalNum：当小分形从L1搬运至L0且调用LoadDataWithTranspose接口时，该接口一次只能转置一个拼合后的方阵，对于int8和F32数据类型来说，FractalShape分别是(16,32)和(16,8)，都需要将连续的两个分形合并为一个方阵后转置，因此该参数表示一个方阵包含几个分形。对于int8和f32数据类型，FractalNum为2。对于f16数据类型，该参数不起作用

4. FractalSize：小分形所包含的元素个数，为16\*(32/sizeof(T))，当输入数据类型分别为int8/f16/f32时，该变量取值为512/256/128

5. DataCopy系列接口使用方式：相关接口[详见](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00127.html)

6. LoadData系列接口使用方式：相关接口[详见](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00169.html)

7. 具体使用情况如下(这里转置指的是L1-小分形到L0-小分形的变化)：

   1. LoadData：非转置 或 转置且为单分形方阵

   2. LoadDataWithTranspose：转置 且 L1排布为连续多分形方阵

   3. LoadData3D：转置 且 L1排布为非连续多分形方阵

   L1到L0的LoadData系列指令的使用情况总结如下

   | 输入矩阵(GM)  | fp16     | int8                  | fp32         |
   | ------------- | -------- | --------------------- | ------------ |
   | A-RowMajor    | LoadData | LoadData              | LoadData     |
   | A-ColumnMajor | LoadData | LoadDataWithTranspose | LoadData3Dv2 |
   | B-RowMajor    | LoadData | LoadDataWithTranspose | LoadData3Dv2 |
   | B-ColumnMajor | LoadData | LoadData              | LoadData     |

#### A-RowMajor-Half

![图1](https://raw.gitcode.com/user-images/assets/9091846/0c69a5ae-56c1-4662-b99d-fa7f4d44afe3/AscendC-basic-knowledge-image.png)

1. GM->L1

使用DataCopy进行数据搬运，具体参数配置如下。参数具体含义详见相关链接

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = m;   //40
nd2nzA1Params.dValue = k;   //56
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = k;

// 以下这个参数取A矩阵在L1上，高度方向的对齐后的长度
// 由于A不转置，因此对于三种数据类型该参数均相同
nd2nzA1Params.dstNzC0Stride = CeilAlign(m, fractalShape[0]);

nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
```

- L1->L0

由于A-Rowmajor的情况下，小分形也是RowMajor的，所以使用LoadData接口

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
    // K轴方向相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔
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

配置方式同上

#### A-RowMajor-Fp32

配置方式同上

#### A-ColumnMajor-Half

![图2：A矩阵ColumnMajor，half数据类型下，GM-->L1-->L0A的数据排布示意图](https://raw.gitcode.com/user-images/assets/9091846/83119a38-4518-42b6-b29e-497641cc6c87/AscendC-basic-knowledge-image-1.png)

1. GM->L1

配置Nd2NzParams结构体成员时需要注意，源操作数的shape为(K, M), dstNzC0Stride的单位为32B，该参数取值为L1上zN矩阵的对齐后的行数，也就是K轴对齐到FractalShape\[0]后的长度

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = k;
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = m;
// 以下这个参数取A矩阵在L1上，高度方向的对齐后的长度
// 由于A转置，因此三种数据类型下，该参数的配置不相同
if constexpr (AscendC::IsSameType<T, half>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzA1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
```

- L1->L0A

当A矩阵排布为ColumnMajor时，小分形在L1上为n，在L0A上为z。针对这种情况，使用LoadData指令完成数据搬运，并且设置ifTranspose=True

如图2所示，以M轴方向作为外轴进行for循环，以K轴方向作为内轴来配置loadDataParams.repeatTimes。在这里对于srcoffset和dstoffset的值为CeilDivision(k, FractalShape\[0]) \* fractalSize

```cpp
uint32_t dstOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
uint32_t srcOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
// Nz -> Zz
AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0]);
//源操作数，内轴，相邻迭代间
loadDataParams.srcStride = 1;
loadDataParams.dstGap = 0;
loadDataParams.ifTranspose = true;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

#### A-ColumnMajor-Int8

![图3：A矩阵ColumnMajor，int8数据类型下，GM-->L1-->L0A数据排布示意图](https://raw.gitcode.com/user-images/assets/9091846/311fc412-cbf4-40e9-a636-4f13ec1d272e/AscendC-basic-knowledge-image-2.png)

1. GM->L1

配置Nd2NzParams结构体成员时需要注意，源操作数的shape为(K, M)，dstNzC0Stride的单位为32B，该参数取值为L1上zN矩阵的对齐后的行数

- L1->L0A

矩阵在GM、L1、L0A上的数据排布分别是RowMajor，nZ和zZ，从nZ到zZ的小分形是转置的，所以需要调用LoadDataWithTranspose接口。如图3所示，以M轴方向作为外轴进行for循环，以K轴方向作为内轴来配置loadDataParams.repeatTimes。需要注意的是，由于转置时连续两个分形合并成一个方阵，因此loadDataParams.repeatTimes=CeilDivision(k, fractalShape\[0] \* fractalNum)。另外，如图2所示，L0A中转置前处于同一个方阵中的两个分形在L1的物理排布上是连续的，转置后前一个分形结束的地址与后一个分形起始地址的间隔为CeilDivision(k, fractalShape\[1]) - 1，单位是512B(16\*32)

```cpp
// dstoffset要根据A矩阵在L0上，宽度方向的对齐来求解
uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize * fractalNum;
// srcoffset要根据A矩阵在L1上，高度方向的对齐来求解
uint32_t srcOffset = CeilDivision(k, fractalShape[0] * fractalNum) * fractalSize * fractalNum;

// Nz -> Zz
AscendC::LoadData2dTransposeParams loadDataParams;
loadDataParams.startIndex = 0;
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0] * fractalNum);
loadDataParams.srcStride = 1;
loadDataParams.dstGap = 0;
// 每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位为512B
loadDataParams.dstFracGap = CeilDivision(k, fractalShape[1]) - 1;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadDataWithTranspose(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

#### A-ColumnMajor-Fp32

![图4：L1上排布的A矩阵，无法调用LoadDataWithTranspose指令示意图](https://raw.gitcode.com/user-images/assets/9091846/27c54a9f-f0ea-49bf-92cb-ae68dfe74d44/AscendC-basic-knowledge-image-3.png)

在L1上的数据不能直接调用LoadDataWithTranspose指令进行转置，因为在K轴方向两个连续的分形不能合并为一个16\*16的方阵

1. GM->L1

配置Nd2NzParams结构体成员时，需要追忆，源操作数的shape为(K, M), dstNzC0Stride的单位为32B，该参数取值为L1矩阵转置后的对齐后的行数，也就是K轴对齐到FractalShape\[0]的长度

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = k;
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = m;

// 以下这个参数取A矩阵在L1上，高度方向的对齐后的长度
// 由于A转置，因此三种数据类型下，该参数的配置不相同
        if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzA1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}

nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
```

- L1->L0

因为无法合并为方阵进行搬运，需要调用LoadData3DV2接口，在写入L0A之前会先分别将A矩阵高度和宽度轴向16, 8对齐，接着该指令会将A矩阵的大小分形都进行转置，最终以zZ排布写入到L0A

下面介绍如何配置Load3Dv2指令的LoadData3DParamsV2结构体成员

根据Load3Dv2指令完成img2col的过程，可知 img2col后A矩阵高度为ho \* wo,根据ho和wo的计算公式，代入卷积核宽度、卷积核滑动步长、卷积核膨胀系数等参数可知：A矩阵的高度为 CeilAlign(k, fractalShape\[0])；img2col后A矩阵宽度为ci \* kh \* kw,代入kh=1,kw=1，可知A矩阵的宽度为CeilAlign(m, fractalShape\[1])。最后，配置loadDataParams.enTranspose = true，将整个A 矩阵转置并且将其中每一个分形转置。

```cpp
// 源操作数height
loadDataParams.l1H = CeilAlign(k, fractalShape[0]);
// 源操作数width
loadDataParams.l1W = 1;
// 源操作数的通道数，
// img2col的结果矩阵高度为ho * wo,根据ho和wo的计算公式，代入卷积核宽度、卷积核滑动步长、卷积核膨胀系数等参数可知：ho * wo = loadDataParams.l1H * loadDataParams.l1w
// img2col的结果矩阵宽度为ci * kh * kw,代入kh=1,kw=1，可知结果矩阵的宽度为ci=loadDataParams.channelSize = m
loadDataParams.channelSize = CeilAlign(m, fractalShape[1]);
// 该指令在目的操作数width维度的传输长度
loadDataParams.kExtension = CeilAlign(m, fractalShape[1]);
// 该指令在目的操作数height维度的传输长度
loadDataParams.mExtension = CeilAlign(k, fractalShape[1] * fractalNum);
// 卷积核在源操作数width维度滑动的步长
loadDataParams.strideW = 1;
// 卷积核在源操作数height维度滑动的步长
loadDataParams.strideH = 1;
// 卷积核width
loadDataParams.filterW = 1;
// 卷积核height
loadDataParams.filterH = 1;
// 卷积核width膨胀系数
loadDataParams.dilationFilterW = 1;
// 卷积核height膨胀系数
loadDataParams.dilationFilterH = 1;
loadDataParams.enTranspose = true;
```

#### B-RowMajor-Half

![图5: B矩阵不转置，half数据类型下，GM-->L1-->L0B的数据搬运示意图](https://raw.gitcode.com/user-images/assets/9091846/227a5505-1fcd-49fa-afd6-601d8543b281/AscendC-basic-knowledge-image-4.png)

1. GM->L1

配置Nd2NzParams结构体成员时，需要注意，源操作数的shape为(K, N)，dstNzC0Stride的单位为32B，该参数取值为L1上分形对齐后的行数

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// 以下这个参数取B矩阵在L1上，高度方向的对齐后的长度
// 由于A转置，因此三种数据类型下，该参数的配置不相同
     if constexpr (AscendC::IsSameType<T, half>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

L1上分形到L0上分形发生转置，且为fp16数据类型方阵，使用LoadData指令进行数据搬运

如图5所示，以K轴为外轴进行for循环，以N轴方向作为内轴配置loadDataParams.repeatTimes

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

![图6: B矩阵RowMajor, int8数据类型下, GM-->L1-->L0B的数据排布示意图](https://raw.gitcode.com/user-images/assets/9091846/7d56129c-58ee-4a8b-bc9c-1a0c9fe67e1f/AscendC-basic-knowledge-image-5.png)

1. GM->L1

配置Nd2NzParams结构体的成员时，需要注意的是源操作数的shape为(K, N), dstNzC0Stride的单位为32B，该参数取值为L1上矩阵对齐后的行数

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// 以下这个参数取B矩阵在L1上，高度方向的对齐后的长度
// 由于A转置，因此三种数据类型下，该参数的配置不相同
if constexpr (AscendC::IsSameType<T, int8_t>::value && AscendC::IsSameType<U, int32_t>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0] * fractalNum);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

由于L1上分形到L0B上的分形发生了转置，且是非FP16方阵场景，调用LoadDataWithTranspose接口

如图6所示，以K轴方向作为外轴进行for循环，以N轴方向作为N轴来配置loadDataParams.repeatTimes。需要注意的是，由于转置时连续两个分形和行为一个方阵因此loadDataParams.repeatTimes=CeilDivision(k, fractalShape\[0]\*fractalNum)。另外，如图6所示，L0A中转置前同一块方阵中的两个分形在L1上是连续的，转置后依然是连续的，因此前一个分形地址和后一个分形地址的间隔为0

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

针对Fp32输入，使用LoadData3DV2接口完成数据搬运。调用LoadData3DV2指令时，在写入L0B之前会先分别将B矩阵K轴和N轴向16和8对齐，并将B矩阵整体进行转置和搬运，最终向L0B写入的是nZ排布

对齐上，L1上B矩阵的K轴向FractalShape\[0]\*fractalNum对齐，N轴向FractalShape\[1]对齐。在L0B上K轴向FractalShape\[0]\*FractalNum对齐，N轴向FractalShape\[1]对齐

1. GM->L1

配置Nd2NzParams结构体成员时，需要注意源操作数的shape为(K, N), dstNzC0Stride的单位为32B，该参数为L1上矩阵对齐后的行数

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
// 以下这个参数取B矩阵在L1上，高度方向的对齐后的长度
// 由于A转置，因此三种数据类型下，该参数的配置不相同
if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

fp32数据类型的搬运需要针对Load3Dv2指令的接口进行相应配置

根据Load3Dv2指令完成img2col的过程，可知 img2col后B矩阵高度为ho \* wo,根据ho和wo的计算公式，代入卷积核宽度、卷积核滑动步长、卷积核膨胀系数等参数可知：B矩阵的高度为 CeilAlign(k, fractalShape\[0])；img2col后B矩阵宽度为ci \* kh \* kw，代入kh=1,kw=1，可知B矩阵的宽度为CeilAlign(n, fractalShape\[1])。需要注意的是 loadDataParams.enTranspose 配置仅仅对A矩阵有效，对B矩阵取值为true或者false不会影响功能。

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

![图7: B矩阵转置，half数据类型下，GM-->L1-->L0B数据排布示意图](https://raw.gitcode.com/user-images/assets/9091846/47b2e408-f9c9-4668-9496-65287a9a3d22/AscendC-basic-knowledge-image-6.png)

1. GM->L1

根据数据排布具体情况配置DataParams，使用DataCopy进行数据搬运

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = n;
nd2nzB1Params.dValue = k;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = k;

// 以下这个参数取B矩阵在L1上，高度方向的对齐后的长度
// 由于B转置，因此三种数据类型下，该参数的配置相同
nd2nzB1Params.dstNzC0Stride = CeilAlign(n, fractalShape[0]);
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

- L1->L0

在B矩阵ColumnMajor输入的情况下，L1上nZ表示，L0上nZ表示，可直接调用Load2D进行数据搬运

如图7所示，以K轴方向作为外轴进行for循环，以N轴方向作为内轴来配置loadDataParams.repeatTimes。如图所示，srcoffset是L1上B矩阵K轴方向每循环一次时LocalTensor的地址偏移量，dstoffset是在L0B上B矩阵按K轴方向循环一次LocalTensor的地址偏移量。由于L1上的B矩阵与L0B上的B矩阵等价，因此srcOffset和dstOffset取值相同

```cpp
__aicore__ inline void SplitB()
{
    AscendC::LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
    AscendC::LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
    // srcOffset和dstOffset相同
    // n轴向fractalShape[0]对齐
    uint32_t dstOffset = CeilDivision(n, fractalShape[0]) * fractalSize;
    uint32_t srcOffset = CeilDivision(n, fractalShape[0]) * fractalSize;
    // Nz -> Zz
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilDivision(n, fractalShape[0]);
    loadDataParams.srcStride = 1;
    // N轴方向相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    // k轴向fractalShape[1]对齐
    for (int i = 0; i < CeilDivision(k, fractalShape[1]); ++i) {
        AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
    }
    inQueueB1.FreeTensor(b1Local);
    inQueueB2.EnQue<T>(b2Local);
}
```

#### B-ColumnMajor-Int

同上

#### B-ColumnMajor-Fp32

同上

#### Fixpipe

通过Fixpipe指令进行搬运时，需要配置MmadParams的结构体成员，具体的含义可参考[Fixpipe-数据搬运](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0251.html)。其中，fixpipeParams.srcStride的单位是32/sizeof(T)个元素，其含义是源NZ矩阵中相邻小分形的起始地址偏移(RowMajor)矩阵中同一行的元素在源NZ矩阵中处于相邻的Z排布，该参数的取值是L0C上C矩阵M轴向16对齐后的长度

```cpp
AscendC::FixpipeParamsV220 fixpipeParams;
fixpipeParams.nSize = n;
fixpipeParams.mSize = m;

// 源操作数来源于L0c，因此m只需要向16对齐，与数据类型无关
//源NZ矩阵中相邻Z排布的起始地址偏移
fixpipeParams.srcStride = CeilAlign(m, fractalShape[0]);
fixpipeParams.dstStride = n;

fixpipeParams.ndNum = 1;
fixpipeParams.srcNdStride = 0;
fixpipeParams.dstNdStride = 0;
```

### 矩阵计算

下面介绍如何配置Mmad指令的MmadParams结构体成员，各个变量的具体含义详见[Mmad-矩阵计算](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0249.html)

需要注意的是，当Mmad指令执行时，矩阵计算单元会从L0A/L0B连续读入多个分形参与矩阵乘计算，读入分形的数量根据MmadParams结构体的成员变量m,n,k的取值以及Mmad指令对L0A/L0B上A矩阵/B矩阵各个轴的对齐要求而来

由于Mmad指令，A矩阵分形为(16,32), B矩阵分形为(32,16)来连续读入分形，也就是说矩阵计算单元从L0A/L0B连续读入的分形总数，L0A为CeilDivision(m, 16)\*CeilDivision(k,32), L0B为CeilDivision(k,32)\*CeilDivision(n,16)

因此当L0A/L0B上对A矩阵和B矩阵在各个轴的实际对齐要求与Mmad指令默认的对齐要求不一致时，就可能导致读入连续分形时，错误读入完全由无效数据填充的分形而忽略了有效数据的情况

如图8所示，以输入数据为int8，A/B矩阵为RowMajor为例，A shape(30, 60), B shape(60, 70)，L0A在M轴和K轴分别向16, 32对齐，L0B在K轴和N轴分别向32, 16\*2对齐，而Mmad指令默认在M/K/N三轴的对齐要求分别是16/32/16, 因此此时N轴实际对齐要求与Mmad指令默认要求不一致。

如图8左图所示，如果设置mmadParams.n = 70，就会导致读入编号为5的分形，同时没能包含编号为10的分形的有效数据

如图8右图所示，如果设置mmadParams.n=CeilAlign(n, FractalShape\[0]\*FractalNum)=96, 此时会读入所有分形。虽然矩阵计算结果中包含了无效数据，但是在Fixpipe指令搬出数据时通过设置fixpipeParams.nSize可以保证无效数据参与计算的结果不会被搬出。

![图8: B矩阵RowMajor，int8数据类型下, N轴实际对齐要求](https://raw.gitcode.com/user-images/assets/9091846/70f3155f-4db7-4c4c-a6a4-95a42c1aa6c1/AscendC-basic-knowledge-image-7.png)

与上述场景类似，当输入数据类型为float且A矩阵为ColumnMajor时，K轴实际对齐要求与MMad指令默认的对齐要求也不一致，但是此种场景下的解决方案与上述场景有所不同，需要单独引入mmadParams.kDirectionAlign参数来解决

根据矩阵乘法的计算公式可知，K轴作为A/B矩阵的公共维度，此时如果像上述场景那样设置mmadParams.k=CeilAlign(k, fractalSh2ape\[1]\*fractalNum)会导致C矩阵中每个元素的数值都受到多读入的无效数据的影响，并且也不同通过设置fixpipeParams的参数在搬出阶段舍弃无效数据

如图9所示，mmadParams.kDirectionAlign仅在输入数据类型为float时生效。当A矩阵是ColumnMajor时，kDirectionAlign设置为true，此时L0A上A矩阵在K方向16对齐，矩阵计算单元从L0A读取数据会跳过填充的无效数据，其余场景下该参数取默认值为false，此时L0A上A矩阵在K方向按8对齐

也就是说，针对L0A矩阵为float的读入时，默认K方向按8元素对齐。但如果是A矩阵转置且为FP32的时候，L1上为nZ的8\*16排布，因为要合并成16\*16的搬运，所以在K方向就是16对齐。

如果kDirectionAlign=false，K方向按8对齐，会读入0-9的分形。读入了5分形未读入10分形，导致计算错误

如果kDirectionAlign=true，K方向即按16元素对齐，并且在连续物理地址读取时跳过5号分形，正确读入数据到cube

所以设置kDirectionAlign=true的目的，就是把L0A上对齐到16并且在计算时候skip不该算的那部分

![图9: A矩阵转置，float数据类型下，K轴实际对齐情况](https://raw.gitcode.com/user-images/assets/9091846/717fe7f3-2267-4603-8e45-a894ad6f5569/AscendC-basic-knowledge-image-8.png)

```cpp
AscendC::MmadParams mmadParams;
// 左矩阵Height
mmadParams.m = m;
// 右矩阵width
mmadParams.n = n;
if constexpr (AscendC::IsSameType<T, int8_t>::value && AscendC::IsSameType<U, int32_t>::value) {
    if constexpr (!isBtranspose) {
        // mmad默认n轴向16对齐，但是由于b转置过程n轴向2 * 16对齐，填充了一个全部由无效数据的32 * 16的分形，
        // 如果仍然设置mmadParams.n = n，cube单元会多读入无效数据的分形同时有效数据的分形也未被读入。
        // 此时可以通过设置n向32对齐，让此分形参与计算，搬出时跳过无效分形参与计算的得到的分形即可
        mmadParams.n = CeilAlign(n, fractalShape[0] * fractalNum);
    }
}
// 左矩阵Width、右矩阵Height
mmadParams.k = k;
if constexpr (AscendC::IsSameType<T, float>::value && AscendC::IsSameType<U, float>::value) {
    if constexpr (isAtranspose) {
        mmadParams.kDirectionAlign = true;
    }
}
```

## 同步指令

于AIC核/AIV核内部的执行单元（如MTE2搬运单元、Vector计算单元等）以异步并行的方式运行，在读写同一存储资源时可能存在数据依赖关系。为确保数据一致性及计算正确性，需通过同步控制协调操作时序

### 核内同步

AIC/AIV内并行的指令流水一共有8条

| 流水类型   | 含义                                                     |
| ---------- | -------------------------------------------------------- |
| PIPE\_S    | 标量流水线，使用Tensor GetValue函数时为此流水            |
| PIPE\_V    | Vector流水线                                             |
| PIPE\_M    | Cube流水线                                               |
| PIPE\_MTE1 | L1 buffer出发的流水线，包括L1->L0A, L1->L0B, L1->BT      |
| PIPE\_MTE2 | GM出发的流水线，包括GM->L1, GM->L0A, GM->L0B, GM->UB     |
| PIPE\_MTE3 | UB/L1回到GM的流水线，包括UB->GM，L1->GM                  |
| PIPE\_FIX  | Fixpipe相关的流水线，包括L0C->GM，L0C->L1，L1->FP buffer |
| PIPE\_ALL  | 所有流水                                                 |

![](https://raw.gitcode.com/user-images/assets/9091846/718dc2fa-e681-453f-9b6f-0a6da9313fd6/AscendC-basic-knowledge-image-9.png)

### 核间同步

核间同步机制仅适用于AIC/AIV Mix算子，不适用于纯AIC算子或纯AIV算子。（AIC指Cube；AIV指Vector）。

核间同步常用指令集有：

- [SyncAll](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0204.html)
- [CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0273.html)配套[CrossCoreWaitFlag](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0274.html)

其中`CrossCoreSetFlag`配套`CrossCoreWaitFlag`实现的同步控制分为以下几种模式，：

- `模式0`：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
- `模式1`：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
- `模式2`：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。

![](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/figure/zh-cn_image_0000002562462709.png)
