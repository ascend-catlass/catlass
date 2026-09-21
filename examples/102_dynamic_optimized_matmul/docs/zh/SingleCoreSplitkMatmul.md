# SingleCoreSplitkMatmul

## 1 模板说明

![image-20260210002058199](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210002058199.png)

CommonMatmul要求各任务块的完整计算结果在一个AI Core上产生。为了避免不必要的L1->L0的重复数据搬运，且能在L0C上完成整个K方向的累加，在CommonMatmul中，要求L0TileM=L1TileM，L0TileN=L1TileN，L0TileK在L0的空间约束下取最大值。

L0C的空间相比L1较小（在A2上L0C为128K，L1为256K），在增大L1TileM和L1TileN的过程中，往往L0C先达到空间约束点，例如最常用的L1TileShape，假设数据类型为half，L1TileM=128，L1TileN=256，L1TileK=256，这时刚好用满L0C（`128*256*sizeof(float)=128K`），但是L1只用了`128*256*2*2+256*256*2*2=384K(双缓冲)`，如果L1TileShape增大为L1TileM=256，L1TileN=256，L1TileK=256，刚好用满L1，但是L0C上无法存放`256*256*sizeof(float)=256K`的基本块，这时就需要进行单核切K，将部分和暂存到GM上，通过原子加累加部分和。在单核切K模板中，可以L0TileM<L1TileM，L0TileN<L1TileN。如图所示，L1TileM=2L0TileM，L1TileN=L0TileN，L0C上每次产生`L0TileM*L0TileN`大小的部分和，之后将其通过原子加累加到GM上的workspace，图中的matmul一共产生了8个部分和，对应位置的部分和进行3次原子加得到最终的结果。

单核切K的收益来源于L1TileM增大，根据前面的理论分析，L1TileM增大，会带来读取数据量的减少，但是每次产生部分和不能在L0C上完成累加，需要写出到GM进行原子加，带来了额外的写出开销。

## 2 优化点

根据不同优化策略，分化出三个SingleCoreSplitK模板。

### 2.1 SingleCoreSplitkKLoopMiddleMatmul

![image-20260210193559636](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210193559636.png)

伪代码描述：

```c++
for gmTileN1 in N:
    for m1 in gmTileM1:
        for k1 in K:
            loadL1A(m1,k1)
            for n1 in gmTileN1:
                loadL1B(k1,n1)
                for m0 in m1:
                    for k0 in k1:
                        loadL0A(m0, k0)
                            for n0 in n1:
                                loadL0B(k0, n0)
                                mmad(A, B)
                                writeC(m0, n0)
```

### 2.1.1 工程优化

- L1常驻

  对于单核切K模板来说，L1常驻是一个自然而然的优化，如上图所示，A矩阵的tile块可以常驻于L1上，例如core0，一个A矩阵tile块需要与两个B矩阵的tile块进行矩阵乘得到部分和，A矩阵tile块只需要加载一次即可，加载后常驻于L1上，减少数据搬运。

- 取消常驻矩阵的double buffer

  由于A矩阵常驻于L1上，所以不需要开辟双buffer空间，多出来的空间可用于增加tile大小。

- 开辟局部workspace，提高写命中率

  每个AICORE分配固定大小的workspace，由于每次Fixpipe都是写出到同一个workspace，对于写出来说，总是缓存命中的，从而提高写出带宽。

- 关闭unitflag，在L0C上采用double buffer

  减小L0TileM或者L0TileN，在L0C上开辟双buffer空间，以实现Mmad计算和Fixpipe写出的高度掩盖，双buffer的掩盖效果要好于使能unitflag。

- 对齐写出

  由于在workspace上使用原子加完成累加，Fixpipe写出到workspace上的时候将stride对齐到512B进行写出，这样能充分发挥NZ2ND写出带宽。

- SetMMLayoutTransform

### 2.1.2 优势与劣势

SingleCoreSplitkKLoopMiddleMatmul的优势在于能够开辟局部workspace空间，每次Fixpipe都是写出到同一个workspace，写总是命中，写出带宽高。

SingleCoreSplitkKLoopMiddleMatmul的劣势在于每个核心负责了比较大的C矩阵块，更容易负载不均衡。

### 2.2 SingleCoreSplitkKLoopOuterMatmul

![image-20260210194055831](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210194055831.png)

伪代码描述：

```c++
for k1 in K:
    for m1 in M:
        loadL1A(m1,k1)
        for n1 in N:
            loadL1B(k1,n1)
              for m0 in m1:
                for k0 in k1:
                    loadL0A(m0, k0)
                    for n0 in n1:
                        loadL0B(k0, n0)
                        mmad(A, B)
                        writeC(m0, n0)
```

SingleCoreSplitkKLoopOuterMatmul必须开辟与C矩阵相同元素量的缓冲区（累加类型与输出类型不同或者C矩阵非对齐的时候）

### 2.2.1 工程优化

- L1常驻，取消常驻矩阵的double buffer，关闭unitflag，在L0C上采用double buffer，对齐写出，SetMMlayoutTransform等优化点与SingleCoreSplitkKLoopMiddleMatmul相同。

- 负载均衡

  SingleCoreSplitkKLoopOuterMatmul模板中，C矩阵基本块按swizzle次序均分到所有AICORE上。

### 2.2.2 优势与劣势

SingleCoreSplitkKLoopOuterMatmul的优势在于比SingleCoreSplitkKLoopMiddleMatmul更容易实现负载均衡。

SingleCoreSplitkKLoopOuterMatmul的劣势在于需要开辟与C矩阵相同元素量的workspace空间，每次计算完整个C矩阵的一层部分和，然后计算下一层，C矩阵很大的时候，写命中率将会很低，也就导致写出带宽低。

### 2.3 SingleCoreSplitkForSmallKMatmul

此模板为针对特殊场景的模板。当K较小的时候，即K<=L1TileK时，K方向不需要进行切分，也就不需要在workspace上通过原子加累加部分和。例如当A、B矩阵类型为half时，如果C矩阵对齐，可以直接写出half数据到C矩阵空间，如果C矩阵非对齐，先将half数据按512B对齐的stride写出到workspace，然后由AIV将workspace中的数据写回到GM C。

![image-20260210212742124](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260210212742124.png)
