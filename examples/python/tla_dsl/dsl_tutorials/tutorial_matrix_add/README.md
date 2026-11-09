# 矩阵加法编程教程

本教程通过四个版本的FP32矩阵加法，逐步学习CATLASS Python的数据划分、数据搬运、计算与同步，再尝试双缓冲和UB布局优化

建议按v0→v1→v2→v3的顺序阅读，每一步都回答三个问题：上一版有什么限制、本版修改了什么、需要用什么实验判断修改是否有效

四个版本计算相同的结果：输入为两个连续RowMajor矩阵`A[M, N]`和`B[M, N]`，输出为`C[M, N]`，满足`C[m, n] = A[m, n] + B[m, n]`，本教程主线使用默认的`8192×8192`输入，每个FP32元素占4字节



## 学习路线

| 步骤 | 本版引入的知识 | 阅读后应能回答的问题 |
| --- | --- | --- |
| 公共流程 | Host与kernel分工、按行分核、Tensor视图、tile分块 | 每个核处理哪些数据，哪些API真正搬运数据 |
| v0 | 单缓冲、Scalar逐元素计算、显式flag | 为什么搬入、计算、搬出之间需要同步 |
| v1 | SIMD寄存器计算、资源mutex | 如何用向量操作替换逐元素计算，何时可以复用UB |
| v2 | 多缓冲、编译期槽位展开 | 为什么不同tile需要独立空间，槽位怎样轮转 |
| v3 | 分配大小、对齐、顺序与UB布局 | 地址布局为什么可能影响性能，如何验证Bank冲突 |



## 代码组织

| 文件 | 作用 |
| --- | --- |
| host.py | 统一运行入口，选择版本、生成数据、分配设备Tensor、编译并启动kernel、保存和校验结果 |
| matrix_add_common.py | 公共Init函数，完成按行分核及每核GM视图构造 |
| matrix_add_v0.py | 单缓冲、标量计算和显式flag同步 |
| matrix_add_v1.py | 单缓冲、SIMD计算和mutex同步 |
| matrix_add_v2.py | 在SIMD实现上增加多缓冲，默认使用双缓冲 |
| matrix_add_v3.py | 在多缓冲基础上调整UB分配大小、对齐及顺序，尝试减少bank冲突 |
| scripts/gen_data.py | 生成FP32输入及CPU参考结果 |
| scripts/verify_result.py | 读取二进制文件并逐元素校验结果 |

## 第一步：理解公共计算流程

### Host准备数据，kernel处理设备上的数据

host.py通过`KERNELS`选择版本，生成CPU输入与golden，将输入放到NPU，并使用`from_dlpack`构造DSL可以接收的Tensor参数

```python
artifact = tla.compile(
    kernel, tla_a, tla_b, tla_c, config, options="--npu-arch 3510"
)
artifact(tla_a, tla_b, tla_c, block_num=block_num)
torch.npu.synchronize()
```

这里包含编译、启动、等待完成三个阶段，`config: tla.Constexpr`提供编译期配置，设备端的`block_idx()`与`block_num()`用于确定当前核的数据范围，最后由Host取回结果并验证

`@tla.kernel`标记kernel入口，公共`Init`使用`@tla.jit`供kernel调用，设备代码虽然采用Python语法编写，动态循环、条件和同步由DSL编译处理，不能简单当作普通Python循环理解

### 按行分核：让每个核处理连续数据

设启动核数为`P`，公共Init先计算：

```text
baseCoreM = M // P
remainderM = M % P
```

前`remainderM`个核处理`baseCoreM + 1`行，其余核处理`baseCoreM`行，每核起始行随前面核的数据量累加

以固定输入`M=N=8192`、启动核数`P=56`为例说明分配过程，实际启动核数由设备决定：

| 核编号 | 行数actualCoreM | 起始行startM | 一维元素起点core_start | 元素数core_data_len |
| --- | --- | --- | --- | --- |
| 0 | 147 | 0 | 0 | 1204224 |
| 15 | 147 | 2205 | 18063360 | 1204224 |
| 16 | 146 | 2352 | 19267584 | 1196032 |
| 55 | 146 | 8046 | 65912832 | 1196032 |

`8192 // 56 = 146`，余数为16，因此前16个核各处理147行，其余40个核各处理146行，总计8192行，选择行方向划分是因为输入为连续RowMajor矩阵，每核负责的多行可以直接表示为一段连续元素

```python
core_data_len = actualCoreM * config.n
core_start = startM * config.n
core_layout = tla.make_layout(
    tla.make_shape(core_data_len),
    tla.make_stride(1),
    layoutTag=tla.arch.RowMajor,
)
core_coord = tla.make_coord(core_start)
gm_a_core = tla.make_tensor(gm_a.ptr, core_layout, coord=core_coord)
```

`core_start`和`core_data_len`以元素为单位，`make_tensor`在原GM指针上构造视图，这一步没有复制矩阵，B和C使用相同的划分

### 按tile分块：把大矩阵映射到有限的UB空间

每核的数据再按`dataBlockLen=4096`个元素分块，tile数量为`ceil_div(core_data_len, dataBlockLen)`，完整tile中每个输入占`4096×4=16384`字节

```python
gm_a_tile = tla.tile_view(
    gm_a_core, tla.make_shape(config.dataBlockLen), tla.make_coord(i)
)
ub_a_tile = tla.make_tensor_like(ub_ptr_a, gm_a_tile, tla.arch.RowMajor)
tla.copy(ub_a_tile, gm_a_tile)
```

这里`i`表示第几个tile，`tile_view`构造GM子视图，`make_tensor_like`使用已分配的UB指针构造对应视图，真正从GM向UB传输数据的是`tla.copy`，本例参数顺序为目标在前、源在后

| 操作 | 在本例中的作用 | 是否实际搬运数据 |
| --- | --- | --- |
| `tla.allocate` | 分配UB存储空间 | 否 |
| `make_tensor`、`tile_view`、`make_tensor_like` | 描述存储上的形状、布局和访问范围 | 否 |
| `tla.copy(ub, gm)` | MTE2将输入搬入UB | 是 |
| `load()`、`store()` | SIMD区域中UB与寄存器之间的数据传递 | 是 |
| `tla.copy(gm, ub)` | MTE3将结果写回GM | 是 |

默认配置下`N=8192`，每行恰好包含两个4096元素的tile，因此每核的tile数量为`actualCoreM×2`，不存在不足4096元素的尾块；每个tile又恰好包含64组64元素的SIMD计算

代码使用`origin_shape[0]`取得有效长度，在本教程默认配置下，每个完整GM tile的有效长度均为4096；修改矩阵尺寸或`dataBlockLen`后可能出现尾块，需要补充对应的正确性验证

`tla.allocate`的第一个参数是元素数量或形状，最后一个参数是字节对齐要求，例如`tla.allocate(4096, tla.Float32, tla.AddressSpace.ub, 256)`申请4096个FP32元素，占16384字节，起始地址按256字节对齐，不能直接把字节数作为第一个参数

## 第二步：v0——用Scalar计算跑通数据流

### 实现方式与关键代码

阅读matrix_add_v0.py，本版为A、B、C各分配一块UB，先搬入A和B，再逐元素相加，最后搬出C

```python
tla.copy(ub_a_tile, gm_a_tile)
tla.copy(ub_b_tile, gm_b_tile)
tla.set_flag(ub_loaded)
tla.wait_flag(ub_loaded)

for j in tla.range(gm_a_tile.origin_shape[0]):
    ub_c_tile[j] = ub_a_tile[j] + ub_b_tile[j]

tla.set_flag(scalar_done)
tla.wait_flag(scalar_done)
tla.copy(gm_c_tile, ub_c_tile)
tla.set_flag(ub_stored)
tla.wait_flag(ub_stored)
```

完整tile执行4096次标量加法，三块UB的申请数据量合计`3×4096×4=49152`字节，即48KiB，每块按256字节对齐

### 为什么需要三组flag

不同流水之间存在生产和消费关系，源代码里先写搬运再写计算，并不能代替跨流水的数据依赖同步

| flag | 依赖方向 | 等待的目的 |
| --- | --- | --- |
| `ub_loaded` | MTE2→SCALAR | A、B搬入完成后才能读取 |
| `scalar_done` | SCALAR→MTE3 | C计算完成后才能搬出 |
| `ub_stored` | MTE3→MTE2 | 本轮搬出完成后再开始下一轮复用 |

本版在每个tile内显式等待，适合先理解完整的数据生命周期，循环结束后的`pipe_barrier(ALL)`用于收尾，不能替代循环内的依赖同步

### 性能观察与下一步

纯scalar实现，从profiling可以看到，scalar占比占满


```Bash
 #执行命令，注意填写kernel-name
 msopprof --application="python host.py --version=v0" --kernel-name=matrix_add_v0
```

计算8192*8192个fp32累加，算子时间16328us

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=NmI1OGVhNjE5MjhmZDJkNGY0YTExOTU3MDBhZGZiMDlfWWtIRDBha3VwYkkxS0RUREtLZWVROEhXSWZzWENaTzVfVG9rZW46VTVqSmIyNnRDb0hQMVZ4UVRJZmNjdkFBbmxmXzE3OTE3NjgwOTk6MTc5MTc3MTY5OV9WNA&add_watermark=true&scene_type=CCM)
![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=NTEyYTI1MDQ0OTA4YWEwMGNlMWQzZDBiZDY3YmE1N2FfZk9DYlA0eEJtTndQOGQwWVNyWlVzbkZxanpLd0tsbG9fVG9rZW46Q0RqRGJESGN3b3d0NHl4TWVmMWNrVWd2bkdlXzE3OTE3NjgxMDY6MTc5MTc3MTcwNl9WNA&add_watermark=true&scene_type=CCM)
下一步保留分核和GM分块方式，把逐元素加法替换为SIMD寄存器计算

## 第三步：v1——使用SIMD与资源同步

### 从逐元素计算变成寄存器向量计算

阅读matrix_add_v1.py，本版仍然使用A、B、C各一块UB，每块容量和GM tile大小与v0一致，在`tla.vec.func(mode="simd")`内部按`VL_LEN=64`继续划分

以下为循环内的关键代码，完整循环及三个UB子视图的构造见源文件：

```python
reg_a = ub_vl_a.load()
reg_b = ub_vl_b.load()
reg_c = tla.add(reg_a, reg_b)
ub_vl_c.store(reg_c)
```

对完整的4096元素tile，源码中共有`4096/64=64`组寄存器计算，每组最多处理64个FP32元素，即每个输入256字节；这不表示整个tile只执行一条机器指令，也不能据此直接推导64倍加速

`ub_vl_a`是UB上的视图，`reg_a`是加载得到的寄存器值，`tla.add`完成向量加法，`store`将结果写回UB，之后才通过MTE3写回GM

### 用mutex表达资源的使用与复用

v1同时把显式flag改为按资源管理的mutex，因此v0→v1的整体耗时变化同时包含计算和同步写法变化

| 资源 | 生产阶段 | 消费阶段 | 下一轮复用前需要保证 |
| --- | --- | --- | --- |
| A的UB | MTE2搬入A | VECTOR读取A | 上一轮读取已完成 |
| B的UB | MTE2搬入B | VECTOR读取B | 上一轮读取已完成 |
| C的UB | VECTOR写入C | MTE3搬出C | 上一轮搬出已完成 |

例如A的搬入被包在对应流水的`lock/unlock`之间：

```python
mutex_ub_a.lock(pipe=tla.arch.MTE2)
tla.copy(ub_a_tile, gm_a_tile)
mutex_ub_a.unlock(pipe=tla.arch.MTE2)
```

计算阶段再使用同一个`mutex_ub_a`在VECTOR流水上声明读取区间，B与C分别使用各自的mutex，关注的是“哪块资源由哪个流水使用”，而非把mutex看作Host线程锁

### 性能观察与下一步

SIMD vector累加实现，搬入register累加

从测试数据可以看到，执行时间和scalar时间明显下降

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=M2EwOTI0NmRhYzA3ODY5MjM3YTkxYWJiZTE2NGE3NjNfWnZHcW91TmpMS3dzcDFScGtvWXRyeWozUFBhcHBNSGtfVG9rZW46WFlVaWJ0NzMyb3NsVEl4eDFKc2M3YzBLblRnXzE3OTE3NjgxMzc6MTc5MTc3MTczN19WNA&add_watermark=true&scene_type=CCM)
![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=NWUyNTY4YTg1ZDhjODQ5ODBmYWQ0NDk1ZmM5ZWYxNjVfWVJ4WWZGbXNLUXdkT3YzTWo3UnNsZFVCZmN0S0pzaTRfVG9rZW46Q0ZyT2JCZnZWb1dYNU54WUdBQmNkMHZsbm1RXzE3OTE3NjgxNDY6MTc5MTc3MTc0Nl9WNA&add_watermark=true&scene_type=CCM)
这里可以看到mte2时间占比是50%左右，aiv计算40%左右，仍然有比较多的时间花费在数据搬运，而且两者没有并行起来

```Bash
# 仿真
msopprof simulator --soc-version=Ascend950PR_9579 --application="python host.py --version=v1" --kernel-name=matrix_add_v1
```
![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=MWFlNmIzNDg5NjU2NmZjZTFlMTFkN2Q4YTk5YWM0MmRfNDJrWkRtcXBmck5iM2ViZ0ZwRHIwNnU3Vmt6VkhFYVNfVG9rZW46Skw4T2JpWG1xb200em94RkpkUGMyZ0dNblJmXzE3OTE3NjgxNjg6MTc5MTc3MTc2OF9WNA&add_watermark=true&scene_type=CCM)



## 第四步：v2——引入双缓冲

### 缓冲槽位与tile编号是两个概念

阅读matrix_add_v2.py，A、B、C各分配`buffer_num`份空间，每份有独立mutex，默认`buffer_num=2`，因此共有六块UB缓冲和六个mutex

每块容量由属性`config.UB_STAGE_BYTES`按`dataBlockLen × ELEM_TYPE_LEN`计算，再除以`ELEM_TYPE_LEN`转换成`tla.allocate`需要的元素数；默认每块4096个FP32元素、16384字节，六块共98304字节，即96KiB，每块按256字节对齐

```python
for group_idx in tla.range(ceil_div(gm_tile_count, config.buffer_num)):
    for buffer_idx in tla.range_constexpr(config.buffer_num):
        slot_idx = group_idx * config.buffer_num + buffer_idx
        if slot_idx < gm_tile_count:
            # 使用buffer_idx选择UB和mutex，使用slot_idx选择GM tile
            ...
```

以上为控制流节选，省略号处对应源码中的搬入、计算和搬出，本教程每核tile数量为偶数，默认双缓冲的前几组对应关系如下：

| group_idx | buffer_idx | slot_idx | 行为 |
| --- | --- | --- | --- |
| 0 | 0 | 0 | tile 0使用槽位0 |
| 0 | 1 | 1 | tile 1使用槽位1 |
| 1 | 0 | 2 | tile 2复用槽位0 |
| 1 | 1 | 3 | tile 3复用槽位1 |
| 2 | 0 | 4 | tile 4复用槽位0 |
| 2 | 1 | 5 | tile 5复用槽位1 |

固定输入下，每核tile数量为`actualCoreM×2`，默认双缓冲的最后一组也有两个有效tile，源码中的`slot_idx < gm_tile_count`判断仍然保留

`buffer_idx`通过`tla.range_constexpr`在编译期展开，便于索引Python列表中的UB指针和mutex；`group_idx`与`slot_idx`用于设备上的动态遍历，不能用运行期tile编号直接替代编译期列表索引

### 双缓冲为什么可能减少等待

同一槽位仍须遵守上一版的资源生命周期，但tile 1可以使用槽位1，而不必等待tile 0释放槽位0，为MTE2搬入后续tile、VECTOR计算当前tile和MTE3搬出已完成tile提供重叠条件


### 性能观察与下一步
修改实现为double buffer实现，数据读取和计算可以并行，降低空泡


实测可以看到，加入DB后性能明显提升

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=MDI2YzBkMTUxZmFkMTdjYjBkMGM0OWZhYWRjOWNhOThfYThEaldXcjFnWFUwUnBNWnBUYzR1TFBTWHJLcnQwSjlfVG9rZW46R2tBdWI4OFNCb3Qyd3N4Yko1SmNma1Uwbjd6XzE3OTE3NjgyMTc6MTc5MTc3MTgxN19WNA&add_watermark=true&scene_type=CCM)

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=Mzk4ZWMwY2RiMmY1MDVjMzMyYzlkM2QwNmM4MDRmYjdfVmVLY044SDl5Z0FOa3YxSDdkbU9kbEZKYXEyS3hqNFhfVG9rZW46RFMwcmJTQVQ5b3RqNlp4UnU2a2NmRk84bk9tXzE3OTE3NjgyMTc6MTc5MTc3MTgxN19WNA&add_watermark=true&scene_type=CCM)


![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=MWI3NjEyOTU4MjU1YTVkYWI0NzY0MWE3NjQzNTY5MTJfZDdxYVpVeGVQQ01aZ1Yycm9mdlJiZHdLc21oSXBNRnlfVG9rZW46SDJ0eWJiN1FMb2xwQjZ4WFRaQmNOc3N5bjNMXzE3OTE3NjgyNTQ6MTc5MTc3MTg1NF9WNA&add_watermark=true&scene_type=CCM)

## 第五步：v3——调整UB布局并验证Bank冲突

### 本版具体修改了什么

阅读matrix_add_v3.py，本版保留多缓冲设计，改变了每块缓冲的容量、对齐和分配顺序

| 项目 | v2默认配置 | v3默认配置 |
| --- | --- | --- |
| 每块申请元素数 | 4096 | 4128，即4096+32 |
| 每块申请字节数 | 16384 | 16512 |
| 六块申请字节数合计 | 98304 | 99072 |
| 起始地址对齐 | 256字节 | 128字节 |
| 源码分配顺序 | A0、B0、C0、A1、B1、C1 | A0、B0、A1、B1、C0、C1 |
| 每个GM tile最大元素数 | 4096 | 4096 |

```python
data_copy_len = config.BANK_CONFLICT_DATA_COPY_BYTES // config.ELEM_TYPE_LEN

for buffer_idx in tla.range_constexpr(config.buffer_num):
    ub_a_ptr[buffer_idx] = tla.allocate(data_copy_len, tla.Float32, tla.AddressSpace.ub, 128)
    ub_b_ptr[buffer_idx] = tla.allocate(data_copy_len, tla.Float32, tla.AddressSpace.ub, 128)
for buffer_idx in tla.range_constexpr(config.buffer_num):
    ub_c_ptr[buffer_idx] = tla.allocate(data_copy_len, tla.Float32, tla.AddressSpace.ub, 128)
```

`BANK_CONFLICT_DATA_COPY_BYTES`使用整数增量128，默认返回16512字节，除以每元素4字节后得到整数4128，可作为`tla.allocate`的静态形状

每块多出的32个FP32元素占128字节，用于改变分配间距，GM数据长度、行跨度和有效计算长度都没有因此增加，六块缓冲的申请数据量合计99072字节

目标架构为3510，UB采用低位地址交织，字节地址对应的Bank Group为`(addr >> 5) & 7`，组内Bank为`(addr >> 8) & 1`，不能将256字节连续块理解为同一个Bank内连续的一行

按当前分配顺序、128字节对齐和整数容量计算，相对于UB起点的六块地址依次为`A0=0x00000`、`B0=0x04080`、`A1=0x08100`、`B1=0x0C180`、`C0=0x10200`、`C1=0x14280`

128字节错位改变了访问映射，但一次64个FP32元素的连续加载覆盖256字节，A0/B0的Bank集合仍部分重叠，A0/C0在相同元素偏移下也具有相同的Bank映射，因此不能仅凭padding认定冲突已经消除，需要结合实际指令调度和profiling判断

参考[避免Bank冲突（NPU架构版本3510）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md)；[官方add_high_performance样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance)包含不同代际的配置，比较时需要核对架构、元素类型、地址间距和测量条件



v3同时改变了多个布局参数，v2→v3的耗时差只能直接说明两份实现的整体差异；若要证明具体是哪项布局调整带来收益，需要分别控制容量、对齐或分配顺序开展对照实验

### 实现约束与验证范围

v3已使用独立的`slot_idx`计算动态tile编号，A、B、C三处GM视图均使用该坐标，初始化mutex的编译期变量`i`不再参与动态tile访问

Host在生成数据和分配设备Tensor之前检查tile长度及UB占用，UB占用包含各缓冲区之间的对齐间隙，当前默认编译选项使用248KiB预算，保留编译器所需的8KiB空间，预算依据为[`TlaSimtUbLimits.h`](../../../../../python/tla_dsl/csrc/mlir/include/Passes/TlaSimtUbLimits.h)

默认配置的UB占用分别为：v0/v1为49152字节，v2为98304字节，v3为99072字节，容量检查通过仅表示分配预算满足要求，不代表非默认tile长度及尾块的正确性已得到验证


## 如何执行

### 环境准备

先按照快速开始完成CATLASS DSL安装和环境配置，运行环境需要可用的Ascend 950系列NPU、配套CANN、PyTorch、`torch_npu`及NumPy

当前`host.py`使用`--npu-arch 3510`编译，四个版本均为纯Vector核样例，仅支持FP32

### 运行四个版本

从仓库根目录进入本教程目录，后续命令均在此目录执行：

```bash
cd examples/python/tla_dsl/dsl_tutorials/tutorial_matrix_add

# 使用默认矩阵规模8192×8192和全部Vector核
python host.py --version v0

# 保持输入规模相同，使用全部Vector核依次运行其余版本
python host.py --version v1
python host.py --version v2
python host.py --version v3

```

以上命令均不指定`--m`、`--n`和`--block-num`，矩阵规模使用默认的8192×8192，核数使用默认值`-1`自动获取并启动设备的全部Vector核

`host.py`会自动生成输入和golden，不需要先执行数据生成脚本，每次运行都会重新生成并覆盖当前工作目录下的相关数据文件

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--version` | `v0` | 选择`v0`、`v1`、`v2`或`v3` |
| `--device` | `0` | NPU设备号 |
| `--m` | `8192` | 矩阵行数，本教程保持默认值 |
| `--n` | `8192` | 矩阵列数，本教程保持默认值 |
| `--dtype` | `f32` | 当前仅支持FP32 |
| `--block-num` | `-1` | 启动核数，`-1`表示自动获取设备Vector核数，也可指定正整数 |
| `--dataBlockLen` | `4096` | 每个GM tile的元素数，不是字节数，本教程主线保持默认值 |

`UB_STAGE_BYTES`和`BANK_CONFLICT_DATA_COPY_BYTES`已改为只读属性，每次根据当前实例的`dataBlockLen`和`ELEM_TYPE_LEN`计算，容量分别为`dataBlockLen × ELEM_TYPE_LEN`字节和在此基础上增加128字节，修改`--dataBlockLen`后容量随之变化，两个属性均返回整数

`VL_LEN=64`、`ELEM_TYPE_LEN=4`和`buffer_num=2`是配置字段，没有对应命令行参数，kernel固定使用`tla.Float32`，不能仅修改`ELEM_TYPE_LEN`就切换数据类型

当前Host检查`m/n`和`dataBlockLen`为正数、`block_num`为`-1`或正数，并按所选版本的缓冲区数量、容量及对齐间隙核对248KiB的UB预算，非法配置在生成数据和分配设备Tensor之前抛出`ValueError`，容量检查通过不替代非默认配置的编译及尾块正确性验证

## 输出与校验

运行后生成以下文件：

```text
input/
├── input_x.bin       # FP32矩阵A
└── input_y.bin       # FP32矩阵B
output/
├── golden.bin        # CPU计算的A+B
└── output.bin        # NPU计算结果
```

Host固定NumPy随机种子为0，生成取值范围为`[1, 100)`的输入，将输出Tensor初始化为`-7.0`，编译并启动所选kernel后同步设备，再保存和校验结果

已完成的上板验证覆盖默认FP32矩阵规模`8192*8192`及设备全部Vector核，不包含200个精度用例、20个内存检测用例以及特殊值和确定性专项检测

以下为输出格式示意：

```text
--- version=v2 dtype=f32 m=8192 n=8192 ---
mismatched elements: 0 / 67108864, error ratio: 0.00000000
passed=True cache_key=<cache_key>
kernel.o=<kernel_binary_path>
```

校验要求文件大小和元素数量正确，所有元素为有限值，并且每个元素都满足`np.isclose`，其中`rtol=1e-4`、`atol=1e-5`，Host通过时返回退出码0，校验失败时返回1

也可以单独生成数据或重新校验已有结果：

```bash
python scripts/gen_data.py
python scripts/verify_result.py output/output.bin output/golden.bin
```

两条命令各自独立，生成数据脚本不会执行kernel，重新校验时应使用同一次计算对应的output和golden，避免重新生成golden后与旧output混用

当前Host不输出kernel耗时，`passed=True`仅表示该次输入的正确性校验通过，四个版本的性能差异需要另外采集设备执行时间
