# GroupedMatmulSliceMGelu Example Readme

## 代码组织

```text
./examples/80_ascend950_grouped_matmul_slice_m_gelu/
├── CMakeLists.txt     // CMake编译文件
├── gen_data.py   // 数据生成脚本
├── 80_ascend950_grouped_matmul_slice_m_gelu.cpp   // 主文件
├── launcher
│   └── grouped_matmul_slice_m_gelu_launcher.hpp   // launcher文件
└── README.md   // 说明文件

```

## 功能介绍

该算子支持A矩阵在m轴切分，然后和B矩阵按照group分组进行矩阵乘。基于Ascend 950架构，使用`MmadPreloadAsyncWithCallbackL0CToUB` dispatch policy、`BlockMmadTla` block组件、`EpilogueElemWiseGeluRegBase` dispatch policy、`BlockEpilogue` block组件、
`TileCopy` tile组件、`TileElemwiseGeluRegbase` tile组件。

## 使用示例

因为GroupedMatmulSliceMGelu参数较多，所以该示例直接在代码中承载输出参数列表`groupList`, 通过`golden::GenerateGroupList`来生成随机切分的序列。
相关输入配置具体详见[ascend950_grouped_matmul_slice_m_gelu.cpp](ascend950_grouped_matmul_slice_m_gelu.cpp)。
如果需要输入grouplist配置(例如通过tensorList方式构造输入)，可以参考python_extension中相应实现

example使用

- 获取代码之后编译相应的算子可执行文件，可参考[快速入门](../../docs/zh/1_Practice/01_quick_start.md)
- 执行算子

```bash

function build() {
    rm -rf ./build
    rm -rf ./output
    bash scripts/build.sh 80_ascend950_grouped_matmul_slice_m_gelu -DCATLASS_ARCH=3510
}
build

group_num=4
m=2048
n=256
k=256
device_id=1

python ./examples/80_ascend950_grouped_matmul_slice_m_gelu/gen_data.py "$group_num" "$m" "$n" "$k" "$device_id"

./output/bin/80_ascend950_grouped_matmul_slice_m_gelu $group_num $m $n $k $device_id
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
