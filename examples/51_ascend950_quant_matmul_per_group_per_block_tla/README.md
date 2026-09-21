# 51_ascend950_quant_matmul_per_group_per_block_tla Example Readme

## 代码组织

```text
├── 51_ascend950_quant_matmul_per_group_per_block_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── matmul_quant_pertile.cpp # 主文件
```

## 功能介绍

- 该算子支持左矩阵采用pergroup，右矩阵采用perblock的量化组合模式。
- cube侧计算时沿L0shape的大小切块，每次计算baseK的数据后，搬运到ub，在vector侧对该部分数据，先取得对应左矩阵的缩放因子和对应右矩阵的缩放因子，再与该块数据计算,
- 最后在vector上完成累加后输出

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 51_ascend950_quant_matmul_per_group_per_block_tla -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./51_ascend950_quant_matmul_per_group_per_block_tla 128 128 128 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
