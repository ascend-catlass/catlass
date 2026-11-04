# MatmulAdd Example Readme

## 代码组织

```text
├── 03_matmul_add
│   ├── CMakeLists.txt  # CMake编译文件
│   ├── README.md
│   └── matmul_add.cpp  # 主文件
```

## 约束说明

约束：`m`、`n`、`k` 均须大于 0；否则样例在参数解析阶段直接退出。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 03_matmul_add
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./03_matmul_add 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
