# StreamkMatmul Example Readme

## 代码组织

```text
├── 66_ascend950_streamk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── streamk_matmul.cpp # 主文件
```

## 模板说明

该模板为切尾轮基本块的多核切K模板，通过切分尾轮基本块的`K`，划分出更多的任务块，从而利用更多的核心参与尾轮的基本块计算。
与`37_ascend950_tail_multi_core_splitk_matmul`不同的是，该模板切分不再通过一个切分因子切分`K`，而是将尾轮的计算量直接均分到所有计算核心上。

```sh
# 编译指定用例
bash scripts/build.sh 66_ascend950_streamk_matmul -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./66_ascend950_streamk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```


## 模板推荐场景

推荐 MNK 范围：约定L1上的分块大小为`m1 n1 k1`，设 `blocks = ceil(M/m1)×ceil(N/n1)`，推荐 `blocks % C ∈ (0, 0.8C) 且 K > 1024`（C 为 AI Core 核数），即尾轮有残余块且不足 0.8 倍核数的深 K 场景。
