# GroupGemm Example Readme

## 功能说明

- 算子功能：完成分组矩阵乘（Group Gemm）计算，各组使用独立的 m、n、k 以及缩放系数 alpha、beta。
- 计算公式：

  $$
    D_i = \alpha_i \times (A_i \times B_i) + \beta_i \times X_i, \quad 0 \le i < g
  $$

  其中 $g$ 为分组数量，各组 $A_i$、$B_i$、$X_i$、$D_i$ 的形状分别为 `(m_i,k_i)`、`(k_i,n_i)`、`(m_i,n_i)`、`(m_i,n_i)`。各组张量沿行方向连续拼接存储，`alpha`、`beta` 为按组给定的标量序列。
- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m_i,k_i)` | `float16` | 各组左矩阵，layout 固定为 `RowMajor`，各组沿行方向拼接 |
| `B` | Input | `(k_i,n_i)` | `float16` | 各组右矩阵，layout 固定为 `RowMajor`，各组沿行方向拼接 |
| `X` | Input | `(m_i,n_i)` | `float16` | 各组累加矩阵，layout 固定为 `RowMajor`，各组沿行方向拼接 |
| `alpha` | Input | `(g,)` | `float32` | 各组的乘积缩放系数 |
| `beta` | Input | `(g,)` | `float32` | 各组的累加系数 |
| `D` | Output | `(m_i,n_i)` | `float16` | 各组计算结果，本样例中与 `X` 复用同一块显存 |

## 使用范围说明

推荐范围：

- 分组数量 `g` 满足 `1 ≤ g ≤ 32`。

分组数量超过 32 时，程序会输出 `Unsupported group count: require groupCnt <= 32.` 并以非零退出码结束，不会启动算子。

## 使用示例

### 命令行参数

```bash
16_group_gemm [groupCnt] [mList] [nList] [kList] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `groupCnt` | 无 | 分组数量，取值范围 1~32 |
| `mList` | 无 | 各组 m 轴大小，按组顺序以英文逗号分隔 |
| `nList` | 无 | 各组 n 轴大小，按组顺序以英文逗号分隔 |
| `kList` | 无 | 各组 k 轴大小，按组顺序以英文逗号分隔 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 16_group_gemm
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./16_group_gemm 3 "128,256,512" "256,512,128" "512,256,128" 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
