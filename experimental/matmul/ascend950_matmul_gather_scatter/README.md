# Ascend 950 MatmulGatherScatter

> **注意**：本样例位于 `experimental/` 目录下。如需独立编译运行，请先将样例目录复制到
> `examples/`，并在 `examples/CMakeLists.txt` 的 `EXAMPLE_ASCEND950` 列表中添加
> `ascend950_matmul_gather_scatter`。

## 功能说明

本样例在单次 MIX kernel 中完成输出清零、Gather、Cube Matmul 和 Scatter：

```text
D[indices, :] = A[indices, :] @ B
```

- `A/B/D` 为 FP16 RowMajor，`indices` 为 INT32。
- `A.shape=(M,K)`、`B.shape=(K,N)`、`indices.shape=(J,)`、`D.shape=(M,N)`。
- 未被 `indices` 选中的 `D` 行保持为零。
- 不创建 `(J,K)` 或 `(J,N)` 的 GM 中间张量。

## 参数说明

| 参数名 | 描述 | 约束 |
| --- | --- | --- |
| `m` | 输入 A 和输出 D 的物理行数 M | `M>0` |
| `j` | indices 长度和参与 Matmul 的逻辑行数 J | `0<J<=M` |
| `n` | 输入 B 和输出 D 的列数 N | `N>0` |
| `k` | 输入 A 的列数和输入 B 的行数 K | `K>0` |
| `device_id` | NPU 设备编号，默认值为 0 | 可选，在设备有效范围内 |

## 约束说明

- indices 的范围和唯一性由调用方保证。
- 任务测试集 shape 使用 FullLoadA/Cube 路径；当完整 Gather A tile 无法安全驻留 L1，或 K 不满足 Cube
  对齐要求时，切换到 AIV SIMT 泛化路径。
- 泛化路径支持任意正数 J/N/K，包括 K 大于 2048 或非 16 对齐的场景。
- 目标架构为 Ascend 950，构建时需要设置 `CATLASS_ARCH=3510`。

## 实现说明

Kernel 按当前设备的 AIC 数量启动，以 MIX `(1 AIC, 2 AIV)` 使用 Cube Core 和 Vector Core：

1. AIV 并行清零 D，并与首个 Gather/Matmul tile 重叠。
2. Host 根据 K 直接分发到 SIMT、AIV Gather FullLoad 或 AIC Gather FullLoad 静态 kernel；每种 Gather
   模式只保留一个 TileShape，不在设备侧执行 selector。
3. AIV MTE Gather 与 AIC 直接 Gather 使用不同的 FullLoad kernel 类型；Dispatch Policy 承载 Gather
   模式和输出 UB 缓冲配置，将逻辑 A tile 转换为 L1 zN。
4. B 使用 CATLASS 的 GM/L2 到 L1 多阶段流水，Cube 使用 FP32 累加。
5. AIV 将输出 Scatter 到 `D[indices, :]`。
6. FullLoadA 不适用时，AIV SIMT 泛化模板直接执行 Gather、FP32 累加和 Scatter。

FullLoadA 只表示 Gather 后的逻辑 A tile `(TileM,K)` 沿 K 维完整驻留 L1，不会将物理 A `(M,K)`
整体搬入 L1。同一 A tile 可在多个 N tile 间复用。

## 代码组织

```text
ascend950_matmul_gather_scatter
├── CMakeLists.txt
├── README.md
├── matmul_gather_scatter.cpp
└── test_79_ascend950_matmul_gather_scatter.py
```

## 使用示例

1. 编译样例：

    ```bash
    bash scripts/build.sh ascend950_matmul_gather_scatter -DCATLASS_ARCH=3510
    ```

2. 运行指定 shape：

    ```bash
    ./output/bin/ascend950_matmul_gather_scatter 400 128 256 2048
    ```

3. 指定设备编号：

    ```bash
    ./output/bin/ascend950_matmul_gather_scatter 400 128 256 2048 0
    ```

程序输出 `Compare success.` 表示数值结果和未选中行清零检查均通过。

对应的 Python optest 接口为：

```python
torch_catlass.ascend950_matmul_gather_scatter(a, b, indices)
```

optest 测试用例保存在当前 experimental 样例目录，不会被全局 CI 自动收集。复验时可将
`test_79_ascend950_matmul_gather_scatter.py` 放入 `tests/optest/tests/` 后，按
`tests/optest/README.md` 的步骤执行。
