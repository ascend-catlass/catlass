# 向量掩码合并

## 功能说明

CATLASS Python 的 `tla.where(mask, x, y)` 按元素选择输出。掩码为真时取 `x` 对应位置的值，为假时取 `y` 对应位置的值。本例将计算结果作为 `x`。将零向量作为 `y` 可以清零未选中的位置，将更新前的目的向量作为 `y` 可以保留原值。

### 加法示例的输入与掩码

以两个 f32 向量相加为例。下表只列出前四个元素，每列对应一个 lane。掩码由 `b > 0` 生成。

| 输入 | lane 0 | lane 1 | lane 2 | lane 3 |
| --- | ---: | ---: | ---: | ---: |
| 向量 `a` | 1 | 2 | 3 | 4 |
| 向量 `b` | 10 | 20 | -30 | -40 |
| 更新前的目的向量 | 90 | 91 | 92 | 93 |
| 掩码 `b > 0` | 真 | 真 | 假 | 假 |

### 加法输出

lane 0 和 1 写入加法结果。lane 2 和 3 取 `tla.where` 第三个参数的对应元素。

| 第三个参数 | lane 0 | lane 1 | lane 2 | lane 3 |
| --- | ---: | ---: | ---: | ---: |
| 零向量 | 11 | 22 | 0 | 0 |
| 更新前的目的向量 | 11 | 22 | 92 | 93 |

掩码全选时，两种写法都返回完整的加法结果。掩码全不选时，输出等于第三个参数。

## 调用示例

在 `tla.vec.func(mode="simd")` 内生成掩码，计算选中 lane 的和。

```python
mask = tla.cmp(b, 0, "gt")
result = tla.add(a, b, mask=mask)
```

**清零未选中的 lane。** 将零向量作为第三个参数。

```python
zero = tla.full(0, dtype=tla.Float32)
out_zero = tla.where(mask, result, zero)
```

**保留未选中 lane 的原值。** 示例先将更新前的目的向量载入 UB 缓冲区 `output_buffer`。在写回前读出该向量，作为第三个参数。

```python
previous = output_buffer.load()
out_merge = tla.where(mask, result, previous)
```

## 各项操作经 `tla.where` 合并后的输出

上例用 `tla.where` 的第三个参数决定未选中位置清零还是保留原值。下表列出本示例脚本中的七项操作。它们都保留未选中位置的原值，第二个参数随操作而变。这样，前两项由各行的操作决定，后两项都保留 92 和 93。

| 操作 | 传给 `tla.where` 的第二个参数 | 合并后的前四项 |
| --- | --- | --- |
| 加法 | `tla.add(a, b, mask=mask)` | `[11, 22, 92, 93]` |
| 取较大值 | `tla.max(a, b, mask=mask)` | `[10, 20, 92, 93]` |
| 取较小值 | `tla.min(a, b, mask=mask)` | `[1, 2, 92, 93]` |
| 填入 7 | `tla.full(7.0, dtype=tla.Float32, mask=mask)` | `[7, 7, 92, 93]` |
| 复制源元素 | `a` | `[1, 2, 92, 93]` |
| f32 转 i32 | `a.to(tla.Int32, ROUND_EVEN, mask=mask)` | `[1, 2, 92, 93]`（i32） |
| i32 转 f32 | `a.to(tla.Float32, ROUND_EVEN, mask=mask)` | `[1.0, 2.0, 92.0, 93.0]`（f32） |

其他运算传入掩码后如何处理未选中位置，见对应接口的参数说明。例如，`tla.relu` 和 `tla.prelu` 的 `mask` 行为见[向量算术接口](../../../../../../python/tla_dsl/docs/zh/api/language/simd/arithmetic.md)。

复制一行的完整写法是 `tla.where(mask, a, previous)`。掩码选中哪里，就从源向量 `a` 复制哪里的值。

类型转换后，传给 `tla.where` 的两个向量要有相同的元素类型和长度。例如，f32 转 i32 后要给未选中位置补零，就用 `tla.full(0, dtype=tla.Int32)` 生成整数零向量。表中的 `ROUND_EVEN` 是这两个转换共用的舍入设置：

```python
ROUND_EVEN = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_RINT)
```

## 归约与写回

### 求和归约

沿用加法示例的 `out_zero` 和 `out_merge`。求和归约使用独立的掩码 `first_four`，选中 lane 0 至 3。

```python
first_four, _ = tla.update_mask(4, dtype=tla.Float32)
sum_zero = out_zero.reduce(tla.ReductionOp.ADD, mask=first_four)
sum_merge = out_merge.reduce(tla.ReductionOp.ADD, mask=first_four)
```

默认归约模式返回单 lane 结果。lane 2 和 3 在加法中未选中，在本次求和中参与计算。

| 未选中 lane 的值 | 参与归约的值 | 和 |
| --- | --- | ---: |
| 零 | `11 + 22 + 0 + 0` | 33 |
| 原值 | `11 + 22 + 92 + 93` | 218 |

最大值归约也会读取补入的零。例如，仅前两个 lane 有效且取值为 -3 和 -5 时，清零后的四项为 `[-3, -5, 0, 0]`，这四项的最大值为 0。

### 多轮更新

继续使用加法示例的 `a`、`b`。第一轮选中前两个 lane，第二轮选中后两个 lane。每轮将更新前的目的向量传给 `tla.where`，下一轮从上一轮的输出继续更新。

| 轮次 | 本轮选中的 lane | 更新前的目的向量 | 本轮输出 |
| --- | --- | --- | --- |
| 第一轮 | 0、1 | `[90, 91, 92, 93]` | `[11, 22, 92, 93]` |
| 第二轮 | 2、3 | `[11, 22, 92, 93]` | `[11, 22, -27, -36]` |

### UB 写回

`tla.where` 决定寄存器各 lane 的值。`store` 的掩码决定哪些 UB 地址会被写入。

假设写入另一块 UB 缓冲区 `target_buffer`，其前四项初值为 `[5, 6, 7, 8]`。下表各行独立执行。带 `mask` 的写回只更新 lane 0 和 1 对应的地址。

| 写回调用 | 写回后 UB 的前四项 | 后两项的来源 |
| --- | --- | --- |
| `target_buffer.store(out_zero)` | `[11, 22, 0, 0]` | 寄存器中补入的零 |
| `target_buffer.store(out_merge)` | `[11, 22, 92, 93]` | 寄存器中保留的原值 |
| `target_buffer.store(out_zero, mask=mask)` | `[11, 22, 7, 8]` | 存储掩码跳过这两个地址，保留 UB 原值 |

随后将整块 UB 复制到输出内存时，复制的是表中写回后的值。

## 运行示例

在已配置 CATLASS Python 的 Ascend 950 环境中，从仓库根目录执行以下命令，在设备 0 上依次运行表中的七项操作。

```bash
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/mask_merging/mask_merging.py
```

[mask_merging.py](mask_merging.py) 验证未选中 lane 保留原值。标量填充使用独立的 `mask_merging_full` kernel，用 `b` 生成掩码并在选中 lane 填 7。其余六项运行 `mask_merging` kernel。两个 kernel 都从 UB 读取目的向量原值，在寄存器中完成合并，再写回同一块 UB。

每次调用处理一个 256 字节的向量寄存器。f32 和 i32 每个元素均占 4 字节，因此每次处理 `256 / 4 = 64` 个 lane。

每项操作覆盖四组掩码：全选、全不选、选中偶数 lane、仅选中前两个 lane。主机端按位比较输出与参考值。f32 目的向量包含负零和指定载荷的 NaN，用于检查未选中 lane 的位模式。加法四组比较均通过时输出 `add: all four masks passed`。
