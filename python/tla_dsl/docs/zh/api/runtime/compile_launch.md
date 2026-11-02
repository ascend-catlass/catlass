---
nav_order: 2
---

<!--
手工维护的中文 程序生命周期参考。
英文源：docs/en/api/runtime/compile_launch.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# 编译与启动

[程序生命周期](index.md) | [API 速查](../quick_reference.md)

将装饰后的 kernel 编译为设备二进制并启动。同一份二进制需要多次启动时，用
`tla.compile` 获取可调用的 `JitCompiledFunction`；直接调用它时会延迟创建并复用内部
executor。缓存 / 架构 / IR dump 等非函数参数见
[环境变量](../../kernel_development/core_concepts/env_vars.md)。

## 编译

生成设备二进制。日常入口是 `tla.compile`；
`TlaJitFunction.compile` 是装饰后函数上的底层辅助接口。

### `tla.compile`

**源码：** [`catlass.base_dsl.compiler.CompileCallable.__call__`](../../../../catlass/base_dsl/compiler.py#L118)

功能说明：

编译 `@tla.kernel` 函数，返回可调用的
`JitCompiledFunction`。这是公开的 `tla.compile` 入口；调用返回的对象即可启动
（`compiled(*tensors, block_num=...)`）。适合编译一次、同一二进制多次启动。

函数原型：

```python
tla.compile(func: Any, *args: Any, **kwargs: Any) -> JitCompiledFunction
```

参数说明：

- *`func`*（`TlaJitFunction | Any`）：被 `@tla.kernel` 装饰的函数，或由 `@tla.jit __call__` 指出其 kernel 的类实例。必填。
- *`args`*（`Any`）：作为编译类型样本的 Host tensor / 标量 / `@dataclass` 实例（如 `from_dlpack` 或 `make_fake_tensor` 的返回值）；不必绑定 NPU 缓冲。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名，全部可用取值见[编译选项](#编译选项)。缓存 / IR dump / 强制重编译由 `CATLASS_DSL_*` 环境变量控制。`block_num` / `stream` 等启动参数写在返回的编译函数上，而不是这里。

约束说明：

- `func` 必须是 `@tla.kernel` 得到的 `TlaJitFunction`，或由 `@tla.jit __call__` 指出其 kernel 的类实例。
- 从实例读到的 kernel 方法（`实例.kernel`）自带该实例作为 receiver，该实例排在编译类型样本的第 0 位，落进 kernel 的 `self` 参数。
- 传入类实例时编译其 `@tla.jit __call__` 转发到的 kernel，因此类必须声明该 `__call__`。转发必须是 `self.<kernel>(...)` 形式的纯调用，并按自身参数顺序传参；其它形式的 `@tla.jit __call__` 函数体不支持。
- kernel 方法的第一个 `self` 参数是编译期配置：不占启动 ABI，`self.attr` 在编译时折叠为常量。

调用示例：

```python
compiled = tla.compile(copy_kernel, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)  # 启动；再次调用复用同一份二进制

class VectorAddWithStaticLength:
    def __init__(self, n_ele: int, vl_ele: int = 64):
        self.n_ele, self.vl_ele = n_ele, vl_ele

    @tla.jit
    def __call__(self, gm_a, gm_b, gm_c):
        self.kernel(gm_a, gm_b, gm_c)

    @tla.kernel
    def kernel(self, gm_a, gm_b, gm_c):
        ...  # gm_c = gm_a + gm_b；循环次数由 self.n_ele 在编译期确定

compiled = tla.compile(
    VectorAddWithStaticLength(384), gm_a, gm_b, gm_c,
    options="--npu-arch 3510",
)
compiled(gm_a, gm_b, gm_c, block_num=1)
```

---

### `tla.TlaJitFunction.compile`

**源码：** [`catlass.dsl.TlaJitFunction.compile`](../../../../catlass/dsl.py#L198)

功能说明：

编译当前 `@tla.kernel` 函数并返回 `JitCompiledFunction`。
日常 Host 入口是 `tla.compile(fn, *args, options=...)`；只有已持有
`TlaJitFunction` 并需要直接获取编译函数所有者时才调用 `.compile()`。

函数原型：

```python
TlaJitFunction.compile(*, type_args: Sequence[Any] | None = None, **kwargs: Any) -> JitCompiledFunction
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：作为编译类型样本的 Host tensor / 标量。
  可选，默认 `None`（不做张量特化）。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名，
  全部可用取值见[编译选项](#编译选项)。
  缓存 / IR dump 由 `CATLASS_DSL_*` 环境变量控制。

约束说明：

- `type_args` 只作编译期类型样本，不必绑定 NPU 缓冲（`make_fake_tensor` 合法）。
- 用 `options="--npu-arch 3510"` 指定芯片名；不支持的取值在编译时报错。

调用示例：

```python
compiled = my_kernel.compile(
    type_args=[tx, ty],
    options="--npu-arch 3510",
)
```

---

### 编译选项

`tla.compile` 与 `TlaJitFunction.compile` 的 `options` 接受一个字符串，按空格
分隔解析。可用选项如下：

| 选项 | 形式 | 说明 |
| --- | --- | --- |
| `--npu-arch <芯片名>` | 键值 | 指定目标芯片，例如 `3510`。 |
| `--cce-disable-asc-reserved-ubuf` | 开关 | 释放编译器为 Ascend C 预留的 2 KB Unified Buffer。 |
| `--cce-disable-vf-stack-reserved-ubuf` | 开关 | 释放编译器为 VF 栈预留的 6 KB Unified Buffer。 |

约束说明：

- 键值选项必须带取值，缺少取值时编译期报错。
- 开关选项不带取值：写出即生效，不需要再写 `1` 或 `true`。
- Unified Buffer 共 256 KB，编译器默认从中预留 8 KB（Ascend C 2 KB 与 VF 栈
  6 KB），kernel 只能用剩下的 248 KB。上面两个开关分别释放这两段，释放后
  kernel 可用的 Unified Buffer 相应变大。
- **`--cce-disable-vf-stack-reserved-ubuf` 有风险。** VF 栈是编译器溢出（spill）
  向量寄存器的去处；释放之后，若 kernel 触发溢出，编译器没有地方可写，而且
  没有任何检查会报错——结果是静默写坏 Unified Buffer 中相邻的数据。只在确认
  kernel 不会溢出时使用。
- `tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)` 返回整块 256 KB，
  而不是扣掉预留之后的大小。kernel 若按这个值划分缓冲又没有带上这两个开关，
  申请量就会超过实际可用，启动时被拒绝。

调用示例：

```python
compiled = tla.compile(
    matmul_add,
    ta, tb, tc,
    options=(
        "--npu-arch 3510"
        " --cce-disable-asc-reserved-ubuf"
        " --cce-disable-vf-stack-reserved-ubuf"
    ),
)
```

---

## 启动

在 NPU 上运行已编译的 kernel。直接调用 `tla.compile` 返回的
`JitCompiledFunction`。

### `tla.JitCompiledFunction.__call__`

**源码：** [`catlass.base_dsl.jit_executor.JitCompiledFunction.__call__`](../../../../catlass/base_dsl/jit_executor.py#L614)

功能说明：

在 NPU 上启动已编译 kernel，传入运行时入参与 `block_num`、`stream` 等启动参数。
executor 和 binary 在首次调用时延迟加载，后续调用直接复用。

函数原型：

```python
JitCompiledFunction.__call__(*launch_args: Any, *, block_num: int | None = None, args: Sequence[Any] | None = None, **launch_kwargs: Any) -> TlaExecutionResult
```

参数说明：

- *`launch_args`*（`Any`）：位置形式的运行时 kernel 入参，与 `@tla.kernel`
  签名对应（已绑定的 Host tensor、标量或 `@dataclass` 实例）。与 `args=` 互斥。
- *`block_num`*（`int | None`）：启动的 block 数。可选，默认 `1`；传入时须为 `int`。
- *`args`*（`Sequence[Any] | None`）：显式运行时实参序列。可选，默认 `None`。
  不能与非空的 `*launch_args` 同时使用。
- *`stream`*（`Any`，经 `**launch_kwargs`）：可选 ACL stream 句柄。
  省略时使用执行器所在设备的当前 stream。
- *`ub`*（`int`，经 `**launch_kwargs`）：为 kernel 中 `tla.arch.get_dyn_ub`
  声明的动态 UB 区域实际提供的字节数，与 CuTe DSL 用 `launch(smem=...)`
  指定动态 shared memory 的方式对应。可选，默认 `0`。kernel 未声明该区域时
  传入会报错；超过 `artifact.max_ub_bytes` 也会报错。按 kernel 实际使用量
  申请即可：launch 只提供所申请的字节数，SIMT 场景下未被占用的 UB 会成为
  Data Cache，申请过多会拖慢访存。

约束说明：

- `*launch_args` 与 `args=` 不能同时非空（`TlaUnsupportedAbiError`）。
- tensor 启动实参须为已绑定的 NPU 缓冲（`from_dlpack`）；
  `make_fake_tensor` 仅用于编译样本。
- `block_num` 须为 `int`（默认 `1`）。

调用示例：

```python
compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
compiled(args=(tx, ty), block_num=1)
```

---

## 查看 IR

导出前端 TLA IR，不生成设备二进制，也不启动。

### `tla.TlaJitFunction.dump_mlir`

**源码：** [`catlass.dsl.TlaJitFunction.dump_mlir`](../../../../catlass/dsl.py#L259)

功能说明：

返回该 kernel 的 TLA IR（`tlair`）MLIR 文本。不编译设备二进制，也不 launch。

函数原型：

```python
TlaJitFunction.dump_mlir(*, type_args: Sequence[Any] | None = None) -> str
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：类型样本，用法同 `.compile()`。
  可选，默认 `None`。

约束说明：

- `type_args` 规则与 `.compile()` 相同。
- 返回的是前端 TLA IR（`tlair`），不是
  `JitCompiledFunction.artifacts.LLVM` 中的 HIVM/LLVM 形式。

调用示例：

```python
text = my_kernel.dump_mlir(type_args=[fa, fb])
print(text[:500])
```
