# EVG Extension Guide

This document organizes the current EVG extension boundaries. It answers two questions: when to add a `ComputeFn` and when to add a node, and what constraints must be followed during extension. For details about the integration methods, see [evg_api](../../3_API/evg_api.md). For details about the overall design, see [01_evg_design](./01_evg_design.md).

## Determining the Extension Type

Do this first to avoid turning a new operator into a new node.

| Scenario                                                                                                                    | Approach             |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Only element-wise computation is performed. The input and output are in the UB. GM, layout, and workspace are not required. | Adding a `ComputeFn` |
| GM read/write, layout management, UB allocation, or your own `Arguments / Params` is required.                              | Adding a node        |

A simple rule of thumb:

- If you only extend how to compute, add a `ComputeFn`.
- If you involve how to load, how to store, how to occupy resources, add a node.

Current reference implementations:

- `ComputeFn`: `Add`, `Muls`, `LeakyRelu`, `AddRelu`
- Nodes: `VisitorAuxLoad`, `VisitorAuxStore`, `VisitorRowBroadcast`

## Adding a ComputeFn

### Location

Currently, `ComputeFn` is placed in:

- `include/catlass/epilogue/fusion/operations.hpp`

`VisitorCompute` instantiates operators here using the template parameter `ComputeFn`. When adding an element-wise operator, follow the existing pattern in this file. Typically, no separate node is needed.

### Required Form

The minimum form is as follows:

```cpp
template <typename T>
struct SomeOp {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1
    ) const {
        // do compute
    }
};
```

Fixed rules:

- The first parameter is the output `dst`.
- The second parameter is `compute_length`.
- Subsequent parameters are inputs.
- The number of inputs is determined by the operator semantics.

If the operator has scalar parameters, package them into an aggregate type.

```cpp
template <typename T>
struct ClampMin {
    T min_value;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src
    ) const {
        AscendC::Maxs(dst, src, min_value, compute_length);
    }
};
```

Corresponding integration:

```cpp
using ClampMinOp =
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::ClampMin, ElementC, ElementC>;

typename ClampMinOp::Arguments args{{0.0f}};
```

### Current Constraints

#### Only Pure Computation

In the current implementation, `ComputeFn` only carries out computation within UB. The following responsibilities remain at the node layer:

- Accessing GM
- Managing layouts
- Allocating workspace
- Handling event synchronization
- Depending on global coordinates

These responsibilities are not handled in `ComputeFn`.

#### Type Rules Follow `VisitorCompute`

Currently, `VisitorCompute` requires that all input types be equal to `ElementCompute`. Therefore:

- In mixed precision scenarios, `VisitorCast` is typically inserted first.
- Input type compatibility is still handled in the graph, not expanded in `ComputeFn`.

#### Barriers Added for Multi-Step V Computation

If an `operator()` contains multi-step V computation, `AscendC::PipeBarrier<PIPE_V>()` should be added between steps, similar to `AddRelu`.

In short:

- Single atomic instructions usually do not require additional barriers.
- Multi-step chained computation usually requires barriers.

#### Multi-Input Operators Follow Existing Expansion

Multi-input operators like `Add` and `Mul` are currently implemented using chained expansion. When adding a multi-input operator, try to follow existing patterns to avoid introducing a new call convention.

#### Keep Aggregate Initialization Friendly

`VisitorCompute` constructs operators as `Op<ElementCompute>{...}`. When adding a `ComputeFn`, keeping simple fields and aggregate initialization better aligns with the existing implementation.

## Adding a Node

### Location

All node implementations are currently placed in:

- `include/catlass/epilogue/fusion/visitor_*.hpp`

After adding a new node, also integrate it into:

- `include/catlass/epilogue/fusion/fusion.hpp`

### Reference Peer Implementations

When adding a new node, maintain the same structure and responsibilities as existing peer implementations:

- `VisitorAuxLoad`: A typical leaf node responsible for reading a tile from GM.
- `VisitorAuxStore`: A typical root node responsible for final write-back.
- `VisitorRowBroadcast`: A typical cross-stage node. `LOAD` fetches data and `COMPUTE` broadcasts it.
- `VisitorCompute`: A pure compute node. If requirements align with it, reuse it directly.

### Minimum Skeleton

```cpp
template <class Element>
struct VisitorSomeNode : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;

    struct Arguments {};
    struct Params {};

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void* workspace) {
        return Params(/* ... */);
    }

    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const&, Arguments const&) {
        return true;
    }

    struct Callbacks : EmptyCallbacks {
        template <VisitStage Stage, class ArchTag, class TensorC, typename... Inputs>
        CATLASS_DEVICE auto visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            Inputs const&... inputs
        ) {
            // stage-specific work
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        return Callbacks(/* ... */);
    }

    Params params;
};
```

### Required Components and Responsibilities

| Component                 | Purpose                                           | Constraint                                       |
| ------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| `ElementOutput`           | Tells the graph organizer the node's output type. | Must match the actual output.                    |
| `Arguments`               | Parameters filled directly by the user.           | Keep it simple and support brace initialization. |
| `Params`                  | Parameters actually used by the device.           | Keep only the information needed for execution.  |
| `to_underlying_arguments` | Converts `Arguments` to `Params`.                 | Perform workspace mapping here if needed.        |
| `get_workspace_size`      | Declares the node's own workspace.                | Calculate only its own portion.                  |
| `can_implement`           | Performs lightweight validity check.              | Check at least the easily misused parameters.    |
| `Callbacks::visit`        | Executes node actions.                            | Write logic strictly by stage.                   |
| `get_callbacks`           | Allocates UB and constructs callback.             | Whoever allocates UB advances `ub_offset`.       |

### Current Constraints

#### Stage Semantics Must Not Be Disrupted

Node logic is organized in three stages:

- `LOAD`: Prepare inputs and read from GM to UB.
- `COMPUTE`: Compute, transform, and broadcast within UB.
- `STORE`: Write out the result.

Clear rules:

- Actions that truly write back to external addresses go in `STORE`.
- Cross-stage nodes are allowed, but each step is placed separately.
- Actions that belong in `STORE` should not be moved to `COMPUTE`.

#### Node Responsibilities Remain Single

The current implementation favors letting one node take on only one type of responsibility, such as reading, computing, writing, or broadcasting.
Compound behaviors like "Read GM + Compute + Write GM" are better split into multiple nodes, which better aligns with the existing graph organization.

#### Layouts Are Always Interpreted as Full Tensors

If a node has a `layout`, it describes a full GM tensor, not the current tile. The current implementation follows this principle. When adding a new node, keep consistency.

#### UB Allocation Convention Remains Consistent

If `get_callbacks` allocates UB, follow these principles:

- Whoever allocates UB advances `ub_offset`.
- Allocate only the portion needed by itself.
- The size is deduced from `compute_length` and element size.
- The allocation result must not exceed the UB upper limit allowed by the current architecture.

#### Do Not Manage Outer Events Inside a Node

EVG's double buffering and event synchronization are managed uniformly by `BlockEpilogue`. Nodes are only responsible for executing by stage. If a node has multi-step V computation, you may add `AscendC::PipeBarrier<PIPE_V>()`. Outer synchronization is still managed by `BlockEpilogue`.

#### `Arguments` Continues to Support Direct Write

After adding a new node, the user should still be able to directly write aggregate initialization for the entire graph.

```cpp
typename EVG::Arguments args{
    {},
    {deviceX, layoutX},
    {{2.0f}},
    {deviceD, layoutD}
};
```

If a node significantly complicates the parameter initialization of the entire graph, it usually indicates that the interface design is somewhat distant from existing conventions.

#### Reuse Existing Nodes First, Then Decide Whether to Add a New One

If the requirement is simply to add an element-wise operator, continue with `ComputeFn + VisitorCompute`.
Add a new node only when the existing nodes cannot express the required data access, layout, or resource behavior.

### `visit` Signature

The `visit` can be written as a variadic parameter or a fixed signature, depending on the node's responsibilities:

- Leaf nodes or nodes that may extend the number of inputs in the future can use variadic parameters.
- Nodes with strictly single or fixed multiple inputs can write the input signature directly.

The key is not uniformity of form, but that the node itself makes clear:

- How many inputs it needs.
- What the input order is.
- Whether the input types meet expectations.

### Implementation Sequence

When adding a `ComputeFn`:

1. Add an operator to `operations.hpp`.
2. Keep the `operator()(dst, compute_length, inputs...)` form.
3. Add `PipeBarrier` for multi-step V computation.
4. Integrate using `VisitorCompute<..., ElementType, Scalars...>`.
5. Validate parameters and type conventions with a minimal graph.

When adding a node:

1. Create a `visitor_xxx.hpp`.
2. Inherit `VisitorImpl<>`.
3. Define `ElementOutput / Arguments / Params`.
4. Implement `to_underlying_arguments / get_workspace_size / can_implement`.
5. Set stage-specific logic in `Callbacks::visit`.
6. Complete UB allocation in `get_callbacks`.
7. Integrate `fusion.hpp`.
8. Validate composition in `TreeVisitor` or `TopologicalVisitor` with a minimal sample.

## Pre-implementation Checklist

Before adding a `ComputeFn`, at least confirm:

- It is pure element-wise computation.
- It truly does not need GM, layout, or workspace.
- Whether multi-step V computation requires barriers.
- Whether a new node is truly necessary.

Before adding a node, at least confirm:

- The requirement cannot be solved solely by `ComputeFn`.
- Whether `Arguments` can remain simple and support brace initialization.
- Whether `LOAD / COMPUTE / STORE` responsibilities are clearly separated.
- Whether UB or workspace is required.
- Whether the layout is interpreted as a full tensor.
- Whether basic `can_implement` is supplemented.

## References

When extending, refer to these existing implementations:

- `include/catlass/epilogue/fusion/operations.hpp`
- `include/catlass/epilogue/fusion/visitor_compute.hpp`
- `include/catlass/epilogue/fusion/visitor_aux_load.hpp`
- `include/catlass/epilogue/fusion/visitor_aux_store.hpp`
- `include/catlass/epilogue/fusion/visitor_row_broadcast.hpp`

Follow this sequence:

- Adding element-wise operators only: Check `operations.hpp` first.
- Adding read/write nodes: Check `visitor_aux_load.hpp` and `visitor_aux_store.hpp` first.
- Adding cross-stage nodes: Check `visitor_row_broadcast.hpp` first.
