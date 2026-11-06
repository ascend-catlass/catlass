"""Execution-mode lowering that emits Tla MLIR directly while running Python frontend code."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from . import _tla_type_bridge
from . import runtime
from . import tla_ast_decorators as ast_decorators
from .base_dsl.ast_preprocessor import (
    _RecursiveJitHelperError,
    _preprocess_for_lowering,
    reject_user_class_value,
)
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.runtime.argument_tree import (
    ArgumentTreeError,
    build_runtime_argument_trees,
    runtime_leaf_templates,
)
from .base_dsl.runtime.argument_binding import (
    RuntimeArgumentTree,
    RuntimeLeafBinding,
    _RuntimeArgumentProxy,
)
from .base_dsl.utils.tree_utils import tree_unflatten
from .base_dsl.typing import Numeric
from .dsl import (
    _jit_helper_inline,
    is_jit_callable,
    unwrap_jit_callable,
)
from .base_dsl.runtime.jit_arg_adapters import (
    _is_compile_time_callable,
    is_arg_annotation_constexpr,
    receiver_argument_name,
)
from .frontend_diagnostics import (
    FrontendDiagnosticError,
    SourceLocation,
    capture_cause_trace,
    syntax_error_location,
    traceback_location_for_code,
    traceback_location_for_user_code,
)
from .tla.typing import Tensor


class TlaLoweringError(FrontendDiagnosticError):
    """Raised when TLA Python lowering fails."""


class UnsupportedExecutionLowering(FrontendDiagnosticError):
    """Raised when execution-mode lowering cannot safely handle a function."""


_SOURCE_INFO_ATTR = "__tladsl_source_info__"


def _execution_source_error_parts(
    fn: Any, exc: Exception
) -> tuple[str, SourceLocation | None, str] | None:
    info = getattr(fn, _SOURCE_INFO_ATTR, None)
    location = syntax_error_location(exc) if isinstance(exc, SyntaxError) else None
    location = location or traceback_location_for_user_code(exc)
    location = location or traceback_location_for_code(exc, fn.__code__)
    if location is None:
        return None
    filename = location.filename
    if location.filename == fn.__code__.co_filename and isinstance(info, dict):
        filename = str(info.get("filename") or filename)
    return (
        f"Execution-mode lowering failed while running `{fn.__name__}`",
        SourceLocation(
            filename=filename,
            lineno=location.lineno,
            col_offset=location.col_offset,
            end_col_offset=location.end_col_offset,
        ),
        f"{type(exc).__name__}: {exc}",
    )


def _wrap_execution_exception(
    error_type: type[FrontendDiagnosticError], fn: Any, exc: Exception
) -> FrontendDiagnosticError:
    parts = _execution_source_error_parts(fn, exc)
    if parts is None:
        return error_type(
            f"Execution-mode lowering failed while running `{fn.__name__}`: {exc}",
            reason=f"{type(exc).__name__}: {exc}",
            cause_trace=capture_cause_trace(exc),
        )
    summary, location, reason = parts
    return error_type(
        summary,
        location=location,
        reason=reason,
        cause_trace=capture_cause_trace(exc),
    )


@dataclass(frozen=True)
class ExternCompileSpec:
    """Compilation requirements for one called external source."""

    source: str
    core_types: frozenset[str]
    include_dirs: tuple[Path, ...]


@dataclass
class LoweredTlaIR:
    """Structured result of execution-mode lowering to TLA MLIR (``tla`` dialect)."""

    context: mlir_ir.Context
    module: mlir_ir.Module
    generic: bool = False
    _asm: str | None = None
    extern_compile_specs: tuple[ExternCompileSpec, ...] = ()
    # One entry per non-Constexpr parameter; annotation-only arguments have no tree.
    argument_trees: tuple[RuntimeArgumentTree | None, ...] = dataclass_field(
        default_factory=tuple
    )

    @property
    def entrypoint(self) -> str:
        """Read the frontend symbol before the first cache lookup."""
        entries = [
            mlir_ir.StringAttr(op.attributes["sym_name"]).value
            for op in self.module.body.operations
            if op.operation.name == "tla.func"
        ]
        if len(entries) != 1:
            raise TlaLoweringError(
                "Runtime compilation requires exactly one TLA entrypoint"
            )
        return entries[0]

    def asm(self, *, generic: bool | None = None) -> str:
        emit_generic = self.generic if generic is None else bool(generic)
        if self._asm is None or emit_generic != self.generic:
            with self.context:
                self._asm = self.module.operation.get_asm(
                    print_generic_op_form=emit_generic,
                    assume_verified=False,
                )
            self.generic = emit_generic
        return self._asm


def lower_jit_to_tlair_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> str:
    return lower_jit_to_tlair_module_by_execution(
        fn,
        kind=kind,
        options=options,
        generic=generic,
        type_args=type_args,
        location=location,
    ).asm(generic=generic)


def lower_jit_to_tlair_module_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> LoweredTlaIR:
    auto_sync = (options or {}).get("auto_sync")
    if kind == "kernel" and auto_sync not in (None, "v0"):
        raise TlaLoweringError(
            f"kernel option auto_sync must be 'v0' or None, got {auto_sync!r}"
        )
    if kind != "kernel" and auto_sync is not None:
        raise TlaLoweringError("auto_sync is supported only for tla.kernel")
    raw_core_ratio = (options or {}).get("core_ratio")
    core_ratio = None
    if raw_core_ratio is not None:
        if kind != "kernel":
            raise TlaLoweringError("core_ratio is supported only for tla.kernel")
        from .dsl import parse_core_ratio

        try:
            # Options already store the decorate-time normalized tuple.
            core_ratio = parse_core_ratio(raw_core_ratio, allow_normalized_tuple=True)
        except ValueError as exc:
            raise TlaLoweringError(str(exc)) from exc
    fn = _preprocess_for_lowering(
        fn,
        internal_for=ast_decorators._internal_frontend_for,
        internal_region=runtime._internal_frontend_region,
        internal_if=ast_decorators._internal_frontend_if,
        internal_if_expr=ast_decorators._internal_frontend_if_expr,
        internal_bool_and=ast_decorators._internal_frontend_bool_and,
        internal_bool_or=ast_decorators._internal_frontend_bool_or,
        internal_bool_not=ast_decorators._internal_frontend_bool_not,
        internal_compare=ast_decorators._internal_frontend_compare,
        internal_any=ast_decorators._internal_frontend_any,
        internal_all=ast_decorators._internal_frontend_all,
        internal_bool=ast_decorators._internal_frontend_bool,
        internal_min=ast_decorators._internal_frontend_min,
        internal_max=ast_decorators._internal_frontend_max,
    )
    sig = BaseDSL()._get_signature(fn)
    params = list(sig.parameters.values())
    arg_names = [p.name for p in params]
    # One rule decides what is compile-time: ``is_arg_annotation_constexpr``
    # reserves a leading ``self`` / ``cls`` receiver in addition to
    # ``Constexpr`` annotations. The receiver carries compile-time configuration,
    # so ``self.attr`` reads fold as the body is staged and the instance takes no
    # runtime ABI leaf.
    constexpr_names = {
        p.name
        for index, p in enumerate(params)
        if is_arg_annotation_constexpr(p.annotation, p.name, index, None)
    }
    receiver_name = receiver_argument_name(params)
    kw_only = inspect.Parameter.KEYWORD_ONLY
    keyword_only_names = {p.name for p in params if p.kind is kw_only}
    call_args = _prepare_call_args(arg_names=arg_names, type_args=type_args)
    for name, value in zip(arg_names, call_args, strict=False):
        if name == receiver_name:
            # The receiver is a compile-time configuration object, not a kernel
            # value; its attributes are read while the body is staged.
            if isinstance(value, (Tensor, Numeric)):
                raise TlaLoweringError(
                    f"kernel parameter {name!r} is the compile-time receiver, "
                    "but a runtime value was passed for it; rename the parameter "
                    "when it carries data"
                )
            continue
        if name in constexpr_names:
            # Constexpr values are compile-time inputs, not runtime ABI leaves.
            if (
                callable(value)
                and not isinstance(value, type)
                and not _is_compile_time_callable(value)
            ):
                reject_user_class_value(value, context=f"kernel argument {name!r}")
            continue
        reject_user_class_value(value, context=f"kernel argument {name!r}")

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        _load_execution_dialects(ctx)
        try:
            runtime_argument_trees = build_runtime_argument_trees(
                arg_names, call_args, constexpr_names, ctx
            )
        except ArgumentTreeError as exc:
            raise TlaLoweringError(str(exc)) from exc
        arg_values_for_types = {
            name: value
            for name, value in zip(arg_names, call_args, strict=False)
            if name not in runtime_argument_trees
        }
        arg_types = _resolve_execution_arg_types(
            arg_values=arg_values_for_types or None,
            ctx=ctx,
            constexpr_names=constexpr_names,
        )
        with mlir_ir.Location.unknown(ctx):
            module = mlir_ir.Module.create()
            with mlir_ir.InsertionPoint(module.body):
                fn_loc = _coerce_location(ctx, location)
                extern_compile_specs, argument_trees = _build_tla_func(
                    fn=fn,
                    module=module,
                    fn_name=fn.__name__,
                    arg_names=arg_names,
                    constexpr_names=constexpr_names,
                    keyword_only_names=keyword_only_names,
                    arg_types=arg_types,
                    call_args=call_args,
                    ctx=ctx,
                    fn_loc=fn_loc,
                    auto_sync=auto_sync,
                    runtime_argument_trees=runtime_argument_trees,
                    core_ratio=core_ratio,
                )
    lowered = LoweredTlaIR(
        context=ctx,
        module=module,
        generic=bool(generic),
        extern_compile_specs=extern_compile_specs,
        argument_trees=argument_trees,
    )
    lowered._asm = module.operation.get_asm(
        print_generic_op_form=bool(generic),
        assume_verified=False,
    )
    return lowered


def _transform_jit_helper(helper: Any) -> Any:
    """Transform one genuine helper with the same frontend hooks as its root."""

    return _preprocess_for_lowering(
        unwrap_jit_callable(helper),
        internal_for=ast_decorators._internal_frontend_for,
        internal_region=runtime._internal_frontend_region,
        internal_if=ast_decorators._internal_frontend_if,
        internal_if_expr=ast_decorators._internal_frontend_if_expr,
        internal_bool_and=ast_decorators._internal_frontend_bool_and,
        internal_bool_or=ast_decorators._internal_frontend_bool_or,
        internal_bool_not=ast_decorators._internal_frontend_bool_not,
        internal_compare=ast_decorators._internal_frontend_compare,
        internal_any=ast_decorators._internal_frontend_any,
        internal_all=ast_decorators._internal_frontend_all,
        internal_bool=ast_decorators._internal_frontend_bool,
        internal_min=ast_decorators._internal_frontend_min,
        internal_max=ast_decorators._internal_frontend_max,
    )


def _prepare_call_args(
    *, arg_names: Sequence[str], type_args: Sequence[Any] | None
) -> tuple[Any, ...]:
    if type_args is None:
        if arg_names:
            raise UnsupportedExecutionLowering(
                "Execution-mode lowering requires type_args for non-empty signatures."
            )
        return ()
    if len(type_args) != len(arg_names):
        raise TlaLoweringError(
            "type_args length must match function arguments: "
            f"expected {len(arg_names)}, got {len(type_args)}"
        )
    return tuple(type_args)


@dataclass
class _BoundTreeLeaf:
    template: Any
    binding: RuntimeLeafBinding
    slots: tuple[int, ...]


@dataclass
class _RuntimePhysicalArgumentLayout:
    """Physical MLIR slots derived from one runtime argument tree pass.

    The tree remains the logical representation.  This object only carries the
    physical information needed by ``_build_tla_func``: block argument types,
    logical-to-physical slot ranges, and the compile-time host templates used to
    rebuild device proxies.
    """

    mlir_arg_types: tuple[Any, ...]
    block_slots: dict[str, tuple[int, ...]]
    tree_records: dict[str, tuple[_BoundTreeLeaf, ...]]


def _dynamic_gm_block_types(
    tensor_ty: Any,
    *,
    bridge_ext: Any,
    ctx: mlir_ir.Context,
    index_ty: Any,
) -> tuple[Any, Any, Any]:
    """Return the three device block argument types for one Dynamic-GM leaf."""

    gm_memref = bridge_ext.dynamic_gm_memref_type(tensor_ty.to_mlir_type(ctx))
    return gm_memref, index_ty, index_ty


def _bind_tree_leaves(
    tree: RuntimeArgumentTree,
    templates: tuple[Any, ...],
    *,
    start_slot: int,
    bridge_ext: Any,
    ctx: mlir_ir.Context,
    index_ty: Any,
) -> tuple[tuple[Any, ...], tuple[_BoundTreeLeaf, ...]]:
    """Allocate physical types and slots together for one argument tree."""

    from .core_api import is_dynamic_gm_tensor_arg

    if len(templates) != tree.runtime_leaf_count:
        raise TlaLoweringError(
            f"runtime tree has {len(templates)} leaves but "
            f"{tree.runtime_leaf_count} bindings"
        )

    physical_types: list[Any] = []
    records: list[_BoundTreeLeaf] = []
    for binding, template in zip(tree.bindings, templates, strict=True):
        if binding.kind == "dynamic_gm":
            if not is_dynamic_gm_tensor_arg(template):
                raise TlaLoweringError(
                    f"runtime tree leaf {binding.path!r} is marked Dynamic-GM "
                    "but its host value is not a Dynamic-GM Tensor"
                )
            tensor_ty = template.tla_tensor_type_descriptor()
            leaf_types = _dynamic_gm_block_types(
                tensor_ty,
                bridge_ext=bridge_ext,
                ctx=ctx,
                index_ty=index_ty,
            )
        else:
            leaf_types = (_coerce_type(ctx, binding.mlir_type),)
        first_slot = start_slot + len(physical_types)
        slots = tuple(range(first_slot, first_slot + len(leaf_types)))
        physical_types.extend(leaf_types)
        records.append(_BoundTreeLeaf(template, binding, slots))
    return tuple(physical_types), tuple(records)


def _build_runtime_physical_argument_layout(
    *,
    runtime_arg_names: Sequence[str],
    arg_names: Sequence[str],
    arg_types: Mapping[str, Any],
    call_args: Sequence[Any],
    runtime_argument_trees: Mapping[str, RuntimeArgumentTree],
    ctx: mlir_ir.Context,
    bridge_ext: Any,
    index_ty: Any,
) -> _RuntimePhysicalArgumentLayout:
    """Build the flat MLIR signature while preserving logical tree bindings."""

    call_args_by_name = dict(zip(arg_names, call_args, strict=True))

    mlir_arg_types: list[Any] = []
    block_slots: dict[str, tuple[int, ...]] = {}
    tree_records: dict[str, tuple[_BoundTreeLeaf, ...]] = {}
    for name in runtime_arg_names:
        tree = runtime_argument_trees.get(name)
        if tree is not None:
            templates = runtime_leaf_templates(call_args_by_name[name], tree.treedef)
            start = len(mlir_arg_types)
            physical_types, records = _bind_tree_leaves(
                tree,
                templates,
                start_slot=start,
                bridge_ext=bridge_ext,
                ctx=ctx,
                index_ty=index_ty,
            )
            mlir_arg_types.extend(physical_types)
            tree_records[name] = records
            continue

        spec = arg_types.get(name)
        if spec is None:
            host = call_args_by_name[name]
            raise TlaLoweringError(
                f"kernel argument {name!r} has no runtime type: a "
                f"{type(host).__name__} value cannot be a kernel argument. "
                "Annotate it ``tla.Constexpr[...]`` to pass it as a "
                "compile-time constant, or pass a tensor / numeric value."
            )
        start = len(mlir_arg_types)
        mlir_arg_types.append(_coerce_type(ctx, spec))
        block_slots[name] = (start,)

    return _RuntimePhysicalArgumentLayout(
        mlir_arg_types=tuple(mlir_arg_types),
        block_slots=block_slots,
        tree_records=tree_records,
    )


def _materialize_tree_dynamic_gm_descriptors(
    *,
    pending_tree_rebuilds: Mapping[int, RuntimeArgumentTree],
    arg_names: Sequence[str],
    tree_records: Mapping[str, tuple[_BoundTreeLeaf, ...]],
    entry: Any,
    fn_loc: Any,
) -> dict[tuple[int, int], tuple[Any, Any, dict[str, Any]]]:
    """Materialize Dynamic-GM leaves before frontend emission."""

    from .core_api import (
        _materialize_dynamic_gm_root_tensor_descriptor,
    )

    descriptors: dict[tuple[int, int], tuple[Any, Any, dict[str, Any]]] = {}
    for index in pending_tree_rebuilds:
        name = arg_names[index]
        records = tree_records[name]
        for binding_index, record in enumerate(records):
            template, binding, slots = record.template, record.binding, record.slots
            if binding.kind == "dynamic_gm":
                tensor_ty = template.tla_tensor_type_descriptor()
                desc, metadata = _materialize_dynamic_gm_root_tensor_descriptor(
                    entry.arguments[slots[0]],
                    entry.arguments[slots[1]],
                    entry.arguments[slots[2]],
                    tensor_ty,
                    loc=fn_loc,
                )
                descriptors[(index, binding_index)] = (desc, tensor_ty, metadata)
    return descriptors


def _rebuild_tree_argument_proxies(
    *,
    pending_tree_rebuilds: Mapping[int, RuntimeArgumentTree],
    arg_names: Sequence[str],
    tree_records: Mapping[str, tuple[_BoundTreeLeaf, ...]],
    tree_dynamic_descriptors: Mapping[tuple[int, int], tuple[Any, Any, dict[str, Any]]],
    entry: Any,
    ctx: mlir_ir.Context,
    frontend_state: Any,
    call_args_for_fn: list[Any],
) -> None:
    """Rebuild logical argument trees from device-side proxy leaves."""

    from .core_api import _wrap_frontend_value

    for index, tree in pending_tree_rebuilds.items():
        leaves: list[Any] = []
        name = arg_names[index]
        records = tree_records[name]
        for binding_index, record in enumerate(records):
            template, binding, slots = record.template, record.binding, record.slots
            if binding.kind == "dynamic_gm":
                desc, _, _ = tree_dynamic_descriptors[(index, binding_index)]
                proxy_value = _RuntimeArgumentProxy()
                frontend_state.arg_bindings[id(proxy_value)] = (proxy_value, desc)
                frontend_state.category_bindings[id(proxy_value)] = (
                    proxy_value,
                    "tensor",
                )
                frontend_state.category_bindings[id(desc)] = (desc, "tensor")
                frontend_state.tensor_host_by_value[desc] = template
            elif isinstance(template, Numeric):
                ssa = entry.arguments[slots[0]]
                proxy_value = type(template)(ssa)
                frontend_state.category_bindings[id(proxy_value)] = (
                    proxy_value,
                    "numeric",
                )
                frontend_state.category_bindings[id(ssa)] = (ssa, "numeric")
            elif isinstance(template, Tensor):
                ssa = entry.arguments[slots[0]]
                frontend_state.tensor_host_by_value[ssa] = template
                proxy_value = _wrap_frontend_value(ssa)
            else:
                ssa = entry.arguments[slots[0]]
                proxy_value = _wrap_frontend_value(ssa)
                category = _category_from_type_like(ctx, binding.mlir_type)
                if category is not None:
                    frontend_state.category_bindings[id(proxy_value)] = (
                        proxy_value,
                        category,
                    )
                    frontend_state.category_bindings[id(ssa)] = (ssa, category)
            leaves.append(proxy_value)
        call_args_for_fn[index] = tree_unflatten(leaves, tree.treedef)


def _build_tla_func(
    *,
    fn: Any,
    module: mlir_ir.Module,
    fn_name: str,
    arg_names: Sequence[str],
    constexpr_names: set[str],
    keyword_only_names: set[str],
    arg_types: Mapping[str, Any],
    call_args: Sequence[Any],
    ctx: mlir_ir.Context,
    fn_loc: mlir_ir.Location,
    auto_sync: str | None,
    runtime_argument_trees: Mapping[str, RuntimeArgumentTree],
    core_ratio: tuple[int, int] | None = None,
) -> tuple[
    tuple[ExternCompileSpec, ...],
    tuple[RuntimeArgumentTree | None, ...],
]:
    runtime_arg_names = [name for name in arg_names if name not in constexpr_names]

    # Dynamic GM host tensors enter as unified GM memref + originShape0/1 index args.
    bridge_ext = _tla_type_bridge._load_bridge_extension()
    index_ty = mlir_ir.IndexType.get(ctx)
    physical_layout = _build_runtime_physical_argument_layout(
        runtime_arg_names=runtime_arg_names,
        arg_names=arg_names,
        arg_types=arg_types,
        call_args=call_args,
        runtime_argument_trees=runtime_argument_trees,
        ctx=ctx,
        bridge_ext=bridge_ext,
        index_ty=index_ty,
    )
    mlir_arg_types = list(physical_layout.mlir_arg_types)
    block_slots = physical_layout.block_slots
    tree_records = physical_layout.tree_records

    fn_type = mlir_ir.FunctionType.get(mlir_arg_types, [])
    func_attrs = {
        "sym_name": mlir_ir.StringAttr.get(fn_name),
        "function_type": mlir_ir.TypeAttr.get(fn_type),
    }
    dynamic_gm_slots = [
        record.slots[0]
        for records in tree_records.values()
        for record in records
        if record.binding.kind == "dynamic_gm"
    ]
    if dynamic_gm_slots:
        arg_attrs = [{} for _ in mlir_arg_types]
        for slot in dynamic_gm_slots:
            arg_attrs[slot]["tla.dynamic_gm"] = mlir_ir.UnitAttr.get()
        func_attrs["arg_attrs"] = mlir_ir.ArrayAttr.get(
            [mlir_ir.DictAttr.get(attrs) for attrs in arg_attrs]
        )
    if auto_sync == "v0":
        func_attrs["tla.auto_sync"] = mlir_ir.StringAttr.get("v0")
    if core_ratio is not None:
        func_attrs["tla.core_ratio"] = mlir_ir.DenseI64ArrayAttr.get(list(core_ratio))
    func_op = mlir_ir.Operation.create(
        "tla.func",
        attributes=func_attrs,
        regions=1,
        loc=fn_loc,
    )
    entry = func_op.regions[0].blocks.append(*mlir_arg_types)

    call_args_for_fn = list(call_args)
    arg_bindings: dict[int, tuple[Any, Any]] = {}
    category_bindings: dict[int, tuple[Any, Any]] = {}
    pending_tree_rebuilds: dict[int, RuntimeArgumentTree] = {}
    for i, name in enumerate(arg_names):
        if name in constexpr_names:
            continue
        tree = runtime_argument_trees.get(name)
        if tree is not None:
            # Rebuild recursively once the frontend emission state is active.
            # The tree is also used by ExecutionArgs for launch flattening.
            pending_tree_rebuilds[i] = tree
            continue
        slots = block_slots[name]
        ssa = entry.arguments[slots[0]]
        proxy = _RuntimeArgumentProxy()
        call_args_for_fn[i] = proxy
        arg_bindings[id(proxy)] = (proxy, ssa)
        category = _category_from_type_like(ctx, arg_types.get(name))
        if category is not None:
            category_bindings[id(proxy)] = (proxy, category)
            category_bindings[id(ssa)] = (ssa, category)
    call_args_for_fn = tuple(call_args_for_fn)

    tensor_host_by_value: dict[Any, Any] = {}

    with mlir_ir.InsertionPoint(entry):
        # Descriptor metadata must exist before the user body reads shape/stride.
        tree_dynamic_descriptors = _materialize_tree_dynamic_gm_descriptors(
            pending_tree_rebuilds=pending_tree_rebuilds,
            arg_names=arg_names,
            tree_records=tree_records,
            entry=entry,
            fn_loc=fn_loc,
        )

        with runtime._frontend_emission(
            arg_bindings=arg_bindings,
            category_bindings=category_bindings,
            tensor_host_by_value=tensor_host_by_value,
            module=module,
            core_ratio=core_ratio,
        ) as frontend_state:
            from .core_api import (
                _register_tla_tensor_metadata,
                _register_tla_tensor_type,
            )

            # Descriptor emission ran before emission state existed; register now.
            for desc, tensor_ty, metadata in tree_dynamic_descriptors.values():
                _register_tla_tensor_type(desc, tensor_ty)
                _register_tla_tensor_metadata(desc, metadata)
            if pending_tree_rebuilds:
                call_args_for_fn = list(call_args_for_fn)
                _rebuild_tree_argument_proxies(
                    pending_tree_rebuilds=pending_tree_rebuilds,
                    arg_names=arg_names,
                    tree_records=tree_records,
                    tree_dynamic_descriptors=tree_dynamic_descriptors,
                    entry=entry,
                    ctx=ctx,
                    frontend_state=frontend_state,
                    call_args_for_fn=call_args_for_fn,
                )
                call_args_for_fn = tuple(call_args_for_fn)
            helper_cache: dict[int, tuple[Any, Any]] = {}
            active_jit_helpers: list[Any] = []

            def transform_helper(helper: Any) -> Any:
                key = id(helper)
                cached = helper_cache.get(key)
                if cached is not None and cached[0] is helper:
                    return cached[1]

                # Helpers discovered only while staging a factory or Python
                # forwarding call have not passed the root-function boundary
                # walk. Validate them before their first transformation.
                transformed = _transform_jit_helper(helper)

                def guarded_helper(*args: Any, **kwargs: Any) -> Any:
                    if any(active is helper for active in active_jit_helpers):
                        raise _RecursiveJitHelperError(
                            "recursive @tla.jit helper calls are not supported"
                        )
                    active_jit_helpers.append(helper)
                    try:
                        return transformed(*args, **kwargs)
                    finally:
                        active_jit_helpers.pop()

                # Retain the wrapper alongside the guarded callable: CPython
                # may recycle object IDs for temporary factory results.
                helper_cache[key] = (helper, guarded_helper)
                return guarded_helper

            try:
                with _jit_helper_inline(transform_helper):
                    if keyword_only_names:
                        # Keyword-only params (``def k(*, sel: Constexpr[str])``)
                        # are ordinary entries in ``arg_names``; passing them
                        # positionally would raise before the body ever runs.
                        _pos = [
                            v
                            for n, v in zip(arg_names, call_args_for_fn, strict=False)
                            if n not in keyword_only_names
                        ]
                        _kw = {
                            n: v
                            for n, v in zip(arg_names, call_args_for_fn, strict=False)
                            if n in keyword_only_names
                        }
                        fn(*_pos, **_kw)
                    else:
                        fn(*call_args_for_fn)
            except runtime.TlaCoreAPIError as exc:
                if exc.location is not None or exc.cause_trace:
                    raise
                raise _wrap_execution_exception(
                    runtime.TlaCoreAPIError, fn, exc
                ) from None
            except TlaLoweringError as exc:
                if exc.location is not None or exc.cause_trace:
                    raise
                raise _wrap_execution_exception(TlaLoweringError, fn, exc) from None
            # ValueError raised from the user body benefits from the same
            # source-framed diagnostic as other execution-mode errors.
            except ValueError as exc:
                if traceback_location_for_code(exc, fn.__code__) is None:
                    raise
                raise _wrap_execution_exception(
                    UnsupportedExecutionLowering, fn, exc
                ) from None
            except SyntaxError as exc:
                if isinstance(exc, _RecursiveJitHelperError):
                    raise
                raise _wrap_execution_exception(
                    UnsupportedExecutionLowering, fn, exc
                ) from None
            except ast_decorators.FrontendControlFlowLoweringError as exc:
                raise UnsupportedExecutionLowering(
                    exc.args[0],
                    location=exc.location,
                    reason=exc.reason,
                    cause_trace=exc.cause_trace,
                ) from None
            except Exception as exc:
                raise _wrap_execution_exception(
                    UnsupportedExecutionLowering, fn, exc
                ) from None
            # Group called externs by source in first-use order. Declarations that
            # share a source must also share one ordered include configuration;
            # merge their call-site core types into a single compile spec. The
            # lowering pass independently keeps each original callee symbol and
            # derives its core type, including AIC_OR_AIV for a symbol used by both.
            # Example (source -> first symbol, include dirs, merged core types):
            # {
            #     'extern "C" { ... }': (
            #         "tla_user_shared_a",
            #         (Path("/project/include"),),
            #         {"aic", "aiv"},
            #     )
            # }
            source_compile_configs: dict[
                str, tuple[str, tuple[Path, ...], set[str]]
            ] = {}
            for usage in frontend_state.extern_usages.values():
                source = usage.function.source
                symbol = usage.function.symbol
                include_dirs = usage.function.include_dirs
                config = source_compile_configs.get(source)
                if config is None:
                    core_types: set[str] = set()
                    source_compile_configs[source] = (
                        symbol,
                        include_dirs,
                        core_types,
                    )
                else:
                    configured_symbol, configured_include_dirs, core_types = config
                    if configured_include_dirs != include_dirs:
                        raise runtime.TlaCoreAPIError(
                            "tla.extern: source compile configuration conflict: "
                            f"symbol {configured_symbol!r} uses include_dirs="
                            f"{[str(path) for path in configured_include_dirs]!r}, "
                            f"but symbol {symbol!r} uses include_dirs="
                            f"{[str(path) for path in include_dirs]!r} for the same source"
                        )
                for _, core_type in usage.calls:
                    core_types.add(core_type)

            extern_compile_specs = tuple(
                ExternCompileSpec(
                    source=source,
                    core_types=frozenset(core_types),
                    include_dirs=include_dirs,
                )
                for source, (
                    _,
                    include_dirs,
                    core_types,
                ) in source_compile_configs.items()
            )
            if core_ratio is not None:
                has_cube = False
                has_vector = False

                def _walk_regions(op: mlir_ir.Operation) -> None:
                    nonlocal has_cube, has_vector
                    name = op.name
                    if name == "tla.cube":
                        has_cube = True
                    elif name == "tla.vector":
                        has_vector = True
                    for region in op.regions:
                        for block in region.blocks:
                            for child in block.operations:
                                _walk_regions(child)

                _walk_regions(func_op)
                if not (has_cube and has_vector):
                    raise TlaLoweringError(
                        "core_ratio is only valid on MIX kernels that contain "
                        "both tla.cube and tla.vector regions"
                    )
        mlir_ir.Operation.create("tla.return", loc=fn_loc)
    # Return external compile requirements and the same tree plans used by
    # device reconstruction and host launch.
    return (
        extern_compile_specs,
        tuple(runtime_argument_trees.get(name) for name in runtime_arg_names),
    )


def _coerce_location(
    ctx: mlir_ir.Context, location: DSLLocation | None
) -> mlir_ir.Location:
    if location is None:
        return mlir_ir.Location.unknown(ctx)
    if location.lineno <= 0:
        return mlir_ir.Location.unknown(ctx)
    file_loc = mlir_ir.Location.file(
        location.filename,
        int(location.lineno),
        int(location.col_offset),
        ctx,
    )
    return mlir_ir.Location.name(location.function_name, childLoc=file_loc, context=ctx)


def _coerce_type(ctx: mlir_ir.Context, type_like: Any) -> mlir_ir.Type:
    if isinstance(type_like, mlir_ir.Type):
        return type_like
    if type_like is None:
        raise TypeError(
            "execution lowering could not resolve a concrete runtime argument type"
        )
    if isinstance(type_like, str):
        with ctx:
            return Numeric.from_dtype_token(type_like).mlir_type(ctx)
    raise TypeError(
        "execution lowering expected mlir.ir.Type or a Tla element token; "
        f"got {type(type_like).__name__}"
    )


def _category_from_type_like(ctx: mlir_ir.Context, type_like: Any) -> str | None:
    from . import _tla_type_bridge

    if type_like is None:
        return None
    try:
        ty = _coerce_type(ctx, type_like)
    except Exception:
        return None
    if isinstance(ty, mlir_ir.IndexType):
        return "index"
    return _tla_type_bridge.tla_type_category(ty)


def _resolve_execution_arg_types(
    *,
    arg_values: Mapping[str, Any] | None,
    ctx: mlir_ir.Context,
    constexpr_names: set[str] | None = None,
) -> Mapping[str, Any]:
    resolved: dict[str, Any] = {}
    skip = constexpr_names or set()
    if arg_values is not None:
        for name, value in arg_values.items():
            # Constexpr params are host values with no MLIR type. Probing them
            # is not just wasted work: a dtype constexpr (``tla.Float32``) is a
            # class whose unbound ``__get_mlir_types__`` is callable, so the
            # probe below would call it with the context as ``self``.
            if name in skip:
                continue
            mlir_types_getter = getattr(value, "__get_mlir_types__", None)
            if callable(mlir_types_getter):
                resolved_types = mlir_types_getter(ctx)
                if resolved_types:
                    if len(resolved_types) != 1:
                        raise TlaLoweringError(
                            f"kernel argument {name!r} exposes "
                            f"{len(resolved_types)} runtime values; use a supported "
                            "ordered container to expose multiple values"
                        )
                    resolved[name] = resolved_types[0]
                    continue
            if is_jit_callable(value):
                # Compile-time helpers do not consume runtime ABI slots.
                continue
    return resolved


def _load_execution_dialects(ctx: mlir_ir.Context) -> None:
    _tla_type_bridge.load_tla_dialect(ctx)
    for dialect in ("arith", "scf", "memref"):
        ctx.dialects[dialect]


__all__ = [
    "TlaLoweringError",
    "LoweredTlaIR",
    "UnsupportedExecutionLowering",
    "lower_jit_to_tlair_by_execution",
    "lower_jit_to_tlair_module_by_execution",
]
