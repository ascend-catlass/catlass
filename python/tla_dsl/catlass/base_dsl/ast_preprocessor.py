"""Minimal AST rewrites used to lower structured frontend control flow."""

from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import inspect
import sys
import symtable
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Callable, Iterable, Iterator, NoReturn


_INTERNAL_FOR = "__tladsl_internal_for__"
_INTERNAL_REGION = "__tladsl_internal_region__"
_INTERNAL_IF = "__tladsl_internal_if__"
_INTERNAL_IF_EXPR = "__tladsl_internal_if_expr__"
_INTERNAL_BOOL_AND = "__tladsl_internal_bool_and__"
_INTERNAL_BOOL_OR = "__tladsl_internal_bool_or__"
_INTERNAL_BOOL_NOT = "__tladsl_internal_bool_not__"
_INTERNAL_COMPARE = "__tladsl_internal_compare__"
_INTERNAL_ANY = "__tladsl_internal_any__"
_INTERNAL_ALL = "__tladsl_internal_all__"
_INTERNAL_BOOL = "__tladsl_internal_bool__"
_INTERNAL_MIN = "__tladsl_internal_min__"
_INTERNAL_MAX = "__tladsl_internal_max__"
_INTERNAL_CF_SYMBOL_CHECK = "__tladsl_internal_cf_symbol_check__"
_INTERNAL_CHECKED_DSL_MEMBER = "__tladsl_internal_checked_dsl_member__"
_INTERNAL_CHECKED_DSL_IDENTITY = "__tladsl_internal_checked_dsl_identity__"
_INTERNAL_INDEX_ADD = "__tladsl_internal_index_add__"
_INTERNAL_INDEX_SUB = "__tladsl_internal_index_sub__"
_INTERNAL_ATTACH_SOURCE_INFO = "__tladsl_internal_attach_source_info__"
_INTERNAL_UNKNOWN_EFFECT_CALL = "__tladsl_internal_unknown_effect_call__"
_INTERNAL_RUNTIME_BINDING_IS_COMPILETIME = (
    "__tladsl_internal_runtime_binding_is_compiletime__"
)
_INTERNAL_RUNTIME_WRITE_CHECK = "__tladsl_internal_runtime_write_check__"
_INTERNAL_STATIC_LOOP_ITERATION = "__tladsl_internal_static_loop_iteration__"
_INTERNAL_RANGE_CONSTEXPR = "__tladsl_internal_range_constexpr__"
_INTERNAL_LAZY_ATTRIBUTE = "__tladsl_internal_lazy_attribute__"
_INTERNAL_LAZY_SUBSCRIPT = "__tladsl_internal_lazy_subscript__"
_INTERNAL_LAZY_BINOP = "__tladsl_internal_lazy_binop__"
_INTERNAL_LAZY_UNARY = "__tladsl_internal_lazy_unary__"
_SOURCE_INFO_ATTR = "__tladsl_source_info__"
_WHILE_SELECTOR = "while_selector"
_WHILE_EXECUTOR = "while_executor"


_UNKNOWN_LANGUAGE_VALUE = object()


class _RecursiveJitHelperError(SyntaxError):
    """Internal marker for recursion detected while staging a JIT helper."""


@dataclass(frozen=True)
class _SourceSnapshot:
    """Keep diagnostic source alive without extracting snippets while compiling."""

    source_text: str

    def snippet(self, span: tuple[int | None, ...]) -> str:
        lineno, col_offset, end_lineno, end_col_offset = span
        location = ast.Constant(
            value=None,
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
        )
        # Use CPython's byte-offset and newline rules, including form feeds.
        # This runs only when reporting an error, never during normal tracing.
        segment = ast.get_source_segment(self.source_text, location) or ""
        return next((line.strip() for line in segment.splitlines() if line.strip()), "")


def _source_text_for_info(info: dict[str, Any]) -> str:
    if "source" not in info:
        reference = info.get("_source_ref")
        if reference is not None:
            snapshot, span = reference
            info["source"] = snapshot.snippet(span)
    return str(info.get("source") or "")


def _is_exact_builtin_dict(value: object) -> bool:
    """Return whether *value* is a dict without subclass-defined behavior."""

    return isinstance(value, dict) and (
        object.__getattribute__(value, "__class__") is dict
    )


def _is_tla_owned_type(value_type: type[Any]) -> bool:
    module = str(getattr(value_type, "__module__", ""))
    if module == "builtins":
        return getattr(builtins, value_type.__name__, None) is value_type
    if module == "enum":
        return getattr(sys.modules.get("enum"), value_type.__name__, None) is value_type
    if not module.startswith("catlass"):
        return False
    return getattr(sys.modules.get(module), value_type.__name__, None) is value_type


def reject_user_class_value(value: Any, *, context: str) -> None:
    """Reject arbitrary Python class instances before frontend evaluation."""

    seen: set[int] = set()

    def check(item: Any) -> None:
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            return
        if (
            inspect.isfunction(item)
            or inspect.ismethod(item)
            or isinstance(item, functools.partial)
        ):
            # Plain callables may be ``Constexpr`` kernel args; if not annotated,
            # lowering fails later with a Constexpr hint.
            return
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, dict):
            for key, nested in item.items():
                check(key)
                check(nested)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            for nested in item:
                check(nested)
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                check(getattr(item, field.name))
            return
        item_type = item if inspect.isclass(item) else type(item)
        if _is_tla_owned_type(item_type):
            return
        raise TypeError(
            f"user-defined class {item_type.__name__!r} is not supported in "
            f"TLA Python functions ({context})"
        )

    check(value)


_BUILTIN_REDIRECTS = {
    "any": _INTERNAL_ANY,
    "all": _INTERNAL_ALL,
    "bool": _INTERNAL_BOOL,
    "min": _INTERNAL_MIN,
    "max": _INTERNAL_MAX,
}
_TRUSTED_LAZY_CALLABLES: tuple[Callable[..., Any], ...] = ()
_TRUSTED_DSL_MODULE: ModuleType | None = None
_TRUSTED_RANGE_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_RANGE_CONSTEXPR_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_CONST_EXPR_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_CUBE_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_VECTOR_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_VEC_NAMESPACE: Any | None = None
_TRUSTED_VEC_FUNC_CALLABLE: Callable[..., Any] | None = None
_TRUSTED_AS_NUMERIC_CALLABLE: Callable[..., Any] | None = None


@dataclass(frozen=True)
class _TrustedDslIdentities:
    range_callable: Callable[..., Any] | None
    range_constexpr_callable: Callable[..., Any] | None
    const_expr_callable: Callable[..., Any] | None
    module: ModuleType | None
    cube_callable: Callable[..., Any] | None
    vector_callable: Callable[..., Any] | None
    vec_namespace: Any | None
    vec_func_callable: Callable[..., Any] | None
    as_numeric_callable: Callable[..., Any] | None


@dataclass(frozen=True)
class _TrustedDslSymbols:
    range_names: frozenset[str]
    builtin_range_names: frozenset[str]
    range_constexpr_names: frozenset[str]
    const_expr_names: frozenset[str]
    as_numeric_names: frozenset[str]
    module_aliases: frozenset[str]


def _trusted_dsl_identities() -> _TrustedDslIdentities:
    return _TrustedDslIdentities(
        range_callable=_TRUSTED_RANGE_CALLABLE,
        range_constexpr_callable=_TRUSTED_RANGE_CONSTEXPR_CALLABLE,
        const_expr_callable=_TRUSTED_CONST_EXPR_CALLABLE,
        module=_TRUSTED_DSL_MODULE,
        cube_callable=_TRUSTED_CUBE_CALLABLE,
        vector_callable=_TRUSTED_VECTOR_CALLABLE,
        vec_namespace=_TRUSTED_VEC_NAMESPACE,
        vec_func_callable=_TRUSTED_VEC_FUNC_CALLABLE,
        as_numeric_callable=_TRUSTED_AS_NUMERIC_CALLABLE,
    )


def _trusted_dsl_symbols(
    global_symbols: dict[str, Any], identities: _TrustedDslIdentities
) -> _TrustedDslSymbols:
    return _TrustedDslSymbols(
        range_names=frozenset(
            _tla_function_names_from_globals(global_symbols, "range", identities)
        ),
        builtin_range_names=frozenset(
            name
            for name in ("range",)
            if global_symbols.get(name, builtins.range) is builtins.range
        ),
        range_constexpr_names=frozenset(
            _tla_function_names_from_globals(
                global_symbols, "range_constexpr", identities
            )
        ),
        const_expr_names=frozenset(
            _tla_const_expr_names_from_globals(global_symbols, identities)
        ),
        as_numeric_names=frozenset(
            _tla_as_numeric_names_from_globals(global_symbols, identities)
        ),
        module_aliases=frozenset(
            _tla_module_aliases_from_globals(global_symbols, identities)
        ),
    )


def _register_trusted_lazy_callables(
    callables: tuple[Callable[..., Any], ...],
    module: ModuleType,
    *,
    range_callable: Callable[..., Any],
    range_constexpr_callable: Callable[..., Any],
    const_expr_callable: Callable[..., Any],
    cube_callable: Callable[..., Any],
    vector_callable: Callable[..., Any],
    vec_namespace: Any,
    vec_func_callable: Callable[..., Any],
    as_numeric_callable: Callable[..., Any],
) -> None:
    """Freeze genuine DSL call identities once core_api has initialized."""

    global _TRUSTED_CONST_EXPR_CALLABLE
    global _TRUSTED_DSL_MODULE, _TRUSTED_LAZY_CALLABLES
    global _TRUSTED_RANGE_CALLABLE, _TRUSTED_RANGE_CONSTEXPR_CALLABLE
    global _TRUSTED_CUBE_CALLABLE, _TRUSTED_VECTOR_CALLABLE
    global _TRUSTED_VEC_NAMESPACE, _TRUSTED_VEC_FUNC_CALLABLE
    global _TRUSTED_AS_NUMERIC_CALLABLE
    if _TRUSTED_LAZY_CALLABLES or _TRUSTED_DSL_MODULE is not None:
        raise RuntimeError("trusted lazy callables are already registered")
    _TRUSTED_LAZY_CALLABLES = callables
    _TRUSTED_DSL_MODULE = module
    _TRUSTED_RANGE_CALLABLE = range_callable
    _TRUSTED_RANGE_CONSTEXPR_CALLABLE = range_constexpr_callable
    _TRUSTED_CONST_EXPR_CALLABLE = const_expr_callable
    _TRUSTED_CUBE_CALLABLE = cube_callable
    _TRUSTED_VECTOR_CALLABLE = vector_callable
    _TRUSTED_VEC_NAMESPACE = vec_namespace
    _TRUSTED_VEC_FUNC_CALLABLE = vec_func_callable
    _TRUSTED_AS_NUMERIC_CALLABLE = as_numeric_callable


@dataclass(frozen=True)
class Binding:
    """One Python lexical binding owned by an analyzed function."""

    scope_id: str
    name: str
    lineno: int
    col_offset: int
    kind: str


@dataclass(frozen=True)
class FunctionBlockPlan:
    """A source-ordered block nested in a transformed function."""

    construct_name: str
    lineno: int
    col_offset: int


@dataclass(frozen=True)
class FunctionPlan:
    """Immutable lexical plan for one user-authored function."""

    name: str
    scope_id: str
    arguments: tuple[Binding, ...]
    local_bindings: tuple[Binding, ...]
    captures: tuple[Binding, ...]
    child_plans: tuple[FunctionBlockPlan, ...]

    @property
    def bindings(self) -> tuple[Binding, ...]:
        return (*self.arguments, *self.local_bindings, *self.captures)

    def resolve(self, name: str) -> Binding | None:
        return next(
            (binding for binding in self.bindings if binding.name == name), None
        )


def _symtable_type(table: Any) -> str:
    """Normalize SymbolTable.get_type() across Python 3.10 through 3.13."""

    kind = table.get_type()
    value = getattr(kind, "value", kind)
    return str(value).lower()


@dataclass(frozen=True)
class _FunctionScopeFacts:
    """Compiler-owned lexical facts for one source function block."""

    local_names: frozenset[str]
    free_names: frozenset[str]
    local_bindings: tuple[Binding, ...]


def _function_scope_id(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    return (
        f"{node.name}@{int(getattr(node, 'lineno', 0) or 0)}:"
        f"{int(getattr(node, 'col_offset', 0) or 0)}"
    )


_FUNCTION_SCOPE_WRAPPERS = frozenset(
    {
        "annotation",
        "type alias",
        "type parameter",
        "type parameters",
        "type variable",
        "typevar bound",
    }
)


def _root_function_tables(namespace: Any) -> list[Any]:
    kind = _symtable_type(namespace)
    if kind == "function":
        # Before comprehensions were inlined, CPython represented them as
        # function tables with the synthetic, unspellable parameter `.0`.
        return [] if ".0" in namespace.get_parameters() else [namespace]
    if kind not in _FUNCTION_SCOPE_WRAPPERS:
        return []
    return [
        function_table
        for child in namespace.get_children()
        for function_table in _root_function_tables(child)
    ]


def _root_function_scope_facts(
    source: str,
    filename: str,
    target: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _FunctionScopeFacts:
    """Return CPython lexical facts for the one inspected root function."""

    root = symtable.symtable(source, filename, "exec")
    try:
        namespaces = root.lookup(target.name).get_namespaces()
    except KeyError:
        namespaces = ()
    tables = [
        function_table
        for namespace in namespaces
        for function_table in _root_function_tables(namespace)
    ]
    if len(tables) != 1:
        raise RuntimeError(
            "expected exactly one root symbol table for "
            f"{target.name!r}, found {len(tables)}"
        )
    table = tables[0]
    symbols = {name: table.lookup(name) for name in table.get_identifiers()}
    collector = _FunctionLocalCollector(_function_scope_id(target))
    for statement in target.body:
        collector.visit(statement)
    ast_owned_names = collector.names | {
        argument.arg for argument in _ordered_function_args(target.args)
    }
    return _FunctionScopeFacts(
        local_names=frozenset(
            name
            for name, symbol in symbols.items()
            if (symbol.is_local() or symbol.is_parameter()) and name in ast_owned_names
        ),
        free_names=frozenset(
            name for name, symbol in symbols.items() if symbol.is_free()
        ),
        local_bindings=tuple(collector.bindings),
    )


def _scope_facts_for_transform(
    source: str,
    filename: str,
    target: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    has_scope_declaration: bool | None = None,
) -> _FunctionScopeFacts | None:
    """Build root compiler facts unless a source diagnostic must run first."""

    if has_scope_declaration is None:
        has_scope_declaration = _has_scope_declaration(target)
    if has_scope_declaration:
        return None
    return _root_function_scope_facts(source, filename, target)


def _has_scope_declaration(
    target: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        isinstance(item, (ast.Global, ast.Nonlocal)) for item in ast.walk(target)
    )


def _contains_runtime_value(value: Any, seen: set[int] | None = None) -> bool:
    """Classify a prepared lowering value without speculating about calls."""

    from catlass._mlir import ir as mlir_ir

    from .typing import Numeric

    from ..core_api import _resolve_bound_value

    seen = seen if seen is not None else set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    resolved = _resolve_bound_value(value)
    if resolved is not value:
        return _contains_runtime_value(resolved, seen)
    if isinstance(value, mlir_ir.Value):
        return True
    # An explicitly typed DSL scalar is a runtime-stage value even when its
    # literal is materialized lazily on first use.
    if isinstance(value, Numeric):
        return True
    numeric_value = getattr(value, "value", None)
    if isinstance(numeric_value, mlir_ir.Value):
        return True
    if isinstance(value, dict):
        return any(
            _contains_runtime_value(item, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_runtime_value(item, seen) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_runtime_value(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    attributes = getattr(value, "__dict__", None)
    return isinstance(attributes, dict) and any(
        _contains_runtime_value(item, seen) for item in attributes.values()
    )


def _runtime_binding_is_compiletime(value: Any) -> bool:
    from .. import runtime as runtime_mod

    # Source transformation is also used by eager Python-only unit execution.
    if runtime_mod._current_frontend_state() is None:
        return False
    return not _contains_runtime_value(value)


def _runtime_write_check(
    compiletime_origin: bool,
    explicit_promotion: bool,
    name: str,
    construct_name: str,
    filename: str,
    lineno: int,
    col_offset: int,
    source_line: str | None,
) -> None:
    from .. import runtime as runtime_mod

    if (
        runtime_mod._current_frontend_state() is None
        or not compiletime_origin
        or explicit_promotion
    ):
        return
    error = SyntaxError(
        f"compile-time binding {name!r} cannot be assigned in dynamic Tla "
        f"{construct_name}; use tla.as_numeric(...) to promote it first"
    )
    error.filename = filename
    error.lineno = lineno
    error.offset = col_offset + 1
    error.text = source_line
    raise error


class _FunctionAnalyzer:
    """Build lexical bindings without entering nested function scopes."""

    def __init__(
        self,
        *,
        global_names: set[str] | None = None,
        global_symbols: dict[str, Any] | None = None,
        scope_facts: _FunctionScopeFacts | None = None,
        root_freevars: set[str] | None = None,
        trusted_identities: _TrustedDslIdentities | None = None,
        trusted_symbols: _TrustedDslSymbols | None = None,
        statement_facts: _StatementFactsCache | None = None,
    ) -> None:
        self.global_names = set(global_names or ())
        self.global_symbols = dict(global_symbols or {})
        self.scope_facts = scope_facts
        self.root_freevars = set(root_freevars or ())
        self.statement_facts = statement_facts or _StatementFactsCache()
        self.trusted_identities = trusted_identities or _trusted_dsl_identities()
        self.trusted_symbols = trusted_symbols or _trusted_dsl_symbols(
            self.global_symbols, self.trusted_identities
        )

    def analyze(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionPlan:
        scope_id = _function_scope_id(node)
        arguments = tuple(
            Binding(
                scope_id=scope_id,
                name=argument.arg,
                lineno=int(getattr(argument, "lineno", node.lineno) or 0),
                col_offset=int(getattr(argument, "col_offset", node.col_offset) or 0),
                kind="argument",
            )
            for argument in _ordered_function_args(node.args)
        )
        scope_facts = self.scope_facts
        if scope_facts is None:
            collector = _FunctionLocalCollector(scope_id)
            for statement in node.body:
                collector.visit(statement)
            collected_bindings = collector.bindings
        else:
            collected_bindings = scope_facts.local_bindings
        local_names = scope_facts.local_names if scope_facts is not None else None
        local_bindings = tuple(
            binding
            for binding in collected_bindings
            if binding.name not in {argument.name for argument in arguments}
            and (local_names is None or binding.name in local_names)
        )
        owned = {binding.name: binding for binding in (*arguments, *local_bindings)}
        free_names = set(scope_facts.free_names) if scope_facts is not None else None
        if scope_facts is not None:
            free_names.update(self.root_freevars)
        elif self.root_freevars:
            free_names = set(self.root_freevars)
        captures = self._captures(node, owned, free_names)
        lexical_shadows = {binding.name for binding in (*owned.values(), *captures)}
        return FunctionPlan(
            name=node.name,
            scope_id=scope_id,
            arguments=arguments,
            local_bindings=local_bindings,
            captures=captures,
            child_plans=tuple(
                _function_child_plans(
                    node,
                    lexical_shadow_names=lexical_shadows,
                    trusted_symbols=self.trusted_symbols,
                    global_symbols=self.global_symbols,
                    statement_facts=self.statement_facts,
                )
            ),
        )

    def _captures(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        owned: dict[str, Binding],
        free_names: set[str] | None,
    ) -> tuple[Binding, ...]:
        if free_names is not None and not free_names:
            return ()
        loaded = _ordered_function_loads(node)
        captures: list[Binding] = []
        seen: set[str] = set()
        for name, load_node in loaded:
            if name in seen or name in owned:
                continue
            if free_names is not None and name not in free_names:
                continue
            if free_names is None and name in self.global_names:
                continue
            seen.add(name)
            binding = Binding(
                scope_id=(
                    f"{node.name}@{int(getattr(node, 'lineno', 0) or 0)}:captures"
                ),
                name=name,
                lineno=int(getattr(load_node, "lineno", 0) or 0),
                col_offset=int(getattr(load_node, "col_offset", 0) or 0),
                kind="capture",
            )
            captures.append(binding)
        if free_names is not None:
            descendant_loads: dict[str, ast.Name] = {}
            for candidate in ast.walk(node):
                if (
                    isinstance(candidate, ast.Name)
                    and isinstance(candidate.ctx, ast.Load)
                    and candidate.id in free_names
                ):
                    previous = descendant_loads.get(candidate.id)
                    if previous is None or (
                        int(getattr(candidate, "lineno", 0) or 0),
                        int(getattr(candidate, "col_offset", 0) or 0),
                    ) < (
                        int(getattr(previous, "lineno", 0) or 0),
                        int(getattr(previous, "col_offset", 0) or 0),
                    ):
                        descendant_loads[candidate.id] = candidate
            remaining = [
                name
                for name in free_names
                if name not in seen and name not in owned and name in descendant_loads
            ]
            remaining.sort(
                key=lambda name: (
                    int(getattr(descendant_loads[name], "lineno", 0) or 0),
                    int(getattr(descendant_loads[name], "col_offset", 0) or 0),
                    name,
                )
            )
            for name in remaining:
                load_node = descendant_loads[name]
                binding = Binding(
                    scope_id=(
                        f"{node.name}@{int(getattr(node, 'lineno', 0) or 0)}:captures"
                    ),
                    name=name,
                    lineno=int(getattr(load_node, "lineno", 0) or 0),
                    col_offset=int(getattr(load_node, "col_offset", 0) or 0),
                    kind="capture",
                )
                captures.append(binding)
        return tuple(captures)


class _FunctionLocalCollector(ast.NodeVisitor):
    def __init__(self, scope_id: str) -> None:
        self.scope_id = scope_id
        self.bindings: list[Binding] = []
        self._names: set[str] = set()
        self._target_kind = "local"

    @property
    def names(self) -> set[str]:
        return set(self._names)

    def _bind(self, name: str, node: ast.AST, kind: str | None = None) -> None:
        if name in self._names:
            return
        self._names.add(name)
        if name == "_":
            return
        self.bindings.append(
            Binding(
                scope_id=self.scope_id,
                name=name,
                lineno=int(getattr(node, "lineno", 0) or 0),
                col_offset=int(getattr(node, "col_offset", 0) or 0),
                kind=kind or self._target_kind,
            )
        )

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id, node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".", 1)[0], alias, "import")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._bind(alias.asname or alias.name, alias, "import")

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._bind(node.name, node, "exception variable")
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern is not None:
            self.visit(node.pattern)
        if node.name is not None:
            self._bind(node.name, node, "pattern variable")

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._bind(node.name, node, "pattern variable")

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        for key in node.keys:
            self.visit(key)
        for pattern in node.patterns:
            self.visit(pattern)
        if node.rest is not None:
            self._bind(node.rest, node, "pattern variable")

    def visit_For(self, node: ast.For) -> None:
        previous_kind = self._target_kind
        self._target_kind = "loop variable"
        self.visit(node.target)
        self._target_kind = previous_kind
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name, node, "nested function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name, node, "nested function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name, node, "nested class")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, node.key, node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, node.elt)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        *values: ast.expr,
    ) -> None:
        # Comprehension targets are owned by the comprehension's implicit
        # function scope, not by the containing function.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for value in values:
            self.visit(value)


@dataclass(frozen=True)
class ControlFlowPlan:
    """Immutable analysis consumed by one runtime control-flow rewrite."""

    construct_name: str
    lineno: int
    col_offset: int
    active_bindings: frozenset[Binding]
    active_callables: frozenset[str]
    assigned_by_region_bindings: tuple[frozenset[Binding], ...]
    assigned_bindings: frozenset[Binding]
    invoked_bindings: frozenset[Binding]
    carried_bindings: tuple[Binding, ...]
    captured_bindings: tuple[Binding, ...]
    full_write_args_count: int
    assignment_targets: tuple[str, ...]
    tensor_store_assignments: frozenset[ast.Assign]
    nested_constructs: tuple[tuple[str, int, int], ...]

    @property
    def active_symbols(self) -> frozenset[str]:
        return frozenset(binding.name for binding in self.active_bindings)

    @property
    def assigned_by_region(self) -> tuple[frozenset[str], ...]:
        return tuple(
            frozenset(binding.name for binding in region)
            for region in self.assigned_by_region_bindings
        )

    @property
    def assigned_names(self) -> frozenset[str]:
        return frozenset(binding.name for binding in self.assigned_bindings)

    @property
    def invoked_names(self) -> frozenset[str]:
        return frozenset(binding.name for binding in self.invoked_bindings)

    @property
    def carried_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.carried_bindings)

    @property
    def captured_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.captured_bindings)


class _ControlFlowAnalyzer:
    """Analyze and validate a runtime construct without rewriting its AST."""

    def __init__(self, statement_facts: _StatementFactsCache | None = None) -> None:
        self._statement_facts = statement_facts

    def analyze(
        self,
        *,
        node: ast.If | ast.For | ast.While | ast.With,
        construct_name: str,
        assigned_regions: list[list[ast.stmt]],
        active_call_nodes: list[ast.AST],
        active_symbols: set[str],
        active_callables: set[str],
        function_plan: FunctionPlan | None = None,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
        is_runtime_for: Callable[[ast.For], bool] | None = None,
        is_static_test: Callable[[ast.AST], bool] | None = None,
    ) -> ControlFlowPlan:
        policy = _DynamicControlFlowPolicy(
            construct_name,
            filename=filename,
            line_offset=line_offset,
            source_text=source_text,
            is_runtime_for=is_runtime_for,
            is_static_test=is_static_test,
        )
        for region in assigned_regions:
            for statement in region:
                policy.visit(statement)
        statement_facts = self._statement_facts or _StatementFactsCache()
        assigned_names_by_region = tuple(
            set().union(
                *(statement_facts.get(statement).assigned_names for statement in region)
            )
            for region in assigned_regions
        )
        assigned_names = set().union(*assigned_names_by_region)
        invoked_names = (
            set().union(
                *(
                    statement_facts.get(statement).invoked_names
                    for statement in active_call_nodes
                )
            )
            & active_symbols
        )
        # Local bare helpers are Python closures captured by the outlined
        # region, not runtime SSA state that must be threaded through it.
        invoked_names.difference_update(active_callables)
        if function_plan is None:
            fallback_scope = f"<analysis>:{construct_name}"
            bindings_by_name = {
                name: Binding(fallback_scope, name, 0, 0, "local")
                for name in active_symbols | assigned_names | invoked_names
            }
        else:
            bindings_by_name = {}
            for binding in function_plan.bindings:
                bindings_by_name[binding.name] = binding
        active_bindings = frozenset(
            bindings_by_name[name]
            for name in active_symbols
            if name in bindings_by_name
        )
        assigned_by_region_bindings = tuple(
            frozenset(
                bindings_by_name[name]
                for name in region_names
                if name in bindings_by_name
            )
            for region_names in assigned_names_by_region
        )
        assigned_bindings = frozenset().union(*assigned_by_region_bindings)
        invoked_bindings = frozenset(
            bindings_by_name[name] for name in invoked_names if name in bindings_by_name
        )
        # Carriers follow control-flow semantics, not the runtime type of the
        # value. A name the body *assigns to* is redefined each iteration and
        # has to be carried; a name the body only invokes a method on is a side
        # effect on an object reached from the enclosing scope -- the same
        # classification a subscript store (`out[i] = v`) already gets.
        #
        # Carrying the latter is what forced a UB tile into scf.for's iter_args,
        # where it has lost the allocation it came from and its loads no longer
        # legalize. Deciding it here rather than by inspecting the value keeps
        # one rule in one place, and covers a tile nested in a dataclass or
        # container, which a check on the carried value alone cannot see.
        carried_bindings = tuple(
            sorted(assigned_bindings & active_bindings, key=lambda b: b.name)
        )
        captured_bindings = tuple(
            sorted(invoked_bindings - set(carried_bindings), key=lambda b: b.name)
        )
        return ControlFlowPlan(
            construct_name=construct_name,
            lineno=int(getattr(node, "lineno", 0) or 0),
            col_offset=int(getattr(node, "col_offset", 0) or 0),
            active_bindings=active_bindings,
            active_callables=frozenset(active_callables),
            assigned_by_region_bindings=assigned_by_region_bindings,
            assigned_bindings=assigned_bindings,
            invoked_bindings=invoked_bindings,
            carried_bindings=carried_bindings,
            captured_bindings=captured_bindings,
            full_write_args_count=len(carried_bindings),
            assignment_targets=tuple(policy.assignment_targets),
            tensor_store_assignments=frozenset(policy.tensor_store_assignments),
            nested_constructs=tuple(policy.nested_constructs),
        )


class ScopeManager:
    """Manage frontend AST variable and callable scopes during preprocessing."""

    def __init__(self) -> None:
        self.scopes: list[set[str]] = []
        self.callables: list[set[str]] = []
        self.callable_definitions: list[dict[str, ast.FunctionDef]] = []

    @classmethod
    def create(cls) -> "ScopeManager":
        return cls()

    def add_to_scope(self, name: str) -> None:
        if name == "_":
            return
        if not self.scopes:
            self.scopes.append(set())
        self.scopes[-1].add(name)

    def add_names_to_scope(self, names: Iterable[str]) -> None:
        for name in names:
            self.add_to_scope(name)

    def add_to_callables(self, name: str) -> None:
        if not self.callables or name == "_":
            return
        self.callables[-1].add(name)

    def add_callable_definition(self, node: ast.FunctionDef) -> None:
        if self.callable_definitions:
            self.callable_definitions[-1][node.name] = node

    def add_callable_alias(self, name: str, target: str) -> None:
        definition = self.get_callable_definition(target)
        if definition is not None and self.callable_definitions:
            self.callable_definitions[-1][name] = definition

    def get_callable_definition(self, name: str) -> ast.FunctionDef | None:
        for definitions in reversed(self.callable_definitions):
            definition = definitions.get(name)
            if definition is not None:
                return definition
        return None

    def get_active_symbols(self) -> set[str]:
        active: set[str] = set()
        for scope in self.scopes:
            active.update(scope)
        return active

    def get_active_callables(self) -> set[str]:
        active: set[str] = set()
        for callables in self.callables:
            active.update(callables)
        return active

    @contextlib.contextmanager
    def enter_local_scope(
        self,
        initial_names: set[str] | list[str] | tuple[str, ...] | None = None,
        initial_callables: set[str] | list[str] | tuple[str, ...] | None = None,
        initial_callable_definitions: dict[str, ast.FunctionDef] | None = None,
    ) -> Iterator[None]:
        self.scopes.append(set(initial_names or ()))
        self.callables.append(set(initial_callables or ()))
        self.callable_definitions.append(dict(initial_callable_definitions or {}))
        try:
            yield
        finally:
            self.callable_definitions.pop()
            self.callables.pop()
            self.scopes.pop()


class _DynamicConditionValidator(ast.NodeVisitor):
    """Reject condition syntax whose Python evaluation cannot be preserved."""

    def __init__(
        self,
        construct_name: str,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
    ) -> None:
        self.construct_name = construct_name
        self.filename = filename
        self.line_offset = line_offset
        self.source_lines = source_text.splitlines()

    def validate(self, node: ast.AST) -> None:
        self.visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._raise(node, "does not support assignment expressions in its condition")

    def visit_Await(self, node: ast.Await) -> None:
        self._raise(node, "does not support await expressions in its condition")

    def visit_Yield(self, node: ast.Yield) -> None:
        self._raise(node, "does not support yield expressions in its condition")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._raise(node, "does not support yield expressions in its condition")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._raise(node, "does not support lambda expressions in its condition")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._raise(node, "does not support generator expressions in its condition")

    def _raise(self, node: ast.AST, message: str) -> None:
        error = SyntaxError(f"dynamic Tla {self.construct_name} {message}")
        relative_lineno = int(getattr(node, "lineno", 0) or 0)
        error.filename = self.filename
        error.lineno = self.line_offset + relative_lineno
        error.offset = (
            int(getattr(node, "col_offset", 0)) + 1
            if getattr(node, "col_offset", None) is not None
            else None
        )
        if 0 < relative_lineno <= len(self.source_lines):
            error.text = self.source_lines[relative_lineno - 1]
        raise error


@dataclass(frozen=True)
class _StatementFacts:
    """Scope-independent facts; populated once, before rewriting source nodes."""

    read_names: frozenset[str]
    assigned_names: frozenset[str]
    invoked_names: frozenset[str]


_EMPTY_STATEMENT_FACTS = _StatementFacts(frozenset(), frozenset(), frozenset())


class _StatementFactsCache:
    """Facts about source nodes, retained unchanged during destructive rewriting.

    Populate before visiting a source statement. Generated statements must be
    analyzed separately: a source node can keep its identity after mutation.
    """

    def __init__(self) -> None:
        self._facts: dict[ast.AST, _StatementFacts] = {}

    def get(self, node: ast.AST) -> _StatementFacts:
        facts = self._facts.get(node)
        if facts is None:
            facts = self._collect(node)
            self._facts[node] = facts
        return facts

    def _collect(self, node: ast.AST) -> _StatementFacts:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                return _StatementFacts(frozenset({node.id}), frozenset(), frozenset())
            if isinstance(node.ctx, ast.Store):
                return _StatementFacts(frozenset(), frozenset({node.id}), frozenset())
            return _EMPTY_STATEMENT_FACTS
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return _StatementFacts(frozenset(), frozenset({node.name}), frozenset())
        if isinstance(node, ast.Lambda):
            return _EMPTY_STATEMENT_FACTS

        read_names: set[str] = set()
        assigned_names: set[str] = set()
        invoked_names: set[str] = set()

        def include(
            facts: _StatementFacts, hidden: set[str] | frozenset[str] = frozenset()
        ) -> None:
            read_names.update(facts.read_names - hidden)
            assigned_names.update(facts.assigned_names)
            invoked_names.update(facts.invoked_names)

        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            bound: set[str] = set()
            for generator in node.generators:
                include(self.get(generator.iter), bound)
                target = self.get(generator.target)
                bound.update(target.assigned_names)
                # Assignment/load collectors exclude comprehension targets;
                # invocation analysis historically visits them without filtering.
                invoked_names.update(target.invoked_names)
                for condition in generator.ifs:
                    include(self.get(condition), bound)
            values = (
                (node.key, node.value)
                if isinstance(node, ast.DictComp)
                else (node.elt,)
            )
            for value in values:
                include(self.get(value), bound)
        else:
            for child in ast.iter_child_nodes(node):
                include(self.get(child))
            if isinstance(node, ast.Call):
                base = _call_base_name(node.func)
                if base is not None:
                    invoked_names.add(base)
            elif isinstance(node, ast.Import):
                assigned_names.update(
                    alias.asname or alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assigned_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name != "*"
                )
        if not read_names and not assigned_names and not invoked_names:
            return _EMPTY_STATEMENT_FACTS
        return _StatementFacts(
            frozenset(read_names), frozenset(assigned_names), frozenset(invoked_names)
        )


@dataclass(frozen=True)
class _BlockLastReadIndex:
    """Index each variable's last read within a statement list."""

    last_read: dict[str, int]

    @classmethod
    def build(
        cls,
        body: list[ast.stmt],
        *,
        statement_facts: _StatementFactsCache | None = None,
    ) -> "_BlockLastReadIndex":
        statement_facts = statement_facts or _StatementFactsCache()
        return cls(
            {
                name: index
                for index, statement in enumerate(body)
                for name in statement_facts.get(statement).read_names
            }
        )

    def used_after(self, index: int, candidates: Iterable[str]) -> set[str]:
        return {name for name in candidates if self.last_read.get(name, -1) > index}


class _FrontendControlFlowTransformer(ast.NodeTransformer):
    def __init__(
        self,
        global_symbols: dict[str, Any] | None = None,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
        trusted_identities: _TrustedDslIdentities | None = None,
        root_plan: FunctionPlan | None = None,
        statement_facts: _StatementFactsCache | None = None,
    ) -> None:
        self._counter = 0
        self._reserved_names: set[str] = set(global_symbols or ())
        self._range_alias_stack: list[set[str]] = []
        self._scope_manager = ScopeManager.create()
        self._following_loads_stack: list[tuple[_BlockLastReadIndex, int]] = []
        self._binding_origin_markers: dict[Binding, str] = {}
        self._runtime_origin_initializers: dict[int, list[ast.stmt]] = {}
        self._runtime_transform_depth = 0
        self._runtime_local_bindings: set[Binding] = set()
        self._tensor_store_assignments: set[ast.Assign] = set()
        self._statement_facts = statement_facts or _StatementFactsCache()
        # Names captured before rewriting, keyed by the emitted statement.
        self._rewritten_assigned_names: dict[ast.stmt, frozenset[str]] = {}
        self._control_flow_analyzer = _ControlFlowAnalyzer(self._statement_facts)
        self._function_plan_stack: list[FunctionPlan] = []
        self._global_symbols = global_symbols or {}
        self._filename = filename
        self._line_offset = line_offset
        self._source_text = source_text
        self._source_snapshot = _SourceSnapshot(source_text)
        self._source_snapshot_name: str | None = None
        self._source_lines = source_text.splitlines()
        self._tensor_store_helper_name: str | None = None
        self._lazy_operand_context: str | None = None
        self._call_shadow_stack: list[set[str]] = []
        self._lexical_shadow_stack: list[set[str]] = []
        self._trusted_identities = trusted_identities or _trusted_dsl_identities()
        self._root_plan = root_plan
        self._source_plan_membership: dict[int, bool] = {}
        trusted_symbols = _trusted_dsl_symbols(
            self._global_symbols, self._trusted_identities
        )
        self._tla_range_names = set(trusted_symbols.range_names)
        self._builtin_range_names = set(trusted_symbols.builtin_range_names)
        self._tla_range_constexpr_names = set(trusted_symbols.range_constexpr_names)
        self._tla_const_expr_names = set(trusted_symbols.const_expr_names)
        self._tla_as_numeric_names = set(trusted_symbols.as_numeric_names)
        self._tla_module_aliases = set(trusted_symbols.module_aliases)

    def visit_Module(self, node: ast.Module) -> Any:
        self._reserved_names.update(_identifier_names(node))
        self._tensor_store_helper_name = self._fresh("tensor_store")
        self._source_snapshot_name = self._fresh("source_snapshot")
        return self.generic_visit(node)

    @property
    def tensor_store_helper_name(self) -> str:
        if self._tensor_store_helper_name is None:
            raise RuntimeError("frontend module has not been initialized")
        return self._tensor_store_helper_name

    def _fresh(self, prefix: str) -> str:
        while True:
            self._counter += 1
            name = f"__tladsl_{prefix}_{self._counter}"
            if name in self._reserved_names:
                continue
            self._reserved_names.add(name)
            return name

    def _ensure_runtime_origin_markers(self, plan: ControlFlowPlan) -> None:
        initializers: list[ast.stmt] = []
        for binding in sorted(
            plan.active_bindings & plan.assigned_bindings,
            key=lambda item: item.name,
        ):
            if binding in self._runtime_local_bindings:
                continue
            # A nested runtime region can receive this binding as a materialized
            # block argument. Preserve the first marker instead of classifying
            # that runtime representation as a new source value: the promotion
            # rule is about the binding's provenance before entering runtime
            # control flow.
            if binding in self._binding_origin_markers:
                continue
            marker = self._fresh(f"{binding.name}_compiletime_origin")
            self._binding_origin_markers[binding] = marker
            assignment = ast.Assign(
                targets=[ast.Name(id=marker, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(
                        id=_INTERNAL_RUNTIME_BINDING_IS_COMPILETIME,
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=binding.name, ctx=ast.Load())],
                    keywords=[],
                ),
            )
            assignment.lineno = plan.lineno
            assignment.col_offset = plan.col_offset
            initializers.append(assignment)
        self._runtime_origin_initializers[id(plan)] = initializers

    def _runtime_write_guards(self, plan: ControlFlowPlan) -> list[ast.stmt]:
        return self._runtime_origin_initializers.pop(id(plan), [])

    def _is_direct_as_numeric_call(self, value: ast.expr) -> bool:
        if not isinstance(value, ast.Call) or value.keywords or len(value.args) != 1:
            return False
        func = value.func
        is_module_member = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self._tla_module_aliases
            and func.value.id not in self._recognition_scope()
            and func.attr == "as_numeric"
        )
        is_direct_import = (
            isinstance(func, ast.Name)
            and func.id in self._tla_as_numeric_names
            and func.id not in self._recognition_scope()
        )
        return is_module_member or is_direct_import

    def _is_static_comprehension_iterable(self, value: ast.expr) -> bool:
        """Accept only fixed Python iterables used to build runtime carriers."""
        if isinstance(value, (ast.List, ast.Tuple)):
            # The literal fixes the carrier's structure.  Its elements may be
            # runtime expressions, because ``as_numeric`` materializes each
            # element independently in the comprehension body.
            return not any(isinstance(node, ast.Starred) for node in ast.walk(value))
        return False

    @staticmethod
    def _is_compiletime_dict_key_literal(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) or (
            isinstance(value, ast.Tuple)
            and all(
                _FrontendControlFlowTransformer._is_compiletime_dict_key_literal(
                    element
                )
                for element in value.elts
            )
        )

    def _compiletime_comprehension_target_names(
        self, generators: list[ast.comprehension]
    ) -> set[str]:
        """Return targets whose literal source values are compile-time keys."""

        name_is_compiletime: dict[str, bool] = {}

        def target_paths(
            target: ast.expr, generator: ast.comprehension, path: tuple[int, ...] = ()
        ) -> None:
            if isinstance(target, ast.Name):
                source_values: list[ast.expr] = []
                if not isinstance(generator.iter, (ast.List, ast.Tuple)):
                    is_compiletime = False
                else:
                    for element in generator.iter.elts:
                        source = element
                        for index in path:
                            if not isinstance(source, (ast.List, ast.Tuple)) or (
                                index >= len(source.elts)
                            ):
                                break
                            source = source.elts[index]
                        else:
                            source_values.append(source)
                    is_compiletime = len(source_values) == len(
                        generator.iter.elts
                    ) and all(
                        self._is_compiletime_dict_key_literal(source)
                        for source in source_values
                    )
                name_is_compiletime[target.id] = (
                    name_is_compiletime.get(target.id, True) and is_compiletime
                )
                return
            if isinstance(target, (ast.List, ast.Tuple)):
                for index, element in enumerate(target.elts):
                    target_paths(element, generator, (*path, index))

        for generator in generators:
            target_paths(generator.target, generator)
        return {
            name
            for name, is_compiletime in name_is_compiletime.items()
            if is_compiletime
        }

    def _is_static_runtime_comprehension(
        self, value: ast.ListComp | ast.DictComp | ast.GeneratorExp
    ) -> bool:
        if not value.generators or any(
            generator.is_async
            or generator.ifs
            or not self._is_static_comprehension_iterable(generator.iter)
            for generator in value.generators
        ):
            return False
        if isinstance(value, ast.DictComp):
            key_is_compiletime = self._is_compiletime_dict_key_literal(value.key) or (
                isinstance(value.key, ast.Name)
                and value.key.id
                in self._compiletime_comprehension_target_names(value.generators)
            )
            return self._is_direct_as_numeric_call(value.value) and key_is_compiletime
        return self._is_direct_as_numeric_call(value.elt)

    def _is_explicit_runtime_promotion(self, value: ast.expr) -> bool:
        if self._is_direct_as_numeric_call(value):
            return True
        if isinstance(value, ast.ListComp):
            return self._is_static_runtime_comprehension(value)
        if isinstance(value, ast.DictComp):
            return self._is_static_runtime_comprehension(value)
        return self._is_builtin_tuple_promotion_call(value)

    def _is_builtin_tuple_promotion_call(self, value: ast.expr | None) -> bool:
        """Whether ``value`` uses the unshadowed built-in tuple constructor."""

        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "tuple"
            and value.func.id not in self._recognition_scope()
            and self._global_symbols.get("tuple", builtins.tuple) is builtins.tuple
            and not value.keywords
            and len(value.args) == 1
            and isinstance(value.args[0], ast.GeneratorExp)
            and self._is_static_runtime_comprehension(value.args[0])
        )

    def _rewrite_builtin_tuple_promotion(self, value: ast.expr | None) -> None:
        """Bind a recognized promotion to the injected built-in tuple object."""

        if not self._is_builtin_tuple_promotion_call(value):
            return
        if not isinstance(value, ast.Call):
            return
        value.func = ast.copy_location(
            ast.Attribute(
                value=ast.Name(id="__tladsl_builtins__", ctx=ast.Load()),
                attr="tuple",
                ctx=ast.Load(),
            ),
            value.func,
        )

    def _runtime_assignment_guards(
        self,
        node: ast.Assign | ast.AnnAssign | ast.AugAssign | ast.For,
        targets: list[ast.expr],
        value: ast.expr | None,
    ) -> list[ast.stmt]:
        if self._runtime_transform_depth == 0 or not self._function_plan_stack:
            return []
        explicit_promotion = value is not None and self._is_explicit_runtime_promotion(
            value
        )
        function_plan = self._function_plan_stack[-1]
        source_lines = self._source_lines
        guards: list[ast.stmt] = []
        target_names = set().union(*(_assigned_names(target) for target in targets))
        for name in sorted(target_names):
            binding = function_plan.resolve(name)
            if binding is None or binding in self._runtime_local_bindings:
                continue
            marker = self._binding_origin_markers.get(binding)
            if marker is None:
                continue
            relative_lineno = int(getattr(node, "lineno", 0) or 0)
            relative_col_offset = int(getattr(node, "col_offset", 0) or 0)
            source_line = (
                source_lines[relative_lineno - 1]
                if 0 < relative_lineno <= len(source_lines)
                else None
            )
            call = ast.Call(
                func=ast.Name(id=_INTERNAL_RUNTIME_WRITE_CHECK, ctx=ast.Load()),
                args=[
                    ast.Name(id=marker, ctx=ast.Load()),
                    ast.Constant(value=explicit_promotion),
                    ast.Constant(value=name),
                    ast.Constant(value=self._runtime_construct_name(node)),
                    ast.Constant(value=self._filename),
                    ast.Constant(value=self._line_offset + relative_lineno),
                    ast.Constant(value=relative_col_offset),
                    ast.Constant(value=source_line),
                ],
                keywords=[],
            )
            guards.append(ast.copy_location(ast.Expr(value=call), node))
        return guards

    def _refresh_compiletime_origin_markers(
        self, node: ast.Assign | ast.AnnAssign, targets: list[ast.expr]
    ) -> list[ast.stmt]:
        """Refresh provenance after an ordinary lexical-scope rebinding."""

        if self._runtime_transform_depth != 0 or not self._function_plan_stack:
            return []
        function_plan = self._function_plan_stack[-1]
        refreshes: list[ast.stmt] = []
        target_names = set().union(*(_assigned_names(target) for target in targets))
        for name in sorted(target_names):
            binding = function_plan.resolve(name)
            marker = self._binding_origin_markers.get(binding)
            if marker is None:
                continue
            refresh = ast.Assign(
                targets=[ast.Name(id=marker, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(
                        id=_INTERNAL_RUNTIME_BINDING_IS_COMPILETIME, ctx=ast.Load()
                    ),
                    args=[ast.Name(id=name, ctx=ast.Load())],
                    keywords=[],
                ),
            )
            refreshes.append(ast.copy_location(refresh, node))
        return refreshes

    def _runtime_construct_name(self, node: ast.AST) -> str:
        # The innermost runtime construct owns the transformed statement. The
        # diagnostic contract only requires the source-local dynamic scope.
        del node
        return "control-flow scope"

    def _source_info_dict(
        self,
        generated_name: str,
        node: ast.AST,
        *,
        construct: str,
        region: str,
    ) -> ast.Dict:
        span = tuple(
            getattr(node, field, None)
            for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
        )
        return ast.Dict(
            keys=[
                ast.Constant(value="filename"),
                ast.Constant(value="lineno"),
                ast.Constant(value="col_offset"),
                ast.Constant(value="construct"),
                ast.Constant(value="region"),
                ast.Constant(value="generated_name"),
                ast.Constant(value="_source_ref"),
            ],
            values=[
                ast.Constant(value=self._filename),
                ast.Constant(
                    value=self._line_offset + int(getattr(node, "lineno", 0) or 0)
                ),
                ast.Constant(value=int(getattr(node, "col_offset", 0) or 0)),
                ast.Constant(value=construct),
                ast.Constant(value=region),
                ast.Constant(value=generated_name),
                ast.Tuple(
                    elts=[
                        ast.Name(id=self._source_snapshot_name, ctx=ast.Load()),
                        ast.Constant(value=span),
                    ],
                    ctx=ast.Load(),
                ),
            ],
        )

    def _source_info_stmt(
        self,
        function_name: str,
        node: ast.AST,
        *,
        construct: str,
        region: str,
    ) -> ast.stmt:
        info = self._source_info_dict(
            function_name, node, construct=construct, region=region
        )
        stmt = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=function_name, ctx=ast.Load()),
                    attr=_SOURCE_INFO_ATTR,
                    ctx=ast.Store(),
                )
            ],
            value=info,
        )
        return ast.copy_location(stmt, node)

    def _range_aliases(self) -> set[str]:
        if not self._range_alias_stack:
            self._range_alias_stack.append(set())
        return self._range_alias_stack[-1]

    def _local_scope(self) -> set[str]:
        return self._scope_manager.get_active_symbols()

    def _trusted_symbol_shadows(self) -> set[str]:
        return {name for scope in self._lexical_shadow_stack for name in scope}

    def _recognition_scope(self) -> set[str]:
        return self._local_scope() | self._trusted_symbol_shadows()

    def _active_callables(self) -> set[str]:
        return self._scope_manager.get_active_callables()

    def _following_loads(self, candidates: Iterable[str]) -> set[str]:
        if not self._following_loads_stack:
            return set()
        last_read_index, index = self._following_loads_stack[-1]
        return last_read_index.used_after(index, candidates)

    def _is_static_control_flow_test(self, node: ast.AST) -> bool:
        return _is_static_python_if_test(
            node,
            self._tla_const_expr_names,
            self._tla_module_aliases,
            self._local_scope() | self._trusted_symbol_shadows(),
            self._global_symbols,
        )

    def analyze_control_flow(
        self,
        *,
        node: ast.If | ast.For | ast.While,
        construct_name: str,
        assigned_regions: list[list[ast.stmt]],
        active_call_nodes: list[ast.AST],
    ) -> ControlFlowPlan:
        self._reject_dynamic_helper_rebindings(active_call_nodes)
        plan = self._control_flow_analyzer.analyze(
            node=node,
            construct_name=construct_name,
            assigned_regions=assigned_regions,
            active_call_nodes=active_call_nodes,
            active_symbols=set(self._local_scope()),
            active_callables=set(self._active_callables()),
            function_plan=(
                self._function_plan_stack[-1] if self._function_plan_stack else None
            ),
            filename=self._filename,
            line_offset=self._line_offset,
            source_text=self._source_text,
            is_runtime_for=lambda nested: _is_tla_range_iter(
                nested.iter,
                self._range_aliases(),
                self._tla_range_names,
                self._tla_module_aliases,
                self._local_scope() | self._trusted_symbol_shadows(),
            ),
            is_static_test=self._is_static_control_flow_test,
        )
        self._tensor_store_assignments.update(plan.tensor_store_assignments)
        self._ensure_runtime_origin_markers(plan)
        return plan

    def _reject_dynamic_helper_rebindings(self, body: list[ast.AST]) -> None:
        transformer = self

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    helper = transformer._scope_manager.get_callable_definition(
                        node.func.id
                    )
                    if helper is not None and _has_nonlocal_or_global_declaration(
                        helper
                    ):
                        transformer._raise_source_syntax(
                            node.func,
                            "bare Python helpers called from runtime control flow "
                            "cannot declare global or nonlocal bindings",
                        )
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                del node

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef

        visitor = Visitor()
        for statement in body:
            visitor.visit(statement)

    def visit_Assign(self, node: ast.Assign) -> Any:
        assigned_names = self._statement_facts.get(node).assigned_names
        self._record_runtime_local_bindings(node)
        guards = self._runtime_assignment_guards(node, node.targets, node.value)
        if self._runtime_transform_depth != 0:
            self._rewrite_builtin_tuple_promotion(node.value)
        refreshes = self._refresh_compiletime_origin_markers(node, node.targets)
        if node not in self._tensor_store_assignments:
            rewritten = self.generic_visit(node)
            self._rewritten_assigned_names[rewritten] = assigned_names
            statements = [*guards, rewritten, *refreshes]
            return statements if len(statements) > 1 else rewritten
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
            raise SyntaxError(
                "dynamic Tla control flow requires a tensor store to have one target"
            )
        target = node.targets[0]
        rewritten = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=self.tensor_store_helper_name, ctx=ast.Load()),
                # Python assignment evaluates RHS before the target and index.
                args=[
                    self.visit(node.value),
                    self.visit(target.value),
                    self.visit(target.slice),
                ],
                keywords=[],
            )
        )
        rewritten = ast.copy_location(rewritten, node)
        self._rewritten_assigned_names[rewritten] = assigned_names
        statements = [*guards, rewritten, *refreshes]
        return statements if len(statements) > 1 else rewritten

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        assigned_names = self._statement_facts.get(node).assigned_names
        self._record_runtime_local_bindings(node)
        guards = self._runtime_assignment_guards(node, [node.target], node.value)
        if self._runtime_transform_depth != 0:
            self._rewrite_builtin_tuple_promotion(node.value)
        refreshes = self._refresh_compiletime_origin_markers(node, [node.target])
        rewritten = self.generic_visit(node)
        self._rewritten_assigned_names[rewritten] = assigned_names
        statements = [*guards, rewritten, *refreshes]
        return statements if len(statements) > 1 else rewritten

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        assigned_names = self._statement_facts.get(node).assigned_names
        guards = self._runtime_assignment_guards(node, [node.target], None)
        rewritten = self.generic_visit(node)
        self._rewritten_assigned_names[rewritten] = assigned_names
        return [*guards, rewritten] if guards else rewritten

    def _record_runtime_local_bindings(self, node: ast.Assign | ast.AnnAssign) -> None:
        if self._runtime_transform_depth == 0 or not self._function_plan_stack:
            return
        active = self._local_scope()
        function_plan = self._function_plan_stack[-1]
        for name in self._statement_facts.get(node).assigned_names - active:
            binding = function_plan.resolve(name)
            if binding is not None:
                self._runtime_local_bindings.add(binding)

    def _record_runtime_for_target(self, target: ast.expr) -> None:
        if self._runtime_transform_depth == 0 or not self._function_plan_stack:
            return
        function_plan = self._function_plan_stack[-1]
        for name in _assigned_names(target):
            binding = function_plan.resolve(name)
            if binding is not None:
                self._runtime_local_bindings.add(binding)

    @contextlib.contextmanager
    def _runtime_transform_scope(self) -> Iterator[None]:
        self._runtime_transform_depth += 1
        try:
            yield
        finally:
            self._runtime_transform_depth -= 1

    def _visit_statement_list(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rewritten_body: list[ast.stmt] = []
        aliases = self._range_aliases()
        last_read_index = _BlockLastReadIndex.build(
            body, statement_facts=self._statement_facts
        )
        for index, stmt in enumerate(body):
            self._following_loads_stack.append((last_read_index, index))
            try:
                rewritten = self.visit(stmt)
                stmts = rewritten if isinstance(rewritten, list) else [rewritten]
                for out_stmt in stmts:
                    if out_stmt is None:
                        continue
                    rewritten_body.append(out_stmt)
                    _update_range_aliases(
                        out_stmt,
                        aliases,
                        self._tla_range_names,
                        self._tla_module_aliases,
                        self._recognition_scope(),
                    )
                    assigned_names = self._rewritten_assigned_names.get(out_stmt)
                    if assigned_names is None:
                        assigned_names = _assigned_names(out_stmt)
                    self._scope_manager.add_names_to_scope(assigned_names)
                    for callable_name in _callable_names(
                        out_stmt, self._scope_manager.get_active_callables()
                    ):
                        self._scope_manager.add_to_callables(callable_name)
                    if isinstance(out_stmt, ast.FunctionDef):
                        self._scope_manager.add_callable_definition(out_stmt)
                    elif (
                        isinstance(out_stmt, ast.Assign)
                        and len(out_stmt.targets) == 1
                        and isinstance(out_stmt.targets[0], ast.Name)
                        and isinstance(out_stmt.value, ast.Name)
                    ):
                        self._scope_manager.add_callable_alias(
                            out_stmt.targets[0].id, out_stmt.value.id
                        )
            finally:
                self._following_loads_stack.pop()
        return rewritten_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self._function_plan_stack:
            # A nested definition is a bare Python helper. Preserve it
            # verbatim whether it appears in the root or an outlined region:
            # its body executes under the caller's active emission context and
            # is never independently transformed.
            return node
        if self._root_plan is None:
            raise RuntimeError(
                "frontend transformer requires an authoritative root FunctionPlan"
            )
        function_plan = self._root_plan
        self._source_plan_membership = _align_function_child_plans(
            node, function_plan.child_plans
        )
        self._function_plan_stack.append(function_plan)
        self._range_alias_stack.append(set())
        local_names = _function_arg_names(node.args) | _assigned_names_from_statements(
            node.body, statement_facts=self._statement_facts
        )
        self._call_shadow_stack.append(local_names)
        lexical_shadows = {binding.name for binding in function_plan.bindings}
        self._lexical_shadow_stack.append(lexical_shadows)
        with self._scope_manager.enter_local_scope(_function_arg_names(node.args)):
            try:
                node.args = self.visit(node.args)
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
                node.decorator_list = [
                    self.visit(decorator) for decorator in node.decorator_list
                ]
                if node.returns is not None:
                    node.returns = self.visit(node.returns)
                return node
            finally:
                self._call_shadow_stack.pop()
                self._lexical_shadow_stack.pop()
                self._range_alias_stack.pop()
                self._function_plan_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._raise_source_syntax(
            node,
            f"nested async function definition '{node.name}' is not supported "
            "in a transformed Tla function",
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._raise_source_syntax(
            node,
            f"nested class definition '{node.name}' is not supported "
            "in a transformed Tla function",
        )

    def visit_Global(self, node: ast.Global) -> Any:
        self._raise_source_syntax(
            node, "global declarations are not supported in a transformed Tla function"
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        self._raise_source_syntax(
            node,
            "nonlocal declarations are not supported in a transformed Tla function",
        )

    def _raise_source_syntax(self, node: ast.AST, message: str) -> None:
        error = SyntaxError(message)
        error.filename = self._filename
        error.lineno = self._line_offset + int(getattr(node, "lineno", 0) or 0)
        error.offset = int(getattr(node, "col_offset", 0) or 0) + 1
        source = ast.get_source_segment(self._source_text, node) or ""
        error.text = next(iter(source.splitlines()), None)
        raise error

    def _validate_reserved_module_call(self, node: ast.AST) -> None:
        """Reject mutation of a recognized canonical ``tla`` member."""

        if not isinstance(node, ast.Call):
            return
        path = self._module_qualified_dsl_path(node.func)
        if path is None:
            return
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            raise RuntimeError("recognized module-qualified DSL call has no name root")
        try:
            _checked_dsl_member(self._global_symbols[root.id], path)
        except (KeyError, RuntimeError):
            self._raise_source_syntax(
                node.func,
                "reserved module-qualified call must use the genuine TLA Python member",
            )

    def _module_qualified_dsl_path(self, func: ast.expr) -> str | None:
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self._tla_module_aliases
            and func.value.id not in self._recognition_scope()
            and func.attr
            in {
                "range",
                "range_constexpr",
                "const_expr",
                "as_numeric",
                "cube",
                "vector",
            }
        ):
            return func.attr
        if _is_tla_vec_func(func, self._tla_module_aliases, self._recognition_scope()):
            return "vec.func"
        return None

    def _checked_module_callable(self, func: ast.expr, path: str) -> ast.expr:
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            raise RuntimeError("recognized module-qualified DSL call has no name root")
        checked = ast.Call(
            func=ast.Name(id=_INTERNAL_CHECKED_DSL_MEMBER, ctx=ast.Load()),
            args=[root, ast.Constant(value=path)],
            keywords=[],
        )
        return ast.copy_location(checked, func)

    def _direct_dsl_path(self, func: ast.expr) -> str | None:
        if not isinstance(func, ast.Name) or func.id in self._recognition_scope():
            return None
        if func.id in self._builtin_range_names:
            return "builtin_range"
        if func.id in self._tla_range_names:
            return "range"
        if func.id in self._tla_range_constexpr_names:
            return "range_constexpr"
        if func.id in self._tla_const_expr_names:
            return "const_expr"
        return None

    def _checked_direct_callable(self, func: ast.expr, path: str) -> ast.expr:
        checked = ast.Call(
            func=ast.Name(id=_INTERNAL_CHECKED_DSL_IDENTITY, ctx=ast.Load()),
            args=[func, ast.Constant(value=path)],
            keywords=[],
        )
        return ast.copy_location(checked, func)

    def visit_For(self, node: ast.For) -> Any:
        self._validate_reserved_module_call(node.iter)
        is_range_constexpr = _is_tla_range_constexpr_call(
            node.iter,
            self._tla_range_constexpr_names,
            self._tla_module_aliases,
            self._recognition_scope(),
        )
        recognized_range = _is_tla_range_call(
            node.iter,
            self._tla_range_names | self._builtin_range_names,
            self._tla_module_aliases,
            self._recognition_scope(),
        )
        if recognized_range:
            negative_step_prelude = self._rewrite_negative_step_range(node)
        else:
            negative_step_prelude = []
        node.iter = self.visit(node.iter)
        if is_range_constexpr:
            if not isinstance(node.iter, ast.Call):
                raise RuntimeError("recognized range_constexpr iterator is not a call")
            check_stmts = (
                [ast.copy_location(ast.Expr(value=node.iter.func), node.iter.func)]
                if isinstance(node.iter.func, ast.Call)
                else _cf_symbol_check_stmts(node.iter)
            )
            node.iter = _static_range_call_from_range_constexpr(node.iter)
            self._scope_manager.add_names_to_scope(_assigned_names(node.target))
            self._range_alias_stack.append(set())
            try:
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
            finally:
                self._range_alias_stack.pop()
            self._range_alias_stack.append(set())
            try:
                node.orelse = self._visit_statement_list(node.orelse)
            finally:
                self._range_alias_stack.pop()
            return [*check_stmts, node]

        planned = self._source_plan_membership.get(id(node))
        is_tla_range = (
            planned
            if planned is not None
            else _is_tla_range_iter(
                node.iter,
                self._range_aliases(),
                self._tla_range_names | self._builtin_range_names,
                self._tla_module_aliases,
                self._recognition_scope(),
            )
        )
        if is_tla_range:
            if node.orelse:
                raise SyntaxError("dynamic Tla for does not support for-else")
        if not is_tla_range:
            # A Python ``for`` nested in runtime control flow executes while
            # lowering, but its target still writes a binding in the enclosing
            # kernel scope.  Check that write before marking the target as a
            # runtime-local binding so it cannot overwrite a compile-time
            # value and later be read as though it were still compile-time.
            guards = self._runtime_assignment_guards(node, [node.target], None)
            self._record_runtime_for_target(node.target)
            self._scope_manager.add_names_to_scope(_assigned_names(node.target))
            self._range_alias_stack.append(set())
            try:
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
            finally:
                self._range_alias_stack.pop()
            self._range_alias_stack.append(set())
            try:
                node.orelse = self._visit_statement_list(node.orelse)
            finally:
                self._range_alias_stack.pop()
            return [*guards, node] if guards else node
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("dynamic Tla for requires a simple local name target")

        analysis = self.analyze_control_flow(
            node=node,
            construct_name="for",
            assigned_regions=[node.body],
            active_call_nodes=node.body,
        )
        _reject_unsupported_dynamic_for_new_defs(
            analysis.active_symbols,
            analysis.assigned_names,
            node.target.id,
            self._following_loads(analysis.assigned_names | {node.target.id}),
        )
        carried_names = list(analysis.carried_names)
        self._scope_manager.add_names_to_scope(carried_names)

        body = self._transform_nested_function_body(
            node.body,
            [*analysis.active_symbols, node.target.id, *carried_names],
            analysis.active_callables,
            {node.target.id},
        )
        range_name = self._fresh("range")
        body_name = self._fresh("loop_body")

        range_assign = ast.Assign(
            targets=[ast.Name(id=range_name, ctx=ast.Store())],
            value=node.iter,
        )
        ast.copy_location(range_assign, node)

        body_fn = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=arg, annotation=None)
                    for arg in [node.target.id, *carried_names]
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=_append_return_for_carried_names(body, carried_names),
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(body_fn, node)

        helper_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_INTERNAL_FOR, ctx=ast.Load()),
                args=[
                    ast.Name(id=range_name, ctx=ast.Load()),
                    ast.Name(id=body_name, ctx=ast.Load()),
                    *[ast.Name(id=name, ctx=ast.Load()) for name in carried_names],
                ],
                keywords=[
                    ast.keyword(
                        arg="carried_names",
                        value=ast.Tuple(
                            elts=[ast.Constant(value=name) for name in carried_names],
                            ctx=ast.Load(),
                        ),
                    ),
                    *_captured_keywords(analysis),
                ],
            )
        )
        ast.copy_location(helper_call, node)

        if not carried_names:
            helper_stmt: ast.stmt = helper_call
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=helper_call.value,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=helper_call.value,
            )
        ast.copy_location(helper_stmt, node)

        result: list[ast.stmt] = []
        if recognized_range and not negative_step_prelude:
            result.extend(_cf_symbol_check_stmts(node.iter))
        result.extend(
            [
                *negative_step_prelude,
                range_assign,
                *self._runtime_write_guards(analysis),
                body_fn,
                self._source_info_stmt(
                    body_name, node, construct="dynamic for", region="body"
                ),
                helper_stmt,
            ]
        )
        return result

    def _rewrite_negative_step_range(self, node: ast.For) -> list[ast.stmt]:
        if not isinstance(node.iter, ast.Call) or not isinstance(node.target, ast.Name):
            return []
        bounds = _extract_range_call_bounds(node.iter)
        if bounds is None or not bounds.has_explicit_step:
            return []

        start_original = self._fresh("start_ori")
        stop_original = self._fresh("stop_ori")
        step_original = self._fresh("step_ori")
        is_negative = self._fresh("is_negative")
        start_name = self._fresh("start")
        stop_name = self._fresh("stop")
        step_name = self._fresh("step")
        offset_name = self._fresh("offset")
        callable_name = self._fresh("range_callable")
        module_path = self._module_qualified_dsl_path(node.iter.func)
        direct_path = self._direct_dsl_path(node.iter.func)
        if module_path == "range":
            checked_callable = self._checked_module_callable(node.iter.func, "range")
        elif direct_path in {"range", "builtin_range"}:
            checked_callable = self._checked_direct_callable(
                node.iter.func, direct_path
            )
        else:
            raise RuntimeError("recognized explicit-step range lost its callable role")

        prelude: list[ast.stmt] = [
            _assign_name(callable_name, checked_callable, node),
            _assign_name(start_original, bounds.start, node),
            _assign_name(stop_original, bounds.end, node),
            _assign_name(step_original, bounds.step, node),
            _assign_name(
                is_negative,
                ast.Compare(
                    left=ast.Name(id=step_original, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)],
                ),
                node,
            ),
            _assign_name(
                start_name,
                _if_expr(
                    is_negative,
                    ast.Name(id=stop_original, ctx=ast.Load()),
                    ast.Name(id=start_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                stop_name,
                _if_expr(
                    is_negative,
                    ast.Name(id=start_original, ctx=ast.Load()),
                    ast.Name(id=stop_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                step_name,
                _if_expr(
                    is_negative,
                    _index_sub_call(
                        ast.Constant(value=0),
                        ast.Name(id=step_original, ctx=ast.Load()),
                    ),
                    ast.Name(id=step_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                offset_name,
                _if_expr(
                    is_negative,
                    _index_add_call(
                        ast.Name(id=start_name, ctx=ast.Load()),
                        ast.Name(id=stop_name, ctx=ast.Load()),
                    ),
                    ast.Constant(value=0),
                ),
                node,
            ),
        ]

        node.iter.args = [
            ast.Name(id=start_name, ctx=ast.Load()),
            ast.Name(id=stop_name, ctx=ast.Load()),
            ast.Name(id=step_name, ctx=ast.Load()),
        ]
        node.iter.func = ast.copy_location(
            ast.Name(id=callable_name, ctx=ast.Load()), node.iter.func
        )

        target_name = node.target.id
        remap = _assign_name(
            target_name,
            _if_expr(
                is_negative,
                _index_sub_call(
                    ast.Name(id=offset_name, ctx=ast.Load()),
                    ast.Name(id=target_name, ctx=ast.Load()),
                ),
                ast.Name(id=target_name, ctx=ast.Load()),
            ),
            node.target,
        )
        node.body.insert(0, remap)

        transformed: list[ast.stmt] = []
        for stmt in prelude:
            visited = self.visit(stmt)
            transformed.extend(visited if isinstance(visited, list) else [visited])
        return transformed

    def visit_If(self, node: ast.If) -> Any:
        self._validate_reserved_module_call(node.test)
        self._validate_dynamic_condition(node.test, "if")
        planned = self._source_plan_membership.get(id(node))
        if planned is False or (
            planned is None and self._is_static_control_flow_test(node.test)
        ):
            node.test = self.visit(node.test)
            node.body = self._transform_static_statement_region(node.body)
            node.orelse = self._transform_static_statement_region(node.orelse)
            if isinstance(node.test, ast.Call) and _is_constexpr_cf_test(
                node.test,
                self._tla_const_expr_names,
                self._tla_module_aliases,
                self._recognition_scope(),
            ):
                return [*_cf_symbol_check_stmts(node.test), node]
            return node
        analysis = self.analyze_control_flow(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[*node.body, *node.orelse],
        )
        then_assigned = analysis.assigned_by_region[0]
        else_assigned = analysis.assigned_by_region[1]
        _reject_unsupported_dynamic_if_new_defs(
            analysis.active_symbols,
            then_assigned,
            else_assigned,
            self._following_loads(then_assigned | else_assigned),
        )
        carried_names = list(analysis.carried_names)
        result = self._runtime_write_guards(analysis)

        test = self.visit(node.test)
        then_body = self._transform_nested_function_body(
            node.body, carried_names, analysis.active_callables
        )
        else_body = (
            self._transform_nested_function_body(
                node.orelse, carried_names, analysis.active_callables
            )
            if node.orelse
            else [ast.Return(value=_names_list(carried_names))]
        )

        then_name = self._fresh("if_then")
        else_name = self._fresh("if_else") if node.orelse or carried_names else None

        then_fn = self._branch_function(then_name, carried_names, then_body)
        result.extend(
            [
                then_fn,
                self._source_info_stmt(
                    then_name,
                    node.body[0] if node.body else node,
                    construct="dynamic if",
                    region="then-region",
                ),
            ]
        )
        if else_name is not None:
            result.append(self._branch_function(else_name, carried_names, else_body))
            result.append(
                self._source_info_stmt(
                    else_name,
                    node.orelse[0] if node.orelse else node,
                    construct="dynamic if",
                    region="else-region",
                )
            )

        helper_call = ast.Call(
            func=ast.Name(id=_INTERNAL_IF, ctx=ast.Load()),
            args=[
                test,
                ast.Name(id=then_name, ctx=ast.Load()),
                (
                    ast.Name(id=else_name, ctx=ast.Load())
                    if else_name is not None
                    else ast.Constant(value=None)
                ),
                *[ast.Name(id=name, ctx=ast.Load()) for name in carried_names],
            ],
            keywords=[
                ast.keyword(
                    arg="carried_names",
                    value=ast.Tuple(
                        elts=[ast.Constant(value=name) for name in carried_names],
                        ctx=ast.Load(),
                    ),
                ),
                *_captured_keywords(analysis),
            ],
        )
        ast.copy_location(helper_call, node)
        if not carried_names:
            helper_stmt: ast.stmt = ast.Expr(value=helper_call)
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=helper_call,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=helper_call,
            )
        ast.copy_location(helper_stmt, node)
        result.append(helper_stmt)
        return result

    def visit_While(self, node: ast.While) -> Any:
        self._validate_reserved_module_call(node.test)
        self._validate_dynamic_condition(node.test, "while")
        planned = self._source_plan_membership.get(id(node))
        if planned is False or (
            planned is None and self._is_static_control_flow_test(node.test)
        ):
            is_constexpr_while = _is_constexpr_cf_test(
                node.test,
                self._tla_const_expr_names,
                self._tla_module_aliases,
                self._local_scope() | self._trusted_symbol_shadows(),
            )
            node.test = self.visit(node.test)
            node.body = self._transform_static_statement_region(node.body)
            node.orelse = self._transform_static_statement_region(node.orelse)
            if is_constexpr_while:
                iteration_name = self._fresh("constexpr_while_iteration")
                initialize = ast.Assign(
                    targets=[ast.Name(id=iteration_name, ctx=ast.Store())],
                    value=ast.Constant(value=0),
                )
                ast.copy_location(initialize, node)
                increment = ast.AugAssign(
                    target=ast.Name(id=iteration_name, ctx=ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(value=1),
                )
                ast.copy_location(increment, node)
                warn = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(
                            id=_INTERNAL_STATIC_LOOP_ITERATION, ctx=ast.Load()
                        ),
                        args=[ast.Name(id=iteration_name, ctx=ast.Load())],
                        keywords=[
                            ast.keyword(
                                arg="once_at_threshold",
                                value=ast.Constant(value=True),
                            )
                        ],
                    )
                )
                ast.copy_location(warn, node)
                node.body = [increment, warn, *node.body]
                return [*_cf_symbol_check_stmts(node.test), initialize, node]
            return node
        if node.orelse:
            raise SyntaxError("dynamic Tla while does not support while-else")

        analysis = self.analyze_control_flow(
            node=node,
            construct_name="while",
            assigned_regions=[node.body],
            active_call_nodes=[node.test, *node.body],
        )
        _reject_unsupported_dynamic_while_new_defs(
            analysis.active_symbols,
            analysis.assigned_names,
            self._following_loads(analysis.assigned_names),
        )
        carried_names = list(analysis.carried_names)
        self._scope_manager.add_names_to_scope(carried_names)

        before_body = self._transform_nested_function_body(
            [
                ast.Return(
                    value=ast.List(
                        elts=[node.test, _names_list(carried_names)],
                        ctx=ast.Load(),
                    )
                )
            ],
            carried_names,
            analysis.active_callables,
        )
        after_body = self._transform_nested_function_body(
            node.body, carried_names, analysis.active_callables
        )
        if not _ends_with_return(after_body):
            after_body.append(ast.Return(value=_names_list(carried_names)))

        before_name = self._fresh("while_before")
        after_name = self._fresh("while_after")
        region_name = self._fresh("while_region")
        before_fn = self._branch_function(before_name, carried_names, before_body)
        after_fn = self._branch_function(after_name, carried_names, after_body)
        ast.copy_location(before_fn, node)
        ast.copy_location(after_fn, node)

        execute_call = ast.Call(
            func=ast.Name(id=_WHILE_EXECUTOR, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg="while_before_block",
                    value=ast.Name(id=before_name, ctx=ast.Load()),
                ),
                ast.keyword(
                    arg="while_after_block",
                    value=ast.Name(id=after_name, ctx=ast.Load()),
                ),
                ast.keyword(
                    arg="write_args",
                    value=ast.List(
                        elts=[
                            ast.Name(id=name, ctx=ast.Load()) for name in carried_names
                        ],
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="full_write_args_count",
                    value=ast.Constant(value=analysis.full_write_args_count),
                ),
                ast.keyword(
                    arg="write_args_names",
                    value=ast.List(
                        elts=[ast.Constant(value=name) for name in carried_names],
                        ctx=ast.Load(),
                    ),
                ),
                *_captured_keywords(analysis),
            ],
        )
        ast.copy_location(execute_call, node)

        before_info = self._source_info_stmt(
            before_name, node.test, construct="dynamic while", region="condition-region"
        )
        after_info = self._source_info_stmt(
            after_name,
            node.body[0] if node.body else node,
            construct="dynamic while",
            region="body-region",
        )

        region_fn = ast.FunctionDef(
            name=region_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name, annotation=None) for name in carried_names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[
                before_fn,
                before_info,
                after_fn,
                after_info,
                ast.Return(value=execute_call),
            ],
            decorator_list=[
                ast.Call(
                    func=ast.Name(id=_WHILE_SELECTOR, ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="write_args",
                            value=ast.List(
                                elts=[
                                    ast.Name(id=name, ctx=ast.Load())
                                    for name in carried_names
                                ],
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                )
            ],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(region_fn, node)

        region_result = ast.Name(id=region_name, ctx=ast.Load())
        ast.copy_location(region_result, node)
        if not carried_names:
            helper_stmt: ast.stmt = ast.Expr(value=region_result)
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=region_result,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=region_result,
            )
        ast.copy_location(helper_stmt, node)
        return [*self._runtime_write_guards(analysis), region_fn, helper_stmt]

    def _validate_dynamic_condition(
        self, condition: ast.AST, construct_name: str
    ) -> None:
        _DynamicConditionValidator(
            construct_name,
            filename=self._filename,
            line_offset=self._line_offset,
            source_text=self._source_text,
        ).validate(condition)

    def _transform_nested_function_body(
        self,
        body: list[ast.stmt],
        initial_scope: list[str] | None = None,
        initial_callables: set[str] | None = None,
        shadowed_range_aliases: set[str] | None = None,
    ) -> list[ast.stmt]:
        self._range_alias_stack.append(
            set(self._range_aliases()) - set(shadowed_range_aliases or ())
        )
        with self._runtime_transform_scope():
            with self._scope_manager.enter_local_scope(
                initial_scope, initial_callables
            ):
                try:
                    transformed = self._visit_statement_list(body)
                finally:
                    self._range_alias_stack.pop()
        return transformed or [ast.Pass()]

    def _transform_static_statement_region(
        self, body: list[ast.stmt]
    ) -> list[ast.stmt]:
        self._range_alias_stack.append(set(self._range_aliases()))
        with self._scope_manager.enter_local_scope(
            self._local_scope(), self._active_callables()
        ):
            try:
                return self._visit_statement_list(body)
            finally:
                self._range_alias_stack.pop()

    def _branch_function(
        self, name: str, args: list[str], body: list[ast.stmt]
    ) -> ast.FunctionDef:
        branch_body = list(body)
        if args and not _ends_with_return(branch_body):
            branch_body.append(ast.Return(value=_names_list(args)))
        fn = ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg, annotation=None) for arg in args],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=branch_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return fn

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        then_fn = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=body,
        )
        else_fn = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=orelse,
        )
        ast.copy_location(then_fn, node.body)
        ast.copy_location(else_fn, node.orelse)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_IF_EXPR, ctx=ast.Load()),
                args=[
                    test,
                    ast.Call(
                        func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                        args=[
                            then_fn,
                            self._source_info_dict(
                                "<if expression then>",
                                node.body,
                                construct="conditional expression",
                                region="then-region",
                            ),
                        ],
                        keywords=[],
                    ),
                    ast.Call(
                        func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                        args=[
                            else_fn,
                            self._source_info_dict(
                                "<if expression else>",
                                node.orelse,
                                construct="conditional expression",
                                region="else-region",
                            ),
                        ],
                        keywords=[],
                    ),
                ],
                keywords=[],
            ),
            node,
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            node = self.generic_visit(node)
            return node
        lhs = self.visit(node.values[0])
        for original_rhs in node.values[1:]:
            with self._lazy_operand("boolean"):
                rhs = self.visit(original_rhs)
            lhs_name = self._fresh("bool_op_lhs")
            bound_lhs = ast.NamedExpr(
                target=ast.Name(id=lhs_name, ctx=ast.Store()),
                value=lhs,
            )
            ast.copy_location(bound_lhs, lhs)
            bound_name = ast.Name(id=lhs_name, ctx=ast.Load())
            if isinstance(node.op, ast.And):
                true_value, false_value = rhs, bound_name
            else:
                true_value, false_value = bound_name, rhs
            lhs = self._lazy_conditional_expression(
                bound_lhs,
                true_value,
                false_value,
                original_rhs,
                construct="boolean expression",
            )
            if isinstance(original_rhs, ast.Constant) and isinstance(
                original_rhs.value, bool
            ):
                if isinstance(node.op, ast.And) and not original_rhs.value:
                    break
                if isinstance(node.op, ast.Or) and original_rhs.value:
                    break
        return lhs

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if self._lazy_operand_context is not None and not isinstance(node.op, ast.Not):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_INTERNAL_LAZY_UNARY, ctx=ast.Load()),
                    args=[ast.Constant(value=type(node.op).__name__), operand],
                    keywords=[],
                ),
                node,
            )
        node.operand = operand
        if not isinstance(node.op, ast.Not):
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_BOOL_NOT, ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            ),
            node,
        )

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        comparators = []
        for index, comparator in enumerate(node.comparators):
            if index == 0:
                comparators.append(self.visit(comparator))
            else:
                with self._lazy_operand("chained-comparison"):
                    comparators.append(self.visit(comparator))

        def compare_pair(lhs: ast.expr, rhs: ast.expr, op: ast.cmpop) -> ast.Call:
            return ast.Call(
                func=ast.Name(id=_INTERNAL_COMPARE, ctx=ast.Load()),
                args=[
                    lhs,
                    ast.Tuple(elts=[rhs], ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Constant(value=_compare_op_name(op))], ctx=ast.Load()
                    ),
                ],
                keywords=[],
            )

        def build(index: int, lhs: ast.expr) -> ast.expr:
            rhs = comparators[index]
            if index == len(comparators) - 1:
                return compare_pair(lhs, rhs, node.ops[index])
            rhs_name = self._fresh("compare_rhs")
            bound_rhs = ast.NamedExpr(
                target=ast.Name(id=rhs_name, ctx=ast.Store()), value=rhs
            )
            condition_name = self._fresh("compare_result")
            bound_condition = ast.NamedExpr(
                target=ast.Name(id=condition_name, ctx=ast.Store()),
                value=compare_pair(lhs, bound_rhs, node.ops[index]),
            )
            later = build(index + 1, ast.Name(id=rhs_name, ctx=ast.Load()))
            return self._lazy_conditional_expression(
                bound_condition,
                later,
                ast.Name(id=condition_name, ctx=ast.Load()),
                node.comparators[index + 1],
                construct="chained comparison",
            )

        return ast.copy_location(build(0, left), node)

    def _lazy_conditional_expression(
        self,
        condition: ast.expr,
        true_value: ast.expr,
        false_value: ast.expr,
        source_node: ast.AST,
        *,
        construct: str,
    ) -> ast.Call:
        def region(value: ast.expr, name: str) -> ast.Call:
            function = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None,
                ),
                body=value,
            )
            ast.copy_location(function, source_node)
            return ast.Call(
                func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                args=[
                    function,
                    self._source_info_dict(
                        name, source_node, construct=construct, region=name
                    ),
                ],
                keywords=[],
            )

        call = ast.Call(
            func=ast.Name(id=_INTERNAL_IF_EXPR, ctx=ast.Load()),
            args=[
                condition,
                region(true_value, "lazy-true-region"),
                region(false_value, "lazy-false-region"),
            ],
            keywords=[],
        )
        return ast.copy_location(call, source_node)

    def visit_Call(self, node: ast.Call) -> Any:
        module_path = self._module_qualified_dsl_path(node.func)
        direct_path = self._direct_dsl_path(node.func)
        if self._lazy_operand_context is not None and not (
            module_path is not None
            or direct_path is not None
            or self._is_dsl_call(node)
        ):
            original = self.generic_visit(node)
            thunk = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=original,
            )
            ast.copy_location(thunk, node)
            attached = ast.Call(
                func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                args=[
                    thunk,
                    self._source_info_dict(
                        "unknown-effect-call",
                        node,
                        construct="runtime-lazy operand",
                        region=self._lazy_operand_context,
                    ),
                ],
                keywords=[],
            )
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_INTERNAL_UNKNOWN_EFFECT_CALL, ctx=ast.Load()),
                    args=[attached],
                    keywords=[],
                ),
                node,
            )
        if module_path is not None:
            node.func = self._checked_module_callable(node.func, module_path)
        elif direct_path is not None:
            node.func = self._checked_direct_callable(node.func, direct_path)
        if not isinstance(node.func, ast.Name):
            node.args = [self.visit(arg) for arg in node.args]
            node.keywords = [self.visit(keyword) for keyword in node.keywords]
            return node
        original_func_name = node.func.id
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        if original_func_name in {"any", "all", "bool"}:
            if node.keywords or len(node.args) != 1:
                return node
            helper = _BUILTIN_REDIRECTS[original_func_name]
        elif original_func_name in {"min", "max"}:
            if node.keywords:
                return node
            helper = _BUILTIN_REDIRECTS[original_func_name]
        else:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=node.args,
                keywords=[],
            ),
            node,
        )

    @contextlib.contextmanager
    def _lazy_operand(self, context: str) -> Iterator[None]:
        previous = self._lazy_operand_context
        self._lazy_operand_context = previous or context
        try:
            yield
        finally:
            self._lazy_operand_context = previous

    def _is_dsl_call(self, node: ast.Call) -> bool:
        # Expanding user-provided iterables or mappings invokes arbitrary Python
        # protocols before the trusted call target itself is entered.
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return False
        target = self._resolve_global_call_target(node.func)
        if target is None:
            return False
        return any(target is trusted for trusted in _trusted_lazy_callables())

    def _resolve_global_call_target(self, node: ast.expr) -> Any | None:
        shadowed_names: set[str] = set()
        for scope in self._call_shadow_stack:
            shadowed_names.update(scope)
        if isinstance(node, ast.Name):
            if node.id in shadowed_names or node.id in _BUILTIN_REDIRECTS:
                return None
            return self._global_symbols.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        attributes: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            attributes.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        if current.id in shadowed_names:
            return None
        target = self._global_symbols.get(current.id)
        for attribute in reversed(attributes):
            try:
                target = inspect.getattr_static(target, attribute)
            except (AttributeError, TypeError):
                members = inspect.getattr_static(target, "_members", None)
                if object.__getattribute__(members, "__class__") is dict:
                    if attribute not in members:
                        return None
                    target = members[attribute]
                    continue
                # Module exports via ``__getattr__`` (e.g. ``catlass.tla.arch``)
                # are invisible to ``getattr_static``.
                try:
                    target = getattr(target, attribute)
                except Exception:
                    return None
        return target

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if self._lazy_operand_context is None or not isinstance(node.ctx, ast.Load):
            node.value = value
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_ATTRIBUTE, ctx=ast.Load()),
                args=[value, ast.Constant(value=node.attr)],
                keywords=[],
            ),
            node,
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        slice_value = self.visit(node.slice)
        if self._lazy_operand_context is None or not isinstance(node.ctx, ast.Load):
            node.value, node.slice = value, slice_value
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_SUBSCRIPT, ctx=ast.Load()),
                args=[value, slice_value],
                keywords=[],
            ),
            node,
        )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        if self._lazy_operand_context is None:
            node.left, node.right = left, right
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_BINOP, ctx=ast.Load()),
                args=[ast.Constant(value=type(node.op).__name__), left, right],
                keywords=[],
            ),
            node,
        )

    def visit_Name(self, node: ast.Name) -> Any:
        if not isinstance(node.ctx, ast.Load):
            return node
        helper = _BUILTIN_REDIRECTS.get(node.id)
        if helper is None:
            return node
        return ast.copy_location(
            ast.IfExp(
                test=ast.Compare(
                    left=ast.Name(id=node.id, ctx=ast.Load()),
                    ops=[ast.Is()],
                    comparators=[
                        ast.Attribute(
                            value=ast.Name(id="__tladsl_builtins__", ctx=ast.Load()),
                            attr=node.id,
                            ctx=ast.Load(),
                        )
                    ],
                ),
                body=ast.Name(id=helper, ctx=ast.Load()),
                orelse=ast.Name(id=node.id, ctx=ast.Load()),
            ),
            node,
        )

    def visit_With(self, node: ast.With) -> Any:
        # Names bound in the enclosing scope(s) before this region. A region body
        # is hoisted into a nested function but is semantically an inline block
        # sharing the enclosing scope, so enclosing names reassigned inside it
        # must be threaded via ``nonlocal`` (see below).
        enclosing_symbols = set(self._local_scope())
        planned = self._source_plan_membership.get(id(node))
        region_name = None
        region_mode = None
        region_thread_block_dim = None
        region_guard: ast.expr | None = None
        analysis: ControlFlowPlan | None = None
        if len(node.items) == 1 and planned is not False:
            region_name = _region_name_from_with_item(
                node.items[0], self._tla_module_aliases, self._recognition_scope()
            )
            if planned is True and region_name is None:
                raise RuntimeError(
                    "authoritative FunctionPlan marked a planned runtime with, "
                    "but region recognition no longer matches"
                )
            if region_name is not None:
                region_mode, region_thread_block_dim = _region_mode_from_with_item(
                    node.items[0],
                    self._tla_module_aliases,
                    self._recognition_scope(),
                )
                context_expr = node.items[0].context_expr
                if not isinstance(context_expr, ast.Call):
                    raise RuntimeError("recognized DSL region context is not a call")
                path = self._module_qualified_dsl_path(context_expr.func)
                if path is not None:
                    region_guard = self._checked_module_callable(
                        context_expr.func, path
                    )
                analysis = self._control_flow_analyzer.analyze(
                    node=node,
                    construct_name="with",
                    assigned_regions=[node.body],
                    active_call_nodes=node.body,
                    active_symbols=set(enclosing_symbols),
                    active_callables=set(self._active_callables()),
                    function_plan=(
                        self._function_plan_stack[-1]
                        if self._function_plan_stack
                        else None
                    ),
                    filename=self._filename,
                    line_offset=self._line_offset,
                    source_text=self._source_text,
                    is_runtime_for=lambda nested: _is_tla_range_iter(
                        nested.iter,
                        self._range_aliases(),
                        self._tla_range_names,
                        self._tla_module_aliases,
                        self._local_scope() | self._trusted_symbol_shadows(),
                    ),
                    is_static_test=self._is_static_control_flow_test,
                )
                self._ensure_runtime_origin_markers(analysis)
        for item in node.items:
            self._validate_reserved_module_call(item.context_expr)
            if isinstance(item.optional_vars, ast.Name):
                self._scope_manager.add_to_scope(item.optional_vars.id)
        node.items = [self.visit(item) for item in node.items]
        # A region body (e.g. ``with tla.vec.func():``) is hoisted into its own
        # nested function below, so process it like a function body: a fresh
        # scope with sequential statement-by-statement registration. This lets
        # loop-carried values defined inside the region (seeded before a
        # ``tla.range`` loop and reassigned in it) be detected as carried.
        with_target_names = set().union(
            *(
                _assigned_names(item.optional_vars)
                for item in node.items
                if item.optional_vars is not None
            )
        )
        self._range_alias_stack.append(set(self._range_aliases()) - with_target_names)
        try:
            with self._runtime_transform_scope():
                with self._scope_manager.enter_local_scope():
                    node.body = self._visit_statement_list(node.body) or [ast.Pass()]
        finally:
            self._range_alias_stack.pop()
        if len(node.items) != 1:
            return node
        if region_name is None:
            return node

        body_name = self._fresh(f"{region_name.replace('.', '_')}_body")
        # The region body shares the enclosing scope semantically. Any enclosing
        # variable reassigned inside the body -- most commonly the carried-value
        # round-trip the if/for lowering emits (``x = internal_if(..., x, ...)``)
        # for a mutex/flag that is method-invoked inside nested control flow --
        # would otherwise be treated as a local of this nested function, so the
        # read on the RHS raises UnboundLocalError. Declare those names nonlocal
        # so they keep referring to the enclosing binding.
        region_body: list[ast.stmt] = node.body or [ast.Pass()]
        nonlocal_names = sorted(
            enclosing_symbols & _assigned_names_from_statements(region_body)
        )
        if nonlocal_names:
            region_body = [ast.Nonlocal(names=nonlocal_names), *region_body]
        body_fn = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=region_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(body_fn, node)

        helper_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_INTERNAL_REGION, ctx=ast.Load()),
                args=[
                    ast.Constant(value=region_name),
                    ast.Name(id=body_name, ctx=ast.Load()),
                ],
                keywords=(
                    (
                        []
                        if region_mode is None
                        else [ast.keyword(arg="mode", value=region_mode)]
                    )
                    + (
                        []
                        if region_thread_block_dim is None
                        else [
                            ast.keyword(
                                arg="thread_block_dim", value=region_thread_block_dim
                            )
                        ]
                    )
                ),
            )
        )
        ast.copy_location(helper_call, node)
        return [
            *self._runtime_write_guards(analysis),
            *(
                [ast.copy_location(ast.Expr(value=region_guard), node)]
                if region_guard is not None
                else []
            ),
            body_fn,
            self._source_info_stmt(
                body_name,
                node.body[0] if node.body else node,
                construct=f"tla.{region_name}",
                region="region-body",
            ),
            helper_call,
        ]


@dataclass(frozen=True)
class _PreparedFunctionSource:
    """Source shared by validation and its subsequent, destructive rewrite."""

    source_text: str
    filename: str
    line_offset: int
    module_ast: ast.Module
    target: ast.FunctionDef | None
    statement_facts: _StatementFactsCache = dataclasses.field(
        default_factory=_StatementFactsCache, compare=False, repr=False
    )


def _prepare_function_source(fn: Callable[..., Any]) -> _PreparedFunctionSource | None:
    try:
        source_lines, first_lineno = inspect.getsourcelines(fn)
    except (OSError, IOError, TypeError):
        return None
    source_text = textwrap.dedent("".join(source_lines))
    filename = inspect.getsourcefile(fn) or "<unknown>"
    module_ast = ast.parse(source_text, filename=filename)
    return _PreparedFunctionSource(
        source_text,
        filename,
        int(first_lineno) - 1,
        module_ast,
        _find_function_def(module_ast, fn.__name__),
    )


def _preprocess_for_lowering(fn: FunctionType, **hooks: Any) -> FunctionType:
    prepared = _prepare_function_source(fn)
    validate_language_boundaries(fn, _prepared_source=prepared)
    return maybe_transform_for_lowering(fn, _prepared_source=prepared, **hooks)


def maybe_transform_for_lowering(
    fn: FunctionType,
    *,
    internal_for: Any,
    internal_region: Any,
    internal_if: Any,
    internal_if_expr: Any,
    internal_bool_and: Any,
    internal_bool_or: Any,
    internal_bool_not: Any,
    internal_compare: Any,
    internal_any: Any,
    internal_all: Any,
    internal_bool: Any,
    internal_min: Any,
    internal_max: Any,
    _prepared_source: _PreparedFunctionSource | None = None,
) -> FunctionType:
    """Return a transformed callable when source-driven control-flow lowering is needed."""

    prepared = _prepared_source or _prepare_function_source(fn)
    if prepared is None:
        return fn
    source = prepared.source_text
    filename = prepared.filename
    line_offset = prepared.line_offset
    module_ast = prepared.module_ast
    target = prepared.target
    if target is None:
        return fn
    has_scope_declaration = _has_scope_declaration(target)
    scope_facts = _scope_facts_for_transform(
        source, filename, target, has_scope_declaration=has_scope_declaration
    )
    root_freevars = set(fn.__code__.co_freevars)
    trusted_identities = _trusted_dsl_identities()
    shadow_plan = _FunctionAnalyzer(
        global_names={*fn.__globals__, *dir(builtins)},
        global_symbols=fn.__globals__,
        scope_facts=scope_facts,
        root_freevars=root_freevars,
        trusted_identities=trusted_identities,
        statement_facts=prepared.statement_facts,
    ).analyze(target)
    root_shadows = {binding.name for binding in shadow_plan.bindings}
    if not has_scope_declaration and not _function_needs_frontend_transform(
        target,
        fn.__globals__,
        root_shadows,
        trusted_identities,
        is_root=True,
        child_plans=shadow_plan.child_plans,
    ):
        return fn

    target.decorator_list = []
    exec_globals = dict(fn.__globals__)
    transformer = _FrontendControlFlowTransformer(
        exec_globals,
        filename=filename,
        line_offset=line_offset,
        source_text=source,
        trusted_identities=trusted_identities,
        root_plan=shadow_plan,
        statement_facts=prepared.statement_facts,
    )
    transformed = transformer.visit(module_ast)
    if root_freevars:
        transformed = _wrap_transformed_closure(transformed, fn.__name__, root_freevars)
    ast.fix_missing_locations(transformed)
    if line_offset:
        ast.increment_lineno(transformed, line_offset)

    exec_globals[_INTERNAL_FOR] = internal_for
    exec_globals[_INTERNAL_REGION] = internal_region
    exec_globals[_INTERNAL_IF] = internal_if
    exec_globals[_INTERNAL_IF_EXPR] = internal_if_expr
    exec_globals[_INTERNAL_BOOL_AND] = internal_bool_and
    exec_globals[_INTERNAL_BOOL_OR] = internal_bool_or
    exec_globals[_INTERNAL_BOOL_NOT] = internal_bool_not
    exec_globals[_INTERNAL_COMPARE] = internal_compare
    exec_globals[_INTERNAL_ANY] = internal_any
    exec_globals[_INTERNAL_ALL] = internal_all
    exec_globals[_INTERNAL_BOOL] = internal_bool
    exec_globals[_INTERNAL_MIN] = internal_min
    exec_globals[_INTERNAL_MAX] = internal_max
    exec_globals[_INTERNAL_CF_SYMBOL_CHECK] = _cf_symbol_check
    exec_globals[_INTERNAL_RUNTIME_BINDING_IS_COMPILETIME] = (
        _runtime_binding_is_compiletime
    )
    exec_globals[_INTERNAL_RUNTIME_WRITE_CHECK] = _runtime_write_check
    exec_globals[_INTERNAL_CHECKED_DSL_MEMBER] = _checked_dsl_member
    exec_globals[_INTERNAL_CHECKED_DSL_IDENTITY] = _checked_dsl_identity
    exec_globals[_INTERNAL_INDEX_ADD] = _index_add
    exec_globals[_INTERNAL_INDEX_SUB] = _index_sub
    exec_globals[transformer.tensor_store_helper_name] = _tensor_store
    exec_globals[transformer._source_snapshot_name] = transformer._source_snapshot
    exec_globals[_INTERNAL_ATTACH_SOURCE_INFO] = _attach_source_info
    from ..tla_ast_decorators import (
        _internal_lazy_attribute,
        _internal_lazy_binop,
        _internal_lazy_subscript,
        _internal_lazy_unary,
        _internal_unknown_effect_call,
    )

    exec_globals[_INTERNAL_UNKNOWN_EFFECT_CALL] = _internal_unknown_effect_call
    exec_globals[_INTERNAL_LAZY_ATTRIBUTE] = _internal_lazy_attribute
    exec_globals[_INTERNAL_LAZY_SUBSCRIPT] = _internal_lazy_subscript
    exec_globals[_INTERNAL_LAZY_BINOP] = _internal_lazy_binop
    exec_globals[_INTERNAL_LAZY_UNARY] = _internal_lazy_unary
    from .ast_helpers import (
        _warn_static_loop,
        range_constexpr,
        while_executor,
        while_selector,
    )

    exec_globals[_INTERNAL_STATIC_LOOP_ITERATION] = _warn_static_loop
    exec_globals[_INTERNAL_RANGE_CONSTEXPR] = range_constexpr

    exec_globals[_WHILE_EXECUTOR] = while_executor
    exec_globals[_WHILE_SELECTOR] = while_selector
    exec_globals["__tladsl_builtins__"] = builtins
    namespace: dict[str, Any] = {}
    code = compile(
        transformed,
        filename=filename,
        mode="exec",
    )
    exec(code, exec_globals, namespace)
    if root_freevars:
        closure_factory = namespace.get(_CLOSURE_FACTORY_NAME)
        if not isinstance(closure_factory, FunctionType):
            return fn
        template = closure_factory(*(object() for _ in root_freevars))
        if not isinstance(template, FunctionType):
            return fn
        original_cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        try:
            closure = tuple(
                original_cells[name] for name in template.__code__.co_freevars
            )
        except KeyError:
            return fn
        rewritten = FunctionType(
            template.__code__,
            exec_globals,
            name=fn.__name__,
            argdefs=fn.__defaults__,
            closure=closure,
        )
    else:
        rewritten = namespace.get(fn.__name__)
    if not isinstance(rewritten, FunctionType):
        return fn
    rewritten.__defaults__ = fn.__defaults__
    rewritten.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    rewritten.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    if hasattr(fn, "__type_params__"):
        rewritten.__type_params__ = fn.__type_params__
    rewritten.__tladsl_source_info__ = {
        "filename": filename,
        "generated_name": rewritten.__name__,
        "construct": "kernel body",
        "region": "frontend-execution",
    }
    rewritten.__module__ = fn.__module__
    rewritten.__qualname__ = fn.__qualname__
    return rewritten


def validate_language_boundaries(
    fn: Callable[..., Any],
    *,
    _prepared_source: _PreparedFunctionSource | None = None,
) -> None:
    """Validate calls and class use reachable from one decorated function."""

    prepared = _prepared_source or _prepare_function_source(fn)
    if prepared is None:
        raise SyntaxError("TLA Python function source must be available for validation")
    if prepared.target is None:
        raise SyntaxError("TLA Python function source could not be analyzed")
    _validate_language_boundaries(
        fn,
        target=prepared.target,
        source=prepared.source_text,
        filename=prepared.filename,
        line_offset=prepared.line_offset,
        statement_facts=prepared.statement_facts,
    )


def _validate_language_boundaries(
    fn: Callable[..., Any],
    *,
    target: ast.FunctionDef,
    source: str,
    filename: str,
    line_offset: int,
    validated: set[int] | None = None,
    validating: set[int] | None = None,
    statement_facts: _StatementFactsCache | None = None,
) -> None:
    """Validate the boundary between Python staging and DSL lowering.

    Ordinary inspectable Python code is valid while a kernel is being staged.
    The frontend owns only values that cross the kernel ABI, explicit runtime
    promotion, or a transformed runtime-control-flow region.
    """

    validated = validated if validated is not None else set()
    validating = validating if validating is not None else set()
    if id(fn) in validated:
        return
    validating.add(id(fn))
    symbols = dict(fn.__globals__)
    try:
        closure = inspect.getclosurevars(fn)
    except (TypeError, ValueError):
        closure = None
    if closure is not None:
        symbols.update(closure.globals)
        symbols.update(closure.nonlocals)
        symbols.update(closure.builtins)
    local_names = _function_arg_names(target.args) | _assigned_names_from_statements(
        target.body, statement_facts=statement_facts
    )
    aliases = {name: _UNKNOWN_LANGUAGE_VALUE for name in local_names}
    source_lines = source.splitlines()

    def fail(
        node: ast.AST,
        message: str,
        error_type: type[SyntaxError] = SyntaxError,
    ) -> NoReturn:
        error = error_type(message)
        relative_lineno = int(getattr(node, "lineno", 0) or 0)
        error.filename = filename
        error.lineno = line_offset + relative_lineno
        error.offset = int(getattr(node, "col_offset", 0) or 0) + 1
        if 0 < relative_lineno <= len(source_lines):
            error.text = source_lines[relative_lineno - 1]
        raise error

    def resolve(node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, symbols.get(node.id, _UNKNOWN_LANGUAGE_VALUE))
        literal_types = {
            ast.Dict: dict,
            ast.List: list,
            ast.Set: set,
            ast.Tuple: tuple,
        }
        for literal_node, literal_type in literal_types.items():
            if isinstance(node, literal_node):
                return literal_type
        if isinstance(node, ast.Subscript):
            container = resolve(node.value)
            if (
                isinstance(container, (tuple, list))
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                try:
                    return container[node.slice.value]
                except IndexError:
                    return _UNKNOWN_LANGUAGE_VALUE
            if _is_exact_builtin_dict(container) and isinstance(
                node.slice, ast.Constant
            ):
                try:
                    return container[node.slice.value]
                except (KeyError, TypeError):
                    return _UNKNOWN_LANGUAGE_VALUE
            return _UNKNOWN_LANGUAGE_VALUE
        if not isinstance(node, ast.Attribute):
            return _UNKNOWN_LANGUAGE_VALUE
        owner = resolve(node.value)
        if owner is _UNKNOWN_LANGUAGE_VALUE:
            return owner
        try:
            return inspect.getattr_static(owner, node.attr)
        except (AttributeError, TypeError):
            members = inspect.getattr_static(owner, "_members", None)
            if isinstance(members, dict) and node.attr in members:
                return members[node.attr]
            return _UNKNOWN_LANGUAGE_VALUE

    def is_jit(value: Any) -> bool:
        from ..dsl import is_jit_callable

        return is_jit_callable(value)

    def validate_jit(node: ast.AST, helper: Callable[..., Any]) -> None:
        if id(helper) in validating:
            fail(
                node,
                "recursive @tla.jit helper calls are not supported",
                _RecursiveJitHelperError,
            )
        if id(helper) in validated:
            return
        try:
            helper_lines, helper_lineno = inspect.getsourcelines(helper)
        except (OSError, IOError, TypeError):
            fail(node, "@tla.jit helper source must be available for validation")
        helper_source = textwrap.dedent("".join(helper_lines))
        helper_filename = inspect.getsourcefile(helper) or "<unknown>"
        helper_target = _find_function_def(
            ast.parse(helper_source, filename=helper_filename), helper.__name__
        )
        if helper_target is None:
            fail(node, "@tla.jit helper source could not be analyzed")
        _validate_language_boundaries(
            helper,
            target=helper_target,
            source=helper_source,
            filename=helper_filename,
            line_offset=int(helper_lineno) - 1,
            validated=validated,
            validating=validating,
        )

    def is_supported_isinstance_type(node: ast.AST) -> bool:
        if isinstance(node, ast.Tuple):
            return bool(node.elts) and all(
                is_supported_isinstance_type(element) for element in node.elts
            )
        value = resolve(node)
        # ``isinstance`` remains Python staging when its type operand is a
        # statically resolved class. User classes are valid staging objects;
        # they are still rejected if they cross the kernel ABI or are promoted.
        return inspect.isclass(value)

    def validate_callable(node: ast.Call, value: Any) -> None:
        if value is _UNKNOWN_LANGUAGE_VALUE:
            # Local lambdas and nested functions deliberately resolve here:
            # they are ordinary Python staging code.
            return
        if is_jit(value):
            from ..dsl import unwrap_jit_callable

            validate_jit(node.func, unwrap_jit_callable(value))
            return
        from ..dsl import TlaJitFunction

        if isinstance(value, TlaJitFunction):
            name = getattr(value.fn, "__name__", "<kernel>")
            fail(
                node.func,
                f"calling @tla.kernel {name!r} from a TLA Python function is not "
                "supported; use @tla.jit",
            )
        if inspect.isclass(value):
            return
        owner = getattr(value, "__self__", None)
        if owner is not None and not inspect.ismodule(owner):
            return
        module = str(getattr(value, "__module__", ""))
        if module.startswith("catlass"):
            return
        if module == "builtins":
            if value is isinstance:
                if len(node.args) != 2 or node.keywords:
                    fail(
                        node.func,
                        "isinstance requires exactly a value and trusted type operand",
                    )
                if not is_supported_isinstance_type(node.args[1]):
                    fail(
                        node.args[1],
                        "isinstance only supports built-in or TLA Python types in "
                        "TLA Python functions",
                    )
                return
            supported_builtins = {
                abs,
                all,
                any,
                bool,
                enumerate,
                float,
                int,
                len,
                list,
                max,
                min,
                range,
                reversed,
                sorted,
                str,
                sum,
                tuple,
                zip,
            }
            if value in supported_builtins:
                return
            name = getattr(value, "__name__", type(value).__name__)
            fail(
                node.func, f"builtin {name!r} is not supported in TLA Python functions"
            )
        if inspect.isfunction(value):
            # Bare Python functions execute directly during staging. They share
            # the caller's active frontend emission context, so TLA operations
            # in their body emit IR without helper transformation or source
            # inspection.
            return
        if callable(value):
            try:
                callable_lines, callable_lineno = inspect.getsourcelines(value)
            except (OSError, IOError, TypeError):
                name = getattr(value, "__name__", type(value).__name__)
                fail(
                    node.func,
                    f"call to Python callable {name!r} requires @tla.jit with "
                    "inspectable source when it is lowered by TLA Python",
                )
            callable_source = textwrap.dedent("".join(callable_lines))
            callable_filename = inspect.getsourcefile(value) or filename
            callable_target = _find_function_def(
                ast.parse(callable_source, filename=callable_filename),
                getattr(value, "__name__", ""),
            )
            if callable_target is not None and _function_needs_frontend_transform(
                callable_target, getattr(value, "__globals__", {})
            ):
                fail(
                    node.func,
                    "Python helper control-flow composition is not supported yet; "
                    "inline the helper body or move the call outside runtime control flow",
                )

    class Validator(ast.NodeVisitor):
        def __init__(self) -> None:
            self._runtime_constructs: list[str] = []

        @contextlib.contextmanager
        def _runtime_construct(self, name: str) -> Iterator[None]:
            self._runtime_constructs.append(name)
            try:
                yield
            finally:
                self._runtime_constructs.pop()

        def _visit_runtime_body(self, name: str, body: list[ast.stmt]) -> None:
            with self._runtime_construct(name):
                for statement in body:
                    self.visit(statement)

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            resolved = resolve(node.value)
            for assignment_target in node.targets:
                if isinstance(assignment_target, ast.Name):
                    aliases[assignment_target.id] = resolved
                else:
                    self.visit(assignment_target)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self.visit(node.value)
                if isinstance(node.target, ast.Name):
                    aliases[node.target.id] = resolve(node.value)

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            if isinstance(node.test, ast.Constant) and isinstance(
                node.test.value, bool
            ):
                for statement in node.body if node.test.value else node.orelse:
                    self.visit(statement)
                return
            before = dict(aliases)
            try:
                for body in (node.body, node.orelse):
                    aliases.clear()
                    aliases.update(before)
                    self._visit_runtime_body("if", body)
            finally:
                aliases.clear()
                aliases.update(before)

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            if _is_syntactic_tla_range_for(node):
                self._visit_runtime_body("for", node.body)
                self._visit_runtime_body("for", node.orelse)
                return
            for statement in [*node.body, *node.orelse]:
                self.visit(statement)

        def visit_While(self, node: ast.While) -> None:
            self.visit(node.test)
            self._visit_runtime_body("while", node.body)
            self._visit_runtime_body("while", node.orelse)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                self.visit(item.context_expr)
            self._visit_runtime_body("with", node.body)

        def visit_Call(self, node: ast.Call) -> None:
            value = resolve(node.func)
            validate_callable(node, value)
            if self._runtime_constructs and inspect.isclass(value):
                if not _is_tla_owned_type(value) and not dataclasses.is_dataclass(
                    value
                ):
                    fail(
                        node.func,
                        f"user-defined class {value.__name__!r} cannot be "
                        "constructed in dynamic Tla control flow",
                    )
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            self.visit(node.value)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Definitions execute during Python staging. Do not recurse into a
            # class body: it is not part of the kernel's lowered body.
            del node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Nested functions are staging-local Python callables.
            del node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self.visit(node.body)

    try:
        validator = Validator()
        for statement in target.body:
            validator.visit(statement)
    finally:
        validating.remove(id(fn))
    validated.add(id(fn))


_CLOSURE_FACTORY_NAME = "__tladsl_internal_closure_factory__"


def _wrap_transformed_closure(
    module: ast.Module, function_name: str, freevars: set[str]
) -> ast.Module:
    """Compile a transformed closure without reading or copying its cells."""

    target = _find_function_def(module, function_name)
    if target is None:
        return module

    # Definition-time expressions have already run for the original callable.
    # Remove them from the private template so rebuilding cannot repeat effects.
    target.args.defaults = []
    target.args.kw_defaults = [None] * len(target.args.kwonlyargs)
    for argument in (
        *target.args.posonlyargs,
        *target.args.args,
        *target.args.kwonlyargs,
    ):
        argument.annotation = None
    if target.args.vararg is not None:
        target.args.vararg.annotation = None
    if target.args.kwarg is not None:
        target.args.kwarg.annotation = None
    target.returns = None
    if hasattr(target, "type_params"):
        target.type_params = []

    ordered_freevars = sorted(freevars)
    factory = ast.FunctionDef(
        name=_CLOSURE_FACTORY_NAME,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in ordered_freevars],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            target,
            ast.Return(value=ast.Name(id=function_name, ctx=ast.Load())),
        ],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    if hasattr(factory, "type_params"):
        factory.type_params = []
    ast.copy_location(factory, target)
    return ast.Module(body=[factory], type_ignores=module.type_ignores)


def _find_function_def(module_ast: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _trusted_lazy_callables() -> tuple[Callable[..., Any], ...]:
    """Return exact frontend call targets allowed to execute in lazy IR regions."""

    return _TRUSTED_LAZY_CALLABLES


def _function_needs_frontend_transform(
    target: ast.FunctionDef,
    global_symbols: dict[str, Any],
    lexical_shadow_names: set[str] | None = None,
    trusted_identities: _TrustedDslIdentities | None = None,
    *,
    is_root: bool = False,
    child_plans: tuple[FunctionBlockPlan, ...] | None = None,
) -> bool:
    """Return whether *target* contains syntax handled by the frontend rewrite."""

    # An authoritative nonempty plan already establishes the need to rewrite.
    # An empty plan still needs expression, static-loop and diagnostic discovery.
    if child_plans:
        return True
    identities = trusted_identities or _trusted_dsl_identities()
    range_names = _tla_function_names_from_globals(global_symbols, "range", identities)
    range_constexpr_names = _tla_function_names_from_globals(
        global_symbols, "range_constexpr", identities
    )
    module_aliases = _tla_module_aliases_from_globals(global_symbols, identities)
    lexical_shadows = set(lexical_shadow_names or ())

    # Keep helper discovery aligned with the root frontend plan: an ``if`` or
    # ``while`` only needs PR-5 composition when it is dynamic control flow.
    # Static Python branches remain ordinary staging-time Python.
    if child_plans is None:
        child_plans = _function_child_plans(
            target,
            lexical_shadow_names=lexical_shadows,
            trusted_symbols=_trusted_dsl_symbols(global_symbols, identities),
            global_symbols=global_symbols,
        )
    if child_plans:
        return True

    class Discovery(ast.NodeVisitor):
        needed = False

        def visit_If(self, node: ast.If) -> None:
            if _is_constexpr_cf_test(
                node.test,
                _tla_const_expr_names_from_globals(global_symbols, identities),
                module_aliases,
                lexical_shadows,
            ):
                self.needed = True
                return
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            # Root kernels must route even static ``while`` through the
            # frontend so its static-loop warning instrumentation is installed.
            # Staging helpers keep ordinary static Python ``while`` semantics
            # until PR 5 transforms their bodies.
            if is_root:
                self.needed = True
                return
            self.generic_visit(node)

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.needed = True

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Nested classes need the frontend's source-location diagnostic.
            del node
            self.needed = True

        def visit_Global(self, node: ast.Global) -> None:
            self.needed = True

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            self.needed = True

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id not in lexical_shadows
                and func.id
                in {
                    "range",
                    "range_constexpr",
                    *_BUILTIN_REDIRECTS,
                    *range_names,
                    *range_constexpr_names,
                }
            ):
                self.needed = True
                return
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in module_aliases
                and func.value.id not in lexical_shadows
                and func.attr in {"range", "range_constexpr", "cube", "vector"}
            ):
                self.needed = True
                return
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "func"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "vec"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in module_aliases
                and func.value.value.id not in lexical_shadows
            ):
                self.needed = True
                return
            self.generic_visit(node)

    discovery = Discovery()
    for statement in target.body:
        discovery.visit(statement)
        if discovery.needed:
            break
    return discovery.needed


def _tla_function_names_from_globals(
    global_symbols: dict[str, Any],
    function_name: str,
    identities: _TrustedDslIdentities,
) -> set[str]:
    target = (
        identities.range_callable
        if function_name == "range"
        else identities.range_constexpr_callable
    )
    return {
        name
        for name, value in global_symbols.items()
        if target is not None and value is target
    }


def _tla_const_expr_names_from_globals(
    global_symbols: dict[str, Any], identities: _TrustedDslIdentities
) -> set[str]:
    return {
        name
        for name, value in global_symbols.items()
        if identities.const_expr_callable is not None
        and value is identities.const_expr_callable
    }


def _tla_as_numeric_names_from_globals(
    global_symbols: dict[str, Any], identities: _TrustedDslIdentities
) -> set[str]:
    return {
        name
        for name, value in global_symbols.items()
        if identities.as_numeric_callable is not None
        and value is identities.as_numeric_callable
    }


def _tla_module_aliases_from_globals(
    global_symbols: dict[str, Any], identities: _TrustedDslIdentities
) -> set[str]:
    return {
        name
        for name, value in global_symbols.items()
        if identities.module is not None and value is identities.module
    }


def _is_tla_range_call(
    node: ast.AST,
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if _is_checked_dsl_call(func, "range"):
        return True
    if (
        isinstance(func, ast.Name)
        and func.id in tla_range_names
        and func.id not in local_names
    ):
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (
            func.value.id in tla_module_aliases
            and func.value.id not in local_names
            and func.attr == "range"
        )
    return False


def _is_tla_range_constexpr_call(
    node: ast.AST,
    tla_range_constexpr_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if _is_checked_dsl_call(func, "range_constexpr"):
        return True
    if (
        isinstance(func, ast.Name)
        and func.id in tla_range_constexpr_names
        and func.id not in local_names
    ):
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (
            func.value.id in tla_module_aliases
            and func.value.id not in local_names
            and func.attr == "range_constexpr"
        )
    return False


def _is_checked_dsl_call(node: ast.AST, path: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {_INTERNAL_CHECKED_DSL_MEMBER, _INTERNAL_CHECKED_DSL_IDENTITY}
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value
        in ({path, "builtin_range"} if path == "range" else {path})
        and not node.keywords
    )


def _cf_symbol_check_stmts(range_call: ast.Call) -> list[ast.stmt]:
    func = range_call.func
    if (
        isinstance(func, ast.Call)
        and isinstance(func.func, ast.Name)
        and func.func.id
        in {_INTERNAL_CHECKED_DSL_MEMBER, _INTERNAL_CHECKED_DSL_IDENTITY}
    ):
        return []
    if not isinstance(func, (ast.Attribute, ast.Name)):
        raise SyntaxError("dynamic Tla for requires a Tla range symbol")
    expressions: list[ast.expr] = []
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        expressions.append(func.value)
    expressions.append(func)
    return [_cf_symbol_check_expression_stmt(expression) for expression in expressions]


def _cf_symbol_check_expression_stmt(expression: ast.expr) -> ast.stmt:
    check_stmt = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=_INTERNAL_CF_SYMBOL_CHECK, ctx=ast.Load()),
            args=[expression],
            keywords=[],
        )
    )
    return ast.copy_location(check_stmt, expression)


def _static_range_call_from_range_constexpr(node: ast.Call) -> ast.Call:
    rewritten = ast.Call(
        func=ast.Name(id=_INTERNAL_RANGE_CONSTEXPR, ctx=ast.Load()),
        args=node.args,
        keywords=node.keywords,
    )
    return ast.copy_location(rewritten, node)


class _RangeCallBounds:
    def __init__(
        self,
        *,
        start: ast.expr,
        end: ast.expr,
        step: ast.expr,
        has_explicit_step: bool,
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.has_explicit_step = has_explicit_step


def _extract_range_call_bounds(node: ast.Call) -> _RangeCallBounds | None:
    if len(node.args) > 3:
        return None

    start: ast.expr
    end: ast.expr
    step: ast.expr
    has_explicit_step = False
    if len(node.args) == 1:
        start = ast.Constant(value=0)
        end = node.args[0]
        step = ast.Constant(value=1)
    elif len(node.args) == 2:
        start = node.args[0]
        end = node.args[1]
        step = ast.Constant(value=1)
    elif len(node.args) == 3:
        start = node.args[0]
        end = node.args[1]
        step = node.args[2]
        has_explicit_step = True
    else:
        return None

    return _RangeCallBounds(
        start=start,
        end=end,
        step=step,
        has_explicit_step=has_explicit_step,
    )


def _assign_name(name: str, value: ast.expr, location: ast.AST) -> ast.Assign:
    return ast.copy_location(
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value),
        location,
    )


def _if_expr(test_name: str, body: ast.expr, orelse: ast.expr) -> ast.IfExp:
    return ast.IfExp(
        test=ast.Name(id=test_name, ctx=ast.Load()),
        body=body,
        orelse=orelse,
    )


def _index_add_call(lhs: ast.expr, rhs: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=_INTERNAL_INDEX_ADD, ctx=ast.Load()),
        args=[lhs, rhs],
        keywords=[],
    )


def _index_sub_call(lhs: ast.expr, rhs: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=_INTERNAL_INDEX_SUB, ctx=ast.Load()),
        args=[lhs, rhs],
        keywords=[],
    )


def _index_add(lhs: Any, rhs: Any) -> Any:
    from catlass.base_dsl.typing import as_numeric

    return as_numeric(lhs) + as_numeric(rhs)


def _index_sub(lhs: Any, rhs: Any) -> Any:
    from catlass.base_dsl.typing import as_numeric

    return as_numeric(lhs) - as_numeric(rhs)


def _tensor_store(value: Any, target: Any, index: Any) -> None:
    """Guard a rewritten subscription assignment before invoking tensor storage."""

    from catlass.core_api import _require_category

    _require_category("tensor_store", "target", target, "tensor", 0)
    target[index] = value


def _contains_slice(node: ast.AST) -> bool:
    return any(isinstance(candidate, ast.Slice) for candidate in ast.walk(node))


def _attach_source_info(fn: Any, info: dict[str, Any]) -> Any:
    setattr(fn, _SOURCE_INFO_ATTR, info)
    return fn


def _cf_symbol_check(symbol: Any) -> None:
    identities = _trusted_dsl_identities()
    if any(
        symbol is trusted
        for trusted in (
            identities.module,
            identities.range_callable,
            identities.range_constexpr_callable,
            identities.const_expr_callable,
            identities.cube_callable,
            identities.vector_callable,
            identities.vec_namespace,
            identities.vec_func_callable,
        )
        if trusted is not None
    ):
        return
    name = getattr(symbol, "__name__", type(symbol).__name__)
    raise RuntimeError(f"Incorrect `{name}` is used. Please use the TLA Python symbol.")


def _checked_dsl_member(module: Any, path: str) -> Any:
    """Resolve one frozen DSL member path exactly once at its Python evaluation point."""

    identities = _trusted_dsl_identities()
    expected_paths = {
        "range": (identities.range_callable,),
        "range_constexpr": (identities.range_constexpr_callable,),
        "const_expr": (identities.const_expr_callable,),
        "as_numeric": (identities.as_numeric_callable,),
        "cube": (identities.cube_callable,),
        "vector": (identities.vector_callable,),
        "vec.func": (identities.vec_namespace, identities.vec_func_callable),
    }
    expected = expected_paths.get(path)
    if module is not identities.module or expected is None:
        _cf_symbol_check(module)
        raise RuntimeError("Unknown reserved TLA Python member path")
    current = module
    for member, trusted in zip(path.split("."), expected):
        try:
            current = getattr(current, member)
        except AttributeError:
            current = None
        if trusted is None or current is not trusted:
            _raise_incorrect_dsl_symbol(current)
    return current


def _checked_dsl_identity(symbol: Any, path: str) -> Any:
    """Require the exact frozen identity assigned to a direct DSL-call role."""

    identities = _trusted_dsl_identities()
    expected = {
        "range": identities.range_callable,
        "builtin_range": builtins.range,
        "range_constexpr": identities.range_constexpr_callable,
        "const_expr": identities.const_expr_callable,
    }.get(path)
    if expected is None or symbol is not expected:
        _raise_incorrect_dsl_symbol(symbol)
    if path == "builtin_range":
        if identities.range_callable is None:
            _raise_incorrect_dsl_symbol(symbol)
        return identities.range_callable
    return symbol


def _raise_incorrect_dsl_symbol(symbol: Any) -> NoReturn:
    name = getattr(symbol, "__name__", type(symbol).__name__)
    raise RuntimeError(f"Incorrect `{name}` is used. Please use the TLA Python symbol.")


def _is_tla_range_iter(
    node: ast.AST,
    aliases: set[str],
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if _is_tla_range_call(node, tla_range_names, tla_module_aliases, local_names):
        return True
    return isinstance(node, ast.Name) and node.id in aliases


def _update_range_aliases(
    stmt: ast.stmt,
    aliases: set[str],
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> None:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return
    if _is_tla_range_call(stmt.value, tla_range_names, tla_module_aliases, local_names):
        aliases.add(target.id)
    else:
        aliases.discard(target.id)


def _region_name_from_with_item(
    item: ast.withitem,
    tla_module_aliases: set[str],
    local_names: set[str],
) -> str | None:
    context_expr = item.context_expr
    if not isinstance(context_expr, ast.Call):
        return None
    func = context_expr.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if (
            func.value.id in tla_module_aliases
            and func.value.id not in local_names
            and func.attr in {"cube", "vector"}
        ):
            return func.attr
    if _is_tla_vec_func(func, tla_module_aliases, local_names):
        return "vec.func"
    return None


def _is_tla_vec_func(
    func: ast.expr,
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "func"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "vec"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id in tla_module_aliases
        and func.value.value.id not in local_names
    )


def _raise_vec_func_error(message: str) -> None:
    from catlass.runtime import TlaCoreAPIError

    raise TlaCoreAPIError(f"tla.vec.func: {message}")


def _region_mode_from_with_item(
    item: ast.withitem,
    tla_module_aliases: set[str],
    local_names: set[str],
) -> tuple[ast.expr | None, ast.expr | None]:
    context_expr = item.context_expr
    if not isinstance(context_expr, ast.Call):
        return None, None
    if not _is_tla_vec_func(context_expr.func, tla_module_aliases, local_names):
        return None, None
    if context_expr.args:
        _raise_vec_func_error("mode must be passed by keyword")
    mode_expr: ast.expr | None = None
    thread_block_dim_expr: ast.expr | None = None
    for keyword in context_expr.keywords:
        if keyword.arg == "mode":
            if mode_expr is not None:
                _raise_vec_func_error("mode was passed multiple times")
            mode_expr = keyword.value
        elif keyword.arg == "thread_block_dim":
            if thread_block_dim_expr is not None:
                _raise_vec_func_error("thread_block_dim was passed multiple times")
            thread_block_dim_expr = keyword.value
        else:
            _raise_vec_func_error(f"unknown keyword argument: {keyword.arg}")
    if mode_expr is None:
        mode_expr = ast.Constant(value="simd")
    return mode_expr, thread_block_dim_expr


def _is_constexpr_cf_test(
    node: ast.AST,
    tla_const_expr_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (
            func.value.id in tla_module_aliases
            and func.value.id not in local_names
            and func.attr == "const_expr"
        )
    if isinstance(func, ast.Name):
        return func.id in tla_const_expr_names and func.id not in local_names
    return False


def _is_static_python_if_test(
    node: ast.AST,
    tla_const_expr_names: set[str] | None = None,
    tla_module_aliases: set[str] | None = None,
    local_names: set[str] | None = None,
    global_symbols: dict[str, Any] | None = None,
) -> bool:
    locals_in_scope = local_names if local_names is not None else set()
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, bool)
        or isinstance(node, ast.Name)
        and node.id not in locals_in_scope
        and isinstance((global_symbols or {}).get(node.id), bool)
        or _is_constexpr_cf_test(
            node,
            (
                tla_const_expr_names
                if tla_const_expr_names is not None
                else {"const_expr"}
            ),
            tla_module_aliases if tla_module_aliases is not None else {"tla"},
            locals_in_scope,
        )
    )


def _compare_op_name(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Is):
        return "is"
    if isinstance(op, ast.IsNot):
        return "is not"
    if isinstance(op, ast.In):
        return "in"
    if isinstance(op, ast.NotIn):
        return "not in"
    raise SyntaxError(f"unsupported comparison operator: {type(op).__name__}")


def _identifier_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, ast.alias):
            names.add(item.asname or item.name.split(".", 1)[0])
        elif isinstance(item, ast.ExceptHandler) and item.name is not None:
            names.add(item.name)
        elif isinstance(item, (ast.Global, ast.Nonlocal)):
            names.update(item.names)
        elif isinstance(item, (ast.MatchAs, ast.MatchStar)) and item.name is not None:
            names.add(item.name)
        elif isinstance(item, ast.MatchMapping) and item.rest is not None:
            names.add(item.rest)
    return names


def _function_arg_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs}
    names.update(arg.arg for arg in args.args)
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _ordered_function_args(args: ast.arguments) -> tuple[ast.arg, ...]:
    ordered = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        ordered.append(args.vararg)
    if args.kwarg is not None:
        ordered.append(args.kwarg)
    return tuple(ordered)


def _ordered_function_loads(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[str, ast.Name], ...]:
    loads: list[tuple[str, ast.Name]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.comprehension_bindings: list[set[str]] = []

        def visit_Name(self, name_node: ast.Name) -> None:
            if isinstance(name_node.ctx, ast.Load) and not any(
                name_node.id in bindings
                for bindings in reversed(self.comprehension_bindings)
            ):
                loads.append((name_node.id, name_node))

        def visit_ListComp(self, comp_node: ast.ListComp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def visit_SetComp(self, comp_node: ast.SetComp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def visit_DictComp(self, comp_node: ast.DictComp) -> None:
            self._visit_comprehension(comp_node, comp_node.key, comp_node.value)

        def visit_GeneratorExp(self, comp_node: ast.GeneratorExp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def _visit_comprehension(
            self,
            comp_node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
            *values: ast.expr,
        ) -> None:
            bindings: set[str] = set()
            self.comprehension_bindings.append(bindings)
            try:
                # Generator iterables are evaluated in order: each iterable can
                # see preceding targets, but not its own target.
                for generator in comp_node.generators:
                    self.visit(generator.iter)
                    bindings.update(_assigned_names(generator.target))
                    for condition in generator.ifs:
                        self.visit(condition)
                for value in values:
                    self.visit(value)
            finally:
                self.comprehension_bindings.pop()

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

    visitor = Visitor()
    for statement in node.body:
        visitor.visit(statement)
    return tuple(loads)


def _function_child_plans(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    lexical_shadow_names: set[str] | None = None,
    trusted_symbols: _TrustedDslSymbols,
    global_symbols: dict[str, Any] | None = None,
    statement_facts: _StatementFactsCache | None = None,
) -> tuple[FunctionBlockPlan, ...]:
    statement_facts = statement_facts or _StatementFactsCache()
    plans: list[FunctionBlockPlan] = []
    lexical_shadows = set(lexical_shadow_names or ())
    tla_range_names = set(trusted_symbols.range_names)
    tla_range_names.update(trusted_symbols.builtin_range_names)
    tla_const_expr_names = set(trusted_symbols.const_expr_names)
    tla_module_aliases = set(trusted_symbols.module_aliases)

    class Planner:
        @staticmethod
        def _record(construct_name: str, block: ast.AST) -> None:
            plans.append(
                FunctionBlockPlan(
                    construct_name=construct_name,
                    lineno=int(getattr(block, "lineno", 0) or 0),
                    col_offset=int(getattr(block, "col_offset", 0) or 0),
                )
            )

        def visit_statements(
            self,
            body: list[ast.stmt],
            active_names: set[str],
            range_aliases: set[str],
        ) -> None:
            for statement in body:
                self.visit_statement(statement, active_names, range_aliases)
                _update_range_aliases(
                    statement,
                    range_aliases,
                    tla_range_names,
                    tla_module_aliases,
                    active_names | lexical_shadows,
                )
                active_names.update(statement_facts.get(statement).assigned_names)

        def visit_statement(
            self,
            statement: ast.stmt,
            active_names: set[str],
            range_aliases: set[str],
        ) -> None:
            if isinstance(statement, ast.If):
                if not _is_static_python_if_test(
                    statement.test,
                    tla_const_expr_names,
                    tla_module_aliases,
                    active_names | lexical_shadows,
                    global_symbols,
                ):
                    self._record("if", statement)
                self._visit_regions(
                    (statement.body, statement.orelse), active_names, range_aliases
                )
                return
            if isinstance(statement, ast.For):
                is_runtime_for = _is_tla_range_iter(
                    statement.iter,
                    range_aliases,
                    tla_range_names,
                    tla_module_aliases,
                    active_names | lexical_shadows,
                )
                if is_runtime_for:
                    self._record("for", statement)
                target_names = statement_facts.get(statement.target).assigned_names
                loop_names = active_names | target_names
                self._visit_regions(
                    (statement.body, statement.orelse),
                    loop_names,
                    range_aliases - target_names if is_runtime_for else set(),
                )
                return
            if isinstance(statement, ast.While):
                if not _is_static_python_if_test(
                    statement.test,
                    tla_const_expr_names,
                    tla_module_aliases,
                    active_names | lexical_shadows,
                    global_symbols,
                ):
                    self._record("while", statement)
                self._visit_regions(
                    (statement.body, statement.orelse), active_names, range_aliases
                )
                return
            if isinstance(statement, ast.With):
                if (
                    len(statement.items) == 1
                    and _region_name_from_with_item(
                        statement.items[0],
                        tla_module_aliases,
                        active_names | lexical_shadows,
                    )
                    is not None
                ):
                    self._record("with", statement)
                with_names = set(active_names)
                for item in statement.items:
                    if item.optional_vars is not None:
                        with_names.update(
                            statement_facts.get(item.optional_vars).assigned_names
                        )
                self._visit_regions(
                    (statement.body,),
                    with_names,
                    range_aliases - (with_names - active_names),
                )
                return
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return
            try_nodes = (ast.Try,)
            try_star = getattr(ast, "TryStar", None)
            if try_star is not None:
                try_nodes += (try_star,)
            if isinstance(statement, try_nodes):
                regions = [statement.body]
                regions.extend(handler.body for handler in statement.handlers)
                regions.extend((statement.orelse, statement.finalbody))
                self._visit_regions(regions, active_names, range_aliases)
                return
            if isinstance(statement, ast.Match):
                self._visit_regions(
                    [case.body for case in statement.cases],
                    active_names,
                    range_aliases,
                )

        def _visit_regions(
            self,
            regions: Iterable[list[ast.stmt]],
            active_names: set[str],
            range_aliases: set[str],
        ) -> None:
            for region in regions:
                self.visit_statements(region, set(active_names), set(range_aliases))

    Planner().visit_statements(
        node.body,
        _function_arg_names(node.args),
        set(),
    )
    return tuple(plans)


def _align_function_child_plans(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    child_plans: tuple[FunctionBlockPlan, ...],
) -> dict[int, bool]:
    """Match planned dynamic blocks to the untouched source AST exactly once."""

    candidates: dict[tuple[str, int, int], list[ast.AST]] = {}

    class Collector(ast.NodeVisitor):
        def _record(self, construct_name: str, block: ast.AST) -> None:
            key = (
                construct_name,
                int(getattr(block, "lineno", 0) or 0),
                int(getattr(block, "col_offset", 0) or 0),
            )
            candidates.setdefault(key, []).append(block)
            self.generic_visit(block)

        def visit_If(self, block: ast.If) -> None:
            self._record("if", block)

        def visit_For(self, block: ast.For) -> None:
            self._record("for", block)

        def visit_While(self, block: ast.While) -> None:
            self._record("while", block)

        def visit_With(self, block: ast.With) -> None:
            self._record("with", block)

        def visit_FunctionDef(self, nested: ast.FunctionDef) -> None:
            del nested

        def visit_AsyncFunctionDef(self, nested: ast.AsyncFunctionDef) -> None:
            del nested

        def visit_ClassDef(self, nested: ast.ClassDef) -> None:
            del nested

        def visit_Lambda(self, nested: ast.Lambda) -> None:
            del nested

    collector = Collector()
    for statement in node.body:
        collector.visit(statement)

    membership = {
        id(candidate): False
        for matching in candidates.values()
        for candidate in matching
    }
    seen: set[tuple[str, int, int]] = set()
    for plan in child_plans:
        key = (plan.construct_name, plan.lineno, plan.col_offset)
        matching = candidates.get(key, ())
        if key in seen or len(matching) != 1:
            raise RuntimeError(
                "authoritative FunctionPlan does not uniquely match source block "
                f"{plan.construct_name} at {plan.lineno}:{plan.col_offset}"
            )
        membership[id(matching[0])] = True
        seen.add(key)
    return membership


def _assigned_names_from_statements(
    body: list[ast.stmt], *, statement_facts: _StatementFactsCache | None = None
) -> set[str]:
    assigned: set[str] = set()
    for stmt in body:
        assigned.update(
            _assigned_names(stmt)
            if statement_facts is None
            else statement_facts.get(stmt).assigned_names
        )
    return assigned


def _assigned_names(node: ast.AST) -> set[str]:
    assigned: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, name_node: ast.Name) -> None:
            if isinstance(name_node.ctx, ast.Store):
                assigned.add(name_node.id)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            assigned.add(function_node.name)

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            assigned.add(function_node.name)

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            assigned.add(class_node.name)

        def visit_Import(self, import_node: ast.Import) -> None:
            for alias in import_node.names:
                assigned.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, import_node: ast.ImportFrom) -> None:
            for alias in import_node.names:
                if alias.name != "*":
                    assigned.add(alias.asname or alias.name)

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def visit_ListComp(self, comp_node: ast.ListComp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def visit_SetComp(self, comp_node: ast.SetComp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def visit_DictComp(self, comp_node: ast.DictComp) -> None:
            self._visit_comprehension(comp_node, comp_node.key, comp_node.value)

        def visit_GeneratorExp(self, comp_node: ast.GeneratorExp) -> None:
            self._visit_comprehension(comp_node, comp_node.elt)

        def _visit_comprehension(
            self,
            comp_node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
            *values: ast.expr,
        ) -> None:
            for generator in comp_node.generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)

    Visitor().visit(node)
    return assigned


def _callable_names(node: ast.AST, active_callables: set[str]) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Name)
        and node.value.id in active_callables
    ):
        return {node.targets[0].id}
    return set()


def _has_nonlocal_or_global_declaration(node: ast.FunctionDef) -> bool:
    class Visitor(ast.NodeVisitor):
        found = False

        def visit_Global(self, declaration: ast.Global) -> None:
            del declaration
            self.found = True

        def visit_Nonlocal(self, declaration: ast.Nonlocal) -> None:
            del declaration
            self.found = True

        def visit_FunctionDef(self, nested: ast.FunctionDef) -> None:
            if nested is node:
                self.generic_visit(nested)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    visitor = Visitor()
    visitor.visit(node)
    return visitor.found


class _DynamicControlFlowPolicy(ast.NodeVisitor):
    """Shared statement and assignment policy for runtime control flow."""

    def __init__(
        self,
        construct_name: str,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
        is_runtime_for: Callable[[ast.For], bool] | None = None,
        is_static_test: Callable[[ast.AST], bool] | None = None,
    ) -> None:
        self.construct_name = construct_name
        self.filename = filename
        self.line_offset = line_offset
        self.source_lines = source_text.splitlines()
        self.is_runtime_for = is_runtime_for or _is_syntactic_tla_range_for
        self.is_static_test = is_static_test or _is_static_python_if_test
        self.python_loop_depth = 0
        self.assignment_targets: list[str] = []
        self.tensor_store_assignments: set[ast.Assign] = set()
        self.nested_constructs: list[tuple[str, int, int]] = []

    def visit_Return(self, node: ast.Return) -> None:
        self._unsupported_exit(node)

    def visit_Break(self, node: ast.Break) -> None:
        if self.python_loop_depth == 0:
            self._unsupported_exit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        if self.python_loop_depth == 0:
            self._unsupported_exit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self._unsupported_exit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._raise(node, "does not support deletion")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            if any(isinstance(target, ast.Subscript) for target in node.targets):
                self._raise(node, "does not support chained tensor stores")
            for target in node.targets:
                self._check_local_target(target)
        else:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                self._check_tensor_store(node, target)
            else:
                self._check_local_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._raise(node, "does not support annotated tensor stores")
        self._check_local_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._raise(node, "does not support augmented tensor stores")
        self._check_local_target(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.nested_constructs.append(self._construct_key("for", node))
        if self.is_runtime_for(node):
            return
        self._check_local_target(node.target)
        self.visit(node.iter)
        self._visit_python_loop(node.body, node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.nested_constructs.append(self._construct_key("while", node))
        if not self.is_static_test(node.test):
            return
        self.visit(node.test)
        self._visit_python_loop(node.body, node.orelse)

    def visit_If(self, node: ast.If) -> None:
        self.nested_constructs.append(self._construct_key("if", node))
        if self.is_static_test(node.test):
            self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._check_local_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support try statements")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support try-star statements")

    def visit_Match(self, node: ast.Match) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support match statements")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._raise(node, "does not support assignment expressions")

    def visit_Await(self, node: ast.Await) -> None:
        self._raise(node, "does not support await expressions")

    def visit_Yield(self, node: ast.Yield) -> None:
        self._raise(node, "does not support yield expressions")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._raise(node, "does not support yield expressions")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def _check_tensor_store(
        self, assignment: ast.Assign, target: ast.Subscript
    ) -> None:
        if _contains_slice(target.slice):
            self._raise(target, "does not support tensor slice assignment")
        self.assignment_targets.append("tensor subscript")
        self.tensor_store_assignments.add(assignment)
        self.visit(target.value)
        self.visit(target.slice)

    def _check_local_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.assignment_targets.append(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and all(
            isinstance(element, ast.Name) for element in target.elts
        ):
            self.assignment_targets.extend(element.id for element in target.elts)
            return
        self._raise(
            target,
            "only supports assignments to local names, tuples/lists of local "
            "names, or Catlass tensor elements",
        )

    def _unsupported_exit(self, node: ast.AST) -> None:
        self._raise(node, "does not support return, break, continue, or raise")

    def _visit_python_loop(self, body: list[ast.stmt], orelse: list[ast.stmt]) -> None:
        self.python_loop_depth += 1
        try:
            for statement in [*body, *orelse]:
                self.visit(statement)
        finally:
            self.python_loop_depth -= 1

    def _raise(self, node: ast.AST, message: str) -> None:
        error = SyntaxError(f"dynamic Tla {self.construct_name} {message}")
        relative_lineno = int(getattr(node, "lineno", 0) or 0)
        error.filename = self.filename
        error.lineno = self.line_offset + relative_lineno
        error.offset = (
            int(getattr(node, "col_offset", 0)) + 1
            if getattr(node, "col_offset", None) is not None
            else None
        )
        if 0 < relative_lineno <= len(self.source_lines):
            error.text = self.source_lines[relative_lineno - 1]
        raise error

    @staticmethod
    def _construct_key(kind: str, node: ast.AST) -> tuple[str, int, int]:
        return (
            kind,
            int(getattr(node, "lineno", 0) or 0),
            int(getattr(node, "col_offset", 0) or 0),
        )


def _is_syntactic_tla_range_for(node: ast.For) -> bool:
    if not isinstance(node.iter, ast.Call):
        return False
    func = node.iter.func
    if isinstance(func, ast.Name):
        return func.id == "tla_range"
    if isinstance(func, ast.Attribute):
        return func.attr == "range" and isinstance(func.value, ast.Name)
    return False


def _reject_unsupported_dynamic_for_new_defs(
    active_names: set[str],
    body_assigned: set[str],
    target_name: str,
    following_loads: set[str],
) -> None:
    if target_name in following_loads:
        raise SyntaxError(
            "dynamic Tla for induction variables cannot be used after the loop: "
            f"{target_name}"
        )
    new_defs = (body_assigned - active_names) - {target_name}
    used_after = sorted(new_defs & following_loads)
    if used_after:
        raise SyntaxError(
            "dynamic Tla for values used after the loop must be initialized "
            f"before the loop: {', '.join(used_after)}"
        )


def _call_base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return _call_base_name(node.value)
    if isinstance(node, ast.Name) and node.id not in {"tla"}:
        return node.id
    return None


def _reject_unsupported_dynamic_if_new_defs(
    active_names: set[str],
    then_assigned: set[str],
    else_assigned: set[str],
    following_loads: set[str],
) -> None:
    newly_assigned = (then_assigned | else_assigned) - active_names
    used_later = newly_assigned & following_loads
    if used_later:
        names = ", ".join(sorted(used_later))
        raise SyntaxError(
            "dynamic Tla if requires variables used after the branch to be "
            f"initialized before the if: {names}"
        )


def _reject_unsupported_dynamic_while_new_defs(
    active_names: set[str],
    body_assigned: set[str],
    following_loads: set[str],
) -> None:
    newly_assigned = body_assigned - active_names
    used_later = newly_assigned & following_loads
    if used_later:
        names = ", ".join(sorted(used_later))
        raise SyntaxError(
            "dynamic Tla while requires variables used after the loop to be "
            f"initialized before the while: {names}"
        )


def _captured_keywords(analysis: Any) -> list[ast.keyword]:
    """Pass the method-only captures to the frontend helper for validation.

    A captured name is never a carrier, so the body traces once and any method
    called on it runs once. That is correct for a TLA handle, whose methods
    emit ops into the region, and silently wrong for a host object, whose
    methods mutate at trace time. The helper can only tell the two apart from
    the runtime value, so send the values along with the names.
    """
    names = list(analysis.captured_names)
    if not names:
        return []
    return [
        ast.keyword(
            arg="captured_names",
            value=ast.Tuple(
                elts=[ast.Constant(value=name) for name in names], ctx=ast.Load()
            ),
        ),
        ast.keyword(
            arg="captured_values",
            value=ast.Tuple(
                elts=[ast.Name(id=name, ctx=ast.Load()) for name in names],
                ctx=ast.Load(),
            ),
        ),
    ]


def _names_list(names: list[str]) -> ast.List:
    return ast.List(
        elts=[ast.Name(id=name, ctx=ast.Load()) for name in names],
        ctx=ast.Load(),
    )


def _ends_with_return(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], ast.Return)


def _append_return_for_carried_names(
    body: list[ast.stmt], carried_names: list[str]
) -> list[ast.stmt]:
    if not carried_names or _ends_with_return(body):
        return body
    return [*body, ast.Return(value=_names_list(carried_names))]
