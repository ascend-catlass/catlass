"""In-process rewrite memo: one AST rewrite per function object."""

from __future__ import annotations

import ast
import contextlib
import types
from collections.abc import Iterator
from typing import Any

import pytest

import catlass.tla as tla
from catlass import runtime
from catlass import tla_ast_decorators as ast_decorators
from catlass.base_dsl import ast_preprocessor as ap
from catlass.dsl import unwrap_jit_callable

_REBOUND = 1
_MUTATED: list[int] = []
_UNRELATED = 1
_HELPER_OP = abs
_HELPER_UNRELATED = 1
# Variables a host script rebinds (``for name in ...``) while a kernel compiles.
# The kernels below use the same names for parameters, locals and inner scopes.
_SHARED_LOCAL = 0
_SHARED_PARAM = 0
_SHARED_ALIAS = 0
_SHARED_LOOP = 0
_SHARED_LAMBDA = 0
_SHARED_COMPREHENSION = 0
# The same, for names that kernels below capture as closure variables.
_SHARED_CLOSURE_VALUE = 0
_SHARED_CLOSURE_ROOT = 0
_SHARED_CLOSURE_OP = 0
_SHARED_CLOSURE_NESTED = 0
_SHARED_CLOSURE_DECLARED = 0
_SHARED_CLOSURE_DEFAULT = 0
_DEFAULT_GLOBAL = 3


def _hooks() -> dict[str, Any]:
    return dict(
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


def _fresh_branching():
    """A new function object each time, so tests never share a memo."""

    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_REBOUND, len(_MUTATED))

    return branching


def _branching_variant():
    """Same name and arity as ``_fresh_branching``'s function, different code."""

    def branching(value: int) -> None:
        if value < 0:
            tla.make_coord(0, _UNRELATED)

    return branching


def _plain(value: int) -> int:
    return value + 1


# Stands in for an imported module: the kernel calls ``_PROBE.helper`` and reads
# ``_PROBE.limit``.
_PROBE = types.ModuleType("probe")
_PROBE.helper = _plain  # type: ignore[attr-defined]
_PROBE.limit = 4  # type: ignore[attr-defined]


def _calls_probe_helper():
    def branching(value: int) -> None:
        if value > 0:
            _PROBE.helper(value)
            tla.make_coord(_PROBE.limit, 0)

    return branching


class _CountingProxy:
    """Counts the ``__getattr__`` lookups made on it, by whoever makes them."""

    def __init__(self) -> None:
        self.lookups = 0
        self.compared = 0

    def __getattr__(self, name: str) -> Any:
        if name == "make_coord":
            self.lookups += 1
            return tla.make_coord
        raise AttributeError(name)

    def __eq__(self, other: object) -> bool:
        self.compared += 1
        return self is other

    __hash__ = object.__hash__


_PROXY = _CountingProxy()


def _calls_proxy():
    def branching(value: int) -> None:
        if value > 0:
            _PROXY.make_coord(value, 0)

    return branching


_CALLABLES: list[Any] = [abs, abs]
_CONFIG: dict[str, Any] = {"op": abs, "unused": abs}


class _LoggingList(list):
    """A list whose ``__getitem__`` is user code."""

    def __getitem__(self, index: Any) -> Any:
        return super().__getitem__(index)


_SUBCLASSED = _LoggingList([abs])


def _calls_list_element():
    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_CALLABLES[0](value), 0)

    return branching


def _calls_list_element_through_an_alias():
    def branching(value: int) -> None:
        if value > 0:
            selected = _CALLABLES[0]
            tla.make_coord(selected(value), 0)

    return branching


def _calls_dict_element():
    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_CONFIG["op"](value), 0)

    return branching


def _calls_missing_dict_element():
    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_CONFIG["late"](value), 0)

    return branching


def _calls_subclassed_element():
    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_SUBCLASSED[0](value), 0)

    return branching


def _closure_over(op: Any):
    """A function calling ``op`` from its closure, and a way to rebind ``op``."""

    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(op(value), 0)

    def rebind(new_op: Any) -> None:
        nonlocal op
        op = new_op

    return branching, rebind


@tla.jit
def _leaf_helper() -> None:
    tla.make_coord(_HELPER_OP(2), 0)


@tla.jit
def _outer_helper() -> None:
    _leaf_helper()


@tla.jit
def _replaceable_helper() -> None:
    tla.make_coord(1, 0)


@tla.jit
def _subclassed_helper() -> None:
    tla.make_coord(_SUBCLASSED[0](2), 0)


def _calls_helper(helper: Any):
    """A function that names a helper behind a branch it may never take."""

    def branching(value: int) -> None:
        if value > 0:
            helper()

    return branching


def _assigns_shared_local():
    def branching(value: int) -> None:
        if value > 0:
            _SHARED_LOCAL = 2
            tla.make_coord(_SHARED_LOCAL, 0)

    return branching


def _takes_shared_param():
    def branching(_SHARED_PARAM: int) -> None:
        if _SHARED_PARAM > 0:
            tla.make_coord(_SHARED_PARAM, 0)

    return branching


def _calls_through_shared_alias():
    def branching(value: int) -> None:
        if value > 0:
            _SHARED_ALIAS = tla
            _SHARED_ALIAS.make_coord(value, 0)

    return branching


def _loops_over_shared_name():
    def branching(value: int) -> None:
        if value > 0:
            for _SHARED_LOOP in (1, 2):
                tla.make_coord(_SHARED_LOOP, 0)

    return branching


def _reads_global_named_like_a_lambda_parameter():
    def branching(value: int) -> None:
        if value > 0:
            double = lambda _SHARED_LAMBDA: _SHARED_LAMBDA * 2  # noqa: E731
            tla.make_coord(_SHARED_LAMBDA, double(value))

    return branching


def _reads_global_named_like_a_comprehension_variable():
    def branching(value: int) -> None:
        if value > 0:
            squares = [_SHARED_COMPREHENSION for _SHARED_COMPREHENSION in (1, 2)]
            tla.make_coord(_SHARED_COMPREHENSION, len(squares))

    return branching


def _rebinds_a_closure_variable():
    count = 0

    def branching(value: int) -> None:
        nonlocal count
        count = value

    return branching


def _captures_shared_value():
    _SHARED_CLOSURE_VALUE = 7

    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_SHARED_CLOSURE_VALUE, 0)

    return branching


def _calls_through_shared_closure_root():
    _SHARED_CLOSURE_ROOT = tla

    def branching(value: int) -> None:
        if value > 0:
            _SHARED_CLOSURE_ROOT.make_coord(value, 0)

    return branching


def _calls_shared_closure_op():
    _SHARED_CLOSURE_OP = abs

    def branching(value: int) -> None:
        if value > 0:
            tla.make_coord(_SHARED_CLOSURE_OP(value), 0)

    return branching


def _captures_shared_value_in_a_lambda():
    _SHARED_CLOSURE_NESTED = 7

    def branching(value: int) -> None:
        if value > 0:
            double = lambda: _SHARED_CLOSURE_NESTED * 2  # noqa: E731
            tla.make_coord(double(), 0)

    return branching


def _captures_shared_value_in_a_default():
    _SHARED_CLOSURE_DEFAULT = 7

    # The default runs in the factory, where the name is the closure variable.
    def branching(value: int = _SHARED_CLOSURE_DEFAULT) -> None:
        if value > 0:
            tla.make_coord(_SHARED_CLOSURE_DEFAULT, 0)

    return branching


def _defaults_to_a_module_global():
    def branching(value: int = _DEFAULT_GLOBAL) -> None:
        if value > 0:
            tla.make_coord(value, 0)

    return branching


def _declares_a_global_over_a_closure_variable():
    _SHARED_CLOSURE_DECLARED = 1

    def branching(value: int) -> None:
        def module_value() -> int:
            global _SHARED_CLOSURE_DECLARED
            return _SHARED_CLOSURE_DECLARED

        from_module = module_value()
        if value > 0:
            tla.make_coord(from_module, _SHARED_CLOSURE_DECLARED)

    return branching


def _closure_kernel():
    host_captured = 7

    @tla.kernel
    def kernel() -> None:
        tla.make_coord(host_captured, 0)

    return kernel


@contextlib.contextmanager
def _replaced(container: Any, key: Any, value: Any) -> Iterator[None]:
    saved = container[key]
    container[key] = value
    try:
        yield
    finally:
        container[key] = saved


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every rewrite the memo could not skip."""

    calls: list[Any] = []
    real = ap._build_rewrite_plan

    def counting(fn: Any, prepared: Any) -> Any:
        calls.append(fn)
        return real(fn, prepared)

    monkeypatch.setattr(ap, "_build_rewrite_plan", counting)
    return calls


def test_second_lowering_reuses_the_rewrite(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    collected: list[ast.FunctionDef] = []
    collect = ap._collect_rewrite_source_facts

    def counting(target: ast.FunctionDef) -> Any:
        collected.append(target)
        return collect(target)

    monkeypatch.setattr(ap, "_collect_rewrite_source_facts", counting)
    fn = _fresh_branching()
    first = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(collected) == 1
    second = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(collected) == 1
    assert len(builds) == 1
    assert first is not fn and second is not fn
    # Each lowering still gets its own function and globals.
    assert first is not second
    assert first.__globals__ is not second.__globals__
    assert first.__code__ is second.__code__


def test_function_needing_no_rewrite_is_memoized(builds: list[Any]) -> None:
    assert ap._preprocess_for_lowering(_plain, **_hooks()) is _plain
    assert ap._preprocess_for_lowering(_plain, **_hooks()) is _plain
    assert len(builds) == 1


def test_changed_hooks_rewrite_again(builds: list[Any]) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    other = _hooks()
    other["internal_for"] = lambda *args, **kwargs: None
    ap._preprocess_for_lowering(fn, **other)
    assert len(builds) == 2


def test_clear_rewrite_memo_rewrites_again(builds: list[Any]) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._clear_rewrite_memo(fn)
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_swapped_code_object_rewrites_again(builds: list[Any]) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    fn.__code__ = _branching_variant().__code__
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_failed_rewrite_is_not_memoized(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _fresh_branching()
    real = ap.validate_language_boundaries
    attempts: list[int] = []

    def flaky(target: Any, **kwargs: Any) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise SyntaxError("rejected")
        real(target, **kwargs)

    monkeypatch.setattr(ap, "validate_language_boundaries", flaky)
    for _ in range(2):
        with pytest.raises(SyntaxError, match="rejected"):
            ap._preprocess_for_lowering(fn, **_hooks())
    assert not builds and getattr(fn, ap._REWRITE_MEMO_ATTR, None) is None
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(attempts) == 3 and len(builds) == 1


def test_rebound_global_rewrites_again(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_REBOUND", object())
    rewritten = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2
    assert rewritten.__globals__["_REBOUND"] is globals()["_REBOUND"]


def test_defined_and_removed_globals_rewrite_again(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    def uses_late_name(value: int) -> None:
        if value > 0:
            tla.make_coord(_LATE_GLOBAL, 0)  # noqa: F821

    ap._preprocess_for_lowering(uses_late_name, **_hooks())
    monkeypatch.setitem(globals(), "_LATE_GLOBAL", 3)
    ap._preprocess_for_lowering(uses_late_name, **_hooks())
    assert len(builds) == 2
    monkeypatch.delitem(globals(), "_LATE_GLOBAL")
    ap._preprocess_for_lowering(uses_late_name, **_hooks())
    assert len(builds) == 3


def test_mutating_a_referenced_object_in_place_keeps_the_memo(
    builds: list[Any],
) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    _MUTATED.append(1)
    try:
        rewritten = ap._preprocess_for_lowering(fn, **_hooks())
    finally:
        _MUTATED.clear()
    assert len(builds) == 1
    assert rewritten.__globals__["_MUTATED"] is _MUTATED


def test_unreferenced_global_keeps_the_memo_but_is_seen_live(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _fresh_branching()
    ap._preprocess_for_lowering(fn, **_hooks())
    marker = object()
    monkeypatch.setitem(globals(), "_UNRELATED", marker)
    rewritten = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    assert rewritten.__globals__["_UNRELATED"] is marker


def test_closure_instances_do_not_share_a_memo(builds: list[Any]) -> None:
    def factory(offset: int):
        def branching(value: int) -> None:
            if value > 0:
                tla.make_coord(offset, 0)

        return branching

    first, second = factory(1), factory(2)
    for _ in range(2):
        ap._preprocess_for_lowering(first, **_hooks())
        ap._preprocess_for_lowering(second, **_hooks())
    assert len(builds) == 2
    # Closure cells stay live, not copied into the memo.
    rewritten = ap._preprocess_for_lowering(first, **_hooks())
    assert rewritten.__closure__[0] is first.__closure__[0]


def test_replaced_module_attribute_call_target_rewrites_again(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _calls_probe_helper()
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    monkeypatch.setattr(_PROBE, "helper", _branching_variant())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_module_attribute_read_as_a_value_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _calls_probe_helper()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setattr(_PROBE, "limit", 8)
    rewritten = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    # The rewritten body still reads the attribute when it runs.
    assert rewritten.__globals__["_PROBE"].limit == 8


def test_recording_call_paths_runs_no_user_code(builds: list[Any]) -> None:
    fn = _calls_proxy()
    prepared = ap._prepare_function_source(fn)
    assert prepared is not None and prepared.target is not None
    lookups, compared = _PROXY.lookups, _PROXY.compared
    deps = ap._rewrite_attribute_deps(
        fn, prepared.target, ap._global_scope_names(fn, prepared, prepared.target)
    )
    assert deps and deps[0][0] == "_PROXY"
    assert (_PROXY.lookups, _PROXY.compared) == (lookups, compared)


def test_reusing_the_memo_runs_no_user_code(builds: list[Any]) -> None:
    fn = _calls_proxy()
    ap._preprocess_for_lowering(fn, **_hooks())
    # Whatever the first rewrite itself did, reusing it must add nothing.
    lookups, compared = _PROXY.lookups, _PROXY.compared
    for _ in range(3):
        ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    assert (_PROXY.lookups, _PROXY.compared) == (lookups, compared)


_KERNEL_PROXY = _CountingProxy()


@tla.kernel
def _proxy_lowering_kernel() -> None:
    _KERNEL_PROXY.make_coord(1, 0)
    with tla.vector():
        tla.print(_KERNEL_PROXY.lookups)


def test_lowering_adds_no_dynamic_lookups_of_its_own() -> None:
    # The kernel prints the counter, so extra lookups would change its IR.
    deltas = []
    for _ in range(3):
        before = _KERNEL_PROXY.lookups
        _proxy_lowering_kernel.dump_mlir(type_args=())
        deltas.append(_KERNEL_PROXY.lookups - before)
    assert deltas == [1, 1, 1]


@tla.kernel
def _memo_lowering_kernel(limit: int) -> None:
    if limit > 0:
        tla.make_coord(limit, 0)


def test_cached_and_rebuilt_lowering_emit_identical_ir(builds: list[Any]) -> None:
    first = _memo_lowering_kernel.dump_mlir(type_args=(2,))
    builds_after_first = len(builds)
    second = _memo_lowering_kernel.dump_mlir(type_args=(2,))
    assert first == second
    assert builds_after_first >= 1 and len(builds) == builds_after_first
    ap._clear_rewrite_memo(_memo_lowering_kernel.fn)
    assert _memo_lowering_kernel.dump_mlir(type_args=(2,)) == first
    assert len(builds) == builds_after_first + 1


def test_rebound_closure_cell_is_validated_again(builds: list[Any]) -> None:
    fn, rebind = _closure_over(abs)
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    rebind(hash)
    with pytest.raises(SyntaxError, match="hash"):
        ap._preprocess_for_lowering(fn, **_hooks())
    # The memo built for ``abs`` is still there and valid once ``abs`` is back.
    rebind(abs)
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_rebound_closure_cell_is_rejected_when_lowering() -> None:
    @tla.kernel
    def kernel() -> None:
        tla.make_coord(op(2), 0)

    op = abs
    kernel.dump_mlir(type_args=())
    op = hash
    with pytest.raises(SyntaxError, match="hash"):
        kernel.dump_mlir(type_args=())
    op = abs
    kernel.dump_mlir(type_args=())


def test_closure_cell_read_as_a_value_keeps_the_memo(builds: list[Any]) -> None:
    def factory(offset: int):
        def branching(value: int) -> None:
            if value > 0:
                tla.make_coord(offset, 0)

        def rebind(new_offset: int) -> None:
            nonlocal offset
            offset = new_offset

        return branching, rebind

    fn, rebind = factory(1)
    ap._preprocess_for_lowering(fn, **_hooks())
    rebind(2)
    rewritten = ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    assert rewritten.__closure__[0].cell_contents == 2


@pytest.mark.parametrize(
    "make",
    [_calls_list_element, _calls_list_element_through_an_alias],
    ids=["direct", "alias"],
)
def test_replaced_list_element_is_validated_again(make: Any, builds: list[Any]) -> None:
    fn = make()
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    with _replaced(_CALLABLES, 0, hash):
        with pytest.raises(SyntaxError, match="hash"):
            ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_replaced_dict_element_is_validated_again(builds: list[Any]) -> None:
    fn = _calls_dict_element()
    ap._preprocess_for_lowering(fn, **_hooks())
    with _replaced(_CONFIG, "op", hash):
        with pytest.raises(SyntaxError, match="hash"):
            ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_dict_key_added_later_is_validated(builds: list[Any]) -> None:
    fn = _calls_missing_dict_element()
    ap._preprocess_for_lowering(fn, **_hooks())
    _CONFIG["late"] = hash
    try:
        with pytest.raises(SyntaxError, match="hash"):
            ap._preprocess_for_lowering(fn, **_hooks())
    finally:
        del _CONFIG["late"]
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_container_elements_the_function_never_reads_keep_the_memo(
    builds: list[Any],
) -> None:
    fn = _calls_list_element()
    ap._preprocess_for_lowering(fn, **_hooks())
    with _replaced(_CALLABLES, 1, hash):
        ap._preprocess_for_lowering(fn, **_hooks())
    _CALLABLES.append(hash)
    try:
        ap._preprocess_for_lowering(fn, **_hooks())
    finally:
        _CALLABLES.pop()
    assert len(builds) == 1

    config_fn = _calls_dict_element()
    ap._preprocess_for_lowering(config_fn, **_hooks())
    with _replaced(_CONFIG, "unused", hash):
        ap._preprocess_for_lowering(config_fn, **_hooks())
    assert len(builds) == 2


def test_element_read_through_user_code_is_never_memoized(
    builds: list[Any],
) -> None:
    fn = _calls_subclassed_element()
    for _ in range(3):
        ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 3
    assert getattr(fn, ap._REWRITE_MEMO_ATTR, None) is None


def test_helper_global_change_is_validated_again(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _calls_helper(_leaf_helper)
    ap._preprocess_for_lowering(fn, **_hooks())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1
    monkeypatch.setitem(globals(), "_HELPER_OP", hash)
    with pytest.raises(SyntaxError, match="hash"):
        ap._preprocess_for_lowering(fn, **_hooks())
    # The memo built for ``abs`` is still there and valid once ``abs`` is back.
    monkeypatch.setitem(globals(), "_HELPER_OP", abs)
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_nested_helper_global_change_is_validated_again(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _calls_helper(_outer_helper)
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_HELPER_OP", hash)
    with pytest.raises(SyntaxError, match="hash"):
        ap._preprocess_for_lowering(fn, **_hooks())


def test_helper_global_change_is_rejected_when_the_helper_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"helper": True}

    @tla.jit
    def helper() -> None:
        tla.make_coord(_HELPER_OP(2), 0)

    @tla.kernel
    def kernel() -> None:
        if calls["helper"]:
            helper()

    kernel.dump_mlir(type_args=())
    monkeypatch.setitem(globals(), "_HELPER_OP", hash)
    calls["helper"] = False
    with pytest.raises(SyntaxError, match="hash"):
        kernel.dump_mlir(type_args=())


def test_replaced_helper_code_rewrites_again(builds: list[Any]) -> None:
    fn = _calls_helper(_replaceable_helper)
    ap._preprocess_for_lowering(fn, **_hooks())
    helper = unwrap_jit_callable(_replaceable_helper)
    original = helper.__code__

    def variant():
        def _replaceable_helper() -> None:
            tla.make_coord(2, 0)

        return _replaceable_helper

    helper.__code__ = variant().__code__
    try:
        ap._preprocess_for_lowering(fn, **_hooks())
    finally:
        helper.__code__ = original
    assert len(builds) == 2


def test_global_no_helper_reads_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _calls_helper(_leaf_helper)
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_HELPER_UNRELATED", object())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_helper_read_through_user_code_is_never_memoized(
    builds: list[Any],
) -> None:
    fn = _calls_helper(_subclassed_helper)
    for _ in range(3):
        ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 3
    assert getattr(fn, ap._REWRITE_MEMO_ATTR, None) is None


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (_assigns_shared_local, "_SHARED_LOCAL"),
        (_takes_shared_param, "_SHARED_PARAM"),
        (_calls_through_shared_alias, "_SHARED_ALIAS"),
        (_loops_over_shared_name, "_SHARED_LOOP"),
    ],
    ids=["local", "parameter", "call-path-root", "loop-target"],
)
def test_host_variable_named_like_a_local_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any], make: Any, name: str
) -> None:
    fn = make()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), name, 0)
    for host_value in range(1, 40):  # what `for name in ...:` does at module level
        globals()[name] = host_value
        ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (_reads_global_named_like_a_lambda_parameter, "_SHARED_LAMBDA"),
        (_reads_global_named_like_a_comprehension_variable, "_SHARED_COMPREHENSION"),
    ],
    ids=["lambda", "comprehension"],
)
def test_global_named_like_an_inner_scope_variable_is_still_tracked(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any], make: Any, name: str
) -> None:
    # The lambda parameter and the comprehension variable live in their own
    # scopes, so the same name read in the function body is a global.
    fn = make()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), name, object())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def _global_scope_names_of(fn: Any) -> frozenset[str]:
    prepared = ap._prepare_function_source(fn)
    assert prepared is not None and prepared.target is not None
    return ap._global_scope_names(fn, prepared, prepared.target)


@pytest.mark.parametrize(
    ("make", "kept", "excluded"),
    [
        (
            _captures_shared_value_in_a_default,
            {"tla"},
            {"_SHARED_CLOSURE_DEFAULT", "value"},
        ),
        (_defaults_to_a_module_global, {"tla", "_DEFAULT_GLOBAL"}, {"value"}),
        (
            _declares_a_global_over_a_closure_variable,
            {"tla", "_SHARED_CLOSURE_DECLARED"},
            {"value", "from_module", "module_value"},
        ),
    ],
    ids=["closure-default", "global-default", "nested-global"],
)
def test_shared_source_facts_preserve_scope_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    make: Any,
    kept: set[str],
    excluded: set[str],
) -> None:
    fn = make()
    prepared = ap._prepare_function_source(fn)
    assert prepared is not None and prepared.target is not None
    target = prepared.target
    facts = ap._collect_rewrite_source_facts(target)
    independent_names = ap._global_scope_names(fn, prepared, target)
    independent_globals = ap._rewrite_global_deps(fn, target, independent_names)
    independent_attributes = ap._rewrite_attribute_deps(fn, target, independent_names)
    walk = ast.walk

    def no_target_rewalk(node: ast.AST) -> Any:
        # Signature subtrees can still be visited; dependency syntax was
        # already collected from the full function and must be shared.
        assert node is not target
        return walk(node)

    monkeypatch.setattr(ast, "walk", no_target_rewalk)
    names = ap._global_scope_names(fn, prepared, target, _source_facts=facts)
    global_deps = ap._rewrite_global_deps(fn, target, names, _source_facts=facts)
    attributes = ap._rewrite_attribute_deps(fn, target, names, _source_facts=facts)
    assert names == independent_names
    assert global_deps == independent_globals
    assert attributes == independent_attributes
    assert kept <= names
    assert not excluded & names
    assert kept <= dict(global_deps).keys()
    assert not excluded & dict(global_deps).keys()


def test_shared_source_facts_filter_local_paths_and_deduplicate_globals(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    nested = types.SimpleNamespace(helper=_plain)
    monkeypatch.setattr(_PROBE, "nested", nested, raising=False)

    def branching(value: int) -> None:
        _SHARED_ALIAS = _PROBE
        _PROBE.nested.helper(1)
        _PROBE.nested.helper(1)
        _SHARED_ALIAS.nested.helper(1)
        if value > 0:
            tla.make_coord(1, 0)

    prepared = ap._prepare_function_source(branching)
    assert prepared is not None and prepared.target is not None
    facts = ap._collect_rewrite_source_facts(prepared.target)
    assert facts.call_paths.count(("_PROBE", ("nested", "helper"))) == 1
    assert ("_SHARED_ALIAS", ("nested", "helper")) in facts.call_paths
    names = ap._global_scope_names(
        branching, prepared, prepared.target, _source_facts=facts
    )
    attributes = ap._rewrite_attribute_deps(
        branching, prepared.target, names, _source_facts=facts
    )
    assert [(root, chain) for root, chain, _, _ in attributes] == [
        ("_PROBE", ("nested", "helper")),
        ("tla", ("make_coord",)),
    ]

    kernel = tla.kernel(branching)
    first = kernel.dump_mlir(type_args=(2,))
    assert kernel.dump_mlir(type_args=(2,)) == first
    assert len(builds) == 1
    monkeypatch.setitem(globals(), "_SHARED_ALIAS", object())
    assert kernel.dump_mlir(type_args=(2,)) == first
    assert len(builds) == 1
    monkeypatch.setattr(nested, "helper", lambda value: value + 2)
    assert kernel.dump_mlir(type_args=(2,)) == first
    assert len(builds) == 2


def test_global_scope_names_follow_lexical_scope() -> None:
    names = _global_scope_names_of(_assigns_shared_local())
    assert "tla" in names
    assert not names & {"_SHARED_LOCAL", "value"}


def test_global_scope_names_leave_out_closure_variables() -> None:
    names = _global_scope_names_of(_captures_shared_value())
    assert "tla" in names
    assert "_SHARED_CLOSURE_VALUE" not in names
    # A closure variable read from a nested scope is one of the function's own.
    assert "_SHARED_CLOSURE_NESTED" not in _global_scope_names_of(
        _captures_shared_value_in_a_lambda()
    )
    # So is one the ``def`` statement reads for a default: it runs in the factory.
    assert "_SHARED_CLOSURE_DEFAULT" not in _global_scope_names_of(
        _captures_shared_value_in_a_default()
    )


def test_global_scope_names_keep_a_global_read_by_the_def_statement() -> None:
    assert "_DEFAULT_GLOBAL" in _global_scope_names_of(_defaults_to_a_module_global())


def test_global_scope_names_keep_a_name_declared_global_in_a_nested_scope() -> None:
    # The function reads the closure variable, but a nested ``global`` statement
    # names the module global, which the rewrite may read.
    names = _global_scope_names_of(_declares_a_global_over_a_closure_variable())
    assert "_SHARED_CLOSURE_DECLARED" in names


def test_source_that_cannot_be_analyzed_alone_tracks_every_name_but_closure_ones() -> (
    None
):
    # ``nonlocal`` of a closure variable has no binding in the extracted source.
    fn = _rebinds_a_closure_variable()
    prepared = ap._prepare_function_source(fn)
    assert prepared is not None
    assert ap._symtable_global_names(prepared) is None
    names = _global_scope_names_of(fn)
    assert "value" in names  # not analyzed, so even a parameter counts
    assert "count" not in names  # a closure variable never does


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (_captures_shared_value, "_SHARED_CLOSURE_VALUE"),
        (_calls_through_shared_closure_root, "_SHARED_CLOSURE_ROOT"),
        (_calls_shared_closure_op, "_SHARED_CLOSURE_OP"),
        (_captures_shared_value_in_a_lambda, "_SHARED_CLOSURE_NESTED"),
        (_captures_shared_value_in_a_default, "_SHARED_CLOSURE_DEFAULT"),
    ],
    ids=["value", "call-path-root", "called-value", "nested-lambda", "default"],
)
def test_host_variable_named_like_a_closure_variable_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any], make: Any, name: str
) -> None:
    fn = make()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), name, 0)
    for host_value in range(1, 40):
        globals()[name] = host_value
        ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 1


def test_global_read_by_the_def_statement_is_still_tracked(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _defaults_to_a_module_global()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_DEFAULT_GLOBAL", 4)
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_global_declared_in_a_nested_scope_is_still_tracked(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    fn = _declares_a_global_over_a_closure_variable()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_SHARED_CLOSURE_DECLARED", object())
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_empty_closure_cell_falls_back_to_the_global_of_that_name(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    def factory():
        def branching(value: int) -> None:
            if value > 0:
                tla.make_coord(_SHARED_CLOSURE_OP(value), 0)

        def fill() -> None:
            nonlocal _SHARED_CLOSURE_OP
            _SHARED_CLOSURE_OP = abs

        if False:  # the cell stays empty until ``fill`` runs
            _SHARED_CLOSURE_OP = None
        return branching, fill

    monkeypatch.setitem(globals(), "_SHARED_CLOSURE_OP", abs)
    fn, fill = factory()
    ap._preprocess_for_lowering(fn, **_hooks())
    # While the cell is empty the validation resolves the global of that name.
    monkeypatch.setitem(globals(), "_SHARED_CLOSURE_OP", hash)
    with pytest.raises(SyntaxError, match="hash"):
        ap._preprocess_for_lowering(fn, **_hooks())
    # Once it is filled the cell shadows the global, whatever the global is.
    fill()
    ap._preprocess_for_lowering(fn, **_hooks())
    monkeypatch.setitem(globals(), "_SHARED_CLOSURE_OP", 0)
    ap._preprocess_for_lowering(fn, **_hooks())
    assert len(builds) == 2


def test_host_loop_variable_named_like_a_kernel_closure_variable_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    kernel = _closure_kernel()
    first = kernel.dump_mlir(type_args=())
    compiled = len(builds)
    monkeypatch.setitem(globals(), "host_captured", 0)
    for value in range(1, 40):
        globals()["host_captured"] = value  # `for host_captured in range(...)`
        assert kernel.dump_mlir(type_args=()) == first
    assert len(builds) == compiled


@tla.kernel
def _host_loop_kernel() -> None:
    host_index = 2
    tla.make_coord(host_index, 0)


def test_host_loop_variable_named_like_a_kernel_local_keeps_the_memo(
    monkeypatch: pytest.MonkeyPatch, builds: list[Any]
) -> None:
    _host_loop_kernel.dump_mlir(type_args=())
    compiled = len(builds)
    monkeypatch.setitem(globals(), "host_index", 0)
    for value in range(1, 40):
        globals()["host_index"] = value  # `for host_index in range(...)` in a script
        _host_loop_kernel.dump_mlir(type_args=())
    assert len(builds) == compiled
