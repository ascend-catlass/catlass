from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
from dataclasses import FrozenInstanceError
from types import ModuleType
from unittest.mock import patch

import pytest

import catlass.tla as catlass_module
import catlass.core_api as core_api
import catlass.tla_ast_decorators as ast_decorators
from catlass.base_dsl.ast_preprocessor import (
    _BlockLastReadIndex,
    _cf_symbol_check,
    _ControlFlowAnalyzer,
    _DynamicConditionValidator,
    _FrontendControlFlowTransformer,
    FunctionBlockPlan,
    FunctionPlan,
    _function_needs_frontend_transform,
    _FunctionAnalyzer,
    _root_function_scope_facts,
    _scope_facts_for_transform,
    _SourceSnapshot,
    _source_text_for_info,
    _StatementFactsCache,
    _trusted_dsl_identities,
    maybe_transform_for_lowering,
)
from catlass.tla_ast_decorators import (
    _dynamic_lazy_region,
    _internal_frontend_bool_not,
    _internal_frontend_compare_pair,
    _internal_frontend_if,
    _internal_frontend_if_expr,
    _internal_lazy_attribute,
    _internal_lazy_binop,
    _internal_lazy_subscript,
    _internal_lazy_unary,
    _internal_unknown_effect_call,
)


def _parse_first_statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def test_statement_facts_preserve_source_reads_after_rewrite() -> None:
    statement = _parse_first_statement(
        "result = [transform(item, other) for item in items "
        "for other in expand(item) if predicate(other)]"
    )
    cache = _StatementFactsCache()
    original = cache.get(statement)
    assert original.read_names == {"transform", "items", "expand", "predicate"}
    assert original.assigned_names == {"result"}
    assert original.invoked_names == {"transform", "expand", "predicate"}

    # Rewriting retains the source node's identity and its original facts.
    statement.value = ast.parse("replacement(seed)", mode="eval").body
    assert cache.get(statement) is original

    generated = ast.Assign(targets=statement.targets, value=statement.value)
    rewritten = cache.get(generated)
    assert rewritten.read_names == {"replacement", "seed"}
    assert rewritten.assigned_names == {"result"}
    assert rewritten.invoked_names == {"replacement"}
    assert cache.get(statement).read_names == {
        "transform", "items", "expand", "predicate"
    }


def test_block_last_read_index_respects_statement_boundaries_and_scopes() -> None:
    body = ast.parse(
        "seed = initial\n"
        "if flag:\n"
        "    consume(seed)\n"
        "result = [item for item in items]\n"
        "sink(result)\n"
        "seed = replacement\n"
        "def helper():\n"
        "    return seed\n"
    ).body
    index = _BlockLastReadIndex.build(body, statement_facts=_StatementFactsCache())
    candidates = {"seed", "flag", "consume", "item", "items", "result", "missing"}

    assert index.used_after(0, candidates) == {
        "seed", "flag", "consume", "items", "result"
    }
    assert index.used_after(1, candidates) == {"items", "result"}
    assert index.used_after(2, candidates) == {"result"}
    # A later store or nested function body does not extend a variable's last read.
    assert index.used_after(3, candidates) == set()
    assert index.used_after(len(body) - 1, candidates) == set()
    assert _BlockLastReadIndex.build([]).used_after(-1, candidates) == set()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("前缀 = '值'; 结果 = 调用(变量)\n", "调用(变量)"),
        ("前缀 = '值'; 结果 = (\n    变量\n    + 1\n)\n", "变量"),
        ("前缀 = '值'\n\f结果 = 调用(\n    变量,\n)\n", "调用("),
    ],
    ids=["utf8-columns", "multiline", "form-feed"],
)
def test_source_snapshot_extracts_and_caches_text_without_source_file(
    source: str, expected: str, tmp_path
) -> None:
    path = tmp_path / "kernel.py"
    path.write_text(source, encoding="utf-8")
    node = ast.parse(source).body[-1].value
    span = (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    with patch.object(ast, "get_source_segment", wraps=ast.get_source_segment) as extract:
        snapshot = _SourceSnapshot(path.read_text(encoding="utf-8"))
        info = {"filename": str(path), "_source_ref": (snapshot, span)}
        path.unlink()
        assert "source" not in info
        extract.assert_not_called()
        assert _source_text_for_info(info) == expected
        assert info["source"] == expected
        assert _source_text_for_info(info) == expected
        extract.assert_called_once()


def _validate_condition(source: str, *, construct: str = "if") -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, (ast.If, ast.While))
    _DynamicConditionValidator(
        construct,
        filename="condition_kernel.py",
        line_offset=40,
        source_text=source,
    ).validate(node.test)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "if (bound := predicate):\n    pass\n",
            "does not support assignment expressions in its condition",
        ),
        (
            "if await predicate:\n    pass\n",
            "does not support await expressions in its condition",
        ),
        (
            "if (yield predicate):\n    pass\n",
            "does not support yield expressions in its condition",
        ),
        (
            "if (yield from predicates):\n    pass\n",
            "does not support yield expressions in its condition",
        ),
        (
            "if (lambda: predicate)():\n    pass\n",
            "does not support lambda expressions in its condition",
        ),
        (
            "if [item for item in values]:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if {item for item in values}:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if {item: item for item in values}:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if (item for item in values):\n    pass\n",
            "does not support generator expressions in its condition",
        ),
    ],
    ids=[
        "walrus",
        "await",
        "yield",
        "yield-from",
        "lambda",
        "list-comprehension",
        "set-comprehension",
        "dict-comprehension",
        "generator-expression",
    ],
)
def test_dynamic_condition_rejects_unsupported_syntax(
    source: str, message: str
) -> None:
    with pytest.raises(SyntaxError, match=message):
        _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if False and (lambda: predicate)():\n    pass\n",
        "if True or [item for item in values]:\n    pass\n",
        "if staged_false() and (item for item in values):\n    pass\n",
        "if (1,) is (1,) is (lambda: predicate)():\n    pass\n",
        "while False and (bound := predicate):\n    pass\n",
    ],
)
def test_forbidden_condition_syntax_is_rejected_even_when_unreachable(
    source: str,
) -> None:
    construct = "while" if source.startswith("while") else "if"
    with pytest.raises(SyntaxError, match="dynamic Tla"):
        _validate_condition(source, construct=construct)


def _compact_if_for_transform_discovery(predicate: object) -> None:
    if(predicate):
        pass


def _compact_while_for_transform_discovery(predicate: object) -> None:
    while(predicate):
        predicate = False


def _irrelevant_function_for_transform_discovery(value: object) -> object:
    return value


def _maybe_transform(fn):
    passthrough = lambda *args, **kwargs: None
    return maybe_transform_for_lowering(
        fn,
        internal_for=passthrough,
        internal_region=passthrough,
        internal_if=passthrough,
        internal_if_expr=passthrough,
        internal_bool_and=passthrough,
        internal_bool_or=passthrough,
        internal_bool_not=passthrough,
        internal_compare=passthrough,
        internal_any=passthrough,
        internal_all=passthrough,
        internal_bool=passthrough,
        internal_min=passthrough,
        internal_max=passthrough,
    )


def _transform_with_eager_if(fn):
    def eager_if(
        condition, then_fn, else_fn, *carried_values, carried_names=None
    ):
        del carried_names
        selected = then_fn if condition else else_fn
        if selected is None:
            return carried_values
        result = selected(*carried_values)
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    passthrough = lambda *args, **kwargs: None
    return maybe_transform_for_lowering(
        fn,
        internal_for=passthrough,
        internal_region=passthrough,
        internal_if=eager_if,
        internal_if_expr=passthrough,
        internal_bool_and=passthrough,
        internal_bool_or=passthrough,
        internal_bool_not=passthrough,
        internal_compare=passthrough,
        internal_any=passthrough,
        internal_all=passthrough,
        internal_bool=passthrough,
        internal_min=passthrough,
        internal_max=passthrough,
    )


@pytest.mark.parametrize(
    "fn",
    [_compact_if_for_transform_discovery, _compact_while_for_transform_discovery],
)
def test_ast_discovery_transforms_compact_control_flow(fn) -> None:
    assert _maybe_transform(fn) is not fn


def test_ast_discovery_does_not_transform_irrelevant_function() -> None:
    assert (
        _maybe_transform(_irrelevant_function_for_transform_discovery)
        is _irrelevant_function_for_transform_discovery
    )


def test_frontend_transform_analyzes_root_exactly_once(monkeypatch) -> None:
    original = _FunctionAnalyzer.analyze
    calls = 0

    def counted(self, node):
        nonlocal calls
        calls += 1
        return original(self, node)

    monkeypatch.setattr(_FunctionAnalyzer, "analyze", counted)

    assert _maybe_transform(_compact_if_for_transform_discovery) is not None
    assert calls == 1


def test_frontend_transformer_rejects_non_unique_child_plan_match() -> None:
    source = "def kernel(predicate):\n    if predicate:\n        pass\n"
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    plan = FunctionPlan(
        name="kernel",
        scope_id="kernel@1:0",
        arguments=(),
        local_bindings=(),
        captures=(),
        child_plans=(FunctionBlockPlan("if", 99, 0),),
    )

    with pytest.raises(RuntimeError, match="does not uniquely match source block"):
        _FrontendControlFlowTransformer(
            {}, source_text=source, root_plan=plan
        ).visit(tree)


def test_transformer_uses_child_plan_after_classifier_drift(monkeypatch) -> None:
    source = (
        "def kernel(predicate, limit):\n"
        "    if predicate:\n"
        "        pass\n"
        "    for i in tla.range(limit):\n"
        "        pass\n"
        "    while predicate:\n"
        "        predicate = False\n"
        "    with tla.vector():\n"
        "        pass\n"
    )
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    globals_ = {"tla": catlass_module}
    scope_facts = _scope_facts_for_transform(source, "<test>", target)
    plan = _FunctionAnalyzer(
        global_symbols=globals_, scope_facts=scope_facts
    ).analyze(target)
    assert [child.construct_name for child in plan.child_plans] == [
        "if",
        "for",
        "while",
        "with",
    ]

    implementation = sys.modules[_FunctionAnalyzer.__module__]
    monkeypatch.setattr(implementation, "_is_static_python_if_test", lambda *a: True)
    monkeypatch.setattr(implementation, "_is_tla_range_iter", lambda *a: False)
    transformed = _FrontendControlFlowTransformer(
        globals_, source_text=source, root_plan=plan
    ).visit(tree)
    ast.fix_missing_locations(transformed)
    rendered = ast.unparse(transformed)

    assert "__tladsl_internal_if__" in rendered
    assert "__tladsl_internal_for__" in rendered
    assert "__tladsl_internal_region__" in rendered


def test_transformer_rejects_planned_with_classifier_drift(monkeypatch) -> None:
    source = "def kernel():\n    with tla.vector():\n        pass\n"
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    globals_ = {"tla": catlass_module}
    scope_facts = _scope_facts_for_transform(source, "<test>", target)
    plan = _FunctionAnalyzer(
        global_symbols=globals_, scope_facts=scope_facts
    ).analyze(target)
    implementation = sys.modules[_FunctionAnalyzer.__module__]
    monkeypatch.setattr(implementation, "_region_name_from_with_item", lambda *a: None)

    with pytest.raises(RuntimeError, match="planned runtime with"):
        _FrontendControlFlowTransformer(
            globals_, source_text=source, root_plan=plan
        ).visit(tree)


class _ProtocolSpy:
    calls = 0

    def _called(self):
        type(self).calls += 1
        return True

    __bool__ = _called
    __len__ = _called
    __getattr__ = lambda self, _name: self._called()
    __getitem__ = lambda self, _key: self._called()
    __add__ = lambda self, _other: self._called()
    __neg__ = _called
    __eq__ = lambda self, _other: self._called()
    __contains__ = lambda self, _item: self._called()
    __hash__ = _called
    __iter__ = _called


@pytest.mark.parametrize(
    "operation",
    [
        lambda spy: _internal_lazy_attribute(spy, "field"),
        lambda spy: _internal_lazy_subscript(spy, 0),
        lambda spy: _internal_lazy_binop("Add", spy, 1),
        lambda spy: _internal_lazy_unary("USub", spy),
        lambda spy: _internal_frontend_compare_pair(spy, 1, "=="),
        lambda spy: _internal_frontend_compare_pair(1, spy, "in"),
        lambda spy: _internal_frontend_compare_pair(spy, {1}, "in"),
        lambda spy: _internal_frontend_if_expr(spy, lambda: 1, lambda: 0),
    ],
    ids=[
        "attribute",
        "subscript",
        "binary",
        "unary",
        "compare",
        "membership",
        "hash-membership",
        "truth",
    ],
)
def test_runtime_lazy_protocols_are_rejected_before_user_code(operation) -> None:
    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        operation(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda spy: _internal_frontend_if(spy, lambda: None, lambda: None),
        lambda spy: _internal_frontend_if_expr(spy, lambda: 1, lambda: 0),
    ],
    ids=["statement-if", "if-expression"],
)
def test_outer_condition_rejects_custom_truthiness_before_user_code(operation) -> None:
    _ProtocolSpy.calls = 0
    with pytest.raises(SyntaxError, match="effect unknown"):
        operation(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


def test_boolean_not_rejects_custom_truthiness_before_user_code() -> None:
    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_frontend_bool_not(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (0, True), (1, False), ("", True), ([], True), ([1], False)],
)
def test_boolean_not_stages_exact_builtin_truthiness(value, expected) -> None:
    assert _internal_frontend_bool_not(value) is expected


def test_dynamic_lazy_context_resets_after_unknown_effect_failure() -> None:
    calls: list[str] = []
    with pytest.raises(SyntaxError, match="effect unknown"):
        with _dynamic_lazy_region():
            _internal_unknown_effect_call(lambda: calls.append("dynamic"))
    assert calls == []
    _internal_unknown_effect_call(lambda: calls.append("staged"))
    assert calls == ["staged"]


def test_dynamic_lazy_context_is_reentrant_and_restores_each_scope() -> None:
    calls: list[str] = []
    with _dynamic_lazy_region():
        with pytest.raises(SyntaxError, match="effect unknown"):
            _internal_unknown_effect_call(lambda: calls.append("outer"))
        with _dynamic_lazy_region():
            with pytest.raises(SyntaxError, match="effect unknown"):
                _internal_unknown_effect_call(lambda: calls.append("inner"))
        with pytest.raises(SyntaxError, match="effect unknown"):
            _internal_unknown_effect_call(lambda: calls.append("outer-again"))
    _internal_unknown_effect_call(lambda: calls.append("staged"))
    assert calls == ["staged"]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda: _internal_lazy_subscript([2, 3], 1), 3),
        (lambda: _internal_lazy_subscript({"key": 4}, "key"), 4),
        (lambda: _internal_lazy_binop("Add", 2, 3), 5),
        (lambda: _internal_lazy_unary("USub", 2), -2),
        (lambda: _internal_frontend_compare_pair((1, 2), (1, 3), "<"), True),
        (lambda: _internal_frontend_compare_pair(2, (1, 2), "in"), True),
        (lambda: _internal_frontend_if_expr([], lambda: "yes", lambda: "no"), "no"),
    ],
)
def test_runtime_lazy_exact_builtin_operations_are_allowed(operation, expected) -> None:
    with _dynamic_lazy_region():
        assert operation() == expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (None, "false"),
        (0, "false"),
        (1, "true"),
        ("", "false"),
        ("value", "true"),
        ([], "false"),
        ([1], "true"),
        ({}, "false"),
        ({"key": 1}, "true"),
    ],
)
def test_outer_if_expression_stages_exact_builtin_truthiness(
    condition, expected
) -> None:
    calls: list[str] = []

    def selected(value: str):
        return lambda: calls.append(value) or value

    result = _internal_frontend_if_expr(
        condition, selected("true"), selected("false")
    )
    assert result == expected
    assert calls == [expected]


@pytest.mark.parametrize("condition", [None, 0, "", [], {}, set()])
def test_outer_statement_if_stages_false_exact_builtin_truthiness(condition) -> None:
    calls: list[str] = []
    result = _internal_frontend_if(
        condition,
        lambda: calls.append("then"),
        lambda: calls.append("else") or 7,
    )
    assert result is None
    assert calls == ["else"]


@pytest.mark.parametrize(
    ("value", "name", "expected"),
    [
        (1, "real", 1),
        ("abc", "upper", str.upper),
        ([], "append", list.append),
        ({}, "keys", dict.keys),
    ],
)
def test_runtime_lazy_exact_builtin_attribute_reads_are_allowed(
    value, name, expected
) -> None:
    with _dynamic_lazy_region():
        result = _internal_lazy_attribute(value, name)
    if callable(expected):
        assert result.__self__ is value
        assert result.__name__ == expected.__name__
    else:
        assert result == expected


def test_runtime_lazy_exact_builtin_missing_attribute_raises_naturally() -> None:
    with _dynamic_lazy_region(), pytest.raises(AttributeError):
        _internal_lazy_attribute([], "missing")


def test_exact_builtin_attribute_does_not_inspect_custom_elements() -> None:
    _ProtocolSpy.calls = 0
    value = [_ProtocolSpy()]
    with _dynamic_lazy_region():
        result = _internal_lazy_attribute(value, "append")
    assert result.__self__ is value
    assert _ProtocolSpy.calls == 0


def test_runtime_lazy_builtin_subclass_is_rejected_without_protocol_dispatch() -> None:
    class ListSubclass(list):
        def __getitem__(self, key):
            _ProtocolSpy.calls += 1
            return super().__getitem__(key)

    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript(ListSubclass([1]), 0)
    assert _ProtocolSpy.calls == 0


def test_runtime_lazy_slice_rejects_custom_bounds_without_index_dispatch() -> None:
    class IndexSpy:
        calls = 0

        def __index__(self):
            type(self).calls += 1
            return 0

    bound = IndexSpy()
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript([1], slice(bound))
    assert IndexSpy.calls == 0


def test_runtime_lazy_dsl_subscript_rejects_tuple_subclass_without_iteration(
    monkeypatch,
) -> None:
    class TupleSpy(tuple):
        calls = 0

        def __iter__(self):
            type(self).calls += 1
            return super().__iter__()

    class TrustedValue:
        def __getitem__(self, index):
            tuple(index)
            return 1

    value = TrustedValue()
    monkeypatch.setattr(
        ast_decorators,
        "_is_trusted_dsl_value",
        lambda candidate: candidate is value,
    )
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript(value, TupleSpy((0,)))
    assert TupleSpy.calls == 0


def _transform_source(source: str, globals_: dict[str, object]) -> str:
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    scope_facts = _scope_facts_for_transform(source, "<test>", target)
    plan = _FunctionAnalyzer(
        global_names={*globals_, *dir(__builtins__)},
        global_symbols=globals_,
        scope_facts=scope_facts,
    ).analyze(target)
    transformed = _FrontendControlFlowTransformer(
        globals_,
        source_text=source,
        root_plan=plan,
    ).visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def test_runtime_lazy_call_trust_uses_exact_identity_not_metadata() -> None:
    def forged():
        return True

    forged.__module__ = "catlass.core_api"
    forged.__name__ = "block_idx"
    rendered = _transform_source(
        "def kernel(p):\n    if p and forged():\n        pass\n", {"forged": forged}
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_nested_dynamic_if_write_keeps_post_scope_python_binding_guard() -> None:
    rendered = _transform_source(
        "def kernel(limit):\n"
        "    scale = 2\n"
        "    if limit > 0:\n"
        "        if limit > 1:\n"
        "            scale = 3\n"
        "    else:\n"
        "        scale = 4\n"
        "    tla.make_coord(scale, 0)\n",
        {"tla": catlass_module},
    )

    assert "__tladsl_internal_runtime_write_check__" in rendered


def test_dynamic_if_annotations_do_not_clear_python_binding_guard() -> None:
    rendered = _transform_source(
        "def kernel(limit):\n"
        "    scale = 2\n"
        "    if limit > 0:\n"
        "        scale: int\n"
        "    else:\n"
        "        scale: int\n"
        "    tla.make_coord(scale, 0)\n",
        {"tla": catlass_module},
    )

    assert "__tladsl_internal_runtime_write_check__" in rendered


def test_runtime_lazy_call_trust_ignores_module_namespace_injection(
    monkeypatch,
) -> None:
    def forged():
        return True

    forged.__module__ = "catlass.core_api"
    monkeypatch.setattr(core_api, "forged", forged, raising=False)
    rendered = _transform_source(
        "def kernel(p):\n    if p and core.forged():\n        pass\n",
        {"core": core_api},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_runtime_lazy_value_trust_ignores_module_namespace_injection(
    monkeypatch,
) -> None:
    class Forged:
        calls = 0

        def __bool__(self):
            type(self).calls += 1
            return True

    Forged.__module__ = "catlass.core_api"
    monkeypatch.setattr(core_api, "Forged", Forged, raising=False)
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_frontend_if_expr(Forged(), lambda: 1, lambda: 0)
    assert Forged.calls == 0


def test_global_call_resolution_does_not_dispatch_custom_member_mapping() -> None:
    class MemberSpy(dict):
        calls = 0

        def __contains__(self, key):
            type(self).calls += 1
            return super().__contains__(key)

        def __getitem__(self, key):
            type(self).calls += 1
            return super().__getitem__(key)

    class Owner:
        _members = MemberSpy(block=core_api.arch.block_idx)

    rendered = _transform_source(
        "def kernel(p):\n    if p and owner.block():\n        pass\n",
        {"owner": Owner()},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered
    assert MemberSpy.calls == 0


@pytest.mark.parametrize("name", ["bool", "any", "all", "min", "max"])
def test_redirected_builtin_is_rejected_only_in_runtime_lazy_operand(name: str) -> None:
    rendered = _transform_source(
        f"def kernel(p):\n    if p and {name}([p]):\n        pass\n", {}
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_exact_core_callable_global_alias_is_trusted() -> None:
    rendered = _transform_source(
        "def kernel(p):\n    if p and block() >= 0:\n        pass\n",
        {"block": core_api.arch.block_idx},
    )
    assert "__tladsl_internal_unknown_effect_call__" not in rendered


@pytest.mark.parametrize(
    "call",
    ["block(*values)", "block(**values)", "block(value, *more)"],
)
def test_trusted_dsl_call_with_argument_expansion_is_effect_unknown(call: str) -> None:
    rendered = _transform_source(
        f"def kernel(p, values, more):\n    if p and {call}:\n        pass\n",
        {"block": core_api.arch.block_idx},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


@pytest.mark.parametrize(
    "source",
    [
        "def kernel(p, block):\n    if p and block():\n        pass\n",
        "def kernel(p):\n    block = lambda: True\n    if p and block():\n        pass\n",
        "def kernel(p):\n    if p and block():\n        pass\n    block = lambda: True\n",
        "def kernel(p):\n    from somewhere import block\n    if p and block():\n        pass\n",
    ],
    ids=["parameter", "assignment", "assignment-after", "local-import"],
)
def test_local_or_free_name_shadows_trusted_global_callable(source: str) -> None:
    rendered = _transform_source(source, {"block": core_api.arch.block_idx})
    assert "__tladsl_internal_unknown_effect_call__" in rendered


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "if predicate and staged_call():\n    pass\n",
            "conditionally skipped boolean operand has trace-time effect unknown",
        ),
        (
            "if lower < value < staged_call():\n    pass\n",
            "conditionally skipped chained-comparison operand has trace-time effect unknown",
        ),
    ],
    ids=["bool-call", "compare-call"],
)
def test_dynamic_condition_defers_unknown_lazy_call_effects(
    source: str, message: str
) -> None:
    del message
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if predicate or owner.attribute:\n    pass\n",
        "if predicate and values[index]:\n    pass\n",
        "if lower < value < owner.attribute:\n    pass\n",
        "if lower < value < values[index]:\n    pass\n",
    ],
)
def test_dynamic_condition_allows_lazy_reads(source: str) -> None:
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if lower < value and value < upper:\n    pass\n",
        "if not predicate or fallback:\n    pass\n",
        "if staged_call() and predicate:\n    pass\n",
        "if lower < staged_call() < upper:\n    pass\n",
        "while lower < value <= upper:\n    pass\n",
    ],
    ids=[
        "pure-comparisons",
        "pure-booleans",
        "eager-simple-call",
        "eager-first-comparator-call",
        "while-pure-chain",
    ],
)
def test_dynamic_condition_allows_pure_expressions_and_eager_calls(
    source: str,
) -> None:
    construct = "while" if source.startswith("while") else "if"
    _validate_condition(source, construct=construct)


def test_dynamic_condition_defers_source_located_unknown_call_diagnostic() -> None:
    source = "if (\n    predicate\n    and staged_call()\n):\n    pass\n"
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if (bound := predicate):\n    pass\n",
    ],
    ids=["if-walrus"],
)
def test_frontend_transformer_validates_each_dynamic_condition(source: str) -> None:
    node = _parse_first_statement(source)
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="condition_kernel.py",
        source_text=source,
    )

    with pytest.raises(SyntaxError, match="dynamic Tla (if|while)"):
        transformer.visit(node)


def _parse_function(source: str) -> ast.FunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _root_facts(source: str, filename: str = "<test>"):
    return _root_function_scope_facts(source, filename, _parse_function(source))


def test_function_analyzer_owns_bindings_and_ordered_child_plans() -> None:
    function = _parse_function(
        "def kernel(limit, predicate):\n"
        "    state = 0\n"
        "    if predicate:\n"
        "        state = 1\n"
        "    with tla.vector():\n"
        "        for i in tla.range(limit):\n"
        "            state = state + i\n"
        "    consume(state)\n"
    )

    plan = _FunctionAnalyzer(
        global_names={"tla", "consume"},
        global_symbols={"tla": catlass_module},
    ).analyze(function)

    assert [(binding.name, binding.kind) for binding in plan.arguments] == [
        ("limit", "argument"),
        ("predicate", "argument"),
    ]
    assert [(binding.name, binding.kind) for binding in plan.local_bindings] == [
        ("state", "local"),
        ("i", "loop variable"),
    ]
    assert [child.construct_name for child in plan.child_plans] == [
        "if",
        "with",
        "for",
    ]
    assert plan.captures == ()


def test_function_analyzer_plans_only_runtime_and_outlined_blocks() -> None:
    function = _parse_function(
        "def kernel(predicate, limit, mutex):\n"
        "    if True:\n"
        "        pass\n"
        "    if tla.const_expr(True):\n"
        "        pass\n"
        "    if predicate:\n"
        "        pass\n"
        "    for _ in range(2):\n"
        "        pass\n"
        "    for _ in tla.range_constexpr(2):\n"
        "        pass\n"
        "    for i in tla.range(limit):\n"
        "        pass\n"
        "    with ordinary_context():\n"
        "        pass\n"
        "    with tla.mutex_guard(mutex):\n"
        "        pass\n"
        "    with tla.vector():\n"
        "        while predicate:\n"
        "            pass\n"
        "    with tla.vec.func():\n"
        "        pass\n"
    )

    plan = _FunctionAnalyzer(
        global_names={"tla", "range"},
        global_symbols={"tla": catlass_module, "range": range},
    ).analyze(function)

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("if", 6),
        ("for", 8),
        ("for", 12),
        ("with", 18),
        ("while", 19),
        ("with", 21),
    ]


def test_builtin_range_only_function_is_planned_and_transformed() -> None:
    source = "def kernel(limit):\n    for i in range(limit):\n        pass\n"
    function = _parse_function(source)
    plan = _FunctionAnalyzer(global_names={"range"}, global_symbols={}).analyze(
        function
    )

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("for", 2)
    ]
    assert _function_needs_frontend_transform(function, {})
    assert "__tladsl_internal_for__" in _transform_source(source, {})


def test_non_builtin_global_range_remains_python() -> None:
    def custom_range(limit):
        return (limit,)

    source = "def kernel(limit):\n    for i in range(limit):\n        pass\n"
    function = _parse_function(source)
    plan = _FunctionAnalyzer(
        global_names={"range"}, global_symbols={"range": custom_range}
    ).analyze(function)

    assert plan.child_plans == ()
    assert "__tladsl_internal_for__" not in _transform_source(
        source, {"range": custom_range}
    )


def test_function_analyzer_child_plans_follow_range_alias_rebinding() -> None:
    function = _parse_function(
        "def kernel(limit):\n"
        "    loop = device_range(limit)\n"
        "    for i in loop:\n"
        "        pass\n"
        "    for _ in range(1):\n"
        "        for nested in loop:\n"
        "            pass\n"
        "    loop = range(limit)\n"
        "    for j in loop:\n"
        "        pass\n"
        "    for k in dsl.range(limit):\n"
        "        pass\n"
        "    with dsl.vector():\n"
        "        pass\n"
    )

    plan = _FunctionAnalyzer(
        global_names={"device_range", "dsl", "range"},
        global_symbols={
            "device_range": core_api.range,
            "dsl": catlass_module,
            "range": range,
        },
    ).analyze(function)

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("for", 3),
        ("for", 5),
        ("for", 6),
        ("for", 9),
        ("for", 11),
        ("with", 13),
    ]


def test_function_analyzer_child_plans_respect_local_range_shadowing() -> None:
    function = _parse_function(
        "def kernel(tla, device_range, limit):\n"
        "    for i in device_range(limit):\n"
        "        pass\n"
        "    for j in tla.range(limit):\n"
        "        pass\n"
        "    with tla.vector():\n"
        "        pass\n"
        "    if tla.const_expr(True):\n"
        "        pass\n"
    )

    plan = _FunctionAnalyzer(
        global_names={"tla", "device_range"},
        global_symbols={
            "tla": catlass_module,
            "device_range": core_api.range,
        },
    ).analyze(function)

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("if", 8)
    ]


def test_frontend_transformer_respects_local_tla_module_shadowing() -> None:
    source = (
        "def kernel(tla, limit):\n"
        "    for i in tla.range(limit):\n"
        "        pass\n"
        "    with tla.vector():\n"
        "        pass\n"
    )

    rendered = _transform_source(source, {"tla": catlass_module})

    assert "__tladsl_internal_for__" not in rendered
    assert "__tladsl_internal_region__" not in rendered
    assert "for i in tla.range(limit)" in rendered
    assert "with tla.vector()" in rendered


def test_function_analyzer_masks_inherited_alias_with_runtime_loop_target() -> None:
    function = _parse_function(
        "def kernel(limit):\n"
        "    loop = device_range(limit)\n"
        "    for loop in device_range(limit):\n"
        "        for i in loop:\n"
        "            pass\n"
    )

    plan = _FunctionAnalyzer(
        global_names={"device_range"},
        global_symbols={"device_range": core_api.range},
    ).analyze(function)

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("for", 3)
    ]


def test_child_plan_and_transformer_track_alias_inside_static_branch() -> None:
    source = (
        "def kernel(limit):\n"
        "    if True:\n"
        "        loop = tla.range(limit)\n"
        "        for i in loop:\n"
        "            pass\n"
    )
    function = _parse_function(source)

    plan = _FunctionAnalyzer(
        global_names={"tla"},
        global_symbols={"tla": catlass_module},
    ).analyze(function)
    rendered = _transform_source(source, {"tla": catlass_module})

    assert [(child.construct_name, child.lineno) for child in plan.child_plans] == [
        ("for", 4)
    ]
    assert rendered.count("__tladsl_internal_for__") == 1


def test_static_sibling_assignment_shadows_dsl_name_function_wide() -> None:
    source = (
        "def kernel(limit):\n"
        "    if True:\n"
        "        device_range = range\n"
        "    else:\n"
        "        for i in device_range(limit):\n"
        "            pass\n"
    )
    function = _parse_function(source)
    globals_ = {
        "device_range": core_api.range,
        "range": range,
    }

    plan = _FunctionAnalyzer(
        global_names=set(globals_), global_symbols=globals_
    ).analyze(function)
    rendered = _transform_source(source, globals_)

    assert plan.child_plans == ()
    assert "__tladsl_internal_for__" not in rendered


@pytest.mark.parametrize(
    ("trusted_name", "source"),
    [
        (
            "tla",
            "def kernel(limit):\n"
            "    for i in tla.range(limit):\n"
            "        pass\n"
            "    tla = object()\n",
        ),
        (
            "device_range",
            "def kernel(limit):\n"
            "    for i in device_range(limit):\n"
            "        pass\n"
            "    device_range = range\n",
        ),
        (
            "const_expr",
            "def kernel():\n"
            "    if const_expr(True):\n"
            "        pass\n"
            "    const_expr = bool\n",
        ),
    ],
)
def test_late_local_shadow_is_function_wide_for_planner_and_transformer(
    trusted_name: str, source: str
) -> None:
    globals_ = {
        "tla": catlass_module,
        "device_range": core_api.range,
        "const_expr": catlass_module.const_expr,
    }
    plan = _FunctionAnalyzer(
        global_names=set(globals_),
        global_symbols=globals_,
        scope_facts=_root_facts(source),
    ).analyze(_parse_function(source))
    rendered = _transform_source(source, globals_)
    expected = "if" if trusted_name == "const_expr" else None
    assert [child.construct_name for child in plan.child_plans] == (
        [expected] if expected else []
    )
    if trusted_name == "const_expr":
        assert "__tladsl_internal_if__" in rendered
    else:
        assert "__tladsl_internal_for__" not in rendered


def test_sibling_branch_assignment_shadows_trusted_range_function_wide() -> None:
    source = (
        "def kernel(flag, limit):\n"
        "    if flag:\n"
        "        device_range = range\n"
        "    else:\n"
        "        for i in device_range(limit):\n"
        "            pass\n"
    )
    globals_ = {"device_range": core_api.range, "range": range}
    plan = _FunctionAnalyzer(
        global_names=set(globals_),
        global_symbols=globals_,
        scope_facts=_root_facts(source),
    ).analyze(_parse_function(source))
    rendered = _transform_source(source, globals_)
    assert [child.construct_name for child in plan.child_plans] == ["if"]
    assert "__tladsl_internal_for__" not in rendered


trusted_device_range = core_api.range


def _closure_shadow_factory():
    device_range = range

    def kernel(limit):
        for _ in device_range(limit):
            pass

    return kernel


def _global_declaration_kernel(limit):
    global trusted_device_range
    for _ in trusted_device_range(limit):
        pass


def _nonlocal_declaration_factory():
    state = 0

    def kernel(limit):
        nonlocal state
        for _ in trusted_device_range(limit):
            state += 1

    return kernel


def _descendant_only_closure_factory():
    state = object()

    def kernel():
        def observe():
            return state

        return observe

    return kernel


def _direct_and_descendant_closure_factory():
    state = object()

    def kernel():
        state

        def observe():
            return state

        return observe

    return kernel


def _branch_closure_factory():
    captured = "captured"

    def kernel(flag):
        result = "fallback"
        if flag:
            result = captured
        return result

    return kernel


def _multiple_closure_factory():
    zebra = "last"
    alpha = "first"

    def kernel(flag):
        result = (alpha, zebra)
        if flag:
            result = (zebra, alpha)
        return result

    return kernel


def _live_closure_factory():
    captured = "before"

    def kernel(flag):
        result = "fallback"
        if flag:
            result = captured
        return result

    def update(value):
        nonlocal captured
        captured = value

    return kernel, update


def _empty_closure_factory():
    captured = "deleted"

    def kernel(flag):
        result = "fallback"
        if flag:
            result = captured  # noqa: F821 - deliberately exercise an empty cell
        return result

    del captured
    return kernel


_closure_default_evaluations = 0


def _record_closure_default():
    global _closure_default_evaluations
    _closure_default_evaluations += 1
    return object()


def _closure_with_default_factory():
    captured = "captured"

    def kernel(flag, marker=_record_closure_default()):
        result = marker
        if flag:
            result = captured
        return result

    return kernel


def _generic_metadata_kernel(value):
    result = None
    if value:
        result = value
    return result


def _generic_metadata_closure_factory():
    captured = "captured"

    def kernel(value):
        result = value
        if value:
            result = captured
        return result

    return kernel


def _analyze_extracted_function(fn) -> object:
    source_lines, _ = inspect.getsourcelines(fn)
    source = textwrap.dedent("".join(source_lines))
    module = ast.parse(source)
    target = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
    )
    scope_facts = _root_function_scope_facts(
        source, inspect.getsourcefile(fn) or "<test>", target
    )
    return _FunctionAnalyzer(
        global_names=set(fn.__globals__),
        global_symbols=fn.__globals__,
        scope_facts=scope_facts,
        root_freevars=set(fn.__code__.co_freevars),
    ).analyze(target)


def test_root_code_freevars_shadow_same_spelled_trusted_global() -> None:
    kernel = _closure_shadow_factory()
    global_symbols = {**kernel.__globals__, "device_range": core_api.range}
    source = (
        "def kernel(limit):\n"
        "    for _ in device_range(limit):\n"
        "        pass\n"
    )
    plan = _FunctionAnalyzer(
        global_names=set(global_symbols),
        global_symbols=global_symbols,
        scope_facts=_root_facts(source),
        root_freevars=set(kernel.__code__.co_freevars),
    ).analyze(_parse_function(source))
    rewritten = _maybe_transform(kernel)

    assert [binding.name for binding in plan.captures] == ["device_range"]
    assert rewritten is kernel


@pytest.mark.parametrize(
    "factory",
    [_descendant_only_closure_factory, _direct_and_descendant_closure_factory],
)
def test_root_plan_captures_closure_used_only_by_unsupported_nested_function(
    factory,
) -> None:
    plan = _analyze_extracted_function(factory())

    assert [binding.name for binding in plan.captures] == ["state"]
    assert plan.resolve("state") is plan.captures[0]
    assert not hasattr(plan, "nested_definitions")


def test_transformed_kernel_preserves_closure_across_both_branches() -> None:
    rewritten = _transform_with_eager_if(_branch_closure_factory())

    assert rewritten(False) == "fallback"
    assert rewritten(True) == "captured"


def test_transformed_kernel_maps_multiple_closure_cells_by_name() -> None:
    original = _multiple_closure_factory()
    rewritten = _transform_with_eager_if(original)

    assert rewritten.__code__.co_freevars == original.__code__.co_freevars
    assert rewritten(True) == ("last", "first")
    assert rewritten(False) == ("first", "last")


def test_transformed_kernel_reuses_live_closure_cells() -> None:
    original, update = _live_closure_factory()
    rewritten = _transform_with_eager_if(original)

    assert rewritten.__closure__ == original.__closure__
    assert rewritten(True) == "before"
    update("after")
    assert rewritten(True) == "after"


def test_transformed_kernel_preserves_empty_closure_cell_behavior() -> None:
    original = _empty_closure_factory()
    rewritten = _transform_with_eager_if(original)

    assert rewritten(False) == "fallback"
    with pytest.raises(NameError) as original_error:
        original(True)
    with pytest.raises(type(original_error.value), match="captured"):
        rewritten(True)


def test_transformed_closure_does_not_repeat_definition_time_defaults() -> None:
    original = _closure_with_default_factory()
    evaluations_before_transform = _closure_default_evaluations

    rewritten = _transform_with_eager_if(original)

    assert _closure_default_evaluations == evaluations_before_transform
    assert rewritten.__defaults__ == original.__defaults__
    assert rewritten(False) is original.__defaults__[0]
    assert rewritten(True) == "captured"


@pytest.mark.parametrize(
    "original", [_generic_metadata_kernel, _generic_metadata_closure_factory()]
)
def test_transformed_kernel_preserves_generic_function_metadata(
    original, monkeypatch: pytest.MonkeyPatch
) -> None:
    type_parameter = object()
    type_params = (type_parameter,)
    monkeypatch.setattr(original, "__type_params__", type_params, raising=False)
    monkeypatch.setattr(
        original,
        "__annotations__",
        {"value": type_parameter, "return": type_parameter},
    )

    rewritten = _transform_with_eager_if(original)

    assert rewritten.__type_params__ is type_params
    assert rewritten.__annotations__ == original.__annotations__
    assert rewritten.__annotations__["value"] is rewritten.__type_params__[0]


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 syntax requires Python 3.12+"
)
def test_transformed_real_generic_functions_preserve_type_parameters(
    tmp_path,
) -> None:
    module_path = tmp_path / "generic_kernels.py"
    module_path.write_text(
        "def plain[T](value: T) -> T:\n"
        "    result = None\n"
        "    if value:\n"
        "        result = value\n"
        "    return result\n"
        "\n"
        "def factory():\n"
        "    captured = 'captured'\n"
        "    def kernel[T](value: T) -> T:\n"
        "        result = value\n"
        "        if value:\n"
        "            result = captured\n"
        "        return result\n"
        "    return kernel\n"
    )
    spec = importlib.util.spec_from_file_location("generic_kernels", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for original in (module.plain, module.factory()):
        rewritten = _transform_with_eager_if(original)

        assert rewritten.__type_params__ is original.__type_params__
        assert rewritten.__annotations__ == original.__annotations__
        assert rewritten.__annotations__["value"] is rewritten.__type_params__[0]


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (_global_declaration_kernel, "global declarations"),
        (_nonlocal_declaration_factory(), "nonlocal declarations"),
    ],
)
def test_maybe_transform_rejects_scope_declaration_before_lexical_discovery(
    kernel, message: str
) -> None:
    with pytest.raises(SyntaxError, match=message) as caught:
        _ = _maybe_transform(kernel)

    assert caught.value.filename == __file__
    assert caught.value.text is not None


def test_comprehension_target_does_not_shadow_trusted_outer_name() -> None:
    source = (
        "def kernel(values, limit):\n"
        "    items = [device_range for device_range in values]\n"
        "    for i in device_range(limit):\n"
        "        pass\n"
    )
    rendered = _transform_source(source, {"device_range": core_api.range})
    assert rendered.count("__tladsl_internal_for__") == 1


def test_comprehension_walrus_owns_outer_trusted_name() -> None:
    source = (
        "def kernel(values, limit):\n"
        "    items = [(device_range := value) for value in values]\n"
        "    for i in device_range(limit):\n"
        "        pass\n"
    )
    rendered = _transform_source(
        source, {"device_range": core_api.range, "range": range}
    )
    assert "__tladsl_internal_for__" not in rendered


@pytest.mark.parametrize(
    ("kind", "source", "global_name"),
    [
        (
            "range",
            "def kernel(limit):\n    for i in forged(limit):\n        pass\n",
            "forged",
        ),
        (
            "range_constexpr",
            "def kernel():\n    for i in forged(2):\n        pass\n",
            "forged",
        ),
        (
            "const_expr",
            "def kernel():\n    if const_expr(True):\n        pass\n",
            "const_expr",
        ),
        (
            "module",
            "def kernel(limit):\n    for i in forged.range(limit):\n        pass\n",
            "forged",
        ),
    ],
)
def test_forged_metadata_is_not_a_trusted_dsl_identity(
    kind: str, source: str, global_name: str
) -> None:
    if kind == "module":
        forged: object = ModuleType("catlass")
    else:
        def forged_callable(*args, **kwargs):
            del args, kwargs
            return ()

        forged_callable.__module__ = (
            "catlass.runtime" if kind == "const_expr" else "catlass.core_api"
        )
        forged_callable.__name__ = kind
        forged = forged_callable
    globals_ = {global_name: forged}
    function = _parse_function(source)
    plan = _FunctionAnalyzer(
        global_names=set(globals_), global_symbols=globals_
    ).analyze(function)
    rendered = _transform_source(source, globals_)
    discovery = _function_needs_frontend_transform(function, globals_)

    if kind == "const_expr":
        assert [child.construct_name for child in plan.child_plans] == ["if"]
        assert discovery
        assert "__tladsl_internal_cf_symbol_check__" not in rendered
        assert "__tladsl_internal_if__" in rendered
    else:
        assert plan.child_plans == ()
        assert not discovery
        assert "__tladsl_internal_for__" not in rendered


def test_control_flow_symbol_check_uses_exact_trusted_identities() -> None:
    for genuine in (
        catlass_module,
        core_api.range,
        core_api.range_constexpr,
        catlass_module.const_expr,
    ):
        _cf_symbol_check(genuine)

    forged_module = ModuleType("catlass")

    def forged_range():
        return None

    forged_range.__module__ = "catlass.core_api"
    forged_range.__name__ = "range"
    for forged in (forged_module, forged_range):
        with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
            _cf_symbol_check(forged)


def test_canonical_module_identity_is_frozen_at_registration(monkeypatch) -> None:
    genuine = catlass_module
    forged = ModuleType("catlass")
    monkeypatch.setitem(sys.modules, "catlass", forged)

    identities = _trusted_dsl_identities()
    assert identities.module is genuine
    assert identities.module is not forged

    genuine_rendered = _transform_source(
        "def kernel(limit):\n"
        "    for i in genuine.range(limit):\n"
        "        pass\n",
        {"genuine": genuine},
    )
    forged_rendered = _transform_source(
        "def kernel(limit):\n"
        "    for i in forged.range(limit):\n"
        "        pass\n",
        {"forged": forged},
    )
    assert "__tladsl_internal_for__" in genuine_rendered
    assert "__tladsl_internal_for__" not in forged_rendered


def test_callable_roles_are_frozen_against_metadata_mutation(monkeypatch) -> None:
    genuine_range = core_api.range
    genuine_range_constexpr = core_api.range_constexpr
    genuine_const_expr = catlass_module.const_expr
    monkeypatch.setattr(genuine_range, "__name__", "range_constexpr")
    monkeypatch.setattr(genuine_range_constexpr, "__name__", "range")
    monkeypatch.setattr(genuine_const_expr, "__module__", "forged.runtime")

    identities = _trusted_dsl_identities()
    assert identities.range_callable is genuine_range
    assert identities.range_constexpr_callable is genuine_range_constexpr
    assert identities.const_expr_callable is genuine_const_expr


@pytest.mark.parametrize(
    ("member", "source"),
    [
        ("range", "def kernel(n):\n    for i in tla.range(n):\n        pass\n"),
        (
            "range_constexpr",
            "def kernel():\n    for i in tla.range_constexpr(2):\n        pass\n",
        ),
        ("const_expr", "def kernel():\n    if tla.const_expr(True):\n        pass\n"),
        ("cube", "def kernel():\n    with tla.cube():\n        pass\n"),
        ("vector", "def kernel():\n    with tla.vector():\n        pass\n"),
        ("vec", "def kernel():\n    with tla.vec.func():\n        pass\n"),
        ("vec.func", "def kernel():\n    with tla.vec.func():\n        pass\n"),
    ],
)
def test_mutated_module_qualified_member_is_rejected_during_transform(
    monkeypatch, member: str, source: str
) -> None:
    def forged(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forged DSL member must not execute")

    if member == "vec.func":
        monkeypatch.setitem(core_api.vec._members, "func", forged)
    else:
        monkeypatch.setattr(catlass_module, member, forged)

    with pytest.raises(SyntaxError, match="genuine TLA Python member") as exc_info:
        _ = _transform_source(source, {"tla": catlass_module})
    assert exc_info.value.lineno == 2


@pytest.mark.parametrize(
    "source, expected_checks",
    [
        ("def kernel(n):\n    for i in tla.range(n):\n        pass\n", ["tla.range"]),
        (
            "def kernel():\n    for i in tla.range_constexpr(2):\n        pass\n",
            ["tla.range_constexpr"],
        ),
        ("def kernel():\n    if tla.const_expr(True):\n        pass\n", ["tla.const_expr"]),
        ("def kernel():\n    with tla.cube():\n        pass\n", ["tla.cube"]),
        ("def kernel():\n    with tla.vector():\n        pass\n", ["tla.vector"]),
        (
            "def kernel():\n    with tla.vec.func():\n        pass\n",
            ["tla.vec.func"],
        ),
    ],
)
def test_module_qualified_lowering_guards_exact_members(
    source: str, expected_checks: list[str]
) -> None:
    rendered = _transform_source(source, {"tla": catlass_module})
    for member_expression in expected_checks:
        path = member_expression.removeprefix("tla.")
        assert f"__tladsl_internal_checked_dsl_member__(tla, '{path}')" in rendered
    assert "__tladsl_internal_cf_symbol_check__" not in rendered


@pytest.mark.parametrize(
    "expression",
    [
        "tla.const_expr(True)",
        "not tla.const_expr(False)",
        "tla.const_expr(1) == tla.const_expr(1)",
        "tla.const_expr(True) and (tla.const_expr(False) or tla.const_expr(True))",
    ],
)
def test_module_member_resolver_is_at_each_nested_call_evaluation_point(
    expression: str,
) -> None:
    rendered = _transform_source(
        f"def kernel():\n    if {expression}:\n        pass\n",
        {"tla": catlass_module},
    )
    assert rendered.count("__tladsl_internal_checked_dsl_member__(tla, 'const_expr')") == expression.count("tla.const_expr")


def test_constexpr_while_warning_runs_before_user_body() -> None:
    rendered = _transform_source(
        "def kernel(limit):\n"
        "    i = 0\n"
        "    while tla.const_expr(i < limit):\n"
        "        i += 1\n"
        "        continue\n",
        {"tla": catlass_module},
    )

    tree = ast.parse(rendered)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    loop = next(statement for statement in function.body if isinstance(statement, ast.While))
    assert isinstance(loop.body[0], ast.AugAssign)
    assert isinstance(loop.body[1], ast.Expr)
    assert "__tladsl_internal_static_loop_iteration__" in ast.unparse(loop.body[1])
    assert isinstance(loop.body[-1], ast.Continue)


def _guarded_module_order(argument):
    for _ in catlass_module.range(argument()):
        pass


def _guarded_module_explicit_step(start, stop, step):
    for _ in catlass_module.range(start(), stop(), step()):
        pass


def _unreachable_guarded_module_calls(false_prefix=False, true_prefix=True):
    if false_prefix and catlass_module.const_expr(True):
        pass
    if true_prefix or catlass_module.const_expr(False):
        pass
    return "ok"


def _unreachable_guarded_module_while_calls(
    false_prefix=False, true_prefix=True
):
    while false_prefix and catlass_module.const_expr(True):
        pass
    iterations = 0
    while iterations < 1 and (
        true_prefix or catlass_module.const_expr(False)
    ):
        iterations += 1
    return "ok"


def test_checked_module_member_preserves_callable_before_arguments(monkeypatch) -> None:
    transformed = _maybe_transform(_guarded_module_order)
    argument_calls = 0

    def argument():
        nonlocal argument_calls
        argument_calls += 1
        raise AssertionError("argument must not run before callable validation")

    monkeypatch.setattr(catlass_module, "range", lambda *_: ())
    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(argument)
    assert argument_calls == 0


@pytest.mark.parametrize("step_value", [1, -1])
def test_explicit_step_checks_callable_before_lifted_arguments(
    monkeypatch, step_value: int
) -> None:
    transformed = _maybe_transform(_guarded_module_explicit_step)
    argument_calls = 0

    def argument(value):
        def evaluate():
            nonlocal argument_calls
            argument_calls += 1
            return value

        return evaluate

    monkeypatch.setattr(catlass_module, "range", core_api.range_constexpr)
    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(argument(0), argument(2), argument(step_value))
    assert argument_calls == 0


def test_checked_module_member_preserves_native_short_circuit(monkeypatch) -> None:
    transformed = _maybe_transform(_unreachable_guarded_module_calls)
    monkeypatch.setattr(
        catlass_module,
        "const_expr",
        lambda *_: (_ for _ in ()).throw(AssertionError("must remain unreachable")),
    )
    assert transformed() == "ok"


def test_pre_mutated_unreachable_module_members_remain_unresolved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(catlass_module, "const_expr", core_api.range)
    transformed = _maybe_transform(_unreachable_guarded_module_calls)
    assert transformed() == "ok"


def test_pre_mutated_unreachable_while_members_are_not_eagerly_resolved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(catlass_module, "const_expr", core_api.range)
    source = inspect.getsource(_unreachable_guarded_module_while_calls)
    rendered = _transform_source(source, {"catlass_module": catlass_module})
    assert "__tladsl_internal_checked_dsl_member__" in rendered


@pytest.mark.parametrize(
    ("member", "replacement", "source"),
    [
        ("range", core_api.range_constexpr, "def kernel(n):\n    for _ in tla.range(n):\n        pass\n"),
        ("range_constexpr", core_api.range, "def kernel():\n    for _ in tla.range_constexpr(1):\n        pass\n"),
        ("const_expr", core_api.range, "def kernel():\n    if tla.const_expr(True):\n        pass\n"),
        ("cube", catlass_module.vector, "def kernel():\n    with tla.cube():\n        pass\n"),
        ("vector", catlass_module.cube, "def kernel():\n    with tla.vector():\n        pass\n"),
    ],
)
def test_module_member_cross_role_swap_is_rejected_before_transform(
    monkeypatch, member: str, replacement, source: str
) -> None:
    monkeypatch.setattr(catlass_module, member, replacement)
    with pytest.raises(SyntaxError, match="genuine TLA Python member"):
        _ = _transform_source(source, {"tla": catlass_module})


def _guarded_module_range(n):
    for _ in catlass_module.range(n):
        pass


def _guarded_module_range_constexpr():
    for _ in catlass_module.range_constexpr(1):
        pass


def _guarded_module_const_expr():
    if catlass_module.const_expr(True):
        pass


def _guarded_module_cube():
    with catlass_module.cube():
        pass


def _guarded_module_vector():
    with catlass_module.vector():
        pass


def _guarded_module_vec_func():
    with catlass_module.vec.func():
        pass


def _guarded_module_vec_func_order(argument):
    with catlass_module.vec.func(mode=argument()):
        pass


direct_range_alias = core_api.range
direct_range_constexpr_alias = core_api.range_constexpr
direct_const_expr_alias = catlass_module.const_expr


def _guarded_direct_range_alias(n):
    for _ in direct_range_alias(n):
        pass


def _guarded_direct_range_constexpr_alias():
    for _ in direct_range_constexpr_alias(1):
        pass


def _guarded_direct_const_expr_alias():
    if direct_const_expr_alias(True):
        pass


@pytest.mark.parametrize(
    ("member", "function", "args"),
    [
        ("range", _guarded_module_range, (1,)),
        ("range_constexpr", _guarded_module_range_constexpr, ()),
        ("const_expr", _guarded_module_const_expr, ()),
        ("cube", _guarded_module_cube, ()),
        ("vector", _guarded_module_vector, ()),
        ("vec", _guarded_module_vec_func, ()),
        ("vec.func", _guarded_module_vec_func, ()),
    ],
)
def test_transformed_function_rejects_later_module_member_mutation(
    monkeypatch, member: str, function, args: tuple[object, ...]
) -> None:
    transformed = _maybe_transform(function)
    calls = 0

    def forged(*forged_args, **forged_kwargs):
        nonlocal calls
        del forged_args, forged_kwargs
        calls += 1
        raise AssertionError("forged DSL member must not execute")

    if member == "vec.func":
        monkeypatch.setitem(core_api.vec._members, "func", forged)
    else:
        monkeypatch.setattr(catlass_module, member, forged)

    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(*args)
    assert calls == 0


@pytest.mark.parametrize(
    ("member", "replacement", "function", "args"),
    [
        ("range", core_api.range_constexpr, _guarded_module_range, (1,)),
        ("range_constexpr", core_api.range, _guarded_module_range_constexpr, ()),
        ("const_expr", core_api.range, _guarded_module_const_expr, ()),
        ("cube", catlass_module.vector, _guarded_module_cube, ()),
        ("vector", catlass_module.cube, _guarded_module_vector, ()),
    ],
)
def test_transformed_function_rejects_module_member_cross_role_swap(
    monkeypatch, member: str, replacement, function, args: tuple[object, ...]
) -> None:
    transformed = _maybe_transform(function)
    monkeypatch.setattr(catlass_module, member, replacement)
    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(*args)


@pytest.mark.parametrize(
    ("name", "replacement", "function", "args"),
    [
        ("direct_range_alias", core_api.range_constexpr, _guarded_direct_range_alias, (1,)),
        ("direct_range_constexpr_alias", core_api.range, _guarded_direct_range_constexpr_alias, ()),
        ("direct_const_expr_alias", core_api.range, _guarded_direct_const_expr_alias, ()),
    ],
)
def test_transformed_direct_alias_rejects_cross_role_rebinding(
    name: str, replacement, function, args: tuple[object, ...]
) -> None:
    transformed = _maybe_transform(function)
    transformed.__globals__[name] = replacement
    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(*args)


def test_transformed_function_rejects_later_module_alias_rebinding(monkeypatch) -> None:
    transformed = _maybe_transform(_guarded_module_range)
    forged_module = ModuleType("catlass")
    forged_module.range = core_api.range
    monkeypatch.setitem(transformed.__globals__, "catlass_module", forged_module)

    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(1)


def test_region_member_validation_precedes_mode_evaluation(monkeypatch) -> None:
    transformed = _maybe_transform(_guarded_module_vec_func_order)
    argument_calls = 0

    def argument():
        nonlocal argument_calls
        argument_calls += 1
        raise AssertionError("mode must not run before callable validation")

    monkeypatch.setitem(core_api.vec._members, "func", lambda **_: None)
    with pytest.raises(RuntimeError, match="Please use the TLA Python symbol"):
        transformed(argument)
    assert argument_calls == 0


def test_function_analyzer_uses_symtable_ownership_with_ast_binding_metadata() -> None:
    source = (
        "def kernel(subject, values):\n"
        "    import package as imported\n"
        "    from module import member as imported_member\n"
        "    try:\n"
        "        pass\n"
        "    except Error as caught:\n"
        "        pass\n"
        "    match subject:\n"
        "        case {'item': captured, **rest}:\n"
        "            pass\n"
        "    annotation_only: int\n"
        "    del deleted\n"
        "    items = [(walrus := value) for value in values]\n"
    )
    function = _parse_function(source)

    plan = _FunctionAnalyzer(scope_facts=_root_facts(source)).analyze(function)

    assert [(binding.name, binding.kind) for binding in plan.local_bindings] == [
        ("imported", "import"),
        ("imported_member", "import"),
        ("caught", "exception variable"),
        ("captured", "pattern variable"),
        ("rest", "pattern variable"),
        ("annotation_only", "local"),
        ("deleted", "local"),
        ("items", "local"),
        ("walrus", "local"),
    ]
    assert plan.resolve("value") is None


def test_symtable_filters_global_store_from_function_locals() -> None:
    source = "def kernel():\n    global state\n    state = 1\n"

    plan = _FunctionAnalyzer(scope_facts=_root_facts(source)).analyze(
        _parse_function(source)
    )

    assert plan.local_bindings == ()


def test_unsupported_nested_function_does_not_change_root_capture_facts() -> None:
    source = (
        "def kernel(state):\n"
        "    def observe():\n"
        "        return state, external\n"
    )
    plan = _FunctionAnalyzer(
        global_names={"state", "external"},
        scope_facts=_root_facts(source),
    ).analyze(_parse_function(source))

    assert plan.resolve("state") is plan.arguments[0]
    assert plan.resolve("observe") is plan.local_bindings[0]
    assert plan.captures == ()
    assert not hasattr(plan, "nested_definitions")


@pytest.mark.parametrize(
    "source",
    [
        "def other():\n    pass\n",
        "def kernel():\n    pass\ndef kernel(value):\n    pass\n",
    ],
)
def test_root_symtable_lookup_mismatch_fails_internally(source: str) -> None:
    target = _parse_function("def kernel():\n    pass\n")

    with pytest.raises(RuntimeError, match="exactly one root symbol table"):
        _root_function_scope_facts(source, "<test>", target)


def test_root_symtable_selects_namespace_bound_to_function_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSymbol:
        def __init__(self, namespaces: tuple[FakeTable, ...]) -> None:
            self._namespaces = namespaces

        def get_namespaces(self) -> tuple[FakeTable, ...]:
            return self._namespaces

        def is_local(self) -> bool:
            return True

        def is_parameter(self) -> bool:
            return True

        def is_free(self) -> bool:
            return False

        def is_declared_global(self) -> bool:
            return False

    class FakeTable:
        def __init__(
            self,
            name: str,
            kind: str,
            *,
            namespaces: tuple[FakeTable, ...] = (),
        ) -> None:
            self._name = name
            self._kind = kind
            self._namespaces = namespaces

        def get_type(self) -> str:
            return self._kind

        def get_name(self) -> str:
            return self._name

        def get_lineno(self) -> int:
            return 1

        def get_children(self) -> tuple[FakeTable, ...]:
            raise AssertionError("function lookup must not scan sibling tables")

        def get_identifiers(self) -> tuple[str, ...]:
            return ("value",)

        def get_parameters(self) -> tuple[str, ...]:
            return ("value",)

        def lookup(self, name: str) -> FakeSymbol:
            if self._kind == "module":
                assert name == "kernel"
                return FakeSymbol(self._namespaces)
            assert name == "value"
            return FakeSymbol(())

    function_table = FakeTable("kernel", "function")
    module = FakeTable("top", "module", namespaces=(function_table,))
    monkeypatch.setattr("symtable.symtable", lambda *args: module)
    source = "def kernel(value):\n    return value\n"

    facts = _root_function_scope_facts(
        source, "<test>", _parse_function(source)
    )

    assert facts.local_names == frozenset({"value"})


@pytest.mark.parametrize(
    ("function_name", "expression"),
    [
        ("genexpr", "(value for value in values)"),
        ("listcomp", "[value for value in values]"),
        ("setcomp", "{value for value in values}"),
        ("dictcomp", "{value: value for value in values}"),
    ],
)
def test_user_function_table_does_not_collide_with_implicit_scope_on_same_line(
    function_name: str, expression: str
) -> None:
    source = f"def {function_name}(values): return {expression}\n"
    function = _parse_function(source)

    facts = _root_function_scope_facts(source, "<test>", function)

    assert facts.local_names == frozenset({"values"})
    assert facts.free_names == frozenset()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def kernel(values): value = 0; "
            "return [value for value in values]\n",
            {"value", "values"},
        ),
        (
            "def kernel(value, values): return [value for value in values]\n",
            {"value", "values"},
        ),
        (
            "def kernel(values): "
            "return [(outer := value) for value in values]\n",
            {"outer", "values"},
        ),
    ],
)
def test_comprehension_keeps_genuine_containing_scope_bindings(
    source: str, expected: set[str]
) -> None:
    function = _parse_function(source)

    facts = _root_function_scope_facts(source, "<test>", function)

    assert facts.local_names == frozenset(expected)


@pytest.mark.parametrize(
    "source",
    [
        "def genexpr(x=(y for y in z)): return x\n",
        "def genexpr(*, x=(y for y in z)): return x\n",
        "def genexpr(x, /, *args, y=(item for item in z), **kwargs): "
        "return x\n",
        "def genexpr() -> (y for y in z): return None\n",
    ],
)
def test_user_function_table_does_not_collide_with_implicit_signature_scope(
    source: str,
) -> None:
    function = _parse_function(source)

    facts = _root_function_scope_facts(source, "<test>", function)

    expected = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    if function.args.vararg is not None:
        expected.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        expected.add(function.args.kwarg.arg)
    assert facts.local_names == frozenset(expected)
    assert facts.free_names == frozenset()


@pytest.mark.parametrize(
    ("function_name", "expression"),
    [
        ("genexpr", "(value for value in values)"),
        ("listcomp", "[value for value in values]"),
        ("setcomp", "{value for value in values}"),
        ("dictcomp", "{value: value for value in values}"),
    ],
)
def test_all_implicit_signature_scopes_are_excluded_from_function_namespace(
    function_name: str, expression: str
) -> None:
    source = f"def {function_name}(result={expression}): return result\n"
    function = _parse_function(source)

    facts = _root_function_scope_facts(source, "<test>", function)

    assert facts.local_names == frozenset({"result"})
    assert facts.free_names == frozenset()


def test_root_symtable_unwraps_python313_type_parameters_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSymbol:
        def __init__(self, namespaces: tuple[FakeTable, ...] = ()) -> None:
            self._namespaces = namespaces

        def get_namespaces(self) -> tuple[FakeTable, ...]:
            return self._namespaces

        def is_local(self) -> bool:
            return True

        def is_free(self) -> bool:
            return False

        def is_declared_global(self) -> bool:
            return False

    class FakeTable:
        def __init__(
            self,
            name: str,
            kind: str,
            *,
            children: tuple["FakeTable", ...] = (),
        ) -> None:
            self._name = name
            self._kind = kind
            self._children = children

        def get_type(self) -> str:
            return self._kind

        def get_name(self) -> str:
            return self._name

        def get_lineno(self) -> int:
            return 1

        def get_children(self) -> tuple["FakeTable", ...]:
            return self._children

        def get_parameters(self) -> tuple[str, ...]:
            return ("value",)

        def get_identifiers(self) -> tuple[str, ...]:
            return ("value",)

        def lookup(self, name: str) -> FakeSymbol:
            if self._kind == "module":
                assert name == "kernel"
                return FakeSymbol(self._children)
            assert name == "value"
            return FakeSymbol()

    function_table = FakeTable("kernel", "function")
    type_parameters = FakeTable(
        "kernel", "type parameters", children=(function_table,)
    )
    module = FakeTable("top", "module", children=(type_parameters,))
    monkeypatch.setattr("symtable.symtable", lambda *args: module)
    source = "def kernel(value):\n    return value\n"
    function = _parse_function(source)

    facts = _root_function_scope_facts(source, "<test>", function)

    assert facts.local_names == frozenset({"value"})


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 syntax requires Python 3.12+"
)
def test_root_symtable_matches_generic_function_without_planning_nested() -> None:
    source = (
        "def kernel[T](value: T):\n"
        "    def nested[U](other: U):\n"
        "        return value, other\n"
        "    return nested\n"
    )
    module = ast.parse(source)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    root_facts = _root_function_scope_facts(source, "<test>", function)

    assert root_facts.local_names == frozenset({"value", "nested"})


def test_function_analyzer_is_compatible_without_ast_try_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(ast, "TryStar", raising=False)
    function = _parse_function("def kernel(value):\n    return value\n")

    plan = _FunctionAnalyzer().analyze(function)

    assert [binding.name for binding in plan.arguments] == ["value"]


def test_function_analyzer_does_not_plan_unsupported_nested_definition() -> None:
    function = _parse_function(
        "def kernel(limit):\n"
        "    state = 0\n"
        "    def observe():\n"
        "        state = limit\n"
        "        consume(state)\n"
        "    state = state + 1\n"
    )

    plan = _FunctionAnalyzer(global_names={"consume"}).analyze(function)

    assert [binding.name for binding in plan.local_bindings] == ["state", "observe"]
    assert not hasattr(plan, "nested_definitions")


def test_function_analyzer_keeps_comprehension_bindings_implicit() -> None:
    function = _parse_function(
        "def kernel(rows, columns, item):\n"
        "    values = [item + column for item in rows "
        "for column in columns[item] if column != item]\n"
        "    consume(item, values)\n"
    )

    plan = _FunctionAnalyzer(global_names={"consume"}).analyze(function)

    assert [binding.name for binding in plan.local_bindings] == ["values"]
    assert [binding.name for binding in plan.captures] == []


def test_function_analyzer_comprehension_generators_bind_sequentially() -> None:
    function = _parse_function(
        "def kernel(rows):\n"
        "    values = [(row, column) for row in rows for column in row]\n"
    )

    plan = _FunctionAnalyzer().analyze(function)

    assert [binding.name for binding in plan.local_bindings] == ["values"]
    assert plan.captures == ()


def test_control_flow_analyzer_does_not_carry_comprehension_target() -> None:
    function = _parse_function(
        "def kernel(predicate, i, rows):\n"
        "    values = []\n"
        "    if predicate:\n"
        "        values = [(i, column) for i in rows for column in i]\n"
    )
    function_plan = _FunctionAnalyzer().analyze(function)
    node = function.body[1]
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "i", "rows", "values"},
        active_callables=set(),
        function_plan=function_plan,
    )

    assert plan.assigned_names == frozenset({"values"})
    assert plan.carried_names == ("values",)


def test_compile_time_for_destructuring_activates_every_target() -> None:
    source = (
        "def kernel(predicate, pairs):\n"
        "    for (left, (right, *rest)) in pairs:\n"
        "        if predicate:\n"
        "            left = left + right\n"
        "            rest = rest\n"
    )
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="destructuring_kernel.py",
        source_text=source,
        root_plan=_FunctionAnalyzer().analyze(target),
    )

    transformed = transformer.visit(tree)

    assert isinstance(transformed, ast.Module)


def test_function_analyzer_does_not_plan_nested_async_definition() -> None:
    function = _parse_function(
        "def kernel(limit):\n"
        "    async def observe():\n"
        "        consume(limit)\n"
    )

    plan = _FunctionAnalyzer(global_names={"consume"}).analyze(function)

    assert [binding.name for binding in plan.local_bindings] == ["observe"]
    assert not hasattr(plan, "nested_definitions")


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("global state", "global declarations"),
        ("nonlocal state", "nonlocal declarations"),
    ],
)
def test_frontend_rejects_source_scope_declarations_with_location(
    declaration: str, message: str
) -> None:
    source = f"def kernel():\n    {declaration}\n"
    root_plan = FunctionPlan("kernel", "kernel@1:0", (), (), (), ())
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="scope_kernel.py",
        line_offset=20,
        source_text=source,
        root_plan=root_plan,
    )

    with pytest.raises(SyntaxError, match=message) as caught:
        transformer.visit(ast.parse(source))

    assert caught.value.filename == "scope_kernel.py"
    assert caught.value.lineno == 22
    assert caught.value.offset == 5
    assert caught.value.text == declaration


def test_frontend_rejects_nested_async_helper_with_location() -> None:
    source = "def kernel():\n    async def helper():\n        pass\n"
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="async_kernel.py",
        line_offset=30,
        source_text=source,
        root_plan=_FunctionAnalyzer().analyze(target),
    )

    with pytest.raises(SyntaxError, match="nested async function.*helper") as caught:
        transformer.visit(tree)

    assert caught.value.filename == "async_kernel.py"
    assert caught.value.lineno == 32
    assert caught.value.offset == 5
    assert caught.value.text == "async def helper():"


@pytest.mark.parametrize(
    ("source", "expected_lineno"),
    [
        (
            "def kernel(pred):\n"
            "    class Helper:\n"
            "        if pred:\n"
            "            value = 1\n"
            "    return Helper\n",
            42,
        ),
        (
            "def kernel(pred, base):\n"
            "    if pred:\n"
            "        pass\n"
            "    class Helper((captured := base)):\n"
            "        pass\n"
            "    return Helper, captured\n",
            44,
        ),
    ],
)
def test_frontend_rejects_nested_class_with_location(
    source: str, expected_lineno: int
) -> None:
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    assert _function_needs_frontend_transform(target, {})
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="class_kernel.py",
        line_offset=40,
        source_text=source,
        root_plan=_FunctionAnalyzer().analyze(target),
    )

    with pytest.raises(SyntaxError, match="nested class definition 'Helper'") as caught:
        transformer.visit(tree)

    assert caught.value.filename == "class_kernel.py"
    assert caught.value.lineno == expected_lineno
    assert caught.value.offset == 5
    assert caught.value.text.startswith("class Helper")


def test_control_flow_state_uses_the_owning_function_binding() -> None:
    function = _parse_function(
        "def kernel(state):\n    if state:\n        state = state + 1\n"
    )
    function_plan = _FunctionAnalyzer().analyze(function)
    node = function.body[0]
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"state"},
        active_callables=set(),
        function_plan=function_plan,
    )

    state = function_plan.resolve("state")
    assert state is not None
    assert plan.assigned_bindings == frozenset({state})
    assert plan.carried_bindings == (state,)


@pytest.mark.parametrize(
    ("source", "kind", "regions"),
    [
        ("if predicate:\n    state = state + 1\n", "if", ("body", "orelse")),
        ("for i in tla.range(limit):\n    state = i\n", "for", ("body",)),
        ("while state < limit:\n    state = state + 1\n", "while", ("body",)),
    ],
)
def test_control_flow_analyzer_builds_immutable_plans(
    source: str, kind: str, regions: tuple[str, ...]
) -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, (ast.If, ast.For, ast.While))
    assigned_regions = [getattr(node, region) for region in regions]

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name=kind,
        assigned_regions=assigned_regions,
        active_call_nodes=[node],
        active_symbols={"predicate", "limit", "state"},
        active_callables=set(),
    )

    assert plan.construct_name == kind
    assert plan.assigned_names == frozenset({"state"})
    assert plan.carried_names == ("state",)
    with pytest.raises(FrozenInstanceError):
        plan.construct_name = "changed"  # type: ignore[misc]


def test_control_flow_plan_records_immutable_assigned_bindings() -> None:
    function = _parse_function(
        "def kernel(predicate):\n"
        "    scale = 2\n"
        "    if predicate:\n"
        "        scale = 3\n"
    )
    function_plan = _FunctionAnalyzer().analyze(function)
    node = function.body[1]
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "scale"},
        active_callables=set(),
        function_plan=function_plan,
    )

    scale = function_plan.resolve("scale")
    assert plan.assigned_bindings == frozenset({scale})
    assert plan.carried_bindings == (scale,)
    with pytest.raises(FrozenInstanceError):
        plan.assigned_bindings = frozenset()  # type: ignore[misc]


def test_control_flow_analyzer_classifies_tensor_store_as_side_effect() -> None:
    node = _parse_first_statement("if predicate:\n    out[index] = value\n")
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "out", "index", "value"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("tensor subscript",)
    assert len(plan.tensor_store_assignments) == 1
    assert plan.assigned_names == frozenset()
    assert plan.carried_names == ()


def test_control_flow_analyzer_collects_tensor_store_in_python_loop() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    for _ in (0,):\n"
        "        out[index] = value\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "out", "index", "value"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("_", "tensor subscript")
    assert len(plan.tensor_store_assignments) == 1


def test_control_flow_analyzer_records_nested_runtime_constructs() -> None:
    node = _parse_first_statement(
        "while state < limit:\n"
        "    if predicate:\n"
        "        state = state + 1\n"
        "    for i in tla.range(limit):\n"
        "        consume(i)\n"
    )
    assert isinstance(node, ast.While)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="while",
        assigned_regions=[node.body],
        active_call_nodes=[node.test, *node.body],
        active_symbols={"state", "limit", "predicate"},
        active_callables=set(),
    )

    assert [kind for kind, _, _ in plan.nested_constructs] == ["if", "for"]


@pytest.mark.parametrize(
    "nested_source",
    [
        "if True:\n    return\n",
        "for _ in range(1):\n    return\n",
        "while True:\n    return\n",
        "with context:\n    return\n",
        "try:\n    return\nexcept Exception:\n    pass\n",
        "match value:\n    case _:\n        return\n",
    ],
    ids=["if", "for", "while", "with", "try", "match"],
)
def test_control_flow_analyzer_recursively_rejects_nested_return(
    nested_source: str,
) -> None:
    indented = "".join(f"    {line}\n" for line in nested_source.splitlines())
    node = _parse_first_statement(f"if predicate:\n{indented}")
    assert isinstance(node, ast.If)

    with pytest.raises(
        SyntaxError, match="dynamic Tla if does not support return"
    ):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "context", "value"},
            active_callables=set(),
        )


def test_control_flow_analyzer_does_not_cross_nested_function_scope() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    def helper():\n"
        "        return\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate"},
        active_callables=set(),
    )

    assert plan.construct_name == "if"


def test_control_flow_analyzer_allows_python_loop_owned_exits() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    for item in values:\n"
        "        continue\n"
        "    for item in values:\n"
        "        break\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "values"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("item", "item")


@pytest.mark.parametrize(
    "source",
    [
        "if predicate:\n    for owner.attribute in values:\n        pass\n",
        "if predicate:\n    with context as owner.attribute:\n        pass\n",
    ],
    ids=["for-target", "with-target"],
)
def test_control_flow_analyzer_rejects_nonlocal_binding_targets(
    source: str,
) -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, ast.If)

    with pytest.raises(SyntaxError, match="only supports assignments to local names"):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "values", "context", "owner"},
            active_callables=set(),
        )


@pytest.mark.parametrize(
    ("nested_source", "message"),
    [
        (
            "try:\n    consume()\nexcept Exception:\n    recover()\n",
            "does not support try statements",
        ),
        (
            "match value:\n    case _:\n        consume()\n",
            "does not support match statements",
        ),
        ("item = (updated := value)\n", "does not support assignment expressions"),
    ],
    ids=["try", "match", "assignment-expression"],
)
def test_control_flow_analyzer_rejects_unsupported_nested_syntax(
    nested_source: str, message: str
) -> None:
    indented = "".join(f"    {line}\n" for line in nested_source.splitlines())
    node = _parse_first_statement(f"if predicate:\n{indented}")
    assert isinstance(node, ast.If)

    with pytest.raises(SyntaxError, match=message):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "value"},
            active_callables=set(),
        )

