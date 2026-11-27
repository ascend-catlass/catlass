from catlass.tla.runtime import make_fake_tensor

from dataclasses import dataclass
import inspect
import re
from typing import Any

import pytest

import catlass.tla as tla
import catlass.tla_ast_decorators as ast_decorators_mod
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass import _tla_type_bridge
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass._mlir import ir as mlir_ir


def _ensure_compute_order_or_skip() -> None:
    with mlir_ir.Context() as context:
        _tla_type_bridge.load_tla_dialect(context)
        try:
            mlir_ir.Attribute.parse("#tla.compute_order<M_FIRST>", context=context)
        except mlir_ir.MLIRError as error:
            if "compute_order" in str(error) and "expected valid keyword" in str(error):
                pytest.skip(
                    "linked BiShengIR lacks tla.compute_order enum used by mmad"
                )
            raise


@pytest.fixture(autouse=True)
def _skip_without_compute_order(request: pytest.FixtureRequest) -> None:
    name = request.node.name.lower()
    if "mmad" in name or ("cube" in name and "mutex" not in name):
        _ensure_compute_order_or_skip()


const_expr = tla.const_expr


def _operation_offsets_nested_in_scf_if(mlir: str, operation: str) -> list[bool]:
    """Report lexical scf.if nesting for every occurrence of an operation."""
    offsets: list[int] = []
    start = 0
    while True:
        offset = mlir.find(operation, start)
        if offset == -1:
            break
        offsets.append(offset)
        start = offset + len(operation)
    results: list[bool] = []
    for operation_offset in offsets:
        results.append(_offset_is_nested_in_scf_if(mlir, operation_offset))
    return results

def _offset_is_nested_in_scf_if(mlir: str, operation_offset: int) -> bool:
    stack: list[str] = []
    last_closed = ""
    for offset, token in enumerate(mlir[:operation_offset]):
        if token == "{":
            header_start = mlir.rfind("\n", 0, offset) + 1
            header = mlir[header_start:offset]
            if "scf.if" in header:
                kind = "scf.if"
            elif "else" in header and last_closed == "scf.if":
                kind = "scf.if"
            else:
                kind = "other"
            stack.append(kind)
            last_closed = ""
        elif token == "}" and stack:
            last_closed = stack.pop()
    return "scf.if" in stack

def _operation_is_nested_in_scf_if(mlir: str, operation: str) -> bool:
    occurrences = _operation_offsets_nested_in_scf_if(mlir, operation)
    return bool(occurrences) and all(occurrences)

@dataclass
class BranchState:
    coord: Any
    offset: Any

@dataclass(frozen=True)
class FrozenBranchState:
    coord: Any
    offset: Any

@dataclass
class OtherBranchState:
    coord: Any
    offset: Any

class CustomBranchState:
    def __init__(self, coord: Any, offset: Any) -> None:
        self.coord = coord
        self.offset = offset

    def __extract_mlir_values__(self) -> list[Any]:
        return [
            core_api_mod._as_branch_value(self.coord),
            core_api_mod._as_branch_value(self.offset),
        ]

    def __new_from_mlir_values__(self, values: list[Any]) -> "CustomBranchState":
        if len(values) != 2:
            raise ValueError(f"CustomBranchState expects 2 values, got {len(values)}")
        return CustomBranchState(
            core_api_mod._wrap_frontend_value(values[0]),
            core_api_mod._wrap_frontend_value(values[1]),
        )

@tla.jit
def _explode() -> None:
    raise AssertionError("unreachable")

@tla.kernel
def const_expr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)

@tla.kernel
def imported_const_expr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if const_expr(flag):
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)

@tla.kernel
def bad_const_expr_dynamic_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if tla.const_expr(i == 0):
            tla.make_coord(i, 0)

@tla.kernel
def static_literal_if_kernel() -> None:
    if True:
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)

@tla.kernel
def inline_if_static_true_lazy_kernel() -> None:
    coord = 1 if True else _explode()
    tla.make_coord(coord, 0)

@tla.kernel
def pointer_conditional_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        loop_range = tla.range(0, 2, 1)
        for i in loop_range:
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0 if i == 0 else ptr1
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.copy(local, tile)

@tla.kernel
def dynamic_mmad_init_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                tla.mmad(acc, lhs, rhs, init_c=True if outer == 0 and inner == 0 else False)

@tla.kernel
def dynamic_mmad_unit_flag_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                unit_flag = 0b11 if (outer == 1) and (inner == 1) else 0b10
                tla.mmad(acc, lhs, rhs, init_c=False, unit_flag=unit_flag)

@tla.kernel
def dynamic_mmad_initc_unit_flag_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 0b11 if (outer == 1) and (inner == 1) else 0b10
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)

@tla.kernel
def inline_if_tuple_result_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = (i, i + 1) if i == 0 else (i + 2, i + 3)
        tla.make_coord(coord[0], coord[1])

@tla.kernel
def inline_if_dataclass_result_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1) if i == 0 else BranchState(i + 2, i + 3)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def builtin_bool_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if bool(i == 0):
            tla.make_coord(i, 0)

@tla.kernel
def builtin_min_max_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        lower = min(i + 1, i + 2)
        upper = max([i, i + 3])
        tla.make_coord(lower, upper)

@tla.kernel
def builtin_name_redirect_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        choose = max
        pred = any
        coord = choose(i, i + 1)
        if pred((i == 0, i == 1)):
            tla.make_coord(coord, 0)

@tla.kernel
def bad_inline_if_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = (i, i + 1) if i == 0 else i + 2
        tla.make_coord(coord[0], coord[1])

@tla.kernel
def statement_if_side_effect_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        else:
            coord = i + 2
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_multiple_carried_values_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        offset = i + 1
        if i == 0:
            coord = i + 2
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset)

@tla.kernel
def statement_if_tuple_assignment_carried_values_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord, offset = i, i + 1
        if i == 0:
            coord, offset = i + 2, i + 3
        else:
            coord, offset = i + 4, i + 5
        tla.make_coord(coord, offset)

@tla.kernel
def statement_if_tuple_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (i + 4, i + 5)
        tla.make_coord(state[0], state[1])

@tla.kernel
def statement_if_list_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = [i, i + 1]
        if i == 0:
            state = [i + 2, i + 3]
        else:
            state = [i + 4, i + 5]
        tla.make_coord(state[0], state[1])

@tla.kernel
def statement_if_dict_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = {"coord": i, "offset": i + 1}
        if i == 0:
            state = {"coord": i + 2, "offset": i + 3}
        else:
            state = {"coord": i + 4, "offset": i + 5}
        tla.make_coord(state["coord"], state["offset"])

@tla.kernel
def statement_if_dataclass_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, (i + 1, i + 2))
        if i == 0:
            state = BranchState(i + 3, (i + 4, i + 5))
        else:
            state = BranchState(i + 6, (i + 7, i + 8))
        tla.make_coord(state.coord, state.offset[0])

@tla.kernel
def statement_if_frozen_dataclass_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = FrozenBranchState(i, i + 1)
        if i == 0:
            state = FrozenBranchState(i + 2, i + 3)
        else:
            state = FrozenBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def statement_if_custom_class_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(i, i + 1)
        if i == 0:
            state = CustomBranchState(i + 2, i + 3)
        else:
            state = CustomBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def loop_carried_index_kernel(limit: int) -> None:
    acc = tla.as_numeric(0)
    for i in tla.range(0, limit, 1):
        acc = acc + i
    tla.make_coord(acc, 0)

@tla.kernel
def loop_tuple_carried_state_kernel(limit: int) -> None:
    zero = tla.as_numeric(0)
    state = (zero, zero + 1)
    for i in tla.range(0, limit, 1):
        state = (state[0] + i, state[1] + i)
    tla.make_coord(state[0], state[1])

@tla.kernel
def loop_list_carried_state_kernel(limit: int) -> None:
    zero = tla.as_numeric(0)
    state = [zero, zero + 1]
    for i in tla.range(0, limit, 1):
        state = [state[0] + i, state[1] + i]
    tla.make_coord(state[0], state[1])

@tla.kernel
def loop_dict_carried_state_kernel(limit: int) -> None:
    zero = tla.as_numeric(0)
    state = {"coord": zero, "offset": zero + 1}
    for i in tla.range(0, limit, 1):
        state = {"coord": state["coord"] + i, "offset": state["offset"] + i}
    tla.make_coord(state["coord"], state["offset"])

@tla.kernel
def loop_dataclass_carried_state_kernel(limit: int) -> None:
    zero = tla.as_numeric(0)
    state = BranchState(zero, (zero + 1, zero + 2))
    for i in tla.range(0, limit, 1):
        state = BranchState(state.coord + i, (state.offset[0] + i, state.offset[1]))
    tla.make_coord(state.coord, state.offset[0])

@tla.kernel
def loop_custom_class_carried_state_kernel(limit: int) -> None:
    zero = tla.as_numeric(0)
    state = CustomBranchState(zero, zero + 1)
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(state.coord + i, state.offset + i)
    tla.make_coord(state.coord, state.offset)

@tla.kernel
def statement_if_implicit_else_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_bool_op_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and i != 1:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_guarded_division_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i != 0 and 10 // i > 2:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_guarded_division_or_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 or 10 // i > 2:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_multi_operand_lazy_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i != 0 and 10 // i > 2 and i < limit:
            tla.make_coord(i, 0)
        if i == 0 or 10 // i > 2 or i >= limit:
            tla.make_coord(i, 1)

@tla.kernel
def statement_if_chained_lazy_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i < 0 < 10 // i:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_dsl_call_lazy_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i < limit and tla.arch.block_idx() == 0:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_chained_dsl_call_kernel() -> None:
    if 0 <= tla.arch.block_idx() < 2:
        tla.make_coord(0, 0)

@tla.kernel
def bad_mask_predicate_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            if mask:
                tla.bitwise_not(mask)

@tla.kernel
def bad_mixed_lazy_branch_value_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        value = i == 0 and 7
        if value:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_bool_op_and_name_collision_kernel(limit: int) -> None:
    __tladsl_bool_op_lhs_1 = False
    if limit > 0 and __tladsl_bool_op_lhs_1:
        tla.make_coord(0, 0)

@tla.kernel
def statement_if_bool_op_or_name_collision_kernel(limit: int) -> None:
    __tladsl_bool_op_lhs_1 = True
    if limit < 0 or __tladsl_bool_op_lhs_1:
        tla.make_coord(0, 0)

@tla.jit
def _raise_if_evaluated() -> bool:
    raise AssertionError("short-circuited boolean RHS was evaluated")

@tla.jit
def _returns_false() -> bool:
    return False

@tla.jit
def _returns_true() -> bool:
    return True

def _record_comparator(events: list[str], name: str, value: int) -> int:
    events.append(name)
    return value

@tla.jit
def _staged_is_zero(value: Any) -> Any:
    return value == 0

def _undecorated_is_zero(value: Any) -> Any:
    return value == 0

class _ExpansionSpy:
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        return iter(())

    def keys(self):
        type(self).calls += 1
        return ()

    def __getitem__(self, key):
        type(self).calls += 1
        raise KeyError(key)

_star_expansion_spy = _ExpansionSpy()
_mapping_expansion_spy = _ExpansionSpy()

_forged_call_count = 0

def _forged_block_idx() -> bool:
    global _forged_call_count
    _forged_call_count += 1
    return True

_forged_block_idx.__module__ = "catlass.core_api"
_forged_block_idx.__name__ = "block_idx"

@tla.kernel
def static_false_and_short_circuit_kernel() -> None:
    if False and _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def static_true_or_short_circuit_kernel() -> None:
    if True or _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def staged_false_and_short_circuit_kernel() -> None:
    if _returns_false() and _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def staged_true_or_short_circuit_kernel() -> None:
    if _returns_true() or _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def staged_empty_list_and_short_circuit_kernel() -> None:
    if [] and _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def staged_nonempty_list_or_short_circuit_kernel() -> None:
    if [1] or _raise_if_evaluated():
        tla.make_coord(1, 0)

@tla.kernel
def staged_empty_dict_statement_kernel() -> None:
    if {}:
        _raise_if_evaluated()

@tla.kernel
def static_false_chained_comparison_short_circuit_kernel() -> None:
    if 2 < 1 < _raise_if_evaluated():
        tla.make_coord(1, 0)

_comparison_events: list[str] = []
_identity_events: list[str] = []

@tla.kernel
def staged_comparison_order_kernel() -> None:
    if (
        _record_comparator(_comparison_events, "left", 1)
        < _record_comparator(_comparison_events, "middle", 2)
        < _record_comparator(_comparison_events, "right", 3)
    ):
        tla.make_coord(1, 0)

@tla.kernel
def staged_identity_chain_kernel() -> None:
    left = (1,)
    middle = (1,)
    if left is middle is (_identity_events.append("marker") or (1,)):
        tla.make_coord(0, 0)

@tla.kernel
def statement_if_simple_staged_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if _staged_is_zero(i):
            tla.make_coord(i, 0)

@tla.kernel
def bad_statement_if_runtime_lazy_unknown_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and _undecorated_is_zero(i):
            tla.make_coord(i, 0)

@tla.kernel
def bad_runtime_lazy_trusted_star_expansion_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and tla.arch.block_idx(*_star_expansion_spy) >= 0:
            tla.make_coord(i, 0)

@tla.kernel
def bad_runtime_lazy_trusted_mapping_expansion_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and tla.arch.block_idx(**_mapping_expansion_spy) >= 0:
            tla.make_coord(i, 0)

@tla.kernel
def bad_runtime_lazy_forged_metadata_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and _forged_block_idx():
            tla.make_coord(i, 0)

@tla.kernel
def bad_statement_if_runtime_lazy_module_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and tla.compile():
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_runtime_lazy_block_idx_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and tla.arch.block_idx() >= 0:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_ordered_comparison_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i < limit:
            tla.make_coord(i, 0)
        if i <= limit:
            tla.make_coord(i, 1)
        if i > 0:
            tla.make_coord(i, 2)
        if i >= 0:
            tla.make_coord(i, 3)

@tla.kernel
def statement_if_chained_comparison_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if 0 <= i < limit:
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_any_all_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if any((i == 0, i == 1, i < limit)):
            tla.make_coord(i, 0)
        if all([i >= 0, i < limit]):
            tla.make_coord(i, 1)

@tla.kernel
def statement_if_any_all_generator_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if any(i == j for j in (0, 1)):
            tla.make_coord(i, 0)
        if all(i >= j for j in (0, 1)):
            tla.make_coord(i, 1)

@tla.kernel
def statement_if_inside_region_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        with tla.cube():
            if i == 0:
                tla.make_coord(i, 0)

@tla.kernel
def statement_if_carried_pointer_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        for i in tla.range(0, 2, 1):
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0
            if i == 0:
                selected = ptr1
            else:
                selected = ptr0
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.copy(local, tile)

@tla.kernel
def statement_if_mixed_index_pointer_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        for i in tla.range(0, 2, 1):
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            coord = i
            selected = ptr0
            if i == 0:
                coord = i + 1
                selected = ptr1
            else:
                coord = i + 2
                selected = ptr0
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.make_coord(coord, 0)
            tla.copy(local, tile)

@tla.kernel
def statement_if_elif_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        elif i == 1:
            coord = i + 2
        else:
            coord = i + 3
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_carried_tensor_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ub_ptr = tla.allocate((16, 8), tla.Float16, tla.AddressSpace.ub, 256)
    ub_tensor = tla.make_tensor_like(ub_ptr, root, tla.arch.RowMajor)
    for i in tla.range(0, 2, 1):
        left = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        right = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        selected = left
        if i == 0:
            selected = left
        else:
            selected = right
        with tla.vector():
            tla.copy(ub_tensor, left)
            tla.copy(ub_tensor, right)
            tla.copy(ub_tensor, selected)

@tla.kernel
def statement_if_mixed_index_tensor_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ub_ptr = tla.allocate((16, 8), tla.Float16, tla.AddressSpace.ub, 256)
    ub_tensor = tla.make_tensor_like(ub_ptr, root, tla.arch.RowMajor)
    for i in tla.range(0, 2, 1):
        left = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        right = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        coord = i
        selected = left
        if i == 0:
            coord = i + 1
            selected = left
        else:
            coord = i + 2
            selected = right
        tla.make_coord(coord, 0)
        with tla.vector():
            tla.copy(ub_tensor, left)
            tla.copy(ub_tensor, right)
            tla.copy(ub_tensor, selected)

@tla.kernel
def statement_if_not_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if not (i == 0):
            tla.make_coord(i, 0)

@tla.kernel
def statement_if_augassign_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord += 1
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_nested_function_local_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:

            def helper() -> None:
                local_only = i
                _ = local_only

            helper()
            coord = i + 1
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_nested_function_return_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:

            def helper() -> None:
                return

            helper()
            coord = i + 1
        tla.make_coord(coord, 0)

@tla.kernel
def bad_statement_if_return_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            return

@tla.kernel
def bad_statement_if_break_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            break

@tla.kernel
def bad_statement_if_continue_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            continue

@tla.kernel
def bad_statement_if_raise_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            raise ValueError("unsupported")

@tla.kernel
def bad_statement_if_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        else:
            coord = True
        tla.make_coord(coord, 0)

@tla.kernel
def bad_statement_if_multiple_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        offset = i + 1
        if i == 0:
            coord = i + 2
            offset = i + 3
        else:
            coord = True
            offset = i + 4
        tla.make_coord(coord, offset)

@tla.kernel
def bad_statement_if_pytree_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (True, i + 4)
        tla.make_coord(state[0], state[1])

@tla.kernel
def bad_statement_if_pytree_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (i + 4,)
        tla.make_coord(state[0], 0)

@tla.kernel
def bad_statement_if_dataclass_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1)
        if i == 0:
            state = BranchState(i + 2, i + 3)
        else:
            state = BranchState(True, i + 4)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def bad_statement_if_dataclass_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1)
        if i == 0:
            state = BranchState(i + 2, i + 3)
        else:
            state = OtherBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def bad_statement_if_custom_class_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(i, i + 1)
        if i == 0:
            state = CustomBranchState(i + 2, i + 3)
        else:
            state = CustomBranchState(True, i + 4)
        tla.make_coord(state.coord, state.offset)

@tla.kernel
def bad_statement_if_new_value_used_after_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        tla.make_coord(coord, 0)

@tla.kernel
def bad_statement_if_new_value_used_after_else_pass_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        else:
            pass
        tla.make_coord(coord, 0)

@tla.kernel
def bad_statement_if_new_value_used_after_elif_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        elif i == 1:
            coord = i + 2
        tla.make_coord(coord, 0)

@tla.kernel
def bad_statement_if_subscript_assignment_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = [i]
        if i == 0:
            values[0] = i + 1

@tla.kernel
def bad_statement_if_attribute_assignment_kernel(limit: int) -> None:
    box = object()
    for i in tla.range(0, limit, 1):
        if i == 0:
            box.value = i

@tla.kernel
def bad_statement_if_starred_assignment_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord, *rest = (i + 1, i + 2)
            _ = rest
        tla.make_coord(coord, 0)

@tla.kernel
def statement_if_compile_time_binding_write_inside_kernel(limit: int) -> None:
    scale = 2
    if limit > 0:
        tla.make_coord(limit * scale, 0)
        scale = 3
        tla.make_coord(limit * scale, 0)

@tla.kernel
def statement_if_definitely_assigned_python_seed_kernel(limit: int) -> None:
    scale = 2
    if limit > 0:
        scale = 3
    else:
        scale = 4
    tla.make_coord(scale, 0)

_ENABLE_DEFINITE_ASSIGNMENT = True

@tla.kernel
def static_config_nested_definite_assignment_kernel(limit: int) -> None:
    scale = 2
    if _ENABLE_DEFINITE_ASSIGNMENT:
        if limit > 0:
            scale = 3
        else:
            scale = 4
    tla.make_coord(scale, 0)

@tla.kernel
def region_compile_time_binding_write_inside_kernel() -> None:
    scale = 2
    with tla.cube():
        tla.make_coord(scale, 0)
        scale = 3
        tla.make_coord(scale, 0)

@tla.kernel
def bad_nested_compile_time_binding_read_after_inner_kernel(limit: int) -> None:
    scale = 2
    for i in tla.range(limit):
        if i > 0:
            scale = 3
        tla.make_coord(scale, 0)

@tla.kernel
def nested_runtime_binding_read_after_inner_kernel(limit: int) -> None:
    scale = tla.as_numeric(0)
    for i in tla.range(limit):
        if i > 0:
            scale = i
        tla.make_coord(scale, 0)

@tla.kernel
def bad_compile_time_binding_reassignment_does_not_restore_kernel(
    limit: int,
) -> None:
    scale = 2
    if limit > 0:
        scale = 3
    scale = 4
    tla.make_coord(scale, 0)

@tla.kernel
def bad_statement_if_active_closure_call_kernel(limit: int) -> None:
    def helper(value: int) -> None:
        tla.make_coord(value, 0)

    for i in tla.range(0, limit, 1):
        if i == 0:
            helper(i)

@tla.kernel
def bad_statement_if_active_object_method_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = []
        if i == 0:
            values.append(i)

@tla.kernel
def statement_if_active_object_method_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = [i, i + 1]
        if i == 0:
            values.reverse()
        tla.make_coord(values[0], values[1])

def test_const_expr_true_branch_stays_static() -> None:
    mlir = const_expr_if_kernel.dump_mlir(type_args=(True,))
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir

def test_const_expr_false_branch_stays_static() -> None:
    mlir = const_expr_if_kernel.dump_mlir(type_args=(False,))
    assert "scf.if" not in mlir
    assert "!tla.coord<2,0>" in mlir
    assert "!tla.coord<1,0>" not in mlir

def test_imported_const_expr_if_stays_static() -> None:
    mlir = imported_const_expr_if_kernel.dump_mlir(type_args=(True,))
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir

def test_const_expr_rejects_dynamic_value() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.const_expr"):
        _ = bad_const_expr_dynamic_kernel.dump_mlir(type_args=(2,))

def test_static_literal_if_stays_static() -> None:
    mlir = static_literal_if_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir

def test_pointer_conditional_expression_lowers_to_scf_if() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = pointer_conditional_kernel.dump_mlir(type_args=(mem,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_tensor_like" in mlir

def test_dynamic_mmad_init_expression_lowers_to_scf_if(compiler_tlair) -> None:
    mem_a = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0a,
                layout_tag=tla.arch.zN,
                coord=(0, 0),
            )
    mem_b = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((1, 256), (16, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0b,
                layout_tag=tla.arch.nZ,
                coord=(0, 0),
            )
    mem_c = make_fake_tensor(
                tla.Float32,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0c,
                layout_tag=tla.arch.L0Clayout,
                coord=(0, 0),
            )
    mlir = compiler_tlair(dynamic_mmad_init_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 2
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = false}> : () -> i1' in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_dynamic_mmad_unit_flag_expression_lowers_to_scf_if(compiler_tlair) -> None:
    mem_a = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0a,
                layout_tag=tla.arch.zN,
                coord=(0, 0),
            )
    mem_b = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((1, 256), (16, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0b,
                layout_tag=tla.arch.nZ,
                coord=(0, 0),
            )
    mem_c = make_fake_tensor(
                tla.Float32,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0c,
                layout_tag=tla.arch.L0Clayout,
                coord=(0, 0),
            )
    mlir = compiler_tlair(dynamic_mmad_unit_flag_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 2
    assert "value = 3" in mlir
    assert "value = 2" in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_dynamic_mmad_initc_unit_flag_expression_lowers_to_scf_if(compiler_tlair) -> None:
    mem_a = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0a,
                layout_tag=tla.arch.zN,
                coord=(0, 0),
            )
    mem_b = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((1, 256), (16, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0b,
                layout_tag=tla.arch.nZ,
                coord=(0, 0),
            )
    mem_c = make_fake_tensor(
                tla.Float32,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0c,
                layout_tag=tla.arch.L0Clayout,
                coord=(0, 0),
            )
    mlir = compiler_tlair(dynamic_mmad_initc_unit_flag_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 4
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = false}> : () -> i1' in mlir
    assert "value = 3" in mlir
    assert "value = 2" in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_inline_if_static_branch_is_lazy() -> None:
    mlir = inline_if_static_true_lazy_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "tla.make_coord" in mlir

def test_inline_if_tuple_result_lowers_to_scf_if_results() -> None:
    mlir = inline_if_tuple_result_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_inline_if_dataclass_result_lowers_to_scf_if_results() -> None:
    mlir = inline_if_dataclass_result_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir

def test_builtin_bool_predicate_lowers() -> None:
    mlir = builtin_bool_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir

def test_builtin_min_max_lower_dynamic_indexes() -> None:
    mlir = builtin_min_max_kernel.dump_mlir(type_args=(2,))
    assert "arith.cmpi" in mlir
    assert "arith.select" in mlir
    assert "tla.make_coord" in mlir

def test_builtin_name_redirects_to_dynamic_helpers() -> None:
    mlir = builtin_name_redirect_kernel.dump_mlir(type_args=(2,))
    assert "arith.ori" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir

def test_inline_if_rejects_structure_mismatch() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="Conditional expression.*structure"):
        _ = bad_inline_if_structure_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_side_effect_lowers_to_scf_if() -> None:
    mlir = statement_if_side_effect_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_carried_index_lowers_to_scf_if_result() -> None:
    mlir = statement_if_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_multiple_carried_values_lower_to_scf_if_results() -> None:
    mlir = statement_if_multiple_carried_values_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir

def test_loop_carried_index_lowers_to_scf_for_result() -> None:
    mlir = loop_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir

def test_loop_structured_carried_values_lower_to_scf_for_results() -> None:
    for kernel in (
        loop_tuple_carried_state_kernel,
        loop_list_carried_state_kernel,
        loop_dict_carried_state_kernel,
        loop_dataclass_carried_state_kernel,
    ):
        mlir = kernel.dump_mlir(type_args=(2,))
        assert "scf.for" in mlir
        assert "scf.yield" in mlir
        assert "arith.addi" in mlir
        assert "tla.make_coord" in mlir
    for kernel in (loop_custom_class_carried_state_kernel,):
        with pytest.raises(SyntaxError, match="user-defined class"):
            kernel.dump_mlir(type_args=(2,))

def test_statement_if_tuple_assignment_carried_values_lower() -> None:
    mlir = statement_if_tuple_assignment_carried_values_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_tuple_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_tuple_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_list_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_list_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_dict_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_dict_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_dataclass_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_dataclass_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir

def test_statement_if_frozen_dataclass_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_frozen_dataclass_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir

def test_statement_if_custom_class_carried_state_lowers_to_scf_if_results() -> None:
    with pytest.raises(SyntaxError, match="user-defined class 'CustomBranchState'"):
        statement_if_custom_class_carried_state_kernel.dump_mlir(type_args=(2,))

def test_statement_if_without_else_yields_original_carried_value() -> None:
    mlir = statement_if_implicit_else_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_predicate_bool_ops_lower() -> None:
    mlir = statement_if_bool_op_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") >= 2

def test_bool_op_and_generated_name_does_not_capture_user_name() -> None:
    mlir = statement_if_bool_op_and_name_collision_kernel.dump_mlir(type_args=(4,))
    assert "arith.constant false" in mlir
    assert mlir.count("scf.if") >= 2

def test_bool_op_or_generated_name_does_not_capture_user_name() -> None:
    mlir = statement_if_bool_op_or_name_collision_kernel.dump_mlir(type_args=(4,))
    assert "arith.constant true" in mlir
    assert mlir.count("scf.if") >= 2

def test_transformed_static_false_and_skips_rhs() -> None:
    mlir = static_false_and_short_circuit_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" not in mlir

def test_transformed_static_true_or_skips_rhs() -> None:
    mlir = static_true_or_short_circuit_kernel.dump_mlir()
    assert mlir.count("scf.if") == 0
    assert "!tla.coord<1,0>" in mlir

@pytest.mark.parametrize(
    "kernel",
    [staged_false_and_short_circuit_kernel, staged_true_or_short_circuit_kernel],
)
def test_staged_boolean_prefix_skips_unknown_rhs_call(kernel: Any) -> None:
    mlir = kernel.dump_mlir()
    assert "scf.if" not in mlir

@pytest.mark.parametrize(
    "kernel",
    [
        staged_empty_list_and_short_circuit_kernel,
        staged_nonempty_list_or_short_circuit_kernel,
        staged_empty_dict_statement_kernel,
    ],
)
def test_safe_builtin_condition_stages_without_dynamic_ir_or_rhs_call(
    kernel: Any,
) -> None:
    mlir = kernel.dump_mlir()
    assert "scf.if" not in mlir

def test_transformed_static_chained_comparison_skips_later_comparator() -> None:
    mlir = static_false_chained_comparison_short_circuit_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" not in mlir

def test_staged_chained_comparison_allows_inspectable_staging_helpers() -> None:
    _comparison_events.clear()
    staged_comparison_order_kernel.dump_mlir()
    assert _comparison_events == ["left", "middle", "right"]

def test_staged_identity_chain_rejects_uninspectable_object_methods() -> None:
    _identity_events.clear()
    with pytest.raises(SyntaxError, match="inspectable source"):
        staged_identity_chain_kernel.dump_mlir()
    assert _identity_events == []

@pytest.mark.parametrize(
    "kernel",
    [statement_if_guarded_division_kernel, statement_if_guarded_division_or_kernel],
)
def test_statement_if_boolean_rhs_is_lowered_in_lazy_region(kernel: Any) -> None:
    mlir = kernel.dump_mlir(type_args=(4,))
    division = max(mlir.find("arith.divsi"), mlir.find("arith.divui"))
    assert division != -1
    division_op = "arith.divsi" if "arith.divsi" in mlir else "arith.divui"
    assert _operation_is_nested_in_scf_if(mlir, division_op)
    assert mlir.count("scf.if") >= 2

def test_statement_if_multi_operand_boolean_chains_nest_left_to_right() -> None:
    mlir = statement_if_multi_operand_lazy_kernel.dump_mlir(type_args=(4,))
    division_op = "arith.divsi" if "arith.divsi" in mlir else "arith.divui"
    assert mlir.count(division_op) == 2
    assert _operation_is_nested_in_scf_if(mlir, division_op)
    assert mlir.count("scf.if") >= 6

def test_chained_comparison_lazily_contains_failing_later_operand() -> None:
    mlir = statement_if_chained_lazy_kernel.dump_mlir(type_args=(4,))
    division_op = "arith.divsi" if "arith.divsi" in mlir else "arith.divui"
    assert mlir.count(division_op) == 1
    assert _operation_is_nested_in_scf_if(mlir, division_op)

def test_recognized_dsl_call_lowers_in_runtime_lazy_region() -> None:
    mlir = statement_if_dsl_call_lazy_kernel.dump_mlir(type_args=(4,))
    assert mlir.count("tla.arch.block_idx") == 1
    assert _operation_is_nested_in_scf_if(mlir, "tla.arch.block_idx")

def test_chained_comparison_evaluates_shared_middle_dsl_call_once() -> None:
    mlir = statement_if_chained_dsl_call_kernel.dump_mlir()
    assert mlir.count("tla.arch.block_idx") == 1

def test_non_scalar_mask_predicate_has_source_located_diagnostic() -> None:
    with pytest.raises(Exception, match="scalar Bool predicate") as exc_info:
        bad_mask_predicate_kernel.dump_mlir()
    assert "test_frontend_branching.py" in str(exc_info.value)

def test_mixed_runtime_lazy_values_have_source_located_diagnostic() -> None:
    with pytest.raises(Exception, match="incompatible|type") as exc_info:
        bad_mixed_lazy_branch_value_kernel.dump_mlir(type_args=(2,))
    assert "test_frontend_branching.py" in str(exc_info.value)

def test_statement_if_allows_simple_staged_condition_call() -> None:
    mlir = statement_if_simple_staged_call_kernel.dump_mlir(type_args=(2,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_rejects_unknown_call_in_runtime_lazy_region() -> None:
    with pytest.raises(Exception, match="trace-time effect unknown") as exc_info:
        bad_statement_if_runtime_lazy_unknown_call_kernel.dump_mlir(type_args=(2,))
    assert "test_frontend_branching.py" in str(exc_info.value)

@pytest.mark.parametrize(
    ("kernel", "spy"),
    [
        (bad_runtime_lazy_trusted_star_expansion_kernel, _star_expansion_spy),
        (bad_runtime_lazy_trusted_mapping_expansion_kernel, _mapping_expansion_spy),
    ],
)
def test_trusted_call_expansion_rejects_before_python_protocols(
    kernel: Any, spy: _ExpansionSpy
) -> None:
    type(spy).calls = 0
    with pytest.raises(Exception, match="trace-time effect unknown"):
        kernel.dump_mlir(type_args=(2,))
    assert type(spy).calls == 0

def test_forged_dsl_metadata_rejects_before_call() -> None:
    global _forged_call_count
    _forged_call_count = 0
    with pytest.raises(Exception, match="trace-time effect unknown"):
        bad_runtime_lazy_forged_metadata_kernel.dump_mlir(type_args=(2,))
    assert _forged_call_count == 0

def test_statement_if_rejects_non_lowerable_catlass_call_in_lazy_region() -> None:
    with pytest.raises(Exception, match="trace-time effect unknown"):
        bad_statement_if_runtime_lazy_module_call_kernel.dump_mlir(type_args=(2,))

def test_statement_if_allows_lowerable_arch_call_in_lazy_region() -> None:
    mlir = statement_if_runtime_lazy_block_idx_kernel.dump_mlir(type_args=(2,))
    assert "tla.arch.block_idx" in mlir
    assert mlir.count("scf.if") >= 2

def test_statement_if_ordered_comparisons_lower() -> None:
    mlir = statement_if_ordered_comparison_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") == 4
    assert mlir.count("arith.cmpi") >= 4
    assert "tla.make_coord" in mlir

def test_statement_if_chained_comparison_lowers() -> None:
    mlir = statement_if_chained_comparison_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") >= 2
    assert mlir.count("arith.cmpi") >= 2
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_any_all_predicates_lower() -> None:
    mlir = statement_if_any_all_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.ori" in mlir
    assert "arith.andi" in mlir
    assert mlir.count("scf.if") == 2
    assert "tla.make_coord" in mlir

def test_statement_if_rejects_generator_predicates() -> None:
    with pytest.raises(SyntaxError, match="generator expressions in its condition"):
        statement_if_any_all_generator_predicate_kernel.dump_mlir(type_args=(2,))

def test_statement_if_inside_region_lowers() -> None:
    mlir = statement_if_inside_region_kernel.dump_mlir(type_args=(2,))
    assert "tla.cube" in mlir
    assert "scf.if" in mlir

def test_statement_if_carried_pointer_lowers_to_scf_if_result() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = statement_if_carried_pointer_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "tla.make_tensor_like" in mlir
    assert "tla.copy" in mlir

def test_statement_if_mixed_index_pointer_lowers_to_scf_if_results() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = statement_if_mixed_index_pointer_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "index" in mlir
    assert "tla.make_tensor_like" in mlir
    assert "tla.copy" in mlir

def test_statement_if_elif_lowers_to_nested_scf_if() -> None:
    mlir = statement_if_elif_carried_index_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") >= 2
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_carried_tensor_lowers_to_scf_if_result() -> None:
    mem = make_fake_tensor(
        tla.Float16,
        (16, 8),
        (8, 1),
        origin_shape=(16, 8),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = statement_if_carried_tensor_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.tensor<!tla.layout<!tla.shape<16,4>" in mlir
    assert "tla.copy" in mlir

def test_statement_if_mixed_index_tensor_lowers_to_scf_if_results() -> None:
    mem = make_fake_tensor(
        tla.Float16,
        (16, 8),
        (8, 1),
        origin_shape=(16, 8),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = statement_if_mixed_index_tensor_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.tensor<!tla.layout<!tla.shape<16,4>" in mlir
    assert "index" in mlir
    assert "tla.copy" in mlir

def test_statement_if_not_predicate_lowers() -> None:
    mlir = statement_if_not_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.xori" in mlir
    assert "scf.if" in mlir

def test_statement_if_augassign_carried_index_lowers() -> None:
    mlir = statement_if_augassign_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir

def test_statement_if_allows_nested_function_with_isolated_local() -> None:
    mlir = statement_if_nested_function_local_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir

def test_statement_if_allows_nested_function_with_return() -> None:
    mlir = statement_if_nested_function_return_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir

def test_statement_if_rejects_return() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_return_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_break() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_break_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_continue() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_continue_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_raise() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_raise_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_mismatched_carried_type() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"coord|i1"):
        _ = bad_statement_if_type_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_mismatched_multiple_carried_type_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"coord|i1"):
        _ = bad_statement_if_multiple_type_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_pytree_type_mismatch_by_leaf_name() -> None:
    # Int32 vs Bool differ in FrontendIfTreeSpec.node_type; rejected as structure mismatch.
    with pytest.raises(tla.TlaCoreAPIError, match=r"'state'.*structure"):
        _ = bad_statement_if_pytree_type_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_pytree_structure_mismatch_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_if_pytree_structure_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_dataclass_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_if_dataclass_type_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_dataclass_structure_mismatch_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_if_dataclass_structure_mismatch_kernel.dump_mlir(
            type_args=(2,)
        )

def test_statement_if_rejects_custom_class_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(SyntaxError, match="user-defined class 'CustomBranchState'"):
        _ = bad_statement_if_custom_class_type_mismatch_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_wrong_carried_result_count_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="coord, offset"):
        ast_decorators_mod._internal_frontend_if(
            True,
            lambda coord, offset: coord,
            None,
            1,
            2,
            carried_names=("coord", "offset"),
        )

def test_statement_if_rejects_new_value_used_after_branch() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_new_value_used_after_else_pass() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_else_pass_kernel.dump_mlir(
            type_args=(2,)
        )

def test_statement_if_rejects_new_value_used_after_elif() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_elif_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_subscript_assignment() -> None:
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"(?s)values\[0\] = i \+ 1.*expected tensor, got list",
    ):
        _ = bad_statement_if_subscript_assignment_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_attribute_assignment() -> None:
    with pytest.raises(SyntaxError, match="assignments to local names"):
        _ = bad_statement_if_attribute_assignment_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_starred_assignment() -> None:
    with pytest.raises(SyntaxError, match="assignments to local names"):
        _ = bad_statement_if_starred_assignment_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_compile_time_binding_write_at_assignment() -> None:
    with pytest.raises(
        Exception,
        match="compile-time binding 'scale'.*use tla.as_numeric",
    ) as exc:
        statement_if_compile_time_binding_write_inside_kernel.dump_mlir(type_args=(2,))

    line = inspect.getsourcelines(statement_if_compile_time_binding_write_inside_kernel.fn)[1] + 5
    message = str(exc.value)
    assert "reason: SyntaxError: compile-time binding 'scale'" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |" in message
    assert "scale = 3" in message

def test_statement_if_rejects_definitely_assigned_python_seed() -> None:
    with pytest.raises(Exception, match="use tla.as_numeric"):
        statement_if_definitely_assigned_python_seed_kernel.dump_mlir(type_args=(2,))

def test_static_global_bool_rejects_nested_compile_time_write() -> None:
    with pytest.raises(Exception, match="use tla.as_numeric"):
        static_config_nested_definite_assignment_kernel.dump_mlir(type_args=(2,))

def test_region_rejects_compile_time_binding_write_at_assignment() -> None:
    with pytest.raises(
        Exception,
        match="compile-time binding 'scale'.*use tla.as_numeric",
    ) as exc:
        region_compile_time_binding_write_inside_kernel.dump_mlir()

    line = inspect.getsourcelines(region_compile_time_binding_write_inside_kernel.fn)[1] + 5
    message = str(exc.value)
    assert "reason: SyntaxError: compile-time binding 'scale'" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |" in message
    assert "scale = 3" in message

def test_nested_scope_rejects_compile_time_binding_write_in_inner_scope() -> None:
    with pytest.raises(
        Exception,
        match="compile-time binding 'scale'.*use tla.as_numeric",
    ):
        bad_nested_compile_time_binding_read_after_inner_kernel.dump_mlir(
            type_args=(2,)
        )

def test_nested_scope_keeps_runtime_binding_readable() -> None:
    mlir = nested_runtime_binding_read_after_inner_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "scf.if" in mlir

def test_reassignment_does_not_restore_compile_time_binding() -> None:
    with pytest.raises(
        Exception,
        match="compile-time binding 'scale'.*use tla.as_numeric",
    ):
        bad_compile_time_binding_reassignment_does_not_restore_kernel.dump_mlir(
            type_args=(2,)
        )

def test_statement_if_allows_active_closure_call() -> None:
    mlir = bad_statement_if_active_closure_call_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir

def test_statement_if_rejects_active_object_method_call() -> None:
    with pytest.raises(SyntaxError, match="requires @tla.jit"):
        statement_if_active_object_method_call_kernel.dump_mlir(type_args=(2,))

def test_statement_if_rejects_active_object_method_call_structure_change() -> None:
    with pytest.raises(SyntaxError, match="requires @tla.jit"):
        _ = bad_statement_if_active_object_method_call_kernel.dump_mlir(type_args=(2,))

@tla.kernel
def dynamic_if_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    if idx == 0:
        values[idx]


@tla.kernel
def nested_dynamic_if_bad_constexpr_kernel() -> None:
    idx = tla.arch.block_idx()
    if idx == 0:
        if idx == 0:
            tla.const_expr(idx)


def _source_line(fn: Any, needle: str) -> int:
    source_lines, first_lineno = inspect.getsourcelines(fn.fn)
    for offset, line in enumerate(source_lines):
        if needle in line:
            return first_lineno + offset
    raise AssertionError(f"Unable to find source line containing {needle!r}")

def test_dynamic_if_then_error_reports_original_source_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = _source_line(dynamic_if_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in dynamic if then-region" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |         values[idx]" in message
    assert "|         ^" in message
    assert "cannot be used as a Python index" in message or "list indices" in message
    assert "Int32" in message
    assert "Captured cause traceback:" not in message
    verbose = excinfo.value.format(verbose=True)
    assert "Captured cause traceback:" in verbose
    assert "TypeError: Int32 SSA cannot be used as a Python index" in verbose
    monkeypatch.setenv("CATLASS_DSL_VERBOSE_ERRORS", "1")
    assert "Captured cause traceback:" in str(excinfo.value)


def test_nested_dynamic_if_preserves_inner_diagnostic_location() -> None:
    line = _source_line(nested_dynamic_if_bad_constexpr_kernel, "tla.const_expr(idx)")

    with pytest.raises(tla.TlaCoreAPIError, match="tla.const_expr") as exc_info:
        nested_dynamic_if_bad_constexpr_kernel.dump_mlir()

    diagnostic = exc_info.value
    assert diagnostic.location is not None
    assert diagnostic.location.filename == __file__
    assert diagnostic.location.lineno == line
    assert f" --> {__file__}:{line}:" in str(diagnostic)
    assert "tla.const_expr(idx)" in str(diagnostic)


@tla.kernel
def kernel_body_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    values[idx]


def test_kernel_body_error_renders_source_code_frame() -> None:
    line = _source_line(kernel_body_bad_list_index_kernel, "values[idx]")
    with pytest.raises(UnsupportedExecutionLowering) as excinfo:
        kernel_body_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert "Execution-mode lowering failed while running" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |     values[idx]" in message
    assert "|     ^" in message
    assert "reason: TypeError: Int32 SSA cannot be used as a Python index" in message
    assert "Captured cause traceback:" not in message
    assert "Captured cause traceback:" in excinfo.value.format(verbose=True)


@tla.kernel
def kernel_body_value_error_kernel() -> None:
    int("not an integer")


def test_kernel_body_value_error_renders_source_code_frame() -> None:
    line = _source_line(kernel_body_value_error_kernel, 'int("not an integer")')
    with pytest.raises(UnsupportedExecutionLowering) as excinfo:
        kernel_body_value_error_kernel.dump_mlir()

    message = str(excinfo.value)
    assert "Execution-mode lowering failed while running" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |     int(\"not an integer\")" in message
    assert "reason: ValueError:" in message
    assert "Captured cause traceback:" not in message
    assert "Captured cause traceback:" in excinfo.value.format(verbose=True)


@tla.kernel
def dynamic_if_constexpr_bool_bad_list_index_kernel(flag: tla.Constexpr[bool]) -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    if flag:
        values[idx]

def test_constexpr_if_body_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_if_constexpr_bool_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_constexpr_bool_bad_list_index_kernel.dump_mlir(type_args=(True,))
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in dynamic if then-region" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |         values[idx]" in message
    assert "|         ^" in message
    assert "reason: TypeError:" in message
    assert "cannot be used as a Python index" in message or "list indices" in message
    assert "Int32" in message
    assert "Captured cause traceback:" not in message
    assert "Captured cause traceback:" in excinfo.value.format(verbose=True)

@tla.kernel
def dynamic_if_expr_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    result = values[idx] if idx == 0 else 0
    tla.make_coord(result, 0)

def test_dynamic_if_expr_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_if_expr_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_expr_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert (
        "Execution-mode lowering failed in conditional expression then-region"
        in message
    )
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |     result = values[idx] if idx == 0 else 0" in message
    assert re.search(r"\n\s+\|\s+\^+", message)
    assert "reason: TypeError:" in message
    assert "cannot be used as a Python index" in message or "list indices" in message
    assert "Int32" in message
    assert "Captured cause traceback:" not in message
    assert "Captured cause traceback:" in excinfo.value.format(verbose=True)

@tla.kernel
def cross_flag_statement_if_kernel() -> None:
    ping = tla.cross_flag("ping")
    pong = tla.cross_flag("pong")
    selected = ping
    if tla.arch.block_idx() == 0:
        selected = ping
    else:
        selected = pong
    with tla.vector():
        tla.cross_core_set_flag(selected, tla.pipes.FIX)
        tla.cross_core_wait_flag(selected, tla.pipes.VECTOR)

@tla.kernel
def cross_flag_inline_if_kernel() -> None:
    ping = tla.cross_flag("ping")
    pong = tla.cross_flag("pong")
    selected = ping if tla.arch.block_idx() == 0 else pong
    with tla.vector():
        tla.cross_core_set_flag(selected, tla.pipes.FIX)
        tla.cross_core_wait_flag(selected, tla.pipes.VECTOR)

@pytest.mark.parametrize(
    "kernel", (cross_flag_statement_if_kernel, cross_flag_inline_if_kernel)
)
def test_dynamic_cross_flag_selection_emits_frontend_ir(kernel: Any) -> None:
    mlir = kernel.dump_mlir()
    assert mlir.count("= tla.cross_flag") == 2
    assert "scf.if" in mlir
    assert "!tla.cross_flag" in mlir
    assert "tla.cross_core_set_flag" in mlir
    assert "tla.cross_core_wait_flag" in mlir
    assert "#tla.pipe<fix>" in mlir
    assert "#tla.pipe<vector>" in mlir
