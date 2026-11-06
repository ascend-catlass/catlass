"""Versioned compiler facts used for launch and disk-cache restoration.

IR assembly remains a diagnostic artifact. These facts are collected from MLIR
operations and attributes, never inferred from its textual representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import prod
from typing import Any, Mapping


@dataclass(frozen=True)
class PrintTensorMetadata:
    shape: tuple[int, ...]
    count: int | None
    dtype: str
    position: str
    call: int = 0
    dynamic_execution: bool = False


@dataclass(frozen=True)
class KernelRuntimeMetadata:
    entrypoint: str
    kernel_mode: str = "aiv"
    target_arch: str = "c310"
    uses_scalar_print: bool = False
    uses_tensor_print: bool = False
    uses_debug_tunnel: bool = False
    print_argument_layout: tuple[str, ...] = ()
    simt_print_formats: str | None = None
    print_helper_core: str | None = None
    local_print_positions: tuple[str, ...] = ()
    tensor_records: tuple[PrintTensorMetadata, ...] = ()
    schema_version: int = 2
    aiv_subblock_count: int = 1

    @property
    def uses_simt_print(self) -> bool:
        return self.simt_print_formats is not None

    @property
    def is_mixed(self) -> bool:
        return self.kernel_mode == "mix"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelRuntimeMetadata:
        """Reject absent/old schemas and inconsistent launch facts explicitly."""
        if not isinstance(data, Mapping) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError(
                "missing or unknown runtime metadata fields; rebuild the DSL or recompile the cache"
            )
        if type(data["schema_version"]) is not int or data["schema_version"] != 2:
            raise ValueError("unsupported runtime metadata schema")
        values = dict(data)
        for name in (
            "print_argument_layout",
            "local_print_positions",
            "tensor_records",
        ):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"invalid runtime metadata sequence: {name}")
            values[name] = tuple(values[name])
        records = []
        for raw in values["tensor_records"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                f.name for f in fields(PrintTensorMetadata)
            }:
                raise ValueError("invalid Tensor print metadata fields")
            shape = raw["shape"]
            if (
                not isinstance(shape, (list, tuple))
                or len(shape) not in (1, 2)
                or any(type(n) is not int or n == 0 or n < -1 for n in shape)
            ):
                raise ValueError("invalid Tensor print shape metadata")
            count = raw["count"]
            if count is not None and (type(count) is not int or count <= 0):
                raise ValueError("invalid Tensor print count metadata")
            dtype = {"ui8": "u8", "ui16": "u16", "ui32": "u32"}.get(
                raw["dtype"], raw["dtype"]
            )
            position = str(raw["position"]).upper()
            if dtype not in {
                "f16",
                "f32",
                "i8",
                "i16",
                "i32",
                "u8",
                "u16",
                "u32",
            } or position not in {"GM", "UB", "L1", "L0C"}:
                raise ValueError("invalid Tensor print dtype or position metadata")
            maximum = 8 if position == "L1" else 256 if position == "L0C" else 262112
            if count is not None and (
                count > maximum or (all(n > 0 for n in shape) and count > prod(shape))
            ):
                raise ValueError("Tensor print count exceeds the shape or record limit")
            if (count is None or any(n < 0 for n in shape)) and not raw[
                "dynamic_execution"
            ]:
                raise ValueError(
                    "dynamic Tensor print metadata is missing its execution flag"
                )
            if (
                type(raw["call"]) is not int
                or raw["call"] != len(records)
                or not isinstance(raw["dynamic_execution"], bool)
            ):
                raise ValueError("invalid Tensor print call metadata")
            records.append(
                PrintTensorMetadata(
                    tuple(shape),
                    count,
                    dtype,
                    position,
                    raw["call"],
                    raw["dynamic_execution"],
                )
            )
        values["tensor_records"] = tuple(records)
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("unsupported runtime metadata schema")
        if (
            type(self.aiv_subblock_count) is not int
            or self.aiv_subblock_count not in (1, 2)
            or (not self.is_mixed and self.aiv_subblock_count != 1)
        ):
            raise ValueError("invalid compiler-produced AIV subblock count")
        if not isinstance(self.entrypoint, str) or not self.entrypoint:
            raise ValueError("missing runtime metadata entrypoint")
        if self.kernel_mode not in {"aic", "aiv", "mix"} or self.target_arch != "c310":
            raise ValueError("unsupported runtime metadata target or core type")
        for flag in (
            self.uses_scalar_print,
            self.uses_tensor_print,
            self.uses_debug_tunnel,
        ):
            if not isinstance(flag, bool):
                raise ValueError("invalid runtime metadata print flag")
        layout = self.print_argument_layout
        has_print = (
            self.uses_scalar_print
            or self.uses_tensor_print
            or self.uses_debug_tunnel
            or self.uses_simt_print
        )
        if any(not isinstance(role, str) for role in layout) or (
            bool(layout) != bool(has_print)
            or (
                layout
                and (layout.count("user") != 1 or len(layout) != len(set(layout)))
            )
            or set(layout) - {"user", "fifo", "simt", "tunnel"}
            or ("fifo" in layout) != (self.uses_scalar_print or self.uses_tensor_print)
            or ("simt" in layout) != self.uses_simt_print
            or ("tunnel" in layout) != self.uses_debug_tunnel
        ):
            raise ValueError("invalid compiler-produced print argument layout")
        if self.simt_print_formats is not None:
            if not isinstance(self.simt_print_formats, str):
                raise ValueError("invalid SIMT print format table")
            formats = bytes.fromhex(self.simt_print_formats)
            if formats and not formats.endswith(b"\0"):
                raise ValueError("unterminated SIMT print format table")
        if self.print_helper_core not in {
            None,
            "aic",
            "aiv",
            "both",
        } or self.uses_tensor_print != (self.print_helper_core is not None):
            raise ValueError("invalid Tensor print helper core metadata")
        if any(
            position not in {"L1", "L0C"} for position in self.local_print_positions
        ):
            raise ValueError("invalid local Tensor print position metadata")
        if len(self.local_print_positions) > 1:
            raise ValueError("one kernel cannot mix L1 and L0C tensor prints")
        if bool(self.local_print_positions) != self.uses_debug_tunnel:
            raise ValueError("local Tensor print and debug tunnel metadata disagree")
        shared = (
            self.uses_scalar_print
            or self.uses_simt_print
            or self.print_helper_core == "both"
        )
        if bool(self.tensor_records) != (self.uses_tensor_print and not shared):
            raise ValueError("missing or unexpected static Tensor print metadata")
