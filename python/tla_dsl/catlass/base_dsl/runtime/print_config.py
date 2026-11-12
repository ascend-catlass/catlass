"""Immutable print launch state shared by the executor and runtime."""

from dataclasses import dataclass

from ...compiler_metadata import PrintTensorMetadata


@dataclass(frozen=True)
class _PrintLaunchConfig:
    uses_scalar_print: bool = False
    mixed_core: bool = False
    print_metadata: tuple[PrintTensorMetadata, ...] | None = None
    print_helper_core: str | None = None
    print_tensor_position: str | None = None
    simt_print_formats: bytes | None = None
    print_argument_layout: tuple[str, ...] = ()
    aiv_subblock_count: int = 1

    @property
    def uses_tensor_print(self) -> bool:
        # Shared scalar/SIMT launches have no static Tensor records; the
        # compiler-provided helper core is the authoritative Tensor-print flag.
        return self.print_helper_core is not None
