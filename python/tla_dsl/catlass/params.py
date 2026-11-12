from __future__ import annotations

import enum
from typing import Any

from .types import TlaTensor
from dataclasses import dataclass


class QuantMode(enum.IntEnum):
    NO_QUANT = 0
    PER_TENSOR = 1  # need quant_scale
    PER_CHANNEL = 2  # need quant_tensor

    def __str__(self):
        return self.name  # "NO_QUANT"/..


class L0C2UBMode(enum.IntEnum):
    NO_SPLIT_VEC_0 = 0
    NO_SPLIT_VEC_1 = 1
    SPLIT_M = 2
    SPLIT_N = 3

    def __str__(self):
        return self.name  # "NO_SPLIT_VEC_0"/..


class ComputeOrder(enum.IntEnum):
    M_FIRST = 0
    N_FIRST = 1

    def __str__(self):
        return self.name  # "M_FIRST"/"N_FIRST"


class HF32Mode(enum.IntEnum):
    HF32_DISABLE = 0
    HF32_NEAREST_ZERO = 1
    HF32_NEAREST_EVEN = 2

    def __str__(self):
        return self.name  # "HF32_DISABLE"/"HF32_NEAREST_ZERO"/"HF32_NEAREST_EVEN"


class AtomicMode(enum.Enum):
    """Atomic operation mode"""

    NONE = "none"  # No atomic operation
    ADD = "add"  # Set atomic add

    def __str__(self) -> str:
        return self.value


class RegPart(enum.Enum):
    """Register half selection."""

    LOW = "low"
    HIGH = "high"

    def __str__(self) -> str:
        return self.value


class ReduceMode(enum.Enum):
    """Vector reduction scope selection.

    INTER_BLOCK (default) reduces the whole vector across all 32-byte data
    blocks to a one-lane fragment; INTRA_BLOCK reduces each 32-byte block
    independently and returns the packed per-block results.
    """

    INTRA_BLOCK = "intra_block"
    INTER_BLOCK = "inter_block"

    def __str__(self) -> str:
        return self.value


@dataclass
class CopyParams:
    """Marker annotation for base copy params."""

    pass


class PostMode:
    POST_MODE_NORMAL = "norm"
    POST_MODE_UPDATE = "update"


class LoadDist:
    DIST_NORM = "norm"
    DIST_BRC_B32 = "brc_b32"
    DIST_DINTLV_B32 = "dintlv_b32"
    DIST_US_B8 = "us_b8"
    DIST_BRC_B16 = "brc_b16"
    DIST_US_B16 = "us_b16"
    DIST_UNPACK_B16 = "unpack_b16"
    DIST_UNPACK4_B8 = "unpack4_b8"
    DIST_E2B_B16 = "e2b_b16"
    DIST_E2B_B32 = "e2b_b32"
    DIST_BLK = "blk"


class MaskLoadDist:
    """Mask load distribution modes for ``MaskLoadParams``.

    - ``DIST_NORM``: bit-packed mask load (default)
    - ``DIST_US``: bit upsample
    - ``DIST_DS``: bit downsample

    For ``DIST_US`` / ``DIST_DS``, the UB element type may be b8/b16/b32/b64
    (address typing only; mask bit layout is independent of element width).
    """

    DIST_NORM = "norm"
    DIST_US = "us"
    DIST_DS = "ds"


class StoreDist:
    DIST_NORM = "norm"
    DIST_PACK_B32 = "pack_b32"
    DIST_PACK_B16 = "pack_b16"
    DIST_PACK4_B32 = "pack4_b32"
    DIST_FIRST_ELEMENT_B8 = "first_element_b8"
    DIST_FIRST_ELEMENT_B16 = "first_element_b16"
    DIST_FIRST_ELEMENT_B32 = "first_element_b32"


class MaskStoreDist:
    """Mask store distribution modes."""

    DIST_NORM = "norm"


class StoreParams:
    """Marker annotation for tensor tile store params."""

    pass


@dataclass
class NormalStoreParams(StoreParams):
    """Default aligned vector store (``post_update_stride`` in elements).

    With ``post_mode=update``, ``post_update_stride=0`` (default) means advance
    by ``tile_extent`` (``origin_shape``); an explicit value keeps fill /
    stride-advance decoupling.
    """

    store_dist: str = StoreDist.DIST_NORM
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: Any = 0


@dataclass
class UnalignStoreParams(StoreParams):
    """Unaligned UB vector store.

    With ``post_mode=update``, leave ``post_update_stride`` at default ``0``:
    the window advances by ``tile_extent`` (``origin_shape``; dense only).
    Passing an explicit ``post_update_stride`` is rejected. Returns the next
    window tile from the post-update pointer.
    """

    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: Any = 0


@dataclass
class BlockStoreParams(StoreParams):
    """UB Block store with stride (8 DataBlocks per call).

    ``block_stride`` is in 32B DataBlock units. Unlike ``BlockLoadParams``,
    NORMAL mode has no software pre-offset: ``repeat_stride`` is only used
    with ``post_mode=update`` (32B DataBlock repeat stride; returns the next
    window tile).
    """

    block_stride: int = 0
    post_mode: str = PostMode.POST_MODE_NORMAL
    repeat_stride: Any = 0


@dataclass
class MaskStoreParams(StoreParams):
    """Continuous-aligned MaskSSA store to 1/2/4-byte UB.

    Spill size is ``N/8`` from the ``MaskSSA`` value. ``N`` must equal
    ``256 / sizeof(T)`` for the destination UB element type (``T`` may be int
    or float of that width). Default distribution is ``DIST_NORM``.
    """

    store_dist: str = MaskStoreDist.DIST_NORM


class LoadParams:
    """Marker annotation for tensor tile load params."""

    pass


@dataclass
class NormalLoadParams(LoadParams):
    """Default aligned vector load (``post_update_stride`` in elements).

    With ``post_mode=update``, ``post_update_stride=0`` (default) means advance
    by ``tile_extent`` (``origin_shape``); an explicit value keeps fill /
    stride-advance decoupling.
    """

    load_dist: str = LoadDist.DIST_NORM
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: Any = 0


@dataclass
class UnalignLoadParams(LoadParams):
    """Unaligned UB vector load.

    With ``post_mode=update``, leave ``post_update_stride`` at default ``0``:
    the window advances by ``tile_extent`` (``origin_shape``; dense only).
    Passing an explicit ``post_update_stride`` is rejected. Returns
    ``(vector, next_tile)`` from the post-update pointer.
    """

    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: Any = 0


@dataclass
class BlockLoadParams(LoadParams):
    """UB Block load with stride (8 DataBlocks per call).

    One load gathers 8 DataBlocks whose heads are ``block_stride``
    DataBlocks (32B units) apart; ``block_stride == 0`` repeats the first
    DataBlock into all 8 slots. With ``post_mode=normal``, ``repeat_stride``
    is a compile-time address pre-offset in 32B DataBlocks; with
    ``post_mode=update`` it is the post-update repeat stride in the same unit,
    and the call returns ``(vector, next_tile)``. Both strides must fit in
    16 bits ([0, 65535]).
    """

    block_stride: int = 0
    post_mode: str = PostMode.POST_MODE_NORMAL
    repeat_stride: Any = 0


@dataclass
class MaskLoadParams(LoadParams):
    """Continuous-aligned MaskSSA load via ``tile.load``.

    - ``DIST_NORM``: UB element is a 1/2/4-byte scalar; predicate width
      ``!tla.mask<N>`` is ``N = 256 / sizeof(T)`` (same rule as
      ``create_mask``).
    - ``DIST_US`` / ``DIST_DS``: UB element type may be any b8/b16/b32/b64
      scalar. The element type only types the UB address; the mask bit
      layout does not depend on element width. ``N`` is a static
      tile-shape product in ``{32, 64, 128, 256}`` when available;
      otherwise ``N = 256``.

    Byte-offset alignment (UB byte address = base + offset×sizeof(T)):
    - ``DIST_NORM`` → ``VL/8`` bytes (VL=256 → 32B)
    - ``DIST_US`` → ``VL/16`` bytes (VL=256 → 16B)
    - ``DIST_DS`` → ``min(32, VL/4)`` bytes (VL=256 → 32B)

    ``load_dist``:
    - ``DIST_NORM``: bit-packed mask load (default)
    - ``DIST_US``: bit upsample (VL/16 → VL/8)
    - ``DIST_DS``: bit downsample (VL/4 → VL/8)
    """

    load_dist: str = MaskLoadDist.DIST_NORM


@dataclass
class CopyUbToGmParams(CopyParams):
    atomic_mode: AtomicMode = AtomicMode.NONE

    def _validate(self):
        VALID_ATOMIC_MODE = (AtomicMode.NONE, AtomicMode.ADD)
        cls_name = type(self).__name__
        if self.atomic_mode not in VALID_ATOMIC_MODE:
            raise ValueError(
                f"{cls_name}.atomic_mode must be one of {VALID_ATOMIC_MODE}"
            )


@dataclass
class CopyL0C2DstParams(CopyParams):
    unit_flag: int = 0
    relu_enable: bool = False
    quant_mode: QuantMode = QuantMode.NO_QUANT
    quant_scale: float | None = None
    quant_offset: int | None = None
    quant_tensor: TlaTensor | None = None
    l0c2ub_mode: L0C2UBMode = L0C2UBMode.NO_SPLIT_VEC_0
    atomic_mode: AtomicMode = AtomicMode.NONE

    def _validate(self):
        from .core_api import _category
        from .execution_lowering import TlaLoweringError

        VALID_UNIT_FLAG = (0b00, 0b11)
        VALID_ATOMIC_MODE = (AtomicMode.NONE, AtomicMode.ADD)
        cls_name = type(self).__name__
        if not isinstance(self.unit_flag, int):
            raise TlaLoweringError(f"{cls_name}.unit_flag must be a compile-time int")
        if self.unit_flag not in VALID_UNIT_FLAG:
            raise ValueError(f"{cls_name}.unit_flag must be one of {VALID_UNIT_FLAG}")
        if not isinstance(self.relu_enable, bool):
            raise TlaLoweringError(
                f"{cls_name}.relu_enable must be a compile-time bool"
            )
        if not isinstance(self.quant_mode, QuantMode):
            raise TlaLoweringError(
                f"{cls_name}.quant_mode must be a {QuantMode}, got {type(self.quant_mode).__name__}"
            )
        if not isinstance(self.l0c2ub_mode, L0C2UBMode):
            raise TlaLoweringError(
                f"{cls_name}.l0c2ub_mode must be a {L0C2UBMode}, got {type(self.l0c2ub_mode).__name__}"
            )
        if self.quant_mode == QuantMode.PER_TENSOR:
            if self.quant_scale is None:
                raise TlaLoweringError(
                    f"{cls_name}.quant_scale is required when quant_mode is PER_TENSOR"
                )
            if self.quant_offset is not None:
                if isinstance(self.quant_offset, int):
                    if not -256 <= self.quant_offset <= 255:
                        raise ValueError(
                            f"{cls_name}.quant_offset must fit the 9-bit signed "
                            f"Offset field [-256, 255], got {self.quant_offset}"
                        )
        elif self.quant_offset is not None:
            raise TlaLoweringError(
                f"{cls_name}.quant_offset is only meaningful with QuantMode.PER_TENSOR"
            )
        if self.quant_mode == QuantMode.PER_CHANNEL:
            if _category(self.quant_tensor) != "tensor":
                raise TlaLoweringError(
                    f"{cls_name}.quant_tensor must be a tensor, got {type(self.quant_tensor).__name__}"
                )
        if self.atomic_mode not in VALID_ATOMIC_MODE:
            raise ValueError(
                f"{cls_name}.atomic_mode must be one of {VALID_ATOMIC_MODE}"
            )


# ---------------------------------------------------------------------------
# CastParams: the four knobs of a tla.cast (VectorSSA.to). The enum member
# *values* are the MLIR-asm keywords; the integer codes match the
# I32EnumAttrCase values in Tla.td and the order in the tla.cast trait array.
# ---------------------------------------------------------------------------


class RegSlot(enum.Enum):
    """Packed-register position the narrow cast result lands in.

    For a 2x-width cast (e.g. f32->f16, i32<->i16) this maps to the AVE even/odd
    ``part`` (only ZERO=part_even / ONE=part_odd are valid there). For a 4x-width
    cast (i32<->i8) it maps to the AVE pack pattern ``pp`` and all four values
    select the pack quarter: ZERO=pp0, ONE=pp1, TWO=pp2, THREE=pp3.
    """

    ZERO = "zero"  # part_even / pp0
    ONE = "one"  # part_odd  / pp1
    TWO = "two"  # pp2 (4x casts only)
    THREE = "three"  # pp3 (4x casts only)

    def __str__(self) -> str:
        return self.value


class SatMode(enum.Enum):
    """Overflow behaviour of the cast (AVE ``sat`` BoolAttr).

    ``SAT`` clamps an out-of-range value to the destination's largest finite
    magnitude. ``NOSAT`` lets it overflow the way IEEE does: to an infinity, or
    to NaN for a format that has none, such as ``Float8E4M3FN``.

    ``SAT`` also turns a NaN input into ``+0`` instead of propagating it. That
    is AVE's saturating-narrow behaviour rather than anything specific to fp8 --
    ``f32 -> f16`` does the same -- so use ``NOSAT`` where NaNs must survive.
    """

    UNKNOWN = "unknown"  # unspecified; the lowering treats this exactly as NOSAT
    SAT = "sat"
    NOSAT = "nosat"

    def __str__(self) -> str:
        return self.value


class RoundMode(enum.Enum):
    """Rounding applied on precision loss (AVE ``tla.cast`` / ``VectorSSA.to``).

    ``CAST_NONE`` matches AscendC: it must be passed explicitly. Lossless casts
    drop the round operand; lossy casts lower as ``CAST_RINT``. ``CAST_HYBRID``
    selects native hybrid rounding for Float32 -> HiFloat8 only.
    """

    CAST_ROUND = "cast_round"  # round to nearest, tie away from zero
    CAST_FLOOR = "cast_floor"  # round toward -inf
    CAST_CEIL = "cast_ceil"  # round toward +inf
    CAST_TRUNC = "cast_trunc"  # round toward zero
    CAST_NONE = "cast_none"  # AscendC CAST_NONE: lossless no-rnd / lossy RINT
    CAST_RINT = "cast_rint"  # round to nearest, tie to even (banker's)
    CAST_ODD = "cast_odd"  # Von Neumann rounding (round to odd)
    CAST_HYBRID = "cast_hybrid"  # native HiF8 hybrid rounding (Float32 -> HiFloat8)

    def __str__(self) -> str:
        return self.value


class ScalarRoundMode(enum.Enum):
    """Rounding for a scalar float32 conversion (``Numeric.to``).

    Distinct from :class:`RoundMode`, which configures the AVE vector cast:
    these are the scalar unit's own conversion instructions, and the two sets
    are not interchangeable -- the scalar unit takes none of the AVE cast's
    saturation or register-slot knobs. Vector ``CAST_RINT`` is the AVE peer of
    scalar ``NEAREST_EVEN``.
    """

    NEAREST_EVEN = "nearest_even"  # ties to even      (conv_f322s32r)
    NEAREST_AWAY = "nearest_away"  # ties away from 0  (conv_f322s32a)
    FLOOR = "floor"  # toward -inf       (conv_f322s32f)
    CEIL = "ceil"  # toward +inf       (conv_f322s32c)
    # There is no conv_f322s32z: the scalar unit has no toward-zero
    # conversion instruction, so this is arith.fptosi rather than a helper.
    TRUNC = "trunc"  # toward zero       (arith.fptosi)
    # f32 -> f16 only, and the only mode that conversion has. Round to odd picks
    # the neighbour with an odd mantissa whenever the value is not exactly
    # representable; it exists to avoid double rounding when the result is
    # narrowed again. It is NOT the default narrowing -- plain .to(Float16)
    # gives correct round-to-nearest-even and should stay the usual choice.
    ODD = "odd"  # to odd            (conv_f322f16o)

    def __str__(self) -> str:
        return self.value


_REG_SLOT_CODE = {RegSlot.ZERO: 0, RegSlot.ONE: 1, RegSlot.TWO: 2, RegSlot.THREE: 3}
_SAT_MODE_CODE = {SatMode.UNKNOWN: 0, SatMode.SAT: 1, SatMode.NOSAT: 2}
_ROUND_MODE_CODE = {
    RoundMode.CAST_ROUND: 0,
    RoundMode.CAST_FLOOR: 1,
    RoundMode.CAST_CEIL: 2,
    RoundMode.CAST_TRUNC: 3,
    RoundMode.CAST_NONE: 4,
    RoundMode.CAST_RINT: 5,
    RoundMode.CAST_ODD: 6,
    RoundMode.CAST_HYBRID: 7,
}


@dataclass(frozen=True)
class CastParams:
    """The three knobs of a ``tla.cast`` (``VectorSSA.to``).

    Args:
        reg_slot: destination packed-register position (:class:`RegSlot`).
        sat_mode: overflow behaviour (:class:`SatMode`).
        round_mode: rounding on precision loss (:class:`RoundMode`).
    """

    reg_slot: RegSlot = RegSlot.ZERO
    sat_mode: SatMode = SatMode.NOSAT
    round_mode: RoundMode = RoundMode.CAST_ROUND

    def codes(self) -> list[int]:
        """The three enum codes for the tla.cast DenseI32ArrayAttr, in order."""
        return [
            _REG_SLOT_CODE[self.reg_slot],
            _SAT_MODE_CODE[self.sat_mode],
            _ROUND_MODE_CODE[self.round_mode],
        ]


class HistogramsBinType(enum.Enum):
    """Which 128-lane half of the u8 domain ``tla.histograms`` writes.

    Semantics depend on ``HistogramsParams.type`` (see :class:`HistogramsParams`).
    """

    BIN0 = "bin0"
    BIN1 = "bin1"

    def __str__(self) -> str:
        return self.value


class HistogramsType(enum.Enum):
    """Histogram kind selected by ``HistogramsParams.type``.

    ``FREQUENCY`` — per-value counts; ``ACCUMULATE`` — prefix / CDF counts.
    """

    FREQUENCY = "frequency"
    ACCUMULATE = "accumulate"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class HistogramsParams:
    """Knobs for ``tla.histograms``: ``mode`` (half) and ``type`` (kind).

    Args:
        mode: which 128-lane half (:class:`HistogramsBinType`) — ``BIN0`` / ``BIN1``.
        type: FREQUENCY / ACCUMULATE (:class:`HistogramsType`).

    Math (masked ``src`` lanes only; result is added onto prior ``dst``):

    * **FREQUENCY** — per-value counts in a half-range:
      - ``BIN0``: ``dst[v] += #{src == v}`` for ``v ∈ [0, 127]``
      - ``BIN1``: ``dst[v - 128] += #{src == v}`` for ``v ∈ [128, 255]``
    * **ACCUMULATE** — CDF over the full u8 domain, then split:
      - ``BIN0``: ``dst[n] += #{src <= n}`` for ``n ∈ [0, 127]``
      - ``BIN1``: ``dst[n] += #{src <= 128 + n}`` for ``n ∈ [0, 127]``
        (includes low-half inputs; not a ``[128, 255]``-only bucket)
    """

    mode: HistogramsBinType
    type: HistogramsType


class MemType(enum.IntEnum):
    VEC_STORE = 0
    VEC_LOAD = 1
    SCALAR_STORE = 2
    SCALAR_LOAD = 3
    VEC_ALL = 4
    SCALAR_ALL = 5
