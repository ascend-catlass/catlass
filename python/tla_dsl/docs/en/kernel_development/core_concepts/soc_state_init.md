---
nav_order: 45
---

# Kernel-Entry SoC State Initialization

[Core Concepts](index.md) | [Kernel Entry](../../api/runtime/kernel_entry.md)

## Overview

Every device kernel entry **automatically** initializes SoC / control state so
that leftover settings from a previous operator cannot leak in. There is no
separate initialization API to call, but the entry does rewrite the states
listed below; developers should treat this document as the post-entry hardware
convention when writing kernels.

Overall order:

```text
1) SoC state reset (atomic mode, mask, dequant scale, CTRL, padding / vector mask, …)
2) user kernel body
3) exit: restore mmad-related CTRL and ensure trailing pipeline sync
```

Applies to: Ascend 950.

---

## AIC entry initialization

| Order | Action | State changed / effect |
| --- | --- | --- |
| 1 | Disable stream atomic mode | Atomic mode = none; later GM stores without atomic semantics overwrite instead of accumulate |
| 2 | Normalize mask mode | Mask working mode = normal |
| 3 | Reset dequant default scale | Default scale = 1 (half); affects dequant / related cast paths that use this default |
| 4 | Bulk CTRL reset | Whole-register write `0x1000000000000008 \| (old CTRL & bit48)`: **CTRL[60]=1**, **CTRL[48]=previous (0 or 1)**, **CTRL[3]=1**, other bits 0 |
| 5 | Clear padding | `padding = 0`; avoid dirty padding affecting LoadData |
| 6 | Reset store-atomic config | `st_atomic_cfg = 0b00100100` |

AIC does **not** change: vector mask or UB↔OUT loop-mode parameters.

---

## AIV entry initialization

| Order | Action | State changed / effect |
| --- | --- | --- |
| 1 | Disable stream atomic mode | Atomic mode = none; normal UB→GM stores overwrite |
| 2 | Normalize mask mode | Mask working mode = normal |
| 3 | Reset dequant default scale | Default scale = 1 (half) |
| 4 | Bulk CTRL reset | Same as AIC: CTRL[60]=1; CTRL[48] preserved; other bits rewritten from the constant |
| 5 | Open full vector mask | `vector_mask = (-1, -1)`; avoid a partial mask dropping elements |
| 6 | Reset UB↔OUT loop parameters | `loop_size_ubtoout` / `loop_size_outtoub = (1<<21)\|1` |
| 7 | Reset store-atomic config | `st_atomic_cfg = 0b00100100` |

AIV does **not** change: padding (AIC only).

---

## Implicit CTRL updates for DSL APIs

InitSoc supplies the entry CTRL defaults. When the following APIs depend on CTRL,
the compiler **automatically** adjusts the relevant bits; there is no user-facing
CTRL API. If the entry default already matches a call's need, nothing extra is
written. Repeated identical needs in the same region with no intervening opposite
requirement take effect only once.

| DSL API | Implicit CTRL effect |
| --- | --- |
| `tla.cast` / `CastParams.sat_mode` (SAT / NOSAT / UNKNOWN) | Clear CTRL[60] so the instruction sat operand applies (entry default CTRL[60]=1 is global override) |
| `tla.mmad` / `tla.mmad_mx` `compute_order=M_FIRST` | Requires CTRL[51]=0 (matches entry default; usually no write) |
| `tla.mmad` / `tla.mmad_mx` `compute_order=N_FIRST` | Set CTRL[51]=1 |
| `tla.mmad` `hf32_mode=HF32_DISABLE` | Requires CTRL[46]=0 and CTRL[47]=0 (matches entry default; usually no write) |
| `tla.mmad` `hf32_mode=HF32_NEAREST_ZERO` | Set CTRL[46]=1 and CTRL[47]=1 |
| `tla.mmad` `hf32_mode=HF32_NEAREST_EVEN` | Set CTRL[46]=1; leave CTRL[47]=0 |

Bits unrelated to the APIs above (covered by entry InitSoc instead):

- **CTRL[48]**: preserved by the bulk write; casts use instruction sat once CTRL[60]=0
- **CTRL[56]**: superseded by InitSoc `mask_norm` / `vector_mask`; no DSL parameter

---

## Exit (same for AIC / AIV)

| Action | State changed / effect |
| --- | --- |
| Clear CTRL[51] | Restore mmad M/N compute-direction priority to the hardware default |
| Clear CTRL[46] | Disable mmad HF32 rounding-mode enable |
| Clear CTRL[47] | Disable mmad HF32 rounding-mode select |
| Trailing `PIPE_ALL` barrier | Ensure exit pipeline synchronization |
