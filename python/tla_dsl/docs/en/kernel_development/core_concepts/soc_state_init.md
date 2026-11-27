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
| 1 | Disable stream atomic mode | GM stores overwrite while atomic mode is disabled |
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
| 1 | Disable stream atomic mode | UB→GM stores overwrite while atomic mode is disabled |
| 2 | Normalize mask mode | Mask working mode = normal |
| 3 | Reset dequant default scale | Default scale = 1 (half) |
| 4 | Bulk CTRL reset | Same as AIC: CTRL[60]=1; CTRL[48] preserved; other bits rewritten from the constant |
| 5 | Open full vector mask | `vector_mask = (-1, -1)`; avoid a partial mask dropping elements |
| 6 | Reset UB↔OUT loop parameters | `loop_size_ubtoout` / `loop_size_outtoub = (1<<21)\|1` |
| 7 | Reset store-atomic config | `st_atomic_cfg = 0b00100100` |

AIV does **not** change: padding (AIC only).

---

## Implicit CTRL updates for DSL APIs

The compiler configures the relevant CTRL bits from the arguments to `tla.mmad`
and `tla.mmad_mx`. It can write a uniform setting once after kernel entry
initialization. When a setting varies, the compiler writes it before each MMAD.
In functions that access CTRL explicitly or contain calls that may modify CTRL,
each MMAD sets the bits it needs.

| DSL API | Implicit CTRL effect |
| --- | --- |
| `tla.mmad` / `tla.mmad_mx` `compute_order=M_FIRST` | Clear CTRL[51] |
| `tla.mmad` / `tla.mmad_mx` `compute_order=N_FIRST` | Set CTRL[51]=1 |
| `tla.mmad` `hf32_mode=HF32_DISABLE` | Clear CTRL[46] and CTRL[47] |
| `tla.mmad` `hf32_mode=HF32_NEAREST_ZERO` | Set CTRL[46]=1 and CTRL[47]=1 |
| `tla.mmad` `hf32_mode=HF32_NEAREST_EVEN` | Set CTRL[46]=1 and clear CTRL[47] |

`tla.cast` preserves the current saturation strategy (CTRL[60]). To use the
instruction's saturation mode, explicitly select `USE_API` with
`tla.arch.set_ctrl`. Under the global saturation strategy, CTRL[48] retains its
pre-initialization value until explicitly written.

In functions that access CTRL explicitly, plain `tla.mmad` with two E4M3 inputs
selects E4M3 interpretation (CTRL[45]).

Copies to GM with the default `atomic_mode=NONE` use the current atomic state.
If you enable atomic writes with `tla.arch.set_ctrl`, set
`atomic_format=AtomicFormat.DISABLED` before an overwrite.

---

## Exit (same for AIC / AIV)

| Action | State changed / effect |
| --- | --- |
| Clear CTRL[51] | Restore mmad M/N compute-direction priority to the hardware default |
| Clear CTRL[46] | Disable mmad HF32 rounding-mode enable |
| Clear CTRL[47] | Disable mmad HF32 rounding-mode select |
| Trailing `PIPE_ALL` barrier | Ensure exit pipeline synchronization |
