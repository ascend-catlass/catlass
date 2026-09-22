# `L0C → UB` rewrite guide — post-GATE-2 implementation

Read this guide **only after GATE 2 authorizes an applicable unit's manifest and strategy**. SCREEN, the committed-TLA preflight, the five applicability rows, measurement, GATE 2, and the `coexist`/`replace` decision belong to `references/07-optimize.md`. This guide does not create a manifest, choose a strategy, or authorize a write.

The fixed illustrations below come only from [CATLASS](https://gitcode.com/cann/catlass) at `89e1fc39881a715882b9b47459add06ba270105c`. They are role models: resolve the corresponding committed target-tree symbols before editing. Do not copy a path, constant, flag range, tile value, or source reading from another unit.

## Build the direct path by roles

The committed CATLASS TLA direct-path implementation is the structural reference. Follow the roles, not a filename:

1. a direct-to-UB tile-copy primitive uses Fixpipe with an explicit destination mode;
2. a selector exposes that primitive to the C-side tile copy;
3. a callback-capable `BlockMmad` waits for a free UB slot before Fixpipe and signals that its Cube write is complete afterwards;
4. a matching epilogue reads the C tile from UB and returns a slot only after Vector consumption;
5. the kernel initializes, advances, and drains the cross-core protocol; and
6. an aggregator or registration surface makes the selected specializations reachable.

The non-TLA implementation follows the same dependency direction. Derive the new `BlockMmad` from the selected callback-capable role when that is the seam the direct path needs; a superficially similar plain block is not interchangeable. Preserve the frozen source Kernel/Block/Epilogue/Scheduler behavior except for the explicitly authorized C transport and the required synchronization it entails.

Do not add a duplicate Fixpipe configuration, selector, policy, or aggregator registration if the target tree already supplies the resolved role. Conversely, do not assume a role is available because a different operator or TLA path has one.

## Choose the in-scope mode from allocation arithmetic

Only two modes are in scope.

| Mode | Direct-path behavior | Per-AIV C ring | Synchronization |
|---|---|---|---|
| `NO_SPLIT` | one AIV sub-block receives the full C tile | `M × N × sizeof(ElementC)` per stage | one credit per ring slot |
| `SPLIT_M` | Fixpipe partitions the C tile across AIV sub-blocks | `(M / 2) × N × sizeof(ElementC)` per stage | paired credits per ring slot |

Calculate both candidates for the selected unit:

```text
split_factor    = 2 for SPLIT_M; 1 for NO_SPLIT
c_stage_bytes   = (tile_M / split_factor) * tile_N * sizeof(ElementC)
total_ub_bytes  = c_stage_bytes * UB_STAGES
                  + epilogue buffers
                  + other live UB intervals

require total_ub_bytes <= ArchTag::UB_SIZE
```

The producer allocation, consumer row pitch, and compile-time UB assertion must use the same byte interval. The assertion is a safety condition, not a substitute for reconciling those three expressions. Verify the destination's row-width and alignment preconditions from the selected copy primitive and assert them where the type values live.

`SPLIT_M` is the normal choice only when its arithmetic fits and the epilogue is M-decomposable. `NO_SPLIT` is required when the epilogue needs the full M range or when the split path cannot meet its contract. Neither choice permits retile, block-count, or core-count tuning. **`SPLIT_N` is excluded:** do not import its rounding, destination control, or flag protocol into either in-scope mode.

### Compile-time tile M is not the runtime tail M

For `SPLIT_M`, the **compile-time tile M** supplied to the direct tile-copy specialization must be even. Assert that type-level property where the specialization is formed. It does **not** require the runtime problem's M dimension, nor the final runtime block's actual M extent, to be even.

A final runtime tile may contain an odd `blockM` even when `tile_M` is even. Split that actual interval with the same ceiling division used by the producer: the first AIV receives the extra row, and the other receives the remainder. Use the runtime sub-block count reported by the execution environment; do not hard-code it. The C data already arrives as each AIV's compact local view, so do not apply a second software M slice to that UB C view.

All GM tensors touched by the epilogue need separate classification:

| Tensor class | `SPLIT_M` treatment |
|---|---|
| M-indexed output or input | offset both GM base and layout by that AIV's actual row interval |
| M-invariant, such as a channel-wise N vector | preserve the full M-independent interpretation; offset only on axes it actually indexes |
| M-coupled reduction or broadcast | cannot use `SPLIT_M` unless the coupling is redesigned outside this rewrite; use a safe `NO_SPLIT` path or mark the unit inapplicable |

A zero-row runtime half is still a synchronization participant. It may skip data processing, but it must not skip the wait, set, release, seed, or drain actions that keep the cross-core credits balanced. Prove an input that exercises an odd final block when hardware access is available.

## Cross-core protocol

The protocol belongs to the selected kernel's local conventions. Reinvestigate its flag bases, valid range, synchronization mode, and pipe ownership from the committed matching implementation. There is no repository-wide flag census to import and no reason to create a shared constant solely for this rewrite.

The invariant is independent of local names:

- `NO_SPLIT` has one producer credit and one consumer action per ring slot. It must not reference a second-sub-block alias.
- `SPLIT_M` has one credit from each AIV sub-block per ring slot. The Cube side waits for both credits before reusing the slot and signals completion to both consumers. Each AIV uses the base identity appropriate to its local view; Cube performs the paired accounting.
- The signal and wait pipes order the operation that actually produced or consumed the data. Fixpipe completion and Vector work are not interchangeable pipe events.
- The initial seed gives every ring slot enough free credits for the first Cube writes. The final drain consumes exactly the credits remaining after steady state.

Express the local flag-range and stage constraints as compile-time checks next to the implementation when the selected APIs permit it. Such checks protect identifier overlap; they do not prove token balance, order, or liveness. A device proof remains required.

## Buffer lifetime and epilogue adaptation

The epilogue's direct-path specialization must agree with the producer on all of the following:

- the per-stage C interval and stage count;
- `SPLIT_M`'s halved C view or `NO_SPLIT`'s full C view;
- the direct path's destination layout and row pitch;
- the event protocol that releases a UB slot only after its Vector work completes; and
- each GM operand's M-indexing classification.

Keep address-space conversions and parameter bridging in the style already used by the selected kernel family. In particular, retain the existing ABI ownership boundary for argument pointers, layouts, and parameter conversion; do not introduce an ad hoc address-space cast just because a direct-path sibling happens to have a similar tensor list. Trace per-token, scale, and output tensors to their actual tile-copy contracts before changing their layouts or offsets.

## Strategy-specific selection

The GATE-2 decision is already fixed; implement it exactly.

- **`coexist`:** add a distinct baseline selection and a distinct direct-path selection using the target tree's discovered build/selection convention. The direct path is the default that unchanged `prove` executes. Confirm that the two selected configurations are observably distinct before treating their measurements as a comparison.
- **`replace`:** select the direct path in place and remove only the baseline aliases, includes, or registrations named by the manifest. Do not leave a hidden fallback or reintroduce a second decision path after the explicit replacement authorization.

The build file or selection mechanism is a manifest path when it changes. A compiler definition, cache variable, or target name is not assumed portable between target trees; reuse the convention identified in `profile.json`.

## Implementation invariants

Before handing the finished tree back to the workflow, inspect the authorized diff against these invariants:

1. Fixpipe still performs the C transport; only its destination and the required partitioning changed.
2. The selected direct tile copy, `BlockMmad`, epilogue, kernel, and registrations form one resolvable type stack.
3. `SPLIT_M` has an even compile-time tile M assertion, while runtime odd tails use ceiling-partitioned actual rows.
4. Every stage has balanced seed, steady-state, and drain credits, including a zero-row AIV half.
5. The producer and consumer agree on byte offsets, stage count, and UB budget.
6. No path outside the GATE-2 manifest changed.
7. No oracle, golden, comparator, tolerance, tensor contract, tile/block/core count, or `SPLIT_N` behavior changed.

After implementation, return to the workflow: take repeated measurements, write `optimize.json`, then run **`check applied` → unchanged `prove` → `check optimized`**. Do not replace those steps with a manual build, a compile result, or a success string.
