# Phases 8–10 — SCREEN, CONFIRM-OPT, APPLY: the `L0C → UB` rewrite

This is one screened rewrite, not a tuning program. It applies only to a **non-TLA, CV-fused** migrated unit whose selected C path is still `L0C → GM → UB`; on 3510, Fixpipe can instead deliver C from L0C to UB. The workflow proves applicability from the migrated unit, obtains a second human decision, and only then changes code.

The only fixed-code illustrations in this document are from [CATLASS](https://gitcode.com/cann/catlass) at `89e1fc39881a715882b9b47459add06ba270105c` and [asc-devkit](https://gitcode.com/cann/asc-devkit) at `512674e996da0feaee6e7f435e4efc1cad1d74fb`. They identify roles and invariants; the selected paths, symbols, build entry, registration surfaces, and architecture gates must be reinvestigated in the target tree and recorded in the run artifacts.

**Out of scope:** tile reshaping, swizzle changes, scheduling, autotuning, `SPLIT_N`, block-count tuning, and core-count tuning. `SPLIT_M` versus `NO_SPLIT` and the minimum safe buffering are data-path correctness decisions, not generic tuning knobs. `NO_SPLIT` refers to the actual runtime M interval, including any tail; the compile-time split tile remains even.

`references/08-l0c-to-ub-rewrite.md` is the implementation guide. Do not read or use its implementation steps until GATE 2 authorizes the manifest.

## Boundaries and decisions

| Phase | Actor | Reads or writes | Outcome |
|---|---|---|---|
| 8 — SCREEN | you | reads the migrated unit; writes only run artifacts and profiler output beneath the run directory | `OPT_SCREENED` |
| 9 — CONFIRM-OPT / GATE 2 | human | reads the screen packet; authorizes apply or skip, and `coexist` or `replace` when applying | `OPT_AUTHORIZED` for authorized work |
| 10 — APPLY | you and one Optimizer per unit | writes only the GATE-2-authorized manifest | `OPTIMIZED` only after the ordered checks below |

Optimization is enabled by default at FRAME. A campaign can opt out globally only in its frozen plan. In an enabled campaign, GATE 1 may set aside individual units early with migration `confirm --skip-optimize <ids>`; that records an optimization skip without changing the migration result. SCREEN runs only for a `PROVEN` unit whose proof is **current**: no later `unit.prove_failed`, and the latest `unit.proven` has `errors: 0` plus the exact declared matching Ascend950 identity (`arch: "3510"`, A5-family SoC, nonempty host/device). A historical `PROVEN` rank without that proof is not eligible. This condition remains mandatory through GATE 2, APPLY, and `OPTIMIZED`; a later matching proof restores it without regressing rank. GATE 2 then decides, per screened applicable unit, whether to apply or skip and, if applying, whether the authorized strategy is `coexist` or `replace`.

`--intent` stores the final human decision **verbatim**. It must state the apply/skip decision and, when applying, the chosen strategy. A partial, conditional, or ambiguous reply is a question, not authorization. `replace` additionally requires explicit assent because it removes the baseline path.

| Strategy | Authorized tree state | Consequence |
|---|---|---|
| `coexist` | baseline and direct paths both exist; the rewrite is the default path | the baseline remains buildable and `optimize.json` must record its non-empty profiler sample remeasured in the same session as the direct path |
| `replace` | only the direct path remains | the previous path is removed; GATE 2 must explicitly authorize that removal |

GATE 2 snapshots the strategy and exact `screen.manifest` paths. That snapshot is the whole write grant. If an implementation reveals that the manifest is wrong, stop, correct the SCREEN artifact, re-run `check --phase screened`, re-render GATE 2, and obtain a new decision; do not widen the grant while writing.

## SCREEN

SCREEN is read-only with respect to the target source tree. It has three products: a compact committed-TLA preflight, five applicability readings, and a repeated baseline measurement. Only after those are complete may it propose a strategy and manifest.

### Compact committed-TLA manifest preflight

Before tracing the non-TLA unit, inspect the **committed TLA** `L0C → UB` implementation at the CATLASS pin. Record this compact role manifest in the SCREEN working record (and summarize it in `screen.notes`); it is a preflight, **not** the eventual write grant.

| Committed TLA role to locate | Why it is read |
|---|---|
| direct-to-UB tile-copy primitive and its Fixpipe configuration | establishes the two destination modes and their explicit destination control |
| tile-copy selector or aggregate that exposes that primitive | identifies the C-side selection seam |
| `BlockMmad` specialization with `callbackBeforeFixpipe` and `callbackAfterFixpipe` | identifies the Cube-side producer and the callback handshake |
| epilogue specialization that consumes the UB C tile | identifies the Vector-side consumer and its buffer contract |
| kernel that connects the producer, consumer, and cross-core protocol | identifies seed, steady-state, and drain ownership |
| aggregator or registration surface that makes the specializations selectable | identifies the registration role that must be included if the target tree has an equivalent surface |

Treat the pinned TLA implementation as an illustration of **roles**, not as a list of target-tree filenames. Trace each role from the migrated example's actual includes and aliases. The result prevents authorizing a manifest that names an incompatible plain `BlockMmad` when the selected direct path requires the callback-capable form.

### The five applicability rows

Read the five rows in order from the migrated unit. Each `screen.rows` value must be a current reading: the selected file, symbol or source location, what was observed, and the resulting verdict. Do not copy a prior run's wording. All five rows are required even when an earlier row makes the unit inapplicable.

| Row | Establish from the selected migrated path | Pass condition |
|---|---|---|
| `gemm_family` | example aliases and the C tile shape | a live `BlockMmad` alias and a `GemmShape`-family C tile with meaningful M and N axes |
| `epilogue` | the selected `BlockEpilogue` alias | a non-`void` epilogue consumes C after Cube work |
| `non_tla` | example, selected kernel, block, and epilogue | the C producer and consumer use the non-TLA `LocalTensor`/`layout` path; no TLA or mixed C path is substituted by resemblance |
| `block_mmad` | the `BlockMmad` specialization actually selected | it still sends the relevant C tile from L0C to GM; a direct-to-UB specialization means this rewrite is already present, not applicable |
| `block_epilogue` | the matching selected epilogue specialization | it still reads that C tile from GM into UB; absence of that relay means the rewrite is already present or the data flow differs |

The rewrite is applicable only if every row passes and the two traced relay legs describe the **same C path**. A `Gemv`, convolution, attention, pure-Cube, TLA, already-direct, or otherwise nonmatching stack is a valid terminal SCREEN result: set `applicable: false`, `strategy: null`, and `manifest: []`, retain all readings and the baseline, and do not send it to GATE 2 for code authorization.

The rows are necessary, not sufficient. Also reject a unit when the selected epilogue couples M halves and no safe `NO_SPLIT` budget exists, when the direct destination layout cannot meet its transfer alignment, or when the selected stack needs a different dataflow. Record the discovered reason; do not invent an extra route or adapt a source reading from another unit.

### Manifest and tiers

For an applicable unit, trace from the migrated example through selected includes, aliases, policies, and registration surfaces. `screen.manifest` lists every path the rewrite would add or modify. It includes only paths actually touched, categorized by role rather than a memorized tree layout:

- the example and, when needed, its build/variant selection;
- the selected baseline kernel and the direct-path kernel;
- the selected `BlockMmad` and direct-path counterpart;
- the selected epilogue and direct-path counterpart;
- the non-TLA direct tile-copy primitive, its selector, and necessary dispatch-policy surfaces;
- the aggregator or registration surface that exposes any new specialization.

Every row is `add` or `modify` and has a tier:

- **Tier 1 declarations:** reusable, operator-agnostic primitive, selection, or policy declarations. The lead lands a needed declaration once per tree before a unit's Tier 2 code.
- **Tier 2:** the unit-specific blocks, kernel, and example selection. An Optimizer may write only these authorized rows.
- **Tier 1 registrations:** includes or registration entries that expose Tier 2 code. The lead lands them only after the corresponding Tier 2 header exists.

A path that is already present is not automatically a manifest row; inspect whether this rewrite must modify it. Conversely, no projected role is optional once it is actually touched. The manifest becomes a write grant only after GATE 2.

### Repeated baseline measurement

Every migrated unit with a current hardware-bound `PROVEN` proof receives a baseline measurement, applicable or not. Use the discovered build and run entries; never assemble an invocation from a target name. Put profiler output under that unit's run-directory logs and supply an explicit output directory for every profiler invocation.

Take repeated launches of the unchanged baseline configuration, normally two or three, into the **same** measurement source directory. Transcribe every duration in `screen.baseline.task_us` as a list. The directory must contain one profiler result for each list element; the result count makes repetitions auditable. A remote 3510 device is measured through its declared transport and the profiler output is fetched beneath the same unit log directory before SCREEN is checked.

`screen.baseline` is the baseline configuration, duration list, and source directory. It is a measurement record, not a claim of improvement. Do not estimate missing measurements, reuse measurements from another run, or profile in the target tree.

## GATE 2

Run the optimization gate only after SCREEN has passed and each candidate still has a current
hardware-bound proof. The packet shows, for each unit, the five current readings, TLA preflight summary,
measured baseline, proposed strategy, and exact manifest. A later `unit.prove_failed` or an unbound or
mismatched latest `unit.proven` withdraws the unit until it is re-PROVEN; its rank remains historical.
The human chooses apply or skip; an apply also chooses `coexist` or `replace`. The gate authorizes a
description of work, not an assumed result. A unit skipped at GATE 2 retains its screened result and
baseline; it is not rewritten. If the same campaign returns to an already authorized unit because its
manifest changed, it must be re-screened and re-authorized as above.

## APPLY — only after GATE 2

Now, and only now, open `references/08-l0c-to-ub-rewrite.md` and implement the authorized direct path.
Before APPLY, the SCREEN/GATE-2 proof must still be current; a historical `PROVEN` rank cannot authorize
the write. The ordered boundary is:

1. The lead lands needed Tier 1 declarations, once per tree and only at authorized paths.
2. One Optimizer writes a unit's Tier 2 implementation at its authorized paths.
3. The lead lands Tier 1 registrations after the referenced Tier 2 headers exist.
4. Build the selected configuration and take the repeated rewritten-path measurement. Under `coexist`, also re-measure the baseline variant in the same session and record its non-empty profiler sample beside the direct-path sample in `optimize.json`; `check --phase applied` refuses a coexist artifact that lacks it. A `replace` artifact requires only the direct-path sample.
5. Run `mig.py check --run-dir <run-dir> --phase applied --unit <id>`.
6. Re-run `mig.py prove --run-dir <run-dir> --unit <id>` **unchanged**. The default selected build must be the rewrite, and the run uses the frozen golden, shape, comparator, and evidence contract.
7. Run `mig.py check --run-dir <run-dir> --phase optimized --unit <id>`.

The order is material: **APPLY and measurement → `check applied` → unchanged `prove` → `check optimized`**. `check applied` records the declared landing and validates the artifact; the following proof is the accuracy evidence for that landing; `check optimized` requires that proof after the apply event to be current and hardware-bound. A compile or profiler result is never an accuracy result.

`optimize.json` records the chosen `SPLIT_M` or `NO_SPLIT` mode, the written manifest paths, and repeated profile samples. Each `task_us` may be a list; every list element needs a matching profiler result in the sample's source directory. A `coexist` artifact must include the remeasured same-session baseline beside the direct-path sample; both samples are validated for non-empty durations and profiler output. A `replace` artifact records the direct-path sample and relies on SCREEN for the removed baseline, so it need not include a baseline field.

## Non-negotiable limits

- Do not change the frozen contract, oracle, golden, tolerance, compared region, dtype, layout, scale, mask, aliasing, or supported domain to make this rewrite fit.
- Do not select `SPLIT_N`, alter tile/block/core counts, or attach unrelated performance tuning to this rewrite.
- Do not add an unreviewed path, include, registration entry, or build variant outside the authorized manifest.
- Do not treat a TLA filename, a direct-path policy elsewhere in the tree, a successful compile, or a profiler duration as proof that the selected non-TLA path is direct or correct.
- Do not report an improvement from one sample, an old run, or a generated estimate. The report presents measurements; it does not turn them into a tuning verdict.
