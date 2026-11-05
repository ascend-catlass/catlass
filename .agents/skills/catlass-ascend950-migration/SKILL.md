---
name: catlass-ascend950-migration
description: "Migrates CATLASS operators from Atlas A2 and Atlas A3 (CATLASS_ARCH=2201) to Ascend 950/A5 (CATLASS_ARCH=3510) in a gated, evidence-led campaign. Use when asked to migrate, port, or retarget CATLASS operators to Ascend 950/A5/3510."
---

# CATLASS A2/A3 → Ascend 950 migration

Migrate one or more operators as one campaign, **one unit at a time**. A unit is one source example
and one target example. A campaign is a cluster of units sharing one dispatch policy; run only one
campaign per `target_root` at a time.

The target must be a CATLASS-family tree. Its build entry, architecture gates, registration surfaces,
naming, golden workflow, test surface, and generated artifact paths are discovered from that tree in
PROBE and recorded in `profile.json`; never infer them from a name, a memory, or another checkout.

All generated state is under `<target_root>/.agents-work/`: campaign state at
`.agents-work/catlass-ascend950-migration/<run-id>/` and pinned references at
`.agents-work/.cache/refs/`. `--run-dir` may name a run inside that fence only. The ignore rule belongs
in `$GIT_DIR/info/exclude`, never the tracked `.gitignore`; do not write generated state elsewhere.

## Sources, decisions, and evidence

### Pinned source before unknown

The target tree is the authority for target-specific observations. Before making a fixed claim that the
target tree cannot establish, run `mig.py refs --run-dir <run-dir>` and inspect only these pinned
sources:

| Source | Pin |
|---|---|
| CATLASS — <https://gitcode.com/cann/catlass> | `89e1fc39881a715882b9b47459add06ba270105c` |
| asc-devkit — <https://gitcode.com/cann/asc-devkit> | `512674e996da0feaee6e7f435e4efc1cad1d74fb` |

If the relevant source or target evidence is unavailable, record the matter as **unknown** or a
blocker. Do not promote a local installation, an unpinned revision, a search result, or recollection
into a fixed fact. Cite the applicable pinned source in an artifact whenever it supplies the fact.

### Three human decisions; two write gates

There are three human decision points:

1. **FRAME (Phase 1)** freezes scope, declared access, the performance input, and the global
   optimization choice. It is a decision point, **not a gate** and authorizes no target source,
   build, or registration write.
2. **CONFIRM (Phase 4, GATE 1)** authorizes migration writes only after the route, frozen contract,
   ledger, and write grant are visible.
3. **CONFIRM-OPT (Phase 9, GATE 2)** authorizes the L0C→UB rewrite only after applicability,
   baseline measurement, strategy, and manifest are visible.

No target source, build, or registration write happens before the gate that describes it. PROBE,
ANALYZE, and SCREEN only read the target tree; FRAME may create campaign metadata under
`.agents-work/`. `--intent` stores the final human decision **verbatim**; never summarize, normalize,
reinterpret it in the recorded command. Questions, conditional replies, and ambiguity are not approval:
resolve them, re-render the appropriate packet, and record the final decision exactly as given.

### Declared repair and evidence boundary

A structural repair may be declared: for example, correcting an artifact's schema, a path boundary, a
missing required field, or an invalid manifest. Declare the repair, its reason, and its affected
artifact before using it. A repair is not evidence.

Routes, source readings, rationales, measurements, profiler results, and optimization strategies are
not repairable prose. Reinvestigate them from the target tree, a declared measurement, or the pinned
sources. Never author, copy forward, estimate, or relabel such evidence to make a phase pass.

## Actors and command discipline

A subagent may write only inside one unit's granted target paths during IMPLEMENT or APPLY. It must not
edit shared headers, registration surfaces, A2 sources, or campaign artifacts, and it never builds or
runs the device. It returns proposed registration text; you land shared declarations and registration
changes, and you run all builds, proofs, and measurements.

`init` is the only command without `--run-dir`; it prints the run directory to reuse. Every other
command requires that same `--run-dir`. `mig.py status --run-dir <run-dir>` is the resume authority:
follow its next command for each unit. Re-running a command must preserve recorded decisions rather
than regress a phase.

## Phases

| # | Phase | Actor | Required action |
|---|---|---|---|
| 1 | FRAME | you + human | `mig.py init --plan plan.json [--access access.json] [--perf-cases <path>]` |
| 2 | PROBE | you | `mig.py profile --run-dir <run-dir>` |
| 3 | ANALYZE | you | `mig.py check --run-dir <run-dir> --phase analyzed` |
| 4 | **CONFIRM / GATE 1** | **human** | `mig.py gate --run-dir <run-dir>` → `mig.py confirm --run-dir <run-dir> --intent '<final human decision, verbatim>' [--exclude <ids>] [--skip-optimize <ids>]` |
| 5 | IMPLEMENT | you + one subagent per unit | `mig.py check --run-dir <run-dir> --phase implemented --unit <id>` |
| 6 | COMPILE | you | `mig.py prove --run-dir <run-dir> --unit <id> --compile-only` |
| 7 | PROVE | you | `mig.py prove --run-dir <run-dir> --unit <id>` |
| 8 | SCREEN | you | `mig.py check --run-dir <run-dir> --phase screened --unit <id>` |
| 9 | **CONFIRM-OPT / GATE 2** | **human** | `mig.py gate --run-dir <run-dir> --phase optimize` → `mig.py confirm --run-dir <run-dir> --phase optimize --intent '<final human decision, verbatim>' [--exclude <ids>]` |
| 10 | APPLY | you + one subagent per authorized unit | apply and measure → `mig.py check --run-dir <run-dir> --phase applied --unit <id>` → unchanged `mig.py prove --run-dir <run-dir> --unit <id>` → `mig.py check --run-dir <run-dir> --phase optimized --unit <id>` |
| 11 | REPORT | script | `mig.py report --run-dir <run-dir>` |

Phases 6 and 7 use the same `prove` command but record separate outcomes. Phase references in this
instruction are exact: Phase 4 is CONFIRM/GATE 1, Phase 9 is CONFIRM-OPT/GATE 2, and Phase 10 has the
required order shown above.

## Phase 1 — FRAME

Never delegate FRAME. Resolve only the requested candidates into existing source directories and
measure target names from the target tree's convention. Every named source becomes a unit; do not
silently exclude one. A possible target counterpart is an ANALYZE finding, not a scope filter.

If no requested operator resolves, ask for the list and show the available candidates. If a requested
source, target name, or request text is ambiguous, ask before `init`; never invent `request` or a unit.
Once scope resolves, issue one intake prompt covering these items:

| Item | Decision |
|---|---|
| unit list | Show the measured `source → target` list for correction; freeze the accepted list in `plan.json`. |
| access | Declare `a2` and `a5` reachability, SoC, host, measured device selector, and remote transport when applicable in `access.json`. `device` identifies the observed hardware or the selector used to reach it; it is not arbitrary prose. No reachable device is valid and never blocks compilation. |
| performance cases | Accept a valid supplied table by path or a valid human description. If none is supplied, stage `assets/perf_case_template.md` as `<run-dir>/perf_cases.md`. If supplied input is unreadable, malformed, contradictory, or cannot be translated without inventing cases or values, escalate it to the human; do not silently repair it or substitute the template. |
| optimization | Optimization is on by default. The only global opt-out is `plan.optimize.enabled: false` at FRAME; it is frozen at `init`. |

For every reachable side, declare the exact architecture (`2201` for `a2`, `3510` for `a5`), the
matching SoC family, and nonempty measured host and device values. `init`, `prove`, gates, remote
device operations, and optimization checks load that identity strictly. Only `status` and `report` load
older declarations leniently so their historical work remains readable; an old, missing, or mismatched
identity is **Not established**, not authority for device work. Repair it only from measured hardware,
then re-run `prove`; until then accuracy and optimization eligibility are not current. Never infer or
invent a missing identity.

A global opt-out skips Phases 8–10. In an optimization-enabled campaign, GATE 1 may set aside selected
otherwise-migrated units from optimization with `confirm --skip-optimize <ids>`; that is a per-unit
early optimization decision, not a migration exclusion. `--exclude <ids>` at GATE 1 instead leaves
those units unimplemented at ANALYZED.

Use a plan with the frozen source pins:

```json
{
  "version": 1,
  "request": "<the user's request, verbatim>",
  "target_root": ".",
  "refs": {
    "catlass": "89e1fc39881a715882b9b47459add06ba270105c",
    "asc-devkit": "512674e996da0feaee6e7f435e4efc1cad1d74fb"
  },
  "optimize": { "enabled": true },
  "units": [
    { "id": "10", "source": "examples/10_<op>", "target": "<measured target name>" }
  ]
}
```

Replace every placeholder. A target must not already exist. One unit maps one source example directory
to one target example directory. The access declaration decides only what later claims are possible;
it must reflect the declared environment and transport rather than an assumed device. With no reachable
A5 device, compilation remains required and the campaign may stop honestly at `COMPILED`.

Run `mig.py refs --run-dir <run-dir>` immediately after `init`, before recording any fixed external
claim or treating an unavailable fact as known. See `references/01-frame.md`.

## Phase 2 — PROBE

PROBE is read-only. Write `<run-dir>/profile.json` from the target tree with four explicit sections:
`build`, `golden`, `registration`, and `arch-gating`. Each section must answer its concern or record a
specific gap; absence is a finding, never a guessed default. Then run `mig.py profile`.

`registration.surfaces` is a typed list of `{path, symbol, required, why}`. Each path names an existing
file outside a unit write grant that must receive a registration change. The unit's own CMake file is
not a registration surface. Do not analyze source routes or create target directories in PROBE.

See `references/02-probe.md`.

## Phase 3 — ANALYZE

Analyze the whole cluster before the first migration write. Read each source stack—example, kernel,
block, tile copy, epilogue, build, and golden—and freeze the external contract in
`units/<id>/findings.json`. The file is evidence from a read-only investigation, not a design proposal.

For each unit, establish and record:

- A nullable `counterpart` verdict with field-by-field contract and type-stack evidence. A matching
  directory name alone is not evidence.
- The frozen tensors, dtype/layout/storage/aliasing, CLI, data generation, golden comparator,
  tolerance selection, the one `golden.compared_tensor` and its exact canonical
  `golden.compared_dtype` evidence label, build argv, run argv, and required evidence shape. The
  compared dtype is not inferred from storage or upgraded-golden precision; read and freeze the
  exact string the example will emit. Read the comparator handoff from the current source; do not
  retain a stale line anchor.
- Every shared component as `generalize` or `add`, its rationale, and its existing consumers where
  relevant. Include every required registration surface in each unit ledger so GATE 1 sees the full
  write blast radius.
- The route ladder with evidence for the selected route and every cheaper route it supersedes:

| Class | Allowed change | Must remain fixed |
|---|---|---|
| `retarget` | Architecture tag and target-required dispatch, copy, synchronization, or launch plumbing | Kernel, Block, Scheduler, tile values, verifier, and external contract |
| `unblock` | One A2-bound policy, checker, guard, or copy wrapper becomes architecture-generic | Data flow and external contract |
| `reimplement` | A target-native architecture-facing layer replaces the source layer | External contract |
| `redesign` | External contract changes | Nothing contractual; this is not a migration |

The default route preserves the source form: non-TLA stays non-TLA and TLA stays TLA. A non-TLA to TLA
switch requires a human GATE 1 decision. A route depending on an earlier shared component must name
that dependency. `redesign` is parked as new-contract work and cannot be represented as a migration.

Run `check --phase analyzed` only after every route, reading, rationale, and writable path has been
reinvestigated. See `references/03-analyze.md` and `references/09-hazards.md`.

## Phase 4 — CONFIRM / GATE 1

Before rendering, report the resolved unit list and one doability verdict per unit: route,
counterpart verdict, diagnostic need, and blocker. Then render the migration packet with `mig.py gate`.
It must show the frozen contract, route ladder and evidence, type stack, build and run argv, shared
ledger and consumers, write grant, and registration surfaces.

The packet's execution order is binding:

1. land shared declarations and preserve their existing consumers;
2. implement units one at a time; and
3. land each registration surface after its target directory exists.

`gate` binds the presented set and the artifacts it renders. If analysis, scope, or a structural repair
changes a presented artifact, re-render before confirmation. A confirmation authorizes only the packet
that was shown.

At GATE 1 the human decides whether to migrate, exclude units, and—when global optimization is
enabled—whether any unit should use `--skip-optimize`. Record that final decision verbatim in `--intent`.
The authorization snapshots the presented `plan.json`, `profile.json`, and that unit's
`findings.json`; IMPLEMENT and PROVE refuse any later mismatch. If a contradiction is discovered
while still AUTHORIZED, park it, then deliberately unpark, re-analyze, re-render GATE 1, and obtain a
replacement confirmation. Once IMPLEMENTED is recorded, a contradiction requires a fresh campaign.
No migration write may precede this confirmation. See `references/04-confirm.md`.

## Phase 5 — IMPLEMENT

Land shared declarations serially, preserve existing consumers, and run the discovered source-architecture
regression where the ledger requires it. Then dispatch one implementer for one authorized unit. After it
returns, land that unit's registration text and run `check --phase implemented --unit <id>` before
starting the next unit.

Preserve the frozen Kernel, Block, Epilogue, Scheduler, tile values, CLI, data generation, CPU golden,
tolerances, comparator entry point, and tolerance-selection input. The permitted additive output is one
`CATLASS_EVIDENCE` line; it must not alter existing success or failure tokens. Do not combine a
migration with retiling, rescheduling, fusion, workspace removal, or another behavior change.

If implementation disproves the confirmed contract while the unit is still AUTHORIZED, park the unit,
declare the contradiction, surface it to the human, and follow the deliberate unpark, re-analysis,
GATE-1 re-render, and replacement-confirmation path before further migration work. Once IMPLEMENTED is
recorded, start a fresh campaign instead. Do not repair the route or rationale by writing new evidence.
See `references/05-implement.md`.

## Phases 6 and 7 — COMPILE, then PROVE

`mig.py prove --run-dir <run-dir> --unit <id>` first executes the discovered build and records the
compile outcome, then runs the discovered device command when matching declared A5 access is available.
`--compile-only` records only Phase 6. Compilation is still required when the device is unavailable.

Before building, inspect the declared build prerequisites and record their readings in
`proof.json.environment`. A missing tool, script, environment setup, or other environment mismatch is
**nonblocking**: attempt the build, record the mismatch, and attribute a resulting failure to the
environment rather than the migrated code. Never turn environment preflight into a gate or suppress the
build because the environment is incomplete.

A device run is proof only when it occurs on declared, matching A5 hardware. A remote side requires a
declared transport; use the skill's remote mechanism rather than an ad-hoc wrapper. If matching access
is absent, record the unit as `COMPILED` with device proof not established and retain the pending proof
command. Do not call it migration complete.

The run must emit exactly one marked `CATLASS_EVIDENCE` record. It must match the frozen shape,
`golden.compared_dtype` exactly for the named `golden.compared_tensor`, and tolerance-selection
value, and report `errors: 0`. A success string, exit code, compilation, or a previous proof is not
accuracy evidence. Before a local run, its argv must resolve at least one in-tree executable or input
artifact that exists and is campaign-fresh; a stale, missing, or undatable artifact is a proof failure.
A later failed build, run, or evidence attempt revokes an earlier pass until a later passing proof is
recorded.

See `references/06-prove.md`.

## Phase 8 — SCREEN

SCREEN is read-only and runs only for optimization-enabled, non-TLA units with a **current** proof that
were not set aside with GATE 1 `--skip-optimize`. Current means the latest proof attempt did not fail and
the latest `unit.proven` binds `errors: 0` to the declared matching Ascend 950 identity (`arch: "3510"`,
an A5-family SoC, and nonempty host/device). A historical `PROVEN` rank without that identity does not
authorize SCREEN: re-run `prove` first. The sole candidate is the non-TLA CV-fused L0C→GM→UB path whose
target can be rewritten to use L0C→UB. Establish all applicability rows from the migrated unit, record
their current source evidence, construct the full file manifest from the target example's actual include
and use graph, and reinvestigate the proposed `coexist` or `replace` strategy.

Take and record the declared profiler baseline and its output directory inside the run directory. Every
reported measurement must be backed by the profiler output that produced it; never estimate, reuse, or
invent a value. Run `check --phase screened --unit <id>` after the screen artifact and baseline are
complete. A unit that is not applicable is finished at `OPT_SCREENED` without a rewrite.

This is not generic performance tuning. Do not change tile shape, scheduling, swizzle, fusion,
block count, core count, or any other generic tuning parameter. Do not introduce or tune `SPLIT_N`.
Only the rewrite-guide-supported L0C→UB data-path modes may be proposed.

`NO_SPLIT` covers the actual runtime M interval, including its tail; it does not make the compile-time
split tile odd. Keep that compile-time tile even.

See `references/07-optimize.md`.

## Phase 9 — CONFIRM-OPT / GATE 2

Render one optimization packet with `mig.py gate --phase optimize`. It must show each screened unit's
applicability readings, baseline evidence, proposed strategy, exact manifest, and what that strategy
changes:

| Strategy | Effect |
|---|---|
| `coexist` | Both paths land; the L0C→UB rewrite is the default path, so unchanged post-APPLY `prove` exercises it. The baseline remains buildable only as the alternate variant. |
| `replace` | The old path is removed; this requires an explicit human yes. |

The human decides apply or skip for each eligible unit and chooses `coexist` or `replace` where
applying. Record the final decision verbatim in the phase-optimization `--intent`. The confirmation
snapshots both strategy and manifest. If either needs correction, declare the structural repair,
reinvestigate the affected readings or strategy, re-screen, re-render, and obtain a new GATE 2 decision.
No rewrite write may precede that confirmation. See `references/04-confirm.md`.
GATE 2 may render or confirm a unit only while that same current hardware-bound proof remains current.
If a later prove failure or an unbound/mismatched `unit.proven` appears, the rank does not regress, but
the unit must be re-PROVEN before a new packet can authorize it.

## Phase 10 — APPLY

First land any shared, authorized tier-1 declarations. Dispatch one optimizer per authorized unit to
write only its authorized tier-2 paths; then land authorized tier-1 registration changes after the
target files exist. The implementation must be the approved L0C→UB rewrite and strategy, not generic
tuning.

The phase order is mandatory:

1. apply the authorized rewrite and take the required post-rewrite measurement; under `coexist`, also
   take the baseline-side measurement required for comparison;
2. run `mig.py check --run-dir <run-dir> --phase applied --unit <id>`;
3. rerun the **unchanged** Phase 7 `mig.py prove --run-dir <run-dir> --unit <id>` command against the
   same frozen contract and golden; then
4. run `mig.py check --run-dir <run-dir> --phase optimized --unit <id>`.

`check --phase optimized` requires a proof recorded after APPLY. Before APPLY and again before
`OPTIMIZED`, require the current proof; APPLY must not proceed from a historical `PROVEN` rank. The
unchanged post-APPLY proof restores eligibility only when it carries the declared matching Ascend 950
hardware identity and no later proof failure overlays it. The evidence boundary still applies:
declaring files or a measurement cannot substitute for the actual proof or profiler output. Read the
rewrite guide only while writing this rewrite: `references/08-l0c-to-ub-rewrite.md`.

## Phase 11 — REPORT

Run `mig.py report --run-dir <run-dir>`. The report must distinguish authorized work, compiled work,
proof on matching hardware, screened units, optimization decisions, measured rewrites, environment
mismatches, and all **Not established** results. It must not imply that unproven or skipped work is
complete.

## Non-negotiable rules

- **NEVER** migrate a unit whose direct Cube/Mmad operands are `AscendC::int4b_t`. Refuse it without an
override, identify it to the user, and treat an A5 int4 Cube backend as separate new-backend work.
- The sole stored-int4 exception is a path that stores `AscendC::int4b_t` but performs the actual
  Vector-side `Cast` to `int8_t` before Cube/Mmad, so the Cube operands are `int8_t`. Investigate the
  operand declarations and Cast data flow; a token search for `int4` is not sufficient. Do not confuse
  this exception with a direct int4 Cube operand.
- **NEVER** call a dtype, layout, scale, mask, aliasing, or supported-domain change a migration; edit a
golden or relax a tolerance to pass; or let a redesign pass through the migration gates.
- **NEVER** report accuracy from a success string, exit code, compile, registration row, or previous
  run. Only the current parsed evidence record on matching declared hardware establishes accuracy.
- **NEVER** run `msprof op` without `--output=` inside the run directory, or use an ad-hoc remote shell
  wrapper in a run command. Declare remote access and transport instead.
- **NEVER** use generic tuning—including block count, core count, or `SPLIT_N`—as part of this rewrite.
- **ALWAYS** distinguish a missing specialization selected by a type stack from a hardware conclusion;
  investigate the target tree and the pinned sources before recording either.

## references/ and scripts/

| File | When to read |
|---|---|
| `references/01-frame.md` | Phase 1 — intake, `plan.json`, `access.json`, candidate resolution, target naming, and performance input |
| `references/02-probe.md` | Phase 2 — the four `profile.json` sections |
| `references/03-analyze.md` | Phase 3 — route ladder, contract freeze, and `findings.json` |
| `references/04-confirm.md` | Phases 4 and 9 — both gates, packet contents, decision recording, and binding |
| `references/05-implement.md` | Phase 5 — shared edits, registration surfaces, and writable scope |
| `references/06-prove.md` | Phases 6–7 — compile, proof, environment attribution, evidence, and triage |
| `references/07-optimize.md` | Phases 8–10 — applicability, measurements, strategies, manifests, and optimization artifacts |
| `references/08-l0c-to-ub-rewrite.md` | Phase 10 — the L0C→UB rewrite guide; read only while writing the rewrite |
| `references/09-hazards.md` | Phases 3, 5, and 7 — hardware limits, repository gaps, int4, and migration pitfalls |
| `assets/perf_case_template.md` | Phase 1 fallback performance table staged when none is supplied |
| `scripts/mig.py` | Engine commands: `init refs profile check gate confirm park prove remote status report` |
| `README.md` | User-facing workflow and scope |
