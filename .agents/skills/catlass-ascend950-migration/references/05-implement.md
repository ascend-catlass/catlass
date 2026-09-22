# Phase 5 — IMPLEMENT: ownership, preservation, and scope

IMPLEMENT begins only after Phase 4 CONFIRM / GATE 1 has authorized the unit. It is
not a design phase: the approved route, frozen external contract, writable paths,
shared-component ledger, registration surfaces, and proof plan are its inputs.

## Ownership and order

The lead owns every shared change and all execution. An Implementer owns code only
inside one authorized unit directory. Work one unit at a time in the ledger order:

| Order | Owner | Required action |
|---:|---|---|
| 1 | Lead | Land each approved shared declaration once, in ledger order. |
| 2 | Lead | Run the discovered source-architecture regression for every affected existing consumer before unit work proceeds. |
| 3 | Implementer | Write only the current unit's `writable_paths`; do not build or run. |
| 4 | Implementer | Return the exact registration text, grouped by declared registration surface. |
| 5 | Lead | Land that text only after the unit directory exists. |
| 6 | Lead | Converge the unit with the implementation scope check. |

```bash
mig.py check --run-dir <run-dir> --phase implemented --unit <id>
```

Registration stays after the unit directory exists because registration is a shared
surface and can make configuration observe a directory before it has been created.
An Implementer never writes a registry, shared header, another unit, or the A2
source. It never performs a build, proof, profile, or device run.

## Preserve the migration contract

The target implementation preserves the source's:

- CLI grammar, arguments, defaults, and accepted domain;
- input generation, output region, dtype/layout/scale/mask/aliasing rules, and
  zero-work behavior;
- CPU golden, comparator entry point, `computeNum`, and tolerance behavior;
- Kernel, Block, Epilogue, Scheduler, data flow, and tile values unless the
  approved route explicitly names an architecture-facing adaptation.

Do not call a changed dtype, layout, scale, mask, aliasing rule, supported domain,
or output region a migration. Do not change a golden, comparator, or tolerance to
make a proof pass.

Generic tuning is outside this phase, including block count, core count, `SPLIT_N`,
retiling, rescheduling, fusion, and workspace removal. A migration may repair the
approved architecture boundary; it must not turn that repair into a performance
experiment.

### The permitted diagnostic addition

Every migrated unit must emit exactly one `CATLASS_EVIDENCE` line during its normal
comparison path. It is an additive diagnostic channel, not part of the frozen
external contract:

- preserve existing success and failure tokens byte-for-byte and in their existing
  streams;
- emit the evidence line after the existing comparison and reuse that comparison's
  result; do not run a second comparison;
- treat a missing line as an implementation defect, not as a contract change.

Phase 7 defines the evidence schema and verdict.

## Shared changes and regression ownership

A shared-component ledger row is either `generalize` or `add`. Before changing its
file, inspect the current target tree's complete declaration and specialization set,
all exact-symbol consumers, default template arguments, includes, guards, and
caller-side contract. Prefer an evidenced, isomorphic `ArchTag` generalization over
a Kernel or Block rewrite. Preserve every source-architecture specialization that
remains consumed.

When a shared declaration modifies an existing file outside unit write grants, the
lead must run the source-architecture regression command discovered by PROBE before
dispatching unit work. A target-architecture build does not replace this regression.
A new file is exempt only when the discovered architecture gate proves it is
unreachable from the source architecture.

Use the target tree's existing guard form, primary-template extension point, trait
member spelling, and registration form. The pinned CATLASS source is a technical
reference for patterns; the target tree and `profile.json` are the source of truth
for names, macros, signatures, build targets, and registration locations.

## Architecture-bound details to investigate

The following are investigation prompts, not templates to paste:

- An architecture-generic policy must preserve the source policy's stages,
  synchronization, and resource semantics. Its admissible specializations must use
  the primary template's existing extension point; do not create a parallel trait or
  duplicate specialization.
- `Arch::Resource<ArchTag>` belongs to the generation selected by the consuming
  kernel. Resource constants are compile-time budgets, not permission to enlarge a
  data path; the consumer's constraints remain authoritative.
- Remove source-architecture synchronization plumbing only from the target
  caller when the complete caller chain shows the value is unused there. Do not
  delete shared APIs, a source-architecture branch, or repository-wide callers by
  name.

These rules describe structural boundaries. Their applicability, symbols, and
rationale must be read again from the target tree and recorded in the unit evidence.

### Direct Cube `int4` refusal

Never migrate a unit whose Cube operands are `AscendC::int4b_t`; this is direct
Cube-int4 work and is refused. Do not emulate or conceal it with a new backend.

The distinct case `stored int4 → Vector Cast → int8 → Cube` is allowed only when the
whole path is evidenced from the unit: storage is int4, the Vector stage performs the
cast, and Cube receives int8. Do not infer that exception from a type name or a
single declaration.

## Registration and executable identity

`profile.json` owns registration facts. The lead uses its declared surfaces and the
unit's discovered leaf build declaration to determine:

- the target directory registration entry;
- the exact leaf target to build;
- the runnable artifact named by the proof command; and
- any target-architecture test registration required by the tree.

A directory, aggregate target, or test row is not automatically a runnable leaf.
Do not derive an executable name from a directory name. The Implementer returns text;
the lead reviews and lands it on each declared surface.

## Scope check and repair boundary

`check --phase implemented` checks the target-tree diff against the baseline
recorded at `init`. It requires a change within the current unit's write grant and
rejects changes outside the approved unit paths, another planned target, a ledger
path, a declared registration surface, or campaign artifacts.

If the check identifies a mismatch, first classify it:

| Finding | Required response |
|---|---|
| Another unit owns the change | Remove it from the current unit and perform it only in the owning unit's turn. |
| The change is a missing shared surface | Stop the unit. Reinvestigate the source and target evidence, update the affected findings and shared-component ledger, run `mig.py check --run-dir <run-dir> --phase analyzed`, render a fresh GATE-1 packet, and obtain and record the final human confirmation verbatim. Only then may the lead land the surface and run the source-architecture regression before unit work resumes. |
| The granted path or factual analysis is incomplete | Reinvestigate the source and target evidence, update the analysis, re-check it, and return to GATE 1 before writing under the broader scope. |
| The frozen contract or approved route is false | If the unit is still AUTHORIZED, park it; do not improvise a new migration. If it is already IMPLEMENTED, begin a fresh campaign rather than reuse that rank. |

A structural repair may declare a malformed field, missing path, or incomplete
manifest. It must not author a route, source reading, rationale, measurement, or
strategy. Those are evidence claims and must be reinvestigated from the target tree
or a fresh command result. Any material repair to GATE-1 inputs requires a new
packet and confirmation.

Park a unit when implementation discovers a contract change or an unapproved route **before**
`IMPLEMENTED` is recorded:

```bash
mig.py park --run-dir <run-dir> --unit <id> --reason '<evidenced mismatch>'
```

This invalidates an AUTHORIZED migration grant. Preserve the reason and evidence; deliberately
unpark, re-analyze with `check --phase analyzed`, render GATE 1 again, and record a replacement
confirmation. The invalidation persists through unpark and re-analysis, and only that replacement
authorization clears it. If implementation is already recorded, a later contradiction requires a
fresh campaign, not a parked re-authorization.

## Pinned technical references

- CATLASS source, including architecture, dispatch, block, epilogue, and device
  interfaces: [CATLASS @ `89e1fc39881a715882b9b47459add06ba270105c`](https://gitcode.com/cann/catlass/tree/89e1fc39881a715882b9b47459add06ba270105c).
- Ascend platform development guidance: [asc-devkit @ `512674e996da0feaee6e7f435e4efc1cad1d74fb`](https://gitcode.com/cann/asc-devkit/tree/512674e996da0feaee6e7f435e4efc1cad1d74fb).
