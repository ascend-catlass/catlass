# Phase 3 — ANALYZE: freeze the source and adjudicate the route

ANALYZE is read-only. The lead reads every source unit in the FRAME cluster and writes one
`<run-dir>/units/<id>/findings.json` per unit. Read each shared dispatch-policy stack once, then apply
that reading to every unit that selects it. Do not turn a repeated policy family into independently
invented analyses.

```sh
mig.py check --run-dir <run-dir> --phase analyzed [--unit <id>]
```

The command validates artifact structure and records the analyzed state. It does **not** establish that
a cited symbol is correct, a path supports the stated conclusion, the route is feasible, or the proof
commands will run. Those are mandatory human investigations in this phase and are rendered for GATE 1.

## Evidence boundary

Target-tree observations are read from the target tree. Fixed upstream claims may use only:

- CATLASS: `https://gitcode.com/cann/catlass` at
  `89e1fc39881a715882b9b47459add06ba270105c`
- asc-devkit: `https://gitcode.com/cann/asc-devkit` at
  `512674e996da0feaee6e7f435e4efc1cad1d74fb`

Every fixed fact cites the immutable revision plus the file or symbol read. A pinned example may
illustrate a pattern but is never a default for the target tree. Do not use a branch, tag, short hash,
third-party source, or remembered behavior as evidence.

## Analysis order and evidence rule

For each unit, in this order:

1. Freeze the source external contract.
2. Walk each selected source type stack and its actual control flow.
3. Investigate the route ladder from cheapest to the selected source-form-preserving route.
4. Investigate counterpart candidates from implementation and contract evidence, not names.
5. Declare shared components and discovered registration responsibilities.
6. Discover the build and run commands that Phase 7 will execute.
7. Run `check --phase analyzed` after every material correction.

Each factual field must state what was read, where it was read, and a command that can re-derive the
reading. Source paths and symbols are stable evidence; line numbers are not required and must not stand
in for the reading.

**Repairs and evidence are different.** A structural repair may be declared: for example, a missing
required field, a stale path form, or an omitted required registration surface may be corrected and
identified as a repair. A declaration is not evidence. Routes, source readings, rationales, measurements,
and optimization strategies must be **reinvestigated from their sources** after any repair; never author,
copy forward, estimate, or infer them to complete a schema. Re-render GATE 1 after a repaired finding is
checked.

Never silently remove a named FRAME candidate. An apparent existing target implementation is recorded in
`counterpart` for the human at GATE 1. The sole hard refusal is a unit whose direct Cube operands are
`AscendC::int4b_t`; report that refusal to the human as specified in FRAME. Storage int4 converted to
int8 before Cube is not that refusal and requires normal analysis.

## `findings.json` shape

Keys are closed. This template gives the complete artifact shape; every placeholder is a reading from the
current source tree, not a value to reuse.

```json
{
  "unit": "<plan unit id>",
  "route": "<retarget | unblock | reimplement | redesign>",
  "counterpart": null,
  "contract": {
    "tensors": [
      {
        "name": "<tensor>",
        "role": "<semantic role>",
        "dtype": "<stored dtype and Cube-operand dtype where different>",
        "layout": "<layout and stride semantics>",
        "storage": "<owner and memory layer>",
        "alias_of": null
      }
    ],
    "output_region": "<complete write and reduction ownership>",
    "supported_domain": "<accepted shapes, tails, and limits>",
    "zero_work": "<source behavior for zero-work inputs>",
    "golden": {
      "function": "<reference-compute symbol and precision>",
      "comparator": "<exact called comparator entry point>",
      "compared_tensor": "<exact contract.tensors[].name passed as the comparison result>",
      "compared_dtype": "<exact canonical CATLASS_EVIDENCE.dtype string>",
      "compute_num": 0,
      "compute_num_read_from": "<path>:<line of the comparator call>"
    },
    "cli": "<argument order, defaults, device selection, and API/Arguments order>",
    "frozen_from": "<source paths and immutable revision if external>"
  },
  "type_stack": {
    "host": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "dispatch": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "block": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "tilecopy_ab": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "tilecopy_cd": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "epilogue": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "scheduler_tiles": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "kernel": { "type": "<expanded selected type and semantics>", "read_from": "<path>" },
    "adapter": { "type": "<expanded selected type and semantics>", "read_from": "<path>" }
  },
  "routes": [
    {
      "route": "<class>",
      "verdict": "<eligible | excluded | needs-diagnostic>",
      "evidence": "<file, symbol, observed dataflow, and re-derivation command>",
      "diagnostic": "<required only for needs-diagnostic>",
      "conditional_on": "<optional component established by an earlier unit>"
    }
  ],
  "shared_components": [
    {
      "path": "<path inside target_root>",
      "symbol": "<exact declared symbol>",
      "kind": "<generalize | add>",
      "why": "<one reviewable mechanism and reason>",
      "consumers_of": "<required existing symbol for generalize only>"
    }
  ],
  "prove": {
    "build": ["<target-architecture build argv>"],
    "run": ["<installed executable and frozen CLI argv>"],
    "cwd": "<relative working directory inside target_root>",
    "shape": [0],
    "device": "<target-device selection and observed capability>"
  },
  "writable_paths": ["<unit target directory only>"],
  "notes": "<non-schema context>"
}
```
The complete template shows local target-device access and therefore deliberately omits `prove.stage`.
When FRAME declares A5 access with SSH transport, add `stage` as a non-empty list of the
tree-relative artifact and run-input paths that must be staged; omit the key for every other access mode.

A layer or contract field that genuinely does not apply is exactly
`{"not_applicable":"<reason>"}`; it is never omitted. `compute_num_read_from` remains a source call
citation because the engine validates that the frozen comparator name occurs at that location. Re-read
the call when its line moves; do not preserve an old anchor. `golden.compared_tensor` must exactly name
one `contract.tensors[].name`. `golden.compared_dtype` is the exact, whitespace-free canonical string
the example emits as `CATLASS_EVIDENCE.dtype`; it is not inferred from storage, Cube-operand, or
upgraded-golden precision.

## Freeze the external contract first

The contract records the source behavior that migration must preserve:

| Field | Required reading |
|---|---|
| `tensors` | Every input, output, accumulator, workspace, scale, mask, group list, bias, and host scalar. Record semantic role, layout, owner/storage, and alias relation. Record stored dtype and Cube-operand dtype when they differ. |
| `output_region` | The complete region written and the reduction ownership boundary. |
| `supported_domain` | Accepted shapes, tails, and domain limits. |
| `zero_work` | Source behavior for zero dimensions, empty groups, and other no-work cases, including host rejection. |
| `golden` | Exact reference-compute symbol and precision, comparator entry point, the one named `compared_tensor`, its exact canonical evidence `compared_dtype`, frozen `compute_num`, and the source comparator call that provides it. `compared_tensor` gives the one `dtype` witness an unambiguous subject; `compared_dtype` is not a translation of that tensor's storage description. |
| `cli` | Argument order, defaults, device selection, and launch or `Arguments` ABI as used by the source. |

A stored int4 value that is converted to int8 before Cube has two facts: the public storage contract and
the Cube operand. Preserve both in `dtype`; neither may be silently changed. A direct Cube
`AscendC::int4b_t` operand is the FRAME refusal, not a route to bypass.

A later change to dtype, layout, stride, scale or mask representation, tensor set, aliasing, output
region, zero-work behavior, or supported domain is `redesign`, not migration. Do not edit a golden,
comparator, or tolerance to make a proposed implementation fit the freeze.

## Walk every selected type stack

All nine `type_stack` layers are required: `host`, `dispatch`, `block`, `tilecopy_ab`, `tilecopy_cd`,
`epilogue`, `scheduler_tiles`, `kernel`, and `adapter`.

For each layer, expand aliases, default template arguments, selected partial specializations, and real
control-flow branches. Record the selected declaration's path in `read_from`; state additional evidence
inside `type`. When branches instantiate different kernels or policies, walk each active stack. Ignore
comments and inactive preprocessor arms.

The walk must establish:

- the host architecture tag and all branches selecting a stack;
- dispatch policy parameters and semantics, including stages, flags, and resource assumptions;
- the exact block specialization key and all arguments, not only its template name;
- A/B and C/D copy aggregators, their selected arity, operands, and memory paths;
- epilogue policy and block semantics, including `void` when applicable;
- scheduler/swizzle and literal tile values, which a migration preserves;
- kernel reduction, workspace, synchronization, and visible flag protocol; and
- adapter and launch ABI.

A symbol's existence, an apparent target branch, and visibility under the target architecture gate are
separate facts. A type name, directory name, or generic-looking template proves none of them. Diagnose a
primary-template `static_assert` by expanding the selected specialization key; it proves only that no
matching specialization was selected, not why.

## Adjudicate the route ladder

The default preserves the source form: non-TLA sources remain non-TLA; TLA sources remain TLA. Do not
silently switch a non-TLA source to a TLA stack merely because one exists. The human makes that **form
choice at GATE 1** after seeing the contract comparison, existing target occupants, and excluded cheaper
routes.

| Class | Meaning | Eligibility boundary |
|---|---|---|
| `retarget` | Switch architecture selection and add target-required plumbing while preserving source Kernel, Block, Scheduler, tile values, verifier, and dataflow. | Every selected layer already admits the target architecture and every required data path remains legal. |
| `unblock` | Retarget plus generalize one dispatch-policy family and its selected declaration surfaces without changing member semantics or dataflow. | The retarget blocker is limited to a reusable declaration surface; target dataflow and control semantics are otherwise preserved. |
| `reimplement` | Replace an architecture-facing layer with a target-native equivalent while preserving every frozen contract field. | Only after GATE 1 explicitly chooses the form, every cheaper class is excluded by evidence, and a contract-equivalent target consumer is compared field by field. |
| `redesign` | Change the external contract. | A specific frozen field must change. This is new-contract work, not a migration. |

For the recommended route, provide a route row for that class and every cheaper class it supersedes.
The selected route must be `eligible` or `needs-diagnostic`; a `needs-diagnostic` row names the exact
future command that will settle it. A dearer route is invalid while a cheaper route remains `eligible`.

When the source-form-preserving route cannot yet be established, report the uncertainty as
`needs-diagnostic`; do not choose `reimplement` preemptively. At GATE 1 the human may explicitly choose
a form override. Update the findings from fresh evidence, re-run ANALYZE validation, re-render the gate,
and only then confirm that decision.

A missing target consumer or specialization is a repository gap, not a hardware limitation and not a
`redesign` conclusion. A hardware limitation excludes only the implementation using the prohibited
path; continue investigating legal implementations within the same class. Cite the pinned source for
any such fixed hardware statement.

## Counterpart finding

`counterpart` is required and nullable.

```json
null
```

means no plausible target-architecture implementation was found. Otherwise use:

```json
{
  "suspect": "<target directory>",
  "verdict": "<is-counterpart | not-counterpart>",
  "evidence": "<type-stack and frozen-contract comparison, with re-derivation command>"
}
```

Compare the suspect's type stack and every frozen contract field. A matching slug, number, directory,
or TLA label is not counterpart evidence. GATE 1 renders this finding so the human, not FRAME or
ANALYZE, decides whether an already-migrated unit is excluded.

## Shared declarations and post-unit registrations

`shared_components` is a closed list of `{path, symbol, kind, why, consumers_of}`. `kind` is exactly
`generalize` or `add`.

- A `generalize` row names an existing reusable component. It **requires** `consumers_of`, the exact
  existing symbol whose consumers must keep building.
- An `add` row names a new component. It **forbids** `consumers_of`.
- A unit that has no pre-unit shared declaration still supplies `shared_components: []` only if the
  profile has no required registration surface. Every non-redesign unit must declare every required
  `profile.registration.surfaces` path; the engine checks this by path.

The artifact keeps one list so the gate can compute a shared ledger, but execution has two explicitly
separate partitions:

1. **Pre-unit declarations:** rows whose path is not a required registration surface. The lead lands
   these shared headers, policies, checkers, or wrappers once before implementing dependent units. For a
   `generalize`, investigate every consumer before declaring the row.
2. **Post-unit registrations:** rows whose path is a required registration surface from
   `profile.registration.surfaces`. The lead lands these only **after that unit's target directory
   exists** and the unit implementer returns its registration text. They are not pre-unit declarations
   and must never be registered before the directory they name exists.

The route, source reading, `why`, consumer set, or registration rationale is not copied from another
unit merely because its shape matches. Reinvestigate it for each unit; the ledger then merges identical
`(path, symbol)` rows and measures existing `consumers_of` use.

## Discover the proof block

`prove` is discovered now and executed unchanged in Phases 6 and 7. Commands are argv lists used without
an implicit shell, glob expansion, or environment injection.

| Key | Required source |
|---|---|
| `build` | `profile.build.target_build`, `profile.arch-gating`, and the target directory's expanded runnable leaf target. |
| `run` | The discovered artifact path and installed binary name plus the frozen source CLI at the frozen shape. |
| `cwd` | A non-empty target-root-relative working directory derived from input and executable requirements. |
| `shape` | The frozen contract's own shape. |
| `device` | Target-device selection and the capability observation declared at FRAME. |
| `stage` | Required only when FRAME declares A5 access with SSH transport: a non-empty list of tree-relative artifact and run-input paths to stage. It must be absent for local target-device access and every other access mode. |

A build target, directory, and installed binary may have different names. Expand the leaf declaration and
trace artifact naming before recording `run[0]`. Environment availability remains nonblocking: record an
unavailable prerequisite as evidence for later attribution, not as a reason to invent a command or alter
the contract.

## GATE 1 decisions

After all unit findings are validated, render the migration gate:

```sh
mig.py gate --run-dir <run-dir>
```

The human decides, per unit, whether to authorize migration, exclude it, choose an explicitly presented
form override, or skip optimization early while retaining migration. Interpret questions, conditional
replies, and ambiguity as questions rather than approval. Re-render any interpretation or repaired
artifact before confirmation.

```sh
mig.py confirm --run-dir <run-dir> --intent '<final human decision, verbatim>' \
  [--exclude <comma-separated-unit-ids>] \
  [--skip-optimize <comma-separated-unit-ids>]
```

`--exclude` keeps a unit at `ANALYZED`. `--skip-optimize` authorizes its migration path but removes it
from later optimization phases when campaign optimization is enabled. `--intent` is the final human
decision **verbatim**, not an agent-written summary. Neither route selection nor an early optimization
skip may be silently inferred from the artifact or from an omitted reply.
