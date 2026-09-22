# Phases 6 and 7 — COMPILE, then PROVE

COMPILE and PROVE are separate outcomes of one command. Phase 6 establishes that the
discovered target-architecture build completed. Phase 7 establishes accuracy against
the frozen CPU golden on declared matching hardware. A build, an exit status, a
registry entry, or an old proof is never accuracy evidence.

```bash
# Phase 6: compile outcome only
mig.py prove --run-dir <run-dir> --unit <id> --compile-only

# Phase 6 followed by Phase 7: compile, then execute and verify
mig.py prove --run-dir <run-dir> --unit <id>
```

Run units serially. The build configuration and target device are shared campaign
resources, and serial execution keeps each proof attributable to one unit.

## Access, environment, and outcomes

Access is declared at FRAME with `init --access`, and `prove` uses that declaration.
The build always runs before a device stage is considered. Environment preflight
records whether the declared build command and sourced scripts are available, but it
is non-blocking: `prove` attempts the build even when a prerequisite is absent. A
subsequent build failure may be recorded as an environment cause; it is not evidence
that the migration code failed to compile.

A device run is valid only on declared, reachable target-architecture hardware. A
transport declaration causes the run stage to execute through that transport using
the staged paths recorded in `findings.prove`; the proof evidence is parsed exactly
as it is for a directly reachable device. Without reachable matching hardware, the
campaign may reach `COMPILED` but not `PROVEN`. Report the pending `prove` command
and that device proof is not established. Never substitute a different architecture
for a target-architecture run.

Before any device or staging command, a reachable `a5` declaration must bind that run to
`arch: "3510"`, a non-empty host and device selector, and an SoC that identifies the A5 family
(`950`, `Ascend950`, `A5`, or `3510`). A 910-family declaration, including `910B`, is A2-family
and is rejected rather than relabeled as an A5 run. The matching `a2` binding is `arch: "2201"` and
an SoC identifying `910`, `Atlas A2`, `Atlas A3`, or `2201`.

`prove`, both gates, and optimization checks load that declaration strictly. Only `status` and `report`
may read a legacy declaration leniently, and they must mark its old, missing, or mismatched identity
**NOT ESTABLISHED** rather than treating it as current accuracy. Record a structural repair only from
measured hardware, then re-run `prove` before SCREEN, GATE 2, APPLY, or `OPTIMIZED`; never infer,
backfill, or invent the missing identity.

On a target-device proof, `proof.json.hardware` and `unit.proven.hardware` retain the measured FRAME
declaration verbatim:

```json
{
  "arch": "3510",
  "soc": "Ascend950",
  "host": "<declared device location>",
  "device": "<declared device selector>"
}
```

A current proof is a `unit.proven` event with `errors: 0` and that exact declared hardware object,
with no later `unit.prove_failed`. It is the only accuracy authority for Phases 8–10. An old `PROVEN`
event without this identity is historical, not accuracy evidence; a run never discovers, infers, or
substitutes hardware identity.

Every attempt writes its proof record. A build failure ends the attempt before a run.
A run failure, timeout, or evidence failure is a reported finding, never a skipped
stage. Before a local run, `findings.prove.run` must resolve at least one in-tree executable
or input artifact that exists and is newer than the campaign's frozen plan. An empty resolved
set, a stale artifact, or a missing artifact records `unit.prove_failed` before execution. Remote
runs retain their separately required non-empty `findings.prove.stage` list.

A later build, run, or evidence failure revokes an earlier pass without changing its historical rank.
Only a later passing proof bound to the declared matching Ascend950 identity clears that revocation;
re-run `prove` before SCREEN, GATE 2, APPLY, or `OPTIMIZED`.

## The frozen accuracy contract

PROVE reuses the contract frozen at ANALYZE:

- the source example's input generation, CPU golden function, comparator entry
  point, `computeNum`, tolerance behavior, and one `compared_tensor` that exactly
  names a frozen contract tensor;
- `golden.compared_dtype`, the exact canonical string the evidence line must emit
  for that tensor, rather than an inferred storage or Cube-operand spelling;
- the frozen proof shape and discovered run arguments;
- upgraded-precision golden computation, including `ElementGolden = float` for a
  half-precision operator; that computation remains frozen independently of the
  evidence-label spelling;
- the target tree's actual comparison result, not a newly authored comparator or
  tolerance.

The comparator and the `computeNum` passed to it are contract facts. `computeNum`
selects the comparator's tolerance band, so the evidence-reported value must equal
the frozen value. Integer comparisons remain exact when the discovered comparator
uses exact integer comparison. Never infer a comparator from dtype alone; reuse the
source call and its current target-tree implementation.

When numerical proof fails, verify the golden first: benchmark choice, upgraded
precision, layouts, source-equivalent input buffers, and a small manually understood
case. Then investigate the architecture-facing migration delta. Do not edit the
golden, relax a tolerance, change the comparator, or narrow accepted inputs to turn
a failure into a pass. If the evidence establishes that the frozen contract is false,
park the unit and return it through analysis and GATE 1.

## `CATLASS_EVIDENCE` is the proof witness

The migrated example emits exactly one JSON line prefixed `CATLASS_EVIDENCE` after
its existing comparison. The line is additive: existing success and failure tokens
remain unchanged, and the evidence must reuse the comparison result rather than
performing another comparison.

| Field | Requirement | Proof check |
|---|---|---|
| `shape` | Required integer list | Equals the frozen `findings.prove.shape`. |
| `argv` | Optional string list | When present, equals the arguments delivered to the example. |
| `dtype` | Required non-empty canonical string | Equals `contract.golden.compared_dtype` exactly; that frozen field names the sole `contract.golden.compared_tensor` this label describes. |
| `computeNum` | Required when the golden contract has one | Equals `contract.golden.compute_num`. |
| `errors` | Required JSON integer | `0` is the numerical-pass condition. |
| `golden_sumsq` | Optional finite number | A diagnostic golden summary; inspect it when reviewing proof evidence. |

For example, a frozen golden `{ "compared_tensor": "D", "compared_dtype": "fp16" }`
requires an evidence field `"dtype":"fp16"` exactly. The evidence does not repeat the tensor name:
the Gate-1 contract already fixes that single subject. If the source golden computes `D` in upgraded
precision, preserve that precision in the frozen golden function; do not change this canonical
evidence label to make it resemble the computation type.

A valid run contains **exactly one** such line. No line, more than one line, invalid
JSON, a field mismatch, or nonzero `errors` fails the proof. `Compare success.` and
process exit status are not evidence because neither identifies what was compared.

The evidence line is excluded from the frozen external contract only as an additive
diagnostic. Its absence is an implementation defect. It does not justify parking the
unit, changing the contract, or weakening the proof requirement.

## Evidence and repair boundary

`proof.json`, captured stage logs, artifact readings, and the parsed evidence line
are evidence records. Do not edit them to repair an outcome. Re-run `prove` after
repairing the implementation, command discovery, access declaration, or factual
analysis that caused the failure.

A structural repair may declare a malformed artifact, missing required key, invalid
path, or stale manifest. It may not author any factual value. Comparator choice,
`computeNum`, source reading, rationale, device reading, performance measurement,
and optimization strategy must be reinvestigated from the target tree or fresh
command output. If that investigation changes a gate-bound input, re-check it and
obtain the required new confirmation.

## Revocation and retained records

A passing evidence line records `unit.proven`. Every later build, run, or evidence
failure records `unit.prove_failed` and revokes the previous accuracy claim until a
later passing proof clears the revocation. Keep the failing attempt's log distinct
from logs a retry may replace. The report must show the latest failure as revoked,
not continue to present an earlier `errors: 0` result.

`PROVEN` covers only the frozen contract shape and the evidence from that execution.
It does not establish behavior for other shapes, performance, or a later rewrite.

## Phase 10 uses the same proof

After GATE 2 authorizes a rewrite, Phase 10 uses this order without changing the
proof contract:

```text
APPLY and measure → check --phase applied → unchanged prove → check --phase optimized
```

The unchanged `prove` reuses the same golden, comparator, shape, and evidence rules.
It is the accuracy record for the path made active by APPLY. `check --phase optimized`
requires a proof recorded after `check --phase applied`; a prior Phase-7 proof cannot
establish the rewritten path.

## Pinned technical references

- CATLASS golden, comparator, and example interfaces: [CATLASS @ `89e1fc39881a715882b9b47459add06ba270105c`](https://gitcode.com/cann/catlass/tree/89e1fc39881a715882b9b47459add06ba270105c).
- Ascend precision-analysis guidance: [asc-devkit @ `512674e996da0feaee6e7f435e4efc1cad1d74fb`](https://gitcode.com/cann/asc-devkit/tree/512674e996da0feaee6e7f435e4efc1cad1d74fb).
