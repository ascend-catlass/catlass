# Phases 4 and 9 — CONFIRM: decisions and write gates

A campaign has **three human decision points** and exactly **two write gates**.

| Decision | Phase | What the human decides | Gate? | What follows |
|---|---:|---|---|---|
| FRAME | 1 | The frozen unit scope, declared access, optional performance cases, and whether optimization is enabled for the campaign | No | Read-only PROBE and ANALYZE (Phases 2–3) |
| CONFIRM | 4 | Which analyzed migrations may proceed, their evidenced route, exclusions, and any per-unit optimization skip | **GATE 1** | IMPLEMENT (Phase 5) |
| CONFIRM-OPT | 9 | For each screened applicable unit, whether to apply the rewrite and, if applied, `coexist` or `replace` | **GATE 2** | APPLY (Phase 10) |

FRAME settles scope but authorizes no write. The two gates are the only authorizations to
write the target tree: GATE 1 authorizes migration work only; GATE 2 authorizes the
non-TLA `L0C→UB` rewrite only. Neither implies the other.

The full phase order is: 1 FRAME, 2 PROBE, 3 ANALYZE, 4 **CONFIRM**, 5
**IMPLEMENT**, 6 COMPILE, 7 **PROVE**, 8 SCREEN, 9 **CONFIRM-OPT**, 10 APPLY,
11 REPORT.

## FRAME decisions that bound both gates

`init` freezes the FRAME decision. Access is declared there, not inferred later:

```bash
mig.py init --plan plan.json --access access.json [--perf-cases <path>]
```

`plan.optimize.enabled` defaults to `true`. Set it to `false` only at FRAME to opt
out the entire campaign; an opt-out campaign has no SCREEN, CONFIRM-OPT, or APPLY
work. When no performance-case table is supplied, `init` stages the bundled
`assets/perf_case_template.md` as the campaign's performance-case table.

A campaign with optimization enabled can still set aside a specific unit at GATE 1
with `--skip-optimize`. That decision preserves the migration work but prevents that
unit from entering the optimization path after it reaches `PROVEN`.

## Phase 4 — CONFIRM / GATE 1

ANALYZE is read-only. After every unparked unit has passed its analyzed check, render
rather than write:

```bash
mig.py gate --run-dir <run-dir>
```

The packet must let the human review, for every unit:

- the FRAME request, source-to-target mapping, declared access, performance-case
  state, and campaign-wide optimization setting;
- the counterpart finding; route ladder; source citations and rationale; frozen
  contract; type stack; build/run evidence plan; and writable paths;
- shared components, their owners, regression obligations, and the ledger order:
  shared declarations first, then one unit at a time, then registration surfaces;
- every diagnostic still needed before a route can be trusted.

A packet is a review of evidence, not an invitation to invent it. A route, source
reading, rationale, measurement, or proposed strategy must be re-investigated from
the target tree or a fresh measurement whenever it is repaired or questioned.

After the human has made a final decision, record that decision exactly as given:

```bash
mig.py confirm --run-dir <run-dir> \
  --intent '<verbatim final human decision>' \
  [--exclude <ids>] [--skip-optimize <ids>]
```

`--intent` stores the **verbatim final human decision**. It is not a paraphrase,
summary, or agent-authored rationale. `--exclude` sets aside migration units; they
remain analyzed and receive no migration write authorization. `--skip-optimize`
sets aside optimization for otherwise authorized units when campaign optimization is
enabled. It does not authorize a rewrite and it does not weaken Phase 7 proof.

A question, condition, or ambiguous reply is not a confirmation. Resolve it, repair
any affected factual artifact through investigation, run the analyzed check again,
and render a fresh GATE-1 packet before calling `confirm`.

### What GATE 1 binds

| Authorized by GATE 1 | Not authorized by GATE 1 |
|---|---|
| The evidenced migration route and declared shared components | A different route, undeclared shared surface, or broader write scope |
| Unit writes within `writable_paths` and lead-owned registration work | A2-source edits, golden/tolerance edits, and contract changes |
| The discovered build and proof plan | Retiling, rescheduling, fusion, workspace removal, or generic tuning |
| The implementation order and source-arch regression obligation | The Phase-10 `L0C→UB` rewrite |

`gate` records the presented `plan.json` and per-unit artifact digests; the migration packet also
records the current `profile.json` digest. `confirm` rejects a changed pending set or changed rendered
artifact. Each migration authorization carries that exact plan/profile/findings snapshot, and
IMPLEMENT and PROVE compare it to the present artifacts before a build or device run. Re-rendering is
required after any packet drift. This detects drift between packet and command; it does not authenticate
an operator. The human decision remains an attended review responsibility.

## Phase 9 — CONFIRM-OPT / GATE 2

GATE 2 exists only when campaign optimization is enabled. It follows Phase 8 SCREEN,
which reads the migrated, proven unit and records fresh applicability, manifest,
baseline measurement, and a proposed strategy. GATE 2 is later than GATE 1 because
none of those facts exist during ANALYZE.

Render the campaign packet:

```bash
mig.py gate --run-dir <run-dir> --phase optimize
```

For every applicable screened unit, the final decision must be one of:

| Decision | Meaning |
|---|---|
| Skip | Leave the proven migration unchanged; its migration result remains reported. |
| Apply `coexist` | Keep both paths and make the rewritten path the selected default through the authorized switch. |
| Apply `replace` | Remove the prior path. This requires an explicit affirmative decision. |

The strategy is not an agent-authored optimization claim. It must be the result of
fresh SCREEN evidence and be identified in the human decision. If the manifest,
reading, rationale, or strategy is corrected, re-run the screened check, re-render
GATE 2, and obtain a new final decision.

Record the decision with the optimization confirmation:

```bash
mig.py confirm --run-dir <run-dir> --phase optimize \
  --intent '<verbatim final human decision>' [--exclude <ids>]
```

Here `--exclude` means skip at GATE 2. It leaves the migration standing and
authorizes no rewrite for that unit. GATE 2 binds only the screened manifest and the
chosen apply/skip and coexist/replace decision. It does not reopen the migration
route, contract, or GATE-1 authorization.

## Repair and evidence boundary

A structural repair may be declared: for example, a missing required artifact field,
an invalid path, or a manifest that no longer describes the files it names. Declare
what is structurally wrong and repair the artifact shape or scope through the
applicable check.

A structural repair is **not** permission to author facts. Routes, source readings,
rationales, measurements, and strategies must be reinvestigated and recorded from
the current source, command output, or profiler evidence. Never carry forward an
old value merely because it fits the schema. If a repair changes material presented
at a gate, check it and present it again before confirmation.

If implementation disproves the approved route or frozen contract while the unit is still AUTHORIZED,
stop it instead of stretching the authorization:

```bash
mig.py park --run-dir <run-dir> --unit <id> --reason '<evidenced mismatch>'
```

Parking at AUTHORIZED invalidates that authorization. Preserve the reason and evidence; deliberately
unpark, re-investigate, run the analyzed check, render a fresh GATE-1 packet, and record a new final
confirmation. The invalidation remains visible through unpark and re-analysis and clears only when
that replacement authorization is recorded. Editing a plan, profile, or findings digest is not a way
to widen the old grant: it follows the same recovery path. If `IMPLEMENTED` was already recorded,
do not park and reuse its rank; begin a fresh campaign. `mig.py status --run-dir <run-dir>` is the
resume authority after any interruption.
