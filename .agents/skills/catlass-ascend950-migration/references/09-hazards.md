# Hazards — hardware limits, repository gaps, and evidence

Read this as a conditional reference during ANALYZE, IMPLEMENT, and PROVE. It does not describe the target tree by default: discover target-tree facts in `profile.json` and `findings.json`, and preserve the reading that supports each route decision.

Fixed facts in this file come only from [CATLASS](https://gitcode.com/cann/catlass) at `89e1fc39881a715882b9b47459add06ba270105c` and [asc-devkit](https://gitcode.com/cann/asc-devkit) at `512674e996da0feaee6e7f435e4efc1cad1d74fb`. Pin citations name a source role or path, not a lasting line number. Re-read the pin before using a fact in a gate packet.

## Hardware limit versus repository gap

A hardware limit and an absent target-tree component require different action.

| | Hardware limit | Repository gap |
|---|---|---|
| Meaning | 3510 does not offer the precise instruction, type, or data path | the target tree has no contract-equivalent implementation yet |
| Evidence | a scoped statement in pinned asc-devkit | recorded search of the target tree, optionally informed by pinned CATLASS |
| Route effect | excludes only the implementation that uses that capability | does not itself exclude a route class |
| Never implies | a new contract, a fabricated replacement, or automatic `reimplement` | impossibility in hardware, `redesign`, or permission to change the contract |

For every proposed exclusion: name the exact capability, locate the scoped asc-devkit fact, and verify that the source path actually uses it. If that chain breaks, record a repository gap and the search that established it. Continue down the preservation ladder: a missing component may be a local generalization or an existing isomorphic dataflow, not a reason to abandon the source contract.

## 3510 capability changes that affect route decisions

The asc-devkit compatibility guide at the pinned revision is the source for these migration facts. They are constraints, not invitations to tune unrelated code.

| Source capability | Pinned fact | Migration consequence |
|---|---|---|
| L1 Buffer → GM direct data path | removed on 3510 | use only an already supported, contract-equivalent path; do not synthesize a replacement from analogy |
| GM → L0A/L0B direct path | removed on 3510 | separate the legal staging steps while preserving the dataflow contract |
| L1 → L0A transpose | no longer supported for the affected form | trace the selected layout and transpose behavior; a different transpose location requires contract analysis |
| L0A/L0B initialization | the former initialization instructions are unavailable; `Fill` does not initialize those LocalTensors | preserve the required padded-tail value with a legal, proven source rather than deleting initialization |
| 4:2 structured sparse Cube path | unsupported | exclude only that sparse instruction path; other formats need independent evidence |
| `SetLoadDataBoundary` | unsupported | re-express the guard explicitly; do not silently delete it |
| L0C → UB Fixpipe destination | available on 3510 | relevant only to the separately screened, non-TLA rewrite in Phases 8–10 |

A removed path never authorizes a contract change. If every legal replacement changes dtype, layout, scale, mask, aliasing, output region, zero-work behavior, or supported domain, park the unit as new-contract work instead of calling it a migration.

### Direct Cube s4 refusal, and the stored-s4 exception

The following three facts must be held together; using only the Cube row produces the wrong diagnosis.

| Interface | Pinned asc-devkit fact | Consequence |
|---|---|---|
| `LoadData` | the Ascend 950 type list in the `LoadData` API does not include `int4b_t` | do not preserve an s4 `LoadData` path into the 3510 Cube pipeline |
| `LoadDataWithTranspose` | the Ascend 950 type list in that API does not include `int4b_t` | do not preserve an s4 transpose-load path into the 3510 Cube pipeline |
| `Mmad` | the 2201→3510 compatibility guide says Cube does not support s4 | `int4b_t` is not a legal direct Cube operand on 3510 |

**Refusal:** a unit whose Cube operands remain `AscendC::int4b_t` is not a direct 2201→3510 migration. Do not reinterpret the operand, alter the golden, or claim that an s4 Cube path will be made to work.

**Narrow stored-s4 exception:** s4 may remain an externally stored representation when the Cube operands are no longer s4. The pinned asc-devkit basic-API migration guide gives the target-native compatibility sequence: a CV-fused Vector Core step casts stored `int4b_t` to `int8_t`, saves the converted result in new GM storage, then stages it through UB/L1 for an `int8_t` Mmad. This is a **stored-int4 → Vector-Cast-to-int8 → Cube-int8** pipeline, not direct s4 Cube support.

Use that exception only after proving that the frozen external contract is preserved and that the selected target implementation performs the full conversion before Cube. Record the Vector Cast, converted storage, and int8 Cube operands in the route evidence. It does not waive the direct-Cube-s4 refusal, does not allow an unverified shortcut, and does not convert a required contract change into migration work.

## `BlockMmad` diagnostics are type-stack evidence

At the CATLASS pin, `BlockMmad` aggregate templates and their architecture-gated specializations are source-level selection machinery. A `static_assert` from a primary template means the selected type arguments matched no specialization; it is not a hardware diagnosis.

Investigate in this order:

1. Trace the exact selected aggregate, dispatch policy, layouts, tile-copy type, arity, and includes.
2. Search the target tree for a specialization matching the required form and for its current consumers.
3. If the specialization exists, repair the arguments or wiring without changing the algorithmic form.
4. If the source form is legal but a policy, checker, guard, or copy wrapper is A2-bound, record a shared `generalize` component and preserve its existing consumers.
5. If a pinned asc-devkit fact excludes the exact source path, exclude only that path and continue the route ladder.

A missing aggregate symbol and a primary-template assertion are distinct diagnostics. Treat each as a repository observation until a scoped hardware fact says otherwise. A clean 3510 build proves only that a type stack resolved; it says nothing about numerical correctness.

## Layout, selector, and provenance hazards

Pinned CATLASS provides architecture-sensitive layout selectors and separate Gemm, Conv, TLA, and non-TLA aggregate families. Follow the selector and the specialization actually instantiated; never hard-code an L0 layout from a name or carry a rule from one aggregate into another.

In particular:

- trace L0A, L0B, L1, and L0 element/layout choices through the selected helper, tile copy, and block;
- treat related aggregate names as separate types unless their base clauses establish a common contract;
- inspect every partial specialization and its ordering before generalizing an A2 specialization for 3510; and
- preserve tensor provenance through epilogue adaptation: C location, scale/per-token inputs, output layout, offsets, and address-space conversion each come from the selected implementation, not template arity or a matching name.

A layout error can compile and still produce wrong values. Compilation is therefore a routing signal, never accuracy evidence.

## Zero work and grouped axes

Expand the selected Block, Epilogue, Scheduler, and golden path before declaring a zero-work case safe. Check these independently:

| Case | Question |
|---|---|
| zero K before the main loop | can any tile count become a divisor or modulus before a zero-length guard? |
| zero K with queued events | is every initial wait paired with a seed or re-arm when the loop executes zero iterations? |
| zero K epilogue | does Cube and Vector handling remain symmetric while preserving required offsets and output semantics? |
| zero-row AIV half | does a data-only guard preserve all required synchronization credits? |
| terminal zero-M grouped state | does a scheduler reset all tail/window state rather than retaining the prior group? |

For group lists, identify the partitioned axis and whether the encoding is cumulative or segmented. Validate list length, element width, monotonicity or non-negativity, total coverage, and explicit zero-work entries. The producer, kernel, golden, and comparator must use the same interpretation and cover every materialized output region. Construct zero-work cases deliberately; a random group list is not evidence that one was exercised.

Source shape establishes a risk and a test case. Only an executed target-device case establishes a device outcome.

## Cross-core synchronization

The paired-AIV direct path has two distinct protocols:

- `SPLIT_M`: each AIV contributes one credit per ring slot, Cube accounts for both, and the epilogue releases each local slot after use.
- `NO_SPLIT`: one credit exists per slot. Referencing a second-sub-block identity waits for a credit that cannot arrive.

For either protocol, derive flag identities, valid ranges, synchronization mode, and pipes from the kernel that actually compiles. Do not infer them from a different header, define a repository-wide `FLAG_ID_MAX`, or rely on a census of similarly named constants. The required invariants are balanced seed/steady-state/drain counts, disjoint identities for every live stage, and ordering on the pipe that performed the work.

A runtime zero-row half may omit data computation but must still preserve those invariants. Static checks can establish range and arithmetic conditions; target-device proof establishes whether the selected implementation is live and correct.

## Resource budgets are not hardware capacities

Pinned CATLASS `Arch::Resource<ArchTag>` and `ArchTag` constants are compile-time allocation budgets for a particular architecture type. Sum live intervals against the selected architecture budget; do not derive a capacity, alignment rule, bank-conflict rule, or cross-generation sharing decision from a single budget constant.

For any direct-path change, one byte expression must reconcile the producer allocation, consumer row pitch, ring stride, and compile-time ceiling. Use the selected API's alignment requirements separately. Preserve the original arch-specific specializations or extract a genuinely shared implementation only after checking specialization ordering and every existing consumer.

## Evidence hygiene

Accuracy evidence is exactly one parsed `CATLASS_EVIDENCE` line from the current run on matching declared hardware, with the frozen shape and `computeNum`. It is not established by any of the following:

- a successful build, registration entry, exit status, or success string;
- a previous run's log or measurement;
- a device identifier remembered from an earlier session;
- an untracked scratch artifact; or
- a source location remembered without re-reading the pinned or target source.

A shared-surface edit also owes a source-architecture regression build of an existing consumer. A Phase-10 rewrite owes the ordered sequence in `references/07-optimize.md`: repeated measurement and APPLY artifact, `check applied`, unchanged `prove`, then `check optimized`. Never change the oracle, golden, tolerances, or compared region to make an outcome pass.
