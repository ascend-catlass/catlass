# Phase 1 — FRAME: scope, pins, access, and the frozen plan

FRAME turns the human request into a frozen campaign scope. It is **not** a write gate: it records
what was requested and the conditions under which later phases may make claims. It does not edit the
target tree, choose an implementation route, or authorize migration or optimization writes.

## Authoritative sources

Use these sources only for fixed upstream facts. Target-tree facts are discovered from the target tree
in PROBE and ANALYZE; they are never inferred from either reference.

- CATLASS: `https://gitcode.com/cann/catlass` at
  `89e1fc39881a715882b9b47459add06ba270105c`
- asc-devkit: `https://gitcode.com/cann/asc-devkit` at
  `512674e996da0feaee6e7f435e4efc1cad1d74fb`

A revision is an immutable, full lowercase hexadecimal commit identifier. Do not cite, configure, or
accept a branch, tag, abbreviated revision, moving URL, or an unpinned checkout as evidence.

## Phase boundary and command sequence

1. Resolve the request to source example directories and measure target-directory naming from the
   target tree (§3 and §4).
2. Measure available hardware using the target tree's device-discovery command and prepare an access
   declaration (§5).
3. Present one intake prompt containing the resolved scope, access declaration, optional performance
   input, and global optimization choice (§8).
4. Write `plan.json` and `access.json` from the interpreted human decision.
5. Initialize the run:

   ```sh
   mig.py init --plan <plan.json> --access <access.json> [--perf-cases <path>]
   ```

   `init` prints the run directory. Use that exact path in every later command.
6. Resolve the two pinned references before relying on an upstream fact:

   ```sh
   mig.py refs --run-dir <run-dir>
   ```

All generated material belongs under `<target_root>/.agents-work/`. A supplied `--run-dir` may name a
run within that fence but must not relocate it outside the fence.

**Return unusable input to the human.** A request that cannot identify source directories, an ambiguous
or colliding target name, malformed access declaration, unreadable or unusable performance input, or a
plan rejected by `init` is a question for the person who owns the request. Do not invent a replacement,
drop a named candidate, or continue with a plausible interpretation. Re-present the resolved state and
ask for the correction needed.

## `plan.json`

Keys are closed at every level. The following schema is the FRAME artifact; placeholders describe shape,
not values to freeze.

```json
{
  "version": 1,
  "request": "<human request, verbatim>",
  "target_root": ".",
  "refs": {
    "catlass": "89e1fc39881a715882b9b47459add06ba270105c",
    "asc-devkit": "512674e996da0feaee6e7f435e4efc1cad1d74fb"
  },
  "optimize": { "enabled": true },
  "units": [
    {
      "id": "<path-safe unique unit id>",
      "source": "<existing source directory relative to target_root>",
      "target": "<non-existing target directory relative to target_root>",
      "note": null
    }
  ]
}
```

| Field | Requirement |
|---|---|
| `version` | Required integer `1`. |
| `request` | Required non-empty string, stored verbatim. Never summarize, improve, or author it. |
| `target_root` | Optional existing directory; defaults to `.`. All planned paths and later write grants must resolve inside it. |
| `refs` | Optional only when the bundled values are used. If supplied, each named key may be only `catlass` or `asc-devkit` and must exactly reproduce that key's full lowercase immutable revision above; it cannot select an alternate commit. |
| `optimize.enabled` | Optional JSON boolean; defaults to `true`. `false` is the **global FRAME-only opt-out** of SCREEN, CONFIRM-OPT, and APPLY. Changing it requires a new run. |
| `units` | Required non-empty list. It is the entire campaign scope. |
| `units[].id` | Required, unique, path-safe token used by `units/<id>/` and `--unit`. |
| `units[].source` | Required existing relative directory inside `target_root`; it must not escape through `..` or a symlink. |
| `units[].target` | Required relative path inside `target_root`, with no placeholder delimiters, and must not already exist. |
| `units[].note` | Optional JSON value for intake context only. Information needed at a gate belongs in ANALYZE findings. |

`init` freezes the plan. Reuse the same run only with the identical plan; correct scope, name, access,
or global optimization policy by creating a new run after the human resolves it.

## Resolve every requested unit

Copy the request into `plan.request` before interpreting it. Resolve words, numeric identifiers, and
family names mechanically against the target tree's source-architecture examples, then show the resulting
`source → target` list to the human before `init`.

1. If no operator is named, enumerate candidate source example directories and ask for scope. Do not
   manufacture a request.
2. Match request terms against directory names and build registration. A word, number, slug, directory,
   or target-architecture sibling is only a lead, never proof of identity or buildability.
3. Check that each candidate is a source-architecture example with an external contract and CPU golden
   suitable for later freezing. Shared libraries, header-only components, and aggregate-only directories
   are analyzed as components of a unit, not made into units themselves.
4. Form a campaign cluster from units sharing a dispatch policy. Determine the grouping by reading the
   policy symbols selected by their source examples, not by a naming family. One campaign operates on one
   cluster; all units using a shared declaration must be in that cluster.
5. Show every named candidate and every cluster member added for its shared policy. Ambiguity is returned
   to the human with the candidate set and the missing distinction.

Never silently drop a named candidate. An apparent target counterpart is an ANALYZE finding, not a FRAME
exclusion. The only hard refusal is a source whose **direct Cube operands** are `AscendC::int4b_t`: no
contract-preserving target Cube route exists. Surface that refusal to the human and stop for an explicit
re-scope or separate backend decision. This refusal does **not** apply where the public storage is int4
but the prologue converts it to int8 before Cube; that unit remains a candidate and must be fully
analyzed.

A target collision is likewise not a reason to remove the unit. State the existing path and ask whether
the target name must change, the alleged counterpart must be investigated, or the human wants the unit
set aside at GATE 1.

## Measure target-directory naming

Measure naming from registered target-architecture entries in this target tree. Directory names on disk
are not registration evidence.

1. Locate the target-architecture registration mechanism in the build files.
2. Enumerate the registered entries and reduce them to forms by replacing the varying number and operator
   slug with placeholders.
3. Count forms from registered entries. Use a strict majority when one exists.
4. Prefer a consistently registered operator-family form over an unrelated global majority.
5. When no target-architecture entry exists, record that no convention is measured and mark a
   source-derived proposal as `proposed`.
6. When forms tie, family evidence conflicts, or the measured target already exists, present all
   alternatives to the human. Do not choose one.

The intake prompt must display each `source → target` mapping with the measurement status (`measured`,
`proposed`, or `tied`) and its supporting build location. PROBE independently records the registration
mechanism; a disagreement means the plan must be corrected before a target-tree write is authorized.

## Access is declared at `init`

Access is measured before the intake prompt and declared when the run is initialized. It determines what
later phases may claim; it never blocks discovery or compilation. An inaccessible target device means a
unit may become `COMPILED` but cannot become device-proven.

Pass a declaration to `init`, including explicit `reachable: false` where no access is available.
Zero, one, or both sides may be reachable; the absence of a reachable side limits later claims but
never blocks compilation:

```json
{
  "a2": {
    "reachable": false,
    "notes": "<measurement or limitation>"
  },
  "a5": {
    "reachable": true,
    "arch": "3510",
    "soc": "Ascend950",
    "host": "<declared device location>",
    "device": "<measured device identity or selector>",
    "notes": "<measurement or limitation>",
    "transport": null
  }
}
```

Each side is closed to `reachable`, `arch`, `soc`, `host`, `device`, `notes`, and `transport`.
`reachable` is a JSON boolean. `a2` denotes the source architecture and `a5` the target architecture.
For every reachable side, `arch`, `soc`, `host`, and `device` are required non-empty strings.
`arch` is exactly `2201` for `a2` and `3510` for `a5`. `soc` must identify the declared side's
family: `910`, `Atlas A2`, `Atlas A3`, or `2201` for `a2`; `950`, `Ascend950`, `A5`, or `3510` for
`a5`. `device` records the measured device identity or selector observed at FRAME, not an arbitrary
selector. This identity is a FRAME measurement, not a device-run inference; an `a5` declaration for
a 910-family SoC is invalid before a device command can execute. An absent device is an honest `false`,
not `unknown` or an implied local device.

For a remote reachable side, `transport` is an object closed to `kind`, `host`, `user`, `port`,
`identity_file`, `password`, `password_env`, `workdir`, and `ssh_options`:

```json
{
  "kind": "ssh",
  "host": "<ssh host>",
  "user": "<optional user>",
  "port": 22,
  "identity_file": "<optional local identity path>",
  "password_env": "<optional local environment-variable name>",
  "workdir": "<absolute remote work directory>",
  "ssh_options": ["<optional ssh option>"]
}
```

`kind` is `local` or `ssh`; `ssh` requires `host` and `workdir`. `password` and `password_env` are
mutually exclusive. The build remains local; only the device run is staged through declared transport.
A remote host without transport is not reachable for a run. Do not hand-roll a remote wrapper into a
later run command.

The declaration is fixed at `init`. A passing proof retains its exact `{arch, soc, host, device}` in
`proof.json` and in the `unit.proven` hardware object. Current accuracy requires the latest proof to
carry that bound identity and no later `unit.prove_failed`; an old `unit.proven` event without it is a
historical observation, not current accuracy. If the declaration is malformed, conflicts with the
measured result, cannot express offered remote access, or cannot be loaded, return it to the human
before initialization.

Operational access loading is strict: a reachable side with an old, missing, or invalid identity
cannot run device work, a gate, or optimization. `status` and `report` alone load an older declaration
leniently so its historical campaign remains readable; their **NOT ESTABLISHED** result does not make
the old proof current. Record a structural declaration repair only from the original or a new hardware
measurement, then re-run `prove` before accuracy or optimization eligibility can be current. Never
backfill, infer, or invent a missing identity.

## Pinned-reference resolution

`mig.py refs --run-dir <run-dir>` resolves only the two source/revision pairs listed at the top of this
document into the run cache. It may use a valid cache, a local checkout containing the exact commit, or
a clone of the stated source URL. It must verify that the requested commit is present before recording
resolution. If it cannot resolve a pin, report the attempted source and commit and stop before an
upstream claim is used.

A target-tree observation does not need a reference citation. A fixed architecture or library claim must
cite one of the two pinned sources and identify the file or symbol inspected; do not borrow facts from
other repositories, articles, or performance guides.

## Performance cases

Performance cases are optional human input. A supplied file is passed on the same `init` command as
`--perf-cases <path>` and staged at `<run-dir>/perf_cases.md`.

When no table is supplied, `init` **automatically stages** the bundled
`assets/perf_case_template.md` at that same path. The fallback is not a later manual step and does not
justify invented measurements. Fill only values that declared access can measure; leave unavailable
values unmeasured.

If a supplied path is unreadable, not a usable table, or cannot be staged, return that result to the
human. They may correct or replace the input, or explicitly choose the bundled fallback. Never silently
discard a supplied table or synthesize cases.

## One intake prompt and the optimization choices

After scope, naming, and access are measured, issue one prompt containing all of the following:

1. the verbatim request and resolved unit list, with each target-name status;
2. the measured access declaration and any remote-access fields required to make an offered box usable;
3. the optional performance-case input, stating that the bundled template is staged when none is supplied;
4. the optional global optimization opt-out, stating that optimization defaults to enabled.

A reply that changes only scope or says to proceed accepts the displayed access, bundled performance
fallback, and optimization default. Re-prompt only for an unresolved or unusable input; always show the
interpreted state rather than asking the human to repeat it.

With `optimize.enabled: true`, GATE 1 has a separate **per-unit early-skip** decision. The human may
authorize migration while excluding selected units from SCREEN and the optimization gate:

```sh
mig.py confirm --run-dir <run-dir> --intent '<final human decision, verbatim>' \
  [--exclude <comma-separated-unit-ids>] \
  [--skip-optimize <comma-separated-unit-ids>]
```

`--exclude` sets a unit aside from migration. `--skip-optimize` retains its migration path but records
that it will not enter optimization phases. The command's `--intent` stores the final human decision
**verbatim**; do not replace it with an agent summary. Form changes and other migration authorization
remain GATE 1 decisions described in `references/03-analyze.md`.
