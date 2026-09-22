# Phase 2 — PROBE: discover this tree before analyzing it

PROBE is read-only with respect to the target tree. The lead performs complete discovery and writes one
`<run-dir>/profile.json` with four top-level concerns: `build`, `golden`, `registration`, and
`arch-gating`.

```sh
mig.py profile --run-dir <run-dir>
```

`mig.py profile` is an **engine structural validation**, not discovery. It verifies that the document
parses, every required concern is a non-empty object, and `registration.surfaces` has its required
typed shape. It cannot establish that a build command works, a golden is comparable, a target is
runnable, or an architecture selector reaches every necessary layer. The procedures in this reference
are mandatory even when the engine cannot check their result.

## Evidence boundary

Use the target tree for all target-tree facts. The only sources permitted for fixed upstream facts are:

- CATLASS: `https://gitcode.com/cann/catlass` at
  `89e1fc39881a715882b9b47459add06ba270105c`
- asc-devkit: `https://gitcode.com/cann/asc-devkit` at
  `512674e996da0feaee6e7f435e4efc1cad1d74fb`

A pinned-source snippet can illustrate a possible CMake, registration, or architecture pattern. It is
an **illustration only**, never a default, a naming convention, or evidence that the target tree uses
that pattern. Cite a pinned source by its immutable revision and the file or symbol inspected; do not
use a branch, tag, short hash, or third-party source.

## Profile envelope and discovery rules

Keep operational answers at the top of their concern. Record derivation and unresolved absences beside
them, without stale line anchors:

```json
{
  "<field>": "<discovered answer or null>",
  "notes": "<context not consumed by another phase>",
  "evidence": {
    "<field>": {
      "where": ["<path>", "<symbol or construct>"],
      "how": "<command that re-derives the reading>"
    }
  },
  "gaps": [
    {
      "field": "<field>",
      "searched": ["<path or glob>"],
      "consequence": "<what a later phase cannot establish>",
      "proposal": "<optional question or next investigation>"
    }
  ]
}
```

Rules for every concern:

- Discover the full relevant surface, not a convenient example. Follow includes, helper macros,
  presets, scripts, generated registries, and callers until the answer is complete.
- Never omit a required answer because the tree lacks it. Use `null` with evidence and a `gaps[]` row.
  `null` means investigated absence; omission means not investigated.
- Never substitute a plausible default. An unknown build entry, artifact, architecture flag, target name,
  comparator, or registration mechanism must reach the human as an unresolved finding.
- Preserve commands as argv arrays where later execution needs them. Do not turn shell syntax into an
  argv list; wrap a genuinely shell-dependent command in an explicit shell argv.
- A missing toolkit loader or unavailable device is recorded as environment evidence and is
  **nonblocking** for discovery. It limits later proof claims; it does not justify a guessed result or
  prevent `profile` from completing.

## Concern `build`

Discover the actual path from the campaign's build entry to an installed or build-tree executable. Read
the entry's argument parsing, configure invocation, build invocation, install behavior, artifact naming,
and any opt-in macro declaration an example owns. A directory name, test-case string, or CMake target
name alone does not establish the runnable artifact.

```json
{
  "build_entry": ["<argv>"],
  "target_build": ["<argv with the target architecture selected explicitly>"],
  "regression_build": ["<argv with the source architecture selected explicitly>"],
  "artifact_path": "<directory containing the executable>",
  "install_required": true,
  "binary_name_source": "<target name | OUTPUT_NAME | install rename | discovered rule>",
  "env_setup": ["<argv that loads required environment>"],
  "env_setup_present": true,
  "optin_switch": {
    "declaration": "<verbatim option/definition pair or null>",
    "where": "<path owning the declaration>",
    "reaches_preprocessor_via": "<compile-definition mechanism or none found>"
  },
  "notes": "<configure cwd, cache location, forwarding rules, and other findings>"
}
```

Required discovery:

1. Locate the top-level build entry or the CI/preset command that is actually authoritative.
2. Read its option handling and final configure/build/install calls. Record the exact argument form that
   selects source and target architectures. Each recorded build command must select its architecture
   explicitly; shared caches must not supply an implicit previous value.
3. Trace the executable from leaf declaration through `OUTPUT_NAME`, install destination, and any rename.
   If no install rule exists, record the build-tree artifact and `install_required: false`.
4. Inspect an existing example for the complete opt-in switch declaration: a CMake cache variable is not
   itself evidence of a preprocessor definition. Record `null` and a gap when no declaration exists.
5. Check whether the environment loader named by the command exists in a fresh shell. Record the reading;
   do not block PROBE when it is absent.

## Concern `golden`

Discover the source example's actual numerical contract: the leaf executable, its argument parsing,
data generation or input files, comparator call, failure propagation, and device observation. A success
string, exit status, or available device is not a proof contract by itself.

```json
{
  "comparator": {
    "entry_point": "<exact source call>",
    "header": "<path>",
    "workflow_class": "<in-process-direct | in-process-statistical | file-backed | external-verifier>",
    "zero_baseline_treatment": "<observed behavior>",
    "false_pass_risk": "<observed risk or none established>"
  },
  "tokens": {
    "positive": ["<observed token>"],
    "negative": ["<observed token>"],
    "stream": "<stdout | stderr | both>",
    "coexisting_faults": ["<fault marker that can accompany success>"],
    "status_propagation": "<propagates nonzero | returns zero on failure | unresolved>"
  },
  "data_files": {
    "generator": "<path or null>",
    "inputs": ["<required path>"],
    "cwd_required": "<path or any>"
  },
  "device": {
    "tool": ["<target-tree device-discovery argv>"],
    "output": "<recorded observation>",
    "can_run_target_arch": "<yes | no | undetermined>"
  },
  "notes": "<golden precision, fill and compute symbols, and other findings>"
}
```

Required discovery:

1. Expand the leaf declaration so an aggregate or helper target is not mistaken for a runnable program.
2. Read the source CLI, input loading, generator, comparator invocation, and all status propagation
   through wrappers and `main`.
3. Determine how zero CPU reference metrics are handled and whether an error can coexist with a success
   token. Record the weaker behavior when multiple comparator paths exist.
4. Discover required data files and working directory from the code that consumes them.
5. Record the target-tree device-discovery command and the observation used by the FRAME access
   declaration. A missing runtime dependency is an environment finding; it is not evidence that a
   matching device is absent.
6. If the source lacks a CPU golden, record the gap and its consequence. Do not nominate another oracle.

## Concern `registration`

Discover how directories enter the architecture-specific build, how a leaf executable is declared, how
tests or other runtime cases register it, and how target-directory names are measured. Fully expand one
or more representative existing target-architecture directories as needed to understand helper macros;
those expansions are illustrations of the target tree's mechanism, not templates for new units.

```json
{
  "dir_registration": {
    "mechanism": "<partitioned-lists | glob | option | other discovered mechanism>",
    "source_arch_list": { "var": "<name or null>", "counted": 0, "where": "<path or null>" },
    "target_arch_list": { "var": "<name or null>", "counted": 0, "where": "<path or null>" },
    "selection": "<architecture-selection mechanism>",
    "entry_loop": "<directory-entry mechanism>"
  },
  "leaf_declaration": {
    "macro": "<macro name | add_executable | other>",
    "signature": "<verbatim declaration>",
    "target_name_supplied_by": "<positional | keyword | discovered rule>"
  },
  "sample_expansion": {
    "directory": "<existing target-architecture directory>",
    "leaf_targets": ["<expanded runnable target>"],
    "aggregate_targets": ["<non-runnable target>"]
  },
  "test_registry": {
    "framework": "<framework or null>",
    "file": "<path or null>",
    "setter": "<registration function or null>",
    "arch_decorator": "<architecture gate or null>",
    "case_lists": {
      "<variable>": { "counted": 0, "count_method": "<programmatic method>" }
    }
  },
  "naming": {
    "dominant_form": "<shape or null>",
    "dominant_count": "<count/total | none yet>",
    "exceptions": ["<registered exception>"],
    "documented_policy": false,
    "status": "<measured | proposed | tied>"
  },
  "surfaces": [
    {
      "path": "<existing file under target_root>",
      "symbol": "<symbol or null>",
      "required": true,
      "why": "<why a target unit must register here>"
    }
  ]
}
```

`surfaces` is the engine-consumed profile field. Each row is closed to `path`, `symbol`, `required`, and
`why`; `path` must already exist inside `target_root`, `required` is a JSON boolean, and `symbol` may be
`null`. A unit's own new `CMakeLists.txt` is not a registration surface because it belongs inside that
unit's write grant.

Required discovery:

1. Read the architecture-specific directory selection and entry loop in full, including included CMake
   modules and generated lists.
2. Expand leaf declarations through helper macros; distinguish runnable leaves from aggregate or custom
   targets.
3. Read the complete test or runtime registry and count generated entries programmatically rather than
   from a line span or visual layout.
4. Derive naming only from registered target-architecture entries. Record a tie rather than choosing it;
   with no entries, record a proposed source-derived form and carry that uncertainty to the human.
5. Identify every central existing file that must receive registration text. These are `surfaces` and
   must later be declared by every unit's findings.

## Concern `arch-gating`

Discover the selector, every propagation path, and the host/device guard forms. The selector may live in
a top-level CMake file, included module, toolchain file, preset, script, or CI configuration; the first
plausible location is not a complete answer.

```json
{
  "selector": {
    "name": "<selector>",
    "kind": "<cache variable, compiler flag, preprocessor definition, or combination>",
    "source_value": "<source value>",
    "target_value": "<target value>",
    "default_when_unset": "<value | none>",
    "set_at": "<path>",
    "gating_location": "<top-level CMakeLists | module | toolchain | preset | script | CI>"
  },
  "propagation": [
    {
      "path": "<path>",
      "form": "<verbatim declaration>",
      "controls": "<compiler architecture, preprocessor partition, or other role>",
      "breaks_if_missing": "<observed consequence>"
    }
  ],
  "guards": {
    "host": "<host/build guard or null>",
    "device": "<device guard or null>",
    "partition_idiom": "<verbatim conditional form or null>"
  },
  "notes": "<tag declarations and unresolved findings>"
}
```

Required discovery:

1. Locate where the selector is set, not merely where it is read.
2. Search its propagation through compiler options, compile definitions, generated configuration, and
   device compilation. Separate propagation roles remain separate findings.
3. Read the actual host and device conditional forms and the architecture-tag declarations they select.
4. If a selector, target tag, or propagation role is absent, record the searched surface and consequence.
   Do not infer that an unset selector has a safe default.

## Convergence

Before invoking `mig.py profile`, re-read `profile.json` as one document. Check cross-concern consistency:

- the target build command passes the selector in the form `arch-gating` discovered;
- the selected leaf target produces the artifact `build` records;
- the `golden` CLI and required working directory match the future run command;
- every required registration surface exists and is reachable under the target architecture;
- the naming state used by FRAME agrees with registration evidence.

Run `mig.py profile --run-dir <run-dir>` only after this full discovery. Structural success records the
profile; it never converts a gap into an answer. Carry every gap or contradiction unchanged into
ANALYZE and GATE 1.
