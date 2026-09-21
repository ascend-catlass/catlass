---
name: catlass-docs-translation
description: Update CATLASS Markdown translations with a Git-only incremental workflow.
---

# CATLASS Docs Translation

Follow the root [`AGENTS.md`](../../../AGENTS.md) for repository terminology and writing conventions. This skill owns the translation workflow.

## Git-only incremental workflow

Git history is the sole translation state. Do not create a configuration file, metadata repository, hash cache, document annotation, or translation harness.

1. A translation becomes **ready** only after its reviewed change is merged in a commit containing this fixed trailer:

   ```text
   Translation-Ready: true
   ```

   The marked merged commit itself is the baseline. For an existing target, find the most recent marked commit that changes that target; for a new target, the first marked merged translation commit establishes its baseline. Do not store or copy a source hash.

2. Fetch upstream and derive work solely from the source delta:

   ```bash
   git fetch upstream master
   git log --format='%H%n%B' -- <target.md>  # find Translation-Ready: true
   git diff --unified=0 <ready-commit>..upstream/master -- <source.md>
   ```

3. Expand every changed hunk to its complete Markdown block: heading section, paragraph, continuous list, table, or fenced code block. Only update paired target blocks for those hunks. Existing target blocks outside the hunk are immutable.

4. An empty target may receive a first full translation. For a non-empty target, establish source/target block alignment before writing. If alignment is ambiguous, leave it unchanged and report it for human review; never replace the whole document to force alignment. When a source block changed after the last `Translation-Ready` commit, always inspect its current target block, even if the target was edited later. Target modification time, target commit time, or the mere existence of target changes is never evidence that translation is complete.

5. If the current target block appears to be a manual translation, assess it against the changed source block for completeness, technical meaning, terminology, links, and Markdown structure. When it passes that review, retain it and do not machine-translate or rewrite it; record that it was accepted as an existing manual translation. If it does not pass, translate only the aligned block.

6. Discover pairs from repository convention: `docs/zh/**` maps to `docs/en/**`; elsewhere use an established same-directory pair such as `name.md` and `name_en.md`. A nested `AGENTS.md` may explicitly override the source direction; for example, `python/tla_dsl/docs/en/api/**` is the English source for its paired `python/tla_dsl/docs/zh/api/**` reference. Inspect neighbouring files for nonstandard layouts; do not invent a target path or parallel directory.

7. Skip files marked generated, auto-generated, or “do not edit manually”; change their generator only when the task explicitly includes it. Use any caller-approved translation executor in bounded file or section units.

8. Review the resulting Git diff: reject residual Chinese in English prose, links newly pointed at Chinese docs, altered anchors, changed generated-file notices, and changes outside approved translation blocks. Split PRs by requested scope. Put `Translation-Ready: true` in the PR description; GitCode carries that description into the merged commit body. The source branch commit should carry the same trailer, but the merged commit marker is authoritative.

## Branch and PR contract

- Create every translation branch from the reviewed upstream base as `agent-translation/<scope>`, where `<scope>` is a short kebab-case directory or document scope. Do not use generic `codex/` or model-specific branch names.
- Every translation PR title must begin with `[AI Skill Translation]`, followed by the conventional type and concise scope. For example: `[AI Skill Translation] docs(tla_dsl): update print guide`.
- The PR body must state that it was produced with `catlass-docs-translation`, list the translated files, and end with `Translation-Ready: true`. Do not put the marker in the title.

Use this body shape:

```markdown
## AI skill translation

Produced with `catlass-docs-translation`.

## Scope

- `<translated-file>`

## Validation

- Reviewed the scoped Markdown diff
- Checked links and residual Chinese prose

Translation-Ready: true
```
