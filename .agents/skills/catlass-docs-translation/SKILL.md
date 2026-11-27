---
name: catlass-docs-translation
description: Translate the English documentation changes that correspond to the current Git diff. Run it after changing Markdown under docs/zh/** so the paired English documents are updated in the same branch and PR.
---

# CATLASS Docs Translation

Follow the root [`AGENTS.md`](../../../AGENTS.md) for repository terminology and writing conventions.

## Workflow

1. Get the diff to translate:

   ```bash
   git diff --unified=0 HEAD -- 'docs/zh/**/*.md'                    # working tree
   git diff --unified=0 upstream/master...HEAD -- 'docs/zh/**/*.md'  # committed
   ```

2. Map each changed file to its English pair. Repository convention is `docs/zh/**` ↔ `docs/en/**`; elsewhere use an established same-directory pair (`name.md` ↔ `name_en.md`, `README.md` ↔ `README_en.md`). A nested `AGENTS.md` may override the direction; for example, `python/tla_dsl/docs/en/api/**` is the source for its paired `python/tla_dsl/docs/zh/api/**`. Inspect neighbouring files for nonstandard layouts; do not invent a target path.

3. Translate by block. Expand each changed hunk to its complete Markdown block (heading section, paragraph, continuous list, table, or fenced code block) and update the corresponding block in the English document. Change nothing outside those blocks.

   - New block in Chinese: add it in English at the aligned position, including any heading the Chinese block introduces.
   - Deleted block in Chinese: delete the corresponding English block.
   - No aligned English block and the alignment is ambiguous: leave the target untouched and report it for human review.

4. Translate in full, never condensed. The English block must carry everything the Chinese block carries: every sentence, list item, table row and column, note, image, image alt text, and code line. A Chinese addition must appear in English; do not summarize or drop details. If the target is marked generated, auto-generated, or “do not edit manually”, change its generator instead of the file.

5. Point every link at its English target:

   - relative link into `docs/zh/**` → the same path under `docs/en/**`
   - `name.md` → `name_en.md`, `README.md` → `README_en.md`
   - `#anchor` → the anchor of the matching English heading, taken from that heading's actual text, for example `#-软硬件配套说明` → `#-software-and-hardware-requirements`; never transliterate an anchor by hand
   - external page → its English edition when one exists (`hiascend.com/en/...`), otherwise keep the original URL; never invent one
   - images follow the same rule, with English alt text
   - when no English counterpart exists, keep the original link and report it

6. Complete the surrounding structure:

   - a new Chinese document needs its English file and an entry in the English index or table of contents (`docs/en/README.md`, `README_en.md`, a section list) wherever the Chinese document was added to its index
   - a renamed or deleted document needs the same change in the English index and in English links that point at it
   - keep the link text in English; do not leave a Chinese label pointing at an English page

7. Self-check the diff:

   ```bash
   python scripts/check_md_links.py <changed English files>
   ```

   Reject and fix anything that fails these checks: Chinese prose left in the English block; a link or anchor still pointing at Chinese content; content dropped or summarized; a broken link or anchor; an edit outside the translated blocks.

8. Commit the English changes together with the Chinese change and open the PR from the same branch.
