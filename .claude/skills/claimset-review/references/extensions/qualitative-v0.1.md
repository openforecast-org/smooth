# Claimset extension: qualitative — v0.1 (draft)

Status: working draft · Date: 2026-11-06 · Core: claimset v0.38 and later; lab record v0.20 and later · Name in files: `qualitative`

Qualitative research reads material, codes it against a codebook, and draws its findings from the coded material: themes with their excerpts, and counts of what was coded. The core holds what every field shares: captured sources, excerpts with their locators, definitions of concepts and themes, interpretive claims with their warrants, and the `qual` evidence profile (claimset §3.4; lab record §5.20). This extension adds the codebook, the codings, and counts of coded material, in the claimset and in the lab record.

## 1. In a claimset

### 1.1 Codes (on a definition)

A code of the study's codebook is a definition of kind `concept` with a `qualitative` block holding its criteria:

```json
{"id": "D4", "term": "arrange for ensemble", "kind": "concept", "broader": ["D3"],
 "definition": "A request that the software arrange existing music for a given set of instruments.",
 "ext": {"qualitative": {"code": {"include": "asks for automatic arranging for named instruments",
                                  "exclude": "manual tools such as explode/implode",
                                  "examples": ["arrange this piano score for brass quintet"]}}}}
```

- `code`: `{include, exclude, examples}`. `include` (required): what a passage must show to get the code; `exclude`: what does not count, though it may look alike; `examples`: typical passages. The criteria are part of what the code means. A code's theme is a broader definition (`broader`, claimset §3.2).

### 1.2 Tests over coded material (on a claim)

A claim whose test was decided over coded material has a `qualitative` block on the claim:

- `test_flags`: weaknesses of such a test, besides the core's (`test.flags`, claimset §3.3.3):
  - `coded_before_plan`: a deciding run read codings, of the codes the test counts (or, when it counts none, of any code), written before the plan: the material was coded before the test was fixed;
  - `codebook_changed_after_plan`: a code the test counts (or a code applied by codings the deciding runs read) was redefined between the plan and a deciding run: the codebook the test used is not the one the plan fixed.
- `counts`: `{name: count}`, the counts the test's quantities bind (§3.4). Each count's `codes` are definition IDs of codes (§1.1). The test has a quantity of the same name binding the counter's result, `{"variable": "count: <name>"}`.

### 1.3 Rules

qualitative.1. A `qualitative` block is on a definition (only `code`) or a claim (only `test_flags` and `counts`).
qualitative.2. A `code` is on a definition of kind `concept`; it has a non-empty `include`, and only `include`, `exclude` (non-empty text) and `examples` (a list of texts).
qualitative.3. `test_flags` is a non-empty list of `coded_before_plan` and `codebook_changed_after_plan`, on a claim that has a `test`.
qualitative.4. `counts` is a non-empty object; each count is well formed (§3.4), names only definitions that are codes, and the claim's test has a quantity of its name with `variable` `count: <name>`.

## 2. In a lab record

The extension is declared in the record by an `extension` event (lab record §5.22); the tool writes it the first time a command needs the extension (recording a code, a coding or a count). Records begun before lab record v0.20 have it from their start: their terms' `code`, their codings, their runs' `codings` and their plans' count quantities are read as this extension's, in the forms written then.

### 2.1 Codes (on a term)

A term of kind `concept` with a `qualitative` block `{code: {include, exclude, examples}}` is a code of the codebook, as in §1.1. Changing its criteria is a correction of the term (`--supersedes`); codings made before name the version they applied.

`claimset lab term "arrange for ensemble" --kind concept --definition "…" --include "…" [--exclude "…"] [--example "…"] [--broader TM3]`

### 2.2 `coding` (prefix `CD`)

`{source, codes, code_versions, quote, unit, pass, memo, withdrawn, none_apply}`: codes of the codebook applied to a captured source (lab record §5.20), or to a passage of it, by the event's writer. Written by `claimset lab code SR1 --codes TM4 TM7 [--excerpt "the words"] [--unit post] [--pass 1] [--memo "…"]`.

- `source`: the source coded. Required.
- `codes`: the codes applied, as term IDs of codes (§2.1). Required, at least one, each once.
- `code_versions`: `{code: version}`: the version of each code's term in force when the coding was written, recorded by the tool, so codings made under an earlier definition of a code can be told apart.
- `quote`: optional: the passage coded, as `claimset lab quote` gives it, `{exact, prefix, suffix, start, end}` (a W3C Web Annotation text-quote selector with the passage's character positions in the source's text). The tool finds the words in the source's text when the coding is written, refuses words it does not find or finds more than once, and `verify` checks the quote is still there. Without `quote`, the codes apply to the whole source.
- `unit`: optional: what was coded (`thread`, `post`, `issue`, `answer`, `paragraph`), so counts can say what they count.
- `pass`: optional: the coding pass or round (`1`, `2`, `"second coder"`). Two coders coding the same sources write their own codings; the coder is the event's `by`. Agreement between them is computed by a run from the codings.
- `memo`: optional: the coder's reasoning.
- `withdrawn`: `true` in a correction that removes the codes from the passage (`codes` is then empty or absent).
- `none_apply`: `true` when the coder read the source and no code applies (`claimset lab code SR6 --none`; no `codes`, no `quote`). Such a source stays in the sample: counts have their denominator and the exported evidence lists it, though no count includes it.

A coding is corrected with `--supersedes` when a coder changes their mind or withdraws it; the record keeps every earlier coding. A coder who works in their own agent log (lab record §3.5) writes codings there until the merge; the log keeps them apart from others' codings, but does not hide the main log from them, so blind second coding is the coders' arrangement, not the tool's.

### 2.3 Codings read by runs (on a run)

Counts come from code (lab record §2, principle 3). When the record has codings, the tool writes the codings in force, as the run's writer sees them (the latest version of each, without withdrawn or withheld ones), to `codings.json` in the run's directory before the command starts, names it in `$CLAIMSET_CODINGS`, and records it on the run as `{"ext": {"qualitative": {"codings": {path, hash, count}}}}`: the file, its hash and how many codings it holds. The file holds `{codes, sources, codings}`: each code with its term, definition, criteria and broader terms; each source coded with its `uri`, `title`, `venue`, `published_at` and `retrieved_at`; each coding with its `source`, `codes` (root IDs), `code_versions`, `coder`, `at`, `unit`, `pass`, `memo` and the `quote`'s `exact`, `start` and `end`.

### 2.4 Counts (on a plan)

A plan whose outcome is a count of coded material gives the counts in its `qualitative` block, and its decision rule binds each count's result by name:

```json
"decision": {
  "framework": "qualitative",
  "parameters": {"sample": "every thread a search of the MuseScore forum for 'arrange' returns, 2010-2026"},
  "quantities": {"years": {"variable": "count: years"}, "threads": {"variable": "count: threads"}},
  "supported": "years >= 5 and threads >= 5",
  "not_supported": "threads < 2"
},
"ext": {"qualitative": {"counts": {
  "years": {"codes": ["TM4"], "unit": "source", "distinct_by": "year"},
  "threads": {"codes": ["TM4"], "unit": "source"}}}}
```

`claimset lab plan … --quantity years='{"count": {…}}'` writes both parts. The counter (`claimset/extensions/qualitative_count.py`, standard library only) is run as the plan's deciding run by `claimset lab count --plan PL1`: it reads `codings.json` and writes, for each count, a result whose `variable` is `count: <name>`, with `n`, the number of qualifying units. `claimset lab count --codes TM4 --distinct-by year` is an exploratory count with the same counter.

A count `{codes, match, unit, distinct_by, where}`:
- `codes`: the codes counted (term IDs). A coding counts for a code when one of its codes is that code or a narrower one (whose `broader` terms reach it).
- `match`: `any` (default): a unit qualifies with any of the codes; `all`: with every one of them.
- `unit`: `source` (default): a source qualifies when its codings, together, have the codes; `coding`: each coding is a unit.
- `distinct_by`: what is counted among the qualifying units: `source` or `coding` (the units themselves, the default), `author`, `year` (of `published_at`) or `venue`, each counted once. Units without that property are not counted; the counter reports how many, as a separate result.
- `where`: keeps only codings with the given `pass`, `unit` or `coder`, or of sources with the given `venue` or published from or to a date (`published_from`, `published_to`, as `YYYY` or `YYYY-MM-DD`, inclusive).

Independence beyond these keys (one thread quoting another, one author under two names) is the plan's to define in its `design` and `exclusions`.

### 2.5 Flags

The tool reports the flags of §1.2 for a decided plan, as it reports the core's (lab record §5.7.5). `code_changed_after_plan` does not count runs of the counter: their code is the tool's, whose version the run records in its `parameters`.

### 2.6 Rules

qualitative.L1. A coding names a captured source and, unless it is withdrawn or `none_apply`, at least one code, each once: a term with a code part (§2.1); its `code_versions` name versions of those codes. (Rule L27 in records begun before lab record v0.20.)
qualitative.L2. A coding's `quote` is in its source's text at its recorded position. (Part of rule L25 before v0.20.)
qualitative.L3. A plan's counts are well formed (§2.4) and name only codes; each count's name is a quantity of the plan's decision rule with `variable` `count: <name>`. (Part of rule L15 before v0.20, where a count was a quantity `{count, run}`.)
qualitative.L4. A `qualitative` block on a term holds only `code`, on a term of kind `concept`, with the criteria of §1.1; on a plan, only `counts`; on a run, only `codings`, written by the tool.

### 2.7 Export

| Lab record | Claimset |
|---|---|
| a code (§2.1) | its definition, with the `qualitative` block `{code}` |
| a plan's counts (§2.4) | `ext.qualitative.counts` on the claims of its verdicts, the codes named by their definition IDs; the test's quantities bind `count: <name>` |
| the flags of §1.2 | `ext.qualitative.test_flags` on the claims of the plan's verdicts |
| runs that read codings | evidence of profile `qual` with `analytic_approach: content_analysis`, `data_sources: [document]`, `n_participants_or_cases` (the sources whose codings they read) and `artefacts` (those sources' artefacts); the claim's support is `empirical_qual`. Each quoted coding of its sources is a result whose `value` is the excerpt `{text, locator: {artefact, start, end, lab_event}}` and whose `variable` is the code's definition; a limitation is added when one coder coded every source in one pass (no agreement assessed); counter runs get an `execution` note naming the counter and its version |

The claimset declares `"qualitative": "0.1"` when any of these is exported.
