---
name: claimset-revise
description: Answer the reviews of a published Claimset paper with a revised version and a response, settling each review comment with evidence. Use when a claimset has reviews or comments to answer, or when asked to revise, respond to reviewers, rebut or dispute a comment, or raise a claimset's scores.
---

# Revising a published claimset

A published claimset is reviewed by the platform's reviewer, and later by others. A review never changes the claimset; it sits beside it, public, with the claimset's versions, the author's responses and any human resolutions (`references/revision-v0.2.md`). Each **comment** names the claims it affects, its category and severity, and a **resolution test**: what evidence would settle it, written before you answer. The reviewer scores every claim's credibility from 0 to 100, and every point below 100 rests on comments, so a score rises only when comments are settled. It also assesses each claim's novelty: the closest earlier result it found, and any comment saying the label conflicts with earlier work.

Answering is optional: you may submit a revised version with a **response** to every unresolved comment, or leave the version as it is. Comments not answered stay visible.

**The tool.** Use `claimset` if it is installed (`claimset --help` works); check that this skill matches it (`claimset skill check`) and, if not, update the skills with `claimset skill install`. Otherwise use the copy bundled with this skill, which needs only Python 3.10+: `python3 <this skill>/scripts/claimset`. Below, `claimset` stands for whichever you use.

## Rules you never break

1. **Answer with evidence, not argument.** A rebuttal cites evidence: an evidence object or reference in the revision, or lab events appended for the revision. `argument` explains evidence; it never replaces it.
2. **Never answer `changed` without a change that meets the comment**, and list exactly the paths that do.
3. **Apply the comment's test as written.** Never edit its rules, and never challenge a test because its result would go against you.
4. **Report a test that went against you.** If the evidence meets the comment's `upheld` rule, the comment stands: change the claim, state the limitation, or decline, and say so.
5. **Never rewrite history.** Build the revision from the current version; keep object IDs; never reuse a removed ID; keep appending to the lab record, never restart or rewrite it.
6. **The human decides what is submitted.** Only a guarantor submits a revision; you prepare it and tell them what it says.

## Steps

### 1. Read the record

Get the claimset's **record folder**: its folder in the platform's registry, with `v1/claimset.json`, `v1/reviews/RV1.json`, `v2/…` (a clone, or the files the human gives you). Then:

```
claimset revision status <record>
```

It shows the current version, the latest scores with the comments each rests on, and every comment with its state, its current test and its claims. Comments marked `*` are unresolved (`open`, `upheld`, `limited` or `disputed`): a revision answers every one of them. Read each comment, its requested action and its test before deciding anything.

### 2. Decide an answer to each comment

One of five:

| Answer | When | What it needs |
|---|---|---|
| `changed` | the claimset now meets the comment | `changes`: the field paths of the revision that do (`C1.statement`, `E1.results`, `C13`) |
| `rebutted` | the comment is wrong, and evidence shows it | `evidence`; best, the result of the comment's own test |
| `accepted_as_limitation` | the comment is right; the claim keeps its form and states the problem | `limitation`: a new limitation affecting the comment's claims, or a narrower scope in `changes` |
| `test_challenged` | the test is unfair, cannot be run or does not decide the comment | `alternative_test` (complete: procedure, framework, rules) and `reason` |
| `declined` | no action | optional `reason`; a declined comment keeps lowering the claims it names |

**Settle tests by running them.**
- A computable test (it has quantities) is run as a planned test in the lab record, so its verdict is computed against the reviewer's rule (claimset-lab skill, *Answering a review*): `claimset lab plan --kind confirmatory --hypotheses H9 --comment <comment ID> --review <record>/v1/reviews/RV1.json --analysis "…"`, then `lab run`, `lab verdict` and `lab status`. If a re-review replaced the test, give that review too: the latest test applies.
- A documentary test is decided by what a source states: cite the source as a reference, with a locator.
- A comment that a change meets needs no test: change the claimset.
- A `novelty` comment says the claim's label conflicts with earlier work. If the earlier work states the claim, relabel it (`replication` if your work confirms it independently, or keep only what is new as an `extension`) and relate it to that work: answer `changed`. If it does not, say how the claim differs, cite the work with a locator, and add the relation with a `note` on the difference: answer `rebutted`, citing the reference. Novelty comments do not lower credibility scores; they put the label in question until settled.

A `disputed` comment met neither rule of its test; the re-review gave a replacement test. Run that one. After two disputed rounds you may ask for a human resolver (`request_human: true` on a `rebutted` or `test_challenged` answer).

### 3. Build the revised claimset

Start from the current version's file, `<record>/v<n>/claimset.json`, never from an older one or a fresh lab export:
- keep `paper.id`; remove what the platform adds: `paper.version`, `previous_version`, `changes`, `date`, `status`, `status_notice`, `answers`, the classification's `subfield` and `topics`, and each claim's `verifiability` and `verifiability_flags`;
- make the changes your answers need. A refined object keeps its ID; a different assertion gets a new ID, the next free number, and is placed where it belongs in the file (claims are in reading order, spec §9); an ID removed in any version is never used again; never run `claimset renumber` on a revision;
- put the evidence a rebuttal rests on into the claimset: a new evidence object with the plan's `lab_runs` and results, an assumption's `check` and `check_evidence`, a claim of the planned test (its `test.addresses` names the comment);
- keep the statement rules and everything else of the claimset-submit skill; validate with `claimset validate revised.json --revision`.

If the claimset has a lab record, the revision brings it with the new events appended (`claimset/<study>/lab/lab.jsonl`).

### 4. Write the response

```
claimset revision respond <record> --revision revised.json --lab claimset/<study>/lab/lab.jsonl -o response.json
```

The draft has one entry per unresolved comment: `rebutted` where a planned test addressing the comment met its `withdrawn` rule, `changed` where the revision changes what the comment is about, `declined` otherwise. It prints what you still have to decide. Every entry is yours to confirm or change. Every change no comment asked for goes in `other_changes`, with a self-contained reason.

### 5. Check

```
claimset revision check <record> --revision revised.json --response response.json --lab claimset/<study>/lab/lab.jsonl
```

Fix every error. Read every warning: a rebuttal that no planned test backs, a declined comment that could reverse a claim, a comment disputed twice with no human asked. `claimset revision patch …` shows the changes the platform will record, each with the comments it answers.

### 6. Hand over

Tell the human, comment by comment: what you answered, on what evidence, and what changed; which comments you declined and why; which tests went against the claimset. Give them the three files: the revised claimset, the response and the lab record. A guarantor submits them; the platform publishes the revision as the next version, and its reviewer re-reviews and re-scores it.

## Reviews and resolutions you receive

`claimset revision check-review <file> --record <record>` checks a review or resolution against the record: its target, comments, tests, verdicts and scores (rules R1–R7, R18). If one breaks a rule (a score below 100 that cites no comment, a comment without a resolution test), tell the human; the platform should not have published it.
