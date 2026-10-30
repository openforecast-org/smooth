# Claimset Revision — Specification v0.2 (draft)

Status: working draft · Date: 2026-10-22 · Supersedes v0.1 (`spec/claimset-revision-v0.1.md`) · Companion to the claimset specification (`spec/claimset-v0.34.md`) and the lab record specification (`spec/lab-record-v0.15.md`)

This specification covers what happens to a claimset after its first publication: how the platform's reviewers comment on it, how the author responds, how a revised version is submitted, how disagreements are settled, and how the platform's reviewer judges each claim's credibility and novelty. It specifies the documents the author receives and sends, and the rules the platform checks. The portal itself (accounts, notifications, the website) is out of scope, and so is the scoring rubric, which the portal publishes (§7).

## Changes since v0.1

Scores are given by the platform's reviewer, not computed by a formula: the reviewer is the one that analyses the claimset in detail. What keeps a score open to challenge is that every point below the top rests on comments the author can answer. Novelty is judged apart from credibility: a true claim is not less credible because it was known, but a claim labelled as found here when earlier work states it is a flagged problem.

- Principle 7 and §7 rewritten: the platform's reviewer scores every claim, from 0 to 100, against the portal's public scoring rubric, in the initial review and every re-review. A score below 100 cites the unresolved comments it rests on. A version has no score until its first review. The formula of v0.1 (base values, weights, dependency caps) is removed; the rubric and the paper score are the portal's.
- Reviews gain `scores`, and an agent reviewer that scores names its `rubric` (§3.1). New rule R18; R2 extended (§8).
- §9: `claimset revision score` removed; `status` shows the latest scores.
- §9: the reviewer's instructions are the public `claimset-review` skill, named in every review.
- Novelty as its own dimension: new comment category `novelty`, without severity, which does not lower credibility scores (§3.3); reviews give a novelty assessment of every claim, with the closest earlier result the reviewer found and the search it made (§7.3). New rule R19; R3 and R18 revised.

## 1. Principles

1. **Publication comes first.** A valid submission is published as version 1. Review improves a claimset; it never decides whether it is published. The only bar to publication is validity (claimset §10).
2. **Nothing is rewritten.** Every version, review, response and resolution is an immutable document, kept and public. A later version never edits an earlier one; it supersedes it as the current version, and the earlier one stays in the history.
3. **Everything is public and logged.** All documents are published on the portal and committed to a public git repository (§2.4). Review prompts, reviewer models and the evidence on both sides are public.
4. **Revision is optional.** The author may answer a review with a new version, to settle the comments and raise the score, or may leave the version as it is. Comments that are not answered stay visible.
5. **Disputes are settled by evidence.** Every comment that can be settled says, before the author answers, what evidence would settle it (its resolution test, §3.4). Argument alone neither keeps a comment alive nor resolves it. When no computable test exists, a human resolver decides.
6. **Flag everything at once.** The first review of a claimset covers the whole of it. Later rounds review what changed. Comments that come up later, from the platform or from humans, are additional reviews of the current version (§3.6).
7. **Judgements rest on comments.** The platform's reviewer judges two things of each claim: its credibility, scored against a public rubric, and its novelty, assessed against the earlier work it finds (§7). Every point below the top and every doubt about novelty rests on comments the author can answer with evidence. A judgement is never a bare opinion that only persuasion could change.

## 2. The record of a claimset

### 2.1 Documents

| Document | Prefix | Written by | Attached to |
|---|---|---|---|
| Claimset version | `v1`, `v2`, … | author agent | the claimset |
| Review | `RV` | platform reviewer agents, or humans | the version it reviews |
| Response | `RP` | author agent | the version that answers it |
| Resolution | `RL` | a human resolver | the version it settles a comment on |
| Lab record | — | author agent (and reviewer agents, in their own records) | each version, as an artefact of kind `lab_record` |

Reviews, responses and resolutions are separate documents, not parts of the claimset file. A claimset holds what its contributors assert (claimset principle 7); what others say about it lives beside it. This also keeps the claimset a distillation (claimset principle 13): a revision does not grow by the length of the discussion that led to it.

The prefixes differ from those of lab-record events (`RN` runs, `RS` results, `PL` plans, …), so a response can cite lab events by their IDs without ambiguity (§4.2).

### 2.2 Identifiers

The claimset's DOI is its **concept identifier**; it resolves to the current version. Versions are cited as `<doi>@v<n>` (claimset §2). Documents of the record extend the version they belong to:

- review: `doi:10.0000/claimset.2026.000789@v1/RV1`
- comment: `doi:10.0000/claimset.2026.000789@v1/RV1.c4`
- response: `doi:10.0000/claimset.2026.000789@v2/RP1`
- resolution: `doi:10.0000/claimset.2026.000789@v2/RL1`

Review IDs are numbered across the whole claimset (`RV1` on v1, `RV2` on v2, `RV3` an additional review on v2, …), so a comment's local ID (`RV1.c4`) is unique within the claimset. Responses and resolutions are numbered the same way (`RP1`, `RP2`, …; `RL1`, `RL2`, …). Wherever a comment is named outside its review (in a response, a verdict, a lab plan's `addresses`, a claim's `test.addresses`), it is named by its global ID.

**Stable object IDs.** Claims, evidence and other objects keep their IDs from version to version. An object that keeps its ID is the same object, refined: reworded, rescoped, with corrected numbers or new support. An assertion that changes into a different assertion gets a new ID, and the old one is removed. An ID that has been removed is never used again in the claimset, so an unversioned global ID (`<doi>#C4`) never resolves to a different object than the one it was written for; if the object was removed, it resolves to the last version that holds it, marked as removed.

**Field paths.** Responses and patches name parts of a claimset by field path: an object's ID, optionally followed by field names separated by dots (`C1`, `C1.statement`, `C1.scope.conditions`, `E4.results`), or `paper.` followed by field names (`paper.title`). A path covers itself and every path that extends it (`C1.scope` covers `C1.scope.conditions`).

### 2.3 The chain

The documents form a chain by hash. Hashes are SHA-256 of the canonical JSON of the document, as in lab record §4.1, written `sha256:<64 hex digits>`.

```
v1 ──hash──▶ RV1 ──hash──▶ RP1 ──▶ v2 ──hash──▶ RV2 ──▶ …
                 additional reviews of v1 (RV…) join the chain the same way
```

- A review names the hash of the version it reviews (the published file).
- A response names the hash of every review it answers.
- A published revision names its response (`paper.answers`, §5.3), whose hash the record keeps.
- A revision's lab record begins with the previous version's lab record unchanged (§5.4).

Anyone can check the chain from the published files alone: no document can be altered after a later one has been built on it.

### 2.4 Layout in git

The platform commits every document to a public registry repository when it is published, in a layout that mirrors the identifiers:

```
<concept id>/
  v1/claimset.json  v1/lab.jsonl  v1/validation.json
  v1/reviews/RV1.json
  v2/claimset.json  v2/lab.jsonl  v2/validation.json  v2/response-RP1.json
  v2/reviews/RV2.json  v2/reviews/RV3.json
  v2/resolutions/RL1.json
```

`<concept id>` is the DOI with `doi:` removed and `/` replaced by `_` (`10.0000_claimset.2026.000789`). Only the platform commits to the registry. Each version is tagged (`<concept id>-v2`). The branch is protected against force-pushes and deletion, so the repository's history is itself an append-only log of the record. The author's own repository, where the research is done, is not part of the record; the lab record published with each version is.

An author works on a revision from a copy of the claimset's folder of the registry, the **record folder** (a clone, or the files the platform sends). The reference tools read the record folder (§9).

## 3. Review

### 3.1 Review document

```json
{
  "schema": "claimset-review",
  "schema_version": "0.1",
  "id": "RV1",
  "kind": "initial",
  "target": {"version": "doi:10.0000/claimset.2026.000789@v1", "hash": "sha256:…"},
  "reviewer": {
    "type": "agent", "name": "Claimset reviewer",
    "models": [{"provider": "…", "name": "…", "version": "…", "identifier": "…"}],
    "prompt": {"version": "review-0.3", "uri": "https://…/prompts/review-0.3.md", "hash": "sha256:…"},
    "system": {"name": "claimset-review", "version": "0.3"},
    "rubric": {"version": "rubric-0.1", "uri": "https://…/rubric-0.1.md", "hash": "sha256:…"}
  },
  "date": "2026-10-23",
  "comments": [],
  "verdicts": [],
  "scores": [],
  "novelty": [],
  "lab_record": null
}
```

- `kind`: `initial` (the platform's review of version 1) · `re_review` (the platform's review of a revision, §3.5) · `additional` (any later review, §3.6).
- `target`: the version reviewed and the hash of its published file.
- `reviewer`: for an agent, its models, the review prompt (version, public URI and hash), the reviewing system and, when the review scores, the scoring rubric (version, public URI and hash, §7). For a human: `{type: human, name, orcid, competing_interests}`, where `competing_interests` is a sentence, `"none"` if none.
- `comments`: the review's comments (§3.2).
- `verdicts`: in a re-review, the verdict on each earlier comment that was still unresolved (§6.2); in other reviews, empty.
- `scores`: the claims' credibility scores (§7.2): in an initial review and a re-review, one for every claim of the version; in an additional review by the platform, optional, for the claims its comments name; absent from human reviews.
- `novelty`: the novelty assessment of the claims (§7.3), given by the platform's reviewer as `scores` are, for every claim that is not an assumption.
- `lab_record`: optional; `{uri, hash}` of the reviewer's own lab record, when the reviewer ran code (to reproduce a result, or to run a resolution test). Its runs can be cited as evidence by the review.

The validator's warnings on a version enter its initial review or re-review as comments of category `validation`, written by the platform, so that the author answers them the same way as the rest.

### 3.2 Comment

```json
{
  "id": "c6",
  "claims": ["C13", "C1"],
  "targets": ["C13"],
  "category": "assumption",
  "severity": "could_reverse",
  "comment": "C13 states that per_lag_max() reproduces the check of smooth 4.6.0, but no recorded run calls smooth 4.6.0; the refutation in C1 rests on it.",
  "requested_action": "Run the check of smooth 4.6.0 from CRAN on the models of C1 and compare its decisions with per_lag_max().",
  "resolution_test": {},
  "evidence": [],
  "references": []
}
```

- `id`: `c1`, `c2`, … unique within the review.
- `claims`: required, at least one: the claims the comment affects. A comment on another object names the claims that object bears on (an evidence object: the claims it supports; a limitation: the claims it affects; a definition: the claims that use it). A comment on the paper as a whole (missing prior work, say) names the principal claims it affects. This is what lets a score rest on comments (§7).
- `targets`: optional: the objects or fields the comment is about, as field paths (`E4`, `C4.statement`, `L2`, §2.2), when more precise than `claims`.
- `category`, `severity`: §3.3.
- `comment`: self-contained; states the problem, not the fix, and follows the statement rules of claimset §8 where they apply (no rhetoric, no instructions to a model).
- `requested_action`: what the author could do: a change, a run, a citation.
- `resolution_test`: §3.4.
- `evidence`: optional: evidence the reviewer offers, as claimset evidence objects (claimset §3.4), for example a reproduction run from the reviewer's lab record.
- `references`: optional: works the comment cites, as claimset references (`{target, id_type, locator}`, claimset §3.13), for example the prior work a novelty comment points to.

### 3.3 Categories and severity

**Category** (what kind of problem):

| Category | For |
|---|---|
| `prior_work` | missing or misattributed earlier work that a claim builds on or uses |
| `novelty` | a novelty label that earlier work contradicts: an earlier result states the claim (`no_prior_found` on a known claim); the claim replicates or extends an earlier result without saying so; or it repeats a claim of the same authors' earlier claimset without relating to it. No severity (below) |
| `scope` | a claim broader or narrower than its evidence; a definition the tests do not match |
| `assumption` | an assumption that is untested or wrongly checked; an assumption or dependency a claim relies on but does not list ("the claim lacks an assumption", always `could_reverse`, §7.2) |
| `test` | how a test was planned and run: data reused from exploration, exclusions, decision rules |
| `measurement` | what was measured and how: reference methods, numerical accuracy, instruments |
| `missing_result` | a number or condition a claim relies on that no result records |
| `reproducibility` | artefacts, environments, links that may not last |
| `reasoning` | a proof step, a warrant, an inference |
| `presentation` | wording, structure, format, where nothing in the content is in question |
| `validation` | a validator warning (written by the platform) |

**Severity** reuses the impact scale of claimset limitations (claimset §3.9), with one level added:

`could_reverse` · `could_bias_magnitude` · `reduces_precision` · `reduces_transferability` · `presentation`

A severity says what happens to the affected claims if the comment is right, not how much work the fix takes. It measures credibility, so `novelty` comments have none: a true claim stays true when it turns out to be known. A `novelty` comment cites the earlier work in `references` and is settled like any other comment, usually by a `documentary` test ("upheld if <work> states the claim under <conditions>; withdrawn if it does not"); the author answers it by relabelling the claim and relating it to that work, or by showing the difference.

### 3.4 Resolution test

Every comment says, when it is written, what evidence would settle it. The test has the structure of a lab-record decision rule (lab record §5.7.1), with the two outcomes named for the comment:

```json
"resolution_test": {
  "procedure": "Load smooth 4.6.0 from CRAN; for the 3042 ETS(A,N,A) and 200 GUM models of C1 (confirm.py --test perlag --seed 2026), compare the decision of smooth 4.6.0's check with the decision of per_lag_max().",
  "framework": "deterministic",
  "parameters": {},
  "quantities": {"diff": {"quantity": "models whose admissibility decision differs, smooth 4.6.0 against per_lag_max()"}},
  "upheld": "diff > 0",
  "withdrawn": "diff == 0"
}
```

- `procedure`: what to run or check, on what. Concrete enough that the author, the reviewer or a third party would run the same thing.
- `framework`, `parameters`, `quantities`: as in lab record §5.7.1, plus the framework `documentary`: the test is decided by what a cited source states (for prior-work and novelty comments: "upheld if the cited work derives the admissible region of ETS(A,N,A) from the discount polynomial; withdrawn if it does not"). A documentary test has no quantities.
- `upheld`: the rule under which the comment stands. `withdrawn`: the rule under which it falls. Expressions over quantities when the outcome can be computed (lab record §5.7.3); text otherwise. If neither rule is met, the test did not settle the comment (§6.3).

A resolution test is required for every comment except those of category `presentation` and `validation`, which are settled by the change they ask for. A comment whose test cannot be computed (most `prior_work` and `reasoning` comments) still states its test in text, so that whoever settles it, the reviewer in the next round or a human resolver, applies a rule written before the author answered.

### 3.5 Re-review

When a revision is published, the platform reviews it (`kind: re_review`). A re-review:

1. gives a verdict on every comment that was unresolved on the previous version, from every review of it, including additional reviews (§6.2);
2. may add new comments only on what the revision changed (an object added, or an object or field that differs from the previous version), or of severity `could_reverse`. Everything else should have been raised in an earlier review; a re-review that reopens unchanged content would make every revision a new first review.

The platform pins the reviewer's models and prompt for each round and records them. When the platform changes its reviewer or its rubric, existing reviews and scores keep the setup that made them.

### 3.6 Additional reviews

Anything that comes up after a version's initial review or re-review is an additional review of the current version (`kind: additional`): a human reviewer's comments, a platform reviewer that found something late, a comment prompted by a new paper that contradicts a claim. Additional reviews:

- take the same form as other reviews (§3.1–3.4), with a resolution test on every comment that needs one;
- target the current version only;
- are not limited to changed content;
- require nothing of the author. Their comments can lower the current version's scores (§7) and, if the author submits a revision, are answered in it like any other unresolved comment.

## 4. Response

### 4.1 Response document

A revision answers every comment that is unresolved on the version it revises: from its initial review or re-review, and from any additional reviews.

```json
{
  "schema": "claimset-response",
  "schema_version": "0.1",
  "id": "RP1",
  "revises": {"version": "doi:10.0000/claimset.2026.000789@v1", "hash": "sha256:…"},
  "responds_to": [
    {"review": "doi:10.0000/claimset.2026.000789@v1/RV1", "hash": "sha256:…"}
  ],
  "entries": [],
  "other_changes": [
    {"path": "A9.uri", "reason": "Lab record link pinned to a commit instead of a branch."}
  ]
}
```

- `revises`: the version the revision is built from, and the hash of its published file.
- `responds_to`: every review the response answers, with its hash.
- `entries`: one per unresolved comment (§4.2).
- `other_changes`: changes the author makes that no comment asked for, each `{path, reason}` with a field path (§2.2). A revision may carry corrections this way.

### 4.2 Entries

`{comment, response, changes, evidence, limitation, alternative_test, argument, reason, request_human}`

- `comment`: the comment's global ID.
- `response`: one of the five below.

| Response | Means | Must include |
|---|---|---|
| `changed` | the claimset was changed to meet the comment | `changes`: the field paths in the revision that answer it (`C1.statement`, `C1.scope.conditions`, `C13`) |
| `rebutted` | the comment is wrong, and evidence shows it | `evidence`: at least one evidence object or reference of the revision, or a lab-record event appended since the previous version (§5.4). Ideally the result of the comment's resolution test |
| `accepted_as_limitation` | the comment is right; the claim keeps its form but states the problem | `limitation`: a new limitation of the revision that affects the comment's claims; or, in `changes`, a narrowed scope |
| `test_challenged` | the resolution test is unfair, cannot be run, or does not decide the comment | `alternative_test`: a complete resolution test (§3.4), and `reason` |
| `declined` | no action | nothing; `reason` optional |

- `changes`: field paths, also allowed with `rebutted` (a rebuttal often adds the evidence to the claimset: a new evidence object, an assumption's `check` moved to `tested`).
- `evidence`: IDs of evidence objects or references of the revision (`E20`, `B7`), or of events of the revision's lab record (`RN47`, `RS112`).
- `argument`: optional text with any response. It explains; it never replaces the evidence a rebuttal needs.
- `request_human`: on a `rebutted` or `test_challenged` entry for a comment that is `disputed`: the author asks for a human resolver (§6.4).

A comment on a claim that the revision removes is answered `changed`, with the removed claim's ID in `changes`.

**The resolution test as a planned test.** When the comment's test is computable, the author can run it as a confirmatory test in the lab record: a hypothesis stating what the comment questions (that per_lag_max() reproduces smooth 4.6.0), and a plan whose decision rule is the comment's test, `supported` being its `withdrawn` rule and `not_supported` its `upheld` rule, with the plan's `addresses` naming the comment (lab record §5.7, §9). The verdict is then computed by the lab tool against a rule the author did not write, at a time the anchor records, and the claim exported from it carries the comment in `test.addresses` (claimset §3.3.3). Evidence from exploratory runs is allowed, but weaker (warning W-R2).

## 5. Revision

### 5.1 What is submitted

A revision submission consists of:

1. the revised claimset: complete and self-contained, a valid submission (claimset §10.1–10.2), with `paper.id` naming the claimset's concept identifier and no `correction_reason`;
2. the response (§4);
3. the lab record, if the claimset has one: the previous version's record with any new events appended.

The revision is built from the current version. A revision of an older version is refused: there is one line of versions.

Only a guarantor of the current version, identified by ORCID, can submit a revision. A revision may add or remove guarantors; the platform obtains every new guarantor's consent, as at first publication.

### 5.2 Publication

The platform validates the revision (§8 and claimset §10), checks the chain (§2.3), publishes the claimset as the next version, makes it the current version, and moves the previous one to the history. Publication does not wait for the re-review, which follows it (§3.5). The status of a revision is `active`.

### 5.3 Fields the platform adds

In addition to those of claimset §2.2:

| Field | Value |
|---|---|
| `paper.version`, `paper.previous_version` | as for corrections (claimset §9) |
| `paper.changes` | the JSON Patch from the previous version, as for corrections; each operation carries `reason` and, when it answers comments, `addresses`: the global IDs of those comments, taken from the response |
| `paper.answers` | the global ID of the response the version carries |

A reader of the version can then tell, for every change, which comment it answers, without reading the response.

The patch is computed object by object: objects of a collection are matched by ID, and within an object, fields by name. An object present in both versions whose fields differ gives `replace`, `add` or `remove` operations on those fields (a list inside an object is replaced whole); an object only in the previous version gives `remove`, one only in the revision gives `add`. The operations apply in order to the previous version and give the revision. An operation's reason and `addresses` come from the response entries and `other_changes` whose field paths cover it (§2.2).

### 5.4 Lab record

A revision's lab record starts with the previous version's record, every event unchanged: the same lines, so the same hash chain. New work done for the revision (resolution tests, new runs, new plans) is appended after it. The platform checks the prefix, and evidence that a response cites from the lab record must come from the appended part. The record therefore shows, in one chain, what was done before the review and what was done in answer to it.

An event withheld in the previous version's record (lab record §8) stays withheld in the same way: the published line is part of the prefix.

A version that had no lab record may add one in a revision. A revision may not drop a lab record its previous version had.

## 6. The life of a comment

### 6.1 States

| State | Means | Set by |
|---|---|---|
| `open` | not addressed, declined, or addressed without meeting the comment | the review that raised it (every comment starts `open`), or a re-review |
| `resolved` | the change made meets the comment | a re-review |
| `withdrawn` | evidence met the test's `withdrawn` rule: the comment was wrong | a re-review or a resolution |
| `upheld` | evidence met the test's `upheld` rule, and the claim was not changed | a re-review or a resolution |
| `limited` | the claim now states the problem as a limitation or a narrower scope | a re-review |
| `disputed` | the evidence met neither rule, or the author challenged the test and the challenge was not accepted | a re-review |

A comment's state is the state set by its latest verdict, in the order the documents were published; a comment without a verdict is `open`. A comment is **unresolved** while it is `open`, `upheld`, `limited` or `disputed`, and the author may answer it in a later revision. A `limited` comment can still be resolved by a later change that removes the problem.

### 6.2 Verdicts

A verdict, in a re-review or a resolution, is `{comment, state, reason, evidence, rule, replacement_test, accepted_test}`:

- `comment`: the comment's global ID.
- `state`: the comment's new state; `reason`: self-contained.
- `rule`: for `withdrawn`, `upheld` and `disputed` reached through the test: `{met, quantities}`, which rule the evidence met (`withdrawn`, `upheld` or `neither`) and, for computed tests, the value bound to each quantity.
- `evidence`: what the verdict rests on: the author's evidence, the reviewer's own run, or both.
- `accepted_test`: when the author challenged the test and the reviewer accepts the alternative, which then replaces the comment's test from that round on.
- `replacement_test`: required for `disputed` (§6.3).

A reviewer weighs a rebuttal by applying the test, not by deciding whether it finds the argument convincing. A rebuttal that meets the `withdrawn` rule withdraws the comment; one that meets neither rule leaves it `disputed`, whatever its argument says.

### 6.3 Disputes

The same evidence cannot settle a question both ways. A comment that stays `disputed` therefore means that its test did not decide it: the test was too vague, measured the wrong thing, or could be met by both sides. The verdict that sets `disputed` carries a **replacement test** that differs from the one it replaces and says what it fixes. The next round applies that test. Either side may run it, the author in the lab record, the reviewer in its own. A comment's **current test** is its latest replacement or accepted test, or its original test.

A comment disputed in two consecutive re-reviews, or a challenged test the reviewer does not accept, can go to a human resolver at the author's request (`request_human`), or at the platform's.

### 6.4 Human resolution

Where no computable test can decide (novelty, interpretation, what a source states), or a dispute persists, a human resolver settles the comment. The resolver is not an author, guarantor, supervisor or contributor of the claimset, nor the reviewer that raised the comment. The resolution is a document attached to the current version:

```json
{"schema": "claimset-resolution", "schema_version": "0.1", "id": "RL1",
 "target": {"version": "doi:10.0000/claimset.2026.000789@v2", "hash": "sha256:…"},
 "resolver": {"name": "…", "orcid": "…", "competing_interests": "none"},
 "date": "2026-11-02", "verdicts": []}
```

Its verdicts take the same form as a re-review's (§6.2), with state `withdrawn` or `upheld`, and say which rule of the test the evidence met. A resolution is final for the comment as it stands; a later revision that changes the claim may still resolve it.

## 7. Credibility and novelty

### 7.1 Who judges

The platform's reviewer judges each claim's credibility and its novelty, the two dimensions a review can establish; others (how widely a claim holds, how much others build on it) are shown by the claimset or the platform, not judged. For credibility, it scores the claims. It is the one that reads the claimset in detail and judges whether each claim is correct, supported by its evidence and placed against the literature, and its comments say why. It scores every claim of a version from 0 to 100, in the initial review of version 1 and in the re-review of every revision, against the portal's **scoring rubric**: a public, versioned document saying what a claim at each level looks like and how much each kind of problem costs. The review names the rubric it applied (§3.1). The paper's score is aggregated by the portal from the claim scores, by the rule the rubric states.

A version has no score until its first review: a score given by the file alone would rank unreviewed papers above reviewed ones.

### 7.2 Credibility: what a score rests on

```json
"scores": [
  {"claim": "C1", "score": 72,
   "comments": ["doi:10.0000/claimset.2026.000789@v1/RV1.c4", "doi:10.0000/claimset.2026.000789@v1/RV1.c6"],
   "reason": "The refutation holds on the models tested (c4 biases the count, not its sign), but rests on C13, which no run has checked against smooth 4.6.0 (c6)."},
  {"claim": "C13", "score": 55, "comments": ["doi:10.0000/claimset.2026.000789@v1/RV1.c6"],
   "reason": "Unchecked: no recorded run calls smooth 4.6.0 (c6)."}
]
```

- `claim`: a claim of the version; `score`: an integer from 0 to 100.
- `comments`: the global IDs of the comments the score rests on. A score below 100 cites at least one; a score of 100 cites none.
- `reason`: required below 100: how the cited comments bear on the claim, in a self-contained sentence.

**No deduction without a comment.** Every point below 100 rests on comments, other than `novelty` comments, that are unresolved once the review is written (its own comments, and earlier comments after its verdicts) and that name the claim or a claim it rests on (`depends_on` and `assumptions`, including the assumptions of the evidence it cites). A claim is only as credible as what it rests on: a comment on an assumption can lower the claims built on it. One problem lowers a claim once: the rubric counts a comment once per claim, whether the claim is named directly or through what it rests on. The reviewer that sees a weakness raises a comment for it, with its resolution test, so the author always has something to answer, and a score can be raised only by settling comments.

### 7.3 Novelty

```json
"novelty": [
  {"claim": "C1", "closest": [{"target": "doi:10.1016/j.ijforecast.2008.00000", "id_type": "doi", "locator": "Theorem 2",
                               "note": "Derives the admissible region of ETS(A,N,A) with one seasonal lag, not with several."}],
   "searched": {"sources": ["claimset", "openalex", "references of the claimset"], "date": "2026-10-23"},
   "comments": []},
  {"claim": "C4", "closest": [{"target": "doi:10.0000/claimset.2026.000412#C2", "id_type": "claimset"}],
   "searched": {"sources": ["claimset", "openalex"], "date": "2026-10-23"},
   "comments": ["doi:10.0000/claimset.2026.000789@v1/RV1.c9"]}
]
```

- `claim`: a claim of the version that is not an assumption.
- `closest`: the earlier results closest to the claim that the reviewer found, as references (claimset §3.13), each with a `note` on how the claim differs; empty if it found none.
- `searched`: where the reviewer looked (`sources`: the platform's claims, literature databases, the claimset's own references and those citing them) and when (`date`). Novelty is relative to a search, so the search is recorded.
- `comments`: the unresolved `novelty` comments on the claim. With none, the reviewer's search agrees with the claim's label; with some, the label is in question until they are settled.

The portal shows each claim's label together with this assessment: `no_prior_found` that the reviewer's search confirms, or a label in question, with the earlier work that questions it.

### 7.4 When judgements change

A re-review scores and assesses every claim of the revision again, after its verdicts: a comment resolved or withdrawn no longer lowers a score or questions a label. An additional review by the platform may score and assess again the claims its comments name; human reviews and resolutions do not score, but a resolution's verdicts are taken into account by the next scoring. A claim's current score and assessment are the latest ones given for the current version; those of earlier versions stay in their reviews.

### 7.5 Recording

Scores and novelty assessments are part of the review that gives them, so they are public, committed with it, and tied to the reviewer's models, prompt and rubric. The portal shows the current version's scores and the paper's score, and the history of every version.

## 8. Validation rules

Errors (the document or submission is refused) and warnings (accepted, recorded). Rule numbers are local to this specification.

### 8.1 Reviews

- R1. A review has `schema: claimset-review`, an `id` `RV<n>`, and a `target` naming a published version and its hash. `kind: initial` targets version 1 and is written once per claimset by the platform; `kind: re_review` targets a revision and is written once per revision by the platform; `kind: additional` targets the current version.
- R2. An agent reviewer lists its models and its prompt (`version`, `uri`, `hash`), and its rubric when it scores (R18); a human reviewer has `name`, `orcid` and `competing_interests`.
- R3. Every comment has an `id` unique in the review, at least one entry in `claims`, each resolving to a claim of the target version, and `category`, `comment` and `requested_action` from §3.2–3.3, and a `severity` unless its category is `novelty`; a `novelty` comment has no severity and cites the earlier work in `references`; `targets`, if present, resolve to objects of the target version.
- R4. Every comment not of category `presentation` or `validation` has a `resolution_test` with `procedure`, `framework`, the parameters its framework requires (lab record §5.7.1), and non-empty `upheld` and `withdrawn` rules; rules over `quantities` are valid expressions (lab record §5.7.3); a `documentary` test has text rules.
- R5. A re-review has a verdict for every comment unresolved on the previous version, and verdicts only for those.
- R6. A new comment in a re-review names a claim or target that the revision added or changed, or has severity `could_reverse`.
- R7. A verdict of `disputed` has a `replacement_test` (as R4) that differs from the test it replaces; `withdrawn` and `upheld` name the `rule` met and cite `evidence`; `resolved` and `limited` say in `reason` what in the revision meets the comment. A resolution's verdicts are `withdrawn` or `upheld`.
- R18. An initial review or a re-review has a score for every claim of the target version; an additional review scores only claims its comments name; a human review or a resolution has no scores. Every score is an integer from 0 to 100 for a claim of the target version, at most one per claim. A score below 100 has a `reason` and cites at least one comment; every cited comment is unresolved after the review, is not a `novelty` comment, and names the claim or a claim it rests on. A score of 100 cites none. An agent review with scores names its rubric (`version`, `uri`, `hash`).

- R19. An initial review or a re-review has a novelty assessment for every claim of the target version that is not an assumption; an additional review assesses only claims its `novelty` comments name; a human review or a resolution has none. Each assessment is for such a claim, at most one per claim, with `searched` (a non-empty list of `sources` and an ISO `date`), `closest` as a list of references with `target` and `id_type`, and `comments` citing only unresolved `novelty` comments that name the claim.

### 8.2 Responses

- R8. A response has `schema: claimset-response`, an `id` `RP<n>`, `revises` naming the version it revises and its hash, exactly one entry for every comment unresolved on that version, and names every review those comments come from, with its hash.
- R9. `changed` ⇒ `changes` is non-empty, and every path in it exists in the revision (or names a removed object) and differs from the previous version.
- R10. `rebutted` ⇒ `evidence` is non-empty, and each entry resolves to an evidence object or reference of the revision, or to a lab-record event appended since the previous version.
- R11. `accepted_as_limitation` ⇒ `limitation` names a limitation new in the revision that affects at least one of the comment's claims, or `changes` names a scope field of one of them.
- R12. `test_challenged` ⇒ `alternative_test` satisfies R4, and `reason` is present.
- R13. `request_human` only on `rebutted` or `test_challenged` entries whose comment is `disputed`.

### 8.3 Revisions

- R14. A revision names the claimset's concept identifier in `paper.id`, has no `correction_reason`, revises the current version, comes with a response, and is submitted by a guarantor of the current version. (Claimset rule 25 allows `paper.id` in a correction or a revision.)
- R15. Every operation of the JSON Patch from the previous version falls under a field path listed in a response entry's `changes` or in `other_changes`.
- R16. No ID removed in an earlier version is used again.
- R17. If the previous version had a lab record, the revision has one, it begins with the previous record unchanged (every line), and it verifies (lab record §6).

### 8.4 Warnings

- W-R1. A review was written by an agent on the same model as an author agent of the claimset.
- W-R2. A `rebutted` entry for a comment with a computable resolution test cites only exploratory runs, or evidence that no confirmatory plan addressing the comment produced: running the test under a confirmatory plan (§4.2) makes the rebuttal stronger.
- W-R3. A comment of severity `could_reverse` is `declined`.
- W-R4. A comment has been `disputed` in two consecutive re-reviews and no human resolution has been requested.

## 9. Tools (informative)

The reference implementation is part of the `claimset` package. Its commands read a record folder (§2.4):

| Command | Effect |
|---|---|
| `claimset revision status RECORD` | the current version, its latest scores and novelty assessments, every comment with its state, current test and claims, and what a revision must answer |
| `claimset revision check-review FILE --record RECORD` | checks a review or resolution against the version it targets (R1–R7, R18, R19, W-R1) |
| `claimset revision respond RECORD --revision FILE [--lab FILE] [-o FILE]` | writes a draft response: an entry for every unresolved comment, `changed` with the paths of the revision that the comment's targets cover, `rebutted` with the evidence of lab plans that address the comment and whose verdict met its `withdrawn` rule, and `declined` for the rest, to be completed by the author |
| `claimset revision check RECORD --revision FILE --response FILE [--lab FILE]` | checks a revision submission (R8–R17, W-R2–W-R4) and prints the patch the platform will record |
| `claimset revision patch RECORD --revision FILE --response FILE` | the JSON Patch from the current version to the revision, with `reason` and `addresses` (§5.3) |

The platform's reviewer follows the public `claimset-review` skill of the same repository; each review names the version of the skill it followed, its public URI and its hash in `reviewer.prompt` (§3.1), so anyone can see, and repeat, how a review was made. Authors answer reviews with the `claimset-revise` skill.

The lab tool runs a comment's resolution test as a planned test: `claimset lab plan --comment <global ID> --review FILE --hypotheses H9 …` takes the decision rule from the comment's current test and records the comment in the plan's `addresses` (lab record §5.7).

## 10. Open questions

1. **The scoring rubric.** What each score level means, how much each category and severity of problem costs, and how the paper's score aggregates the claims' scores; to be designed with the portal, and calibrated on real claimsets and their reviews, for example in the evaluation study.
2. **Status of a revision that carries corrections.** Whether a revision whose `other_changes` correct errors sets `status: corrected`, or stays `active` with the corrections visible in `changes`.
3. **Human resolvers.** Who appoints them, how their competing interests are checked, and what happens when none is available.
4. **Additional reviews in volume.** Whether additional reviews need limits (per reviewer, per version) to prevent flooding a claimset with comments.
5. **Scores of cited claims.** Whether a cited claimset claim's score should bear on the citing claim's score, and be updated when the cited claim's score changes.
6. **Reviewer upgrades.** When the platform's reviewer changes, whether current versions are re-reviewed by the new setup, and how that review is labelled.
7. **Consistency of scores.** How the portal keeps scores comparable across papers and reviewer versions (a pinned reviewer and rubric per round, repeated scoring, reporting the spread).
8. **Searching earlier claims from the lab.** When the platform offers a search of its claims, the lab tool could search for a hypothesis when it is recorded, and again at export, and record what it found, so the authors learn early whether the work is a replication or an extension, and the record shows when they looked. Queries are unpublished ideas: the platform would neither keep nor show them.

## Appendix A: example

From the claimset *Admissibility of lagged single-source-of-error state space models* (version 1). Two comments of its initial review, the author's response, and the re-review's verdicts.

**RV1, two comments:**

```json
[
  {"id": "c4", "claims": ["C1"], "targets": ["C1.statement", "E1"],
   "category": "test", "severity": "could_bias_magnitude",
   "comment": "The count of 1482 ETS(A,N,A) models without decaying weights leaves out 78 grid models with alpha + gamma = 2, whose largest mode modulus is 1 and which also lack decaying weights under D3; the test reports no count of the models excluded by the band of 1e-8.",
   "requested_action": "Report the excluded models, and state the count either with them or with the band in the statement.",
   "resolution_test": {"procedure": "Count the ETS(A,N,A) grid models of C1 whose PBH reference lies within 1e-8 of 1.",
                       "framework": "deterministic", "parameters": {},
                       "quantities": {"edge": {"quantity": "ETS(A,N,A) grid models within 1e-8 of the unit circle"}},
                       "upheld": "edge > 0", "withdrawn": "edge == 0"}},
  {"id": "c6", "claims": ["C13", "C1"], "targets": ["C13"],
   "category": "assumption", "severity": "could_reverse",
   "comment": "C13 states that per_lag_max() reproduces the check of smooth 4.6.0, but no recorded run calls smooth 4.6.0; the refutation in C1 rests on it.",
   "requested_action": "Run the check of smooth 4.6.0 from CRAN on the models of C1 and compare its decisions with per_lag_max().",
   "resolution_test": {"procedure": "Load smooth 4.6.0 from CRAN; on the 3042 ETS(A,N,A) and 200 GUM models of C1, compare its check's decisions with those of per_lag_max().",
                       "framework": "deterministic", "parameters": {},
                       "quantities": {"diff": {"quantity": "models whose admissibility decision differs, smooth 4.6.0 against per_lag_max()"}},
                       "upheld": "diff > 0", "withdrawn": "diff == 0"}}
]
```

**RP1, the two entries** (the author ran c6's test as plan PL7, with `addresses` naming the comment; its run RN47 found no difference):

```json
[
  {"comment": "doi:10.0000/claimset.2026.000789@v1/RV1.c4", "response": "changed",
   "changes": ["C1.statement", "E1.results"],
   "argument": "C1 now counts 1560 models: 1482 above the band and 78 on the unit circle, reported as a result of E1."},
  {"comment": "doi:10.0000/claimset.2026.000789@v1/RV1.c6", "response": "rebutted",
   "evidence": ["E20", "RN47"], "changes": ["C13.check", "C13.check_evidence"],
   "argument": "Planned test PL7 ran the comment's resolution test: 0 of 3242 decisions differ. C13 is now tested, with E20 as its evidence."}
]
```

**RV2 (re-review of v2), the two verdicts:**

```json
[
  {"comment": "doi:10.0000/claimset.2026.000789@v1/RV1.c4", "state": "resolved",
   "reason": "C1 states 1560 models and E1 reports the 78 models on the unit circle as R7."},
  {"comment": "doi:10.0000/claimset.2026.000789@v1/RV1.c6", "state": "withdrawn",
   "rule": {"met": "withdrawn", "quantities": {"diff": 0}},
   "evidence": ["doi:10.0000/claimset.2026.000789@v2#E20"],
   "reason": "The resolution test was run under plan PL7: no decision differs, so per_lag_max() reproduces smooth 4.6.0 on the models of C1."}
]
```

The effect on the scores: in v1, RV1 scored C13 55 and C1 72, both citing c6 (C1 rests on C13). In v2, C13 is `tested` and c6 is `withdrawn`, so RV2 can cite neither c6 nor c4 against C1 or C13: with no other unresolved comment on them or on what they rest on, both score 100.
