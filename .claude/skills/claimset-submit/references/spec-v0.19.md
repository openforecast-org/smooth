# Claimset — Specification v0.19 (draft)

Status: working draft · Date: 2026-10-22 · Supersedes v0.18 (`spec/claimset-v0.18.md`) · Companion: lab record specification (`spec/lab-record-v0.6.md`)

Claimset is the name of both the open format and the platform.

## Changes since v0.18

A repository can hold several studies and several papers, which may build on each other before they are published.

- New §2.4: papers live in `claimset/<study>/<paper>.claimset.json`, with their reading view beside them; a version submitted or published is marked by a git tag, not by renaming the file.
- `paper.study`: the name of the study a paper comes from (§3.1).
- `exploration.covers`: a paper that reports some of its lab's questions says which, gives the whole lab's counts and names the other papers (§3.1.1).
- New identifier type `claimset_local`: a claim or paper of the same repository that is not yet published, replaced by its global ID at publication (§4, §3.7, §3.13).
- New rules 55, 56 and 57 (§10).

## Changes in v0.18 (since v0.17)

Claims are not all equal: a reader needs the contributions and what they rest on in full, and the rest briefly. Prior work a paper uses as data, software or an implementation is linked like other prior work.

- New claim role `secondary`: a finding of the paper that no contribution rests on (a side result, a negative or null result reported in its own right). `supporting` now means only a claim that a contribution rests on, directly or through other claims (§3.3).
- §3.3.4: prior work enters in four ways, the fourth being data, software or an implementation the evidence uses (an artefact pointing to the reference); what the paper cites as an example or as background is left out. A claim the paper makes in several places (abstract, introduction, body, conclusions) is one claim; a conclusion broader than what the body establishes is its own claim. In converted papers, secondary claims are kept only when they report results.
- Artefacts gain `reference`: the cited work the artefact is taken from (§3.6, §3.13).
- Reading view: claims are grouped by role, principal first; non-principal claims whose only support is argument or none are grouped last, under CLAIMS WITHOUT EVIDENCE (§2.3, Appendix D).
- New rule 54; rules 4 and 51 revised; new warnings 15 and 16 (§10).

## Changes in v0.17 (since v0.16)

A claimset is stored as JSON for tools and read by models as plain text. The file of a converted paper carries only what its source says, not the converter's judgements.

- New §2.3: the reading view, a plain-text rendering of a valid claimset for reading models, made mechanically by `claimset render`. It keeps everything a reader needs to understand, use or challenge the claims, and leaves out what serves tools, audit and publication. Reference rendering in Appendix D.
- Strength assessments: `by: converter` is removed. In converted papers, assessments record only a strength the source itself states (`by: author`); judgements of a conversion and of its claims are platform assessments (§5, §6). New rule 53.

## Changes in v0.16 (since v0.15)

A claimset contains only the paper's own claims. Prior work is linked, not restated: as cited support of the claims that use it, as relations that position the paper's claims, and as borrowed definitions. Cited works are listed once.

- Claims are the paper's own: `role: background` and `novelty: restatement` are removed; rule 4 revised (§3.3, §3.3.4).
- New top-level `references`: each cited work once, with an ID (`B1`, …), cited by ID from support, relations and definitions (§3.13, §4).
- Relations gain type `addresses` (the claim answers a problem, gap or limitation stated in the target), an optional one-sentence `note`, and may point to a reference (§3.7).
- §3.3.4 rewritten around the three ways prior work enters a claimset.
- New rules 51 and 52; rules 4, 10, 34 and 41 revised (§10).

## Changes in v0.15 (since v0.14)

What belongs in a claimset is decided by two tests that need no knowledge beyond the paper: attribution (whose is it, as the paper's own citations say) and function (does the paper's argument use it). Converted papers can cite works whose reference list gives no identifier, and record the source's evident errors they correct.

- Principle 13 corrected: a claimset is shorter than the body of the paper (its text without the reference list); measured conversions do not support "much shorter" for dense papers (§1).
- New §3.3.4: what belongs in a claimset. Definitions explain in full only the terms the paper introduces or uses in its own sense; borrowed terms are named with their source and what this paper specifies. General statements the paper does not attribute are kept only if its argument uses them (§3.2, §3.3.4).
- New identifier type `reference` for cited support and definition sources: the reference as printed in the paper's reference list, for works the list gives no identifier for (§4).
- `converted_from.corrections`: evident errors in the source that the conversion corrected, each with what was printed and what was recorded (§3.1).
- New rules 49 and 50; new warning 14 (§10).

## Changes in v0.14 (since v0.13)

A claimset is a distillation of a paper's argument, not a re-encoding of its data and derivations. Converting a results-heavy, derivation-heavy paper under v0.13 produced a claimset about four times the paper's length; this version makes the compact form the normal one.

- New principle 13: a claimset is a distillation (§1).
- Evidence gains result tables (`tables`): a table of numbers with shared conditions, row labels and named columns, instead of one `results` entry per number; rows can be cited (§3.4, §4).
- Evidence gains `source_spans`. In converted papers, a derivation is referenced in the source rather than restated: formal evidence may give `source_spans` instead of `steps` (§3.4).
- Granularity rule extended: a comparison the paper makes across several measures or datasets is one claim stating where the method is better and where it is not (§8).
- New rule 48; rules 7 and 38 and warning 8 revised (§10).

## Changes in v0.13 (since v0.12)

Converted papers may come from sources under non-commercial licences, and keep their restrictions.

- Rule 31 revised: a source may be converted when its licence permits derivatives, including non-commercial (NC) licences; licences that forbid derivatives (ND) or reserve all rights still exclude conversion. The claimset's licence keeps every condition of the source licence: NonCommercial stays NonCommercial, ShareAlike stays ShareAlike (§3.1, §10.3).

## Changes in v0.12 (since v0.11)

Planned tests carry the decision rule they were planned with, so readers can check the verdict against a rule fixed before the data were seen.

- New claim field `test`: the inferential framework, its parameters (significance level, test, sidedness, correction for multiple tests, sample size and stopping rule, equivalence bounds, prior, thresholds), the rules for `supported` and `not_supported`, and whether the verdict was computed from recorded results or judged (§3.3.3).
- `not_supported` now means refuted under a rule stated in advance; a test that met neither rule is `inconclusive` (§3.3.3).
- A hypothesis tested more than once gives one claim per test, so no test goes unreported (§3.3.3).
- `exploration` gains `hypotheses.open` (hypotheses never closed) and `tests` (planned tests and their verdicts); its counts are lower bounds of the exploration (§3.1.1).
- New rule 47; rules 45 and 46 revised for the new counts (§10).

## Changes in v0.11 (since v0.10)

Links claimsets to the lab record, the running record of the research from which a claimset is distilled.

- Claims record the hypothesis they came from and its outcome (`supported`, `not_supported`, `inconclusive`), so negative results are ordinary claims in the same claimset (§3.3.3).
- Claims record whether they were planned: `planning`, `planned_at` and the strength of the timestamp, `planned_at_anchor` (§3.3.3).
- Paper-level `exploration` summary: how many hypotheses were tested, supported, refuted, revised and abandoned, and how many runs were made (§3.1).
- New artefact kind `lab_record`, with `withheld_events`; evidence can name the lab runs it came from (`lab_runs`) (§3.4, §3.6).
- New verifiability flags `lab_record_withheld`, `lab_events_withheld` and `unanchored_plan` (§5).
- New rules 42–46 and warning 13. Earlier rules are unchanged.

## Changes in v0.10 (since v0.9)

Prompted by the second test claimset.

- Checks are recorded, not just their result: new claim field `checks` lists each independent check with its outcome (`confirmed`, `disputed`, `inconclusive`) and extent, so a failed or partial check is visible instead of collapsing into `unchecked`. Checks apply to any claim: re-deriving a formal claim or reproducing an empirical one (§3.3, §5).
- New verification level `disputed`. `verified_by` is replaced by `checks` (§5).
- Independence of checks: a warning when a checking agent runs on the same model as an author agent; the platform weights checks by independence (§5, §10).
- Numbers quoted in a statement are checked against the structured results the claim cites (§3.3, §10).
- New rules 39–41 and warnings 11–12; rule 36 revised. Rules 1–35 and 37–38 are unchanged.

## Changes in v0.9 (since v0.8)

Prompted by the first test claimset (a methodology paper on the observed-data likelihood of ADAM and iETS).

- Refutations and replications must link to what they refute or replicate; relations gain `id_type`, `locator`, `archive_uri` and `self`, and may target any identified work, not only claimsets (§3.7).
- Scope describes its own claim, not the section the claim came from (§3.3).
- Guidance for choosing the claim type; measured results of computations, simulations and benchmarks are `descriptive` (§3.3).
- Formal claims carry `verification` and no strength assessments. New verification level `agent_checked`, with `verified_by` (§3.3, §5).
- Evidence gains a structured `results` list and an `execution` record of the code actually run; support entries may point at specific results (§3.4, §4).
- Formal evidence drops `theorem`: the formal claim's statement is the theorem (§3.4).
- `author` covers every agent that produced the paper's content, including turning the team's own draft into a claimset; `converter` is only for third-party sources. New artefact kind `manuscript` (§3.6, §3.11).
- Artefact `licence` recommended (§3.6).
- New rules 33–38 and warnings 5–10 (§10). Rule numbers 1–32 are unchanged.

## Changes in v0.8 (since v0.7)

- Position `director` renamed `supervisor`, and narrowed: the supervisor oversees the agents' work and approves the final version. Who set the research question is recorded by the CRediT role `conceptualization` (§3.11).
- Autonomy value `human_directed` renamed `human_supervised`; autonomy values defined (§3.1).

## Changes in v0.7 (since v0.6)

- Two forms of the file: the **submission** (written by the agent) and the **published file** (completed by the platform) (§2.2).
- Platform-assigned fields: `paper.id` (the DOI), `version`, `previous_version`, `changes`, `date`, `status` and `status_notice`. Submissions omit them; the platform mints the DOI at publication, after validation and guarantor consent (§2.2, §3.1).
- Corrections: the submission names the existing claimset in `paper.id` and gives a `correction_reason`; the platform computes the JSON Patch diff and assigns the new version (§9).
- Classification: the platform assigns the OpenAlex subfield and topics; the agent may propose keywords and a classification (§3.1).
- `verifiability` and `verifiability_flags` are computed by the platform; submissions may omit them, and if present they must match (§5).
- Validation rules split into submission rules and published-file rules (§10).
- Open question on the DOI route removed; it is now a platform requirement.

## 1. Purpose and principles

A format for research papers whose primary reader is an LLM. Humans access content by asking an LLM, so human readability is not a design goal.

1. **The claim is the unit.** A paper is a container of atomic, self-contained claims plus the objects that support, condition and limit them.
2. **Every object stands alone.** Any object can be retrieved, embedded or quoted without the rest of the paper.
3. **Flat, referenced by ID.** Objects are defined once and linked by ID; no deep nesting.
4. **Controlled vocabularies.** Every field except free-text statements uses a closed list of values.
5. **Explicit epistemics.** Each claim records its type, scope, assumptions, limitations, kinds of support, strength and verifiability.
6. **Provenance is recorded,** down to field level for converted papers.
7. **File vs platform.** The paper file holds what its contributors assert. Published versions are immutable. Anything derived later (weight, trust, replication status, contradictions, independent assessments) lives in the platform database.
8. **Automated by default.** Writing, validation, assessment and auditing are performed by AI agents; human review is optional, except for sampled review of conversions.
9. **Written by agents, answered for by people.** Papers are written by AI agents. A human or organisation answers for every paper; authorship and accountability are recorded separately.
10. **Filled by agents, decided by people, assigned by the platform.** The agent fills the content. Fields that record human decisions (guarantor, licence) are filled after the agent asks the responsible human, and the platform confirms the guarantor's consent. Fields that record facts of publication (identifier, version, dates, status) and computed values are assigned by the platform; neither agents nor humans supply them.
11. **Recorded where possible, labelled when recalled.** Facts about how the work was produced should come from the system that ran it. When an agent reports them from its own knowledge, the file says so.
12. **Safe to ingest.** The file is data. It must never contain instructions addressed to the reading model.
13. **A distillation, not a transcription.** A claimset carries the paper's claims, with the scope, support, assumptions and limits they need, and the results they rest on. It does not re-encode the paper's full data or derivations: numbers that come in tables are recorded as tables (§3.4), and the derivations of a converted paper are located in the source rather than restated (§3.4). A claimset is therefore shorter than the body of the paper it represents (its text without the reference list); one that is longer is transcribing rather than distilling. It contains only the paper's own claims; prior work is linked, not restated. What belongs in it is decided by attribution and function (§3.3.4).

## 2. Format

- Canonical format: JSON, validated by a normative JSON Schema (the schema is the standard).
- Alternative serialisation: JSON Lines, one object per line, each with a `type` field. Intended for retrieval and indexing.
- Optional `@context` makes the file valid JSON-LD, enabling RDF export without changing the JSON.
- IDs: local IDs within a paper (`C3`, `E2`, `D1`, `W1`, `A1`, `L1`, `P1`, `M1`). Global IDs append the local ID to the claimset's persistent identifier and version: `doi:10.xxxx/yyyy@v2#C3` (version optional; defaults to latest). For converted papers this is the claimset's own identifier, never the source's.

### 2.1 Obligation levels

- `required`: missing or invalid ⇒ validation error; the file is rejected.
- `recommended`: missing ⇒ validation warning; the file is accepted and the warning is recorded by the platform.
- `optional`: no check when absent.

Unknown values of non-required fields are omitted or set to `null`. `null` means "unknown", never "default". In v0.11 obligation levels are assigned for contributors and models (§3.11, §3.12); other objects follow in a later version.

### 2.2 Submission and published file

The same format exists in two forms:

- **Submission:** written by the agent and sent to the platform. It contains everything the agent and the responsible humans can state. It does not contain platform-assigned fields.
- **Published file:** the submission after the platform has validated it, obtained guarantor consent, minted the DOI and filled the platform-assigned fields. Published files are immutable.

**Platform-assigned fields** (absent from submissions, required in published files unless noted):

| Field | Assigned when |
|---|---|
| `paper.id` | At publication: the platform mints a DOI. Exception: a correction submission names the existing claimset's `paper.id` (§9) |
| `paper.version` | At publication: 1 for a new paper; incremented for corrections |
| `paper.previous_version` | At publication of a correction |
| `paper.changes` | At publication of a correction: JSON Patch computed by the platform |
| `paper.date` | At publication: the publication date |
| `paper.status` | At publication (`active`), and later by governance actions |
| `paper.status_notice` | When status changes from `active` |
| `paper.classification.subfield`, `paper.classification.topics` | At validation: assigned by the platform's classifier; agent proposals are considered |
| `claims[].verifiability`, `claims[].verifiability_flags` | At validation, and recomputed when availability changes |

Before publication, a submission is identified by an internal submission ID issued by the platform. It is not part of the file and is never cited.

Top-level layout:

```json
{
  "schema_version": "0.17",
  "@context": "optional",
  "paper": {},
  "contributors": [],
  "models": [],
  "definitions": [],
  "claims": [],
  "evidence": [],
  "warrants": [],
  "limitations": [],
  "artefacts": [],
  "references": []
}
```

### 2.3 Reading view

A claimset is stored and exchanged as JSON: the canonical file, which tools validate, the platform publishes and corrections patch. A model that reads the paper is given its **reading view**: a plain-text rendering of the same claimset. JSON field names, quotation marks and the fields that serve tools, audit and publication take a large share of a file and carry nothing a reader needs; the reading view leaves them out.

- **Kept:** the title, licence, source of a converted paper, contributions and exploration counts; definitions; claims with their type, role, novelty (when not `new`), support with its pointers and locators, assumptions, `depends_on`, limitations, warrant, strength, verification, verifiability, planning, outcome, scope, relations with their notes, and test; evidence with its description, design fields, estimate, results, tables and proof steps; warrants; limitations; artefact descriptions; references.
- **Left out:** `provenance`, `field_provenance`, `source_spans`, `corrections`, availability, artefact locations, versions and licences, contributors, models, execution records, lab-record identifiers, `schema_version` and the platform's publication fields.

- **Order:** claims are grouped by role: principal claims, then supporting claims, then secondary claims, each group in file order. Claims that are not principal and whose support is only `argument` or `unstated` follow, under CLAIMS WITHOUT EVIDENCE, so a reader sees at once what the paper asserts or argues without evidence. Principal claims stay with the principal claims whatever their support.

The reading view is made mechanically from a valid file (`claimset render`); it adds nothing and is never edited. Its layout is the reference rendering in Appendix D. A reader that needs a field left out (a source span to check a claim against its source, an artefact's location) reads the canonical file.

### 2.4 Files in a repository

A paper written alongside its research lives in the research's repository, in the folder of its study (lab record specification §3):

```
claimset/
  <study>/
    lab/                         the study's lab record
    code/                        code only this study uses (optional)
    <paper>.claimset.json        the paper
    <paper>.claimset.txt         its reading view (claimset render)
```

- A study and a paper are named with lower-case letters, digits and hyphens, starting with a letter (`eigen-checker`, `eigen-checker-cpp`). A study can give several papers; each reports whole questions of the study's lab (lab record §5.17). Papers from different studies are independent unless their claims link to each other.
- A paper may cite or relate to a paper of the same repository that is not yet published, with `id_type: claimset_local` and the target `<study>/<paper>.claimset.json#C3` (a claim) or `<study>/<paper>.claimset.json` (the paper), relative to `claimset/` (§4). The validator checks that the file and the claim exist when the file lies in a `claimset/` folder. At publication the platform replaces each local reference with the global ID of the published claim, so the cited paper is published first, or both are submitted together.
- The file keeps its name through revisions: the date and version of a submitted or published version are recorded by the platform (§9) and in git. A version is marked with the git tag `claimset/<study>/<paper>/v<n>` (`claimset tag`), made on the commit that holds it.
- The `claimset/` folder is left out of package builds (lab record §3.3).

## 3. Object types

### 3.1 Paper

| Field | Type | Notes |
|---|---|---|
| `id` | string | Platform-assigned. Persistent identifier of this claimset: a DOI minted by the platform. For converted papers, distinct from the source's identifier |
| `title` | string | Required |
| `version` | integer | Platform-assigned. 1, 2, …; each published version is immutable |
| `previous_version` | string | Platform-assigned. Global ID of the previous version; absent for version 1 |
| `changes` | list | Platform-assigned for corrections: JSON Patch (RFC 6902) operations against the previous version, each with the `reason` taken from the correction submission |
| `correction_reason` | string | Submission only, required for corrections: what was corrected and why, self-contained |
| `status` | enum | Platform-assigned. `active` · `corrected` · `retracted` · `withdrawn` · `expression_of_concern` |
| `status_notice` | object | Platform-assigned unless `active`: `{date, reason_code, statement, issued_by}` |
| `date` | ISO date | Platform-assigned. Publication date |
| `licence` | string | Required. SPDX identifier, e.g. `CC-BY-4.0`. Native papers: chosen by the supervisor or guarantor. Converted papers: see *Licence of a converted paper* below and rule 31 |
| `classification` | object | `subfield` (one OpenAlex subfield ID) and `topics` (1–3 OpenAlex topic IDs): platform-assigned; the agent may propose values. `keywords` (free text, for specifics the taxonomy does not reach): supplied by the agent, recommended |
| `contributions` | list of claim IDs | Required. Ordered by priority; principal claims only |
| `autonomy` | enum | Native papers only, required: `autonomous` (agents ran the work end to end) · `human_supervised` (humans oversaw the agents' work and approved it) · `human_assisted` (humans carried out substantial parts of the research) · `unknown` |
| `contribution_record` | enum | Required. `complete` · `partial` · `unknown`. For converted papers, refers to the conversion |
| `origin` | enum | Required. `native` · `converted` |
| `exploration` | object | Required when any claim has `hypothesis`: summary of the research process (§3.1.1) |
| `study` | string | The name of the study the paper comes from, when it is written in a repository (§2.4) |
| `converted_from` | object | Required if `converted` (below) |

**`converted_from`**

| Field | Type | Notes |
|---|---|---|
| `source_id` | string | Identifier of the source paper, e.g. a DOI or arXiv ID |
| `id_type` | enum | As for cited support (§4) |
| `source_version` | string | The exact version converted, e.g. `v2` |
| `source_licence` | string | SPDX identifier of the source's licence |
| `source_authors` | list | `{name, orcid}` |
| `source_status_at_conversion` | enum | Status of the source when converted, same values as `status` |
| `converter` | contributor ID | The converting agent |
| `date` | ISO date | Date of conversion |
| `corrections` | list | Evident errors in the source corrected in the conversion (below) |

**Corrections.** A conversion records what the source says, including its errors, except an evident error (a typographical error in a formula or a number that the source itself contradicts elsewhere), which it corrects visibly: the corrected field's provenance is `inferred` (`field_provenance`), and the correction is listed in `corrections` as `{field, where, printed, recorded, reason}`:
- `field`: the corrected location in the claimset, as an object ID and a field path, e.g. `C28.statement`.
- `where`: the source span of the error.
- `printed`: what the source says. `recorded`: what the claimset says instead. `reason`: why the source is evidently in error, e.g. `"Akaike's information criterion is 2k − 2ℓ(θ); the sign of ℓ is inverted in Equation (1)"`.
Disagreements with the source that are not evident errors are not corrections: the claim records what the source says, and the platform's review of the conversion may question it (§6).

**Licence of a converted paper.** A claimset converted from a source is an adaptation of it, so the source's licence decides whether conversion is allowed and what licence the claimset may carry:

| Source licence | Conversion | Claimset licence |
|---|---|---|
| `CC0-1.0`, `CC-BY-2.0`, `CC-BY-3.0`, `CC-BY-4.0` | allowed | the source licence, `CC0-1.0`, `CC-BY-4.0` or `CC-BY-SA-4.0` |
| `CC-BY-SA-3.0`, `CC-BY-SA-4.0` | allowed | a `CC-BY-SA` licence |
| `CC-BY-NC-3.0`, `CC-BY-NC-4.0` | allowed | the source licence, `CC-BY-NC-4.0` or `CC-BY-NC-SA-4.0` |
| `CC-BY-NC-SA-3.0`, `CC-BY-NC-SA-4.0` | allowed | a `CC-BY-NC-SA` licence |
| Any licence with NoDerivatives (`ND`), all rights reserved, or no licence | not allowed | — |

A claimset under a non-commercial licence may be used only for non-commercial purposes. Platforms and tools that use claimsets commercially (for example to train, tune or promote a commercial service) exclude claimsets whose `paper.licence` contains `NC`.

Contributors are listed in the top-level `contributors` list (§3.11), not in the paper object.

#### 3.1.1 Exploration summary

`exploration` tells readers how much was tried to reach the published claims, so positive results can be weighed against the number of attempts. It is exported from the lab record.

```json
"exploration": {
  "lab_record": "A10",
  "questions": 2,
  "hypotheses": {"total": 10, "supported": 3, "not_supported": 2, "inconclusive": 1, "revised": 2, "abandoned": 1, "open": 1},
  "tests": {"total": 5, "supported": 2, "not_supported": 2, "inconclusive": 1},
  "runs": {"total": 214, "confirmatory": 18, "exploratory": 196, "failed": 11}
}
```

- `hypotheses` counts recorded hypotheses by their final status; `open` counts those never closed. `total` is the sum of the other hypothesis counts.
- `tests` counts planned tests that have run, by verdict: one per plan and hypothesis it names. `total` is the sum of the three verdicts.
- `runs.total` is the sum of `confirmatory` and `exploratory`; `failed` counts runs of either kind whose exit status was not `success`.
- `covers`, when the paper reports only some of its lab's questions: `questions` (their IDs in the lab record), `lab` (the whole lab's counts of `questions`, `hypotheses` and `tests`, in the same form) and `other_papers` (each other paper drawn from the lab, `{name, questions}`). `questions`, `hypotheses` and `tests` then count the paper's questions only; `runs` is always the whole lab's, since a run is not tied to a question. Readers can see what the lab tried that this paper does not report, and where it is reported.

```json
"covers": {"questions": ["Q1"],
           "lab": {"questions": 2, "hypotheses": {"total": 9, "supported": 6, "not_supported": 1, "inconclusive": 1, "revised": 0, "abandoned": 1, "open": 0},
                   "tests": {"total": 7, "supported": 6, "not_supported": 1, "inconclusive": 0}},
           "other_papers": [{"name": "eigen-checker-cpp", "questions": ["Q2"]}]}
```

The counts cover what was recorded. Exploration is recorded where its results informed decisions, not step by step, so the counts are lower bounds of the exploration; the counts of planned tests are complete.

Retracted papers are never deleted. All their claims inherit the retraction status for readers and the platform. A converted claimset whose source is retracted is shown with the source's retraction, whatever its own status.

### 3.2 Definition

| Field | Type | Notes |
|---|---|---|
| `id` | string | `D1` … |
| `term` | string | |
| `definition` | string | One term, one meaning, used rigidly throughout |
| `source` | object or string | Where a borrowed term is defined: a reference ID (`B3`, §3.13), `{target, id_type}` as for cited support (§4), or a global claim or definition ID of another claimset |

A definition explains in full only a term the paper introduces or uses in its own sense. A term the paper takes from elsewhere (it cites where the term comes from) is a **borrowed term**: its `definition` names the standard meaning in one line and states only what this paper specifies (e.g. `"Mean scaled interval score; here of the 95% prediction interval"`), and `source` says where the term is defined. A term the paper uses without defining or citing it, in its usual sense, needs no definition.

### 3.3 Claim

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `C1` … |
| `statement` | string | Self-contained sentence (rules in §8) |
| `type` | enum | `descriptive` · `associational` · `causal` · `predictive` · `interpretive` · `formal` · `methodological` (choosing: §3.3.1) |
| `role` | enum | `principal` (a contribution) · `supporting` (one of the paper's own claims that a contribution rests on, directly or through other claims, by `depends_on` or `assumptions`) · `secondary` (a finding of the paper that no contribution rests on: a side result, a negative or null result reported in its own right) |
| `novelty` | enum | `new` · `replication` · `extension` · `refutation` |
| `scope` | object | `population`, `setting`, `geography`, `language`, `period`, `data_type`, `conditions` (list), `excludes` (list: where it explicitly does not apply). Describes this claim only (§3.3.2) |
| `support` | list | Kinds of support (§4); at least one entry |
| `assumptions` | list of claim IDs | Assumption claims (§3.8) this claim relies on |
| `limitations` | list of limitation IDs | Threats to this claim within its scope (§3.9) |
| `assessments` | list | Strength assessments (§5); not for formal or assumption claims |
| `verifiability` | enum | Platform-assigned (§5); optional in submissions |
| `verifiability_flags` | list | Platform-assigned (§5); optional in submissions |
| `warrant` | warrant ID | Reasoning behind the claim; required for `interpretive` |
| `depends_on` | list of claim IDs | Claims within this paper that this one relies on |
| `relations` | list | Typed links to claims or works outside this paper (§3.7); required for refutations and replications |
| `verification` | enum | Formal claims proved or argued in this paper, required: `machine_checked` · `agent_checked` · `peer_checked` · `disputed` · `unchecked` (§5) |
| `checks` | list | Independent checks of this claim and their outcomes (§5): re-derivations of formal claims, reproductions of empirical ones |
| `hypothesis` | object | `{lab_event, outcome}`: the lab-record hypothesis this claim came from, and its outcome `supported` · `not_supported` · `inconclusive` (§3.3.3) |
| `planning` | enum | `confirmatory` · `confirmatory_with_deviations` · `exploratory` (§3.3.3) |
| `planned_at` | datetime | When the plan that tested this claim was recorded (ISO 8601, UTC) |
| `planned_at_anchor` | enum | Strength of `planned_at`: `platform` · `rfc3161` · `git_remote` · `none` |
| `test` | object | The decision rule fixed before the test (§3.3.3); required for confirmatory claims |
| `source_spans` | list | Locations in the source text; required for converted papers |
| `provenance` | enum | Default for the claim: `author_stated` · `extracted` · `inferred` |
| `field_provenance` | object | Overrides per field path, e.g. `{"scope.excludes": "inferred"}`; any field whose provenance differs from the default must be listed |

#### 3.3.1 Choosing the claim type

| If the claim states… | type |
|---|---|
| what something is, how often, how much, or what was measured in a computation, simulation or benchmark under stated conditions (including comparisons of methods measured there) | `descriptive` |
| a relationship observed in data the work did not control | `associational` |
| that X causes Y in a real-world system, from a randomised or otherwise credible causal design | `causal` |
| what will happen | `predictive` |
| what something means (qualitative interpretation) | `interpretive` |
| a mathematical or logical result | `formal` |
| that a method, procedure or implementation does what is specified | `methodological` |

#### 3.3.2 Scope

Scope describes the conditions under which this claim holds, not the section or study it came from. Two claims from the same section often have different scopes: a result about series where every period is observed does not carry the scope "series with unobserved periods". Omit scope fields that do not apply rather than copying them from neighbouring claims.

#### 3.3.3 Hypotheses, planning and negative results

A claim exported from a lab record carries the hypothesis it came from (`hypothesis.lab_event`) and its outcome. A claim with outcome `not_supported` is a **negative result**: its statement says plainly what was not found ("Under conditions …, method M did not reduce …"), with scope, support and evidence like any other claim. Negative results belong in the same claimset as the positive results they qualify.

`planning` says whether the test was planned before it was run:
- `confirmatory`: tested by a run under a plan, recorded before the run, that named the hypothesis, with no deviation.
- `confirmatory_with_deviations`: as above, but the run deviated from the plan; the deviation is recorded as a limitation.
- `exploratory`: every other case, including results found while exploring.

**Outcomes.** `supported` and `not_supported` are conclusive: each means that the rule stated for it in advance was met. A test that met neither rule is `inconclusive`. In particular, failing to reject a null hypothesis is `inconclusive`, not `not_supported`, unless the plan stated a rule for refutation (for example equivalence bounds).

**One claim per planned test.** A hypothesis tested under several plans (a replication, a larger sample) gives one claim per test, each with its own outcome and `test`, so every planned test that ran is reported. The statement of each is the hypothesis as it stood when its plan was written.

**Decision rule (`test`).** A confirmatory claim records the rule it was planned with, as fixed before the data were seen:

```json
"test": {
  "framework": "frequentist",
  "parameters": {"alpha": 0.05, "test": "one-sided paired t-test on per-series MAE differences", "sides": "greater",
                 "correction": "none", "sample": "500 series, fixed in advance; no interim analysis"},
  "quantities": {"p": {"quantity": "p-value, paired t-test of MAE difference"},
                 "d_upper": {"quantity": "mean MAE difference", "field": "interval.upper"}},
  "supported": "p < alpha",
  "not_supported": "d_upper < 0.01",
  "verdict": "computed",
  "runs": 1,
  "lab_plan": "PL2",
  "registration": null
}
```

- `framework`: `frequentist` · `equivalence` · `bayesian` · `deterministic` · `formal` · `qualitative`, as defined in the lab record specification §5.7.1.
- `parameters`: required per framework: `frequentist`: `alpha`, `test`, `sides`, `correction`, `sample`; `equivalence`: `alpha`, `test`, `lower_bound`, `upper_bound`, `sample`; `bayesian`: `prior`, `threshold`, `sample`.
- `quantities`: when the rules are expressions, the results they are evaluated on, each `{quantity, conditions, field, run}`; they correspond to results in the claim's evidence.
- `supported`, `not_supported`: the two rules: expressions over quantities and numeric parameters (lab record §5.7.3), or text saying what observation decides.
- `verdict`: `computed` (the rules are expressions and the outcome was computed from the recorded results by the lab tool) · `judged` (the rules are text and the outcome was stated by the writer, who said which rule the evidence met).
- `runs`: how many runs decided the test. `lab_plan`: the plan's lab-record ID, when there is one. `registration`: the URI of an external pre-registration, when there is one.
- `flags`: optional list of weaknesses of the test, from the lab record (lab record §5.7.5): `code_changed_after_plan` (the analysis code changed between the plan and the test) · `holdout_used_elsewhere` (data held out for the test was also used in other runs).

`planned_at_anchor` says how far `planned_at` can be trusted (lab record §7): from strongest, `platform` (receipt from the Claimset platform), `rfc3161` (trusted timestamp authority), `git_remote` (pushed commit), to `none`.

#### 3.3.4 What belongs in a claimset

A claimset contains the paper's own claims, with what they need: definitions of the paper's own terms, evidence, warrants, assumptions, scope and limitations. It does not restate what others have found. Two tests, both answerable from the paper itself, decide what is the paper's own and what it needs. Neither asks whether something is new to the field, which would need knowledge of the literature that a converter or reader may not have; the paper's authors have already recorded that knowledge in their citations.

1. **Attribution:** whose is it, as the paper's own citations say? What the paper introduces is its own. What it attributes to others (it cites them) is theirs, and enters the claimset only by reference, in one of the three ways below.
2. **Function:** does the paper's argument use it? A general statement the paper makes without attributing it is kept only if one of the paper's claims rests on it (then it is an assumption, §3.8, or `consensus` support of that claim) or the paper contrasts its contribution with it (then the claim's statement says so). Justifications of routine choices (which accuracy measure, which software) that no finding depends on are left out, or belong in the evidence's `description`.

Prior work enters a claimset in four ways, each linked to the paper's own content, never as a claim of its own. Each is a function a citation has in the paper; a work cited for several functions is linked for each.

| The paper uses prior work … | Recorded as |
|---|---|
| as a premise of one of its claims (a theorem it applies, a result it builds on): the claim fails if the cited work is wrong | `cited` support of that claim, pointing to the reference, with a `locator` (§4) |
| to position a claim: what the claim answers, extends, refines, supports or contradicts | a relation of that claim (§3.7), with a one-sentence `note` saying what the target states |
| as data, software or an implementation that its evidence uses | an artefact of that evidence (§3.6) with `reference` pointing to the cited work |
| as vocabulary | a borrowed definition (§3.2) |

Prior work that the paper cites as an example or an illustration ("used, e.g., in retail"), or as background (general motivation, surveys of the field), is left out: none of its claims uses it. When the tests do not decide, the item is kept: an unnecessary item costs length, a missing one loses content.

**One claim, several places.** Papers state their findings more than once: in the abstract, the introduction, the body and the conclusions. Each assertion is one claim, with source spans for every place the paper makes it, worded as precisely as its most specific statement (usually the body's, with its conditions). The introduction's and abstract's statements show which claims are contributions. A statement in the conclusions (or abstract) that is broader than what the body establishes (more general data, a stronger verb, a recommendation) is not merged into the claim it generalises: that would make the narrower claim stronger, or hide what the paper concludes. It is a claim of its own, with the scope the paper gives it, that depends on the claims it generalises.

**Secondary claims in converted papers.** The paper's side results, including negative and null results, are kept as `secondary` claims. Its side remarks, illustrations and explanations that it neither tests nor uses for a contribution are left out.

### 3.4 Evidence

Common fields: `id`, `profile`, `description`, `artefacts` (list of artefact IDs), `data_origin` (`primary` · `secondary`), `preregistered` (boolean), `availability` (§3.10), `models` (list, below), `results` (list, below), `execution` (object, below), `lab_runs` (list of lab-record run IDs, e.g. `RN4` or `AC2.RN4`, that produced this evidence), `tables` (list, below), `source_spans` (locations of the evidence in the source text, for converted papers).

**Results** (`results`, quant and simulation profiles): structured numbers, so readers can query numbers instead of parsing prose. A single number is a result entry; numbers that form a table (several quantities for several settings) are a result table (`tables`, below). `{id, quantity, value, unit, interval, conditions, artefact}`
- `id`: `R1`, `R2`, … unique within the evidence object; referenced from elsewhere as `<evidence id>.<result id>`, e.g. `E23.R2`.
- `quantity`: what was measured, e.g. `"largest KL divergence per observation, mixture vs exact"`.
- `value`: number. `unit`: optional. `interval`: optional `{level, lower, upper}`.
- `conditions`: object of the settings that define this result, e.g. `{"distribution": "Gamma", "form": "conventional"}`.
- `artefact`: optional artefact ID of the table holding the full results.
A single-estimate study may use `estimand`, `estimate` and `interval` instead.

**Result tables** (`tables`, quant and simulation profiles): numbers that the paper reports as a table, recorded as one. `{id, description, conditions, row_labels, columns, units, rows, artefact, source_spans}`
- `id`: `T1`, `T2`, … unique within the evidence object; referenced from elsewhere as `<evidence id>.<table id>`, e.g. `E2.T1`.
- `description`: what the table reports, self-contained, e.g. `"Mean and median forecast errors of six methods on the M3 monthly series, horizons 1 to 18"`.
- `conditions`: optional object of the settings shared by every cell, e.g. `{"dataset": "M3 monthly", "horizon": "1-18"}`.
- `row_labels`: list of the conditions that distinguish rows, e.g. `["method"]`; may be empty for a single row.
- `columns`: list of the quantities, one per value column, e.g. `["mean RMSSE", "median RMSSE", "mean MASE"]`. Names are unique within the table. `units`: optional list of units, one per column (`null` where none).
- `rows`: list of `{id, label, values}`. `id`: `r1`, `r2`, … unique within the table; a row is referenced as `<evidence id>.<table id>.<row id>`, e.g. `E2.T1.r3`. `label`: the values of `row_labels`, in order, e.g. `["CES"]`. `values`: one number per column, in order; `null` where the paper reports no value.
- `artefact`: optional artefact ID of the source of the full table. `source_spans`: optional locations of the table in the source text (converted papers).

Intervals, ranks and test statistics are columns of their own (e.g. `"mean MASE, 95% interval lower"`). A table records what the paper reports; it is not split into result entries, and result entries are not repeated in a table.

```json
"tables": [{"id": "T1", "description": "Mean MASE and MSIS of three methods on the M3 monthly series, horizons 1 to 18",
  "conditions": {"dataset": "M3 monthly"}, "row_labels": ["method"], "columns": ["mean MASE", "mean MSIS"],
  "rows": [{"id": "r1", "label": ["CES"], "values": [0.861, 6.95]},
           {"id": "r2", "label": ["ETS"], "values": [0.874, 6.71]},
           {"id": "r3", "label": ["ARIMA"], "values": [0.902, null]}]}]
```

**Execution** (`execution`, optional): what was actually run, when it differs from or adds to the artefact versions. `{code_version, software, hardware, notes}`; `code_version` is the commit or version that produced the results.

**Models in the research design** (`models`, optional, any profile): `{model, role, settings, record_source, prompts}`
- `model`: model ID (§3.12).
- `role`: `subject` (the model is what is studied) · `instrument` (the model performs a measurement or analysis step) · `annotator` (the model labels or codes data) · `generator` (the model produces data or stimuli) · `baseline` (the model is a comparison method).
- `settings`: object, e.g. `{effort, temperature, top_p, seed, max_tokens}`; individual values may be `null`.
- `record_source`: recommended (§3.11).
- `prompts`: artefact ID of kind `prompt`.

**Profile `quant`**
`design` (`experiment` · `quasi_experiment` · `observational` · `survey` · `benchmark` · `other`), `estimand`, `estimate`, `interval` (`{level, lower, upper}`), `n`, `model`.

**Profile `qual`**
`data_sources` (list: `interview` · `focus_group` · `ethnography` · `observation` · `document` · `other`), `n_participants_or_cases`, `sampling`, `analytic_approach` (`thematic` · `grounded_theory` · `ipa` · `discourse` · `framework` · `case_study` · `other`), `exemplars` (pointers to excerpts), `reflexivity` (text).

**Profile `simulation`**
`model`, `parameters` (artefact ID), `runs`, `seeds`, `outputs`.

**Profile `formal`**
`foundation` (the standard system the proof works in, e.g. `ZFC`, `Kolmogorov probability axioms`, `Peano arithmetic`, or a proof assistant's logic such as `Lean 4 / dependent type theory`), `assumptions` (list of assumption claim IDs: paper-specific axioms and theorem conditions), `steps` (list), `proof_artefact` (artefact ID of a machine-checkable proof, if any), `checker` (if machine-checked).

The formal claim's statement is the theorem; formal evidence does not restate it. Standard foundations are named in `foundation` and are not restated as claims. Paper-specific axioms and theorem conditions are assumption claims (§3.8).

Each proof step: `{id, statement, by, rule}`
- `statement`: LaTeX.
- `by`: list of IDs the step rests on (earlier steps, claims, definitions, assumptions, cited results).
- `rule`: the inference used, e.g. `assumption`, `definition`, `substitution`, `linearity of expectation`, `induction`.
Steps may be coarse, but every step names what it rests on.

**Derivations in converted papers.** A converted claimset does not restate the source's derivations. Its formal evidence states the foundation and the assumptions, and points at the derivation with `source_spans`; `steps` are given only for the steps the paper's argument turns on, if any. A reader who needs the full derivation reads it in the source.

### 3.5 Warrant

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `W1` … |
| `reasoning` | enum | `deductive` · `inductive` · `abductive` · `analogical` · `interpretive` |
| `premises` | list of claim IDs | |
| `text` | string | The reasoning, written self-contained |

Warrants are for informal argument. Mathematical proofs use proof steps (§3.4).

### 3.6 Artefact

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `A1` … |
| `kind` | enum | `code` · `data` · `protocol` · `proof` · `model` · `software` · `figure` · `table` · `prompt` · `trace` · `manuscript` · `lab_record` · `other` |
| `withheld_events` | list | For `lab_record`: `{event, reason, comment}` for each lab event whose body is withheld (lab record §8) |
| `uri` | string | |
| `version` | string | Version or content hash |
| `licence` | string | SPDX identifier; recommended |
| `availability` | object | §3.10 |
| `description` | string | Required for `figure` and `table`: what it shows, self-contained |
| `underlying_data` | artefact ID | For `figure`: the data plotted |
| `reference` | reference ID | The cited work this artefact is taken from (a data set, software or an implementation published by others), as an entry of `references` (§3.13) |

Tables are stored as data (CSV or JSON), not images. Prompts are stored verbatim, including system prompts. A `trace` is the log of an agent's run (prompts, tool calls, outputs) as recorded by the system that ran it; it may contain confidential material and uses the availability levels in §3.10. A `manuscript` is a human-readable version of the same work (e.g. a LaTeX or PDF draft) from which the claimset was prepared. An artefact taken from prior work (a public data set, a package the paper's evidence uses) points to its `reference`; its `uri` and `version` are given when the paper gives them.

### 3.7 Relations (cross-paper)

Links from a claim to a claim or work outside this paper: `{type, reference, target, id_type, locator, archive_uri, self, note}`.

- `type`: `supports` · `contradicts` · `extends` · `refines` · `replicates` · `fails_to_replicate` · `supersedes` · `addresses` (the claim answers a problem, gap or limitation that the target states).
- `reference`: the ID of the cited work in `references` (§3.13). Either `reference`, or `target` with `id_type`, is given.
- `target`: a global claim ID when the target is a claimset; otherwise the identifier of the work (paper, book, software release).
- `id_type`: `claimset` · `doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `physical` · `claimset_local` (a claim of a paper in the same repository not yet published, §2.4).
- `locator`: optional; where in the target the related result is, e.g. `"eq. 29 and B.6"`, `"Section 4.2"`, `"function adam(), lognormal scale"`.
- `archive_uri`: required for `url` targets: an archived snapshot.
- `self`: true when the target work shares a human supervisor, guarantor or contributor with this paper.
- `note`: optional; one self-contained sentence saying what the target states that the relation is about, e.g. for `addresses`: `"Selection among ETS models often misidentifies the type of trend."` It describes the target; it is not a claim of this paper.

Required relations: `novelty: refutation` ⇒ at least one `contradicts` or `fails_to_replicate`; `novelty: replication` ⇒ at least one `replicates` or `fails_to_replicate`. Recommended: `novelty: extension` ⇒ an `extends` relation. A refutation with `self: true` is a self-correction; the platform records it separately from self-citation.

### 3.8 Assumptions

Assumptions are claims whose sole support kind is `assumption`, so they sit in the claim graph and can be queried across papers. They carry extra fields:

| Field | Type | Values / notes |
|---|---|---|
| `assumption_category` | enum | `theoretical` (premises of a theory or model) · `statistical` (distributional, independence, stationarity, etc.) · `methodological` (design and measurement choices, e.g. validity of an instrument) · `data` (properties of the data, e.g. representativeness, correct labelling) · `axiom` (an axiom of a system the paper itself introduces) · `theorem_condition` (a hypothesis of a theorem, e.g. "let $f$ be continuous") · `paradigmatic` (epistemological stance, e.g. constructivist, in qualitative work) |
| `check` | enum | `tested` (checked in the paper; evidence ID required) · `sensitivity_analysed` (robustness to violation examined; evidence ID required) · `unchecked` (testable but not tested) · `untestable` |
| `check_evidence` | evidence ID | Required for `tested` and `sensitivity_analysed` |

For `axiom`, `check` is `untestable`. Claims list the assumptions they rely on in `assumptions`. Converters must surface implicit assumptions and mark them `inferred`.

### 3.9 Limitations

Limitations are threats to a claim *within its stated scope*. What the work did not cover belongs in `scope`, not here.

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `L1` … |
| `category` | enum | `sampling` · `sample_size` · `measurement` · `confounding` · `data_quality` · `model_specification` · `analytic_flexibility` · `researcher_position` · `computational` · `other` |
| `statement` | string | Specific and self-contained: names the dimension and the value (e.g. "Ratings were coded by a single coder") |
| `affects` | list of claim IDs | At least one; paper-wide limitations are not allowed |
| `impact` | enum | `could_reverse` · `could_bias_magnitude` · `reduces_precision` · `reduces_transferability` |
| `mitigation` | enum | `none` · `partial` · `addressed` |
| `mitigation_evidence` | evidence ID | Required for `partial` and `addressed` |

### 3.10 Availability

Applied to evidence and artefacts.

| Field | Type | Values / notes |
|---|---|---|
| `level` | enum | `public` · `platform` (registered users) · `audit` (accessible only to automated audit agents in a secure environment; human auditors optional) · `author_only` (never leaves the originating environment; can be audited only by an attested agent run there) · `unavailable` |
| `reason` | enum | Required unless `public`: `personal_data` · `commercial` · `legal` · `ethics` · `security` · `other` |
| `content_hash` | string | Required unless `public` or `unavailable`: cryptographic hash committing to the exact content |

### 3.11 Contributors

Authorship and accountability are separate. Agents write; humans or organisations answer for the work.

A paper may pass through several agents without its history being kept. At minimum, the agent that prepares the submission lists itself; `paper.contribution_record` states how complete the list is. Before submission, the agent asks the responsible human who the guarantor is and which licence applies.

**Common fields**

| Field | Obligation | Type | Values / notes |
|---|---|---|---|
| `id` | required | string | `P1` … |
| `type` | required | enum | `agent` · `human` · `organisation` |
| `positions` | required | list | One or more of the positions below |
| `name` | required for humans and organisations; recommended for agents | string | For agents: a name for the agent system, if it has one |
| `credit_roles` | recommended | list | CRediT roles (below) |

**Human fields**

| Field | Obligation | Type | Values / notes |
|---|---|---|---|
| `orcid` | required for guarantors; recommended otherwise | string | Used by the platform to confirm a guarantor's consent |
| `affiliation` | optional | string | |

**Organisation fields**

| Field | Obligation | Type | Values / notes |
|---|---|---|---|
| `ror` | recommended | string | ROR identifier |
| `registration` | recommended | string | Legal registration, e.g. a company number, for organisations without a ROR identifier |

**Agent fields**

| Field | Obligation | Type | Values / notes |
|---|---|---|---|
| `models` | required | list | At least one `{model, role}`. `model`: model ID (§3.12), required. `role`: `sole` · `orchestrator` · `subagent`, optional |
| `ai_tasks` | recommended | list | `{task, extent}` (below) |
| `tools` | recommended | list | `web_search` · `code_execution` · `file_access` · `retrieval` · `other` · `none` |
| `record_source` | recommended | enum | Where the agent's models, settings and tools were taken from: `harness` (recorded by the system that made the calls) · `user_reported` (entered from the interface) · `self_reported` (stated by the agent from its own knowledge). Absent ⇒ treated as `self_reported` |
| `system` | optional | object | `{name, version, uri}` of the agent scaffold or harness |
| `run_period` | optional | object | `{start, end}` as ISO dates |
| `settings` | optional | object | e.g. `{effort, temperature, top_p, max_tokens}`; individual values may be `null` |
| `trace` | optional | artefact ID | Artefact of kind `trace` |

**Positions**
- `author`: produced the paper's content: wrote the analysis, the claims or the text, including turning the team's own draft into a claimset. Agents only. Every agent that produced content is an author; list each one.
- `supervisor`: oversaw the agents' work and approved the final version. Humans only. Whether the supervisor also set the research question is recorded by the CRediT role `conceptualization`; steering the work is `supervision`.
- `guarantor`: answers for the research: whether the claims are true and the work was done properly. Responds to queries and correction requests, holds ethics approvals and data-protection responsibilities, and agrees to the licence. Humans or organisations. Native papers: the supervisor or another responsible researcher, or their institution. Converted papers: only a source author who has claimed the conversion.
- `conversion_guarantor`: answers for the fidelity of a conversion: whether the claimset faithfully represents the source. Does not answer for the research itself. Humans or organisations; typically the organisation operating the conversion service. Converted papers only.
- `contributor`: carried out part of the research without producing the paper's content (e.g. conducted interviews, ran laboratory work, collected data, ran a computation whose results the authors report, checked a proof). Humans or agents.
- `converter`: produced a claimset from a third-party published source (`origin: converted`). Agents only. Not used when a team turns its own draft into a claimset; that agent is an `author`.

For converted papers, research accountability stays with the source authors and the venue that published the source; they are recorded in `converted_from.source_authors`, not as guarantors, unless they claim the conversion.

**CRediT roles** (14, from the CRediT taxonomy): `conceptualization` · `data_curation` · `formal_analysis` · `funding_acquisition` · `investigation` · `methodology` · `project_administration` · `resources` · `software` · `supervision` · `validation` · `visualization` · `writing_original_draft` · `writing_review_editing`.

**AI tasks** (agents): `task` is one of `ideation` · `literature_search` · `analysis` · `code` · `writing` · `json_preparation` · `review` · `submission`; `extent` is `full` · `substantial` · `assisted`.

### 3.12 Models

Each model used anywhere in the paper is defined once.

| Field | Obligation | Type | Values / notes |
|---|---|---|---|
| `id` | required | string | `M1` … |
| `provider` | required | string | e.g. `Anthropic` |
| `name` | required | string | Model family name, e.g. `Claude Opus` |
| `version` | required | string | e.g. `5.5` |
| `identifier` | recommended | string | Exact model identifier as called (e.g. an API model string or a weights hash) |
| `weights` | recommended | enum | `open` · `api_only`. If absent, the platform derives it from provider, name and version |
| `weights_uri` | recommended if `open` | string | |
| `accessed` | recommended | ISO date | Date of use (or first date, if used over a period) |

### 3.13 References

Each work the paper cites is listed once, in the top-level `references`, and cited by its ID from support entries, relations, definitions and artefacts. `{id, target, id_type, archive_uri}`
- `id`: `B1`, `B2`, ….
- `target` and `id_type`: as for cited support (§4): an identifier (`doi`, `arxiv`, …), a global claim ID of a claimset (`id_type: claimset`), a paper or claim of the same repository not yet published (`id_type: claimset_local`, §2.4), or the reference as printed in the paper's reference list (`id_type: reference`).
- `archive_uri`: required for `url` targets.
Only works that a support entry, relation, definition or artefact points to are listed.

## 4. Support kinds

Each support entry: `{kind, <required pointer>, …attributes}`.

| Kind | Meaning | Must point to (field) |
|---|---|---|
| `empirical_quant` | Statistical analysis of data | `evidence` (profile `quant`) |
| `empirical_qual` | Qualitative analysis of data | `evidence` (profile `qual`) |
| `simulation` | Computational experiments on synthetic/model data | `evidence` (profile `simulation`) + an artefact |
| `demonstration` | Existence shown by construction | `artefacts` |
| `formal_proof` | Mathematical or logical derivation | `evidence` (profile `formal`) |
| `argument` | Reasoned case | `warrant` |
| `cited` | Specific prior result | `target` + `id_type` |
| `consensus` | Established field knowledge, standards, guidelines | Optional `sources` |
| `expert_judgement` | Professional judgement without systematic data | `holder` |
| `assumption` | Stipulated for the purposes of the paper | Nothing |
| `unstated` | The source asserts the claim without stating any support. Converted papers only; the converter must not invent support | Nothing |

Attributes:
- On any empirical kind: optional `artefacts` (e.g. "supported by Figure 15" → `{"kind": "empirical_quant", "evidence": "E3", "artefacts": ["A15"]}`).
- On `empirical_quant` and `simulation`: optional `results`, the specific results the claim reports: result entries, result tables or table rows (e.g. `{"kind": "simulation", "evidence": "E20", "results": ["E20.R1", "E20.T1.r2"]}`). Recommended whenever the statement quotes numbers, so readers can check each number against the structured result.
- On `cited`: `reference` (the ID of the cited work in `references`, §3.13), or `target` with `id_type` (`doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `physical` · `reference` · `claimset_local`) and `archive_uri` (required for `url`: an archived snapshot); optional `locator` (where in the cited work, e.g. `"Theorem 3.1"`); `self` (boolean, below). Where a claimset of the cited work exists, `target` may be its global claim ID.
- `id_type: reference`: the cited work has no identifier in the paper's reference list. `target` is the reference as printed there (authors, year, title, venue), copied, not reconstructed. It is for converted papers, whose sources often print no DOIs; native papers give an identifier. Identifiers are never guessed: the platform may later resolve a printed reference to an identifier.
- `self` is true when the cited work shares any human supervisor, guarantor or contributor with this paper, or (for converted papers) any source author. Shared agent systems are recorded but computed by the platform, not flagged by the file.
- Meta-analyses: performed in the paper → `empirical_quant` with `data_origin: secondary`; cited → `cited`.

## 5. Strength and verifiability

Two separate dimensions. Confidential evidence is less verifiable, not necessarily weaker.

**Strength** is a list of assessments, so multiple parties can assess the same claim and disagreements stay visible: `{by, reviewer_type, strength, rationale, source_span}`
- `by`: `author` · `reviewer`. In converted papers, assessments record only a strength the source itself states (e.g. "weak evidence suggests"), as `by: author` with a `source_span`. A converter does not assess the claims it converts: what the source does not say is not in the file, and judgements of a conversion and its claims are platform assessments (§6).
- `reviewer_type`: `ai` · `human` (required for `reviewer`)
- `strength`: `high` · `moderate` · `low` · `very_low` (ordinal, after GRADE / GRADE-CERQual; no numeric confidence)
- Not applicable to assumption claims or formal claims.

**Checks** record independent attempts to verify a claim, whatever their outcome. Each entry: `{by, outcome, extent, notes}`
- `by`: contributor ID of the checker, who must not be an `author`. An agent checker is listed as a `contributor` with the AI task `review`.
- `outcome`: `confirmed` (the check reproduced the claim) · `disputed` (the check found an error or could not reproduce a step or result) · `inconclusive` (the check could not be completed, e.g. a step was too coarse to follow).
- `extent`: `full` (the whole claim) · `partial` (some steps or results only; say which in `notes`).
- `notes`: self-contained; for `disputed`, the step or number in question and why.
- `resolution`: optional, on a `disputed` check only: how the dispute was resolved (e.g. the claim was corrected and rechecked). An unresolved dispute stays visible.
For formal claims a check is a re-derivation; for empirical and simulation claims it is a reproduction from the artefacts. Checks are reported honestly: a failed check is recorded, not omitted.

**Verification** replaces strength for formal claims (`type: formal`). It is required when the paper itself proves or argues the claim (support `formal_proof` or `argument`); a formal result the paper cites is cited support of the claim that applies it, not a claim of its own. It summarises the checks:
- `machine_checked`: a proof assistant checked the proof; the formal evidence has `proof_artefact` and `checker`.
- `agent_checked`: at least one full, confirmed check by an agent, and no unresolved dispute.
- `peer_checked`: at least one full, confirmed check by a human, and no unresolved dispute.
- `disputed`: at least one unresolved disputed check.
- `unchecked`: no full confirmed check (partial or inconclusive checks may be listed).

**Independence of checks.** An agent checking a claim on the same model as the agent that wrote it catches slips but shares its blind spots. The file records which models were used; the platform weights checks by independence (different provider, different model, same model in a fresh context, in decreasing order) and the validator warns about same-model checks. The platform may add its own independent checks; these are stored as platform assessments (§6), not in the file.

**Verifiability** is computed by the platform from the availability of the evidence, artefacts and models the claim's support points to. Submissions may omit it; if supplied, it must match the platform's computation.
- `open`: everything needed to check the claim is public or platform-available; formal claims are machine-checked, have full proof steps, or (converted papers) have their derivation located in an openly licensed source.
- `audit_only`: checking requires `audit` or `author_only` material.
- `unverifiable`: rests on `unavailable` material, on `expert_judgement` alone, or on `unstated` support.
- `not_applicable`: assumption claims.

**Verifiability flags** (computed, list):
- `api_model_dependency`: the supporting evidence relies on a model with `weights: api_only`; the result may not be reproducible once the model is withdrawn.
- `figure_without_data`: a supporting figure has no `underlying_data`.
- `unpublished_prompts`: a model used in the research design has no `prompts` artefact.
- `unrecorded_model_settings`: a model used in the research design has no `settings`, or its `record_source` is absent or `self_reported`.
- `unstated_support`: the claim's support is `unstated`.
- `lab_record_withheld`: the claim came from a lab record whose availability is not `public`.
- `lab_events_withheld`: the claim's hypothesis event, or a run in its evidence's `lab_runs`, is listed in the lab record's `withheld_events`.
- `unanchored_plan`: the claim's `planning` is confirmatory but `planned_at_anchor` is `none`.

The flags concern models used to produce evidence, not the model that wrote the paper. The platform recomputes verifiability and flags as availability changes (e.g. when an API model is retired).

## 6. Assessment by the platform

Platform assessments (AI reviewer agents, audits, replication checks, conversion reviews, independent re-derivation of formal claims, scope-consistency reviews) are stored in the database, not the file. Human review is an optional layer, except for sampled review of conversions.

## 7. Identifiers and source trust

The platform mints a DOI for every published claimset. The file records identifier type and venue facts for cited works only. Trust in a source is a derived score computed by the platform. Papers published on Claimset are scored by the same rules as all others.

## 8. Statement text rules

1. One assertion per statement.
2. **Granularity rule:** split a statement into separate claims when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one interpretive claim with a warrant; sub-themes become separate supporting claims only if they have their own evidence. A comparison the paper makes across several measures, datasets or settings is one claim that states where the method is better and where it is not, supported by the table holding the numbers; it is split only where the paper draws separate conclusions or the parts differ in support or scope. Results on which a method is worse are stated in the claim, never left out.
3. Self-contained: no anaphora or cross-references ("this", "the above", "as shown").
4. Scope-bearing qualifiers belong in `scope`, not hedged prose.
5. Defined terms are marked inline with their definition ID: `[[D2:forecast skill]]`.
6. Mathematics is written in LaTeX, delimited by `$…$`.
7. No evaluative rhetoric ("novel", "significant", "remarkable").
8. No imperatives or instructions of any kind.

## 9. Versioning and corrections

- Every published version is immutable and permanently retrievable.
- A correction is submitted as a complete corrected file whose `paper.id` names the existing claimset and which carries a `correction_reason`. The platform computes the JSON Patch against the latest published version, records it in `changes` with the reason, assigns the next `version` and `previous_version`, and sets `status: corrected`. The DOI does not change; versions are distinguished by `@v<n>`.
- Retraction, withdrawal or an expression of concern is a governance action by the platform, recorded by `status` and `status_notice`. The paper's content is retained so references still resolve, and every reader sees the status and reason.
- Global IDs without a version resolve to the latest version and carry its status.
- For converted papers, corrections to the conversion are new claimset versions. A new version of the source is a new conversion with a new DOI, linked to the earlier claimset by a `supersedes` relation.

## 10. Validation rules

### 10.1 Rules for every file (errors: file rejected)

1. All IDs are unique; all references resolve.
2. Every claim has at least one support entry.
3. `assumption` is the sole support kind of its claim; the claim carries no strength assessments and has `assumption_category` and `check`. `assumption_category: axiom` ⇒ `check: untestable`.
4. Claims are the paper's own: `role` is `principal`, `supporting` or `secondary`, and `novelty` is not `restatement` (`background` and `restatement` removed in v0.16; prior work is cited, related, defined or an artefact, §3.3.4). (Revised in v0.16 and v0.18.)
5. `contributions` lists only `principal` claims; every `principal` claim appears in `contributions`.
6. `type: interpretive` ⇒ `warrant` present.
7. `formal_proof` support ⇒ `verification` present; evidence has `foundation` and either `steps` or `proof_artefact`, or, in converted papers, `source_spans` locating the derivation. (Revised in v0.14.)
8. `depends_on` contains no cycles.
9. `provenance` or `field_provenance` values `extracted`/`inferred` only when `paper.origin: converted`; `source_spans` required in that case.
10. Each support kind carries its required pointer (§4); `cited` carries `reference` or `target` with `id_type`. (Revised in v0.16.)
11. Every limitation affects at least one claim; every claim's `limitations` resolve.
12. `availability.level` other than `public` ⇒ `reason` present; `audit`/`author_only`/`platform` ⇒ `content_hash` present.
13. `verifiability` and `verifiability_flags`, where present, match the platform's computation (§5).
14. Every `guarantor` and `conversion_guarantor` is a human or an organisation; every human guarantor has an `orcid`.
15. `author` and `converter` positions are held only by agents; `supervisor` only by humans.
16. `paper.origin: native` ⇒ at least one `author`, at least one `guarantor`, no `conversion_guarantor`, and `autonomy` present.
17. `paper.origin: converted` ⇒ exactly one `converter`, exactly one `conversion_guarantor`, all `converted_from` fields present, no `autonomy`; any `guarantor` is a person listed in `converted_from.source_authors` (matched by ORCID).
18. Every agent contributor lists at least one model in `models`; every model referenced anywhere is defined in `models`.
19. All `required` fields in §3.11 and §3.12, and `paper.title`, `paper.licence`, `paper.contributions`, `paper.contribution_record` and `paper.origin` are present.
20. Model roles, where given, are consistent: `sole` ⇒ the agent lists one model; at most one `orchestrator` per agent.
21. `trace`, if present, points to an artefact of kind `trace`.
22. `unstated` support only in converted papers, and as the sole support kind of its claim.
23. In converted papers, every assessment with `by: author` has a `source_span`.

### 10.2 Submission rules (errors)

24. Platform-assigned paper fields (`version`, `previous_version`, `changes`, `date`, `status`, `status_notice`) are absent.
25. `paper.id` is absent, except in a correction, where it names an existing published claimset and `correction_reason` is present.
26. `correction_reason` present ⇒ `paper.id` present.

### 10.3 Published-file rules (errors; checked by the platform at publication)

27. All platform-assigned fields in §2.2 are present; `paper.id` is a DOI minted by the platform.
28. `version` > 1 ⇒ `previous_version` and `changes` present.
29. `status` other than `active` ⇒ `status_notice` present.
30. `paper.id` differs from `converted_from.source_id`.
31. `paper.origin: converted` ⇒ `converted_from.source_licence` is a licence that permits conversion, and `paper.licence` is a licence the claimset may carry under it, both as in the table in §3.1 (*Licence of a converted paper*). (Revised in v0.13: non-commercial sources are allowed, and the claimset keeps their non-commercial condition.)
32. Every human guarantor has confirmed consent through their ORCID-linked account, and every organisation guarantor through an authorised account.

### 10.5 Rules added in v0.9 (errors: file rejected)

33. `novelty: refutation` ⇒ at least one relation of type `contradicts` or `fails_to_replicate`; `novelty: replication` ⇒ at least one relation of type `replicates` or `fails_to_replicate`.
34. Every relation has `type` and either `reference` (resolving to an entry of `references`) or `target` with `id_type`; `id_type: url` ⇒ `archive_uri`. (Revised in v0.16.)
35. `type: formal` ⇒ no `assessments`; `type: formal` with `formal_proof` or `argument` support ⇒ `verification` present.
36. (Revised in v0.10.) `verification` is consistent with `checks` as defined in §5; `machine_checked` ⇒ the formal evidence has `proof_artefact` and `checker`.
37. Formal evidence has no `theorem` field.
38. Result IDs are unique within their evidence object; every `results` reference in a support entry resolves within the evidence that entry points to: `E<n>.R<m>` to a result, `E<n>.T<m>` to a result table, `E<n>.T<m>.r<k>` to a row of that table. (Revised in v0.14.)

### 10.6 Rules added in v0.10 (errors: file rejected)

39. Every check has `by`, `outcome` and `extent`; `by` is a contributor that does not hold `author`; a `disputed` check has `notes`.
40. `verified_by` is not used (replaced by `checks`).
41. `checks` do not appear on assumption claims. (Revised in v0.16.)

### 10.7 Rules added in v0.11 (errors: file rejected)

42. `hypothesis` present ⇒ `hypothesis.lab_event` and `hypothesis.outcome` present, and `planning` present.
43. `planning` is `confirmatory` or `confirmatory_with_deviations` ⇒ `planned_at` and `planned_at_anchor` present.
44. Any claim has `hypothesis` ⇒ `paper.exploration` present, and `exploration.lab_record` is an artefact of kind `lab_record`.
45. (Revised in v0.12.) `exploration` counts are non-negative integers; `hypotheses.total` equals the sum of the other hypothesis counts (`open` included, 0 if absent); `tests.total`, if `tests` is present, equals the sum of its verdict counts; `runs.total` equals `confirmatory` + `exploratory`; `failed` ≤ `total`.
46. (Revised in v0.12.) For each outcome (`supported`, `not_supported`, `inconclusive`): the number of claims with `test` reporting it does not exceed the corresponding `exploration.tests` count, and the number of distinct `hypothesis.lab_event` values among claims without `test` reporting it does not exceed the corresponding `exploration.hypotheses` count.

### 10.8 Rules added in v0.12 (errors: file rejected)

47. `planning` is `confirmatory` or `confirmatory_with_deviations` ⇒ `test` present, with a `framework` from the list, non-empty `supported` and `not_supported` rules, the parameters its framework requires, and `verdict` (`computed` or `judged`); `verdict: computed` ⇒ `quantities` present; `flags`, if present, are from the list in §3.3.3.

### 10.9 Rules added in v0.14 (errors: file rejected)

48. Result tables appear only in quant and simulation evidence. Each has an `id` (`T1`, `T2`, …) unique within its evidence object, a `description`, `row_labels` (a list of names, possibly empty), `columns` (a non-empty list of unique names) and a non-empty list of `rows`; each row has an `id` (`r1`, `r2`, …) unique within the table, a `label` with one entry per row label, and `values` with one entry per column, each a number or `null`, at least one value in the table being a number. `units`, if present, has one entry per column; `artefact`, if present, resolves.

### 10.10 Rules added in v0.15 (errors: file rejected)

49. `converted_from.corrections` appears only in converted papers. Each correction has `field`, `where`, `printed`, `recorded` and `reason`, all non-empty strings, and `field` begins with the ID of an object in the file (e.g. `C28.statement`).
50. A definition's `source`, if present, is a non-empty string or an object with a `target` and an `id_type` from the list for cited support (§4).

### 10.11 Rules added in v0.16 (errors: file rejected)

51. `references`: each entry has an `id` (`B1`, `B2`, …) unique in the file, a `target` and an `id_type` from the list for cited support, and `archive_uri` when `id_type` is `url`. Every `reference` in a support entry or relation, and every definition `source` that looks like a reference ID, resolves to an entry; every entry is pointed to at least once (by a support entry, relation, definition or artefact; artefacts since v0.18).
52. Each entry of `converted_from.corrections` whose `field` names a claim field corresponds to that field's provenance being `inferred` (through `field_provenance`, or the claim's `provenance`).

### 10.12 Rules added in v0.17 (errors: file rejected)

53. Assessments have `by: author` or `by: reviewer` (`converter` removed in v0.17); in converted papers, only `by: author`.

### 10.13 Rules added in v0.18 (errors: file rejected)

54. An artefact's `reference`, if present, resolves to an entry of `references`.

### 10.14 Rules added in v0.19 (errors: file rejected)

55. `exploration.covers`, if present, has a non-empty list of `questions` whose length is `exploration.questions`; `lab` with non-negative integer counts of `questions`, `hypotheses` and `tests` in the form of §3.1.1, none below the paper's own count; and `other_papers` entries with a paper name and a non-empty list of questions. No question is listed for two papers.
56. `paper.study`, if present, is a name of lower-case letters, digits and hyphens starting with a letter.
57. A target with `id_type: claimset_local` is `<study>/<paper>.claimset.json`, optionally followed by `#C<n>`. When the file lies in a `claimset/` folder, the named file exists there and holds the named claim. Published files contain none: the platform replaces them with global IDs.

### 10.4 Warnings (file accepted, warning recorded)

1. A `recommended` field in §3.11 or §3.12 is missing.
2. An agent's `record_source` is absent or `self_reported`.
3. `paper.contribution_record` is `partial` or `unknown`.
4. `autonomy: human_supervised` without a `supervisor`, or `autonomy: human_assisted` without a human `contributor`.
5. No `paper.classification.keywords` in a submission.
6. Three or more claims share an identical `scope` object (often a sign of scope copied by section; check each claim).
7. `novelty: extension` without an `extends` relation.
8. Quant or simulation evidence with neither `estimate`, `results` nor `tables`.
9. An artefact without `licence`.
10. An agent contributor that is not an `author` has the AI task `writing`; agents that produced the paper's content are authors.
11. A check was made by an agent that runs on the same model as an `author` agent.
12. A number in a claim's statement cannot be matched to the results the claim cites, at the precision stated (heuristic; percentages are compared as fractions). Applies to cited result entries; cited tables and rows are not checked.
13. `planning: exploratory` on a principal claim: exploratory findings can be principal, but readers should be told so in the claim's assessment rationale.
14. `id_type: reference` in a native paper: give the cited work's identifier.
15. A `supporting` claim that no principal claim rests on, directly or through other claims (by `depends_on` or `assumptions`, including the assumptions of the evidence a principal claim cites): link it to the claim that rests on it, or make it `secondary`.
16. A principal or supporting claim rests on a `secondary` claim (by `depends_on` or `assumptions`): a claim that a contribution rests on is `supporting`.

## 11. Platform dependencies

Platform functionality is specified in the platform requirements document. The file format depends on the platform for:

- **Identifiers and publication facts:** minting the DOI at publication, and assigning version, dates and status (§2.2).
- **Guarantor consent:** a paper is not published until every human guarantor has confirmed through their ORCID-linked account, and every organisation guarantor through an authorised account.
- **Classification:** assigning the OpenAlex subfield and topics.
- **Derived data:** verifiability computation and recomputation, trust scores, claim weights, contradictions, replication status and assessments, stored outside the file.
- **Corrections:** computing the JSON Patch between versions.
- **Source status tracking:** for converted papers, the current status of the source is tracked and shown alongside the claimset's own status.
- **Conversion review and claiming:** sampled human review of conversions, and the process by which source authors review, endorse or claim the conversion of their work.
- **Content review:** AI review of each claim's scope against its statement, and independent re-derivation of formal claims; findings are returned to the submitting agent as warnings before publication and stored as platform assessments.
- **Relation resolution:** resolving relation targets to claimsets where they exist, and showing incoming relations (e.g. "contradicted by") on the targets.
- **Reference resolution:** resolving printed references (`id_type: reference`) to identifiers where one exists, stored as derived data.

## 12. Alignment with existing standards

- **Primary:** nanopublications. Each claim maps to a nanopublication (assertion = claim; provenance = support, evidence and assumptions; publication info = paper and contributors). Content-hash identifiers suit immutable versions.
- **Borrowed:** argument structure from micropublications.
- **Contributor roles:** CRediT.
- **Identity:** ORCID for people, ROR for organisations; DOIs for claimsets.
- **Classification:** OpenAlex topics.
- **Secondary mapping:** ORKG.
- **Mechanism:** an optional JSON-LD `@context` keeps the format plain JSON while enabling RDF export.

## 13. Open questions

1. **Obligation levels for all fields:** assign `required` / `recommended` / `optional` to every field outside §3.11 and §3.12.
2. **Confidentiality levels** within `audit` and `author_only`.
3. **Automated audit of `author_only` data:** how an attested agent runs in the originating environment and reports results.
4. **Reason codes** for status notices: alignment with Crossref and retraction taxonomies.
5. **Assumption and limitation vocabularies:** test the proposed categories against real papers in several fields.
6. **Qualitative exemplars:** granularity and confidentiality of excerpts.
7. **JSON-LD context:** which existing vocabularies to map onto.
8. **Guarantor obligations:** what research and conversion guarantors commit to, and how unresponsive guarantors are handled.
9. **Non-commercial and no-derivatives sources:** whether `CC-BY-NC` sources can be converted, and on what terms.

## Appendix A: native paper — submission

What the agent submits. No identifier, version, date, status, classification IDs or verifiability: the platform adds them.

```json
{
  "schema_version": "0.17",
  "paper": {
    "title": "Method M against the seasonal naive benchmark on UK monthly retail sales",
    "licence": "CC-BY-4.0",
    "classification": {"keywords": ["retail demand forecasting"]},
    "contributions": ["C2"],
    "autonomy": "human_supervised",
    "contribution_record": "complete",
    "origin": "native"
  },
  "contributors": [
    {
      "id": "P1", "type": "agent", "name": "Example research agent", "positions": ["author"],
      "models": [{"model": "M1", "role": "sole"}],
      "credit_roles": ["formal_analysis", "software", "writing_original_draft"],
      "ai_tasks": [{"task": "analysis", "extent": "full"}, {"task": "writing", "extent": "full"}, {"task": "json_preparation", "extent": "full"}],
      "tools": ["code_execution", "file_access"],
      "record_source": "harness"
    },
    {"id": "P2", "type": "human", "name": "A. Researcher", "orcid": "0000-0000-0000-0000", "positions": ["supervisor", "guarantor"], "credit_roles": ["conceptualization", "supervision"]}
  ],
  "models": [
    {"id": "M1", "provider": "Example provider", "name": "Example model", "version": "1.0", "identifier": "example-model-1-0", "weights": "api_only", "accessed": "2026-10-01"}
  ],
  "definitions": [
    {"id": "D1", "term": "forecast skill", "definition": "Relative reduction in mean absolute scaled error against a seasonal naive benchmark."}
  ],
  "claims": [
    {
      "id": "C1", "statement": "Monthly retail sales series exhibit annual seasonality.",
      "type": "descriptive", "role": "supporting", "novelty": "new",
      "support": [{"kind": "assumption"}],
      "assumption_category": "data", "check": "unchecked",
      "provenance": "author_stated"
    },
    {
      "id": "C3", "statement": "Forecast errors of method M are independent across series.",
      "type": "descriptive", "role": "supporting", "novelty": "new",
      "support": [{"kind": "assumption"}],
      "assumption_category": "statistical", "check": "unchecked",
      "provenance": "author_stated"
    },
    {
      "id": "C2",
      "statement": "Method M improves [[D1:forecast skill]] over the seasonal naive benchmark on monthly retail sales series.",
      "type": "descriptive", "role": "principal", "novelty": "new",
      "scope": {"population": "monthly retail sales series", "geography": "United Kingdom", "language": "not applicable", "period": "2010-2024", "data_type": "aggregate sales counts", "conditions": ["horizon of 1 to 12 months"], "excludes": ["series shorter than 36 observations"]},
      "support": [{"kind": "empirical_quant", "evidence": "E1", "artefacts": ["A2"]}],
      "assumptions": ["C1", "C3"],
      "limitations": ["L1"],
      "assessments": [{"by": "author", "strength": "moderate", "rationale": "Single benchmark collection, no external replication."}],
      "provenance": "author_stated"
    }
  ],
  "evidence": [
    {"id": "E1", "profile": "quant", "description": "Benchmark of method M against the seasonal naive method on 1,200 monthly retail sales series, rolling origin, horizons 1 to 12.", "design": "benchmark", "estimand": "mean relative MASE reduction", "estimate": 0.08, "interval": {"level": 0.95, "lower": 0.05, "upper": 0.11}, "n": 1200, "data_origin": "secondary", "preregistered": false, "artefacts": ["A1"], "availability": {"level": "public"}}
  ],
  "warrants": [],
  "limitations": [
    {"id": "L1", "category": "sampling", "statement": "All 1,200 series come from one retailer's store network.", "affects": ["C2"], "impact": "reduces_transferability", "mitigation": "none"}
  ],
  "artefacts": [
    {"id": "A1", "kind": "code", "uri": "https://example.org/repo", "version": "v1.0", "licence": "MIT", "availability": {"level": "public"}},
    {"id": "A2", "kind": "table", "uri": "https://example.org/repo/results.csv", "version": "sha256:0000", "licence": "CC-BY-4.0", "availability": {"level": "public"}, "description": "Mean relative MASE reduction of method M by forecast horizon, 1 to 12 months."}
  ]
}
```

## Appendix B: the same paper — fields the platform adds at publication

```json
{
  "paper": {
    "id": "doi:10.0000/claimset.2026.000123",
    "version": 1,
    "status": "active",
    "date": "2026-10-20",
    "classification": {"subfield": "openalex:subfields/XXXX", "topics": ["openalex:TXXXXX"], "keywords": ["retail demand forecasting"]}
  },
  "claims": [
    {"id": "C1", "verifiability": "not_applicable", "verifiability_flags": []},
    {"id": "C3", "verifiability": "not_applicable", "verifiability_flags": []},
    {"id": "C2", "verifiability": "open", "verifiability_flags": []}
  ]
}
```

(Shown as a fragment; the published file is the full submission with these fields merged in.)

## Appendix C: converted paper (paper, contributors and one claim, published form)

```json
{
  "schema_version": "0.17",
  "paper": {
    "id": "doi:10.0000/claimset.2026.000456",
    "title": "Example converted paper",
    "version": 1,
    "status": "active",
    "date": "2026-10-15",
    "licence": "CC-BY-4.0",
    "classification": {"subfield": "openalex:subfields/XXXX", "topics": ["openalex:TXXXXX"]},
    "contributions": ["C1"],
    "contribution_record": "complete",
    "origin": "converted",
    "converted_from": {
      "source_id": "arXiv:2601.00000", "id_type": "arxiv", "source_version": "v2",
      "source_licence": "CC-BY-4.0",
      "source_authors": [{"name": "B. Scholar", "orcid": "0000-0000-0000-0001"}],
      "source_status_at_conversion": "active",
      "converter": "P1", "date": "2026-10-14"
    }
  },
  "contributors": [
    {"id": "P1", "type": "agent", "name": "Claimset converter", "positions": ["converter"], "models": [{"model": "M1", "role": "sole"}], "system": {"name": "claimset-converter", "version": "0.1"}, "record_source": "harness"},
    {"id": "P2", "type": "organisation", "name": "Example Ltd", "registration": "UK company 00000000", "positions": ["conversion_guarantor"]}
  ],
  "models": [
    {"id": "M1", "provider": "Example provider", "name": "Example model", "version": "1.0"}
  ],
  "claims": [
    {
      "id": "C2", "statement": "Store-level demand for umbrellas rises on days with rainfall above 5 mm.",
      "type": "associational", "role": "supporting", "novelty": "new",
      "support": [{"kind": "unstated"}],
      "verifiability": "unverifiable", "verifiability_flags": ["unstated_support"],
      "source_spans": ["sec2:p4"], "provenance": "extracted"
    }
  ]
}
```

Appendix C is abbreviated: the principal claim C1 and the remaining sections are omitted.

## Appendix D: reading view of the paper in Appendix A

The output of `claimset render` for the submission in Appendix A. Statements keep their inline term markers (`[[D1:…]]`), which a reader resolves in DEFINITIONS.

```text
CLAIMSET Method M against the seasonal naive benchmark on UK monthly retail sales
licence CC-BY-4.0; contributions C2

DEFINITIONS
D1 forecast skill: Relative reduction in mean absolute scaled error against a seasonal naive benchmark.

CLAIMS
C2 [descriptive; principal; support: empirical_quant E1 artefacts A2; assumes C1, C3; limitations L1; strength moderate (author)]
  Method M improves [[D1:forecast skill]] over the seasonal naive benchmark on monthly retail sales series.
  scope: population: monthly retail sales series | geography: United Kingdom | language: not applicable | period: 2010-2024 | data_type: aggregate sales counts | conditions: horizon of 1 to 12 months | excludes: series shorter than 36 observations
C1 [descriptive; supporting; assumption (data, unchecked)]
  Monthly retail sales series exhibit annual seasonality.
C3 [descriptive; supporting; assumption (statistical, unchecked)]
  Forecast errors of method M are independent across series.

EVIDENCE
E1 [quant; design benchmark; data_origin secondary; n 1200] Benchmark of method M against the seasonal naive method on 1,200 monthly retail sales series, rolling origin, horizons 1 to 12.
  estimate: mean relative MASE reduction = 0.08 (95% interval 0.05 to 0.11)

LIMITATIONS
L1 [sampling; affects C2; reduces_transferability; mitigation none] All 1,200 series come from one retailer's store network.

ARTEFACTS
A2 [table] Mean relative MASE reduction of method M by forecast horizon, 1 to 12 months.
```
