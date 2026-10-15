# Claimset — Specification v0.9 (draft)

Status: working draft · Date: 2026-10-15 · Supersedes v0.8 (`spec/claimset-v0.8.md`)

Claimset is the name of both the open format and the platform.

## Changes since v0.8

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

## 2. Format

- Canonical format: JSON, validated by a normative JSON Schema (the schema is the standard).
- Alternative serialisation: JSON Lines, one object per line, each with a `type` field. Intended for retrieval and indexing.
- Optional `@context` makes the file valid JSON-LD, enabling RDF export without changing the JSON.
- IDs: local IDs within a paper (`C3`, `E2`, `D1`, `W1`, `A1`, `L1`, `P1`, `M1`). Global IDs append the local ID to the claimset's persistent identifier and version: `doi:10.xxxx/yyyy@v2#C3` (version optional; defaults to latest). For converted papers this is the claimset's own identifier, never the source's.

### 2.1 Obligation levels

- `required`: missing or invalid ⇒ validation error; the file is rejected.
- `recommended`: missing ⇒ validation warning; the file is accepted and the warning is recorded by the platform.
- `optional`: no check when absent.

Unknown values of non-required fields are omitted or set to `null`. `null` means "unknown", never "default". In v0.9 obligation levels are assigned for contributors and models (§3.11, §3.12); other objects follow in a later version.

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
  "schema_version": "0.9",
  "@context": "optional",
  "paper": {},
  "contributors": [],
  "models": [],
  "definitions": [],
  "claims": [],
  "evidence": [],
  "warrants": [],
  "limitations": [],
  "artefacts": []
}
```

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
| `licence` | string | Required. SPDX identifier, e.g. `CC-BY-4.0`. Native papers: chosen by the supervisor or guarantor. Converted papers: see §10.3 |
| `classification` | object | `subfield` (one OpenAlex subfield ID) and `topics` (1–3 OpenAlex topic IDs): platform-assigned; the agent may propose values. `keywords` (free text, for specifics the taxonomy does not reach): supplied by the agent, recommended |
| `contributions` | list of claim IDs | Required. Ordered by priority; principal claims only |
| `autonomy` | enum | Native papers only, required: `autonomous` (agents ran the work end to end) · `human_supervised` (humans oversaw the agents' work and approved it) · `human_assisted` (humans carried out substantial parts of the research) · `unknown` |
| `contribution_record` | enum | Required. `complete` · `partial` · `unknown`. For converted papers, refers to the conversion |
| `origin` | enum | Required. `native` · `converted` |
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

Contributors are listed in the top-level `contributors` list (§3.11), not in the paper object.

Retracted papers are never deleted. All their claims inherit the retraction status for readers and the platform. A converted claimset whose source is retracted is shown with the source's retraction, whatever its own status.

### 3.2 Definition

| Field | Type | Notes |
|---|---|---|
| `id` | string | `D1` … |
| `term` | string | |
| `definition` | string | One term, one meaning, used rigidly throughout |
| `source` | claim/DOI ref | Optional, if the definition is borrowed |

### 3.3 Claim

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `C1` … |
| `statement` | string | Self-contained sentence (rules in §8) |
| `type` | enum | `descriptive` · `associational` · `causal` · `predictive` · `interpretive` · `formal` · `methodological` (choosing: §3.3.1) |
| `role` | enum | `principal` · `supporting` · `background` |
| `novelty` | enum | `new` · `replication` · `extension` · `refutation` · `restatement` |
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
| `verification` | enum | Formal claims proved or argued in this paper, required: `machine_checked` · `agent_checked` · `peer_checked` · `unchecked` (§5) |
| `verified_by` | string | Required for `agent_checked` and `peer_checked`: the contributor ID of the checker |
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

### 3.4 Evidence

Common fields: `id`, `profile`, `description`, `artefacts` (list of artefact IDs), `data_origin` (`primary` · `secondary`), `preregistered` (boolean), `availability` (§3.10), `models` (list, below), `results` (list, below), `execution` (object, below).

**Results** (`results`, quant and simulation profiles): one entry per reported number, so readers can query numbers instead of parsing prose. `{id, quantity, value, unit, interval, conditions, artefact}`
- `id`: `R1`, `R2`, … unique within the evidence object; referenced from elsewhere as `<evidence id>.<result id>`, e.g. `E23.R2`.
- `quantity`: what was measured, e.g. `"largest KL divergence per observation, mixture vs exact"`.
- `value`: number. `unit`: optional. `interval`: optional `{level, lower, upper}`.
- `conditions`: object of the settings that define this result, e.g. `{"distribution": "Gamma", "form": "conventional"}`.
- `artefact`: optional artefact ID of the table holding the full results.
A single-estimate study may use `estimand`, `estimate` and `interval` instead.

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
| `kind` | enum | `code` · `data` · `protocol` · `proof` · `model` · `software` · `figure` · `table` · `prompt` · `trace` · `manuscript` · `other` |
| `uri` | string | |
| `version` | string | Version or content hash |
| `licence` | string | SPDX identifier; recommended |
| `availability` | object | §3.10 |
| `description` | string | Required for `figure` and `table`: what it shows, self-contained |
| `underlying_data` | artefact ID | For `figure`: the data plotted |

Tables are stored as data (CSV or JSON), not images. Prompts are stored verbatim, including system prompts. A `trace` is the log of an agent's run (prompts, tool calls, outputs) as recorded by the system that ran it; it may contain confidential material and uses the availability levels in §3.10. A `manuscript` is a human-readable version of the same work (e.g. a LaTeX or PDF draft) from which the claimset was prepared.

### 3.7 Relations (cross-paper)

Links from a claim to a claim or work outside this paper: `{type, target, id_type, locator, archive_uri, self}`.

- `type`: `supports` · `contradicts` · `extends` · `refines` · `replicates` · `fails_to_replicate` · `supersedes`.
- `target`: a global claim ID when the target is a claimset; otherwise the identifier of the work (paper, book, software release).
- `id_type`: `claimset` · `doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `physical`.
- `locator`: optional; where in the target the related result is, e.g. `"eq. 29 and B.6"`, `"Section 4.2"`, `"function adam(), lognormal scale"`.
- `archive_uri`: required for `url` targets: an archived snapshot.
- `self`: true when the target work shares a human supervisor, guarantor or contributor with this paper.

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
- On `empirical_quant` and `simulation`: optional `results`, the specific results the claim reports (e.g. `{"kind": "simulation", "evidence": "E20", "results": ["E20.R1", "E20.R2"]}`). Recommended whenever the statement quotes numbers, so readers can check each number against the structured result.
- On `cited`: `target`, `id_type` (`doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `physical`), `archive_uri` (required for `url`: an archived snapshot), `self` (boolean, below). Where a claimset of the cited work exists, `target` may be its global claim ID.
- `self` is true when the cited work shares any human supervisor, guarantor or contributor with this paper, or (for converted papers) any source author. Shared agent systems are recorded but computed by the platform, not flagged by the file.
- Meta-analyses: performed in the paper → `empirical_quant` with `data_origin: secondary`; cited → `cited`.

## 5. Strength and verifiability

Two separate dimensions. Confidential evidence is less verifiable, not necessarily weaker.

**Strength** is a list of assessments, so multiple parties can assess the same claim and disagreements stay visible: `{by, reviewer_type, strength, rationale, source_span}`
- `by`: `author` · `converter` · `reviewer`. In converted papers, `author` records only a strength the source itself states (e.g. "weak evidence suggests"), and `source_span` is required; all other assessments of a converted claim are `converter` or `reviewer`.
- `reviewer_type`: `ai` · `human` (required for `reviewer`)
- `strength`: `high` · `moderate` · `low` · `very_low` (ordinal, after GRADE / GRADE-CERQual; no numeric confidence)
- Not applicable to assumption claims or formal claims.

**Verification** replaces strength for formal claims (`type: formal`). It is required when the paper itself proves or argues the claim (support `formal_proof` or `argument`); a formal result cited as background relies on the cited work and carries no verification:
- `machine_checked`: a proof assistant checked the proof; the formal evidence has `proof_artefact` and `checker`.
- `agent_checked`: an agent that did not author the claim re-derived it step by step; `verified_by` names that agent, listed as a `contributor`.
- `peer_checked`: a human checked it; `verified_by` names that human.
- `unchecked`: no independent check.
The platform may add its own independent checks; these are stored as platform assessments (§6), not in the file.

**Verifiability** is computed by the platform from the availability of the evidence, artefacts and models the claim's support points to. Submissions may omit it; if supplied, it must match the platform's computation.
- `open`: everything needed to check the claim is public or platform-available; formal claims are machine-checked or have full proof steps.
- `audit_only`: checking requires `audit` or `author_only` material.
- `unverifiable`: rests on `unavailable` material, on `expert_judgement` alone, or on `unstated` support.
- `not_applicable`: assumption claims; background claims supported only by `consensus`.

**Verifiability flags** (computed, list):
- `api_model_dependency`: the supporting evidence relies on a model with `weights: api_only`; the result may not be reproducible once the model is withdrawn.
- `figure_without_data`: a supporting figure has no `underlying_data`.
- `unpublished_prompts`: a model used in the research design has no `prompts` artefact.
- `unrecorded_model_settings`: a model used in the research design has no `settings`, or its `record_source` is absent or `self_reported`.
- `unstated_support`: the claim's support is `unstated`.

The flags concern models used to produce evidence, not the model that wrote the paper. The platform recomputes verifiability and flags as availability changes (e.g. when an API model is retired).

## 6. Assessment by the platform

Platform assessments (AI reviewer agents, audits, replication checks, conversion reviews, independent re-derivation of formal claims, scope-consistency reviews) are stored in the database, not the file. Human review is an optional layer, except for sampled review of conversions.

## 7. Identifiers and source trust

The platform mints a DOI for every published claimset. The file records identifier type and venue facts for cited works only. Trust in a source is a derived score computed by the platform. Papers published on Claimset are scored by the same rules as all others.

## 8. Statement text rules

1. One assertion per statement.
2. **Granularity rule:** split a statement into separate claims when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one interpretive claim with a warrant; sub-themes become separate supporting claims only if they have their own evidence.
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
4. `role: background` ⇒ `novelty: restatement` ⇒ support kinds limited to `cited` and/or `consensus`.
5. `contributions` lists only `principal` claims; every `principal` claim appears in `contributions`.
6. `type: interpretive` ⇒ `warrant` present.
7. `formal_proof` support ⇒ `verification` present; evidence has `foundation` and either `steps` or `proof_artefact`.
8. `depends_on` contains no cycles.
9. `provenance` or `field_provenance` values `extracted`/`inferred` only when `paper.origin: converted`; `source_spans` required in that case.
10. Each support kind carries its required pointer (§4).
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
31. `paper.origin: converted` ⇒ `converted_from.source_licence` permits derivatives and redistribution (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0` or equivalents), and `paper.licence` equals the source licence or a compatible one.
32. Every human guarantor has confirmed consent through their ORCID-linked account, and every organisation guarantor through an authorised account.

### 10.5 Rules added in v0.9 (errors: file rejected)

33. `novelty: refutation` ⇒ at least one relation of type `contradicts` or `fails_to_replicate`; `novelty: replication` ⇒ at least one relation of type `replicates` or `fails_to_replicate`.
34. Every relation has `type`, `target` and `id_type`; `id_type: url` ⇒ `archive_uri`.
35. `type: formal` ⇒ no `assessments`; `type: formal` with `formal_proof` or `argument` support ⇒ `verification` present.
36. `verification: agent_checked` ⇒ `verified_by` is an agent contributor that does not hold `author`; `peer_checked` ⇒ `verified_by` is a human contributor; `machine_checked` ⇒ the formal evidence has `proof_artefact` and `checker`.
37. Formal evidence has no `theorem` field.
38. Result IDs are unique within their evidence object; every `results` reference in a support entry (`E<n>.R<m>`) resolves to a result of the evidence that entry points to.

### 10.4 Warnings (file accepted, warning recorded)

1. A `recommended` field in §3.11 or §3.12 is missing.
2. An agent's `record_source` is absent or `self_reported`.
3. `paper.contribution_record` is `partial` or `unknown`.
4. `autonomy: human_supervised` without a `supervisor`, or `autonomy: human_assisted` without a human `contributor`.
5. No `paper.classification.keywords` in a submission.
6. Three or more claims share an identical `scope` object (often a sign of scope copied by section; check each claim).
7. `novelty: extension` without an `extends` relation.
8. Quant or simulation evidence with neither `estimate` nor `results`.
9. An artefact without `licence`.
10. An agent contributor that is not an `author` has the AI task `writing`; agents that produced the paper's content are authors.

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
  "schema_version": "0.9",
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
      "type": "descriptive", "role": "background", "novelty": "restatement",
      "support": [{"kind": "consensus"}], "provenance": "author_stated"
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
      "assumptions": ["C3"],
      "limitations": ["L1"],
      "assessments": [{"by": "author", "strength": "moderate", "rationale": "Single benchmark collection, no external replication."}],
      "depends_on": ["C1"], "provenance": "author_stated"
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
  "schema_version": "0.9",
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
      "type": "associational", "role": "supporting", "novelty": "restatement",
      "support": [{"kind": "unstated"}],
      "assessments": [{"by": "converter", "strength": "very_low", "rationale": "Asserted in the source without data or citation."}],
      "verifiability": "unverifiable", "verifiability_flags": ["unstated_support"],
      "source_spans": ["sec2:p4"], "provenance": "extracted"
    }
  ]
}
```

Appendix C is abbreviated: the principal claim C1 and the remaining sections are omitted.
