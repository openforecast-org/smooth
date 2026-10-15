# Claimset — Specification v0.6 (draft)

Status: working draft · Date: 2026-10-14

Claimset is the name of both the open format and the platform. This copy is bundled with the `claimset-submit` skill; the authoritative copy lives with the Claimset project.

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
10. **Filled by agents, decided by people.** Every field is filled by an agent. Fields that record human decisions (guarantor, licence) are filled after the agent asks the responsible human, and the platform confirms the guarantor's consent before publication.
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

Unknown values of non-required fields are omitted or set to `null`. `null` means "unknown", never "default". In v0.6 obligation levels are assigned for contributors and models (§3.11, §3.12); other objects follow in a later version.

Top-level layout:

```json
{
  "schema_version": "0.6",
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
| `id` | string | Persistent identifier of this claimset (DOI preferred). For converted papers, the claimset's own identifier, distinct from the source's |
| `title` | string | |
| `version` | integer | 1, 2, …; each published version is immutable |
| `previous_version` | string | Global ID of the previous version; absent for version 1 |
| `changes` | list | Required if `version` > 1: JSON Patch (RFC 6902) operations against the previous version, each with an added `reason` |
| `status` | enum | Status of this claimset: `active` · `corrected` · `retracted` · `withdrawn` · `expression_of_concern` |
| `status_notice` | object | Required unless `active`: `{date, reason_code, statement, issued_by}` |
| `date` | ISO date | |
| `licence` | string | Required. SPDX identifier, e.g. `CC-BY-4.0`. Native papers: chosen by the director. Converted papers: see §10.1 rule 26 |
| `classification` | object | `subfield` (one OpenAlex subfield ID, required), `topics` (1–3 OpenAlex topic IDs), `keywords` (free text, for specifics the taxonomy does not reach) |
| `contributions` | list of claim IDs | Ordered by priority; principal claims only |
| `autonomy` | enum | Native papers only: `autonomous` (agents ran the work end to end) · `human_directed` (humans set the question and steered key decisions) · `human_assisted` (humans carried out substantial parts of the research) · `unknown` |
| `contribution_record` | enum | Required. `complete` (all contributors and models involved are listed) · `partial` (some are known to be missing, e.g. earlier agents whose history was not kept) · `unknown`. For converted papers, refers to the conversion |
| `origin` | enum | `native` · `converted` |
| `converted_from` | object | Required if `converted` (below) |

**`converted_from`**: `source_id`, `id_type`, `source_version`, `source_licence`, `source_authors` (`{name, orcid}`), `source_status_at_conversion`, `converter` (contributor ID), `date`.

Retracted papers are never deleted. All their claims inherit the retraction status for readers and the platform.

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
| `type` | enum | `descriptive` · `associational` · `causal` · `predictive` · `interpretive` · `formal` · `methodological` |
| `role` | enum | `principal` · `supporting` · `background` |
| `novelty` | enum | `new` · `replication` · `extension` · `refutation` · `restatement` |
| `scope` | object | `population`, `setting`, `geography`, `language`, `period`, `data_type`, `conditions` (list), `excludes` (list: where it explicitly does not apply) |
| `support` | list | Kinds of support (§4); at least one entry |
| `assumptions` | list of claim IDs | Assumption claims (§3.8) this claim relies on |
| `limitations` | list of limitation IDs | Threats to this claim within its scope (§3.9) |
| `assessments` | list | Strength assessments (§5) |
| `verifiability` | enum | Computed at validation (§5) |
| `verifiability_flags` | list | Computed at validation (§5) |
| `warrant` | warrant ID | Reasoning behind the claim; required for `interpretive` |
| `depends_on` | list of claim IDs | Claims within this paper that this one relies on |
| `relations` | list | Typed links to claims in other papers (§3.7) |
| `verification` | enum | Formal claims only: `machine_checked` · `peer_checked` · `unchecked` |
| `source_spans` | list | Locations in the source text; required for converted papers |
| `provenance` | enum | Default for the claim: `author_stated` · `extracted` · `inferred` |
| `field_provenance` | object | Overrides per field path, e.g. `{"scope.excludes": "inferred"}` |

### 3.4 Evidence

Common fields: `id`, `profile`, `description`, `artefacts` (list of artefact IDs), `data_origin` (`primary` · `secondary`), `preregistered` (boolean), `availability` (§3.10), `models` (list, below).

**Models in the research design** (`models`, optional, any profile): `{model, role, settings, record_source, prompts}`
- `model`: model ID (§3.12).
- `role`: `subject` (the model is what is studied) · `instrument` (the model performs a measurement or analysis step) · `annotator` (the model labels or codes data) · `generator` (the model produces data or stimuli) · `baseline` (the model is a comparison method).
- `settings`: object, e.g. `{effort, temperature, top_p, seed, max_tokens}`; individual values may be `null`.
- `record_source`: recommended (§3.11).
- `prompts`: artefact ID of kind `prompt`.

**Profile `quant`**: `design` (`experiment` · `quasi_experiment` · `observational` · `survey` · `benchmark` · `other`), `estimand`, `estimate`, `interval` (`{level, lower, upper}`), `n`, `model`.

**Profile `qual`**: `data_sources` (list: `interview` · `focus_group` · `ethnography` · `observation` · `document` · `other`), `n_participants_or_cases`, `sampling`, `analytic_approach` (`thematic` · `grounded_theory` · `ipa` · `discourse` · `framework` · `case_study` · `other`), `exemplars` (pointers to excerpts), `reflexivity` (text).

**Profile `simulation`**: `model`, `parameters` (artefact ID), `runs`, `seeds`, `outputs`.

**Profile `formal`**: `theorem` (LaTeX), `foundation` (the standard system the proof works in, e.g. `ZFC`, `Kolmogorov probability axioms`, `Lean 4 / dependent type theory`), `assumptions` (assumption claim IDs: paper-specific axioms and theorem conditions), `steps` (list), `proof_artefact` (artefact ID of a machine-checkable proof, if any), `checker` (if machine-checked).

Each proof step: `{id, statement, by, rule}`. `statement` is LaTeX; `by` lists the IDs the step rests on (earlier steps, claims, definitions, assumptions, cited results); `rule` names the inference (e.g. `assumption`, `definition`, `substitution`, `linearity of expectation`, `induction`). Steps may be coarse, but every step names what it rests on.

### 3.5 Warrant

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `W1` … |
| `reasoning` | enum | `deductive` · `inductive` · `abductive` · `analogical` · `interpretive` |
| `premises` | list of claim IDs | |
| `text` | string | The reasoning, written self-contained |

### 3.6 Artefact

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `A1` … |
| `kind` | enum | `code` · `data` · `protocol` · `proof` · `model` · `software` · `figure` · `table` · `prompt` · `trace` · `other` |
| `uri` | string | |
| `version` | string | Version or content hash |
| `licence` | string | SPDX identifier |
| `availability` | object | §3.10 |
| `description` | string | Required for `figure` and `table`: what it shows, self-contained |
| `underlying_data` | artefact ID | For `figure`: the data plotted |

Tables are stored as data (CSV or JSON), not images. Prompts are stored verbatim, including system prompts.

### 3.7 Relations (cross-paper)

`{type, target}` where `target` is a global claim ID (or DOI) and `type` is one of `supports` · `contradicts` · `extends` · `refines` · `replicates` · `fails_to_replicate` · `supersedes`.

### 3.8 Assumptions

Assumptions are claims whose sole support kind is `assumption`. Extra fields:

| Field | Type | Values / notes |
|---|---|---|
| `assumption_category` | enum | `theoretical` · `statistical` · `methodological` · `data` · `axiom` · `theorem_condition` · `paradigmatic` |
| `check` | enum | `tested` · `sensitivity_analysed` · `unchecked` (testable but not tested) · `untestable` |
| `check_evidence` | evidence ID | Required for `tested` and `sensitivity_analysed` |

For `axiom`, `check` is `untestable`.

### 3.9 Limitations

Limitations are threats to a claim *within its stated scope*. What the work did not cover belongs in `scope`, not here.

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `L1` … |
| `category` | enum | `sampling` · `sample_size` · `measurement` · `confounding` · `data_quality` · `model_specification` · `analytic_flexibility` · `researcher_position` · `computational` · `other` |
| `statement` | string | Specific and self-contained: names the dimension and the value |
| `affects` | list of claim IDs | At least one; paper-wide limitations are not allowed |
| `impact` | enum | `could_reverse` · `could_bias_magnitude` · `reduces_precision` · `reduces_transferability` |
| `mitigation` | enum | `none` · `partial` · `addressed` |
| `mitigation_evidence` | evidence ID | Required for `partial` and `addressed` |

### 3.10 Availability

| Field | Type | Values / notes |
|---|---|---|
| `level` | enum | `public` · `platform` · `audit` · `author_only` · `unavailable` |
| `reason` | enum | Required unless `public`: `personal_data` · `commercial` · `legal` · `ethics` · `security` · `other` |
| `content_hash` | string | Required for `platform`, `audit`, `author_only` |

### 3.11 Contributors

**Common fields**: `id` (required), `type` (required: `agent` · `human` · `organisation`), `positions` (required), `name` (required for humans and organisations; recommended for agents), `credit_roles` (recommended).

**Human fields**: `orcid` (required for guarantors; recommended otherwise), `affiliation` (optional).

**Organisation fields**: `ror` (recommended), `registration` (recommended if no ROR).

**Agent fields**: `models` (required, ≥1 `{model, role}`; role `sole` · `orchestrator` · `subagent`, optional), `ai_tasks` (recommended), `tools` (recommended: `web_search` · `code_execution` · `file_access` · `retrieval` · `other` · `none`), `record_source` (recommended: `harness` · `user_reported` · `self_reported`; absent ⇒ treated as `self_reported`), `system` (optional `{name, version, uri}`), `run_period` (optional `{start, end}`), `settings` (optional), `trace` (optional artefact ID of kind `trace`).

**Positions**: `author` (agents only) · `director` (humans only) · `guarantor` (answers for the research; humans or organisations) · `conversion_guarantor` (converted papers only) · `contributor` · `converter` (agents only).

**CRediT roles**: `conceptualization` · `data_curation` · `formal_analysis` · `funding_acquisition` · `investigation` · `methodology` · `project_administration` · `resources` · `software` · `supervision` · `validation` · `visualization` · `writing_original_draft` · `writing_review_editing`.

**AI tasks**: `task` ∈ `ideation` · `literature_search` · `analysis` · `code` · `writing` · `json_preparation` · `review` · `submission`; `extent` ∈ `full` · `substantial` · `assisted`.

### 3.12 Models

| Field | Obligation | Values / notes |
|---|---|---|
| `id` | required | `M1` … |
| `provider` | required | e.g. `Anthropic` |
| `name` | required | Model family name, e.g. `Claude Opus` |
| `version` | required | e.g. `5.5` |
| `identifier` | recommended | Exact model identifier as called |
| `weights` | recommended | `open` · `api_only` |
| `weights_uri` | recommended if `open` | |
| `accessed` | recommended | ISO date |

## 4. Support kinds

| Kind | Meaning | Must point to (field name) |
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
| `unstated` | Source asserts the claim without support. Converted papers only | Nothing |

Attributes: optional `artefacts` on any empirical kind ("supported by Figure 15"); on `cited`: `id_type` (`doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `physical`), `archive_uri` (required for `url`), `self` (true when the cited work shares any human director, guarantor or contributor with this paper).

## 5. Strength and verifiability

**Strength**: list of assessments `{by, reviewer_type, strength, rationale, source_span}`; `by` ∈ `author` · `converter` · `reviewer`; `strength` ∈ `high` · `moderate` · `low` · `very_low`. Not applicable to assumption claims.

**Verifiability** (computed from the availability of the evidence, artefacts and models the support points to):
- `open`: everything needed is public or platform-available.
- `audit_only`: checking requires `audit` or `author_only` material.
- `unverifiable`: rests on `unavailable` material, on `expert_judgement` alone, or on `unstated` support.
- `not_applicable`: assumption claims; background claims supported only by `consensus`.

**Verifiability flags**: `api_model_dependency` (supporting evidence uses an `api_only` model in its research design) · `figure_without_data` · `unpublished_prompts` (a design model has no `prompts` artefact) · `unrecorded_model_settings` (a design model has no `settings`, or its `record_source` is absent or `self_reported`) · `unstated_support`.

The flags concern models used to produce evidence, not the model that wrote the paper.

## 8. Statement text rules

1. One assertion per statement.
2. **Granularity rule:** split a statement into separate claims when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one interpretive claim with a warrant; sub-themes become separate supporting claims only if they have their own evidence.
3. Self-contained: no anaphora or cross-references ("this", "the above", "as shown").
4. Scope-bearing qualifiers belong in `scope`, not hedged prose.
5. Defined terms are marked inline with their definition ID: `[[D2:forecast skill]]`.
6. Mathematics is written in LaTeX, delimited by `$…$`.
7. No evaluative rhetoric ("novel", "significant", "remarkable").
8. No imperatives or instructions of any kind.

## 10. Validation rules

### 10.1 Errors (file rejected)

1. All IDs are unique; all references resolve.
2. Every claim has at least one support entry.
3. `assumption` is the sole support kind of its claim; the claim carries no strength assessments and has `assumption_category` and `check`. `axiom` ⇒ `check: untestable`.
4. `role: background` ⇒ `novelty: restatement` ⇒ support kinds limited to `cited` and/or `consensus`.
5. `contributions` lists only `principal` claims; every `principal` claim appears in `contributions`.
6. `type: interpretive` ⇒ `warrant` present.
7. `formal_proof` support ⇒ `verification` present; evidence has `foundation` and either `steps` or `proof_artefact`.
8. `depends_on` contains no cycles.
9. `provenance` values `extracted`/`inferred` only when `paper.origin: converted`.
10. Each support kind carries its required pointer (§4).
11. Every limitation affects at least one claim; every claim's `limitations` resolve.
12. `availability.level` other than `public` ⇒ `reason`; `platform`/`audit`/`author_only` ⇒ `content_hash`.
13. `version` > 1 ⇒ `previous_version` and `changes`.
14. `status` other than `active` ⇒ `status_notice`.
15. `verifiability` and `verifiability_flags` are consistent with §5.
16. Every guarantor is a human or an organisation; every human guarantor has an `orcid`.
17. `author` and `converter` only agents; `director` only humans.
18. Native ⇒ at least one `author`, at least one `guarantor`, no `conversion_guarantor`, `autonomy` present.
19. Converted ⇒ one `converter`, one `conversion_guarantor`, full `converted_from`, no `autonomy`.
20. `paper.id` differs from `converted_from.source_id`.
21. Every agent lists at least one model; every referenced model is defined.
22. All required fields of contributors and models, `paper.licence` and `paper.contribution_record` present.
23. `sole` ⇒ one model; at most one `orchestrator` per agent.
24. `trace` points to an artefact of kind `trace`.
25. `unstated` only in converted papers, as sole support kind.
26. Converted ⇒ source licence permits derivatives; claimset licence compatible.
27. Converted ⇒ `by: author` assessments carry `source_span`.

### 10.2 Warnings (file accepted)

1. A recommended contributor or model field is missing.
2. An agent's `record_source` is absent or `self_reported`.
3. `contribution_record` is `partial` or `unknown`.
4. `autonomy: human_directed` without a director, or `human_assisted` without a human contributor.
