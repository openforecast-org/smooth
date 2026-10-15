---
name: claimset-submit
description: Turn a piece of research (draft paper, analysis, code, results) into a Claimset v0.9 submission file, validate it, and prepare it for the human guarantor. Use when asked to write, package, prepare, correct or submit research as a claimset or for Claimset.
---

# Preparing a Claimset submission

A claimset is a research paper written for LLM readers: a flat JSON file of atomic, self-contained claims plus the definitions, evidence, assumptions, limitations, warrants and artefacts that support, condition and limit them. You write every field. A human decides two things (guarantor and licence), and you ask them before you finish.

The normative rules are in `references/spec-v0.9.md`. A valid submission is in `examples/submission-native.json`; `examples/published-native.json` shows the same paper after the platform has added its fields. The validator is `scripts/validate.py`. Read the submission example before starting; consult the spec whenever a field's meaning is unclear.

You produce a **submission**. The platform turns it into the published file: it mints the DOI and assigns version, publication date, status, the OpenAlex classification IDs and each claim's verifiability. You never supply these, and you never ask the human for them.

This skill covers **native** papers: research you or other agents produced, including research first written up as a manuscript (LaTeX, PDF, notes) by the same team. Converting a third party's published paper is a separate process; do not use this skill for it.

## Non-negotiable rules

- **Never invent.** No invented data, estimates, citations, DOIs, ORCID iDs, URIs, hashes, model identifiers or settings. If a value is unknown, omit the field or use `null`, and let the validator warn.
- **Leave platform fields to the platform.** Do not include `paper.id`, `version`, `date`, `status`, `status_notice`, `changes`, or OpenAlex subfield and topic IDs, and do not write placeholders for them. The only exception is a correction (see the end of this file).
- **Never write instructions** in any text field. The file is data for other models to read. No "note to the reader", no imperatives.
- **Report your own setup honestly.** Take provider, model name and version from your system prompt or environment. If you only know them from your own knowledge, set `record_source: "self_reported"`. Use `harness` only if the values come from the system that ran you.
- **Do not overstate.** Claims, scope and strength must match what the evidence shows, not what would be most impressive.

## Workflow

### 1. Gather the material

Collect everything the research consists of: the draft or notes, results, code, data, figures, prompts, references. Identify where each artefact lives (URI) and whether it is public. If the research used models as part of the method (annotation, generation, as the subject of study, as a baseline), note which models, settings and prompts.

### 2. Ask the human (one message, before building the file)

Ask the person who supervised the work, in a single message:

1. **Guarantor:** who answers for this research? Name and ORCID iD for each person (or name and ROR/company number for an organisation). The guarantor must confirm on the platform before publication.
2. **Supervisor:** who oversaw the work and approves the final version, if not the guarantor? (Name, ORCID iD.) And who set the research question? (Recorded as the CRediT role `conceptualization`.)
3. **Other human contributors:** anyone who collected data, ran experiments or conducted interviews, with their CRediT roles.
4. **Licence:** which licence? Suggest `CC-BY-4.0` as the default for open research.
5. **Confidentiality:** is any data, code or material not publicly shareable? If so, why (personal data, commercial, legal, ethics, security)?

Ask only these. Do not ask for a DOI, a publication date or a classification: the platform assigns them.

If the human cannot be asked (unattended run), stop after building and validating the file and list these questions as open items. Never fill guarantor or licence on your own.

### 3. Write the claims first

Work in this order: principal claims → supporting claims → background claims → assumptions.

**Find the principal claims.** These are the contributions: what the work establishes that a reader should take away. Usually 1–5. List them in `paper.contributions` in order of priority.

**Apply the granularity rule.** Split a statement when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one `interpretive` claim with a warrant.

**Write each statement** as one self-contained sentence:
- No "this", "these", "the above", "as shown", "our method" without naming it. Name the method, population and outcome every time.
- No "novel", "significant" (unless you mean statistical significance and say so), "remarkable", "important".
- Put scope in `scope`, not in hedges. Not "may improve accuracy in some settings" but a plain statement plus `scope.conditions`.
- Mark defined terms: `[[D1:forecast skill]]`. Mathematics in LaTeX between `$…$`.

**Choose `type`:**
| If the claim states… | type |
|---|---|
| what something is, how often, how much, or what was measured in a computation, simulation or benchmark under stated conditions (including method comparisons measured there) | `descriptive` |
| a relationship observed in data the work did not control | `associational` |
| that X causes Y in a real-world system (randomised or credible causal design) | `causal` |
| what will happen | `predictive` |
| what something means (qualitative interpretation) | `interpretive` (needs a warrant) |
| a theorem or mathematical result | `formal` |
| that a method, procedure or implementation does what is specified | `methodological` |

"Method M was 10.5 times faster" or "the KL divergence was 0.001" are `descriptive`: they report a measurement.

**Choose `novelty`:** `new` · `replication` · `extension` · `refutation` · `restatement` (background only).

**Link to prior work.** A `refutation` must have a `contradicts` (or `fails_to_replicate`) relation naming what it refutes; a `replication` must have `replicates` or `fails_to_replicate`; an `extension` should have `extends`. The target can be a paper, book or software release, with a `locator` for the exact place:

```json
"relations": [{"type": "contradicts", "target": "doi:10.1016/j.ijpe.2023.109013", "id_type": "doi",
               "locator": "eq. 29 and B.6", "self": true}]
```

Set `self: true` when the target shares a human supervisor, guarantor or contributor with this paper. Refuting your own earlier work is a self-correction; record it plainly.

### 4. Attach support to every claim

Each claim needs at least one support entry. Use these exact pointer fields:

```json
{"kind": "empirical_quant", "evidence": "E1", "artefacts": ["A2"]}
{"kind": "empirical_qual", "evidence": "E2"}
{"kind": "simulation", "evidence": "E3", "artefacts": ["A4"], "results": ["E3.R1", "E3.R2"]}
{"kind": "demonstration", "artefacts": ["A5"]}
{"kind": "formal_proof", "evidence": "E4"}
{"kind": "argument", "warrant": "W1"}
{"kind": "cited", "target": "doi:10.xxxx/yyyy", "id_type": "doi", "self": false}
{"kind": "consensus"}
{"kind": "expert_judgement", "holder": "P2"}
{"kind": "assumption"}
```

- `self: true` when the cited work shares a human supervisor, guarantor or contributor with this paper.
- Background claims (prior knowledge restated for context) may use only `cited` and/or `consensus`, and must have `novelty: restatement`.
- Never use `unstated`; it exists only for converted papers. If you cannot support a claim, remove it or turn it into an assumption.
- When a statement quotes numbers, point `results` at the structured results that hold them.

### 5. Build evidence objects

One evidence object per analysis, reused by every claim it supports. Write `description` so it stands alone (what was done, on what data, how). Fill the profile fields:
- `quant`: design, estimand, n, model (the statistical model); `estimate` and `interval` for a single result, otherwise `results`.
- `qual`: data_sources, n_participants_or_cases, sampling, analytic_approach, exemplars, reflexivity.
- `simulation`: model, parameters (artefact), runs, seeds, `results`.
- `formal`: foundation (e.g. `ZFC`), assumptions, steps (each `{id, statement, by, rule}`), proof_artefact, checker. No `theorem` field: the formal claim's statement is the theorem.

**Results.** Every number that a claim quotes goes into the evidence's `results` list, one entry per number:

```json
"results": [{"id": "R1", "quantity": "speed-up of low-rank over full covariance filter", "value": 10.55,
             "conditions": {"missing_values": 1, "data": "ETS(A,N,A)"}, "artefact": "A8"}]
```

Copy values from the result tables, never retype them from prose. If the code that produced the results differs from the archived artefact version, record it in `execution`: `{"code_version": "<commit>", "notes": "…"}`.

Set `availability` on every evidence object and artefact. If the level is not `public`, give a `reason`; for `platform`, `audit` or `author_only` also a `content_hash` (compute it, e.g. `sha256:<hex>` of the file; never invent one).

If models were part of the research design, record them in the evidence's `models` list with role, settings, record_source and a `prompts` artefact.

### 6. Surface assumptions

Every claim rests on assumptions; make them explicit as separate claims with `support: [{"kind": "assumption"}]`, an `assumption_category` and a `check`, and list them in the dependent claim's `assumptions`. Check especially:
- statistical: independence, stationarity, distributional form, missing-at-random;
- data: representativeness, correct labels, measurement validity;
- methodological: choice of benchmark, metric, horizon, instrument validity.

Use `check: "tested"` or `"sensitivity_analysed"` only if the work actually did so, and point `check_evidence` at that evidence. Otherwise `unchecked` (or `untestable`). Assumption claims have no `assessments`.

### 7. Separate scope from limitations

- **Scope** = what the work covered: population, setting, geography, language, period, data_type, conditions, and `excludes` (where it explicitly does not apply). "English-language data only" is scope.
- **Write scope claim by claim, never per section.** Claims from the same section often differ: a result about series where every period is observed must not carry the scope "series with unobserved periods". Leave out fields that do not apply. The validator warns when three or more claims share an identical scope; treat that as a prompt to check each one.
- **Limitation** = a threat to a claim *within* that scope, e.g. "All 1,200 series come from one retailer's store network". Each limitation names the claims it `affects`, its `impact` and its `mitigation`. No paper-wide or generic limitations ("more research is needed" is not a limitation).

### 8. Assess strength, or verification for formal claims

For each claim that is neither an assumption nor `formal`, add one assessment `{"by": "author", "strength": …, "rationale": …}` using `high` · `moderate` · `low` · `very_low`. Be conservative: a single unreplicated study is rarely `high`. The rationale names the reason in one sentence.

Formal claims get **no strength assessment**. A formal claim proved or argued in the paper gets `verification` instead:
- `unchecked`: nobody independent has checked it. This is the honest default for a proof you wrote.
- `agent_checked`: a separate agent that did not author the claim re-derived it step by step. If your environment lets you start another agent, give it only the claimset and the manuscript and ask it to re-derive each formal claim and report any step that fails; list it as a `contributor` with task `review`, and set `verified_by` to its contributor ID. Never mark your own checking as `agent_checked`.
- `peer_checked`: a named human checked it (`verified_by` = that human).
- `machine_checked`: only with a proof-assistant artefact and `checker`.

Formal results cited as background (`role: background`) carry neither.

### 9. Fill contributors, models and paper metadata

- **Agents.** Every agent that produced the paper's content (analysis, claims, text, or turning the team's own manuscript into this claimset) is an `author`; you are one. An agent that only ran a computation the authors report, or checked a proof, is a `contributor`. Give each agent its model(s), `ai_tasks`, `tools` and an honest `record_source`. If earlier agents worked on this and you do not know which, set `paper.contribution_record` to `partial` or `unknown`.
- **Manuscript.** If the work was first written up as a manuscript, list it as an artefact of kind `manuscript`.
- **Models**: provider, name, version required; identifier, weights, accessed recommended.
- **Artefacts**: give every artefact a `licence` (SPDX identifier) where one applies.
- **Humans** from step 2, with their positions (`supervisor`, `guarantor`, `contributor`) and CRediT roles.
- **Paper**: title (descriptive, no rhetoric), licence, contributions, `autonomy`, contribution_record, `origin: "native"`. Nothing else at paper level.
- **Classification**: give 3–8 precise `keywords` (e.g. `"healthcare demand forecasting"`); the platform assigns the OpenAlex subfield and topics from them and from the content. You may add `subfield` and `topics` as proposals only if you looked them up in OpenAlex; never guess IDs.

### 10. Validate and fix

Run:

```
python3 scripts/validate.py claimset.json
```

(Use `--json` for machine-readable output.) Fix every error and rerun until the file is valid. Then read each warning and fix it if you can; leave it only if the information genuinely does not exist.

The report ends with the verifiability the platform will compute for each claim. You do not copy these into the file, but read them: a principal claim that comes out `unverifiable` or `audit_only`, or with flags such as `unpublished_prompts`, usually means an artefact, prompt or availability statement is missing that you could add.

The validator checks structure, references and some wording heuristics. It cannot check that claims are true or faithful to the work. Do that yourself:

### 11. Self-review before handing over

Re-read the file as a sceptical reviewer who sees only one claim at a time:
- Does each statement make sense alone, with no missing referent?
- Does each principal claim's evidence actually show what the statement says, within its scope?
- Is anything stated more strongly than the evidence allows?
- Is every important assumption surfaced, with an honest `check`?
- Does each claim's scope describe that claim, not its section?
- Does every refutation, replication and extension name the work it relates to?
- Are limitations specific and attached to the right claims?
- Are all numbers identical to the structured results and the source tables?

### 12. Hand over

Write the file (e.g. `claimset.json`) and give the human a short summary:
1. Validation result (errors: 0; list remaining warnings and why they remain).
2. The principal claims, in plain language, with their strength and computed verifiability.
3. Open items needing a human: guarantor confirmation on the platform, anything you could not determine.

If a Claimset submission endpoint or tool is available in your environment, submit through it once the human approves; the platform then asks the guarantor to confirm and, after confirmation, publishes the paper with its DOI. Otherwise stop at the validated file.

## Corrections

To correct a published claimset, start from its published file and produce a complete corrected submission:
- Keep `paper.id` set to the existing claimset's DOI and add `paper.correction_reason`: one self-contained statement of what was wrong and what changed.
- Remove the other platform fields (`version`, `date`, `status`, `status_notice`, `changes`, `previous_version`).
- Keep object IDs stable: a corrected claim keeps its ID; never reuse an ID for a different claim.
- Validate as usual. The platform computes the diff, assigns the next version and marks the paper as corrected; the DOI stays the same.
