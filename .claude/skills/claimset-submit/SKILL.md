---
name: claimset-submit
description: Turn a piece of research (draft paper, analysis, code, results) into a Claimset v0.8 submission file, validate it, and prepare it for the human guarantor. Use when asked to write, package, prepare, correct or submit research as a claimset or for Claimset.
---

# Preparing a Claimset submission

A claimset is a research paper written for LLM readers: a flat JSON file of atomic, self-contained claims plus the definitions, evidence, assumptions, limitations, warrants and artefacts that support, condition and limit them. You write every field. A human decides two things (guarantor and licence), and you ask them before you finish.

The normative rules are in `references/spec-v0.8.md`. A valid submission is in `examples/submission-native.json`; `examples/published-native.json` shows the same paper after the platform has added its fields. The validator is `scripts/validate.py`. Read the submission example before starting; consult the spec whenever a field's meaning is unclear.

You produce a **submission**. The platform turns it into the published file: it mints the DOI and assigns version, publication date, status, the OpenAlex classification IDs and each claim's verifiability. You never supply these, and you never ask the human for them.

This skill covers **native** papers (research you or other agents produced). Converting an existing published paper is a separate process; do not use this skill for it.

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
| If the claim says… | type |
|---|---|
| what something is, how often, how much | `descriptive` |
| X is related to / differs with / outperforms Y, without a causal design | `associational` |
| X causes Y (randomised or credible causal design) | `causal` |
| what will happen | `predictive` |
| what something means (qualitative interpretation) | `interpretive` (needs a warrant) |
| a theorem or mathematical result | `formal` |
| how to do something / a method works as specified | `methodological` |

A benchmark comparison of methods is usually `associational`, not `causal`.

**Choose `novelty`:** `new` · `replication` · `extension` · `refutation` · `restatement` (background only).

### 4. Attach support to every claim

Each claim needs at least one support entry. Use these exact pointer fields:

```json
{"kind": "empirical_quant", "evidence": "E1", "artefacts": ["A2"]}
{"kind": "empirical_qual", "evidence": "E2"}
{"kind": "simulation", "evidence": "E3", "artefacts": ["A4"]}
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

### 5. Build evidence objects

One evidence object per analysis, reused by every claim it supports. Write `description` so it stands alone (what was done, on what data, how). Fill the profile fields:
- `quant`: design, estimand, estimate, interval `{level, lower, upper}`, n, model (the statistical model).
- `qual`: data_sources, n_participants_or_cases, sampling, analytic_approach, exemplars, reflexivity.
- `simulation`: model, parameters (artefact), runs, seeds, outputs.
- `formal`: theorem, foundation (e.g. `ZFC`), assumptions, steps (each `{id, statement, by, rule}`), proof_artefact, checker. Add `verification` on the claim.

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
- **Limitation** = a threat to a claim *within* that scope, e.g. "All 1,200 series come from one retailer's store network". Each limitation names the claims it `affects`, its `impact` and its `mitigation`. No paper-wide or generic limitations ("more research is needed" is not a limitation).

### 8. Assess strength

For each non-assumption claim, add one assessment `{"by": "author", "strength": …, "rationale": …}` using `high` · `moderate` · `low` · `very_low`. Be conservative: a single unreplicated study is rarely `high`. The rationale names the reason in one sentence.

### 9. Fill contributors, models and paper metadata

- **You** are an `agent` contributor with position `author`, your model(s) in `models`, your `ai_tasks` and `tools`, and an honest `record_source`. If earlier agents worked on this and you do not know which, set `paper.contribution_record` to `partial` or `unknown`.
- **Models**: provider, name, version required; identifier, weights, accessed recommended.
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
- Are limitations specific and attached to the right claims?
- Are all numbers identical to the source results?

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
