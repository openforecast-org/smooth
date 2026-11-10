---
name: claimset-submit
description: Turn a piece of research (draft paper, analysis, code, results) into a Claimset v0.40 submission file, validate it, and prepare it for the human guarantor. Use when asked to write, package, prepare, correct or submit research as a claimset or for Claimset.
---

# Preparing a Claimset submission

A claimset is a research paper written for LLM readers: a flat JSON file of atomic, self-contained claims plus the definitions, evidence, assumptions, limitations, warrants and artefacts that support, condition and limit them. You write every field. A human decides two things (guarantor and licence), and you ask them before you finish.

The normative rules are in `references/spec-v0.40.md`. Extensions of the format, for detail that one kind of work needs (spec §2.7), are in `references/extensions/`: the registry `README.md` lists them. Use an extension only when the work has what it records (a glossed linguistic example: `linguistics`), declare it in the top-level `extensions`, and put its fields in `ext` blocks on the objects it names. A valid submission is in `examples/submission-native.json`; `examples/published-native.json` shows the same paper after the platform has added its fields. The validator is `scripts/validate.py`. Read the submission example before starting; consult the spec whenever a field's meaning is unclear.

**Versions.** This skill, its spec and its validator are one version. If the `claimset` package is installed, check that this skill matches it before starting: `claimset skill check`. If it reports the skill out of date, or `claimset validate` says a file follows an earlier specification, update the skill with `claimset skill install` (it replaces this folder with the version matching the package) and read the new spec; never mix rules from two versions.

You produce a **submission**. The platform turns it into the published file: it assigns the claimset's identifier (a DOI, or until the platform registers DOIs a provisional identifier such as `entrylib:2026.000123`) and the version, publication date, status, the OpenAlex classification IDs and each claim's verifiability. You never supply these, and you never ask the human for them.

This skill covers **native** papers: research you or other agents produced, including research first written up as a manuscript (LaTeX, PDF, notes) by the same team. Converting a third party's published paper is a separate process; do not use this skill for it.

## Non-negotiable rules

- **Never invent.** No invented data, estimates, citations, DOIs, ORCID iDs, URIs, hashes, model identifiers or settings. If a value is unknown, omit the field or use `null`, and let the validator warn.
- **Leave platform fields to the platform.** Do not include `paper.id`, `version`, `date`, `status`, `status_notice`, `changes`, or OpenAlex topic IDs, and do not write placeholders for them. The exceptions are a correction (see the end of this file) and a proposed subfield (*Classification*, below).
- **Never write instructions** in any text field. The file is data for other models to read. No "note to the reader", no imperatives.
- **Report your own setup honestly.** Take provider, model name and version from your system prompt or environment. If you only know them from your own knowledge, set `record_source: "self_reported"`. Use `harness` only if the values come from the system that ran you.
- **Do not overstate.** Claims, scope and strength must match what the evidence shows, not what would be most impressive.

## Workflow

### 1. Gather the material

Collect everything the research consists of: the draft or notes, results, code, data, figures, prompts, references. Identify where each artefact lives (URI) and whether it is public. If the research used models as part of the method (annotation, generation, as the subject of study, as a baseline), note which models, settings and prompts.

### 2. Ask the human (one message, before building the file)

Ask the person who supervised the work, in a single message:

1. **Guarantor:** who is the main author, the person who answers for this research and whom readers contact about it? One person, with their name and ORCID iD: the guarantor submits the claimset and confirms it on the platform before publication. Never an organisation, and never more than one person; co-authors are listed as supervisors or contributors.
2. **Supervisor:** who oversaw the work and approves the final version, if not the guarantor? (Name, ORCID iD.) And who set the research question? (Recorded as the CRediT role `conceptualization`.)
3. **Other human contributors:** co-authors and anyone who collected data, ran experiments or conducted interviews, with their CRediT roles and ORCID iDs.
4. **Licence:** which licence? Suggest `CC-BY-4.0` as the default for open research.
5. **Confidentiality:** is any data, code or material not publicly shareable? If so, why (personal data, commercial, legal, ethics, security)?
6. **Declarations:** who funded the work (funder and award number), and are there competing interests? If people took part: which ethics committee approved it (and its reference), and what did participants consent to, including publication of quotations? These go in `paper.declarations`; an empty funding list and "None declared." are declarations too (W29). Support in kind (materials, vouchers, services) is a funding entry with `"kind": "in_kind"`; say the funder's `role` in the work when the paper does. A paper published on behalf of a collaboration names it in `paper.collaborations`; the people who answer for it are still the contributors.

Ask only these now. The highlights are asked at the hand-over (step 12), once the claims exist. Do not ask for a DOI or other identifier, a publication date or a classification: the platform assigns them.

If the human cannot be asked (unattended run), stop after building and validating the file and list these questions as open items. Never fill guarantor, licence or highlights on your own.

### 2a. If the research has a lab record

If the work was done with a lab record (`claimset/<study>/lab/lab.jsonl`, or an older `lab/lab.jsonl`; specified in `references/lab-record-v0.21.md`), start from it rather than from a manuscript:
- Run `claimset lab export [--paper NAME]` if the tool is available (with several papers recorded, one paper at a time). It writes a draft that holds everything the record states and prints a **completion list**: the decisions and wording still needed (licence, guarantor, principal claims, claim types, scopes, evidence profiles and descriptions, statements of negative results). Work through that list; do not change what the record states, such as `hypothesis`, `planning`, `planned_at`, `lab_runs` or the exploration counts. If export refuses (an open hypothesis, an unmerged agent log, a record that does not verify), fix the record with `claimset lab` commands, never by editing it.
- Without the tool, build the claims from the record by hand, following the mapping in lab record §9.
- Every hypothesis that reached `supported`, `not_supported` or `inconclusive` becomes a claim with `hypothesis: {lab_event, outcome}`, `planning`, and for planned tests `planned_at`, `planned_at_anchor` and `test` (the decision rule) copied from the record. Never upgrade `exploratory` to `confirmatory`, and never change a `test`: it is what the plan fixed before the data were seen.
- **Negative results are claims too**, in this claimset: state plainly what was not found, with scope and evidence (see `examples/submission-from-lab.json`, claim C4).
- Fill `paper.exploration` with the counts from the record, and list the record as an artefact of kind `lab_record`, with any `withheld_events` and their reasons.
- Give evidence objects the `lab_runs` they came from. Export records their numbers as **tables** (rows the compared units, columns the variables) and writes every result of the record, at full precision, to a CSV beside the claimset (`files/<paper>.results.csv`, a `data` artefact the evidence and tables name). Check each table: a self-contained description, each column named by a variable term where the study defines one (`m4 mce` → the term for MCE), the conditions that set columns apart. Claims cite the rows they rest on (`E12.T1.r1`). Package the claimset (`claimset package`) so the CSV goes with it.

Then continue with the steps below to choose principal claims, review wording and scope, and ask the human.

**Planned tests without a lab record.** If a claim was tested under a plan registered before the data were seen (a preregistration on OSF or AsPredicted, a registered report's Stage 1 protocol, a trial registry entry), list the registration in `references` (its DOI, a trial number as a CURIE such as `clinicaltrials:NCT01234567`, or an archived URL) and give the claim `hypothesis: {outcome}` (no `lab_event`), `planning: confirmatory` (or `confirmatory_with_deviations`, with each departure from the registration a limitation of category `deviation`), `planned_at` (the registration's date; leave it out if neither the paper nor the registry gives it, never estimate it), `planned_at_anchor: registry`, and `test`: the framework, its parameters (for a significance test: `alpha`, `test`, `sides`, `correction`, `sample`), the rules for `supported` and `not_supported` exactly as registered (expressions with `quantities` when the registration fixes thresholds, otherwise text with `verdict: judged`), and `registration`: the reference's ID. `not_supported` means the registered refutation rule was met; otherwise a test that was not significant is `inconclusive`. A hypothesis the paper states but did not register is `exploratory` (spec §3.3.3, *Tests registered outside a lab record*).

### 3. Write the claims first

Work in this order: principal claims → supporting claims → assumptions.

**Find the principal claims.** These are the contributions: what the work establishes that a reader should take away. Usually 1–5. List them in `paper.contributions` in order of priority.

**Decide what belongs (spec §3.3.4).** Claims are the work's own: the contributions, the claims they rest on, the assumptions and conditions, the results (including those against the method). Never restate what others found. Prior work enters only by reference, listed once in `references` (`B1`, …):
- a result your claim builds on → `cited` support of that claim, `{"kind": "cited", "reference": "B3", "locator": "Theorem 2"}`;
- what your claim answers, extends or contradicts → a relation of that claim, `{"type": "addresses", "reference": "B4", "note": "Selection among ETS models often misidentifies the type of trend."}`;
- a borrowed term (MASE, ETS) → a one-line definition with what this work specifies (e.g. the interval level) and `"source": "B5"`.
- your own method or model → a definition with its defining equations as you write them, every one of them, in LaTeX, displayed (`$$…$$`); a variant (seasonal, multivariate) with its own equations in full, never "the same equations at lag $m$" (W26).
- every definition gets a `kind` saying what the term names: `method` · `intervention` · `population` · `data` · `setting` · `design` · `variable` · `concept` · `theory` · `substance` · `organism` · `biological_material` · `entity` (a gene or protein, a particle, a device) · `source` (one primary source) · `other` (spec §3.2). A compound, a species, a cell line or a gene is a thing of its own kind, with `identifiers` from its registry (`chebi:`, `ncbitaxon:`, `cellosaurus:`, `hgnc:`); giving, sampling or knocking it down is an `intervention` or a `population` of its own. It is required on every definition (rule 63). A theme or a newly named phenomenon you find is a `concept`; what you record about each case (a score, an outcome, an error measure, a judgement) is a `variable`. A `setting` may give its `location` as points `{latitude, longitude, label}`; a `variable` that measures a construct names it in `measures` (the concept's ID). A `variable` or `setting` used as a dimension with numbers gives their `unit`; an `intervention` names what it acts on in `targets` (the gene a knockdown targets). A reaction, decay or pathway is a `process`.
- link terms: a group lists its `members` (the compared methods); a parameter, component or data subset is `part_of` its whole (the smoothing parameter of M, M3 monthly of M3); a variant is a kind of its `broader` term (seasonal M of M). A term with a registry identifier gives it in `identifiers` (`chebi:15377`, `ncbitaxon:9606`, `rrid:…`, `ecli:…`, a data set's `doi:…`). Assumptions that hold wherever a method, design or data set is used go on its definition (`assumptions`), once; evidence whose setup names it inherits them. Definitions mark the other terms they use (`[[D4:…]]`); an algorithm is written as pseudocode in a fenced block.
Leave out general knowledge that no claim rests on and justifications of routine choices (which accuracy measure, which software); put such choices in the evidence's `description` if they matter. Define in full only the terms the work introduces or uses in its own sense.

**A claim is what the work shows or uses.** A statement that nothing in the work supports (no evidence, no citation, no warrant) and that no claim rests on is not a claim: framing ("accurate forecasts are crucial for business"), promise ("paves the way for"), sweeping generalisation ("it is well known"), speculation ("this could transform practice"). Leave it out. If it is a question the work leaves open (a conjecture it did not test, a case it could not settle, a direction for further work), record it under `open_questions`, as a question, with the claims it arises from: `{"id": "O1", "question": "Whether method M helps on weekly series is not established.", "claims": ["C2"]}`. The validator warns (W23) about a claim that nothing supports and nothing rests on.

**Apply the granularity rule.** Split a statement when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one `interpretive` claim with a warrant.

**One claim per statement.** The test: could one part be true and another false? Then they are two claims, each self-contained, with the shared context in definitions rather than repeated. Signs of two claims in one sentence:
- a semicolon;
- contrast: "but", "while", "whereas", "although", "however";
- addition: "as well as", "also", "in addition", "moreover", "not only";
- explanation: "because", "since", "due to", "thus", "therefore", "which explains": a result and its explanation are two claims, often of different types (descriptive and causal or interpretive);
- an assertion in a side clause (", which …") or in parentheses;
- a recommendation added to a finding ("…, so practitioners should …");
- results cited from two evidence objects, two scopes, or two different outcomes;
- "and" joining two verbs ("reduces MASE and improves coverage").
"M is more accurate than ETS at short horizons and faster to fit" is two claims. The validator flags most of these signs (W22); check each flagged statement.

Write each claim as an **atom** before writing its statement: *subject* (the method, model, group or population it is about), *relation* (what is asserted), *comparator* (what it is compared with, if anything; a named group counts as one), *conditions* (data, setting, sample size, horizon). One claim is one atom. Two subjects, two comparators or two sets of conditions are two claims, whatever the wording: "M beats N on monthly data and P on quarterly data" is two claims. Atoms are grouped only when they share subject, comparator and conditions and differ in the measure or summary: "M has a lower mean and median MASE than N on monthly data, but a higher MSIS" stays one claim, with the table that holds the numbers. A conclusion you draw over several atoms ("M is the most robust variant") is a claim of its own, with `depends_on` listing the atoms.

**Record the atom in `about`** (spec §3.3.5; required for principal claims, rule 64; recommended for the others, W27; never on assumptions): `{"subject": "D2", "relation": "has lower error than", "object": "D8", "comparator": "D3", "conditions": ["D5"]}`. `object` is what the relation links the subject to (the variable, outcome, property or theme), or `null`; a list of variable IDs when the atom is grouped by variable. Each slot is a definition ID, or a short noun phrase for something you do not define; `relation` is always text and carries any negation ("does not reduce"). Every ID in `about` must be marked in the statement. Examples by kind of work:
- benchmark: subject the method, object the error measures (variables), comparator the other method, conditions the data set;
- trial: subject the intervention, object the outcome, comparator the control, conditions the population and setting;
- qualitative study: subject the phenomenon or the participants, relation "is experienced as" or "is described as", object the theme (a `concept`), conditions the sample and setting; the interpretation stays in the warrant;
- formal result: subject the model, relation "is identifiable when", object the property or `null`, conditions the theorem's conditions.
A finding that is not one relation (a typology, a process with stages) is a `concept` definition, named as the `object` with relation "is characterised by". `about` never changes what the statement says.

**Groups.** When claims compare against a group whose members you also define (the benchmark methods, the variants of your model), define the group once with `members` listing their definitions, all of one `kind` (spec §3.2, rule 65), and name the group in `about` when the claim is about the group as a whole, a member when it is about that member. A claim about a group is not a claim about each member: if it holds for each, the statement says so ("lower error than each of the benchmark methods").

Not every pattern in a table is a claim. Write a side result as a `secondary` claim only when another claim, a warrant or an open question rests on it, or when it states a loss, a limit or a null result of your method; the result table records the rest (W25).

**Write each statement** as one self-contained sentence:
- No "this", "these", "the above", "as shown", "our method" without naming it. Name the method, population and outcome every time.
- No "novel", "remarkable", "important". "Significant" only as "statistically significant", when the claim reports a test.
- Put what bounds the claim (population, setting, period, conditions) in `scope`, not in the statement. Keep epistemic qualifiers ("suggests", "may") and magnitude ("slightly") in the statement, as the work states them; confidence goes in the assessment.
- Mark defined terms: `[[D1:forecast skill]]`. Mathematics in LaTeX: `$…$` within a sentence for symbols and short conditions ($T$, $\epsilon_t \sim N(0, \sigma^2)$); `$$…$$` for an equation on a line of its own, as a paper would display it: a model's or method's defining equations, and long expressions. Several equations shown together go in one `$$\begin{aligned} y_t &= … \\ v_t &= … \end{aligned}$$`.

**Choose `type`:**
| If the claim states… | type |
|---|---|
| what something is, how often, how much, or what was measured in a computation, simulation or benchmark under stated conditions (including method comparisons measured there) | `descriptive` |
| a relationship observed in data the work did not control | `associational` |
| that X causes Y in a real-world system (randomised or credible causal design) | `causal` |
| how an effect comes about: its steps, intermediates or processes (a reaction mechanism, a pathway) | `mechanistic` |
| what will happen | `predictive` |
| what something means (qualitative interpretation) | `interpretive` (needs a warrant) |
| a theorem or mathematical result | `formal` |
| that a method, procedure or implementation does what is specified | `methodological` |

"Method M was 10.5 times faster" or "the KL divergence was 0.001" are `descriptive`: they report a measurement.

**Choose `novelty`:** `no_prior_found` (you found no earlier result stating the claim: it reports your search, not a proof that none exists) · `replication` · `extension` · `refutation` · `not_applicable` (only for a secondary or supporting claim that is not a finding: a control experiment, a routine check). Assumption claims have no novelty.

**Before labelling a claim `no_prior_found` or `refutation`, find the closest prior result.** Start from the work's own references, then search with whatever is available to you (literature search, web search, the project's notes). Cite the closest result you find: as `cited` support if the claim rests on it, or as a relation (`extends`, `addresses`, `contradicts`, with a `note`) if it positions the claim. For a `no_prior_found` claim, relate it to the closest result you found, with a `note` saying how the claim differs, so the reviewer sees what you found. If prior work already states the claim, it is a `replication` (if your work repeats it independently) or you keep only what is new as an `extension`. If you could not search, say in the strength rationale which sources you checked.

**Check that `consensus` support is a premise, not the claim.** `consensus` means the claim rests on something established. If the established knowledge already says what the claim says, the claim is known; if it does not, it is not the claim's only support. The validator warns (W19) when a `no_prior_found` claim has no support but `consensus`.

**Link to prior work.** A `refutation` must have a `contradicts` (or `fails_to_replicate`) relation naming what it refutes; a `replication` must have `replicates` or `fails_to_replicate`; an `extension` should have `extends`. A claim that the target's evidence or reasoning does not establish it, without asserting the opposite, has `disputes_evidence`. The target can be a paper, book or software release, with a `locator` for the exact place:

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
{"kind": "empirical_obs", "evidence": "E5", "results": ["E5.R1"]}
{"kind": "simulation", "evidence": "E3", "artefacts": ["A4"], "results": ["E3.R1", "E3.R2"]}
{"kind": "demonstration", "artefacts": ["A5"]}
{"kind": "formal_proof", "evidence": "E4"}
{"kind": "argument", "warrant": "W1"}
{"kind": "cited", "reference": "B1", "locator": "Section 3", "self": false}
{"kind": "consensus"}
{"kind": "expert_judgement", "holder": "P2"}
{"kind": "assumption"}
```

- `self: true` when the cited work shares a human supervisor, guarantor or contributor with this paper.
- Each cited work is listed once in `references`: `{"id": "B1", "target": "10.xxxx/yyyy", "id_type": "doi"}`, and cited by its ID.
- Never use `unstated`; it exists only for converted papers. If you cannot support a claim, remove it or turn it into an assumption.
- When a statement quotes numbers, point `results` at the structured results that hold them. Every number in the statement of an empirical or simulation claim should be a recorded result, table cell, test parameter, scope condition or part of a referenced definition; the validator warns (W17) about one that is only in an evidence description.

### 5. Build evidence objects

One evidence object per analysis, reused by every claim it supports; the validator warns (W18) about two evidence objects with the same code version and the same results. Write `description` so it stands alone (what was done, on what data, how).

**Split the setup into its parts.** A study is made of parts that many claims share and that readers and reviewers need to name one at a time. Record each part once as a definition (with its `kind`) and list them in the evidence's `setup`, by role (spec §3.4; required on all evidence except formal, rule 61):
- `studied`: what the study is about or compares: methods with the settings that define them, interventions, exposures, the phenomenon explored (e.g. "ETS, automatic selection by AICc, smooth 4.6.0");
- `cases`: what or whom it is studied on: data sets, populations, participants, sites, documents, with size and selection (e.g. "M3 monthly: 1,428 series, holdout of 18"; "18 nurses in their first two years at two rural hospitals");
- `procedure`: how the cases are obtained and analysed: evaluation scheme (origins, horizons, refits), trial or survey design, sampling and data collection, analytic approach;
- `lens`: what is recorded and through what it is read: error measures and outcomes with their aggregation, instruments, a coding frame, a theory used to read the data (e.g. "MASE, averaged over series and horizons");
- `context`: where and when, when it bounds the study.

```json
"definitions": [{"id": "D5", "term": "M3 monthly", "kind": "data", "definition": "The 1,428 monthly series of the M3 competition; the last 18 observations of each held out.", "source": "B4"},
                {"id": "D7", "term": "rolling origin", "kind": "design", "definition": "Eight forecast origins, 18 steps apart; each method refitted at every origin; horizons 1 to 18."}],
"evidence": [{"id": "E2", "setup": {"studied": ["D1", "D2", "D3"], "cases": ["D5"], "procedure": ["D7"], "lens": ["D8", "D9"]}, …}]
```

A part's role is what it does in this study; its kind is what it is. A regression used as the analysis is `procedure`, a regression method being compared is `studied`. A borrowed part (a published data set, method, measure, instrument or analytic approach) names its `source`; the definition says what this work specifies. Evidence objects that share a part point to the same definition, and statements mark the parts they name (`[[D7:rolling origin]]`). The description still says in a sentence what was done. Files written under v0.29 or earlier: `claimset migrate FILE` renames the old roles (compared, data, design, measures) and lists the kinds and atoms left to add.

Fill the profile fields:
- `quant`: design, estimand, n, model (the statistical model); `estimate` and `interval` for a single result, otherwise `results`.
- `qual`: data_sources, n_participants_or_cases, sampling, analytic_approach, reflexivity; excerpts are results (below).
- `observation`: when the claim is what a record shows and the paper presents the record (a blot, a micrograph, a spectrum, a crystal structure, a described specimen, a telescope image, a trace): `records`, each `{kind, artefact, description}` with `kind` `image` · `spectrum` · `structure` · `specimen` · `recording` · `other` and `artefact` the figure (better, the original file); what was seen goes in `results` (a band present, a peak at 7.26 ppm). A statistical analysis across many records is `quant` evidence `derived_from` it (spec §3.4, *Choosing the profile*).
- `simulation`: model, parameters (artefact), runs, seeds, `results`.
- `formal`: foundation (e.g. `ZFC`), assumptions, steps (each `{id, statement, by, rule}`), proof_artefact, checker. No `theorem` field: the formal claim's statement is the theorem.

**Tables are the default** (spec §3.4, *Tables are the standard form*): more than a handful of numbers from one evidence object (as a rule, more than eight that share a variable or a condition) go into a table, rows the compared units and columns the variables, and claims cite its rows. The validator warns (W40) about a longer list of result entries. The worked example is evidence E12 of entrylib:2026.000001: fifteen result entries in v1, one table of two rows and eight columns in v2, cited by its claim C5 as `E12.T1.r1` and `E12.T1.r2`. Keep the full-precision values in a data artefact (a CSV) that the table names in `artefact`, and give values in the table to the precision the paper reports (four significant digits for a lab record's results).

**Results are observations** (spec §3.4): each the value of a variable, observed under conditions. A single number a claim reports, with no table to hold it, is a result entry, as is a value a cell cannot hold (with an `interval`, an `uncertainty` or `n`, or an excerpt), in any evidence except formal:

```json
"results": [{"id": "R1", "variable": "D11", "value": 10.55, "unit": "1", "n": 500,
             "conditions": {"D7": "D2", "missing values": 1}, "artefact": "A8"}]
```

- `variable`: what is observed, a definition of kind `variable` (or a `concept` whose presence is observed), or text.
- `value`: a number, a category (a definition ID or a word), a boolean, or an excerpt `{text, locator}` (an interview passage, a passage of a judgment, a linguistic example with its `translation`; its gloss is `ext.linguistics.gloss`). A value known only as a bound is a limit, `{"bound": "upper", "limit": 0.5, "level": 0.95}` (a 95% CL upper limit; no `level` for a detection limit); a value that is a span is a range, `{"min": 36, "max": 77}` (yields across substrates, an excluded mass region with `level`). Never put a limit's number in `value` as if it were measured.
- `conditions`: where the observation lies, `{dimension: value}`, dimensions and values naming definitions where they exist. When the dimension is a group (the compared methods), the value is one of its members.
- `unit` (a UCUM code: `ms`, `%`, `mg/L`), `interval` with its `kind` (`confidence`, `credible` or `prediction`), `n`.
- `statistic`, what kind of number the value is (`mean`, `median`, `sd`, `difference`, `odds_ratio`, `standardised_mean_difference`, `nnt`, `p_value`, …; spec §3.4), and `approximate: true` when the paper gives it only approximately ("about 1.5-fold"): record the number it states. A family with parameters is an object: a geometric mean `{"name": "mean", "kind": "geometric"}`, a 90th percentile `{"name": "quantile", "probability": 0.9}`, a third central moment `{"name": "moment", "order": 3, "central": true}`. The statistic says how values were summarised (a mean over series); the error measure itself is the variable. Use `{"name": "other", "label": ...}` only when no name fits. A table column object may carry both; so may a row, when the paper's rows are statistics (mean, median), but never a row and a column for the same cell.
- A ± goes in `uncertainty`, typed: `[{"kind": "sd", "value": 1.1}]` for a standard deviation, `se` for a standard error, `stat` and `sys` for a measurement's statistical and systematic components (each `sys` with a `label` naming its source), `total`; `{"kind": "sys", "plus": 0.5, "minus": 0.3}` when it is asymmetric. Ask the author what an unexplained ± is; `unspecified` is for converted papers that do not say. In a table, a column of standard deviations is `{"variable": "D8", "conditions": {"uncertainty": "sd"}}`.
- A value read off a figure says so: `"read_from": {"artefact": "A5", "precision": 0.01}`.
- A contrast (a difference, ratio, effect size or the p-value of a comparison) names the other side with the reserved condition `comparator`: `{"D7": "D2", "comparator": "D3"}`.

Values that form a table go into a table instead, one per table: rows the compared units (methods, cases, groups, sample sizes), columns the variables, the conditions that set a variable apart (a nominal level, a subset, a statistic) as column conditions, and the conditions every value shares on the table. Cells may be any value type:

```json
"tables": [{"id": "T1", "description": "Mean MASE and MSIS of three methods on the M3 monthly series, horizons 1 to 18",
            "conditions": {"D5": "D14"}, "row_labels": ["D7"], "columns": ["D8", "D9"], "units": ["1", "1"],
            "rows": [{"id": "r1", "label": ["D1"], "values": [0.861, 6.95]},
                     {"id": "r2", "label": ["D2"], "values": [0.874, 6.71]}]}]
```

A column may carry conditions of its own: `{"variable": "D8", "conditions": {"horizon": 1}}`. Data larger than the paper reports (per series, per participant) go in a CSV artefact with `columns` naming each column's variable or dimension; cite its rows as `A4#row=12`.

**Figures** (`kind: figure`) have a `function`: `data_plot` (record its values as observations and list them in `shows`), `specimen` (an image or recording that is evidence: record what you read from it with `read_from`), or `diagram` (an illustration: never cited as support; write what it shows into definitions). Images are files of the submission package, never embedded in the JSON.

Claims cite a table (`E2.T1`) or a row (`E2.T1.r1`) in their support's `results`, not result IDs of values a table holds. A claimset distils the argument; it does not re-encode every table of the research, so record the tables the claims rest on, and leave stray numbers no claim uses to the data artefact.

The table holds the numbers; a claim holds what you conclude from them. State the pattern (which method is better or worse, on which measures, under which conditions, including where it is worse) and cite the table or rows; repeat a value only when the conclusion turns on it ("selected for 6 of 11 groups"). Do not write a claim that only retells a table: cite the table from the claim that uses it. The validator warns (W24) when a statement repeats three or more values of the tables it cites.

Copy values from the result tables, never retype them from prose. If the code that produced the results differs from the archived artefact version, record it in `execution`: `{"code_version": "<commit>", "notes": "…"}`.

Set `availability` on every evidence object and artefact (rule 75). If it is not stated anywhere, use `not_stated`; never guess. If the level is otherwise not `public`, give a `reason`; for `platform`, `audit` or `author_only` also a `content_hash` (compute it, e.g. `sha256:<hex>` of the file; never invent one).

If models were part of the research design, record them in the evidence's `models` list with role, settings, record_source and a `prompts` artefact.

### 6. Surface assumptions

Every claim rests on assumptions; make them explicit as separate claims with `support: [{"kind": "assumption"}]`, an `assumption_category` and a `check`, and list them in the dependent claim's `assumptions`. Check especially:
- statistical: independence, stationarity, distributional form, missing-at-random;
- data: representativeness, correct labels, measurement validity;
- methodological: choice of benchmark, metric, horizon, instrument validity.

Use `check: "tested"` or `"sensitivity_analysed"` only if the work actually did so, and point `check_evidence` at that evidence. Otherwise `unchecked` (or `untestable`). Assumption claims have no `assessments` and no `novelty`.

### 7. Separate scope from limitations

- **Scope** = what the work covered: population, setting, geography, language, period, data_type, conditions, and `excludes` (where it explicitly does not apply). "English-language data only" is scope.
- **Write scope claim by claim, never per section.** Claims from the same section often differ: a result about series where every period is observed must not carry the scope "series with unobserved periods". Leave out fields that do not apply. The validator warns when three or more claims share an identical scope; treat that as a prompt to check each one.
- **Limitation** = a threat to a claim *within* that scope, e.g. "All 1,200 series come from one retailer's store network". Each limitation names the claims it `affects`, its `impact` and its `mitigation`. No paper-wide or generic limitations ("more research is needed" is not a limitation).

### 8. Assess strength, or verification for formal claims

For each claim that is neither an assumption nor `formal`, add one assessment `{"by": "author", "strength": …, "rationale": …}` using `high` · `moderate` · `low` · `very_low`. Be conservative: a single unreplicated study is rarely `high`. The rationale names the reason in one sentence. When a principal claim has a limitation that `could_reverse` it with `mitigation: none`, the rationale says how that risk was weighed, naming the limitation (`L2`), or the strength is `low` or `very_low` (W20).

Formal claims get **no strength assessment**. A formal claim proved or argued in the paper gets `verification` instead, and every independent check goes into `checks`.

**Checking.** If your environment lets you start another agent, use it to check the claims, principal claims and refutations first:
- Give the checker only the claimset and the manuscript, not your reasoning. Ask it to re-derive each formal claim step by step, and to reproduce numerical claims from the artefacts where feasible, reporting for each claim: confirmed, disputed (which step or number, and why) or inconclusive.
- Prefer a checker on a **different model** from yours, ideally from a different provider; a same-model check shares your blind spots and the validator warns about it.
- List the checker as a `contributor` with AI task `review`, and record each attempt on the claim:
  `"checks": [{"by": "P5", "outcome": "confirmed", "extent": "full", "notes": "re-derived steps E2.1–E2.4"}]`
- **Record every outcome, including failures.** A disputed check stays in the file. If you then correct the claim and a recheck confirms it, add `resolution` to the disputed check and record the new check. Never drop a failed check.

Then set `verification` to match the checks: `agent_checked` (a full confirmed check by an agent, no unresolved dispute), `peer_checked` (the same by a named human), `disputed` (an unresolved dispute), `unchecked` (no full confirmed check), or `machine_checked` (only with a proof-assistant artefact and `checker`). Never record your own checking as a check.

A formal result you cite and apply is `cited` support of your claim, not a claim of its own, and carries neither.

### 9. Fill contributors, models and paper metadata

- **Agents.** Every agent that produced the paper's content (analysis, claims, text, or turning the team's own manuscript into this claimset) is an `author`; you are one. An agent that only ran a computation the authors report, or checked a proof, is a `contributor`. Give each agent its model(s), `ai_tasks`, `tools` and an honest `record_source`. If earlier agents worked on this and you do not know which, set `paper.contribution_record` to `partial` or `unknown`.
- **Manuscript.** If the work was first written up as a manuscript, list it as an artefact of kind `manuscript`.
- **Models**: provider, name, version required; identifier, weights, accessed recommended.
- **Artefacts**: give each of the work's own artefacts and each packaged file a `licence` (SPDX identifier); an artefact taken from a cited work (it names its `reference`) carries none, since its licence belongs with that work. Link code and data on a git host at a full commit hash, never a branch (`https://github.com/org/repo/tree/<40-hex commit>/path`): a branch moves (W21). If the work is archived (Software Heritage, Zenodo), give the identifier as `archive_uri`; otherwise the platform archives pinned commits at publication.
- **Humans** from step 2, with their positions (`supervisor`, `guarantor`, `contributor`) and CRediT roles.
- **Paper**: title (descriptive, no rhetoric), licence, contributions, highlights (step 12, chosen by the guarantor), `autonomy`, contribution_record, `origin: "native"`. Nothing else at paper level.
- **Classification**: give 3–8 precise `keywords` (e.g. `"healthcare demand forecasting"`); the platform assigns the OpenAlex topics from them and from the content, and the subfield too unless you propose one. Propose `subfield` when the paper's field is clear: take its ID from the list at https://entrylib.org/subfields.json (or OpenAlex), confirm it with the human, and never guess an ID. The platform keeps a valid proposal, and the guarantor sees it on the consent page. Leave `topics` to the platform.

### 9a. Package the files

Figures, specimen images, recordings, bulk data, instruments and stimuli that are not already held elsewhere (a commit on a git host, a data repository) travel with the claimset in a **submission package** (spec §2.5), a BagIt folder or zip:

```
bagit.txt
manifest-sha256.txt          (sha256 and path of every file under data/)
data/paper.claimset.json
data/files/fig3.png
```

Each packaged artefact has a `uri` relative to `data/` (`"files/fig3.png"`) and `version` `sha256:<hex>` of the file, which the manifest repeats. In a repository, keep the files in `claimset/<study>/files/`. Never put an image into the JSON. Build the package with `claimset package claimset/<study>/<paper>.claimset.json -o <paper>.zip` (add `--update` to write the checksums into the claimset file too); it validates what it built. The platform accepts a package within limits (spec §2.5, rule 77): at most 200 files, 50 MB a file and 200 MB in all, accepted file types only (no archives or executables; a gzipped data file such as `results.csv.gz` is fine), and `claimset package` refuses one that breaks them.

### 10. Validate and fix

Run:

```
claimset validate claimset.json          # if the claimset package is installed
claimset validate submission/            # a submission package: the claimset and its files
python3 scripts/validate.py claimset.json  # otherwise, the bundled copy
```

(Use `--json` for machine-readable output.) Fix every error and rerun until the file is valid. Then read each warning and fix it if you can; leave it only if the information genuinely does not exist.

The report ends with the verifiability the platform will compute for each claim. You do not copy these into the file, but read them: a principal claim that comes out `unverifiable` or `audit_only`, or with flags such as `unpublished_prompts`, usually means an artefact, prompt or availability statement is missing that you could add.

The validator checks structure, references and some wording heuristics. It cannot check that claims are true or faithful to the work. Do that yourself:

### 11. Self-review before handing over

Re-read the file as a sceptical reviewer who sees only one claim at a time. If the `claimset` package is installed, read its reading view (`claimset render claimset.json`), which is what reading models will see:
- Does each statement make sense alone, with no missing referent?
- Does each principal claim's evidence actually show what the statement says, within its scope?
- Is anything stated more strongly than the evidence allows?
- Is every important assumption surfaced, with an honest `check`?
- Does each claim's scope describe that claim, not its section?
- Does every refutation, replication and extension name the work it relates to?
- Are limitations specific and attached to the right claims?
- Does the definition of your method or model give every defining equation, and each variant its own (W26)? Could a reader compute the method from it?
- Does every secondary result have something resting on it, or state a loss, limit or null result (W25)?
- Is each claim one atom, with one subject, one comparator and one set of conditions, recorded in its `about` with every ID marked in the statement? Does each statement hold one claim (W22)? Is each part of the setup a definition with a `kind`, listed in the evidence's `setup` (W28)?
- Does any statement retell a table instead of stating a conclusion (W24)?
- Are all numbers identical to the structured results and the source tables? Does each claim cite only the results its statement reports? (The validator warns, W12, when a cited result does not appear in the statement, and W17 when a number in the statement matches no result.)
- Does every `no_prior_found` or `refutation` claim cite the closest prior result you found?
- Do the most important claims, especially refutations, have independent checks, with every outcome recorded?
- If there is a lab record: are negative results included, are the exploration counts right, and is every exploratory claim labelled as such?

### 12. Hand over

Before the first submission, put the claims in reading order and close the gaps in their IDs: `claimset renumber claimset.json` (it rewrites every reference; validate and render again afterwards). A claimset exported from a lab record keeps its link to the record through `hypothesis.lab_event`, so renumbering is safe. Never renumber a published claimset: a correction or revision keeps every ID and places a new claim where it belongs in the file, with the next free number (spec §9, *Order and IDs*).

Write the file where it belongs (spec §2.4): `claimset/<study>/<paper>.claimset.json`, which in a git repository is the checkout of the study's branch (lab record §3; the tool commits and pushes what `claimset lab` commands write there, and you commit a file you edit by hand in that checkout and push the branch), with its reading view as `claimset/<study>/<paper>.claimset.txt` (`claimset render … -o …`). The name stays the same through revisions; never put a date or version in it. When the human submits a version, commit it and mark it with `claimset tag claimset/<study>/<paper>.claimset.json` (a git tag `claimset/<study>/<paper>/v<n>`). If a claim builds on a paper of the same repository that is not yet published, cite or relate it with `id_type: claimset_local` and the target `<study>/<paper>.claimset.json#C3`; the platform replaces it at publication, so that paper is published first or submitted together.

Give the human a short summary:
1. Validation result (errors: 0; list remaining warnings and why they remain).
2. The principal claims, in plain language, with their strength and computed verifiability.
3. **Highlights, for the guarantor to choose** (spec §3.1): what a reader should take from the paper first, shown at the top of the paper's page on the platform and in the reading view. Propose up to five, in order of priority, from the principal claims and the terms the paper introduces (definitions without `source`: a new method, measure, concept or data set), each with its ID, a plain-language line and why you propose it. Ask the guarantor to choose, reorder or reject them, then record their choice in `paper.highlights` (`["C2", "D4"]`) and validate again. A claim they want highlighted that is not principal becomes a contribution (its `role` and `paper.contributions`). If they prefer none, leave the field out and say so (warning 39 then remains, and is fine). Never choose them yourself.
4. Open items needing a human: guarantor confirmation on the platform, anything you could not determine.

Submit only once the human approves, by whichever route your environment has:
- **A terminal with the `claimset` package:** `claimset submit <paper>.zip` (a package from `claimset package`), or `claimset submit <paper>.claimset.json` for a claimset with no packaged files (its artefacts linked by https URLs, such as Zenodo or a commit on GitHub). For a package deposited elsewhere, `claimset submit https://…/paper.zip`. It validates the file first and sends nothing if it has errors.
- **The EntryLib MCP server** (`https://entrylib.org/mcp`): the `validate` tool, then `submit` with the claimset (a JSON object or its text) or with `package_url`, an https URL of a deposited package (for anything over 10 MB).
- **Neither:** stop at the validated file and tell the human they can upload it at https://entrylib.org/submit.

The answer gives a **consent link** for the guarantor: read the whole output and give the link to the human at once, with what it asks of them (open it, sign in with ORCID, check the claimset as reading models will see it, and confirm within 48 hours, or it is deleted). The platform shows the link only in this answer; `claimset submit` keeps it in `.claimset/submissions/`, and `claimset status` shows it again. Without the link, the guarantor confirms at https://entrylib.org/confirm, signing in with ORCID. Never submit again only because a link was lost. Then follow the submission with `claimset status <id>` (or the MCP tool `submission_status`) until it is `published`, and give the human the identifier it was published as (`entrylib:<year>.<number>@v1`). Refused means it failed the platform's checks: fix what the reasons say and submit again. A new claimset with the same title and guarantor as one already published is refused as a duplicate: if it is a new version, set `paper.id` (*Corrections*, below); only a genuinely different paper is submitted again with `--new-paper` (MCP: `new_paper`).

## Revisions: answering a review

A published claimset that has been reviewed is revised with the `claimset-revise` skill: it answers each review comment with evidence, builds the revised version and writes the response.

## Corrections

To correct a published claimset, start from its published file and produce a complete corrected submission:
- Keep `paper.id` set to the existing claimset's identifier (its DOI or `entrylib:` identifier) and add `paper.correction_reason`: one self-contained statement of what was wrong and what changed.
- Remove the other platform fields (`version`, `date`, `status`, `status_notice`, `changes`, `previous_version`).
- Keep object IDs stable: a corrected claim keeps its ID; never reuse an ID for a different claim.
- Validate as usual. The platform computes the diff, assigns the next version and marks the paper as corrected; the identifier stays the same.
- A correction is the author's own initiative. To answer reviews, submit a revision with a response instead (above); a revision can also carry corrections, as `other_changes`.
