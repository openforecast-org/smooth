# Claimset — Specification v0.34 (draft)

Status: working draft · Date: 2026-10-29 · Supersedes v0.33 (`spec/claimset-v0.33.md`) · Companions: lab record specification (`spec/lab-record-v0.15.md`), revision specification (`spec/claimset-revision-v0.2.md`)

Claimset is the name of both the open format and the platform.

## Changes since v0.33

Anyone in any field should be able to write a claimset, and search should rest on what every claimset states. So what search and verification depend on, and every field can state honestly, is required; and what wrongly excluded a field's normal practice is allowed.

- §3.2: `kind` is required on every definition, so that every term can be found and linked by what it names. Rule 63 revised.
- §3.4: a `setup` is required on every evidence object except formal ones: search by what was studied, on what cases, how and through what lens rests on it, and every empirical field can fill its roles. Rule 61 revised; warning 28 keeps only its second part.
- §3.10: `availability` is required on every evidence object and artefact, with the new level `not_stated` for material whose availability the paper does not state, so that a paper's silence is recorded rather than guessed. New rule 75; rule 12 revised; new verifiability flag `availability_not_stated` (§5).
- §8 rule 4: qualifiers that bound where a claim holds go in `scope`; epistemic qualifiers ("suggests", "may") and magnitude ("slightly") are part of what is claimed and stay in the statement. Qualitative and humanities work state claims with such qualifiers as a matter of method.
- Companion lab record v0.14: a supervisor's approval can be recorded by an agent on the supervisor's behalf, with the supervisor's words.
- §2.4: converted papers are kept in a collection folder, `claimset/<collection>/<name>.claimset.json`, named from the source's identifier (`doi-10-1016-j-ejor-2022-04-040`, `arxiv-2401-01234v2`). New warning 31.
- §7: the claimset's DOI: registered with DataCite, one DOI for all versions of a claimset, with the suffix `claimset.<year>.<number>` that carries no other information.

## Changes in v0.33 (since v0.32)

Assumptions listed on a term's definition were inherited through the evidence: every claim resting on an evidence object took the assumptions of every term its setup names. A comparison of five methods then gave each claim the assumptions of all five, so that "VETS F has the highest pinball values of the five models" was shown as resting on the error assumption of VETS D, and a claim about ETS on the error assumptions of CES, against which it was only compared.

- §3.2, *Assumptions of a term*: a claim inherits the assumptions of the data, design, variables and setting its evidence's setup names (roles `cases`, `procedure`, `lens`, `context`), and, of the things studied (role `studied`), only those of its own subject: the term its `about.subject` names, or each member when the subject is a group. The comparator's assumptions are not the claim's. A claim without `about` inherits from every term its setup names, as before.
- Warning 15 and the reading view (§2.3) count and show the assumptions inherited this way.
- §3.6, warning 9: an artefact that is another work the paper uses and cites (it names its `reference`), and that is not a file of the submission package, needs no `licence`: the licence is the cited work's, recorded where that work is described. The paper's own artefacts and every packaged file still state theirs.
- §8 rule 5: a term marker's text is plain; it holds no other marker. A nested marker is an error (rule 1).
- Warning 25 does not ask about a secondary result that stands in a relation to another claim of the paper (one that qualifies a principal claim states its limit), and §8 rule 7 does not count "statistically significant" as rhetoric.

## Changes in v0.32 (since v0.31)

Every field reports its findings as values of something observed on something under some conditions: a reaction time per participant and condition, the expression of a gene per cell line and treatment, the yield of a reaction per catalyst and temperature, whether a construction is attested per language and period, the outcome of an appeal per court and year, the accuracy of a model per benchmark, whether a theme appears per participant. Until v0.31 results were numbers labelled with free text, tables held numbers only, figures were opaque, the materials, registries and declarations of most fields had no place, and files could not travel with a submission. This version records results as observations over the terms of the study, the model of the W3C RDF Data Cube, SDMX and the observation models of the sciences, so that results can be found and compared across papers in any field.

- §3.4, *Observations*: a result is an observation: a `variable` (was `quantity`), a typed `value` (a number, a category, a boolean, or an excerpt), its `conditions` as coordinates over dimensions, and optionally a unit (UCUM), an interval and `n`. Variables, dimensions and category values name definitions; a dimension that is a group (§3.2, *Groups*) takes its members as values. A table is a compact form of many observations, with typed cells and columns that may carry their own conditions. Results and tables may appear in any evidence except formal. Rules 38 and 48 revised; new rule 66.
- §3.4: values read off a figure or a recording record it in `read_from`, with their precision. Excerpts replace the `exemplars` of the qualitative profile; they carry their locator and, for linguistic examples, a gloss and translation.
- §3.6, *Figures*: a figure has a `function`: `data_plot` (shows observations), `specimen` (an image or recording that is itself evidence, read into observations) or `diagram` (illustrates; never evidence). Data plots name the observations they show (`shows`) and may give their `encoding`. New artefact kinds `instrument`, `stimulus`, `codebook` and `recording`. Bulk data files name what each column holds (`columns`, after W3C CSVW), and their rows are citable (`A4#row=12`, RFC 7111). New rule 67; the flag `figure_without_data` revised.
- §3.2: the kind `measure` is renamed `variable`: what is recorded or observed about each case, in any field. Definitions gain `identifiers`: registry identifiers of the thing the term names (`chebi:15377`, `ncbitaxon:9606`, `rrid:AB_2138153`, `ecli:…`, a data set's DOI). Definitions gain `part_of` (a parameter or component of a method, a subset of a data set, a subscale of an instrument, an article of a statute) and `broader` (a variant or subtype of another term), beside the `members` of a group. Algorithms are written as pseudocode. New rule 68.
- §3.1: papers gain `declarations`: funding (funder by ROR, award), competing interests, ethics approval and consent. New rule 69; new warning 29.
- §4: identifier type `curie`, a compact identifier `prefix:local` from a shared registry of prefixes, for works and things that DOIs do not cover (legal sources, database records, biological resources).
- §2.5, new: a submission is a package, the claimset with the files it names (BagIt); the platform stores the files and assigns their URIs at publication. New rule 70.
- §2.3: the reading view shows observations, figure functions and what they show, identifiers and declarations.
- Links the format could not express until v0.31:
  - §3.7: a relation may target a claim of the same paper (`id_type: local`), with the new types `explains` (the claim explains the target's result) and `qualifies` (it states an exception, boundary or condition of the target). `depends_on` stays for premises. Rule 34 revised; new rule 71.
  - §3.4: evidence names the evidence it is `derived_from` (a re-analysis, a subset, a pooling), so readers see when results share data. New rule 72.
  - §3.2: a definition lists the `assumptions` the thing it names rests on (the error distribution of a model, the randomisation of a design); evidence whose setup names it inherits them. New rule 73; warning 15 counts them.
  - §3.4: a contrast between two positions on a dimension (a difference, a ratio, an effect size, the p-value of a comparison) is an observation with the reserved condition `comparator`. New rule 74.
  - §3.2, §8: definitions mark the terms they use (`[[D4:…]]`), and a term the paper defines in its own words that is a standard or another paper's term says so by `identifiers` or `source`.
- Companion lab record v0.13 renames the term kind `measure` to `variable`. Files written under v0.31 are brought over with `claimset migrate`, which renames `measure` and `quantity` and lists what needs judgement (figure functions, excerpts that were `exemplars`, declarations); `claimset validate` accepts a submission package as well as a file.

## Changes in v0.31 (since v0.30)

A claim about a group of methods ("CES has the lowest error of ETS, ARIMA and the Theta method") and a claim about one of them ("CES has a lower MASE than ETS") were not connected: the group was one definition whose text listed its members, so a reader looking for what is known about ETS did not find the first claim.

- §3.2, *Groups*: a definition can name its `members`, the definitions of the things it groups. A claim about a group is not a claim about each member unless its statement says so. New rule 65.
- Warning 28 counts a group or its members as named in a setup that names the other, and asks nothing of claims resting on formal evidence, which has no setup.

## Changes in v0.30 (since v0.29)

A reader looking for papers asks what they studied, on what, how, and what they found about it: "which papers compare ETS with ARIMA on the M4 data by MASE", "what is known about role ambiguity among rural nurses". Until v0.29 the answer was spread over three partial records of the same thing (free-text `scope`, the forecasting-shaped `setup` of evidence, the closed lists of the qualitative profile), and the atom that §8 asks a writer to form first was never written down. This version records one frame of a study, in terms that fit any empirical design, and records each claim's atom over that frame, so that papers and claims can be found by the things they are about, whatever their field or method.

- §3.4, *Setup*: the roles are renamed and generalised so that they fit quantitative, qualitative, simulation and mixed designs: `studied` (was `compared`), `cases` (was `data`), `procedure` (was `design`), `lens` (was `measures`), and a new role `context`. A table maps the roles onto PICO (trials) and SPIDER (qualitative evidence synthesis). Setup is recommended for every evidence object except formal ones. Rule 61 revised; new warning 28.
- §3.2: definitions gain `kind`, a closed list saying what sort of thing a term names (a method, a population, a measure, a concept, …). Required for definitions named in a setup or in a claim's `about`. New rule 63.
- §3.3.5, new: claims gain `about`, the claim's atom (§8 rule 1) recorded as fields: `subject`, `relation`, `object`, `comparator`, `conditions`. Slots hold definition IDs, or short text for things the paper does not define. Required for principal claims, recommended for the others except assumptions. New rule 64; new warning 27.
- §2.3: the reading view shows each definition's kind and leaves `about` out (the statement says it in words).
- §11, §12: the platform resolves defined terms to shared entities across papers; alignment with PICO, SPIDER and the statement model of nanopublications and Wikidata.
- §13: new open question 10: the frame is to be tested by a coding exercise across paper types; roles, kinds or slots that coders cannot assign consistently are revised in a later version.
- Appendix A and D updated; new Appendix E: a qualitative paper.
- Companion lab record v0.12 records terms, hypothesis atoms and plan setups before the data are seen, and exports them as definitions, `about` and `setup`. Files written under v0.23 to v0.29 are brought over with `claimset migrate`, which renames the setup roles and lists the kinds and atoms left to add.

## Changes in v0.29 (since v0.28)

A definition of a paper's own method written as a description of its formula ("an exponentially weighted sum, the weights being powers of …"), or of a variant as "the same equations at lag m", leaves a reader unable to compute or check the method, and the claims about it then rest on a model nobody can reconstruct.

- §3.2, *The paper's own method or model*: its definition gives the equations that define it, as the paper writes them, every one of them; a variant gives its own equations in full; a form derived from another (an underlying state-space model) is a claim, not part of the definition.
- New warning 26 flags a definition of the paper's own that a formal claim names but that contains no mathematics (§10.4).

## Changes in v0.28 (since v0.27)

A claimset exported from a lab record numbers its claims in the order the hypotheses were written, and a hand-made one in the order its claims were found; neither is the order of the paper's argument, and the reading view follows file order.

- §9: claims are listed in reading order, and before the first submission their IDs follow that order (`claimset renumber`). After publication IDs never change: a revision keeps every ID and places a new claim where it belongs in the file, with the next free number.

## Changes in v0.27 (since v0.26)

Splitting comparisons into atoms showed how many claims of a converted paper only read off its tables: each sentence of a results section became several claims that nothing rests on, burying the contributions, while the tables already record every value.

- §3.3, *Secondary claims*: a secondary claim resting on evidence (a result) is written only when a claim, a warrant or an open question rests on it, or when it states a loss, limit or null result of the paper's own method. Other results stay in the result tables of their evidence (§3.4).
- New warning 25 flags a secondary descriptive claim resting on evidence on which nothing rests (§10.4).

## Changes in v0.26 (since v0.25)

"One assertion per statement" was judged sentence by sentence, and comparisons kept growing: several methods, several comparators, several conditions in one statement. A claim is now defined by its parts, so that it can be checked.

- §8 rule 1: a claim is one atom: one subject, one relation, one comparator (if any), one set of conditions. Rule 2: atoms are grouped only when they share subject, comparator and conditions and differ in the measure or summary; the paper's conclusions over several atoms are claims of their own that depend on them.

## Changes in v0.25 (since v0.24)

Tables carry the numbers; claims carry what the paper concludes from them. Statements that retell a table's values add length and transcription errors, and tell a reader nothing the table does not.

- §3.4 *Tables and claims*: a claim is written for a conclusion the paper draws from a table, states the pattern and cites the table or its rows; it repeats a value only when the conclusion turns on it. A claim that only retells a table is not written.
- New warning 24 flags a statement that repeats three or more values of the tables it cites (§10.4).

## Changes in v0.24 (since v0.23)

Papers carry statements that are neither shown nor used: framing, promise, asides that distract from what the paper establishes. And they carry honest questions they leave open, which have had nowhere to go but a claim.

- New top-level `open_questions`: questions the paper raises and leaves open, conjectures and directions for further work, as questions rather than claims (§3.14). New rule 62.
- §8 rule 9: a claim the paper neither shows nor uses is not a claim. New warning 23: a claim that is not principal, has no warrant, is not formal, rests only on argument, consensus, expert judgement or nothing, and on which no claim rests; the message names rhetorical wording when there is some (§10.4).

## Changes in v0.23 (since v0.22)

An evaluation is made of parts that many claims share and that a reader, a reviewer or another paper needs to name one at a time: what is compared, on what data, by which design, measured how. And a sentence of a paper often carries more than one claim.

- Evidence gains `setup`: the parts of the evaluation, each a definition, by role: `compared`, `data`, `design`, `measures` (§3.4). New rule 61.
- §8: how to tell that a statement holds several claims; new warning 22 flags the usual signs (§10.4).

## Changes in v0.22 (since v0.21)

Novelty is a statement about a search, not a fact: a claim is new only as far as its authors looked. Models read labels literally, so the label says what is known.

- `novelty: new` is renamed `no_prior_found`: the authors found no earlier result stating the claim. A claim with this label cites the closest earlier result they found, as a relation with a note on how the claim differs, when there is one (§3.3).
- Assumption claims have no novelty: they are taken as given, not found. New rule 60 (§10).
- Novelty is judged in review apart from credibility (revision specification §7); §6 corrected: review scores are given by the reviewer, not computed.

## Changes in v0.21 (since v0.20)

A published claimset is reviewed, and its author may answer the review with a revised version. Reviews, responses and resolutions are documents beside the claimset, specified in the revision specification (`spec/claimset-revision-v0.1.md`); this specification changes only where the claimset file meets them.

- Versions arise by correction or by revision (§9). A revision submission names the claimset in `paper.id` without a `correction_reason`; rule 25 revised (§10).
- New platform-assigned field `paper.answers`, the response a revision carries; the operations of `paper.changes` gain `addresses`, the review comments they answer (§2.2).
- `test.addresses`: the review comments whose resolution test a planned test ran (§3.3.3).
- §6: reviews, responses, resolutions and scores are platform records; the reading view the platform serves may add their state (§2.3).
- New rule 59 (§10).

## Changes in v0.20 (since v0.19)

Checks for problems in how a study was run and reported, found when reviewing a real claimset: numbers that no recorded result produced, the same computation recorded twice, new claims resting on established knowledge, contributions an unmitigated limitation could reverse, and links to code that move. They are warnings: nothing new is required.

- New warnings 17 to 21 (§10.4).
- New test flag `reproduces_prompting_result`: the confirmatory test gave the same result as the exploratory run that suggested its hypothesis (§3.3.3; lab record §5.7.5).
- `exploration.hypotheses.from_exploration`: how many hypotheses exploration suggested (§3.1.1); new rule 58 (§10).
- Artefacts gain `archive_uri`, a persistent copy of the exact version; links to code on a git host name a commit, and the platform archives that commit at publication (§3.6, §11).

## Changes in v0.19 (since v0.18)

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
| `paper.id` | At publication: the platform mints a DOI. Exception: a correction or revision submission names the existing claimset's `paper.id` (§9) |
| `paper.version` | At publication: 1 for a new paper; incremented for corrections and revisions |
| `paper.previous_version` | At publication of a correction or revision |
| `paper.changes` | At publication of a correction or revision: JSON Patch computed by the platform, each operation with `reason` and, for a revision, `addresses` (the global IDs of the review comments it answers) |
| `paper.answers` | At publication of a revision: the global ID of the response it carries (revision §5.3) |
| `paper.date` | At publication: the publication date |
| `paper.status` | At publication (`active`), and later by governance actions |
| `paper.status_notice` | When status changes from `active` |
| `paper.classification.subfield`, `paper.classification.topics` | At validation: assigned by the platform's classifier; agent proposals are considered |
| `claims[].verifiability`, `claims[].verifiability_flags` | At validation, and recomputed when availability changes |
| `artefacts[].uri`, `artefacts[].archive_uri` of files in the submission package | At publication: the platform's permanent location of each file (§2.5) |

Before publication, a submission is identified by an internal submission ID issued by the platform. It is not part of the file and is never cited.

Top-level layout:

```json
{
  "schema_version": "0.34",
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
  "references": [],
  "open_questions": []
}
```

### 2.3 Reading view

A claimset is stored and exchanged as JSON: the canonical file, which tools validate, the platform publishes and corrections patch. A model that reads the paper is given its **reading view**: a plain-text rendering of the same claimset. JSON field names, quotation marks and the fields that serve tools, audit and publication take a large share of a file and carry nothing a reader needs; the reading view leaves them out.

- **Kept:** the title, licence, source of a converted paper, declarations, contributions and exploration counts; definitions with their kind, members, the wholes they are part of, their broader terms, assumptions and identifiers; claims with their type, role, novelty (when not `no_prior_found`), support with its pointers and locators, assumptions, `depends_on`, limitations, warrant, strength, verification, verifiability, planning, outcome, scope, relations with their notes (relations within the paper as well), assumptions inherited from the terms they rest on (§3.2, *Assumptions of a term*), and test; evidence with its description, design fields, setup, what it is derived from, estimate, observations (results and tables, with their values, units, conditions and where a value was read from) and proof steps; warrants; limitations; open questions; artefact descriptions, figure functions and the observations a figure shows; references.
- **Left out:** `about` (the statement says the same in words; `about` serves search), `provenance`, `field_provenance`, `source_spans`, `corrections`, availability, artefact locations, versions and licences, contributors, models, execution records, lab-record identifiers, `schema_version` and the platform's publication fields.

- **Order:** claims are grouped by role: principal claims, then supporting claims, then secondary claims, each group in file order. Claims that are not principal and whose support is only `argument` or `unstated` follow, under CLAIMS WITHOUT EVIDENCE, so a reader sees at once what the paper asserts or argues without evidence. Principal claims stay with the principal claims whatever their support.

The reading view is made mechanically from a valid file (`claimset render`); it adds nothing and is never edited. The reading view the platform serves may append, under REVIEW STATUS, each claim's unresolved review comments with their category, severity and state; that is derived data, and `claimset render` of the file alone does not add it. Its layout is the reference rendering in Appendix D. A reader that needs a field left out (a source span to check a claim against its source, an artefact's location) reads the canonical file.

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
- **Converted papers.** A converted paper comes from no study of the repository and has no lab. Converted papers are kept in a collection folder, named as a study is: `claimset/<collection>/<name>.claimset.json`, with the reading view beside it (`<name>.claimset.txt`). The name is made from the source's identifier, so that a conversion of the same source has the same name in every repository and no two sources share a name:
  1. the identifier type (`converted_from.id_type`), a hyphen, and the identifier (`converted_from.source_id`) without a prefix (`doi:`, `arXiv:`) or a resolver address (`https://doi.org/`, `https://arxiv.org/abs/`); for arXiv, followed by the version (`converted_from.source_version`) when the identifier does not end with it;
  2. in lower case, with every run of characters other than the letters a–z and the digits 0–9 replaced by one hyphen, and no hyphen at the end.

  The DOI `10.1016/j.ejor.2022.04.040` gives `doi-10-1016-j-ejor-2022-04-040`; the arXiv identifier `2401.01234` with version `v2` gives `arxiv-2401-01234v2`. Other files a repository keeps for the conversion (the source text its `source_spans` point into, a conversion report) carry the same name with their own extension. Warning 31 flags a converted paper in a `claimset/` folder whose file is named otherwise.
- The `claimset/` folder is left out of package builds (lab record §3.3).

### 2.5 Submission package

A submission is a **package**: the claimset with the files its artefacts name that are not already held elsewhere (figures, specimen images, recordings, bulk data, instruments, stimuli). The package is a BagIt bag (RFC 8493), a folder or a zip of one, whose manifest lists every file with its SHA-256 checksum:

```
bagit.txt
manifest-sha256.txt
data/
  paper.claimset.json
  files/fig3.png
  files/results-per-series.csv
  files/interview-guide.pdf
```

- An artefact held in the package has a `uri` relative to `data/` (`"files/fig3.png"`) and `version` `sha256:` followed by the file's checksum, which equals the manifest's.
- An artefact held elsewhere (a commit on a git host, a data repository, a DOI) is not packaged: it keeps its `uri` and `version` as before, and the platform archives it (§3.6).
- A package with no files is the claimset file alone, which may be submitted as it is.
- At publication the platform stores each packaged file by its checksum, so a correction or revision that keeps a file stores it once, and replaces its relative `uri` with the file's permanent location (§2.2). A published file names no relative paths.
- In a repository, a paper's files live in its study folder, `claimset/<study>/files/` (§2.4), and the package is built from there (`claimset package`, which copies the files, sets each `version` and writes the manifest).
- A converted paper packages a file taken from its source (a figure, a supplementary table) only under the source's licence, which conversion already requires to permit derivatives (§3.1); the artefact names its `reference` and `locator` in the source.

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
| `declarations` | object | Funding, competing interests, ethics and consent (below). Recommended for native papers (warning 29); in converted papers, what the source declares |
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

**`declarations`.** What a reader needs to weigh the work beyond its evidence, as its responsible people declare it. Native papers: the agent asks the guarantor (principle 10), and records what they say.
- `funding`: list of `{funder, funder_id, award}`: the funder's name, its identifier (a ROR ID such as `ror:00k4n6c32`, or a Crossref Funder Registry DOI), and the award or grant number. An empty list declares no funding.
- `competing_interests`: text, self-contained; `"None declared."` when there are none.
- `ethics`: list of `{body, reference, decision}`: the ethics committee or review board, its reference, and `decision` `approved` · `exempt` · `not_required`. An empty list when the work needs no ethics review.
- `consent`: text, for work with human participants: how consent was obtained, and for what (participation, publication of excerpts).

Contributors are listed in the top-level `contributors` list (§3.11), not in the paper object.

#### 3.1.1 Exploration summary

`exploration` tells readers how much was tried to reach the published claims, so positive results can be weighed against the number of attempts. It is exported from the lab record.

```json
"exploration": {
  "lab_record": "A10",
  "questions": 2,
  "hypotheses": {"total": 10, "supported": 3, "not_supported": 2, "inconclusive": 1, "revised": 2, "abandoned": 1, "open": 1,
                 "from_exploration": 4},
  "tests": {"total": 5, "supported": 2, "not_supported": 2, "inconclusive": 1},
  "runs": {"total": 214, "confirmatory": 18, "exploratory": 196, "failed": 11}
}
```

- `hypotheses` counts recorded hypotheses by their final status; `open` counts those never closed. `total` is the sum of the status counts.
- `hypotheses.from_exploration` (optional, also in `covers.lab.hypotheses`): how many of the `total` hypotheses were recorded as suggested by exploration (lab record: `origin: from_exploration`). It is not a status and is not part of `total`. A hypothesis that exploration suggested is tested only by data the exploration did not use, so readers weigh its test by what was new.
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
| `kind` | enum | Required. What sort of thing the term names (below) |
| `members` | list | For a term that names a group: the IDs of the definitions of its members, each of the same `kind` as the group (below) |
| `part_of` | list | The definitions of the wholes this term is a part of: a parameter or component of a method, a subset of a data set, a subgroup of a population, a subscale of an instrument (below) |
| `broader` | list | The definitions of the more general terms this term is a kind of: a variant of a method, a subtype of a condition, a narrower category (below) |
| `assumptions` | list of claim IDs | Assumption claims (§3.8) that the thing the term names rests on wherever the paper uses it: a model's error distribution, a design's randomisation, a data set's representativeness (below) |
| `source` | object or string | Where a borrowed term is defined: a reference ID (`B3`, §3.13), `{target, id_type}` as for cited support (§4), or a global claim or definition ID of another claimset |
| `identifiers` | list | Registry identifiers of the thing the term names, each a compact identifier `prefix:local` (§4): a chemical (`chebi:15377`, `inchikey:…`), an organism (`ncbitaxon:9606`), a gene or protein (`hgnc:1100`, `uniprot:P04637`), an antibody or cell line (`rrid:AB_2138153`), a data set (`doi:10.…`), a court decision (`ecli:EU:C:2014:317`), a concept (`wikidata:Q…`). `source` says where a term is defined; `identifiers` say which thing it is |

**Kinds.** The kind says what a term names, whatever role it plays in a study; the role is given by the setup (§3.4) and the claim's `about` (§3.3.5). One term has one kind.

| Kind | The term names … | Examples |
|---|---|---|
| `method` | a method, model, algorithm, estimator or analysis procedure, with the settings this paper uses | ETS with AICc selection; a logistic regression; a proposed forecasting model |
| `intervention` | something applied to, or experienced by, the cases: a treatment, programme, policy, exposure or event | a nurse-led discharge programme; a minimum-wage increase; shift work |
| `population` | the people, organisations or other units sampled or studied | registered nurses in rural hospitals; UK listed firms |
| `data` | a data set, corpus, document collection or data-generating process | the M3 monthly series; 34 interview transcripts; a simulated AR(1) process |
| `setting` | a place, institution, period or circumstance in which cases are observed | two district hospitals in Wales; 2010–2024 |
| `design` | how a study or evaluation is arranged: trial or survey design, evaluation scheme, sampling and data collection, analytic approach | rolling origin with 8 origins; a cluster-randomised trial; purposive sampling with semi-structured interviews; reflexive thematic analysis |
| `variable` | what is recorded or observed about each case, and how it is scored or aggregated: a measured quantity, score, outcome, category, judgement, or the presence of a theme | reaction time in ms; 30-day readmission; yield in %; MASE averaged over series; grammaticality judged by native speakers; appeal outcome; presence of the theme "loss of clinical identity" |
| `concept` | a construct, property, phenomenon or theme the paper names, studies or finds | role ambiguity; forecastability; "loss of clinical identity" |
| `theory` | a theory or theoretical framework the paper uses or proposes | role theory; the resource-based view |
| `other` | anything else; a term often given this kind is a sign that the list needs a value | |

A term the paper introduces as a finding (a theme, a typology, a newly named phenomenon) is a `concept`. The paper's own method or model is a `method`, defined in full as below. An instrument (a questionnaire, a scale, an interview guide, a coding frame) is defined as the `variable` it records, or as the `design` it is part of, and the document itself is an artefact (§3.6). (The kind `variable` was `measure` in v0.30 and v0.31.)

**Dimensions.** A term by which observations are distinguished (the method, the data set, the participant group, the temperature, the year, the language) is a **dimension** of the observations that name it (§3.4). A dimension whose values are terms of the paper (the methods compared) is a group whose `members` are those terms; a dimension whose values are literals (a temperature, a year) is a `variable` or `setting` whose definition gives its unit.

**Groups.** A term that names a group of things the paper also names one by one (the benchmark methods, the variants of a model, the data sets of an evaluation) is defined once as the group, with `members` listing the definitions of its members, and has the same kind as its members: `{"id": "D26", "term": "the benchmark methods", "kind": "method", "definition": "ETS, ARIMA and the Theta method, as the paper runs them", "members": ["D8", "D9", "D10"]}`. A claim's `about` names the group when the claim is about the group as a whole and a member when it is about that member; a claim may name a group as its subject and one of its members as its comparator. Membership lets a reader find the claims about a group when looking for one of its members; it does not make a claim about the group a claim about each member. "The PIC(*,S,N) variants are the most robust" says nothing of each variant alone, while "CES has the lowest error of the benchmark methods" says CES beats each: the statement says which. A group is not a member of itself, directly or through other groups.

**Parts and narrower terms.** Besides membership, two links say how one term stands to another; each is a list of definition IDs on the term that is the part or the narrower one:
- `part_of`: the term is a part of each listed whole. A parameter or component of a method (the complex smoothing parameter and the information potential of CES), a subset of a data set (M3 monthly of M3), a subgroup of a population, a subscale of a questionnaire, a functional group of a compound, an article of a statute, a morpheme of a construction. A part may have a kind of its own: a parameter whose estimated values the paper reports is a `variable`, a component it names and makes claims about a `concept`, a data subset is `data`.
- `broader`: the term is a kind of each listed term. A variant of a method (seasonal CES of CES), a subtype of a disease, a narrower legal category. A narrower term has the kind of the broader one.
A group's members, a whole's parts and a term's narrower terms are three different links: ETS is one of the benchmark methods, not a part or a kind of them; the smoothing parameter is part of CES, not a kind of it; seasonal CES is a kind of CES, not a part of it. Like membership, the links let a reader looking for a term find what is said about its parts and narrower terms, and say nothing more: a claim about a parameter of CES is not a claim about CES as a whole, and a claim about seasonal CES is not a claim about every variant of CES.

**Assumptions of a term.** An assumption that holds wherever the paper uses a method, design or data set ("the errors of ETS are Gaussian", "allocation was concealed", "the sample is representative of UK retailers") is listed once, on the term's definition, not again on every claim. A claim resting on evidence with a `setup` inherits the assumptions of the terms that setup names, as follows:
- the terms in the roles `cases`, `procedure`, `lens` and `context` (the data, the design, the variables, the setting): every claim resting on the evidence inherits their assumptions, since every result of the study rests on them;
- the terms in the role `studied` (the methods, models or interventions compared): a claim inherits the assumptions only of its own subject, the definition its `about.subject` names, when the setup names it (directly, as a member of a group the setup names, or as a group whose members it names all, §3.2 *Groups*); when the subject is a group, the claim inherits the assumptions of each of its members. The assumptions of the comparator and of the other methods studied are not the claim's: "VETS D has the lowest error of the five models" rests on the error assumptions of VETS D, not on those of ETS, with which it is compared;
- a claim without `about` inherits the assumptions of every term its evidence's setup names.
A narrower term inherits the assumptions of its broader terms. A claim still lists in its own `assumptions` those that hold for it alone, including the assumptions of a comparator when the claim does rest on them. The reading view shows inherited assumptions with the claim, marked as inherited. (Revised in v0.33.)

Parameters, components and variants get definitions of their own only when something in the claimset names them apart from the whole: a claim, an `about` slot, an observation's variable or condition, a setup. Otherwise they are the named symbols of the whole's definition (below).

A definition marks the other terms of the paper it uses, as statements do (`"Seasonal [[D1:CES]] with a second [[D6:information potential]] for the seasonal component"`), so that a reader can follow what it rests on.

A definition explains in full only a term the paper introduces or uses in its own sense. A term the paper takes from elsewhere (it cites where the term comes from) is a **borrowed term**: its `definition` names the standard meaning in one line and states only what this paper specifies (e.g. `"Mean scaled interval score; here of the 95% prediction interval"`), and `source` says where the term is defined. A term the paper uses without defining or citing it, in its usual sense, needs no definition. A term the paper defines in its own words that is in fact a standard term, or another claimset's term, says so: by `identifiers` when the thing has a registry identifier, or by `source` naming the other claimset's definition by its global ID; its definition then states what this paper specifies, as for a borrowed term. Readers and the platform then treat the two as one term.

**The paper's own method or model.** An algorithm is defined by its steps, as pseudocode in a fenced block (§8 rule 6), in the paper's order, with each input, output and parameter named. Its definition gives the equations that define it, as the paper writes them (LaTeX, §8 rule 6), with each symbol named: every equation of the method or model, not a description of them ("the weights being powers of …" is not a definition of a weighted sum; the sum is). A variant the paper introduces (seasonal, multivariate, with exogenous variables) is defined with its own equations in full, never as "the same equations at lag $m$": where its parameters and lags enter is what the variant is. When the paper states a method in more than one form (a recursion and a weighted sum), the definition gives each form. A form the paper derives from another (an underlying state-space or ARIMA model) is not part of the definition: like every result of a derivation (a property, an equivalence, a condition), it is a claim (§3.3).

### 3.3 Claim

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `C1` … |
| `statement` | string | Self-contained sentence (rules in §8) |
| `about` | object | The claim's atom as fields: `subject`, `relation`, `object`, `comparator`, `conditions` (§3.3.5). Required for principal claims; recommended for supporting and secondary claims; absent on assumption claims |
| `type` | enum | `descriptive` · `associational` · `causal` · `predictive` · `interpretive` · `formal` · `methodological` (choosing: §3.3.1) |
| `role` | enum | `principal` (a contribution) · `supporting` (one of the paper's own claims that a contribution rests on, directly or through other claims, by `depends_on` or `assumptions`) · `secondary` (a finding of the paper that no contribution rests on: a side result, a negative or null result reported in its own right) |
| `novelty` | enum | `no_prior_found` · `replication` · `extension` · `refutation`. Not on assumption claims (§3.8) |
| `scope` | object | `population`, `setting`, `geography`, `language`, `period`, `data_type`, `conditions` (list), `excludes` (list: where it explicitly does not apply). Describes this claim only (§3.3.2) |
| `support` | list | Kinds of support (§4); at least one entry |
| `assumptions` | list of claim IDs | Assumption claims (§3.8) this claim relies on |
| `limitations` | list of limitation IDs | Threats to this claim within its scope (§3.9) |
| `assessments` | list | Strength assessments (§5); not for formal or assumption claims |
| `verifiability` | enum | Platform-assigned (§5); optional in submissions |
| `verifiability_flags` | list | Platform-assigned (§5); optional in submissions |
| `warrant` | warrant ID | Reasoning behind the claim; required for `interpretive` |
| `depends_on` | list of claim IDs | Claims within this paper that this one relies on |
| `relations` | list | Typed links to claims or works outside this paper, and to other claims of this paper (§3.7); required for refutations and replications |
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

When `about.conditions` names a defined population, data set or setting (§3.3.5), scope does not restate it; scope then records only what narrows the claim further (a horizon, a subgroup, an exclusion).

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
- `quantities`: when the rules are expressions, the results they are evaluated on, each `{quantity, conditions, field, run}`; `quantity` is the `variable` of the results it binds (§3.4), and they correspond to results in the claim's evidence.
- `supported`, `not_supported`: the two rules: expressions over quantities and numeric parameters (lab record §5.7.3), or text saying what observation decides.
- `verdict`: `computed` (the rules are expressions and the outcome was computed from the recorded results by the lab tool) · `judged` (the rules are text and the outcome was stated by the writer, who said which rule the evidence met).
- `runs`: how many runs decided the test. `lab_plan`: the plan's lab-record ID, when there is one. `registration`: the URI of an external pre-registration, when there is one.
- `addresses`: optional list of the global IDs of review comments (`doi:…@v1/RV1.c6`) whose resolution test this test ran (revision §4.2). The test's rules are then the comment's: `supported` its `withdrawn` rule, `not_supported` its `upheld` rule.
- `flags`: optional list of weaknesses of the test, from the lab record (lab record §5.7.5): `code_changed_after_plan` (the analysis code changed between the plan and the test) · `holdout_used_elsewhere` (data held out for the test was also used in other runs) · `reproduces_prompting_result` (a result the decision rule uses has the same quantity, conditions and value as a result of a run that suggested the hypothesis: the test may have repeated the exploration rather than tested the hypothesis on anything new).

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

**Secondary claims in converted papers.** The paper's side results are kept as `secondary` claims only when something rests on them or they qualify what the paper establishes. A secondary descriptive claim resting on evidence (a result; a conclusion the paper draws from results is not one, *One claim, several places* above) is written when a claim depends on it, a warrant names it as a premise or an open question concerns it; or when it states a loss, a limit or a null result of the paper's own method ("M has higher MSIS than N on short series", "M does not improve on N on the yearly data"). Any other result the paper reports, including a pattern its results section reads off a table, is not a claim: it stays in the result table of its evidence, which records it whole (§3.4, *Tables and claims*). Side remarks, illustrations and explanations that the paper neither tests nor uses for a contribution are left out.

#### 3.3.5 What a claim is about

`about` records the claim's atom (§8 rule 1) as fields, so that claims can be found and compared by what they are about rather than by their wording. It is the same atom a writer forms before writing the statement; the statement says it in words, and `about` names its parts.

```json
"about": {"subject": "D2", "relation": "has lower error than", "object": ["D8", "D9"], "comparator": "D3", "conditions": ["D5"]}
```

| Field | Value | Notes |
|---|---|---|
| `subject` | a slot value | What the claim is about: the method, intervention, population, phenomenon or object of which something is asserted. One subject; a named group ("the univariate methods") is one subject, defined as such |
| `relation` | text | What is asserted of the subject, as a short verb phrase in the paper's own terms: `has lower error than`, `increases`, `is associated with higher`, `is experienced as`, `is stationary when`, `does not reduce`. A negation belongs here. Never an ID. No controlled list: the claim's `type` gives the class of the relation |
| `object` | a slot value, a list of `variable` definition IDs, or `null` | What the relation links the subject to: an outcome, a property, a variable, a theme. `null` when the relation needs none (`exists`, `is identifiable`). A list only for atoms grouped by variable (§8 rule 2): one subject against one comparator under one set of conditions, on several variables |
| `comparator` | a slot value or `null` | What the subject is compared with; at most one |
| `conditions` | list of slot values | The cases, setting and other conditions under which the claim holds: the data set, the population, the period, the horizon, the parameter region. Empty when the claim holds without conditions |

**Slot values.** A slot holds the ID of a definition (`D5`) or, for a thing the paper does not define, a short noun phrase (`"the seasonal naive method"`). A term that a setup of the claim's evidence names, or that the paper defines, is given by its ID, not as text. Every definition ID in `about` is marked in the statement (`[[D5:M3 monthly]]`), so the two cannot drift apart. Text slots are resolved by the platform like printed references (§11); they make a claim findable, IDs make it exact.

**By type of claim.**

| Claim | subject | relation | object | comparator | conditions |
|---|---|---|---|---|---|
| descriptive, a comparison of methods | the method (`method`) | `has lower error than` | the error measures (`variable`) | the other method | the data set, the horizon |
| causal, a trial | the intervention | `reduces` | the outcome (`variable`) | the control condition | the population, the setting |
| associational, a survey | the exposure or property | `is associated with higher` | the outcome | `null` | the population, the period |
| interpretive, a qualitative study | the phenomenon or the participants' experience (`concept` or `population`) | `is experienced as` | the theme (`concept`) | `null` | the sample, the setting |
| formal | the model or object of the theorem | `is identifiable when` | the property (`concept`) or `null` | `null` | the theorem's conditions, as text or assumption-claim terms |
| methodological | the method or implementation | `computes` | what it computes | `null` | the inputs for which it does |

**Conclusions over atoms.** A conclusion the paper draws over several atoms (§8 rule 2) has an `about` of its own: its subject is what the conclusion is about ("the paper's method", or a defined group), its conditions those shared by the atoms it rests on.

**Interpretation is not flattened.** An interpretive claim keeps its reasoning in its warrant; `about` only names what the claim is about. A finding that is not one relation between two things (a typology, a process with stages, a model of how a phenomenon unfolds) is defined as a `concept` holding that finding, the claim's `object` names it and its `relation` is `is characterised by`. `about` never forces a finding into a shape it does not have, and never changes what the statement says.

### 3.4 Evidence

Common fields: `id`, `profile`, `description`, `artefacts` (list of artefact IDs), `data_origin` (`primary` · `secondary`), `preregistered` (boolean), `availability` (§3.10), `models` (list, below), `results` (list, below), `execution` (object, below), `lab_runs` (list of lab-record run IDs, e.g. `RN4` or `AC2.RN4`, that produced this evidence), `tables` (list, below), `derived_from` (list, below), `source_spans` (locations of the evidence in the source text, for converted papers).

**Derived evidence** (`derived_from`, optional): the evidence objects of this paper whose data this evidence re-uses, each `{evidence, how}`, `how` one of `reanalysis` (the same cases analysed differently: a robustness check, a sensitivity analysis, another model), `subset` (part of the cases: a subgroup analysis, a restriction), `pooled` (several evidence objects combined: a pooled analysis, a meta-analysis within the paper), `extension` (the same cases with more added). Two results whose evidence derives from the same evidence are not independent, whatever their number; the platform counts them once (§11).

**Observations.** What a study found is recorded as observations: each the value of a variable, observed on something, under conditions that tell it apart from the others. A reaction time per participant and condition, the expression of a gene per cell line and treatment, the yield of a reaction per catalyst and temperature, whether a construction is attested per language and period, the outcome of an appeal per court and year, the accuracy of a model per benchmark, whether a theme appears per participant: each is an observation, and every field's results take this one form, so that readers can query results instead of parsing prose, and compare them across papers. A single observation is a result entry (`results`); observations that the paper reports as a table are a table (`tables`), a compact form of the same thing. Results and tables appear in evidence of any profile except `formal`.

**Results** (`results`): `{id, variable, value, unit, interval, n, conditions, read_from, artefact}`
- `id`: `R1`, `R2`, … unique within the evidence object; referenced from elsewhere as `<evidence id>.<result id>`, e.g. `E23.R2`.
- `variable`: what is observed: the ID of a definition of kind `variable` (§3.2), or of a `concept` whose presence is observed (a theme in an excerpt), or text for one the paper does not define, e.g. `"D8"` or `"largest KL divergence per observation, mixture vs exact"`. (Was `quantity` until v0.31.)
- `value`: one of
  - a number;
  - a category: the ID of a definition (an outcome the paper defines, such as `"upheld"`) or a short word or code (`"present"`, `"B"`);
  - a boolean (attested or not, significant or not);
  - an **excerpt**, `{text, locator, language, gloss, translation}`: words from the data that are themselves the observation (an interview passage, a passage of a judgment, a linguistic example). `text` is the words as the source gives them; `locator` where they are (a transcript and line, a document and paragraph, a corpus and sentence ID); `language` a BCP 47 tag when not English; `gloss` and `translation` for linguistic examples (Leipzig glossing rules). Excerpts from people's data carry no personal data beyond what the participants consented to publish (`declarations.consent`).
- `unit`: optional; a UCUM code (`ms`, `%`, `mg/L`, `Cel`, `1` for a dimensionless ratio) or, where UCUM has none, text. `interval`: optional `{level, lower, upper}`. `n`: optional, the number of cases the value rests on.
- `conditions`: the observation's coordinates: an object whose keys are dimensions and whose values are where the observation lies on each, e.g. `{"D26": "D9", "D5": "D14", "horizon": 12}`. A key is the ID of a definition or text. A value is the ID of a definition, a number, text or a boolean. When a key is a group (§3.2, *Groups*), its value is one of the group's members; when it is a `variable` or `setting` with a unit, its value is a literal in that unit.
- `read_from`: optional `{artefact, precision}`, when the value was read off a figure or a recording rather than taken from data or a table: `artefact` is the figure or recording, `precision` how exactly it could be read (a number in the observation's unit, or text). Values read off a figure are recorded only with it, so a reader knows they are approximate.
- `artefact`: optional; the artefact holding the full data this observation summarises.
A single-estimate study may use `estimand`, `estimate`, `interval` and `n` instead: one observation whose variable is the estimand.

**Tables** (`tables`): observations that the paper reports as a table, recorded as one. `{id, description, conditions, row_labels, columns, units, rows, artefact, source_spans}`
- `id`: `T1`, `T2`, … unique within the evidence object; referenced from elsewhere as `<evidence id>.<table id>`, e.g. `E2.T1`.
- `description`: what the table reports, self-contained, e.g. `"Mean and median forecast errors of six methods on the M3 monthly series, horizons 1 to 18"`.
- `conditions`: optional coordinates shared by every cell, as for results.
- `row_labels`: the dimensions that distinguish rows, each a definition ID or text, e.g. `["D26"]`; may be empty for a single row.
- `columns`: one per value column: the column's variable (a definition ID or text), or `{variable, conditions}` when columns also differ along a dimension, e.g. `{"variable": "D8", "conditions": {"horizon": 1}}`. Columns are distinct.
- `units`: optional list of units, one per column (`null` where none).
- `rows`: list of `{id, label, values}`. `id`: `r1`, `r2`, … unique within the table; a row is referenced as `<evidence id>.<table id>.<row id>`, e.g. `E2.T1.r3`. `label`: the row's value on each row label, in order: a definition ID (a member of the dimension when it is a group), a number or text. `values`: one value per column, in order, of any type a result value may have; `null` where the paper reports no value.
- `artefact`: optional artefact ID of the source of the full table. `source_spans`: optional locations of the table in the source text (converted papers).

Each cell is one observation: its variable is the column's; its conditions are the table's, the row's labels and the column's. Intervals, ranks and test statistics are columns of their own (`{"variable": "D8", "conditions": {"statistic": "95% interval, lower"}}`). A table records what the paper reports; it is not split into result entries, and result entries are not repeated in a table.

**Contrasts.** A value that compares two positions on one dimension (a difference in means, a ratio, an odds ratio, a standardised effect size, the p-value of a comparison, a rank difference) is an observation of the pair. Its conditions name one position as usual and the other under the reserved key `comparator`; its variable says what the contrast is (a defined variable such as "difference in mean MASE", or text). `{"variable": "D11", "conditions": {"D26": "D1", "comparator": "D2", "D5": "D14"}, "value": -0.013, "interval": {"level": 0.95, "lower": -0.021, "upper": -0.005}}` is CES against ETS on the M3 monthly series. The comparator lies on the dimension whose value it replaces: a member of the same group, or a value of the same variable. A table of contrasts has `comparator` in its conditions, its row labels or its columns' conditions. The atom of a claim resting on a contrast has the same comparator (§3.3.5).

**Tables and claims.** The table is the record of its values; a claim is the record of what the paper concludes from them. A claim resting on a table states the pattern the paper draws (which method, treatment or group is higher or lower, on which variables, under which conditions, including where the paper's own is worse) and cites the table or the rows that hold the values (`E2.T1`, `E2.T1.r3`); it repeats a value only when the conclusion turns on that value ("selected for 6 of the 11 groups", "coverage 0.947, the closest to 95%"). A claim that only retells a table's values, with no conclusion of the paper's own, is not written: the claim that uses the table cites it directly. A comparison stated this way is still one claim (§8 rule 2). A pattern that no claim, warrant or open question uses and that states no loss, limit or null result of the paper's method is not written either (§3.3, *Secondary claims in converted papers*).

```json
"tables": [{"id": "T1", "description": "Mean MASE and MSIS of three methods on the M3 monthly series, horizons 1 to 18",
  "conditions": {"D5": "D14"}, "row_labels": ["D26"], "columns": ["D8", "D9"], "units": ["1", "1"],
  "rows": [{"id": "r1", "label": ["D1"], "values": [0.861, 6.95]},
           {"id": "r2", "label": ["D2"], "values": [0.874, 6.71]},
           {"id": "r3", "label": ["D3"], "values": [0.902, null]}]}]
```

Here D26 is the group of compared methods (members D1, D2, D3), D5 the group of data sets, D14 the M3 monthly series, and D8 and D9 the error measures, of kind `variable`. The same form holds a table of a psychology experiment (rows: condition × group; columns: mean reaction time, its standard deviation), of a chemistry paper (rows: catalyst; columns: yield at each temperature, `{"variable": "D4", "conditions": {"D6": 350}}` with D6 the temperature in `Cel`), of a linguistic survey (rows: language; columns: whether each construction is attested, boolean values) or of a legal study (rows: court × year; columns: the number of appeals by outcome). A participant or sample flow (screened, excluded for each reason, analysed) is a table of counts whose rows are the stages, so a flow diagram is a figure that shows it (§3.6).

**Excerpts and qualitative evidence.** A qualitative finding rests on excerpts. Each excerpt the paper gives for a theme is an observation whose variable is the theme's presence (or the theme itself), whose conditions name the participant or document, and whose value is the excerpt: `{"id": "R3", "variable": "D7", "conditions": {"participant": "P07", "D3": "D2"}, "value": {"text": "I used to be the stroke nurse. Here I'm just a nurse.", "locator": "transcript P07, lines 112-113"}}`. A linguistic example is an excerpt with its gloss and translation, a cited passage of a judgment an excerpt with its paragraph. (The `exemplars` of the `qual` profile until v0.31 are recorded this way.)

**Setup** (`setup`; recommended for evidence of every profile except `formal`): the frame of the study, its parts each recorded once as a definition (§3.2) and named here by role, so that claims, evidence objects, reviews and other papers can refer to each part on its own. `{studied, cases, procedure, lens, context}`, each a list of definition IDs; roles that do not apply are left out:
- `studied`: what the study is about or compares: methods and models, interventions and exposures, the phenomenon of interest (in forecasting: the competing methods, each with the settings that define it; in a qualitative study: the experience or phenomenon explored).
- `cases`: what or whom it is studied on: data sets, populations, samples, participants, sites, documents (the series collections with their frequency and size; a simulation's data-generating process; the interviewees with how many there were).
- `procedure`: how the cases are obtained and analysed: the evaluation scheme (fixed or rolling origin, origins, horizons, refits), the trial or survey design, the sampling and data collection, the analytic approach (thematic analysis, grounded theory, a regression model used as the analysis).
- `lens`: what is recorded about each case and through what it is read: the variables observed (error measures, outcomes, scores, judgements) with their aggregation, instruments and scales, the coding frame, a theory used to read the data.
- `context`: where and when, when it bounds the study: the setting, institutions, period.

The roles are those of the frames used to specify studies across fields, under names that fit any design:

| Role | Forecasting benchmark | Trial (PICO) | Qualitative study (SPIDER) | Simulation |
|---|---|---|---|---|
| `studied` | the competing methods | intervention, comparator | phenomenon of interest | the methods or models run |
| `cases` | the series collections | population | sample | the data-generating process |
| `procedure` | the evaluation scheme | trial design | design, research type | the experimental design (replications, parameter grid) |
| `lens` | error measures | outcomes | evaluation (what was explored, and how it was read) | the variables computed |
| `context` | rarely needed | setting | setting | not applicable |

A part's role is the role it plays in this study; its `kind` (§3.2) is what it is. They usually agree (a `variable` in `lens`, a `population` in `cases`) but need not: a method can be `studied` in one paper and part of the `procedure` in another (a regression used as the analysis), and a theory can be `studied` or be the `lens`. The boundary most often in doubt is between `procedure` and `lens`: `procedure` is what was done to obtain and analyse the cases; `lens` is what was recorded about them and the terms in which they were read.

Definitions of borrowed parts name their `source` (a data set, a published method, a standard measure or instrument, an analytic approach); the definition says what this paper specifies (the subset, the settings, the horizon, how the approach was applied). Evidence objects that share a part point to the same definition, and a claim's statement marks the parts it names (`[[D7:rolling origin]]`). `setup` adds no content of its own: the description still says, in a sentence, what was done. The closed lists of the `qual` profile (`data_sources`, `analytic_approach`) are kept: they classify, and the setup's definitions say what exactly was done.

```json
"setup": {"studied": ["D1", "D2", "D3"], "cases": ["D5", "D6"], "procedure": ["D7"], "lens": ["D8", "D9"]}
```

The keys of v0.23 to v0.29 map to these: `compared` → `studied`, `data` → `cases`, `design` → `procedure`, `measures` → `lens`.

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
`data_sources` (list: `interview` · `focus_group` · `ethnography` · `observation` · `document` · `other`), `n_participants_or_cases`, `sampling`, `analytic_approach` (`thematic` · `grounded_theory` · `ipa` · `discourse` · `framework` · `case_study` · `other`), `reflexivity` (text). Excerpts are results (above).

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
| `kind` | enum | `code` · `data` · `protocol` · `proof` · `model` · `software` · `figure` · `table` · `prompt` · `trace` · `manuscript` · `lab_record` · `instrument` (a questionnaire, scale, test or interview guide, as given to participants) · `stimulus` (the materials presented in an experiment: items, texts, images, sounds) · `codebook` (a coding frame with its codes and rules) · `recording` (audio or video) · `other` |
| `withheld_events` | list | For `lab_record`: `{event, reason, comment}` for each lab event whose body is withheld (lab record §8) |
| `uri` | string | |
| `version` | string | Version or content hash |
| `licence` | string | SPDX identifier. Recommended for the paper's own artefacts and for every file of the submission package (§2.5); an artefact taken from a cited work (`reference`) and not packaged carries none, since its licence is the cited work's, recorded where that work is described (warning 9) |
| `availability` | object | §3.10 |
| `description` | string | Required for `figure` and `table`: what it shows, self-contained |
| `underlying_data` | artefact ID | For `figure`: the data plotted |
| `function` | enum | For `figure`, required: `data_plot` · `specimen` · `diagram` (below) |
| `shows` | list | For a `data_plot`: the observations it shows: result, table or row references (`E2.R1`, `E2.T1`, `E2.T1.r3`) or a bulk data artefact (`A4`) |
| `encoding` | object | For a `data_plot`, optional: which dimension or variable each visual channel carries, `{x, y, colour, shape, size, panel}`, each a definition ID or text |
| `columns` | list | For bulk data (`data`, `table`) in a tabular file: `{name, variable, dimension, unit}` for each column: its name in the file, and the definition (or text) of the variable it holds or the dimension it lies on; after W3C CSV on the Web (CSVW) |
| `locator` | string | For an artefact taken from a cited work: where in it, e.g. `"Figure 3"`, `"Supplementary Table S2"` |
| `reference` | reference ID | The cited work this artefact is taken from (a data set, software or an implementation published by others), as an entry of `references` (§3.13) |
| `archive_uri` | string | A persistent copy of exactly this version: a Software Heritage identifier (SWHID), a DOI or an archived URL. Optional; the platform adds it at publication for public artefacts pinned to a commit (§11) |

**Figures.** A figure is recorded by what it does:
- `data_plot`: a view of observations (a chart, a forest plot, a dose–response curve, a spectrum, a flow diagram of counts). It `shows` the observations, which are recorded as results, tables or bulk data; claims cite the observations, and may cite the figure besides. Values that exist only in the plot are read off it and recorded with `read_from` (§3.4).
- `specimen`: an image or recording that is itself evidence (a micrograph, a gel or blot, a scan, a photograph of a document or object, an audio sample). What the paper reads from it is recorded as observations whose `read_from` names it (a band present, a structure counted, a script identified).
- `diagram`: shows an idea, not data (a model's structure, a mechanism, a pipeline, a conceptual framework). It is never evidence: what it shows belongs in definitions (a method, a procedure) or concepts, written out in text, and support does not cite it.
A reading model learns little from an image; it learns from the observations. Images are files of the submission package (§2.5), never embedded in the claimset.

**Bulk data.** Data larger than the paper reports (results per series or per participant, raw measurements, simulation outputs) are artefacts in tabular files (CSV), with `columns` naming the variable or dimension of each column. Their rows are observations like the file's own, and are cited by RFC 7111 fragment, `A4#row=12` (a row) or `A4#row=12-40` (rows). What the claims rest on is recorded in the file as results and tables; bulk data supports and extends them.

Tables are stored as data (CSV or JSON), not images. Prompts are stored verbatim, including system prompts. A `trace` is the log of an agent's run (prompts, tool calls, outputs) as recorded by the system that ran it; it may contain confidential material and uses the availability levels in §3.10. A `manuscript` is a human-readable version of the same work (e.g. a LaTeX or PDF draft) from which the claimset was prepared. An artefact taken from prior work (a public data set, a package the paper's evidence uses) points to its `reference`; its `uri` and `version` are given when the paper gives them.

A `uri` on a git host (GitHub, GitLab, Bitbucket, Codeberg and the like) names a full commit hash, e.g. `https://github.com/org/repo/tree/<40 hex digits>/path`, not a branch: a branch moves, so the link would later show other content than the evidence used (warning 21). A repository can still be deleted; `archive_uri` keeps a copy that outlives it.

### 3.7 Relations

Typed links from a claim to a claim or work outside this paper, or to another claim of this paper: `{type, reference, target, id_type, locator, archive_uri, self, note}`.

- `type`: `supports` · `contradicts` · `extends` · `refines` · `replicates` · `fails_to_replicate` · `supersedes` · `addresses` (the claim answers a problem, gap or limitation that the target states) · `explains` (the claim explains why the target's result holds) · `qualifies` (the claim states an exception, a boundary or a condition of the target).
- `reference`: the ID of the cited work in `references` (§3.13). Either `reference`, or `target` with `id_type`, is given.
- `target`: a global claim ID when the target is a claimset; otherwise the identifier of the work (paper, book, software release).
- `id_type`: `claimset` · `doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `curie` · `physical` · `claimset_local` (a claim of a paper in the same repository not yet published, §2.4) · `local` (a claim of this paper: `target` is its ID, e.g. `"C4"`).
- `locator`: optional; where in the target the related result is, e.g. `"eq. 29 and B.6"`, `"Section 4.2"`, `"function adam(), lognormal scale"`.
- `archive_uri`: required for `url` targets: an archived snapshot.
- `self`: true when the target work shares a human supervisor, guarantor or contributor with this paper.
- `note`: optional; one self-contained sentence saying what the target states that the relation is about, e.g. for `addresses`: `"Selection among ETS models often misidentifies the type of trend."` It describes the target; it is not a claim of this paper.

**Novelty** says what the claim is relative to earlier work, as far as the authors searched: `no_prior_found` (the authors found no earlier result stating the claim; this reports their search, not a proof that none exists), `replication` (it repeats an earlier result, on new data or by other means), `extension` (it carries an earlier result to new conditions), `refutation` (it shows an earlier result false). A `no_prior_found` claim cites the closest earlier result the authors found, when there is one, as a relation (`addresses` or `extends`) with a `note` saying how the claim differs; a reader then sees what was found and can judge the difference. For a converted paper, `no_prior_found` records that the source presents the claim as its own finding.

**Relations within the paper.** `depends_on` says that a claim's argument needs another claim as a premise. Claims of one paper stand in other relations too, and these are relations with `id_type: local`: an explanation `explains` the result it accounts for ("M wins on short series because its initial states are estimated jointly" explains "M has the lowest error on series shorter than 36 observations"); an exception or boundary `qualifies` the claim it limits ("M does not improve on N for intermittent series" qualifies "M improves on N on the retail data"); and the paper's results may `support`, `contradict` or `refine` one another. An explanation does not depend on the result it explains, and a result does not depend on its explanation: neither is the other's premise. The types `replicates`, `fails_to_replicate`, `supersedes` and `addresses` are for other works only.

Required relations: `novelty: refutation` ⇒ at least one `contradicts` or `fails_to_replicate`; `novelty: replication` ⇒ at least one `replicates` or `fails_to_replicate`. Recommended: `novelty: extension` ⇒ an `extends` relation. A refutation with `self: true` is a self-correction; the platform records it separately from self-citation.

### 3.8 Assumptions

Assumptions are claims whose sole support kind is `assumption`, so they sit in the claim graph and can be queried across papers. They are taken as given, not found, so they have no `novelty`. They carry extra fields:

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
| `level` | enum | Required. `public` · `platform` (registered users) · `audit` (accessible only to automated audit agents in a secure environment; human auditors optional) · `author_only` (never leaves the originating environment; can be audited only by an attested agent run there) · `unavailable` · `not_stated` (the paper does not say whether the material is available, or where; never a guess at another level) |
| `reason` | enum | Required unless `public` or `not_stated`: `personal_data` · `commercial` · `legal` · `ethics` · `security` · `other` |
| `content_hash` | string | Required unless `public`, `unavailable` or `not_stated`: cryptographic hash committing to the exact content |

Every evidence object and every artefact has an `availability` (rule 75): whether the material can be checked is what verifiability is computed from (§5), and readers filter on it. A paper that says nothing about it gives `not_stated`, which records a fact, not a default.

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

### 3.14 Open questions

Questions the paper raises and does not answer: a conjecture it states without testing, a case it could not settle, a direction it proposes for further work. They are not claims: nothing supports them and nothing rests on them, and they are not reviewed or scored. Recording them as questions keeps what is useful to a reader looking for open problems, without presenting it as a finding.

| Field | Type | Values / notes |
|---|---|---|
| `id` | string | `O1` … |
| `question` | string | Self-contained, as a question or an open statement: "Whether the polynomial check extends to multiplicative ETS models is not established." Statement rules apply (§8) |
| `claims` | list of claim IDs | Optional: the claims the question arises from |
| `source_spans` | list | Converted papers: where the source raises it |

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
- On any empirical kind: optional `results`, the specific observations the claim reports: result entries, tables, table rows, or rows of a bulk data artefact of the evidence (e.g. `{"kind": "simulation", "evidence": "E20", "results": ["E20.R1", "E20.T1.r2", "A4#row=12"]}`). Recommended whenever the statement quotes values, so readers can check each value against the recorded observation. A diagram (§3.6, *Figures*) is never among a support entry's `artefacts`.
- On `cited`: `reference` (the ID of the cited work in `references`, §3.13), or `target` with `id_type` (`doi` · `arxiv` · `handle` · `pmid` · `isbn` · `url` · `curie` · `physical` · `reference` · `claimset_local`) and `archive_uri` (required for `url`: an archived snapshot); optional `locator` (where in the cited work, e.g. `"Theorem 3.1"`); `self` (boolean, below). Where a claimset of the cited work exists, `target` may be its global claim ID.
- `id_type: curie`: a compact identifier `prefix:local` whose prefix is registered in the Bioregistry (which also resolves the prefixes of identifiers.org), for works and things that DOIs, arXiv and PubMed do not cover: legal sources (`ecli:` for court decisions, `eli:` for legislation), database records (`pdb:1abc`, `geo:GSE12345`), biological resources (`rrid:…`), standards and archival records. The same form is used by definitions' `identifiers` (§3.2).
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
- `availability_not_stated`: an evidence object or artefact the claim's support points to has availability `not_stated`; the claim is computed as `open` if nothing else limits it, and the flag says that this was not confirmed.
- `figure_without_data`: a supporting figure is a `data_plot` with neither `underlying_data` nor `shows`, or a `specimen` that no observation is read from (§3.6). (Revised in v0.32.)
- `unpublished_prompts`: a model used in the research design has no `prompts` artefact.
- `unrecorded_model_settings`: a model used in the research design has no `settings`, or its `record_source` is absent or `self_reported`.
- `unstated_support`: the claim's support is `unstated`.
- `lab_record_withheld`: the claim came from a lab record whose availability is not `public`.
- `lab_events_withheld`: the claim's hypothesis event, or a run in its evidence's `lab_runs`, is listed in the lab record's `withheld_events`.
- `unanchored_plan`: the claim's `planning` is confirmatory but `planned_at_anchor` is `none`.

The flags concern models used to produce evidence, not the model that wrote the paper. The platform recomputes verifiability and flags as availability changes (e.g. when an API model is retired).

## 6. Assessment by the platform

Platform assessments (AI reviewer agents, audits, replication checks, conversion reviews, independent re-derivation of formal claims, scope-consistency reviews) are stored in the database, not the file. Human review is an optional layer, except for sampled review of conversions.

Reviews of a published claimset, the author's responses, human resolutions of disputed comments, and the credibility scores and novelty assessments the reviewer gives are platform records beside the file, specified in the revision specification. A review never changes a published version; the author may answer it with a revision (§9).

## 7. Identifiers and source trust

**The claimset's DOI.** The platform mints a DOI for every published claimset, at publication, and registers it with DataCite under the platform's DOI prefix.

- **Form:** `doi:<prefix>/claimset.<year>.<number>`. `<year>` is the year of first publication; `<number>` is the claimset's sequence number among those first published that year, with at least six digits, zero-padded. Example: `doi:10.0000/claimset.2026.000123`, where `10.0000` stands for the platform's prefix throughout this specification. The suffix holds nothing else (no title, author, subject or source), so nothing in it can later become wrong.
- **One DOI for all versions.** Corrections and revisions keep the DOI; a version is named by appending `@v<n>`, and an object by appending its local ID: `doi:10.0000/claimset.2026.000123@v2#C3` (§2, §9). The DOI resolves to the claimset's page on the platform, which shows the latest version, its status and the list of earlier versions.
- **Converted papers** get a DOI of their own by the same rule; a claimset never takes its source's DOI (rule 30), and the source is recorded in `converted_from`. A conversion of a new version of the source is a new claimset with a new DOI (§9).
- **Never reused.** A DOI is never deleted or given to another claimset. A retracted or withdrawn claimset keeps its DOI, which resolves to the claimset and its status (§9).
- Before publication a submission has only the platform's internal submission ID (§2.2), which is never cited. In a repository, a paper that is not yet published is named by its path (§2.4).

The file records identifier type and venue facts for cited works only. Trust in a source is a derived score computed by the platform. Papers published on Claimset are scored by the same rules as all others.

## 8. Statement text rules

1. One assertion per statement. A claim is one **atom**: one *subject* (the method, model, group of methods, population or object the claim is about), one *relation* (what is asserted of it: is lower than, is stationary when, improves, is equivalent to), at most one *comparator* (what it is compared with; a named group, such as "the univariate methods", is one comparator), and one set of *conditions* (the data, setting, sample size, horizon or parameter region under which it holds). A statement with two subjects, two comparators or two sets of conditions is two or more claims, whatever its wording. Write the atom first and the statement from it; the atom is recorded in the claim's `about` (§3.3.5). A statement also holds more than one claim when one part could be true and another false. Signs of it: a semicolon; contrast (`but`, `while`, `whereas`, `although`, `however`); addition (`as well as`, `also`, `in addition`, `moreover`, `furthermore`, `not only`); explanation (`because`, `since`, `due to`, `thus`, `therefore`, `hence`, `which explains`), which usually joins a result to its explanation, two claims of different types linked by `explains` (§3.7); an assertion in a side clause (`, which …`) or in parentheses; a recommendation added to a finding (`should`, `recommend`); a claim citing results of more than one evidence object; and `and` joining two verbs. The grouping of rule 2 (one subject against one comparator under one set of conditions, better on some measures and worse on others) stays one claim.
2. **Granularity rule:** split a statement into separate claims when its parts differ in support, scope or status; keep it whole when splitting would lose meaning. A qualitative theme is one interpretive claim with a warrant; sub-themes become separate supporting claims only if they have their own evidence. Atoms are grouped into one claim only when they share subject, comparator and conditions and differ in the measure or summary: "M has a lower mean and median MASE than N, but a higher MSIS" is one claim, supported by the table holding the numbers, whose values it does not repeat (§3.4, *Tables and claims*). The same comparison on two data sets, against two comparators or for two methods is two claims. A conclusion the paper draws over several atoms ("M is the most robust of the variants") is a claim of its own, whose `depends_on` lists the atoms it rests on; it is not written by merging them. Results on which a method is worse are stated in the claim, never left out.
3. Self-contained: no anaphora or cross-references ("this", "the above", "as shown").
4. Qualifiers that bound where the claim holds (the population, setting, period, conditions, exclusions) belong in `scope`, not in the statement. Epistemic qualifiers ("suggests", "may", "is consistent with") and magnitude ("slightly", "by about a fifth") are part of what is claimed: they stay in the statement, worded as the paper words them, and confidence in the claim is recorded in its assessments (§5). (Revised in v0.34.)
5. Defined terms are marked inline with their definition ID: `[[D2:forecast skill]]`, in statements and in definitions. The text of a marker is plain: it holds no other marker (`[[D24:VETS F]]`, never `[[D24:V[[D27:ETS]] F]]`).
6. Mathematics is written in LaTeX, delimited by `$…$`. Algorithms in definitions are pseudocode in a fenced block (three backticks before and after).
7. No evaluative rhetoric ("novel", "significant", "remarkable"). "Statistically significant", the outcome of a test, is a result, not rhetoric.
8. No imperatives or instructions of any kind.
9. A claim is what the paper shows or uses. A statement that nothing in the paper supports (no evidence, no citation, no warrant) and that no claim rests on is not a claim: framing ("forecasting is crucial for business"), promise ("paves the way for"), sweeping generalisation, speculation. Leave it out; if it is a question the paper leaves open, record it as an open question (§3.14).

## 9. Versioning, corrections and revisions

- Every published version is immutable and permanently retrievable.
- **Order and IDs.** Claims are listed in reading order: principal claims, then supporting claims, then secondary claims and assumptions, each in the order of the paper's argument, since the reading view follows file order (§2). Before the first submission, IDs follow that order without gaps: `claimset renumber FILE` renumbers claims, warrants, definitions, limitations and open questions and rewrites every reference to them; a claimset exported from a lab record keeps its link to the record through `hypothesis.lab_event`, not through claim IDs. After publication IDs never change (revision §2, *Stable object IDs*): a revision places a new claim where it belongs in the file, with the next free number, and an ID that is dropped is not reused.
- A correction is submitted as a complete corrected file whose `paper.id` names the existing claimset and which carries a `correction_reason`. The platform computes the JSON Patch against the latest published version, records it in `changes` with the reason, assigns the next `version` and `previous_version`, and sets `status: corrected`. The DOI does not change; versions are distinguished by `@v<n>`.
- A revision is submitted as a complete file built from the current version, whose `paper.id` names the existing claimset and which has no `correction_reason`, together with a response to the unresolved review comments (revision §4–5). The platform records the patch as for a correction, with the comments each operation answers, sets `paper.answers`, and keeps `status: active`. Object IDs are stable across versions: an object that keeps its ID is the same object, refined, and an ID removed from the claimset is never used again.
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
12. `availability.level` other than `public` and `not_stated` ⇒ `reason` present; `audit`/`author_only`/`platform` ⇒ `content_hash` present. (Revised in v0.34: `not_stated`.)
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
25. `paper.id` is absent, except in a correction, where it names an existing published claimset and `correction_reason` is present, or in a revision, where it names an existing published claimset and a response accompanies the file. (Revised in v0.21.)
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
34. Every relation has `type` and either `reference` (resolving to an entry of `references`) or `target` with `id_type`; `id_type: url` ⇒ `archive_uri`. (Revised in v0.16 and v0.32: `id_type: local`, rule 71.)
35. `type: formal` ⇒ no `assessments`; `type: formal` with `formal_proof` or `argument` support ⇒ `verification` present.
36. (Revised in v0.10.) `verification` is consistent with `checks` as defined in §5; `machine_checked` ⇒ the formal evidence has `proof_artefact` and `checker`.
37. Formal evidence has no `theorem` field.
38. Result IDs are unique within their evidence object; every `results` reference in a support entry resolves within the evidence that entry points to: `E<n>.R<m>` to a result, `E<n>.T<m>` to a result table, `E<n>.T<m>.r<k>` to a row of that table, `A<n>#row=<k>` or `A<n>#row=<k>-<l>` to rows of a bulk data artefact of that evidence. (Revised in v0.14 and v0.32.)

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

48. Tables appear in evidence of any profile except `formal`. Each has an `id` (`T1`, `T2`, …) unique within its evidence object, a `description`, `row_labels` (a list, possibly empty) and `columns` (a non-empty list of distinct entries, each a non-empty string or `{variable, conditions}`), and a non-empty list of `rows`; each row has an `id` (`r1`, `r2`, …) unique within the table, a `label` with one entry per row label, and `values` with one entry per column, each a value as in rule 66 or `null`, at least one value in the table not `null`. `units`, if present, has one entry per column; `artefact`, if present, resolves. (Revised in v0.32: values of any type; tables in any profile except formal.)

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

### 10.15 Rules added in v0.20 (errors: file rejected)

58. `from_exploration` in `exploration.hypotheses` or in `exploration.covers.lab.hypotheses`, if present, is an integer from 0 to that object's `total`.

### 10.16 Rules added in v0.21 (errors: file rejected)

59. Review-comment references are global IDs `<claimset ID>@v<n>/RV<k>.c<m>`: each entry of a claim's `test.addresses` and of an operation's `addresses` in `paper.changes`. `paper.answers`, if present, is a response ID `<paper.id>@v<paper.version>/RP<k>`.

### 10.17 Rules added in v0.22 (errors: file rejected)

60. A claim that is not an assumption has a `novelty`; an assumption claim (support of kind `assumption`) has none. The value `new` of earlier versions is `no_prior_found`.

### 10.18 Rules added in v0.23 (errors: file rejected)

61. (Revised in v0.30 and v0.34.) Every evidence object whose profile is not `formal` has a `setup`, with only the keys `studied`, `cases`, `procedure`, `lens` and `context`, each a non-empty list of IDs of definitions, at least one key present; formal evidence has no `setup`.

### 10.19 Rules added in v0.24 (errors: file rejected)

62. Each open question has an `id` `O<n>`, a non-empty `question` that follows the text rules, and only the fields of §3.14; its `claims`, if present, resolve to claims.

### 10.20 Rules added in v0.30 (errors: file rejected)

63. Every definition has a `kind`, one of the values of §3.2. (Revised in v0.34: required of every definition, not only those named in a setup or a claim's `about`.)
64. `about`: every principal claim has one; no assumption claim has one. It has `subject` and `relation`, each a non-empty string, and `conditions`, a list (possibly empty); `object` and `comparator` are present, each `null` or a slot value; `object` may instead be a non-empty list of IDs of definitions of kind `variable` (`measure` until v0.31). A slot value is a non-empty string: when it has the form of a definition ID (`D` followed by digits) it resolves to a definition, and that definition is marked in the claim's statement (`[[Dn:…]]`). `relation` is text, never an ID.

### 10.21 Rules added in v0.31 (errors: file rejected)

65. A definition's `members`, if present, is a non-empty list of IDs of other definitions, each of the same `kind` as the definition, which has one; no definition is a member of itself, directly or through other definitions' members.

### 10.22 Rules added in v0.32 (errors: file rejected)

66. Observations. Results appear in evidence of any profile except `formal`. Each result has an `id` and a `variable` (a non-empty string; `quantity` is not used) and a `value` that is a number, a non-empty string, a boolean, or an excerpt with a non-empty `text` and `locator` and only the fields of §3.4. Every string of the form of a definition ID in a result's or table's `variable`, `conditions` (keys and values), `row_labels`, `columns` and row labels resolves to a definition; a `variable` that names a definition names one of kind `variable` or `concept`, or a group of them; a condition or row label whose dimension is a group is one of its members; `read_from`, if present, names an artefact of kind `figure` or `recording` and has a `precision`. Evidence of profile `qual` has no `exemplars`.
67. Artefacts and figures. A `figure` has a `function`; `shows`, if present, is a list of references that resolve as in rule 38, given only for a `data_plot`; `encoding`, if present, has only the keys of §3.6; a support entry's `artefacts` names no figure of function `diagram`. `columns`, if present, is a list of `{name, variable, dimension, unit}` with a non-empty `name` and exactly one of `variable` and `dimension`, each resolving as in rule 66 when it is a definition ID.
68. Definitions. A definition's `kind` is not `measure` (renamed `variable` in v0.32); `part_of` and `broader`, if present, are non-empty lists of IDs of other definitions, and neither link forms a cycle (no term is part of, or a kind of, itself, directly or through others); a definition with `broader` has the kind of each broader definition that has one; its `identifiers`, if present, is a list of compact identifiers `prefix:local`, the prefix of lower-case letters, digits, dots, hyphens and underscores, starting with a letter. A target with `id_type: curie` has the same form.
69. Declarations. `paper.declarations`, if present, has only `funding`, `competing_interests`, `ethics` and `consent`; each funding entry has a non-empty `funder`; each ethics entry has a non-empty `body` and a `decision` from the list of §3.1; `competing_interests` and `consent` are non-empty text.
70. Package. When the claimset is validated as part of a package (§2.5): every artefact whose `uri` is a relative path names a file of the package, its `version` is `sha256:` followed by that file's checksum, and the file is in the package's manifest with the same checksum. A published file has no relative `uri`.

71. Relations within the paper. A relation with `id_type: local` has a `target` that is the ID of another claim of this paper and a `type` from `supports`, `contradicts`, `refines`, `explains` and `qualifies`; the relations `explains` among the paper's claims form no cycle; a claim does not both `explains` a claim and list it in `depends_on`.
72. Derived evidence. Each entry of an evidence object's `derived_from` names another evidence object of this paper and has a `how` from the list of §3.4; `derived_from` forms no cycle.
73. Assumptions of a term. A definition's `assumptions`, if present, is a non-empty list of IDs of assumption claims (support of kind `assumption`).
74. Contrasts. A `comparator` in an observation's conditions (of a result, a table, a row label or a column) is a definition ID or text; when it is a definition ID and another condition of the observation has a value that is a member of a group, the comparator is a member of the same group.

### 10.23 Rules added in v0.34 (errors: file rejected)

75. Availability. Every evidence object and every artefact has an `availability` with a `level` from §3.10.

### 10.4 Warnings (file accepted, warning recorded)

1. A `recommended` field in §3.11 or §3.12 is missing.
2. An agent's `record_source` is absent or `self_reported`.
3. `paper.contribution_record` is `partial` or `unknown`.
4. `autonomy: human_supervised` without a `supervisor`, or `autonomy: human_assisted` without a human `contributor`.
5. No `paper.classification.keywords` in a submission.
6. Three or more claims share an identical `scope` object (often a sign of scope copied by section; check each claim).
7. `novelty: extension` without an `extends` relation.
8. Quant or simulation evidence with neither `estimate`, `results` nor `tables`.
9. An artefact without `licence`, unless it names the cited work it is taken from (`reference`) and is not a file of the submission package (§2.5). (Revised in v0.33.)
10. An agent contributor that is not an `author` has the AI task `writing`; agents that produced the paper's content are authors.
11. A check was made by an agent that runs on the same model as an `author` agent.
12. A number in a claim's statement cannot be matched to the results the claim cites, at the precision stated (heuristic; percentages are compared as fractions). Applies to cited result entries; cited tables and rows are not checked.
13. `planning: exploratory` on a principal claim: exploratory findings can be principal, but readers should be told so in the claim's assessment rationale.
14. `id_type: reference` in a native paper: give the cited work's identifier.
15. A `supporting` claim that no principal claim rests on, directly or through other claims (by `depends_on` or `assumptions`, including the assumptions of the evidence a principal claim cites and the assumptions it inherits from the terms it rests on, §3.2 *Assumptions of a term*): link it to the claim that rests on it, or make it `secondary`. (Revised in v0.32 and v0.33.)
16. A principal or supporting claim rests on a `secondary` claim (by `depends_on` or `assumptions`): a claim that a contribution rests on is `supporting`.
17. A number in the statement of a claim with `empirical_quant` or `simulation` support matches, at the precision stated, no value in: the results (value, interval, conditions) and tables (conditions, cells, row labels) of the evidence the claim cites; that evidence's `estimate`, `interval` and `n`; the claim's `test.parameters` and `scope`; the definitions its statement references. Evidence descriptions do not count. Numbers inside LaTeX, `[[D…]]` markers and model notation (`ETS(A,N,A)`, `AR(1)`), and numbers of sections, tables, figures, equations and theorems, are ignored. The reverse of warning 12: a number a statement reports is a recorded result, or a setting in the scope.
18. Two evidence objects have the same `execution.code_version` and the same results (quantity, value and conditions): one computation recorded twice. Record it once and cite it from every claim it supports.
19. `novelty: no_prior_found` on a claim whose only support is `consensus`: if established knowledge already shows the claim, the claim is known (cite it, or relate to it); if it is not, give the support that shows it. (Revised in v0.22.)
20. In a native paper, a principal claim lists a limitation with `impact: could_reverse` and `mitigation: none`, and its author assessment neither has strength `low` or `very_low` nor names that limitation by ID in its rationale: say how the risk was weighed, or lower the strength.
21. A public artefact's `uri` is on a git host and contains no full (40-hexadecimal-digit) commit hash: a branch or tag link can change what it shows.
22. A statement shows a sign of holding more than one claim (§8 rule 1): a semicolon, a word of contrast, addition or explanation, a side clause with `, which`, a parenthesis holding `and`, `also` or `is`/`are`, a recommendation, or results cited from more than one evidence object. Check it: if one part could be true and another false, split it into claims, each self-contained. A comparison that states where a method is better and where it is not stays one claim.
23. A claim that is not principal, has no warrant and is not formal, whose only support is `argument`, `consensus`, `expert_judgement` or `unstated`, and on which no claim rests (by `depends_on` or `assumptions`): the paper neither shows nor uses it (§8 rule 9). Support it and link it, record it as an open question, or leave it out; in a converted paper, check that the source's argument uses it. The message names rhetorical wording when the statement has some: importance framing ("crucial", "paramount"), promise ("promising", "paves the way"), sweeping generalisation ("always", "it is well known"), speculation ("may", "could"). The wording alone never raises the warning.
24. The statement of a claim with `empirical_quant` or `simulation` support repeats three or more values of the result tables it cites (§3.4, *Tables and claims*): state the pattern the paper concludes and cite the table or rows, repeating a value only when the conclusion turns on it; or, if the claim only retells the table, remove it and cite the table from the claims that use it. Numbers are found as for warning 17.
25. A `secondary` `descriptive` claim with `empirical_quant` or `simulation` support (a result, not a conclusion drawn from results) on which nothing rests: no claim depends on it or lists it in `assumptions`, no warrant names it as a premise, no open question concerns it, and it stands in no relation to another claim of the paper (it neither `explains`, `qualifies`, `supports`, `contradicts` nor `refines` one, nor is the target of such a relation, §3.7). Keep it only if it states a loss, limit or null result of the paper's own method; otherwise the result table of its evidence holds it (§3.3, *Secondary claims in converted papers*). (Revised in v0.33.)
26. A definition without `source` (a term of the paper's own) is named in the statement of a `formal` claim and contains no mathematics (`$…$`): if it is the paper's own method or model, give its defining equations (§3.2, *The paper's own method or model*).
27. A supporting or secondary claim that is not an assumption has no `about` (§3.3.5). Also: a slot of `about` holds text that equals, ignoring case, the `term` of a definition in the file: give the definition's ID instead.
28. A definition is named in a claim's `about` and in no setup of the non-formal evidence that claim's support cites, although its kind is `method`, `intervention`, `population`, `data`, `design` or `variable` (a part of the study that the setup leaves out). A definition counts as named when the setup names it, names a group it is a member of, or names every member of it (§3.2, *Groups*). (Revised in v0.31.) In v0.34 an evidence object without a setup is an error (rule 61).
29. A native paper has no `declarations`, or its declarations leave out `funding` or `competing_interests`: ask the guarantor and record what they declare (an empty funding list and "None declared." are declarations).
30. A `data_plot` figure without `shows` or `underlying_data`; a `specimen` from which no observation is read (`read_from`); or a result whose `variable` or a table column whose variable is text equal, ignoring case, to the `term` of a definition in the file (give the ID).
31. A converted paper lies in a `claimset/` folder and its file is not named `<name>.claimset.json` with the name made from its source's identifier (§2.4). The message gives the name.

## 11. Platform dependencies

Platform functionality is specified in the platform requirements document. The file format depends on the platform for:

- **Identifiers and publication facts:** minting the DOI at publication and registering it with DataCite (§7), and assigning version, dates and status (§2.2).
- **Guarantor consent:** a paper is not published until every human guarantor has confirmed through their ORCID-linked account, and every organisation guarantor through an authorised account.
- **Classification:** assigning the OpenAlex subfield and topics.
- **Derived data:** verifiability computation and recomputation, trust scores, claim weights, contradictions, replication status and assessments, stored outside the file.
- **Corrections:** computing the JSON Patch between versions.
- **Source status tracking:** for converted papers, the current status of the source is tracked and shown alongside the claimset's own status.
- **Conversion review and claiming:** sampled human review of conversions, and the process by which source authors review, endorse or claim the conversion of their work.
- **Content review:** AI review of each claim's scope against its statement, and independent re-derivation of formal claims; findings are returned to the submitting agent as warnings before publication and stored as platform assessments.
- **Relation resolution:** resolving relation targets to claimsets where they exist, and showing incoming relations (e.g. "contradicted by") on the targets.
- **Artefact archiving:** at publication, archiving each public artefact whose `uri` names a commit on a git host (in Software Heritage or a comparable archive) and recording its `archive_uri`, unless the file gives one.
- **Reference resolution:** resolving printed references (`id_type: reference`) to identifiers where one exists, stored as derived data.
- **Package storage:** storing the files of each submission package by checksum and assigning their permanent locations at publication (§2.5).
- **Independence of evidence:** following `derived_from`, so that results resting on the same data are not counted as independent in claim weights, replication status and syntheses.
- **Term links:** following `members`, `part_of` and `broader`, within a claimset and through the shared entities they resolve to, so that a search for a term also finds what is said about its members, parts and narrower terms, labelled as such.
- **Observation index:** every result, table cell and row of bulk data, with its variable, conditions and value resolved to shared entities, so that observations can be queried and compared across claimsets.
- **Entity resolution:** resolving definitions (by kind, term and `source`) and the text slots of `about` to shared entities across claimsets, so that the M3 monthly data, the MASE or role ambiguity is one entity whichever paper names it; linking entities to external identifiers (data set and software DOIs, Wikidata, subject thesauri) where they exist. Stored as derived data, never written into the file. Search over papers and claims by the entities in their setups and in the slots of `about` depends on it.

## 12. Alignment with existing standards

- **Primary:** nanopublications. Each claim maps to a nanopublication (assertion = claim; provenance = support, evidence and assumptions; publication info = paper and contributors). Content-hash identifiers suit immutable versions. A claim's `about` gives the assertion its triple, subject, relation and object, with the comparator and conditions as qualifiers, in the statement model also used by Wikidata.
- **Study frames:** the roles of a setup (§3.4) correspond to PICO (population, intervention, comparator, outcome), used to specify trials and clinical questions, and to SPIDER (sample, phenomenon of interest, design, evaluation, research type), its counterpart for qualitative evidence synthesis.
- **Borrowed:** argument structure from micropublications.
- **Observations:** the W3C RDF Data Cube vocabulary and SDMX (an observation is a measure's value at a point given by its dimensions); the observation models OBOE and SOSA; I-ADOPT for describing variables. Units: UCUM. Bulk tabular data: W3C CSV on the Web (CSVW), with rows cited by RFC 7111 fragments.
- **Packages:** BagIt (RFC 8493).
- **Identifiers of things:** compact identifiers with prefixes from the Bioregistry and identifiers.org; ROR and the Crossref Funder Registry for funders; ECLI and ELI for legal sources.
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
6. **Excerpts:** granularity and confidentiality of excerpts (§3.4), and how consent to publish them is recorded beyond `declarations.consent`.
7. **JSON-LD context:** which existing vocabularies to map onto.
8. **Guarantor obligations:** what research and conversion guarantors commit to, and how unresponsive guarantors are handled.
9. **Non-commercial and no-derivatives sources:** whether `CC-BY-NC` sources can be converted, and on what terms.
10. **The study frame and the atom across fields** (v0.30): whether the roles of a setup, the kinds of definitions and the slots of `about` can be assigned consistently across paper types (forecasting and other benchmarks, trials, surveys, qualitative studies, simulation studies, theory papers). To be settled by independent coders on papers of each type; roles, kinds or slots on which coders disagree are merged, redefined or dropped in a later version.

## Appendix A: native paper — submission

What the agent submits. No identifier, version, date, status, classification IDs or verifiability: the platform adds them.

```json
{
  "schema_version": "0.34",
  "paper": {
    "title": "Method M against the seasonal naive benchmark on UK monthly retail sales",
    "licence": "CC-BY-4.0",
    "classification": {"keywords": ["retail demand forecasting"]},
    "declarations": {"funding": [], "competing_interests": "None declared."},
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
    {"id": "D1", "term": "forecast skill", "kind": "variable", "definition": "Relative reduction in mean absolute scaled error against a seasonal naive benchmark, averaged over series and horizons."},
    {"id": "D2", "term": "method M", "kind": "method", "definition": "The forecasting method studied in this paper, as implemented in version 1.0 of the code."},
    {"id": "D3", "term": "seasonal naive method", "kind": "method", "definition": "Forecasts each month by the value of the same month one year earlier."},
    {"id": "D4", "term": "UK monthly retail sales", "kind": "data", "definition": "1,200 monthly sales series of one UK retailer's store network, 2010-2024, each with at least 36 observations."},
    {"id": "D5", "term": "rolling-origin evaluation", "kind": "design", "definition": "Rolling origin with forecasts for horizons 1 to 12 months from each origin."}
  ],
  "claims": [
    {
      "id": "C1", "statement": "Monthly retail sales series exhibit annual seasonality.",
      "type": "descriptive", "role": "supporting",
      "support": [{"kind": "assumption"}],
      "assumption_category": "data", "check": "unchecked",
      "provenance": "author_stated"
    },
    {
      "id": "C3", "statement": "Forecast errors of method M are independent across series.",
      "type": "descriptive", "role": "supporting",
      "support": [{"kind": "assumption"}],
      "assumption_category": "statistical", "check": "unchecked",
      "provenance": "author_stated"
    },
    {
      "id": "C2",
      "statement": "[[D2:Method M]] improves [[D1:forecast skill]] over the [[D3:seasonal naive method]] on the [[D4:UK monthly retail sales]].",
      "about": {"subject": "D2", "relation": "improves", "object": "D1", "comparator": "D3", "conditions": ["D4"]},
      "type": "descriptive", "role": "principal", "novelty": "no_prior_found",
      "scope": {"conditions": ["horizons of 1 to 12 months"]},
      "support": [{"kind": "empirical_quant", "evidence": "E1", "artefacts": ["A2"]}],
      "assumptions": ["C1", "C3"],
      "limitations": ["L1"],
      "assessments": [{"by": "author", "strength": "moderate", "rationale": "Single benchmark collection, no external replication."}],
      "provenance": "author_stated"
    }
  ],
  "evidence": [
    {"id": "E1", "profile": "quant", "description": "Benchmark of method M against the seasonal naive method on 1,200 monthly retail sales series, rolling origin, horizons 1 to 12.", "setup": {"studied": ["D2", "D3"], "cases": ["D4"], "procedure": ["D5"], "lens": ["D1"]}, "design": "benchmark", "estimand": "mean relative MASE reduction", "estimate": 0.08, "interval": {"level": 0.95, "lower": 0.05, "upper": 0.11}, "n": 1200, "data_origin": "secondary", "preregistered": false, "artefacts": ["A1"], "availability": {"level": "public"}}
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
      "type": "associational", "role": "supporting", "novelty": "no_prior_found",
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
declarations funding: none | competing interests: None declared.

DEFINITIONS
D1 [variable] forecast skill: Relative reduction in mean absolute scaled error against a seasonal naive benchmark, averaged over series and horizons.
D2 [method] method M: The forecasting method studied in this paper, as implemented in version 1.0 of the code.
D3 [method] seasonal naive method: Forecasts each month by the value of the same month one year earlier.
D4 [data] UK monthly retail sales: 1,200 monthly sales series of one UK retailer's store network, 2010-2024, each with at least 36 observations.
D5 [design] rolling-origin evaluation: Rolling origin with forecasts for horizons 1 to 12 months from each origin.

CLAIMS
C2 [descriptive; principal; support: empirical_quant E1 artefacts A2; assumes C1, C3; limitations L1; strength moderate (author)]
  [[D2:Method M]] improves [[D1:forecast skill]] over the [[D3:seasonal naive method]] on the [[D4:UK monthly retail sales]].
  scope: conditions: horizons of 1 to 12 months
C1 [descriptive; supporting; assumption (data, unchecked)]
  Monthly retail sales series exhibit annual seasonality.
C3 [descriptive; supporting; assumption (statistical, unchecked)]
  Forecast errors of method M are independent across series.

EVIDENCE
E1 [quant; design benchmark; data_origin secondary; n 1200] Benchmark of method M against the seasonal naive method on 1,200 monthly retail sales series, rolling origin, horizons 1 to 12.
  setup: studied D2, D3; cases D4; procedure D5; lens D1
  estimate: mean relative MASE reduction = 0.08 (95% interval 0.05 to 0.11)

LIMITATIONS
L1 [sampling; affects C2; reduces_transferability; mitigation none] All 1,200 series come from one retailer's store network.

ARTEFACTS
A2 [table] Mean relative MASE reduction of method M by forecast horizon, 1 to 12 months.
```

## Appendix E: a qualitative paper (definitions, claims, evidence and warrant of a submission)

An invented interview study, to show the frame of §3.4 and `about` (§3.3.5) outside quantitative work. The paper, contributor, model and reference sections are omitted.

```json
{
  "definitions": [
    {"id": "D1", "term": "transition to rural practice", "kind": "concept", "definition": "The first two years of work of a registered nurse who moves from an urban hospital to a rural one."},
    {"id": "D2", "term": "rural transition nurses", "kind": "population", "definition": "18 registered nurses in their first two years at one of two rural district hospitals, after at least three years in urban hospitals; 15 women, 3 men."},
    {"id": "D3", "term": "two Welsh rural hospitals", "kind": "setting", "definition": "Two district general hospitals in rural Wales, interviews held in 2025."},
    {"id": "D4", "term": "purposive sampling with semi-structured interviews", "kind": "design", "definition": "Nurses chosen for variety of specialty and length of urban experience; one interview each, 45 to 70 minutes, by a guide of eight open questions; recruitment stopped when two interviews in a row added no new codes."},
    {"id": "D5", "term": "reflexive thematic analysis", "kind": "design", "definition": "Thematic analysis in the six phases of Braun and Clarke, by two researchers who coded independently and agreed themes in discussion.", "source": "B1"},
    {"id": "D6", "term": "role theory", "kind": "theory", "definition": "The view that people act according to expectations attached to positions; here used to read accounts of changed expectations.", "source": "B2"},
    {"id": "D7", "term": "loss of clinical identity", "kind": "concept", "definition": "A theme: nurses describe no longer being recognised, by themselves or colleagues, as specialists in the field they practised in before."},
    {"id": "D8", "term": "generalist expectation", "kind": "concept", "definition": "A theme: nurses describe being expected to cover any ward or task, regardless of specialty."}
  ],
  "claims": [
    {
      "id": "C1",
      "statement": "[[D2:Rural transition nurses]] at the [[D3:two Welsh rural hospitals]] describe the [[D1:transition to rural practice]] as a [[D7:loss of clinical identity]].",
      "about": {"subject": "D1", "relation": "is described as", "object": "D7", "comparator": null, "conditions": ["D2", "D3"]},
      "type": "interpretive", "role": "principal", "novelty": "no_prior_found",
      "support": [{"kind": "empirical_qual", "evidence": "E1"}],
      "warrant": "W1", "depends_on": ["C2"],
      "provenance": "author_stated"
    },
    {
      "id": "C2",
      "statement": "[[D2:Rural transition nurses]] report a [[D8:generalist expectation]] at the [[D3:two Welsh rural hospitals]].",
      "about": {"subject": "D2", "relation": "report", "object": "D8", "comparator": null, "conditions": ["D3"]},
      "type": "interpretive", "role": "supporting", "novelty": "no_prior_found",
      "support": [{"kind": "empirical_qual", "evidence": "E1"}],
      "warrant": "W2",
      "provenance": "author_stated"
    }
  ],
  "evidence": [
    {
      "id": "E1", "profile": "qual",
      "description": "Semi-structured interviews with 18 nurses in their first two years at two rural hospitals, analysed thematically.",
      "setup": {"studied": ["D1"], "cases": ["D2"], "procedure": ["D4", "D5"], "lens": ["D6"], "context": ["D3"]},
      "results": [
        {"id": "R1", "variable": "D7", "conditions": {"participant": "P07", "D3": "Hospital B"},
         "value": {"text": "I used to be the stroke nurse. Here I am just a nurse, and people ask me for everything.", "locator": "transcript P07, lines 112-114"}},
        {"id": "R2", "variable": "participants whose accounts show the theme", "conditions": {"theme": "D7"}, "value": 14, "n": 18}
      ],
      "data_sources": ["interview"], "n_participants_or_cases": 18, "sampling": "purposive", "analytic_approach": "thematic",
      "data_origin": "primary", "preregistered": false,
      "availability": {"level": "unavailable", "reason": "personal_data"}
    }
  ],
  "warrants": [
    {"id": "W1", "reasoning": "interpretive", "premises": ["C2"], "text": "Read through role theory, the generalist expectation removes the position from which nurses' specialist expectations of themselves were met; 14 of 18 participants link the two when describing who they now are at work."},
    {"id": "W2", "reasoning": "interpretive", "premises": [], "text": "The theme draws on accounts from 16 of 18 participants across both hospitals and all specialties sampled, including accounts that contradict it, which concern night shifts only."}
  ]
}
```

The phenomenon is `studied`, the participants are the `cases`, sampling, interviewing and the analysis are the `procedure`, and the theory through which the accounts were read is the `lens`. The themes are `concept` definitions, so a reader looking for work on loss of clinical identity among nurses finds C1 by its `object`, and a reader looking for studies using role theory finds E1 by its `lens`, without either matching the wording of a statement.
