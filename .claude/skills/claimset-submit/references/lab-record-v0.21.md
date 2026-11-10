# Claimset Lab Record — Specification v0.21 (draft)

Status: working draft · Date: 2026-11-10 · Supersedes v0.20 · Companion to the claimset specification (`spec/claimset-v0.40.md`), its extensions (`spec/extensions/`) and the revision specification (`spec/claimset-revision-v0.6.md`)

## Changes since v0.20

A record said which model an agent was, but not which harness ran it: the same model in Claude Code, Codex CLI, Gemini CLI or opencode, with different tools and instructions, is a different agent. And the protections assumed one harness.

- §5.2: an agent actor may record its `system`, the harness it runs in, as for agent contributors (claimset §3.11). `claimset lab actor` fills it in when the harness identifies itself in the environment (§4.6) and none is given, and takes the model from `$CLAIMSET_MODEL`, set by the human, when the agent gives none (`record_source: user_reported`). Export carries `system` into the contributor (§9).
- §4.5, §4.6 (new), §10: the guard works in every harness that can run a command before a tool call, from one configuration that `claimset init` writes for each harness; harnesses without such hooks rely on the other layers.
- Records begun before v0.21 are read as they were written; `system` is optional in every record.

Amended on 2026-11-10 (§9): export records the results of an evidence object as tables (claimset §3.4, *Tables are the standard form*), not one result entry per number; claims cite table rows; every result of the record is written, at full precision, to a CSV file beside the claimset, listed as a data artefact; results of the cited runs that a claim's status leaves out are not exported.

## Changes in v0.20 (since v0.19)

The claimset specification is this record's vocabulary. Lists this specification used to repeat (the kinds of terms, the kinds of artefacts, actors, assumption categories, check outcomes, reasons for withholding, the frameworks of a decision rule and their parameters, the grammar of rule expressions, the test flags, the form of an observation) are now only in the claimset specification, which this one names; the tool takes them from the claimset package, so the two cannot drift apart.

- §2.1 (new): what this record takes from the claimset specification, and which version.
- §5.10, §5.10.1, §5.7.1: a result names what it observes by `variable`, as a claimset observation does, and a decision rule's quantities bind results by `variable` (both were `quantity`). Results files may still say `quantity`.
- §2.1, §5.22 (new), §6 rule L30: extensions. The record uses the extensions of the claimset format (claimset §2.7): an `extension` event declares one, and events then carry its blocks (`ext`) or are of its event types. Export carries the blocks into the claimset and declares the extensions there.
- The codebook, codings and counts of coded material (v0.17 to v0.18: a term's `code`, the `coding` event, a run's `codings`, count quantities and their flags, rule L27) move to the qualitative extension (`spec/extensions/qualitative-v0.1.md`), which the tool declares the first time it is needed. Captured and private sources stay here.
- Amended with claimset v0.39: a result may carry `uncertainty` (claimset §3.4), and a decision rule's quantity may read one of its components (`field: "uncertainty.se"`). The fields come from the claimset specification, so any record may use them.
- Records begun before v0.20 are read as they were written: `quantity`, a term's `code`, codings, a run's `codings` and count quantities, with the qualitative extension declared from their start.

## Changes in v0.19 (since v0.18)

A private source's files, and the salt of the study's pseudonyms, were committed on the study branch: anyone who could read the branch could read every page captured as private, and could check a guessed handle against the pseudonyms. Private meant only "left out of the package".

- §3, §5.20, *Private sources*: a private source's files are kept in `lab/private/sources/<id>/`, which the lab keeps out of git (`lab/private/.gitignore`); the record holds their hashes and everything else the event records. The pseudonym salt is kept in `lab/private/pseudonym-salt`, also out of git. A public source's files are committed in `lab/sources/<id>/`, as before.
- §5.20: a checkout without `lab/private/` (another clone of the branch) cannot read a private source's text, and refuses `capture --author` once the record has pseudonyms, since its pseudonyms would not match; copying `lab/private/` from the checkout that captured the sources restores both.
- §6: L24 and L25 are checked for a private source wherever its files are present; where they are absent, `verify` says which sources it could not check.
- Records begun before v0.19 keep the private files and the salt they committed (the branch's history is never rewritten); `verify` notes them. Sources they capture from now on are kept out of git.

The lab skill had the agent type the human's registration, which the record then showed as written by the human: the one place the log said a person wrote what they did not.

- §5.2, *Registration*: an agent registers itself first. A person registers themselves, or an agent registers them with their reply quoted verbatim (`words`, and optionally `given_at` and `channel`), as an agent records an approval (§5.8); the event is the agent's.
- §6: new rule L28. An agent's registration of a person without their words is refused; `words` are given only by an agent quoting a person.
- `verify` notes a record with no human actor (an agent working alone), and, in records begun before v0.19, people who registered themselves and then the agent, as the skill then had the agent type both.


Amended on 2026-11-05: new rule L29, a record begun under a later version than the tool is refused with its own message, so the reader knows the tool is behind, not the record.

## Changes in v0.18 (since v0.17)

A qualitative test's rules were text, judged by the writer: "supported if at least 5 independent threads from different years ask for automatic arranging" could be written but not computed, and nothing showed whether the threads had been coded before the test was fixed. A one-day desk scan also had to keep a git remote it did not need.

- §5.21.1, *Counts*: a plan's quantity can be a count over the recorded codings, `{"count": {codes, match, unit, distinct_by, where}}`: the sources (or codings) with the codes or narrower ones, counted once per source, author, year or venue. `claimset lab count --plan PL1` runs the built-in counter as the deciding run, and the tool computes the verdict; `claimset lab count --codes …` is an exploratory count.
- §5.20: a source records its `author`, as a pseudonym stable within the study (`a-…`, from `capture --author HANDLE`), so counts can tell authors apart without names in the record.
- §5.7.5: new flags `coded_before_plan` (the deciding run read codings, of the codes the test counts, written before the plan) and `codebook_changed_after_plan` (a counted code was redefined between the plan and the test).
- §5.21: a coding can say that a source was read and no code applies (`none_apply`, `claimset lab code SR6 --none`), so the sample keeps its denominator.
- §5.20: a source's description is corrected with the same files (`claimset lab capture --supersedes SR1 --archive-uri …`); private sources are committed on the study branch, so the branch's remote must be as private as they are.
- §6: L15 covers counts; L27 applies to the codes a plan counts.
- §7.1, *Desk mode*: `claimset setup --study NAME --desk` keeps a study on its branch without a remote: every command is committed, nothing is pushed or anchored; the hash chain, the order of plan and test, results from code and quotes from sources are kept, and the claims' `planned_at_anchor` is `none`. Giving a remote later (`claimset setup --study NAME --remote origin`) anchors from then on.
- §9: the pinned link to the exported lab record is made for GitHub, GitLab, Codeberg and other Gitea or Forgejo hosts and Bitbucket; the `lab_record` description says when a record was kept in desk mode.
- §9: runs that read codings are exported as qualitative evidence (`content_analysis` of documents) linked to the sources they read, with `empirical_qual` support; the counter's runs do not raise `code_changed_after_plan`, since their code is the tool's (§5.7.5).
- §10, §11 updated.

## Changes in v0.17 (since v0.16)

Coding was work done outside the record: a theme count ("forums mention X often") was an impression, not a number from recorded codes. With sources captured (v0.16), coding becomes recorded work, and counts come from code that reads it.

- §5.19: a term of kind `concept` can be a **code** of the codebook: `code: {include, exclude, examples}` says what a passage must show to get it and what does not count. Terms gain `broader` (the broader codes or themes a code belongs to). Corrections of a code are the codebook's versions.
- §5.21: a new event, `coding` (prefix `CD`), written by `claimset lab code SR1 --codes TM4 --excerpt "…"`: codes applied to a captured source, or to a passage of it (`quote`, checked against the source's text), by its writer, in a `pass`, with a `unit` and a `memo`, and the version of each code applied. Recoding and withdrawing are corrections.
- §5.9: a run receives the codings in force, as its writer sees them, in `codings.json` (`$CLAIMSET_CODINGS`), recorded in the run's `codings` with its hash: counts and agreement between coders come from scripts over recorded codes.
- §6: L25 is checked by `verify` for every coding's quote; new rule L27 (a coding names a captured source and codes of the codebook).
- §9: a code's definition carries its criteria and its broader terms; the completion list says how excerpts from codings are given.
- §10, §11 updated.

## Changes in v0.16 (since v0.15)

A plan could name several hypotheses but had one pair of rules, so every hypothesis it named got the verdict of that pair. Hypotheses that one computation decides by different rules ("the share is below 0.10" for one, "this theme's total exceeds every other's" for another) needed a plan each, with the same computation run once per plan: the duplicate-run note, the evidence warning about identical results (claimset W18) and a false `holdout_used_elsewhere` flag on each plan followed.

- §5.7.1, *Rules for each hypothesis*: a decision rule may give each hypothesis of the plan its own pair of rules, `rules: {"H1": {supported, not_supported}, "H2": {…}}`, over the quantities the plan shares, with an optional `framework` and `parameters` of its own. One deciding run gives a verdict per hypothesis. A rule with one pair, as before, decides every hypothesis the plan names.
- L15: with `rules`, there is one entry for each hypothesis the plan names and none for another, and no shared pair; each entry, with the shared parts, is a complete rule (§6).
- L17: the verdict on a hypothesis is the one its own rules give (§6).
- §9: a claim's `test` carries the rules of its hypothesis and only the quantities they use. The `lab_record` artefact's description names the specification version the record was begun under, the date it was begun and the version of the tool that exported it, from the record itself.
- §10: `--supported` and `--not-supported` take a hypothesis prefix (`--supported "H1: s < threshold"`) and can be repeated.

Outside sources (a forum thread, a web page, a transcript) were things a note mentioned, not data: a link in a note can change or disappear, and what it said cannot be quoted or counted from the record.

- §5.20: a new event, `source` (prefix `SR`), written by `claimset lab capture`: the file as the researcher saved it (the default) or as the tool fetched it, under `sources/<id>/` with its hash and its extracted text, with where it was found (`uri`), how it was obtained (`method`: `saved`, `browser` or `fetched`), when it was retrieved, its format, whose it is (`rights`) and whether it is published (`visibility`). Content written by others is private by default.
- §2, principle 3: quotes come from captured sources; `claimset lab quote` checks an excerpt against a source's text and gives its locator (a W3C Web Annotation text-quote selector).
- §5.7.5: a new flag, `sources_captured_after_plan`: a deciding run used a source captured after its plan.
- §6: new rules L24 (a source's files are as captured), L25 (an excerpt is in its source's text) and L26 (content written by others is published only under its own licence or terms).
- §8, §9: a private source is listed with its hash and left out of the package; content written by others is exported as a cited work (`reference`), never under the paper's licence.
- §3, §10, §11 updated.

## Changes in v0.15 (since v0.14)

On a study branch the tool committed and pushed the record after every command, then wrote the push's receipt into `anchors.jsonl`. A commit cannot hold the receipt of its own push, so the receipt stayed in the checkout, uncommitted: every command left the study checkout dirty, and tools that require a clean checkout (a hook that stops on uncommitted changes) failed after every command.

- §7.1, *Receipts on a study branch*: the receipt of a push is kept in the checkout's git directory, outside the working tree, and the next commit carries it into `anchors.jsonl`: the receipt of push n is in commit n+1. The receipt of the newest pushed commit, which no commit holds yet, is the one kept locally, or, in a checkout that did not push it, is derived from the remote-tracking branch. A command that succeeds leaves the study checkout clean.
- L14 applies to every receipt, committed, kept locally or derived (§6).
- A receipt names the heads of the logs as they are in the commit it names, read from that commit, not from the working tree, where another writer may have appended events the commit does not hold (§7.1). Receipts that earlier tools made in that race name an event their commit lacks; `verify --remote` reports such a receipt as a warning when a later commit of the branch holds the event with the same header hash, not as an L14 error (§6, §7.1).
- Records kept by earlier tools have the receipt of their last push uncommitted in `anchors.jsonl`; the next command commits it (*Records from before v0.15*, §7.1). `anchors.jsonl` keeps its format, and the receipts already in it keep their meaning.

## Changes in v0.14 (since v0.13)

The research is done by agents and decided by people (claimset principle 10). An approval was written only by the human, by running the tool themselves; that asked a supervisor to work the tool to give a decision they had already given in words. The decision stays the human's; the agent records it.

- An approval may be written by an agent on behalf of a human actor (`on_behalf_of`), quoting the human's words (`words`) and when they were given (`given_at`). The agent asks, waits for the answer and records it; it never approves on its own judgement, and silence or a general instruction is not approval (§5.8, L22).
- A plan may require approval (`requires_approval`): no confirmatory run is made under it until an approval of its current version is recorded (§5.7, L23). The agent waits for the supervisor's answer before running.
- New rules L22 and L23 (§6).

## Changes in v0.13 (since v0.12)

Claimset v0.32 records results as observations of variables in any field.

- The term kind `measure` is `variable` (§5.19, L21), as in claimset §3.2. Terms recorded with `measure` are read as `variable`.
- Export writes each result as an observation (claimset §3.4): the result's `quantity` becomes the observation's `variable`, the term it names when one matches, and its `conditions` keep their keys and values, with term IDs replaced by definition IDs (§9).

## Changes in v0.12 (since v0.11)

A pre-registration says what will be studied, on what, how and measured how, before the data are seen; the frames that specify studies across fields (PICO for trials, SPIDER for qualitative work) are that list. Claimset v0.30 records the same frame in a claimset: the parts of a study as definitions with a kind, named by role in each evidence object's `setup`, and each claim's atom in `about`. The lab record is where that frame is fixed in advance, so it now records it, and the claimset is exported from it instead of being written afterwards.

- New event `term` (prefix `TM`): a method, data set, population, design, measure, concept or theory the study names, with its kind and definition (§5.19). The claimset's definitions are exported from the terms (§9).
- Hypotheses gain `about`, the atom of the statement (claimset §3.3.5), with slots that name terms or give text; a statement marks the terms it names, `[[TM3:MASE]]` (§5.4).
- Plans gain `setup`: the terms of the test by role, `studied`, `cases`, `procedure`, `lens`, `context` (claimset §3.4) (§5.7).
- New warnings (§5.7.6): a confirmatory plan without a setup; a plan whose setup does not name a method, intervention, population, data set, design or measure that a hypothesis it tests is about, the mark of a test that would decide a hypothesis about something it does not study or measure (a switched outcome or comparator).
- New rule L21 (§6). Export writes definitions, each claim's `about` and each evidence object's `setup` (§9).

## Changes in v0.11 (since v0.10)

- The completion list asks for the definition of each method the planned tests compare: every defining equation as the code that ran implements it, checked against that code (claimset v0.29, §3.2), and it names definitions among the parts the supervisor's approval covers (§9).

## Changes in v0.10 (since v0.9)

- Export follows claimset v0.22: claims get `novelty: no_prior_found` by default (the authors found no earlier result stating the claim; the completion list asks for the closest one they found), and assumption claims get no novelty (§9).

## Changes in v0.9 (since v0.8)

A published claimset is reviewed, and every review comment says in advance what evidence would settle it (its resolution test, revision §3.4). The lab record is where the author runs that test, so the verdict is computed against a rule the author did not write, at a time the record anchors.

- Plans gain `addresses`: the global IDs of the review comments whose resolution test they run (§5.7.7).
- `plan --comment ID --review FILE` takes the decision rule from the comment's resolution test (§5.7.7, §10).
- Export carries `addresses` into the claim's `test` (§9).
- The record of a revised claimset continues the record of the previous version: new events are appended, nothing before them changes (§5.7.7).

## Changes in v0.8 (since v0.7)

A record can follow every rule and still hide a weak test: a confirmatory run that recomputes the exploration that suggested its hypothesis, units left out of the analysis without a count, one computation run twice under two plans. The tool now says so. Nothing new is required: these are flags and warnings.

- Plans gain `new_data` (what the test uses that the runs which suggested its hypotheses did not) and `exclusions` (rules that leave units out of the analysis, `{name, rule}`) (§5.7).
- New flag `reproduces_prompting_result`: a result the decision rule uses equals a result of a run that suggested the hypothesis (§5.7.5).
- New §5.7.6, warnings: at plan time, a hypothesis from exploration without `new_data`, and a parameter that looks like an exclusion without a declared exclusion; at the verdict, an exclusion that no deciding run counts; at a confirmatory run, the same command, code, seeds and parameters as another plan's run.
- One plan can test several hypotheses (§5.7).
- Export counts the hypotheses suggested by exploration, `hypotheses.from_exploration` (§9; claimset §3.1.1).

## Changes in v0.7 (since v0.6)

The record and the code are kept apart by the tools, not by care. A study's record copied along code branches diverged, was carried into pull requests, and stopped being kept when the work turned to implementation.

- Each study lives on its own branch, `claimset/<study>`, which holds only the study's folder and shares no history with the code. It is checked out as a git worktree at `claimset/<study>/`; the code branches ignore `claimset/`. Study branches are never merged into code branches, nor code into them (§3).
- After every command that changes the record, the tool commits the study branch and pushes it; the push is the anchor, so the separate anchor branches go (§7.1). The commit after a run takes the run's code commit as a further parent, so the code it ran stays reachable when its branch is squash-merged or deleted (§5.9).
- Code only a study uses lives on the code branches, in `claimset-code/<study>/`, left out of package builds like `claimset/` (§3, §3.3).
- `claimset setup` creates, attaches or adopts study branches; a study committed on a code branch is adopted onto its own branch with a relocation (§3.4, §5.18).
- New rule L20: a record is written only on its study branch. The guard refuses git commands that would mix study branches and code, a GitHub check refuses pull requests that carry study records, and every session starts with the state of the studies (§4.5).

## Changes in v0.6 (since v0.5)

A repository can hold several studies, each with its own lab, and a study can give several papers. Where they live and what they are called is fixed, so tools and readers find them.

- Layout: every study lives in `claimset/<study>/` at the top of the repository: its lab in `lab/`, code only it uses in `code/`, its papers as `<paper>.claimset.json` (§3). Study and paper names are lower-case letters, digits and hyphens. `init` records the study's name (§5.1). Tools find the lab of a command by `--lab`, `--study` or the only study in the repository (§3.2).
- Package builds leave `claimset/` out: the tool adds the lines that do it to the build files a repository has (`.Rbuildignore`, `MANIFEST.in`, `.npmignore`, `.gitattributes`) and `verify` reports a missing one (§3.3).
- New event `relocation` (prefix `MV`): the lab moved, with the files the record names; anchors and remote checks follow the move (§3.4, §5.18, §7.1).
- New event `paper` (prefix `PA`): a paper drawn from the record and the questions it reports. A question is reported in one paper; `verify` names hypotheses with a result that no paper reports (§5.17, rule L18).
- Export by paper: only the paper's questions, with the whole lab's counts beside the paper's (claimset §3.1.1) and the other papers named (§9).
- Anchor branches are per study: `claimset-lab/<study>` (§7.1).
- New rules L18 and L19 (§6).

## Changes in v0.5 (since v0.4)

Strict where it matters, light everywhere else. The path from a plan to a claim is strict: every analysis choice that could be tuned after seeing the data is fixed in the plan, each plan gets one attempt, and every planned test that ran is reported with the verdict its rule gives. Exploration is recorded where its results inform decisions, without further restrictions.

- A confirmatory plan carries a **decision rule** instead of a free-text criterion: the inferential framework, its parameters (significance level, test, sidedness, correction for multiple tests, sample size and stopping rule, equivalence bounds, prior, thresholds), and rules for both conclusive outcomes, `supported` and `not_supported`; anything else is `inconclusive`. Rules may be expressions over named results, in which case the tool computes the verdict, or text, in which case the writer states which rule was met (§5.7.1–5.7.3).
- A plan states how many runs decide it. Every confirmatory run under it counts, whatever its exit status, and further confirmatory runs are refused (§5.7.2). Each run records the plan version in force when it was made (`plan_version`, §5.9).
- A verdict names the plan and agrees with its rule. Every planned test that ran must have its verdict recorded before export, and each is exported as its own claim, with the hypothesis as it stood when the plan was written; a hypothesis may be tested again under a new plan (§5.5, §5.7.4, §9).
- Exploration need not be recorded step by step (§2); hypotheses left open are counted as open rather than closed by force (§9).
- Confirmatory runs in an agent log are made under plans written in that log (§3.5).
- Result values are finite numbers (§5.10).
- Plans record the commit they were written at and may declare held-out data; a test whose code changed after the plan, or whose held-out data was used elsewhere, is flagged rather than refused (§5.7.5).
- New rules L15–L17 (§6).
- Records begun under v0.4 or earlier keep their meaning: their plans have `criterion`, and L15–L17 do not apply to them (§5.1).

## Changes in v0.4 (since v0.3)

- Exact bodies for `plan`, `approval`, `run`, `result` and `deviation` (§5.7–5.10, §5.12).
- How the tool runs a command, captures its output and reads results (§5.9.1); the results file format (§5.10.1).
- Approvals are made by humans (§5.8).
- Rules L4 and L5 stated precisely; new rule L13 (run outputs) (§6).
- Protecting the log: what is prevented and what is detected at each level of protection (§4.5).
- Git anchoring specified: configuration, what is committed, receipts, automatic anchoring, checking against the remote (§7.1); new rule L14 (§6).
- Export specified precisely: preconditions, the draft and its completion list, defaults, how planning labels are derived (§9).
- Agent logs specified precisely: how one is started, what an agent sees and may write, the order and layout of a merge, and when a merge is refused (§3.5, §5.16); rule L10 stated precisely; anchors cover agent logs (§7).

## Changes in v0.3 (since v0.2)

Clarifications needed to implement the reference tool. No change to the meaning of v0.2 records.

- Canonical JSON, hash format and header hash defined exactly (§4.1).
- `by` of the `init` event is `null`; an `actor` event may name its own ID in `by` (self-registration) (§4.2).
- Event numbering within each prefix (§4.2).
- Corrections: what `supersedes` means and how references to a corrected event resolve (§4.3).
- Writers take a lock on the log (§4.4).
- New rules L11 (corrections) and L12 (event IDs). New informative commands `show` and `list` (§10).

## Changes in v0.2 (since v0.1)

- Several agents working at once each write their own log, merged into the main log when they finish (§3.5). Event IDs from agent logs carry the actor prefix (`AC2.RN4`); headers gain `origin`.

## 1. Purpose

A lab record is the running record of a piece of research as it happens: the questions asked, the hypotheses proposed and revised, the plans made, every run executed, every result produced, and every decision taken along the way. Agents (and humans) write it while doing the research. The claimset submitted for publication is distilled from it, and the lab record is published alongside the claimset.

Research is iterative: ideas are formed, tested, revised and abandoned. Conventional papers keep only the surviving story. The lab record keeps everything, so that:

- nothing produced during the research is lost or silently overwritten;
- readers can see how many ideas were tried before the published ones, and which results were planned and which were found by exploring;
- negative results are captured as a by-product of doing the work;
- an agent can resume the work after a context reset or hand it to another agent.

Version 0.1 targeted computational research (simulation, analysis of existing data, method development, proofs). Laboratory and qualitative research fit the same model, with humans recording data-collection events. Since v0.16, outside material a study reads (pages, posts, transcripts, documents) is captured as data (§5.20). What one kind of work records beyond that (codes and codings of qualitative research, and in future samples, instrument runs or interview sessions) is recorded through extensions (§2.1). Data-collection sessions are an open question (§11).

## 2. Principles

1. **Append only.** Events are never edited or deleted. A correction is a new event that names the event it supersedes.
2. **Recorded by the tool.** Events are written by the lab tool, which assigns sequence numbers, times and hashes. Agents do not write the log by hand.
3. **Results come from code; quotes come from sources.** Numbers enter the record only from files written by the code that produced them, never typed by an agent; counts of coded material come from runs that read the recorded codings (qualitative extension §2.3). Words quoted from outside material are checked against a captured source (§5.20), never retyped from memory or from a summary.
4. **Plan before run, or say it was exploratory.** A run is confirmatory only if a plan written earlier covers it; every other run is labelled exploratory. Exploration is legitimate; it is just labelled. It is recorded where its results inform decisions (a run whose output is looked at to choose a direction, a setting or a hypothesis), not step by step, so the record's account of exploration is a lower bound. The rules are strict only on the path from a plan to a claim.
5. **Every outcome is recorded.** Failed runs, crashed runs, null results, refuted and abandoned hypotheses are all events.
6. **Exploration cannot confirm itself.** Evidence that suggested a hypothesis cannot also be the evidence that supports it.
7. **Every change of mind has a reason.** Status changes, revisions and deviations from a plan carry a rationale and point at what prompted them.
8. **Resumable.** The current state of the research can always be rebuilt from the log, and the tool summarises it for whoever picks up the work.

### 2.1 The claimset specification is the vocabulary

A lab record is distilled into a claimset, so it speaks the claimset's language. Every list of values, every form and every meaning this record shares with a claimset is defined once, in the claimset specification of the version the tool exports to (the companion named above), and named here by its section, never copied:

| In the record | Defined in the claimset specification |
|---|---|
| a term's `kind` | the kinds of definitions (§3.2) |
| an actor's `kind`, `models`, `system`, `record_source` | contributors and models (§3.11, §3.12) |
| an assumption's `category` and `check` | assumptions (§3.8) |
| a check's `outcome` and `extent` | checks (§5) |
| an artefact's `kind` | artefacts (§3.6) |
| the reason a body is withheld | availability reasons (§3.10) |
| a result: `variable`, `value`, `unit`, `interval`, `uncertainty`, `conditions` | observations (§3.4) |
| a hypothesis's `about`, a plan's `setup` | atoms (§3.3.5), setup roles (§3.4) |
| a statement | statement rules (§8) |
| a decision rule's `framework`, required `parameters`, `quantities`, rule expressions and outcomes | the test of a claim (§3.3.3) |
| the flags of a decided plan | test flags (§3.3.3) and those of the extensions |
| an anchor's `service` | `planned_at_anchor` (§3.3.3) |

The tool reads these from the claimset package it ships with, so a value the claimset specification adds (a new kind of definition, say) is accepted in the record at once. This specification adds only what the record needs besides: events, their order and rules, runs, anchoring, and the lists that belong to the record alone (a hypothesis's `origin`, a status's `to`, a run's `label` and `exit_status`, a source's `method`, `visibility` and `rights.holder`).

**Extensions.** The extensions of the claimset format (claimset §2.7, `spec/extensions/`) may have a part for the lab record: blocks on lab events and event types of their own. A record declares each extension it uses with an `extension` event (§5.22); an event's body then holds the extension's fields in its `ext` block (`"ext": {"qualitative": {…}}`), on the event types the extension allows, or the event is of a type the extension defines (`coding`, qualitative extension). Export carries the blocks into the claimset objects they correspond to and declares the extensions there (§9). As for claimsets, an extension never changes what a field of this specification means.

## 3. Layout

A **study** is a piece of research with its own lab record: one or more related questions, and the hypotheses, plans and runs that answer them. A repository can hold several studies, and a study can give several papers.

**The record and the code are kept apart.** Each study lives on its own branch of the project's repository, `claimset/<study>`, which holds only the study's folder and shares no history with the code. The branch is checked out as a git worktree at `claimset/<study>/` in the code checkout, and the code branches ignore `claimset/` (`.gitignore`: `/claimset/`). So the code checkout shows every study in one place, while no code branch ever holds a record:

```
claimset/                      ignored by the code branches
  <study>/                     the checkout of branch claimset/<study>
    lab/                       the lab record
      lab.jsonl                the main event log, one event per line
      agents/<actor>.jsonl     per-agent logs while agents work concurrently (§3.5)
      anchors.jsonl            timestamp anchors of the log (§7); on a study branch, receipts of earlier pushes
      config.json              the study branch and its remote (§7.1)
      runs/<run id>/           per-run outputs: stdout, stderr, results.json, other files
      sources/<source id>/     captured public sources: the original file and its text (§5.20)
      private/                 kept out of git: private sources' files, the pseudonym salt (§5.20)
    <paper>.claimset.json      a paper drawn from the study (claimset §2.4)
    <paper>.claimset.txt       its reading view
claimset-code/<study>/         on the code branches: code only this study uses (prototypes, benchmarks)
```

- **Never merged.** A study branch is never merged into a code branch, and nothing is merged into it; it is never rebased, force-pushed or deleted. The code goes through its own pull requests; the record is pushed to its study branch.
- **Names.** A study's and a paper's name is lower-case letters, digits and hyphens, starting with a letter, at most 60 characters (`eigen-checker`). The study's name is the name of its branch and folder and is recorded in `init` (§5.1). A paper's name is unique within its study; the first paper of a study usually takes the study's name.
- **Code.** All code is on the code branches. Code that belongs to the project (a package's source) stays where it belongs; code only a study uses goes in `claimset-code/<study>/`. Runs record the commit they ran (§5.9), and the study branch keeps that commit reachable, so the record need not hold code and the code need not reach the main branch.
- **Setting up.** `claimset setup --study NAME --title …` creates the study branch with its first commit and lab, and checks it out at `claimset/NAME/`; in a fresh clone, `claimset setup` checks out every study branch of the remote. Outside a git repository a study is a plain folder `claimset/<study>/`.
- Records begun before v0.7 may live on a code branch (in `claimset/<study>/`, `lab/` at the top, or elsewhere); they are moved to their own branch by adoption (§3.4).

### 3.2 Finding the lab

A command works on the lab named by `--lab DIR` or `$CLAIMSET_LAB`; else on the study named by `--study NAME` or `$CLAIMSET_STUDY` (`claimset/NAME/lab`); else on the only study in the repository; else on `lab/` in the working directory (records before v0.6). With several studies and none named, the command is refused and names them.

### 3.3 Package builds

The studies are part of the repository, not of what it ships, and a build reads the working directory, worktrees included. When a study is set up, the tool adds `/claimset/` to `.gitignore` and, to each of these files the repository already has, the lines that leave `claimset/` and `claimset-code/` out; `verify` reports a file without them:

| File | Lines |
|---|---|
| `.Rbuildignore` (R packages) | `^claimset$`, `^claimset-code$` |
| `MANIFEST.in` (Python source distributions) | `prune claimset`, `prune claimset-code` |
| `.npmignore` (npm packages) | `claimset/`, `claimset-code/` |
| `.gitattributes` (repository statistics) | `claimset/** linguist-documentation`, `claimset-code/** linguist-documentation` |

### 3.4 Moving a lab

A lab is moved only by the tool, which moves the folder, moves any other files the record names (its code), and records a `relocation` event (§5.18) in the moved log. Paths in earlier events are not rewritten: a reader maps them through the relocations. Anchors made before the move hold the log at its old path; remote verification looks for it there (§7.1). A lab moved by other means fails verification once it is in `claimset/` under a name other than its study's (rule L19).

**Adoption** moves a study committed on a code branch onto its own branch (`claimset setup --study NAME --adopt`). The study's folder must be at `claimset/NAME/` and committed (`claimset lab move` brings a lab from elsewhere there first). The tool makes the study branch from that folder, without its `code/` subfolder, which moves on the code branch to `claimset-code/NAME/`; a second commit on the study branch takes the code branch's commit as a further parent, so the code the record names stays reachable. It removes the folder from the code branch (to be committed with the `.gitignore` change), checks the branch out in its place, and records a `relocation` with the branch. From then on the record is written only there. A study folder without a lab (a paper written before any record) is adopted the same way and gets a new lab, begun at the adoption with the working title given; what was done before it is not recorded, and its papers say so.

`lab.jsonl` is UTF-8 JSON Lines. Every line is one event.

### 3.5 Several agents

When several agents work on one lab at the same time, each can write its own log instead of the main log, for example in its own working copy of the repository. The agent logs are merged into the main log when the agents finish.

1. **Starting.** Every agent is registered by an `actor` event in the main log. `start` then creates the empty file `lab/agents/<actor id>.jsonl`. From then on, every event written by that actor goes to its agent log, until the log is merged.
2. **Chain.** An agent log has its own `seq` (from 1) and its own hash chain. Its first event's `prev` is the header hash of the latest main-log event when that first event was written. That main-log event is the log's **base**: it records which state of the lab the agent worked from.
3. **IDs.** Event IDs in an agent log are prefixed by the actor ID: `AC2.H1`, `AC2.RN4`, numbered within the log. IDs therefore never collide between agents. Run output directories use the same IDs (`runs/AC2.RN4/`).
4. **What an agent sees.** An agent's events may refer to main-log events up to its base and to earlier events in its own log, and to nothing else: not to later main-log events, and not to events in other agent logs.
5. **What an agent may write.** Only its own actor writes an agent log. Its confirmatory runs are made under plans written in the same log, so that two logs never race to run the same planned test. `init`, `actor` and `merge` events are written only to the main log. A correction (`supersedes`) in an agent log corrects only an event of the same agent log; other events are corrected in the main log after merging. Approvals written by the human are therefore given in the main log, after the log they approve is merged; an agent records an approval of a plan of its own log in that log, on the human's behalf (§5.8).
6. **Once.** An agent log is merged once. After the merge, the actor's events go to the main log; for new concurrent work, an agent is registered as a new actor.
7. **Merging.** `merge` appends to the main log a `merge` event (§5.16) followed immediately by the events of the agent logs it names, interleaved in order of `at`, each agent's own order kept, ties broken by actor ID, then `seq`. Each merged event keeps its ID, type, `at`, `by`, `supersedes`, `content_hash` and body, gets a new main-log `seq` and `prev`, and records in `origin` its position and header hash in the agent log: `{log: "agents/AC2.jsonl", seq, header_hash}`. The agent log is kept unchanged, so both chains remain verifiable.
8. **Refusal.** Before writing anything, the tool verifies each agent log on its own and checks every event against the main log as it will be after the events before it. If any check fails (for example, a status change that is no longer valid because the main log has moved on), nothing is written and the problem is reported.

Events written directly to the main log have `origin: null`.

## 4. Event structure

Each event has a **header**, written by the tool, and a **body**, specific to the event type:

```json
{"seq": 17, "id": "RN4", "type": "run", "at": "2026-10-15T14:03:22Z", "by": "AC1",
 "prev": "sha256:…", "content_hash": "sha256:…", "supersedes": null, "origin": null,
 "body": { … }}
```

| Field | Notes |
|---|---|
| `seq` | Integer, starting at 1, increasing by exactly 1 per event |
| `id` | Event ID, prefixed by type (§5), unique in the log |
| `type` | Event type (§5) |
| `at` | UTC time the tool wrote the event, ISO 8601 with seconds and `Z` (`2026-10-15T10:00:00Z`) |
| `by` | Actor ID (§5.2) of whoever caused the event |
| `prev` | `sha256` of the previous event's header (all header fields, canonical JSON); for `seq` 1, the hash of the empty string |
| `content_hash` | `sha256` of the body (canonical JSON: sorted keys, no insignificant whitespace) |
| `supersedes` | ID of the event this one corrects, or `null` |
| `origin` | For events merged from an agent log: `{log, seq, header_hash}`; otherwise `null` (§3.5) |
| `body` | Type-specific content; it may hold extension blocks in `ext` (§2.1) |

The headers form a hash chain. Because each header commits to its body only through `content_hash`, a body can be withheld at publication (§8) without breaking the chain: a reader can still verify the order, timing and integrity of every event.

### 4.1 Canonical JSON and hashes

- **Canonical JSON** of a value is its JSON serialisation with object keys sorted, no whitespace outside strings (separators `,` and `:`), non-ASCII characters written as UTF-8 rather than escaped, encoded as UTF-8 bytes. (In Python: `json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`.)
- Every hash is written `sha256:` followed by 64 lowercase hexadecimal digits.
- `content_hash` is the SHA-256 of the canonical JSON of `body`.
- The **header hash** of an event is the SHA-256 of the canonical JSON of an object holding all header fields: `seq`, `id`, `type`, `at`, `by`, `prev`, `content_hash`, `supersedes`, `origin`. The `prev` of the next event is this value.
- For `seq` 1, `prev` is the SHA-256 of the empty byte string.

### 4.2 Actors and IDs

- `by` is an actor ID. Two exceptions: the `init` event has `by: null`, and an `actor` event may name its own ID (an actor registering itself).
- An event ID is its type prefix (§5) followed by a positive integer: `H1`, `H2`, `RN17`. Within each prefix, numbers are allocated in order: the next number is one more than the highest used so far in the log. Events from agent logs carry the actor prefix (`AC2.H1`) and are numbered within their agent log, independently of the main log (§3.5).

### 4.3 Corrections

Events are never edited. To correct one, write a new event of the **same type** whose header names the old one in `supersedes`. The new event has its own ID and a complete new body. An event can be superseded at most once, so corrections form a chain.

The first event of a chain is the **root**. Other events keep referring to the root's ID (or any ID in the chain), and every reference resolves to the latest event of the chain. Status changes, checks and evidence that point at a hypothesis therefore still apply after its statement is corrected. A correction is for errors in what was recorded (a typo, a wrong reference); a change of mind about a hypothesis is a `status` event with `to: revised` and a new hypothesis.

### 4.4 Writing

Only the tool writes the log. A writer takes an exclusive lock, the file `lab/.lock` created atomically, appends one complete line and releases the lock. A lock older than 60 seconds is treated as stale. Several agents working at the same time write separate logs (§3.5) rather than contending for one lock.

### 4.5 Protecting the log

An agent that can run commands as the researcher can, in principle, do anything the researcher can do to a file, including rewriting the whole log and recomputing every hash. Protection therefore comes in levels:

| Level | How | Effect |
|---|---|---|
| Tamper-evident | the hash chain (§4.1) and `verify` | edits, deletions and reordering are detected, unless the whole chain is rewritten |
| Guarded | the tool's guard, installed as a hook in the agent's harness (§4.6), refuses tool calls that would modify the lab directory other than through `claimset lab` | accidental or casual modification is prevented; a determined agent can work around it. Only in harnesses that run a command before a tool call |
| Append-only file | the operating system's append-only attribute on the log files (on Linux, `chattr +a`, set by an administrator) | the operating system refuses any change except appending, as long as the agent cannot act as an administrator |
| Anchored | the head of the chain is regularly recorded outside the agent's reach (§7) | any rewriting before an anchor is detected, including a full recomputation of the chain |
| Remote append-only | the authoritative copy is held by a service that accepts only appends (the Claimset platform; or a git branch protected against force-pushes and deletion, pushed to with credentials that cannot change the protection) | rewriting is prevented, not just detected |

Locally, the log is tamper-evident. With anchoring, tampering is always detected. With a remote append-only copy, it is prevented. A lab that matters should use at least anchoring; a lab on a study branch is anchored at every event.

**Keeping record and code apart** (§3) is protected in the same layers:
- the tool refuses to write a record off its study branch (L20), and refuses runs started from the study's folder;
- the guard refuses git commands that would merge, rebase or cherry-pick between study branches and code, check a study branch out in the code checkout, force-add `claimset/` to a code branch, or force-push, delete or rename a study branch;
- on the host, protect `claimset/*` against force-pushes and deletion, and add the check that `claimset setup --ci` writes: a pull request fails if it changes `claimset/`, comes from or goes to a study branch, or contains a study branch's own history;
- the guard's configuration also shows the state of every study (`claimset studies`) at the start of each agent session, so that work, implementation included, begins inside the record.

### 4.6 Agent harnesses

Agents run in harnesses (Claude Code, Codex CLI, Gemini CLI, Qwen Code, opencode, GitHub Copilot, Cursor, Cline and others), which differ in where they read skills and instructions, in how they let a command inspect a tool call before it runs, and in what their shells may reach. The record and its rules are the same in all of them; only the setup differs, and `claimset init --agent NAME` makes it (`claimset init --list` shows what it knows of each harness):

- **Skills and instructions.** The skills are installed once in each folder the chosen harnesses read: `.agents/skills/` reaches most of them; Claude Code, among others, reads only its own. The instructions are one block in `AGENTS.md`; a harness that reads another file and can import (Claude Code's `CLAUDE.md`, Gemini CLI's `GEMINI.md`) gets a line importing `AGENTS.md`, never a copy. The block names the skills by path, so an agent whose harness does not load skills reads them as files.
- **The guard** (§4.5) reads a proposed tool call in the shape any of these harnesses sends it, as files the call would write (a patch's files included) and commands it would run, and refuses it by exit code 2 with the reason on standard error (Cline: its JSON answer). `claimset init --guard` registers it in each harness's own hook configuration; a harness without such hooks has no guard, and its record relies on the other layers. Some harnesses run project hooks only once the human has trusted them: the guard is in place only from then.
- **The harness** that identifies itself in the environment is recorded as the agent's `system` (§5.2).
- **The network.** A harness may run commands in a sandbox without network. Anchoring (§7.1) and submission then fail; the tool says so, and how that harness lets the git host and the platform through. A study whose agent cannot reach the remote is anchored from a shell that can, or kept in desk mode (§3).

## 5. Event types

### 5.1 `init` (one per lab, `seq` 1, ID `INIT`)

`lab_spec_version` records the version of this specification the record follows. Rules introduced in a later version do not apply to records begun under an earlier one.

`{lab_spec_version, working_title, repository, started_at, study}`. Its `by` is `null`. `study` is the study's name (§3); records begun before v0.6 may lack it, and a relocation can give it.

### 5.2 `actor` (prefix `AC`)

Registers a person or agent who will write events. Mirrors claimset contributors.

`{kind, name, orcid, models: [{provider, name, version, identifier}], system, record_source, words, given_at, channel}`: `kind` is a contributor type, and `models`, `system` and `record_source` are as for agent contributors (claimset §3.11, §3.12). `system` (`{name, version, uri}`, agents only) is the harness the agent runs in; `claimset lab actor` records the harness that identifies itself in the environment (§4.6) unless `--system` names one. An agent that does not know its model from its system prompt or environment is given it by the human in `$CLAIMSET_MODEL` (`provider|name|version|identifier`), recorded with `record_source: user_reported`.

**Registration.** An actor is registered by itself (the event's `by` is its own ID) or by an actor already registered (L8). Every event's `by` is the actor who wrote it, so:

- **An agent registers itself first**, before anyone else: `claimset lab actor --kind agent --name … --model …`. That is the only registration it writes as itself.
- **A person** registers themselves (`claimset lab actor --kind human --name …`, run by them), or an agent registers them: `claimset lab actor --kind human --name … --orcid … --words "…" --by <agent>`. The event is the agent's, and `words` is the person's reply giving their name (and ORCID iD), verbatim; `given_at` (when they gave it, UTC) and `channel` (chat, email, in person) are optional. An agent correcting a person's registration quotes them too (L28).
- **No person present.** An agent works alone and registers only itself; `verify` notes that the record has no human actor. A person is registered later the same way, and their ID is used only from then on. Approvals (§5.8) and the guarantor export asks for need one.
- `words` is given only when an agent records what a person told it; an actor registering itself, or a person registering someone, writes as themselves and gives none.

The tool cannot prove who typed a command: once a person's ID exists, nothing in the record stops an agent from writing with it. The rule makes the honest path the normal one and puts the person's words in the record; the protections of §4.5 keep the log from being rewritten afterwards, but do not stop a false `by` when an event is written. A person who wants their events to be beyond doubt writes them themselves. Records begun before v0.19 had no rule L28: their registrations stay valid, and `verify` notes people who registered themselves and then the agent, as the lab skill then had the agent type both.

### 5.3 `question` (prefix `Q`)

`{text}`: a research question or aim.

### 5.4 `hypothesis` (prefix `H`)

`{statement, question, origin, prompted_by}`

- `statement`: follows the claimset statement rules (one assertion, self-contained, scope stated, no rhetoric).
- `question`: the `Q` it addresses.
- `origin`: `a_priori` (proposed before looking at results) or `from_exploration` (suggested by results).
- `prompted_by`: required for `from_exploration`: the result, run or note IDs that suggested it.
- `about`: optional, recommended: the atom of the statement, as in claimset §3.3.5: `{subject, relation, object, comparator, conditions}`. Each slot except `relation` names a term (`TM3`) or gives short text; `object` may be a list of terms of kind `variable`; `object` and `comparator` may be `null`; `conditions` is a list. Every term `about` names is marked in the statement (`[[TM3:MASE]]`), and every mark names a term (L21). Recording the atom when the hypothesis is written fixes what it is about before any test: the measure, the comparator, the cases.

A revised hypothesis is a new `H` event; the old one gets a `status` event with `to: revised` and `revised_to` naming the new one.

### 5.5 `status` (prefix `ST`)

`{target, to, rationale, evidence, revised_to}`

- `target`: a hypothesis ID.
- `to`: `testing` · `supported` · `not_supported` · `inconclusive` · `revised` · `abandoned`.
- `rationale`: required, self-contained.
- `evidence`: required for `supported`, `not_supported` and `inconclusive`: run and result IDs.
- `revised_to`: required for `revised`.
- `plan`: the decided confirmatory plan whose test this status reports (a **verdict**, §5.7.4).
- `rule`: for plans whose rules are text: which rule the evidence met, `supported` · `not_supported` · `neither`.

`abandoned` means dropped without a conclusive test; `not_supported` means tested and refuted under a rule stated in advance; `inconclusive` means tested without either rule being met.

**Verdicts on planned tests (L17).** A status that names a plan is a verdict: the plan is decided and names the hypothesis, the status cites the plan's deciding runs or their results, and its `to` is the outcome the plan's rule gives: computed by the tool from the deciding runs' results for expression rules (§5.7.3), or `supported`, `not_supported` or `inconclusive` for a `rule` of `supported`, `not_supported` or `neither`. A status that cites a decided plan's deciding runs or their results as evidence for a hypothesis that plan names is a verdict, and names the plan.

### 5.6 `assumption` (prefix `AS`)

`{statement, category, check, check_evidence}`. Categories and check values as in the claimset specification (§3.8).

### 5.7 `plan` (prefix `PL`)

`{kind, hypotheses, design, analysis, setup, new_data, exclusions, addresses, decision, runs, requires_approval, data, holdout, code_version, dirty}`

- `kind`: `confirmatory` or `exploratory`. Required.
- `hypotheses`: hypothesis IDs the plan tests. Required (at least one) for `confirmatory`. One plan can test several hypotheses: when one computation decides several, they are named in one plan, with a quantity and rules that decide each (a verdict is recorded for each, §5.7.4), and the computation is run once.
- `design`: what will be run, on what, with which settings. Required.
- `setup`: optional, recommended for `confirmatory`: the parts of the test as terms by role, `{studied, cases, procedure, lens, context}`, each a non-empty list of term IDs (claimset §3.4): what is studied or compared, on what cases, by what procedure, measured or read through what. Exported as the setup of the evidence its runs give (§9).
- `analysis`: what will be computed from the outputs. Required.
- `new_data`: optional text: what the test uses that the runs which suggested its hypotheses did not (new data, a seed that changes the sample, new models or settings). Recommended when a hypothesis it names is `from_exploration` (§5.7.6): a test on what the exploration already saw can only repeat it.
- `exclusions`: optional list of `{name, rule}`: each rule that leaves units out of the analysis (models near a boundary, outliers, trimmed observations), with a name and the rule as text. The deciding runs report how many units each excluded, as a result whose `variable` contains `excluded` or the exclusion's name (§5.7.6), so readers see how much of the sample the verdict rests on.
- `addresses`: optional list of the global IDs of review comments (`doi:…@v1/RV1.c6`) whose resolution test the plan runs (§5.7.7).
- `decision`: the decision rule (§5.7.1). Required for `confirmatory`.
- `runs`: how many successful confirmatory runs decide the plan (§5.7.2). A positive integer; default 1.
- `data`: optional list of `{description, hash}` for data that exists when the plan is written.
- `holdout`: optional list of `{path, hash}`: data kept for this test only (§5.7.5).
- `requires_approval`: optional, `true` when the plan is to be run only after a supervisor approves it (L23). Set it for every confirmatory plan when the study has a supervisor.
- `code_version`, `dirty`: the commit of the repository when the plan was written, and whether it had uncommitted changes outside the lab directory; recorded by the tool, as for runs (§5.9).

Plans are immutable. A changed plan is a new plan that supersedes the old one; runs already made under the old plan keep pointing at it. The decision rule that decides a plan is the one in the version its first deciding run names in `plan_version` (§5.9); corrections written after that do not change the verdict.

Records begun under v0.4 or earlier have `criterion` (a free-text judgement rule) instead of `decision` and `runs`.

#### 5.7.1 Decision rule

Everything that decides the outcome is fixed in the plan, before any run: the framework, its parameters, the results the decision is made on, and the rules. Nothing is chosen after the results are seen.

```json
"decision": {
  "framework": "frequentist",
  "parameters": {"alpha": 0.05, "test": "one-sided paired t-test on per-series MAE differences, naive minus SES",
                 "sides": "greater", "correction": "none",
                 "sample": "500 simulated series per ratio, fixed in advance; no interim analysis"},
  "quantities": {
    "p": {"variable": "p-value, paired t-test of MAE difference naive minus SES", "conditions": {"ratio": 1}},
    "d": {"variable": "mean MAE difference, naive minus SES", "conditions": {"ratio": 1}},
    "d_upper": {"variable": "mean MAE difference, naive minus SES", "conditions": {"ratio": 1},
                "field": "interval.upper"}
  },
  "supported": "p < alpha and d > 0",
  "not_supported": "d_upper < 0.01"
}
```

The decision rule is the test a claim carries (claimset §3.3.3), written before any run:
- `framework` and `parameters`: one of the claimset's frameworks, with the parameters it requires (claimset §3.3.3), and any others the analysis fixes in advance.
- `quantities`: optional. When given, the rules are expressions over them (claimset §3.3.3) and the tool computes the verdict (§5.7.3); when absent, the rules are text and the writer states which rule the evidence met. Computed verdicts are stronger, and recommended whenever the outcome can be computed from recorded results. Each name binds exactly one result of the deciding runs, by its `variable`, `conditions`, `field` and, with several deciding runs, `run` (1 for the first deciding run, and so on), as claimset §3.3.3 says; it is bound among the results of the deciding runs (that run's only, when `run` is given). Records begun before v0.20 name the variable `quantity`. An extension may add quantities of its own kind: counts of coded material are the qualitative extension's (`spec/extensions/qualitative-v0.1.md` §2.4).
- `supported` and `not_supported`: the rules for the two conclusive outcomes (claimset §3.3.3); if neither holds, the outcome is `inconclusive`.

**Rules for each hypothesis.** One computation can decide several hypotheses by different rules: a share below a threshold for one, an ordering of totals for another. The plan then gives each hypothesis its own pair of rules, keyed by its ID:

```json
"decision": {
  "framework": "deterministic",
  "parameters": {"threshold": 0.10, "sample": "all 654 held-out issues; one run"},
  "quantities": {
    "s": {"variable": "share of issues in theme", "conditions": {"theme": "arranging"}},
    "n": {"variable": "total reactions on issues in theme", "conditions": {"theme": "notation"}},
    "ui": {"variable": "total reactions on issues in theme", "conditions": {"theme": "input_ui"}}
  },
  "rules": {
    "H1": {"supported": "s < threshold", "not_supported": "s >= threshold"},
    "H2": {"supported": "n > ui", "not_supported": "n <= ui"}
  }
}
```

- `rules` has one entry for each hypothesis the plan names (by the ID the plan names it by) and none for another. An entry has `supported` and `not_supported`, and optionally `framework` and `parameters`: its own framework, and parameters added to the shared ones (overriding a shared parameter of the same name). With `rules`, the decision has no shared `supported` or `not_supported`.
- The rule that decides a hypothesis is the shared part (`framework`, `parameters`, `quantities`) with its entry: it must be complete as a rule with one pair is (L15). The quantities are shared, so all rules are expressions, or, without quantities, all are text.
- The deciding runs are the plan's, once for all its hypotheses, and each hypothesis gets the verdict its own rules give (§5.7.3, §5.7.4).
- A decision without `rules` gives one pair that decides every hypothesis the plan names, as before.

**Why both rules.** Failing to reject a null hypothesis is not evidence against the hypothesis. A negative result is conclusive only if the plan said in advance what would refute the hypothesis: an equivalence bound, a significant effect in the opposite direction, a posterior threshold, a counterexample. Without that, the outcome is `inconclusive`.

#### 5.7.2 Deciding runs

The **deciding runs** of a confirmatory plan are its first `runs` confirmatory runs, whatever their exit status. Once a plan has its deciding runs it is **decided**, and further confirmatory runs under it are refused (L16): repeating a test until it comes out favourably is exactly the flexibility the plan removes. A run that fails still counts: a script could otherwise write its results and then fail on purpose. If a needed result is missing because a run failed, the verdict is `inconclusive`. Check that scripts work with exploratory runs on other data before the confirmatory run.

#### 5.7.3 Rule expressions

Rule expressions are those of claimset §3.3.3. The tool evaluates both rules on the results of the deciding runs and gives the outcome claimset §3.3.3 defines: `supported` if only the `supported` rule holds, `not_supported` if only the `not_supported` rule holds, `inconclusive` if neither or both hold, or if a quantity cannot be bound (no result matches, or several do).

#### 5.7.4 Reporting

Every planned test that ran is reported. Once a plan is decided, a verdict (§5.5) is recorded for each hypothesis it names; export refuses a record in which one is missing (§9). Statuses recorded before or after the verdict do not replace it: each verdict becomes a claim of its own.

A hypothesis may be tested again under a new plan (a replication, a larger sample); each test gives its own verdict and claim, so a favourable test cannot hide an unfavourable one. A claim from a planned test states the hypothesis as it stood when the plan was written, so correcting the hypothesis afterwards does not move the verdict to a different statement.

#### 5.7.5 Flags

Some things weaken a planned test without making it invalid. The tool does not refuse them; it reports them as flags of the decided plan (in `verify`, in `verdict`, and in the exported claim's `test.flags`). What each flag means is defined with the test flags of the claimset (claimset §3.3.3); here is when the tool raises it:
- `code_changed_after_plan`: a deciding run used a different commit from the one recorded with the plan, or the code had uncommitted changes when the plan was written or a deciding run was made. Commit the analysis code before writing the plan to avoid it. Runs whose code is the tool's (an extension's counter) are not counted.
- `holdout_used_elsewhere`: a run other than the plan's deciding runs declared a held-out file of the plan (by hash) as input data. Only inputs declared to the tool (`run --data`) can be detected.
- `reproduces_prompting_result`: a plan names a hypothesis recorded `from_exploration`, and a result of a deciding run has the same `variable`, the same `conditions` and the same `value` as a result of a run that suggested it (a run named in the hypothesis's `prompted_by`, or the run of a result named there). Under a rule with quantities, only the results the quantities bind are compared; under text rules every result is. Results with value 0 are not compared: a count of zero is what a correct method gives on any data.
- `sources_captured_after_plan`: a deciding run declared as input data (`run --data`) a file of a source (§5.20) captured after the plan was written. A plan that samples sources still to be collected ("the next 20 threads that a search returns") says so in its `design` and `new_data`.

An extension may define flags of its own, which the tool reports the same way and export writes into the extension's block on the claim: the qualitative extension's `coded_before_plan` and `codebook_changed_after_plan` (qualitative §2.5).

#### 5.7.6 Warnings

The tool warns, and records nothing, when:
- a plan names a hypothesis recorded `from_exploration` and has no `new_data` (at `plan`);
- a plan has no `exclusions` and a parameter of its decision rule has a name that suggests one: split at `_`, `-` and spaces, it contains `edge`, `band`, `exclude`, `exclusion`, `excluded`, `trim`, `trimmed`, `outlier`, `outliers` or `cutoff` (at `plan`);
- a decided plan declares an exclusion and no result of its deciding runs has a `variable` containing `excluded` or the exclusion's name (at `verdict`);
- a confirmatory run has the same `command`, `code_version`, `seeds` and `parameters` as a successful confirmatory run of another plan (at `run`, and as a note of `verify`): one computation recorded twice; one plan naming both hypotheses would have needed one run;
- a confirmatory plan has no `setup` (at `plan`);
- a plan's `setup` does not name a term that the `about` of a hypothesis it tests names, when the term's kind is `method`, `intervention`, `population`, `data`, `design` or `variable` (at `plan`): the test would decide a hypothesis about something it does not study or measure. A hypothesis about MASE tested by a plan that measures MSIS is a switched outcome; one about a comparison with N tested by a plan that runs only M and P is a switched comparator. Revise the hypothesis, or add the part to the setup.

#### 5.7.7 Plans that answer a review

A review comment on a published claimset carries a resolution test: a procedure and two rules, `upheld` (the comment stands) and `withdrawn` (it falls), fixed before the author answers (revision §3.4). The author runs a computable test as a confirmatory plan:
- a hypothesis states what the comment questions, as the author holds it (`per_lag_max() reproduces the check of smooth 4.6.0 on the models of C1`);
- the plan's decision rule is the comment's current test: its framework, parameters and quantities, `supported` its `withdrawn` rule and `not_supported` its `upheld` rule, unchanged; `design` is its procedure;
- `addresses` names the comment.

The verdict then says which rule of the comment's test the evidence met: `supported` means the comment's `withdrawn` rule was met, `not_supported` its `upheld` rule, `inconclusive` neither. All rules of planned tests apply (L15–L17); a test that comes out against the author is reported like any other. A plan may address several comments when one run settles them all. Tests of the framework `documentary` (decided by what a cited source states) and text rules are not run in the record: the author answers them in the response.

The claimset's next version comes with the record of the previous version followed by the events written since (revision §5.4): the record is never restarted or rewritten for a revision, and the work that answers a review is appended to it.

### 5.8 `approval` (prefix `AP`)

`{target, comment, on_behalf_of, words, given_at, channel}`: a human approves a plan (or another event) before it is acted on. Optional unless the plan requires it (`requires_approval`, L23).

An approval is written in one of two ways:
- **By the human** (`by` an actor of kind `human`), running the tool themselves.
- **By an agent on the human's behalf** (`by` an agent actor, `on_behalf_of` the human actor). The agent asked the human, waited, and records the answer: `words` is the human's reply as they gave it, verbatim (`"Yes, go ahead with PL2."`); `given_at` is when they gave it (ISO 8601, UTC; the event's own time when absent); `channel` optionally says where (`conversation`, `email`, `meeting notes`). The human makes the decision; the agent only records it.

An agent never approves on its own judgement. It does not treat silence, a general instruction ("do everything you need"), or an answer to a different question as approval; a standing approval counts only if the human gave it for the plans it is used for ("approve every confirmatory plan for Q2 as written"), and then `words` quotes it and `given_at` is when it was given. The tool cannot tell whether the words are genuine: like every event, a relayed approval is tamper-evident once written, and it is shown as relayed, so a reader sees that the supervisor's words were recorded by the agent. An approval of a plan applies to the version it follows: a plan corrected after its approval needs a new approval (L23).

### 5.9 `run` (prefix `RN`), written by the tool

`{plan, plan_version, label, command, code_version, dirty, parameters, seeds, data, environment, started_at, ended_at, exit_status, exit_code, outputs}`

- `plan`: plan ID or `null`.
- `plan_version`: the ID of the version of that plan in force for the run's writer when the run was made (the latest correction it could see). The version named by the first deciding run decides the plan (§5.7.2); this holds also for runs merged from agent logs, whose writer may not have seen corrections made meanwhile in the main log.
- `label`: `confirmatory` (only if rule L5 holds) or `exploratory`.
- `command`: the command as a list of arguments.
- `code_version`: commit hash of the working directory's repository, or `null` if it is not a git repository; `dirty`: true if it had uncommitted changes outside the lab directory. A run is made from the code checkout, never from the study's folder. The commit of the study branch that records the run takes `code_version` as a further parent, so the code stays reachable from the record (§7.1).
- `parameters`: optional object of named settings given to the tool; `seeds`: optional list of random seeds given to the tool.
- `data`: list of `{path, hash}` for input files declared to the tool.
- `environment`: `{software: {python, platform}, hardware: {machine, cpus}}`, as recorded by the tool.
- `started_at`, `ended_at`: UTC times as in `at`.
- `exit_status`: `success` (exit code 0) · `failed` (non-zero exit code) · `crashed` (killed by a signal, or the command could not be started) · `cancelled` (interrupted by the user).
- `exit_code`: the process's exit code, or `null`.
- `outputs`: list of `{path, hash}` for every file in the run's directory after the run, with paths relative to that directory.

#### 5.9.1 How the tool runs a command

1. It checks rule L5 before starting. A run is confirmatory only when it names a confirmatory plan; otherwise, or when asked, it is exploratory.
2. It creates a pending run directory under `runs/` and starts the command with two environment variables: `CLAIMSET_RUN_DIR` (the directory, for any output files) and `CLAIMSET_RESULTS` (the path of `results.json` inside it). An extension the record declares may add files and variables of its own, recorded in its block on the run: when the record has codings, the qualitative extension writes them to `codings.json` and names it in `CLAIMSET_CODINGS` (qualitative §2.3).
3. It writes the command's standard output and standard error to `stdout.txt` and `stderr.txt` in that directory, and shows them as the command runs.
4. When the command ends, however it ends, it hashes every file in the directory, writes the `run` event, and renames the directory to `runs/<run id>/`.
5. If `results.json` exists, it writes one `result` event per entry (§5.10.1).
6. It exits with the command's exit code.

### 5.10 `result` (prefix `RS`), written by the tool

`{run, variable, value, unit, interval, uncertainty, conditions, source}`

An observation (claimset §3.4: `variable`, `value`, `unit`, `interval`, `uncertainty`, `conditions`), plus `run` and `source`: the output file the value was read from, whose path is in the run's `outputs`. Records begun before v0.20 have `quantity` for `variable`; it is read as `variable`.

#### 5.10.1 Results file

A script reports results by writing a JSON array to the file named by `CLAIMSET_RESULTS`:

```json
[{"variable": "largest KL divergence per observation, mixture vs exact", "value": 0.00115,
  "conditions": {"distribution": "Gamma", "form": "conventional"}},
 {"variable": "share of failed fits", "value": 0.18, "unit": "proportion",
  "interval": {"level": 0.95, "lower": 0.15, "upper": 0.21}}]
```

Each entry needs `variable` (text; `quantity`, its older name, is read as `variable`) and `value` (a finite number: not NaN or infinite); `unit`, `interval`, `uncertainty` and `conditions` are optional, in the forms of claimset §3.4; an entry whose `uncertainty` is not well formed (claimset rule 82) is not valid. Values are written exactly as computed. An entry that is not valid produces no event and is reported by the tool.

### 5.11 `decision` (prefix `DC`)

`{text, prompted_by, affects}`: a choice that changes the direction of the work (e.g. switching metric, dropping a model class, changing the data). This is where the "garden of forking paths" is recorded.

### 5.12 `deviation` (prefix `DV`)

`{plan, run, text, reason}`: a run that departed from its plan, and why. `plan` and `run` are required, `text` says what differed, and `reason` (required) says why.

### 5.13 `check` (prefix `CK`)

`{target, outcome, extent, notes, resolution}`: an independent check of a hypothesis, result or derivation; same semantics as claimset checks (claimset §5). The checking actor must not have written the target.

### 5.14 `artefact` (prefix `AR`)

`{kind, path_or_uri, hash, licence, description}`: registers a file that is not a run output (data set, figure, manuscript, proof file); `kind` is an artefact kind (claimset §3.6).

### 5.15 `note` (prefix `NT`)

`{text, refers_to}`: an observation that is not yet a hypothesis, decision or result. Notes are kept but not exported as claims.

### 5.16 `merge` (prefix `MG`), written by the tool

`{agents, events_merged}`: records that the agent logs of the listed actors were merged into the main log (§3.5). The merge event is followed immediately by exactly `events_merged` events, each with an `origin` in one of the listed logs: every listed log whole, from its `seq` 1, in the order of §3.5. A merge event is never withheld (§8).

### 5.17 `paper` (prefix `PA`)

`{name, questions, title}`: a paper drawn from the record, with the questions it reports (each by its ID, at least one) and an optional working title. A question is reported in one paper (rule L18); to move a question from one paper to another, correct both (`--supersedes`). Written only to the main log. A record without paper events gives one paper, from all its questions.

Papers split a study's results between them; the split must not decide what is reported. So a paper covers whole questions, never chosen hypotheses; each paper's exploration summary gives the whole lab's counts beside its own and names the other papers (§9); and `verify` names every hypothesis with a result (a verdict, or a final status `supported`, `not_supported` or `inconclusive`) whose question no paper reports.

### 5.18 `relocation` (prefix `MV`), written by the tool

`{from, to, study, moved, branch}`: the lab moved from `from` to `to` (paths of the lab folder relative to the top of the repository, or, for `to` with `branch`, to the top of the study branch); `study` names the study when the move gives it one; `moved` lists other files or folders the record names that moved with it, as `{from, to}`; `branch` is the study branch the lab now lives on, after an adoption. Written only to the main log, after the move (§3.4).

### 5.19 `term` (prefix `TM`)

`{term, kind, definition, source, broader}`: a term the study names: what it studies or compares, the cases, the procedure, what it measures, the concepts and theories it uses or finds. It becomes a definition of the claimset (claimset §3.2), whose fields it has.
- `term`: the term as the study writes it. Required.
- `kind`: a kind of definition (claimset §3.2). Required. (`measure` in records written under v0.12 is read as `variable`.)
- `definition`: what the term means in this study, with what the study specifies (settings, subset, horizon, aggregation). For the study's own method or model, every equation that defines it, as the code implements it (claimset §3.2). Required.
- `source`: optional: where a borrowed term is defined (a DOI, an arXiv ID, a citation).
- `broader`: optional list of term IDs: the more general terms this one is a kind of (a code's theme, a variant's method), as in claimset §3.2.

A code of a qualitative codebook is a term of kind `concept` with the qualitative extension's block (qualitative §2.1). Records begun before v0.20 have it as the term's `code`.

A term is recorded once and corrected with `--supersedes` when what it means changes; the correction keeps its place, so hypotheses and plans that named it keep naming it, and the record shows when its meaning changed. A term changed after a plan that names it in its setup was written is a change to the plan's test, and is recorded as a deviation if the runs used the new meaning.

### 5.20 `source` (prefix `SR`), written by the tool

`{uri, title, method, retrieved_at, format, files, visibility, rights, archive_uri, venue, published_at, description}`: outside material kept as data: a web page, a forum thread, an issue, a transcript, a document. `claimset lab capture FILE` copies the file into `sources/<id>/` (a public source) or `private/sources/<id>/` (a private one), hashes it and writes the event; `claimset lab capture --fetch --uri URL` has the tool fetch a public page itself.

- `uri`: where the source was found. Required when `method` is `fetched`; a transcript or a document with no address has none.
- `method`: how the file was obtained. `saved`: the researcher saved it by hand (the default; it records what the researcher saw, including a page that needed a sign-in or was read after a cookie banner); `browser`: captured from a browser by a tool (an extension, print to file); `fetched`: fetched by the lab tool. Fetching is for public pages and repeatable re-captures; it records a different method because the tool, not the researcher, saw the page.
- `retrieved_at`: when the source was retrieved, a UTC time or a date: the researcher's account for `saved` and `browser`, the tool's time for `fetched`. The event's `at` is when it was recorded; both are kept.
- `format`: the original file's media type (`text/html`, `application/pdf`, `text/plain`).
- `files`: `{path, hash, role}` for the files of the source's folder: exactly one `original` (the file as saved or fetched) and, when the tool can read the format or the researcher gives one (`--text`), its `text`: the visible text of an HTML page (script, style and head left out; a line per block), or a text the researcher made of a PDF or image. Excerpts are checked against the text (L25).
- `visibility`: `private` (its files are kept out of git and out of the published package; the record holds their hashes, and its metadata and hashes are published, with short excerpts and their locators) or `public` (published with the package). Content written by others is `private` unless the researcher says otherwise.
- `rights`: `{holder, licence, terms}`: `holder` is `study` (the study's own material) or `third_party` (written by others: posts, issues, pages, documents); `licence` is the source's SPDX licence, if it has one; `terms`, if it has none, the terms under which it is shared (a site's terms of service, a permission). The paper's licence never applies to content written by others. A `public` third-party source has a `licence` or `terms` that permit publishing it (L26).
- `archive_uri`: an archived copy of exactly this page (a web archive snapshot): a witness, independent of the study, that the page said this at that time. The record's hash shows only that the file has not changed since it was captured.
- `author`: who wrote it. `capture --author HANDLE` records a pseudonym, `a-` and 12 hexadecimal digits of the SHA-256 of the study's salt (the file `private/pseudonym-salt` in the lab, made on first use and kept out of git) and the handle (whitespace collapsed, case folded), so the same author has the same pseudonym across the study's sources and no handle appears in the record. Because the salt is not in the record, someone who has the record cannot check a guessed handle against a pseudonym. `--author-as-given` records an organisation or a public author as written.
- `venue`, `published_at`, `description`: where and when the source was written or published, and anything else a reader needs.

**Correcting a source.** What was recorded about a source (an archived copy found later, the date it was published, its author, whether it may be published) is corrected with `claimset lab capture --supersedes SR1 --archive-uri …` and no file: the correction has the same files, copied into its own folder, and the fields given. A file captured by mistake is not corrected; capture the right one as a new source and leave the wrong one uncoded.

**Private sources.** A private source's files are kept in `lab/private/sources/<id>/`; the folder `lab/private/` holds a `.gitignore` that keeps everything in it out of git, so the files are never committed, pushed or published. The record holds their hashes, so a copy given to a reviewer or an auditor can be checked against it, and everything else its event says. The pseudonym salt is kept in `lab/private/` too.

- **Another checkout** of the study branch (a clone, a collaborator's machine) has the record without `lab/private/`. There, `verify` checks everything else and says which private sources it could not check; `claimset lab quote` and `capture --supersedes` refuse for those sources; and `capture --author` refuses once the record has pseudonyms, since a new salt would give the same author another pseudonym. Copying `lab/private/` from the checkout that captured the sources, by a channel as private as they are, restores all of it, and `verify` then checks their hashes and quotes (L24, L25).
- **Keep a copy.** `lab/private/` exists only where it was made: back it up with the study's other private data. If it is lost, the record still shows what was captured and its hashes, but the quotes can no longer be checked and new pseudonyms will not match.
- **What is still in the record.** Codings quote passages (qualitative §2.2), and a run's `codings.json` repeats them, so short excerpts of a private source are in the record; a source's `uri`, `title` and other metadata are too. Withhold events that must not be published (§8).
- **Records begun before v0.19** committed private sources' files in `lab/sources/<id>/` and the salt in `lab/pseudonym-salt`. They stay there (the branch's history is never rewritten), the tool keeps reading them, and `verify` notes them: such a branch's remote must be as private as its sources. Sources captured from now on are kept out of git; the committed salt stays in use, so pseudonyms stay stable.

A source is captured once, when it is first read for the study; a later capture of the same page is a new source, so a page that changed shows as two sources with different hashes. Runs read sources by their files (`run --data lab/private/sources/SR1/text.txt`, or `lab/sources/SR1/text.txt` for a public source), so the hashes tie each run to the material it read.

**Quoting.** `claimset lab quote SR1 "the words"` checks that the words are in the source's text (ignoring differences in whitespace) and prints their locator as a W3C Web Annotation text-quote selector, `{exact, prefix, suffix}` with the words' character positions in the text (`start`, `end`). An excerpt a claim gives (claimset §3.4, excerpts) carries that locator, so a reader with access to the source can find it, and a reader without access knows it was checked.

### 5.21 `coding` (prefix `CD`)

Defined by the qualitative extension (qualitative §2.2), with the counts of coded material (qualitative §2.4). In records begun before v0.20, as they were written then.

### 5.22 `extension` (prefix `EX`)

`{name, version}`: the record uses an extension of the claimset format (§2.1; claimset §2.7) from this event on: `name` is a registered extension and `version` the version of it the record follows, as the claimset declares them (`"qualitative"`, `"0.1"`). Written by `claimset lab extension NAME`, or by the tool itself the first time a command needs the extension (recording a code writes the qualitative extension's). An extension is declared once per record; events written after it may carry its blocks or be of its types (L30). Written only to the main log, before agent logs that use it start. Records begun before v0.20 have the qualitative extension, version 0.1, declared from their start.

## 6. Rules

L1. No line of `lab.jsonl` is modified or removed after it is written.
L2. `seq` increases by exactly 1; every `prev` matches the previous header; every `content_hash` matches its body (or its withheld body, §8).
L3. Every executed command started through the tool produces a `run` event, whatever its exit status.
L4. `result` events are written only by the tool. A result names an earlier `run`, and its `source` is a path listed in that run's `outputs`.
L5. A run is labelled `confirmatory` only if it names a `confirmatory` plan, that plan was written earlier (lower `seq`) and, when anchoring is in use, anchored before the run started (§7). Otherwise it is `exploratory`. A run labelled `exploratory` may name any plan or none.
L6. A `status` event moving a hypothesis to `supported` must not list as evidence any event in that hypothesis's `prompted_by`, nor results of runs listed there.
L7. Every `status`, `decision` and `deviation` has a rationale or reason; every `revised` status names `revised_to`.
L8. Every `by` names an `actor` event with a lower `seq`.
L9. A `check` is not made by the actor who wrote its target.
L10. Agent logs follow L1–L9 and L11–L12 on their own, starting from their base, and the restrictions of §3.5 (who writes them, what they may refer to and correct, which event types they hold). A merged event equals the event its `origin` names, whose header hash it records; an agent log is merged once, and nothing is added to it afterwards. Only merged events carry an actor prefix in the main log.
L11. A `supersedes` reference names an earlier event of the same type that has not already been superseded.
L12. Every `id` has the prefix of its `type` and is unique in the log; every reference in a body names an event with a lower `seq`.
L13. The files of a run directory match the hashes in the run's `outputs`. Run directories are never changed after the run.
L14. Every anchor receipt (committed in `anchors.jsonl`, kept in the checkout's git directory, or derived from the remote-tracking branch, §7.1) names a `seq` present in the log, and its `head_hash` equals the header hash of the event at that `seq`; the same holds for each agent log the entry names. A mismatch means the log was rewritten after it was anchored. Exception, for receipts made by tools before v0.15 while several writers appended to one lab (§7.1, *Heads of a receipt*): when a receipt names a `seq` that the log in its commit does not hold, and a later commit of the same branch holds that `seq` with the same header hash, `verify --remote` reports a warning that names the race instead of an L14 error. A named event that no later commit holds, or holds with another header hash, is an L14 error.
L15. A confirmatory plan has a complete decision rule (§5.7.1): a framework, the parameters it requires, and both rules; when quantities are given, every quantity is well formed (with `run` when there are several deciding runs) and every rule is a valid expression whose names are defined (§5.7.3); `runs` is a positive integer. A rule with `rules` has one entry for each hypothesis the plan names and none for another, no shared `supported` or `not_supported`, entries with only `framework`, `parameters`, `supported` and `not_supported`, and each entry with the shared parts is complete in this sense. In records begun before v0.20, a count quantity has only `count` and `run`, and its `count` is valid as qualitative §2.4 says; from v0.20, counts are the qualitative extension's (rule qualitative.L3).
L16. A confirmatory run is not made under a plan that already has its deciding runs (§5.7.2), or, in an agent log, under a plan written outside that log (§3.5).
L17. A status that names a plan, or that cites a decided plan's deciding runs or their results for a hypothesis the plan names, is a verdict: the plan is decided and names the hypothesis, the deciding runs or their results are cited, and the outcome is the one the plan's rule gives (§5.5); under `rules`, the rule of that hypothesis (§5.7.1).
L18. A paper has a valid name, unique among the record's papers, and names at least one question; no question is named by two papers (each correction chain counted once, its latest version in force).
L19. A lab inside `claimset/<name>/lab` belongs to the study `<name>`: its study (from `init` or the latest relocation that names one) is `<name>`, and records begun under v0.6 or later name a study.
L20. A record with a study branch is written only in the checkout of that branch; a lab in `claimset/<study>/lab` inside a git repository is written only on its study branch (adopted, §3.4).
L21. A hypothesis's `about` and a plan's `setup` name only `term` events; every term mark in a hypothesis's statement (`[[TM3:…]]`) names a term; every term `about` names is marked in the statement; a list in `about.object` names only terms of kind `variable`.
L22. An approval's `by` is a human actor, or an agent actor with `on_behalf_of` naming a human actor and non-empty `words`; an approval by a human has no `on_behalf_of`.
L23. A confirmatory run is not made under a plan with `requires_approval` unless an approval of that plan was written after its current version (the version the run names in `plan_version`).
L24. Each source's folder (`sources/<id>/`, or `private/sources/<id>/` for a private source) holds exactly the files its event records, with the recorded hashes (§5.20). A private source's folder is kept out of git, so it is checked where it is present; where it is absent, `verify` says so.
L25. An excerpt quoted from a source is in that source's text (§5.20); in records begun before v0.20, a coding's `quote` is in its source's text at its recorded position (from v0.20, rule qualitative.L2).
L26. A source whose `rights.holder` is `third_party` is `public` only with a `licence` or `terms` (§5.20).
L27. In records begun before v0.20: a coding names a captured source and, unless it is withdrawn, at least one code: a term with a `code` part; its `code_versions` name versions of those codes. The codes a plan's counts name are codes of the codebook. (From v0.20, rules qualitative.L1 and qualitative.L3.)
L28. In records begun under v0.19 or later: an `actor` event of kind `human` written by an agent actor (its `by` is an agent, not the actor registered) has non-empty `words`, the person's own reply; `words` appear only in such events (§5.2).
L29. A record begun under a version of this specification later than the tool's own (`init.lab_spec_version`) is refused: the tool does not know that version's rules. The record may be valid; read it with a current tool (update the claimset package). Records of every earlier version are checked by the rules of the version they were begun under (§5.1).
L30. In records begun under v0.20 or later: an event carries a block (`ext`) only of an extension declared by an earlier `extension` event, on an event type that extension allows, and is of a type an extension defines only when that extension is declared; an extension is declared once, at a version the tool knows (an extension or version it does not know is refused with a message saying so, as claimset rule 80). The extension's own rules apply to its blocks and events.

The tool enforces L1–L5 and L8–L23 mechanically, and refuses L6 and L7 violations. L15–L17 apply to records begun under v0.5 or later, and to records whose `init` is withheld. Where a check depends on withheld content (a withheld run, result or plan), `verify` does not report it as a violation; withholding lowers verifiability instead (§8).

## 7. Timestamp anchoring

A timestamp written by the tool proves order, not time. To make "this plan existed before this run" verifiable to others, the tool periodically anchors the head of the chain (the latest `seq` and its header hash) with an external service and records the receipt in `anchors.jsonl`:

`{seq, head_hash, service, receipt, at, agents}`

`agents` (omitted when there are none) records the head of each non-empty agent log in the same way: `{"AC2": {seq, head_hash}}`. An event is covered by an anchor if the anchor's `seq` is at least the event's main-log `seq`, or, for an event written to an agent log, if the anchor's entry for that log has a `seq` at least the event's `seq` there.

The services are the values of a claim's `planned_at_anchor` (claimset §3.3.3), in decreasing order of strength: `platform` (the platform records the hash and returns a signed receipt), `rfc3161` (a trusted timestamp authority), `git_remote` (a commit containing the log pushed to a remote host; weaker, as the host's clock and history are trusted) and `none`.

The anchor level of a plan becomes `planned_at_anchor` in the exported claimset.

**Automatic anchoring.** When anchoring is configured, the tool anchors after every `plan` event, and before a confirmatory run it makes sure the run's plan is covered by an anchor. If that anchor cannot be made, the run is refused as confirmatory (rule L5); it can still be run as exploratory. The tool can also be configured to anchor after every event.

### 7.1 Git anchoring

**On a study branch** (§3) anchoring is part of keeping the record: after every command that changes it, the tool commits the study's folder (the logs, run outputs, papers) to the study branch and pushes it. Each push is an anchor (`service: git_remote`, with the pushed commit as the receipt), so every event is anchored when it is written. The commit after a run takes the run's `code_version` as a further parent: pushing the study branch pushes the code the run used, and keeps it reachable whatever happens to its own branch. The record's own history is the first-parent history of the branch. Offline, the commits stay local and are pushed with the next command; until then `verify` and `claimset studies` report them as not pushed, and a confirmatory run is refused until its plan is pushed (L5). Configuration in `lab/config.json`: `{"record": {"remote": "origin", "branch": "claimset/<study>"}}`.

**Desk mode.** A short study (a one-day scan of forum threads, a document review) may have no remote to push to. `claimset setup --study NAME --title … --desk` keeps it on its study branch all the same: after every command the tool commits the study's folder, as above, and pushes nothing. Kept: the hash chain, the append-only logs, the rule that a plan precedes its test (L5 by order in the log), results from code and quotes from sources, and the clean checkout. Lost: anchoring. No receipt is made, so the times of events are the recording tool's clock, a confirmatory run is not refused for want of a push, the claims' `planned_at_anchor` is `none`, and the exported `lab_record` says the record was kept in desk mode. Configuration: `{"record": {"remote": null, "branch": "claimset/<study>", "mode": "desk"}}`. Running `claimset setup --study NAME --remote origin` later pushes the branch and anchors every command from then on; what was recorded before is anchored by that first push, not before.

**Receipts on a study branch.** The receipt of a push names the commit pushed, so that commit cannot hold it, and writing it into the checkout after the push would leave the checkout with an uncommitted change after every command. So:

- After a push, the tool keeps its receipt (`{seq, head_hash, service: "git_remote", receipt: {remote, branch, commit}, at, agents}`) in the checkout's git directory (`claimset-anchors-pending.jsonl` in the directory `git rev-parse --absolute-git-dir` names), which no commit and no `git status` sees.
- The next commit appends the kept receipts to `anchors.jsonl` and commits them with whatever else changed: the receipt of push n is in commit n+1. Receipts alone are not a change: a command that records nothing makes no commit, and pushing a commit already received adds no receipt.
- The receipt of the newest pushed commit is in no commit yet. In the checkout that pushed it, it is the receipt kept locally. In another checkout (a fresh clone, an agent's copy) it is derived: the remote-tracking branch `refs/remotes/<remote>/<branch>` names a commit the remote has; when that commit is in the checkout's history and no receipt names it, its receipt is made from the logs in that commit (the last event's `seq` and header hash, and each agent log's), with the commit's date as `at` and `"derived": true`. The next commit made in that checkout commits the derived receipt like a kept one.
- Every reader of the anchors (coverage of events and plans, L5, L14, `verify --remote`, export's `planned_at_anchor` and the pinned version of the exported lab record) reads the committed receipts, then the kept ones, then the derived one. A receipt is never trusted for more than it is: `verify --remote` checks every commit a receipt names against the remote branch.
- A command that succeeds leaves the study checkout clean: nothing it wrote is uncommitted.

**Heads of a receipt.** A receipt says what the commit it names holds, so its `seq` and `head_hash`, and each agent log's in `agents`, are read from that commit (the last event of each log in it, as `git show <commit>:lab/agents/<actor>.jsonl` gives it), never from the working tree. When several writers share one lab (a human and agents with their own agent logs, §3.5), another writer may append between the commit and the push; its event is in the working tree but not in the commit, and is received with the commit that holds it.

Tools before v0.15 read the heads from the working tree after the commit, so a receipt made in that race can name an event its commit lacks. Records are append-only and such a receipt is not rewritten. `verify --remote` recognises it: when the event a receipt names is absent from the receipt's commit but a later commit of the same branch holds it at that `seq` with the same header hash, the event existed when the receipt was written and was committed afterwards, and nothing was removed. It is reported as a warning naming the race, the receipt's commit and the later commit; the record verifies. Any other mismatch remains an L14 error.

**Records from before v0.15.** Earlier tools wrote the receipt of each push into `anchors.jsonl` after the push, so a study checkout holds that line uncommitted. Before any command that writes, the tool commits it on its own (`claimset lab: commit anchor receipts (lab record v0.15)`), and the command's push sends it to the remote; `claimset lab anchor` does the same when there is nothing else to record. Receipts already committed are read as before.

**Without a study branch** (records outside the study layout, or before v0.7), the head is recorded by pushing the lab's logs to a dedicated anchor branch of a git remote, as follows. The branch is `claimset-lab/<study>` unless the configuration names another (`claimset-lab` for a record without a study). A relocation keeps the branch the record was anchored on: the tool writes it into the configuration. Anchor commits hold the logs at their paths when they were made; remote verification looks for the main log at its current path and, in commits made before a relocation, at the paths it had before. When such a record is adopted onto a study branch, its earlier receipts keep naming the anchor branch they were pushed to, and are checked there.

- **Configuration** is kept in `lab/config.json`: `{"anchor": {"service": "git_remote", "remote": "origin", "branch": "claimset-lab", "every": "plan"}}`. `every` is `plan` (default) or `event`.
- **What is committed:** `lab.jsonl`, `anchors.jsonl` and the files in `agents/`, at their paths in the repository. The commit is made on the anchor branch without touching the researcher's working tree or current branch. Run outputs are not committed; they are covered by the hashes in the log.
- **Order:** before committing, the tool checks that each log already on the remote branch (main and agent logs) is a prefix of the local one. If it is not, the local log has been rewritten since the last anchor; the tool refuses to anchor and reports it.
- **Pushing** never forces. A rejected push means the remote branch has changed in a way the tool did not make, and is reported.
- **Receipt:** `{remote, branch, commit}` recorded in the `anchors.jsonl` entry, which itself is committed with the next anchor. (This mode writes the receipt into `anchors.jsonl` in the working tree, which is the researcher's own checkout, not a study checkout.)
- **Protection of the branch:** to make the remote copy append-only, protect the branch on the host so that force-pushes and deletion are refused, and give the agent credentials that can push but cannot change branch protection. This turns detection into prevention.
- **Several working copies:** when agents work in their own clones (§3.5), each copy anchors to the same branch. The commit starts from the branch as it is on the remote: a local log that extends its anchored copy replaces it; a log that is behind (another copy has added to it) is kept as anchored; a log that differs is refused. `anchors.jsonl` on the branch holds the receipts of all copies. In the repository, `lab/.gitattributes` (written by `init`) marks `anchors.jsonl` with `merge=union`, so merging the copies' branches combines their receipts without conflicts; agent logs never conflict because each has its own file.
- **Checking:** `verify --remote` fetches the branch and checks that each local log agrees with its copy there (a local log that is behind is reported as a note, not an error), that each anchor's commit is on the branch its receipt names, and that it holds every anchored log as it was at that anchor.

## 8. Publication and redaction

By default the whole lab record is published with the claimset, as an artefact of kind `lab_record`, together with run outputs. Private sources (§5.20) are not: their events (metadata and hashes) are published, their files are not, and are never committed.

Individual events can be withheld. A withheld event keeps its header and replaces its body with:

`{"withheld": true, "reason": …, "comment": "…"}`, where `reason` is an availability reason (claimset §3.10).

Withholding hides the body only: the header, including the event's type, actor and corrections, is still checked against the rules. `merge` events are never withheld.

The header still carries the original `content_hash`, so the chain verifies and the platform's audit agents can check the original if it is provided to them. Run output files can be withheld the same way, keeping their hashes. Withheld events lower the verifiability of the claims that depend on them (claimset §5).

## 9. Export to a claimset

`export` produces a **draft** claimset submission from the record, and a **completion list**: the decisions and wording the draft still needs. The claimset-submit workflow then completes it (principal claims, wording, scope, guarantor and licence). The draft holds only what the record states. Where a field needs a judgement the record cannot supply, the draft uses the neutral default given below, or leaves the field out, and the completion list says so. The completion list is kept separate from the draft, because a claimset never contains instructions (claimset §1).

**Preconditions.** `export` refuses, and says why, unless:
- the record verifies (§6);
- no agent log is unmerged (§3.5);
- every decided plan has a verdict recorded for each hypothesis it names (§5.7.4).

Hypotheses without a final status are counted as open; they are not forced closed.

**Mapping.** The latest correction of each event is used throughout. A hypothesis's **final status** is its last `status` event, a correction standing in its original's place: correcting an earlier status does not make it the latest. An actor's events are those written under its ID or any correction of it. Withheld content (§8) is never guessed: a withheld event is left out of the draft, or the fields it would supply are, and the completion list names it.

| Lab record | Claimset |
|---|---|
| `init` | `paper.title` (the working title), `paper.origin: native` |
| `actor`, agent | contributor of type `agent` with its models (each model defined once in `models`, `accessed` the date of the earliest event by an agent using it), `name`, `system`, `record_source`, `run_period` (dates of its first and last events) and `tools: [code_execution]` if it ran commands. Position `author` if it wrote any event other than `actor` and `check`; otherwise `contributor`, with AI task `review` |
| `actor`, human or organisation | contributor with `name` and `orcid`, position `contributor`. Who is supervisor and guarantor is decided by people (completion list) |
| a verdict on a planned test (§5.7.4) | a claim per decided plan and hypothesis it names, in the order of the hypotheses and then the plans: `statement` and `about` from the hypothesis as it stood when the plan was written, `hypothesis: {lab_event, outcome}` (`lab_event` is that version's ID, `outcome` the verdict), defaults `type: descriptive`, `role: supporting`, `novelty: no_prior_found`, `provenance: author_stated` |
| `term` | a definition (`D1`, `D2`, … in record order, each correction chain once, its latest version) with `term`, `kind`, `definition`, `source` (as text: the completion list asks for it to become an entry of `references`) and `broader`. Term marks and term IDs in statements, `about` and `setup` are replaced by the definition IDs |
| `hypothesis` named by no decided plan, with final status `supported`, `not_supported` or `inconclusive` | an exploratory claim, the same way, from the latest version of the hypothesis and its final status |
| `hypothesis` otherwise | not a claim; counted in `paper.exploration` |
| the runs behind a verdict or final status | one evidence object per claim, from the runs the status names as evidence, directly or through their results: `lab_runs`, `tables`, `results`, `artefacts`, `execution` (`code_version` when the runs share one), `setup` (the setups of the plans the runs name, merged by role), default `profile: quant`, `design: other`, `availability: public`. **Which results:** when the status names results, those and the ones the test's quantities bind; when it names only runs, every result of those runs. **Tables** (claimset §3.4, *Tables are the standard form*): the results are grouped by the condition keys that identify the compared units (every key but those shared by all the group's results and those that set a variable apart: a nominal level, a subset, a statistic, a bound, a quantile); each group is a table whose rows are the distinct values of those keys, whose columns are the variables (the term's definition ID when a term of kind `variable` has that name) with the remaining keys as column conditions, and whose conditions are the shared keys. Results from several runs that would fill the same cell gain the row label `run`. Values are rounded to four significant digits, except those a test quantity binds. **Results** remain for a group of one observation and for observations a cell cannot hold (with `interval`, `uncertainty`, `n` or `approximate`, or a value that is not a number, text or a boolean). **The CSV:** every result of the record that is not withheld, at full precision, is written to `files/<paper>.results.csv` beside the claimset (`lab_result`, `run`, `quantity`, `conditions` as JSON, `value` as JSON), listed as an artefact of kind `data` with its columns and checksum, and named by each evidence object and table. The claim's support is `empirical_quant` pointing to the evidence and to the rows holding the results its status names and those its test's quantities bind, or to the evidence's tables and results when it names only runs. An extension may say otherwise for runs it concerns: runs that read codings give qualitative evidence (qualitative §2.7) |
| `plan` | `planning`, `planned_at`, `planned_at_anchor` and `test` on the claims of its verdicts (below). `test` is the plan's decision rule as written before the run: `framework`, `parameters`, `quantities` (each with `variable`), `supported`, `not_supported` (under `rules`, the hypothesis's own rules, framework and parameters, and only the quantities its rules use), `verdict` (`computed` when the rule has quantities, otherwise `judged`), `runs`, `lab_plan` (the plan's ID), `flags` (§5.7.5, the core's) and `addresses` (§5.7.7) |
| `deviation` | `planning: confirmatory_with_deviations`, and a limitation on the affected claim: `category: analytic_flexibility`, `impact: could_bias_magnitude`, `mitigation: none`, statement naming the run, the plan, what differed and why |
| `assumption` | assumption claim (without novelty) with `assumption_category` and `check`; `check_evidence` when it names a run or result used in an evidence object |
| `check` | a `checks` entry on the claim whose hypothesis, final status, or evidence run or result it targets. A check by an actor exported as an author is not exported (claimset rule 39) and is named in the completion list |
| `artefact` | artefact with `kind`, `uri` (the path or URI), `version` (the hash), `licence`, `description` |
| `source` | artefact of kind `data` after the `artefact` events, with `version` (the original's hash) and a `description` naming the source, its method, when it was retrieved and recorded, and, for content written by others, its terms. Private: `uri` is the source's address, `availability: {level: author_only, reason: legal, content_hash}` (the completion list asks to confirm the level and reason), and the file is not packaged. Public: `uri` is the file's path, `availability: public`. `licence` is the source's own licence, never the paper's. Content written by others with an address is also a cited work: a `references` entry `{target: uri, id_type: url, archive_uri}` that the artefact names in `reference` (the completion list asks for `archive_uri` when the source has none) |
| `extension` and blocks | `extensions` of the claimset declares each extension the draft uses, at the version the record declared; each event's block goes into the claimset object the event becomes (a term's to its definition, a result's to its observation, a source's or artefact's to its artefact, a hypothesis's to its claim). An extension's own events and flags are exported as its specification says |
| the whole record | artefact of kind `lab_record`: `uri` (for a repository on GitHub, GitLab, Codeberg or another Gitea or Forgejo host, or Bitbucket, whose last event is anchored by a git commit, the log in that commit; otherwise the repository URI from `init` followed by the log's path, and the completion list asks for a fixed URI, or the path alone), `version` (the hash of `lab.jsonl`), `availability: public`, `withheld_events`, and a `description` that names, from the record and the tool, the version of this specification the record was begun under (`init.lab_spec_version`), the date it was begun and the version of the tool that exported it; referenced from `paper.exploration.lab_record`; kept in desk mode, the description says so |

**Papers.** With paper events, export makes one paper at a time: `--paper NAME`, or the only paper. It exports the hypotheses of the paper's questions only. `paper.study` is the study's name; the exploration summary counts the paper's questions and hypotheses and its tests, and `covers` gives the questions it reports, the whole lab's counts (`lab`) and the other papers with their questions (`other_papers`), so readers see what the paper does not report (claimset §3.1.1). Runs are not tied to questions and are counted for the whole lab. Assumptions are exported from the whole record, and the completion list asks for those no claim of the paper rests on to be removed. The draft is written to `claimset/<study>/<paper>.claimset.json` unless another file is named.

**Planning label of an exported claim.** A claim from a verdict is `confirmatory`, or `confirmatory_with_deviations` if a deviation is recorded for the plan and one of its deciding runs. Its `planned_at` is the `at` of the plan version that decides the plan (the one its first deciding run names in `plan_version`), and `planned_at_anchor` is the service of the earliest anchor that covers that plan version and was made before the first deciding run started, or `none`. A claim from a hypothesis named by no decided plan is `exploratory`.

**Exploration summary.** `questions` counts questions, each correction chain once; `hypotheses` counts hypotheses by final status, each correction chain once, with `open` for those without one, and `from_exploration` those whose latest version has `origin: from_exploration`; `tests` counts verdicts, one per decided plan and hypothesis it names; `runs` counts all run events by label, and `failed` those whose exit status is not `success`.

**Defaults the completion list always names:** `paper.licence` and the guarantor (asked of the responsible human), `paper.contributions` and principal roles, `paper.autonomy` (exported as `unknown`), claim types, statements of negative results (to say plainly what was not found), scopes, evidence profiles and descriptions, links from claims to assumptions, an `about` for each claim whose hypothesis has none and a `setup` for each evidence object whose plans have none, and the **definitions**, exported from the record's terms: missing ones are added, and the definition of each method or model of the study's own that a planned test compares gives every equation that defines it, as the code of the runs implements it, and is checked against that code (the runs' code versions are in each evidence's `execution`); a variant gives its own equations in full (claimset §3.2, warning 26). The supervisor's approval of the completed claimset covers its definitions as well as its claims.

**Negative results** are exported as claims in the same claimset: the statement says what was not found, under what scope, and `hypothesis.outcome` is `not_supported`.

## 10. Tool commands (informative)

The reference tool is part of the `claimset` package. Indicative commands:

| Command | Effect |
|---|---|
| `claimset setup --study NAME --title …` (or `claimset lab init --study NAME --title …`) | creates the study branch `claimset/NAME` with its lab, checks it out at `claimset/NAME/`, and adds the ignore lines (§3, §3.3) |
| `claimset setup` | in a fresh clone: checks out every study branch of the remote |
| `claimset setup --study NAME --adopt --by AC2` | moves a study committed on the code branch onto its own branch (§3.4) |
| `claimset setup --study NAME --title … --desk` | creates a study kept in desk mode, without a remote (§7.1); `--remote origin` later starts anchoring it |
| `claimset setup --ci` | writes the GitHub check that keeps study records out of pull requests (§4.5) |
| `claimset studies` | the studies of the repository: events, open hypotheses, pending plans, commits not pushed |
| `claimset lab paper NAME --questions Q1 Q2` | records a paper and the questions it reports (§5.17) |
| `claimset lab move [--to DIR] [--study NAME] [--also OLD=NEW …]` | moves the lab (default `claimset/<study>/lab`) and other files, and records the relocation (§3.4) |
| `claimset lab actor --kind agent\|human\|organisation --name … [--orcid …] [--model …] [--words "…" [--given-at T] [--channel …]]` | registers an actor: an agent itself, a person themselves, or an agent a person with their words (§5.2) |
| `claimset lab question` / `hypothesis` / `assume` / `note` | writes the corresponding event |
| `claimset lab term MASE --kind variable --definition "…" [--source …]` | records a term (§5.19) |
| `claimset lab extension NAME [--version V]` | declares an extension of the claimset format (§5.22); commands that need one declare it themselves |
| `claimset lab term "arrange for ensemble" --kind concept --definition "…" --include "…" [--exclude "…"] [--example "…"] [--broader TM3]` | records a code of the codebook (qualitative extension §2.1) |
| `claimset lab code SR1 --codes TM4 [TM7 …] [--excerpt "the words"] [--unit post] [--pass 1] [--memo "…"]` | records a coding (qualitative extension §2.2); `--supersedes CD3` recodes, `--supersedes CD3 --withdraw` withdraws; `claimset lab code SR6 --none` records a source read with no code applying |
| `claimset lab count --plan PL1` | the deciding run of a plan decided by counts (qualitative extension §2.4); `claimset lab count --codes TM4 [--distinct-by year] [--where pass=1]` counts for exploration |
| `claimset lab hypothesis "[[TM1:M]] has lower [[TM3:MASE]] than [[TM2:N]]." --question Q1 --subject TM1 --relation "has lower error than" --object TM3 --comparator TM2 --condition TM4` | a hypothesis with its atom (§5.4) |
| `claimset lab status H3 supported --evidence RS12 RS13 --why "…" [--plan PL2] [--rule supported]` | status change, with L6, L7 and L17 checked |
| `claimset lab verdict PL2` | the verdict a decided plan's rule gives, with the value bound to each quantity (§5.7.3) |
| `claimset lab plan --kind confirmatory --hypotheses H1 --design … --analysis … --framework frequentist --parameter alpha=0.05 … --quantity p="…" --supported "p < alpha" --not-supported "…" [--runs N] [--decision FILE] [--new-data "…"] [--exclusion name="rule"] [--studied TM1 TM2 --cases TM4 --procedure TM5 --lens TM3]` | writes a plan with its decision rule (§5.7.1) and anchors it; warns as in §5.7.6 |
| `claimset lab plan --kind confirmatory --hypotheses H1 H2 … --quantity s="…" --quantity n="…" --supported "H1: s < threshold" --not-supported "H1: s >= threshold" --supported "H2: n > ui" --not-supported "H2: n <= ui"` | a plan whose hypotheses have rules of their own (§5.7.1): a rule prefixed with a hypothesis ID is that hypothesis's; repeat for each |
| `claimset lab plan --kind confirmatory --hypotheses H9 --comment doi:…@v1/RV1.c6 --review RV1.json --analysis "…"` | writes a plan that runs a review comment's resolution test: the decision rule is the comment's current test and `addresses` names the comment (§5.7.7); `--design` defaults to the test's procedure |
| `claimset lab approve PL2 --by AC1` | approval by the human, who runs the command |
| `claimset lab approve PL2 --by AC2 --on-behalf-of AC1 --words "Yes, go ahead with PL2." [--given-at …] [--channel conversation]` | approval recorded by an agent on the human's behalf, with their words (§5.8) |
| `claimset lab run [--plan PL2] [--exploratory] [--param k=v] [--seed N] [--data FILE] -- <command>` | runs the command, records `run` and `result` events (§5.9.1) |
| `claimset lab capture FILE --uri URL [--retrieved-at T] [--method saved\|browser] [--title …] [--venue …] [--published-at D] [--author HANDLE [--author-as-given]] [--private\|--public] [--own] [--licence SPDX \| --terms "…"] [--archive-uri URL] [--text FILE]` | records a source the researcher saved (§5.20); `--fetch --uri URL` instead has the tool fetch a public page; `claimset lab capture --supersedes SR1 --archive-uri URL` (no file) corrects what was recorded about it |
| `claimset lab quote SR1 "the words"` | checks that the words are in the source's text and prints their locator (§5.20) |
| `claimset lab decide` / `deviate` / `check` / `artefact` | writes the corresponding event |
| `claimset lab summary` | current state: open questions, hypotheses by status, pending plans, recent decisions; for an actor with an open agent log, what that agent sees |
| `claimset lab show H3` | one event, with corrections resolved |
| `claimset lab list [--type hypothesis] [--last 20]` | events in order, one line each |
| any writing command `--supersedes ID` | writes a correction of `ID` (§4.3) |
| `claimset lab start --by AC2` | starts an agent log for AC2: its events go there until merged (§3.5) |
| `claimset lab merge [AC2 AC3]` | merges agent logs (default: all unmerged ones) into the main log (§3.5) |
| `claimset lab verify` | checks the main and agent chains, the rules and the run files |
| `claimset lab anchor [--git] [--remote origin] [--branch claimset-lab] [--every plan\|event]` | anchors the head of the chain; the first use with `--git` records the configuration (§7.1) |
| `claimset lab verify --remote` | also checks the log against the anchor branch on the remote |
| `claimset lab guard` | the guard for agent harnesses: reads a proposed tool call as JSON on standard input and exits with code 2 if it would modify the lab directory (§4.5); `--print-config [--agent NAME]` prints the hook configuration for a harness (Claude Code by default); `claimset init --guard` writes it (§4.6) |
| `claimset lab protect` | sets the operating system's append-only attribute on the log files, or explains how an administrator can (§4.5) |
| `claimset lab publish --withhold RN7 --reason personal_data --comment "…"` | prepares the publishable record |
| `claimset lab export [--paper NAME] [-o FILE]` | writes the draft claimset submission of one paper (default `claimset/<study>/<paper>.claimset.json`) and prints the completion list (§9) |
| `claimset tag claimset/<study>/<paper>.claimset.json` | marks the committed version of a paper with the git tag `claimset/<study>/<paper>/v<n>` (claimset §2.4) |

## 11. Open questions

1. Anchoring service: platform receipts, RFC 3161 providers, or both.
2. Storage of large run outputs (content-addressed storage, external archives).
3. Event types for laboratory and field research. Terms (§5.19) record a study's setup and sources (§5.20) what it reads; the qualitative extension records codes, codings and counts. Still open, as extensions: data-collection sessions (interviews, observations, samples, instrument runs, survey waves), and blind second coding enforced by the tool (an agent log keeps codings apart, but does not hide others').
4. Very long explorations (thousands of runs): summarised export without losing the counts.
5. Whether abandoned hypotheses with substantial evidence should be exportable as claims.
