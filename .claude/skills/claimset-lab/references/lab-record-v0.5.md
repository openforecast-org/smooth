# Claimset Lab Record — Specification v0.5 (draft)

Status: working draft · Date: 2026-10-16 · Supersedes v0.4 · Companion to the claimset specification (`spec/claimset-v0.18.md`)

## Changes since v0.4

Strict where it matters, light everywhere else. The path from a plan to a claim is strict: every analysis choice that could be tuned after seeing the data is fixed in the plan, each plan gets one attempt, and every planned test that ran is reported with the verdict its rule gives. Exploration is recorded where its results inform decisions, without further restrictions.

- A confirmatory plan carries a **decision rule** instead of a free-text criterion: the inferential framework, its parameters (significance level, test, sidedness, correction for multiple tests, sample size and stopping rule, equivalence bounds, prior, thresholds), and rules for both conclusive outcomes, `supported` and `not_supported`; anything else is `inconclusive`. Rules may be expressions over named results, in which case the tool computes the verdict, or text, in which case the writer states which rule was met (§5.7.1–5.7.3).
- A plan states how many runs decide it. Every confirmatory run under it counts, whatever its exit status, and further confirmatory runs are refused (§5.7.2). Each run records the plan version in force when it was made (`plan_version`, §5.9).
- A verdict names the plan and agrees with its rule. Every planned test that ran must have its verdict recorded before export, and each is exported as its own claim, with the hypothesis as it stood when the plan was written; a hypothesis may be tested again under a new plan (§5.5, §5.7.4, §9).
- Exploration need not be recorded step by step (§2); hypotheses left open are counted as open rather than closed by force (§9).
- Confirmatory runs in an agent log are made under plans written in that log (§3.1).
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
- Agent logs specified precisely: how one is started, what an agent sees and may write, the order and layout of a merge, and when a merge is refused (§3.1, §5.16); rule L10 stated precisely; anchors cover agent logs (§7).

## Changes in v0.3 (since v0.2)

Clarifications needed to implement the reference tool. No change to the meaning of v0.2 records.

- Canonical JSON, hash format and header hash defined exactly (§4.1).
- `by` of the `init` event is `null`; an `actor` event may name its own ID in `by` (self-registration) (§4.2).
- Event numbering within each prefix (§4.2).
- Corrections: what `supersedes` means and how references to a corrected event resolve (§4.3).
- Writers take a lock on the log (§4.4).
- New rules L11 (corrections) and L12 (event IDs). New informative commands `show` and `list` (§10).

## Changes in v0.2 (since v0.1)

- Several agents working at once each write their own log, merged into the main log when they finish (§3.1). Event IDs from agent logs carry the actor prefix (`AC2.RN4`); headers gain `origin`.

## 1. Purpose

A lab record is the running record of a piece of research as it happens: the questions asked, the hypotheses proposed and revised, the plans made, every run executed, every result produced, and every decision taken along the way. Agents (and humans) write it while doing the research. The claimset submitted for publication is distilled from it, and the lab record is published alongside the claimset.

Research is iterative: ideas are formed, tested, revised and abandoned. Conventional papers keep only the surviving story. The lab record keeps everything, so that:

- nothing produced during the research is lost or silently overwritten;
- readers can see how many ideas were tried before the published ones, and which results were planned and which were found by exploring;
- negative results are captured as a by-product of doing the work;
- an agent can resume the work after a context reset or hand it to another agent.

Version 0.1 targets computational research (simulation, analysis of existing data, method development, proofs). Laboratory and qualitative research fit the same model, with humans recording data-collection events; dedicated event types for them are an open question.

## 2. Principles

1. **Append only.** Events are never edited or deleted. A correction is a new event that names the event it supersedes.
2. **Recorded by the tool.** Events are written by the lab tool, which assigns sequence numbers, times and hashes. Agents do not write the log by hand.
3. **Results come from code.** Numbers enter the record only from files written by the code that produced them, never typed by an agent.
4. **Plan before run, or say it was exploratory.** A run is confirmatory only if a plan written earlier covers it; every other run is labelled exploratory. Exploration is legitimate; it is just labelled. It is recorded where its results inform decisions (a run whose output is looked at to choose a direction, a setting or a hypothesis), not step by step, so the record's account of exploration is a lower bound. The rules are strict only on the path from a plan to a claim.
5. **Every outcome is recorded.** Failed runs, crashed runs, null results, refuted and abandoned hypotheses are all events.
6. **Exploration cannot confirm itself.** Evidence that suggested a hypothesis cannot also be the evidence that supports it.
7. **Every change of mind has a reason.** Status changes, revisions and deviations from a plan carry a rationale and point at what prompted them.
8. **Resumable.** The current state of the research can always be rebuilt from the log, and the tool summarises it for whoever picks up the work.

## 3. Layout

A lab lives in a directory, normally inside the project's version-controlled repository:

```
lab/
  lab.jsonl              the main event log, one event per line
  agents/<actor>.jsonl   per-agent logs while agents work concurrently (§3.1)
  anchors.jsonl          timestamp anchors of the log (§7)
  runs/<run id>/         per-run outputs: stdout, stderr, results.json, other files
```

`lab.jsonl` is UTF-8 JSON Lines. Every line is one event.

### 3.1 Several agents

When several agents work on one lab at the same time, each can write its own log instead of the main log, for example in its own working copy of the repository. The agent logs are merged into the main log when the agents finish.

1. **Starting.** Every agent is registered by an `actor` event in the main log. `start` then creates the empty file `lab/agents/<actor id>.jsonl`. From then on, every event written by that actor goes to its agent log, until the log is merged.
2. **Chain.** An agent log has its own `seq` (from 1) and its own hash chain. Its first event's `prev` is the header hash of the latest main-log event when that first event was written. That main-log event is the log's **base**: it records which state of the lab the agent worked from.
3. **IDs.** Event IDs in an agent log are prefixed by the actor ID: `AC2.H1`, `AC2.RN4`, numbered within the log. IDs therefore never collide between agents. Run output directories use the same IDs (`runs/AC2.RN4/`).
4. **What an agent sees.** An agent's events may refer to main-log events up to its base and to earlier events in its own log, and to nothing else: not to later main-log events, and not to events in other agent logs.
5. **What an agent may write.** Only its own actor writes an agent log. Its confirmatory runs are made under plans written in the same log, so that two logs never race to run the same planned test. `init`, `actor` and `merge` events are written only to the main log. A correction (`supersedes`) in an agent log corrects only an event of the same agent log; other events are corrected in the main log after merging. Human approvals are therefore given in the main log, after the log they approve is merged.
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
| `origin` | For events merged from an agent log: `{log, seq, header_hash}`; otherwise `null` (§3.1) |
| `body` | Type-specific content |

The headers form a hash chain. Because each header commits to its body only through `content_hash`, a body can be withheld at publication (§8) without breaking the chain: a reader can still verify the order, timing and integrity of every event.

### 4.1 Canonical JSON and hashes

- **Canonical JSON** of a value is its JSON serialisation with object keys sorted, no whitespace outside strings (separators `,` and `:`), non-ASCII characters written as UTF-8 rather than escaped, encoded as UTF-8 bytes. (In Python: `json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`.)
- Every hash is written `sha256:` followed by 64 lowercase hexadecimal digits.
- `content_hash` is the SHA-256 of the canonical JSON of `body`.
- The **header hash** of an event is the SHA-256 of the canonical JSON of an object holding all header fields: `seq`, `id`, `type`, `at`, `by`, `prev`, `content_hash`, `supersedes`, `origin`. The `prev` of the next event is this value.
- For `seq` 1, `prev` is the SHA-256 of the empty byte string.

### 4.2 Actors and IDs

- `by` is an actor ID. Two exceptions: the `init` event has `by: null`, and an `actor` event may name its own ID (an actor registering itself).
- An event ID is its type prefix (§5) followed by a positive integer: `H1`, `H2`, `RN17`. Within each prefix, numbers are allocated in order: the next number is one more than the highest used so far in the log. Events from agent logs carry the actor prefix (`AC2.H1`) and are numbered within their agent log, independently of the main log (§3.1).

### 4.3 Corrections

Events are never edited. To correct one, write a new event of the **same type** whose header names the old one in `supersedes`. The new event has its own ID and a complete new body. An event can be superseded at most once, so corrections form a chain.

The first event of a chain is the **root**. Other events keep referring to the root's ID (or any ID in the chain), and every reference resolves to the latest event of the chain. Status changes, checks and evidence that point at a hypothesis therefore still apply after its statement is corrected. A correction is for errors in what was recorded (a typo, a wrong reference); a change of mind about a hypothesis is a `status` event with `to: revised` and a new hypothesis.

### 4.4 Writing

Only the tool writes the log. A writer takes an exclusive lock, the file `lab/.lock` created atomically, appends one complete line and releases the lock. A lock older than 60 seconds is treated as stale. Several agents working at the same time write separate logs (§3.1) rather than contending for one lock.

### 4.5 Protecting the log

An agent that can run commands as the researcher can, in principle, do anything the researcher can do to a file, including rewriting the whole log and recomputing every hash. Protection therefore comes in levels:

| Level | How | Effect |
|---|---|---|
| Tamper-evident | the hash chain (§4.1) and `verify` | edits, deletions and reordering are detected, unless the whole chain is rewritten |
| Guarded | the tool's guard, installed as a hook in the agent's harness, refuses tool calls that would modify the lab directory other than through `claimset lab` | accidental or casual modification is prevented; a determined agent can work around it |
| Append-only file | the operating system's append-only attribute on the log files (on Linux, `chattr +a`, set by an administrator) | the operating system refuses any change except appending, as long as the agent cannot act as an administrator |
| Anchored | the head of the chain is regularly recorded outside the agent's reach (§7) | any rewriting before an anchor is detected, including a full recomputation of the chain |
| Remote append-only | the authoritative copy is held by a service that accepts only appends (the Claimset platform; or a git branch protected against force-pushes and deletion, pushed to with credentials that cannot change the protection) | rewriting is prevented, not just detected |

Locally, the log is tamper-evident. With anchoring, tampering is always detected. With a remote append-only copy, it is prevented. A lab that matters should use at least anchoring.

## 5. Event types

### 5.1 `init` (one per lab, `seq` 1, ID `INIT`)

`lab_spec_version` records the version of this specification the record follows. Rules introduced in a later version do not apply to records begun under an earlier one.

`{lab_spec_version, working_title, repository, started_at}`. Its `by` is `null`.

### 5.2 `actor` (prefix `AC`)

Registers a person or agent who will write events. Mirrors claimset contributors.

`{kind: agent | human | organisation, name, orcid, models: [{provider, name, version, identifier}], record_source: harness | user_reported | self_reported}`

### 5.3 `question` (prefix `Q`)

`{text}`: a research question or aim.

### 5.4 `hypothesis` (prefix `H`)

`{statement, question, origin, prompted_by}`

- `statement`: follows the claimset statement rules (one assertion, self-contained, scope stated, no rhetoric).
- `question`: the `Q` it addresses.
- `origin`: `a_priori` (proposed before looking at results) or `from_exploration` (suggested by results).
- `prompted_by`: required for `from_exploration`: the result, run or note IDs that suggested it.

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

`{kind, hypotheses, design, analysis, decision, runs, data, holdout, code_version, dirty}`

- `kind`: `confirmatory` or `exploratory`. Required.
- `hypotheses`: hypothesis IDs the plan tests. Required (at least one) for `confirmatory`.
- `design`: what will be run, on what, with which settings. Required.
- `analysis`: what will be computed from the outputs. Required.
- `decision`: the decision rule (§5.7.1). Required for `confirmatory`.
- `runs`: how many successful confirmatory runs decide the plan (§5.7.2). A positive integer; default 1.
- `data`: optional list of `{description, hash}` for data that exists when the plan is written.
- `holdout`: optional list of `{path, hash}`: data kept for this test only (§5.7.5).
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
    "p": {"quantity": "p-value, paired t-test of MAE difference naive minus SES", "conditions": {"ratio": 1}},
    "d": {"quantity": "mean MAE difference, naive minus SES", "conditions": {"ratio": 1}},
    "d_upper": {"quantity": "mean MAE difference, naive minus SES", "conditions": {"ratio": 1},
                "field": "interval.upper"}
  },
  "supported": "p < alpha and d > 0",
  "not_supported": "d_upper < 0.01"
}
```

- `framework`: `frequentist` (a null-hypothesis significance test) · `equivalence` (an equivalence test against stated bounds) · `bayesian` · `deterministic` (a measured quantity compared with fixed thresholds, without sampling uncertainty, e.g. a benchmark or a simulation with fixed seeds) · `formal` (proof or counterexample) · `qualitative` (a stated pattern in qualitative data).
- `parameters`: the parameters the framework requires (table below), and any others the analysis fixes in advance. Values are numbers or text. Numeric parameters can be used in rules by name.
- `quantities`: optional. When given, the rules are expressions over them and the tool computes the verdict (§5.7.3); when absent, the rules are text and the writer states which rule the evidence met. Computed verdicts are stronger, and recommended whenever the outcome can be computed from recorded results. Each name (letters, digits and `_`, starting with a letter) binds to exactly one result of the deciding runs: the result whose `quantity` equals the given text, whose `conditions` include every given key with the given value, read from `field`: `value` (default), `interval.lower` or `interval.upper`. When the plan has more than one deciding run, each quantity also names the run it comes from, `run` (1 for the first deciding run, and so on), and is bound among that run's results only.
- `supported` and `not_supported`: the rules for the two conclusive outcomes. If neither rule holds, the outcome is `inconclusive`. With quantities each rule is an expression (§5.7.3); otherwise it is self-contained text stating what observation decides (a p-value below `alpha` with the effect in the predicted direction; a confidence interval within the equivalence bounds; a proof of the statement or a counterexample; the pattern in the data that would confirm or disconfirm the hypothesis).

| Framework | Required parameters |
|---|---|
| `frequentist` | `alpha` (significance level), `test`, `sides` (`two_sided` · `greater` · `less`), `correction` (for multiple tests within the plan: `none` · `bonferroni` · `holm` · `fdr_bh` or text), `sample` (planned sample size and stopping rule) |
| `equivalence` | `alpha`, `test`, `lower_bound` and `upper_bound` (the equivalence margins: the smallest effect of interest), `sample` |
| `bayesian` | `prior`, `threshold` (e.g. a Bayes factor or posterior probability that decides), `sample` |
| `deterministic`, `formal`, `qualitative` | none |

**Why both rules.** Failing to reject a null hypothesis is not evidence against the hypothesis. A negative result is conclusive only if the plan said in advance what would refute the hypothesis: an equivalence bound, a significant effect in the opposite direction, a posterior threshold, a counterexample. Without that, the outcome is `inconclusive`.

#### 5.7.2 Deciding runs

The **deciding runs** of a confirmatory plan are its first `runs` confirmatory runs, whatever their exit status. Once a plan has its deciding runs it is **decided**, and further confirmatory runs under it are refused (L16): repeating a test until it comes out favourably is exactly the flexibility the plan removes. A run that fails still counts: a script could otherwise write its results and then fail on purpose. If a needed result is missing because a run failed, the verdict is `inconclusive`. Check that scripts work with exploratory runs on other data before the confirmatory run.

#### 5.7.3 Rule expressions

A rule expression is at most 1,000 characters long and uses finite numbers, names (quantities and numeric parameters), `+ - * /`, unary minus, `abs()`, comparisons (`< <= > >= == !=`, which may be chained: `0 < d < 1`), `and`, `or`, `not` and parentheses. Nothing else is allowed. Every name must be defined in `quantities` or as a numeric parameter (L15).

The tool evaluates both rules on the results of the deciding runs. The verdict is `supported` if only the `supported` rule holds, `not_supported` if only the `not_supported` rule holds, and `inconclusive` if neither holds. If both hold, the rules overlap and the verdict is `inconclusive`. If a quantity cannot be bound (no result matches, or several do), no conclusive verdict can be computed, and only `inconclusive` can be recorded.

#### 5.7.4 Reporting

Every planned test that ran is reported. Once a plan is decided, a verdict (§5.5) is recorded for each hypothesis it names; export refuses a record in which one is missing (§9). Statuses recorded before or after the verdict do not replace it: each verdict becomes a claim of its own.

A hypothesis may be tested again under a new plan (a replication, a larger sample); each test gives its own verdict and claim, so a favourable test cannot hide an unfavourable one. A claim from a planned test states the hypothesis as it stood when the plan was written, so correcting the hypothesis afterwards does not move the verdict to a different statement.

#### 5.7.5 Flags

Two things weaken a planned test without making it invalid. The tool does not refuse them; it reports them as flags of the decided plan (in `verify`, in `verdict`, and in the exported claim's `test.flags`):
- `code_changed_after_plan`: a deciding run used a different commit from the one recorded with the plan, or the code had uncommitted changes when the plan was written or a deciding run was made. The analysis was not fixed when the plan was. Commit the analysis code before writing the plan to avoid it.
- `holdout_used_elsewhere`: a run other than the plan's deciding runs declared a held-out file of the plan (by hash) as input data. The data meant to test the hypothesis was also used to explore it. Only inputs declared to the tool (`run --data`) can be detected.

### 5.8 `approval` (prefix `AP`)

`{target, comment}`: a human approves a plan (or another event) before it is acted on. The `by` of an approval is an actor of kind `human`. Optional; lets a supervisor sign off confirmatory plans.

### 5.9 `run` (prefix `RN`), written by the tool

`{plan, plan_version, label, command, code_version, dirty, parameters, seeds, data, environment, started_at, ended_at, exit_status, exit_code, outputs}`

- `plan`: plan ID or `null`.
- `plan_version`: the ID of the version of that plan in force for the run's writer when the run was made (the latest correction it could see). The version named by the first deciding run decides the plan (§5.7.2); this holds also for runs merged from agent logs, whose writer may not have seen corrections made meanwhile in the main log.
- `label`: `confirmatory` (only if rule L5 holds) or `exploratory`.
- `command`: the command as a list of arguments.
- `code_version`: commit hash of the working directory's repository, or `null` if it is not a git repository; `dirty`: true if it had uncommitted changes outside the lab directory.
- `parameters`: optional object of named settings given to the tool; `seeds`: optional list of random seeds given to the tool.
- `data`: list of `{path, hash}` for input files declared to the tool.
- `environment`: `{software: {python, platform}, hardware: {machine, cpus}}`, as recorded by the tool.
- `started_at`, `ended_at`: UTC times as in `at`.
- `exit_status`: `success` (exit code 0) · `failed` (non-zero exit code) · `crashed` (killed by a signal, or the command could not be started) · `cancelled` (interrupted by the user).
- `exit_code`: the process's exit code, or `null`.
- `outputs`: list of `{path, hash}` for every file in the run's directory after the run, with paths relative to that directory.

#### 5.9.1 How the tool runs a command

1. It checks rule L5 before starting. A run is confirmatory only when it names a confirmatory plan; otherwise, or when asked, it is exploratory.
2. It creates a pending run directory under `runs/` and starts the command with two environment variables: `CLAIMSET_RUN_DIR` (the directory, for any output files) and `CLAIMSET_RESULTS` (the path of `results.json` inside it).
3. It writes the command's standard output and standard error to `stdout.txt` and `stderr.txt` in that directory, and shows them as the command runs.
4. When the command ends, however it ends, it hashes every file in the directory, writes the `run` event, and renames the directory to `runs/<run id>/`.
5. If `results.json` exists, it writes one `result` event per entry (§5.10.1).
6. It exits with the command's exit code.

### 5.10 `result` (prefix `RS`), written by the tool

`{run, quantity, value, unit, interval, conditions, source}`

Same fields as claimset results (`quantity`, `value`, `unit`, `interval`, `conditions`), plus `run` and `source`: the output file the value was read from, whose path is in the run's `outputs`.

#### 5.10.1 Results file

A script reports results by writing a JSON array to the file named by `CLAIMSET_RESULTS`:

```json
[{"quantity": "largest KL divergence per observation, mixture vs exact", "value": 0.00115,
  "conditions": {"distribution": "Gamma", "form": "conventional"}},
 {"quantity": "share of failed fits", "value": 0.18, "unit": "proportion",
  "interval": {"level": 0.95, "lower": 0.15, "upper": 0.21}}]
```

Each entry needs `quantity` (text) and `value` (a finite number: not NaN or infinite); `unit`, `interval` and `conditions` are optional. Values are written exactly as computed. An entry that is not valid produces no event and is reported by the tool.

### 5.11 `decision` (prefix `DC`)

`{text, prompted_by, affects}`: a choice that changes the direction of the work (e.g. switching metric, dropping a model class, changing the data). This is where the "garden of forking paths" is recorded.

### 5.12 `deviation` (prefix `DV`)

`{plan, run, text, reason}`: a run that departed from its plan, and why. `plan` and `run` are required, `text` says what differed, and `reason` (required) says why.

### 5.13 `check` (prefix `CK`)

`{target, outcome, extent, notes, resolution}`: an independent check of a hypothesis, result or derivation; same semantics as claimset checks (claimset §5). The checking actor must not have written the target.

### 5.14 `artefact` (prefix `AR`)

`{kind, path_or_uri, hash, licence, description}`: registers a file that is not a run output (data set, figure, manuscript, proof file).

### 5.15 `note` (prefix `NT`)

`{text, refers_to}`: an observation that is not yet a hypothesis, decision or result. Notes are kept but not exported as claims.

### 5.16 `merge` (prefix `MG`), written by the tool

`{agents, events_merged}`: records that the agent logs of the listed actors were merged into the main log (§3.1). The merge event is followed immediately by exactly `events_merged` events, each with an `origin` in one of the listed logs: every listed log whole, from its `seq` 1, in the order of §3.1. A merge event is never withheld (§8).

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
L10. Agent logs follow L1–L9 and L11–L12 on their own, starting from their base, and the restrictions of §3.1 (who writes them, what they may refer to and correct, which event types they hold). A merged event equals the event its `origin` names, whose header hash it records; an agent log is merged once, and nothing is added to it afterwards. Only merged events carry an actor prefix in the main log.
L11. A `supersedes` reference names an earlier event of the same type that has not already been superseded.
L12. Every `id` has the prefix of its `type` and is unique in the log; every reference in a body names an event with a lower `seq`.
L13. The files of a run directory match the hashes in the run's `outputs`. Run directories are never changed after the run.
L14. Every entry in `anchors.jsonl` names a `seq` present in the log, and its `head_hash` equals the header hash of the event at that `seq`; the same holds for each agent log the entry names. A mismatch means the log was rewritten after it was anchored.
L15. A confirmatory plan has a complete decision rule (§5.7.1): a framework, the parameters it requires, and both rules; when quantities are given, every quantity is well formed (with `run` when there are several deciding runs) and every rule is a valid expression whose names are defined (§5.7.3); `runs` is a positive integer.
L16. A confirmatory run is not made under a plan that already has its deciding runs (§5.7.2), or, in an agent log, under a plan written outside that log (§3.1).
L17. A status that names a plan, or that cites a decided plan's deciding runs or their results for a hypothesis the plan names, is a verdict: the plan is decided and names the hypothesis, the deciding runs or their results are cited, and the outcome is the one the plan's rule gives (§5.5).

The tool enforces L1–L5 and L8–L17 mechanically, and refuses L6 and L7 violations. L15–L17 apply to records begun under v0.5 or later, and to records whose `init` is withheld. Where a check depends on withheld content (a withheld run, result or plan), `verify` does not report it as a violation; withholding lowers verifiability instead (§8).

## 7. Timestamp anchoring

A timestamp written by the tool proves order, not time. To make "this plan existed before this run" verifiable to others, the tool periodically anchors the head of the chain (the latest `seq` and its header hash) with an external service and records the receipt in `anchors.jsonl`:

`{seq, head_hash, service, receipt, at, agents}`

`agents` (omitted when there are none) records the head of each non-empty agent log in the same way: `{"AC2": {seq, head_hash}}`. An event is covered by an anchor if the anchor's `seq` is at least the event's main-log `seq`, or, for an event written to an agent log, if the anchor's entry for that log has a `seq` at least the event's `seq` there.

Services, in decreasing order of strength:
- `platform`: the Claimset platform records the hash and returns a signed receipt.
- `rfc3161`: a trusted timestamp authority under RFC 3161.
- `git_remote`: a commit containing the log pushed to a remote host (weaker: the host's clock and history are trusted).
- `none`: no anchoring.

The anchor level of a plan becomes `planned_at_anchor` in the exported claimset.

**Automatic anchoring.** When anchoring is configured, the tool anchors after every `plan` event, and before a confirmatory run it makes sure the run's plan is covered by an anchor. If that anchor cannot be made, the run is refused as confirmatory (rule L5); it can still be run as exploratory. The tool can also be configured to anchor after every event.

### 7.1 Git anchoring

With `git_remote` anchoring, the head is recorded by pushing the lab's logs to a dedicated branch of a git remote.

- **Configuration** is kept in `lab/config.json`: `{"anchor": {"service": "git_remote", "remote": "origin", "branch": "claimset-lab", "every": "plan"}}`. `every` is `plan` (default) or `event`.
- **What is committed:** `lab.jsonl`, `anchors.jsonl` and the files in `agents/`, at their paths in the repository. The commit is made on the anchor branch without touching the researcher's working tree or current branch. Run outputs are not committed; they are covered by the hashes in the log.
- **Order:** before committing, the tool checks that each log already on the remote branch (main and agent logs) is a prefix of the local one. If it is not, the local log has been rewritten since the last anchor; the tool refuses to anchor and reports it.
- **Pushing** never forces. A rejected push means the remote branch has changed in a way the tool did not make, and is reported.
- **Receipt:** `{remote, branch, commit}` recorded in the `anchors.jsonl` entry, which itself is committed with the next anchor.
- **Protection of the branch:** to make the remote copy append-only, protect the branch on the host so that force-pushes and deletion are refused, and give the agent credentials that can push but cannot change branch protection. This turns detection into prevention.
- **Several working copies:** when agents work in their own clones (§3.1), each copy anchors to the same branch. The commit starts from the branch as it is on the remote: a local log that extends its anchored copy replaces it; a log that is behind (another copy has added to it) is kept as anchored; a log that differs is refused. `anchors.jsonl` on the branch holds the receipts of all copies. In the repository, `lab/.gitattributes` (written by `init`) marks `anchors.jsonl` with `merge=union`, so merging the copies' branches combines their receipts without conflicts; agent logs never conflict because each has its own file.
- **Checking:** `verify --remote` fetches the branch and checks that each local log agrees with its copy there (a local log that is behind is reported as a note, not an error), that each anchor's commit is on the branch, and that it holds every anchored log as it was at that anchor.

## 8. Publication and redaction

By default the whole lab record is published with the claimset, as an artefact of kind `lab_record`, together with run outputs.

Individual events can be withheld. A withheld event keeps its header and replaces its body with:

`{"withheld": true, "reason": personal_data | commercial | legal | ethics | security | other, "comment": "…"}`

Withholding hides the body only: the header, including the event's type, actor and corrections, is still checked against the rules. `merge` events are never withheld.

The header still carries the original `content_hash`, so the chain verifies and the platform's audit agents can check the original if it is provided to them. Run output files can be withheld the same way, keeping their hashes. Withheld events lower the verifiability of the claims that depend on them (claimset §5).

## 9. Export to a claimset

`export` produces a **draft** claimset submission from the record, and a **completion list**: the decisions and wording the draft still needs. The claimset-submit workflow then completes it (principal claims, wording, scope, guarantor and licence). The draft holds only what the record states. Where a field needs a judgement the record cannot supply, the draft uses the neutral default given below, or leaves the field out, and the completion list says so. The completion list is kept separate from the draft, because a claimset never contains instructions (claimset §1).

**Preconditions.** `export` refuses, and says why, unless:
- the record verifies (§6);
- no agent log is unmerged (§3.1);
- every decided plan has a verdict recorded for each hypothesis it names (§5.7.4).

Hypotheses without a final status are counted as open; they are not forced closed.

**Mapping.** The latest correction of each event is used throughout. A hypothesis's **final status** is its last `status` event, a correction standing in its original's place: correcting an earlier status does not make it the latest. An actor's events are those written under its ID or any correction of it. Withheld content (§8) is never guessed: a withheld event is left out of the draft, or the fields it would supply are, and the completion list names it.

| Lab record | Claimset |
|---|---|
| `init` | `paper.title` (the working title), `paper.origin: native` |
| `actor`, agent | contributor of type `agent` with its models (each model defined once in `models`, `accessed` the date of the earliest event by an agent using it), `name`, `record_source`, `run_period` (dates of its first and last events) and `tools: [code_execution]` if it ran commands. Position `author` if it wrote any event other than `actor` and `check`; otherwise `contributor`, with AI task `review` |
| `actor`, human or organisation | contributor with `name` and `orcid`, position `contributor`. Who is supervisor and guarantor is decided by people (completion list) |
| a verdict on a planned test (§5.7.4) | a claim per decided plan and hypothesis it names, in the order of the hypotheses and then the plans: `statement` from the hypothesis as it stood when the plan was written, `hypothesis: {lab_event, outcome}` (`lab_event` is that version's ID, `outcome` the verdict), defaults `type: descriptive`, `role: supporting`, `novelty: new`, `provenance: author_stated` |
| `hypothesis` named by no decided plan, with final status `supported`, `not_supported` or `inconclusive` | an exploratory claim, the same way, from the latest version of the hypothesis and its final status |
| `hypothesis` otherwise | not a claim; counted in `paper.exploration` |
| the runs behind a verdict or final status | one evidence object per claim, from the runs the status names as evidence, directly or through their results: `lab_runs`, `results` (one per `result` event: `quantity`, `value`, `unit`, `interval`, `conditions`), `execution` (`code_version` when the runs share one), default `profile: quant`, `design: other`, `availability: public`. The claim's support is `empirical_quant` pointing to the evidence and its results |
| `plan` | `planning`, `planned_at`, `planned_at_anchor` and `test` on the claims of its verdicts (below). `test` is the plan's decision rule as written before the run: `framework`, `parameters`, `quantities`, `supported`, `not_supported`, `verdict` (`computed` when the rule has quantities, otherwise `judged`), `runs`, `lab_plan` (the plan's ID) and `flags` (§5.7.5) |
| `deviation` | `planning: confirmatory_with_deviations`, and a limitation on the affected claim: `category: analytic_flexibility`, `impact: could_bias_magnitude`, `mitigation: none`, statement naming the run, the plan, what differed and why |
| `assumption` | assumption claim with `assumption_category` and `check`; `check_evidence` when it names a run or result used in an evidence object |
| `check` | a `checks` entry on the claim whose hypothesis, final status, or evidence run or result it targets. A check by an actor exported as an author is not exported (claimset rule 39) and is named in the completion list |
| `artefact` | artefact with `kind`, `uri` (the path or URI), `version` (the hash), `licence`, `description` |
| the whole record | artefact of kind `lab_record`: `uri` (the repository URI from `init` followed by the log's path, or the path alone), `version` (the hash of `lab.jsonl`), `availability: public`, `withheld_events`; referenced from `paper.exploration.lab_record` |

**Planning label of an exported claim.** A claim from a verdict is `confirmatory`, or `confirmatory_with_deviations` if a deviation is recorded for the plan and one of its deciding runs. Its `planned_at` is the `at` of the plan version that decides the plan (the one its first deciding run names in `plan_version`), and `planned_at_anchor` is the service of the earliest anchor that covers that plan version and was made before the first deciding run started, or `none`. A claim from a hypothesis named by no decided plan is `exploratory`.

**Exploration summary.** `questions` counts questions, each correction chain once; `hypotheses` counts hypotheses by final status, each correction chain once, with `open` for those without one; `tests` counts verdicts, one per decided plan and hypothesis it names; `runs` counts all run events by label, and `failed` those whose exit status is not `success`.

**Defaults the completion list always names:** `paper.licence` and the guarantor (asked of the responsible human), `paper.contributions` and principal roles, `paper.autonomy` (exported as `unknown`), claim types, statements of negative results (to say plainly what was not found), scopes, evidence profiles and descriptions, and links from claims to assumptions.

**Negative results** are exported as claims in the same claimset: the statement says what was not found, under what scope, and `hypothesis.outcome` is `not_supported`.

## 10. Tool commands (informative)

The reference tool is part of the `claimset` package. Indicative commands:

| Command | Effect |
|---|---|
| `claimset lab init` | creates `lab/`, writes `init` |
| `claimset lab actor` | registers an actor |
| `claimset lab question` / `hypothesis` / `assume` / `note` | writes the corresponding event |
| `claimset lab status H3 supported --evidence RS12 RS13 --why "…" [--plan PL2] [--rule supported]` | status change, with L6, L7 and L17 checked |
| `claimset lab verdict PL2` | the verdict a decided plan's rule gives, with the value bound to each quantity (§5.7.3) |
| `claimset lab plan --kind confirmatory --hypotheses H1 --design … --analysis … --framework frequentist --parameter alpha=0.05 … --quantity p="…" --supported "p < alpha" --not-supported "…" [--runs N] [--decision FILE]` | writes a plan with its decision rule (§5.7.1) and anchors it |
| `claimset lab approve PL2` | human approval |
| `claimset lab run [--plan PL2] [--exploratory] [--param k=v] [--seed N] [--data FILE] -- <command>` | runs the command, records `run` and `result` events (§5.9.1) |
| `claimset lab decide` / `deviate` / `check` / `artefact` | writes the corresponding event |
| `claimset lab summary` | current state: open questions, hypotheses by status, pending plans, recent decisions; for an actor with an open agent log, what that agent sees |
| `claimset lab show H3` | one event, with corrections resolved |
| `claimset lab list [--type hypothesis] [--last 20]` | events in order, one line each |
| any writing command `--supersedes ID` | writes a correction of `ID` (§4.3) |
| `claimset lab start --by AC2` | starts an agent log for AC2: its events go there until merged (§3.1) |
| `claimset lab merge [AC2 AC3]` | merges agent logs (default: all unmerged ones) into the main log (§3.1) |
| `claimset lab verify` | checks the main and agent chains, the rules and the run files |
| `claimset lab anchor [--git] [--remote origin] [--branch claimset-lab] [--every plan\|event]` | anchors the head of the chain; the first use with `--git` records the configuration (§7.1) |
| `claimset lab verify --remote` | also checks the log against the anchor branch on the remote |
| `claimset lab guard` | the guard for agent harnesses: reads a proposed tool call as JSON on standard input and exits with code 2 if it would modify the lab directory (§4.5); `--print-config` prints the Claude Code hook configuration |
| `claimset lab protect` | sets the operating system's append-only attribute on the log files, or explains how an administrator can (§4.5) |
| `claimset lab publish --withhold RN7 --reason personal_data --comment "…"` | prepares the publishable record |
| `claimset lab export [-o FILE]` | writes the draft claimset submission and prints the completion list (§9) |

## 11. Open questions

1. Anchoring service: platform receipts, RFC 3161 providers, or both.
2. Storage of large run outputs (content-addressed storage, external archives).
3. Event types for laboratory and qualitative research (data-collection sessions, instruments, coding rounds).
4. Very long explorations (thousands of runs): summarised export without losing the counts.
5. Whether abandoned hypotheses with substantial evidence should be exportable as claims.
