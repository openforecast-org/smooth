---
name: ub-mode
description: >
  Follow the reduced engineering principles when writing or changing code in this
  repository: First Make it Work, Then Make it Right, Follow Practice. Adapted from
  unclebob/ub-mode to the rules of this repository's CLAUDE.md (topic branches, no new
  function or name without asking, no summaries, a clean root, R/Python parity), with
  the quality tools crapper, dryer and mutator on the Python port. Use when the user
  runs /ub-mode or asks for the reduced principles.
---

# ub-mode

Adapted from [unclebob/ub-mode](https://github.com/unclebob/ub-mode). Where it and
`CLAUDE.md` differ, `CLAUDE.md` wins.

Every time you are asked to make a change follow:
* First Make it Work.
* Then Make it Right.
* Follow Practice.

# First Make it Work

## Laziness Protocol

* One decision lives in one place.
* Ship the smallest edit that solves the problem.

## Foundational Thinking

* Get the data shape right before you write the logic.
* Do first whatever helps every later phase.
* If a concurrent writer would matter, do not share state.

## TDD

* Write the failing test before you change the code: `tests/testthat/` for R,
  `python/tests/` for Python, and an R-parity test for a behaviour the port shares.
* See it fail for the reason you expect.
* Make the smallest change that makes it pass.
* If an honest test is impractical, do not write a bad one.

## Redesign From First Principles

* Do not bolt a new requirement onto the old design.
* Ask what you would have built if it had been there from the start, think that
  through, and deliver it in pieces.
* The Python port is a direct translation of R: a redesign lands in both languages,
  in the same commit where practical, never in one alone.

## Attack the Premise

* When two fixes that share one premise fail the same gate, write that premise in one
  sentence and count which actors hold the imbalance.
* Remove the assignment.
* Do not start another fix until you have done that.

## Subtract Before You Add

* Take complexity out before you add anything.

## Outcome-Oriented Execution

* Aim at the end state you can check.
* A break in the middle is allowed when you planned it and can undo it.

## Build the Lever

* If the work is not obvious at a glance, build the tool that does it.
* Do one unit by hand, then make the tool match that unit.
* A throwaway tool lives in the scratchpad; one that would join the repository is a new
  function or file, so ask first.

## Architecture

### Model the Domain

* Put the domain in a structure that makes the illegal states impossible.
* The domain terms are the names `adam()` and `ADAM` set, listed in the name map of
  the `smooth-translation` skill; a new name needs asking, and goes into that map.
* If the code is already clear and local, leave it.

### Boundary Discipline

* Check the data where it enters: `parametersChecker()` in R, `parameters_checker()`
  in Python.
* Inside, trust it.
* Keep the business logic in a pure function.

### Type System Discipline

* Make the illegal value impossible to write.
* Parse outside data once, where it comes in.
* Do not silence the compiler, mypy or ruff.

### Make Operations Idempotent

* Running it twice, or running it after a crash halfway through, ends in the same
  state.

### Separate Before Serializing Shared State

* If they do not need the same mutable thing, give each its own.
* If they do, allow one writer, and enforce that in the structure.

# Then Make it Right

## Fix Root Causes

* Reproduce it.
* Fix the cause.
* A guard that only hides the crash is not a fix: no clipping, clamping or floors.
* A failing test is a real signal, never dismissed.

## Sequence work into verifiable units

* Finish one small unit and check it.
* Then start the next.

## Test Behavior, Not Implementation

* The test calls the code the way its user does and checks the result that user
  would see.
* If the test would still pass when every function returned nothing, throw it out.

## Crapper, Dryer and Mutator

They read Python (not R or C++), so they apply to `python/src/` only.

* Clone [crapper](https://github.com/unclebob/crapper),
  [dryer](https://github.com/unclebob/dryer) and
  [mutator](https://github.com/unclebob/mutator) side by side outside the repository
  (mutator needs `../crapper`); each makes its own `.venv` in its checkout on first run.
* Run them from `python/`, never from the root: they write `.metrics/` and `target/`
  where they run, both git-ignored there.
* Start with the files you changed (`--changed`, or the paths), not the whole tree.
* A function to split or a duplicate to extract means a new function: ask first, and
  change R the same way when the Python code mirrors it.

### Crapper

* `crapper --changed` (or the paths): reduce each function under CRAP 10 by testing
  it, by splitting it, or both.
* CRAP is never below the cyclomatic complexity, so a function with CC 10 or more needs
  splitting whatever its coverage.
* Its coverage comes from the default suite, without the R-parity tests: a function
  tested only through them shows as uncovered (`AutoCES.fit`, 1.1%).
* Consider whether case statements are worth reducing.

### Dryer

* `dryer --threshold 0.8 --changed`: eliminate the duplication that ranks 0.8 or above.

### Mutator

* It does not work on this package yet: it mutates a copy of the file in
  `target/mutation-workers`, but the editable install of scikit-build-core sends
  `import smooth` to `python/src`, where the compiled extensions are, so the tests never
  load the mutant and every one survives (0 of 84 killed on `utils/bootstrap.py`, where
  the same mutant applied in place fails `test_coefbootstrap.py`).
* Until it does, trust no survivor without applying that mutant in place, running the
  tests and restoring the file with `git checkout`.
* `mutator --test-command "<repo>/python/.venv/bin/python -m pytest -q tests/<file>.py" <file>`:
  the absolute interpreter of `python/.venv`, and the tests of that file.
* Cover the uncovered.
* Kill the survivors.
* Delete dead code, unless R still has it.

## Refactoring

* Pin what the code does.
* Change the structure.
* Prove the behavior did not change: the tests, and R parity for the Python port.

# Practice

## Keep a Log

* Track the actions and their reasons as you go, in the task list (in the lab record
  for research), not in a file of the repository.
* At the end, report the outcome and its evidence (the tests, the numbers), not a
  summary of what you did.

## Guard the Context Window

* Keep the main thread small.
* Send the bulk elsewhere, and keep the summary.

## Worktree Cleanup

* Find what is safe to remove, confirm nothing still in use needs it, then reclaim the
  disk.
* Uncommitted work and branches wait for a decision.

## Commit and Push

* Commit often, in small ordered commits; R and Python changes that mirror each other
  go in the same commit.
* Each commit says why, what changed, and how you proved it, with no session link.
* Push to the topic branch, never to `master`; open a pull request only when asked.
