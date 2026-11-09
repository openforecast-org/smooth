# Registry of extensions

Extensions add to the claimset core (claimset specification §2.7), and to the lab record (lab record §2.1, §5.22), the detail that one kind of work needs. One extension covers both: what the work records in the lab and what its claimset says. Each is specified in its own file, `<name>-v<major>.<minor>.md`, and follows the compatibility rule of §2.6 on its own: a block valid under a version of an extension stays valid under it for good.

An extension is registered here before it is used. A validator knows the extensions registered when it was released; an extension or a version registered later is refused by it with a message saying so (rule 80), and a current validator checks it.

| Extension | Versions | Adds | Objects | Core and lab record |
|---|---|---|---|---|
| `linguistics` | 0.1 (`linguistics-v0.1.md`) | Interlinear glosses of linguistic examples, their segmentation into morphemes, acceptability judgements | result entries with an excerpt value | from v0.38 |
| `qualitative` | 0.1 (`qualitative-v0.1.md`) | Codebooks, codings and counts of coded material | claimset: definitions (codes), claims (test flags, counts); lab record: terms, plans, runs, and the event type `coding` | from v0.38; lab record from v0.20 |

`candidates.md` surveys the extensions the format may need, by kind of work, and the needs found to belong in the core instead (informative).

## Writing an extension

- **Name** the kind of work, not the discipline (`synthesis`, not `chemistry`): one paper may use several extensions, and one kind of work happens in several disciplines.
- **Add fields only**, inside `ext.<name>` blocks on the objects the extension lists. Never change what a core field means, add values to a core list, or make a core field required. A file with every block removed must say everything the core says.
- **Point to the file's objects** by ID (a definition, an artefact, a result); do not repeat them.
- **Rules** are numbered within the extension (`<name>.1`, `<name>.2`, warnings `<name>.w1`), each with a test: a valid block that passes and a broken one that fails with that rule.
- **What two extensions would both define belongs in the core.** Propose it there instead.
- **Versions:** the first is 0.1. A new version may add optional fields and loosen rules; stricter rules apply to blocks of the new version only. A released version's validator is never edited.
- **Lab record part**, when the work records something in the lab: blocks on lab events (`LAB_BLOCKS`), event types of its own (`LAB_EVENTS`, with their ID prefixes), and how export carries them into the claimset.
- **Implementation:** a module `src/claimset/extensions/<name>_v<major>_<minor>.py`, listed in `KNOWN` in `src/claimset/extensions/__init__.py`, and a row in the table above; its lab part is implemented by the lab tool in the module its `LAB` names.
