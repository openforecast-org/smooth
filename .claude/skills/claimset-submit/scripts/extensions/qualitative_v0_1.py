"""The qualitative extension, version 0.1 (spec/extensions/qualitative-v0.1.md): codebooks, codings and counts of
coded material. Never edit once released: a new version of the extension is a new module.

The claimset part (check) is here. The lab record part is implemented by the lab tool, in the module LAB names:
its event types (LAB_EVENTS) and the lab events its blocks may appear on (LAB_BLOCKS).
"""

NAME = "qualitative"
VERSION = "0.1"
OBJECTS = {"definition", "claim"}

#: Lab record: event types this extension defines (type -> ID prefix), events its blocks may appear on, and the
#: module that implements them.
LAB_EVENTS = {"coding": "CD"}
LAB_BLOCKS = {"term", "plan", "run"}
LAB = "claimset.lab.qualitative"

CODE_FIELDS = {"include", "exclude", "examples"}
TEST_FLAGS = ("coded_before_plan", "codebook_changed_after_plan")
DEFINITION_FIELDS = {"code"}
CLAIM_FIELDS = {"test_flags", "counts"}


def code_problems(code):
    """What makes a code's criteria invalid: {include, exclude, examples} (§2.1)."""
    if not isinstance(code, dict) or not isinstance(code.get("include"), str) or not code["include"].strip():
        return ["a code needs include: what a passage must show to get the code"]
    out = []
    extra = sorted(set(code) - CODE_FIELDS)
    if extra:
        out.append(f"a code has only include, exclude and examples, not {', '.join(extra)}")
    if code.get("exclude") is not None and not (isinstance(code["exclude"], str) and code["exclude"].strip()):
        out.append("a code's exclude is non-empty text")
    ex = code.get("examples")
    if ex is not None and not (isinstance(ex, list) and all(isinstance(x, str) and x.strip() for x in ex)):
        out.append("a code's examples are a list of texts")
    return out


def _counter():
    try:
        from . import qualitative_count
    except ImportError:                          # the submit skill: extensions/ is a top-level package
        import importlib
        qualitative_count = importlib.import_module(__name__.rsplit(".", 1)[0] + ".qualitative_count")
    return qualitative_count


def check(blocks, doc, r):
    defs = {d.get("id"): d for d in doc.get("definitions") or [] if isinstance(d, dict)}

    def is_code(did):
        d = defs.get(did)
        ext = d.get("ext") if isinstance(d, dict) else None
        block = ext.get("qualitative") if isinstance(ext, dict) else None
        return isinstance(block, dict) and isinstance(block.get("code"), dict)

    for where, typ, obj, block in blocks:
        allowed = DEFINITION_FIELDS if typ == "definition" else CLAIM_FIELDS
        extra = set(block) - allowed
        if extra:
            r.err("qualitative.1", where, f"on a {typ}: only {', '.join(sorted(allowed))}; not "
                                          f"{', '.join(sorted(extra))}")
        if typ == "definition" and "code" in block:
            if obj.get("kind") != "concept":
                r.err("qualitative.2", f"{where}.code", "a code of a codebook is a definition of kind concept")
            for p in code_problems(block["code"]):
                r.err("qualitative.2", f"{where}.code", p)
        if typ != "claim":
            continue
        test = obj.get("test") if isinstance(obj.get("test"), dict) else None
        if "test_flags" in block:
            flags = block["test_flags"]
            if test is None:
                r.err("qualitative.3", f"{where}.test_flags", "flags of a planned test: the claim has a test")
            if not isinstance(flags, list) or not flags or any(f not in TEST_FLAGS for f in flags):
                r.err("qualitative.3", f"{where}.test_flags", f"a non-empty list of {', '.join(TEST_FLAGS)}")
        if "counts" in block:
            counts = block["counts"]
            quantities = test.get("quantities") if test and isinstance(test.get("quantities"), dict) else {}
            if not isinstance(counts, dict) or not counts:
                r.err("qualitative.4", f"{where}.counts", "an object {name: count}")
                continue
            for name, spec in counts.items():
                for p in _counter().spec_problems(name, spec):
                    r.err("qualitative.4", f"{where}.counts.{name}", p)
                for c in (spec.get("codes") or []) if isinstance(spec, dict) else []:
                    if not is_code(c):
                        r.err("qualitative.4", f"{where}.counts.{name}.codes",
                              f"{c} is not a code: a definition with a qualitative block holding its criteria")
                q = quantities.get(name)
                want = _counter().result_variable(name)
                if not (isinstance(q, dict) and q.get("variable") == want):
                    r.err("qualitative.4", f"{where}.counts.{name}",
                          f"the claim's test has a quantity {name} that binds the counter's result, "
                          f"{{\"variable\": \"{want}\"}}")
