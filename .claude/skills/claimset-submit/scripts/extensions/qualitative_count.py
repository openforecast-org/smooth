"""Counts of coded material: the counter of the qualitative extension (spec/extensions/qualitative-v0.1.md, §3.4).

A plan's quantity can be a count over the recorded codings instead of a result a script writes: "the number of
threads from different years coded 'arrange for ensemble'". `claimset lab count` runs this file as the deciding
run's command: it reads the codings in force from $CLAIMSET_CODINGS and writes one result per count to
$CLAIMSET_RESULTS, whose variable is `count: <name>`, so the counts come from code like any other result, and
anyone can re-run it on the run's `codings.json`. The extension's rules use its `spec_problems` to check counts.

This file uses only the standard library and imports nothing from the package, so it runs as a script.
"""

import argparse
import json
import os
import sys

COUNT_KEYS = ("codes", "match", "unit", "distinct_by", "where")
UNITS = ("source", "coding")
DISTINCT = ("source", "coding", "author", "year", "venue")
MATCH = ("any", "all")
WHERE_KEYS = ("pass", "unit", "coder", "venue", "published_from", "published_to")


def result_variable(name):
    """The variable of the result that holds a count, which the plan's quantity binds to."""
    return f"count: {name}"


def spec_problems(name, spec):
    """What makes a count specification incomplete or invalid (rule qualitative.L3; L15 in records begun before
    lab record v0.20)."""
    if not isinstance(spec, dict):
        return [f"count of quantity '{name}' must be an object {{codes, match, unit, distinct_by, where}}"]
    out = []
    extra = sorted(set(spec) - set(COUNT_KEYS))
    if extra:
        out.append(f"count of quantity '{name}' has {', '.join(extra)}; it has only {', '.join(COUNT_KEYS)}")
    codes = spec.get("codes")
    if not isinstance(codes, list) or not codes or not all(isinstance(c, str) and c.strip() for c in codes):
        out.append(f"count of quantity '{name}' needs codes: a list of the codes counted (term IDs in a lab record, definition IDs in a claimset)")
    for key, allowed, default in (("match", MATCH, "any"), ("unit", UNITS, "source"),
                                  ("distinct_by", DISTINCT, None)):
        v = spec.get(key, default)
        if v is not None and v not in allowed:
            out.append(f"{key} of the count of quantity '{name}' must be one of {', '.join(allowed)}")
    where = spec.get("where")
    if where is not None:
        if not isinstance(where, dict):
            out.append(f"where of the count of quantity '{name}' must be an object")
        else:
            bad = sorted(set(where) - set(WHERE_KEYS))
            if bad:
                out.append(f"where of the count of quantity '{name}' has {', '.join(bad)}; it has only "
                           f"{', '.join(WHERE_KEYS)}")
            for k in ("published_from", "published_to"):
                v = where.get(k)
                if v is not None and not (isinstance(v, str) and len(v) >= 4 and v[:4].isdigit()):
                    out.append(f"where.{k} of the count of quantity '{name}' must be a date (YYYY, YYYY-MM-DD)")
    return out


def _ancestors(codes):
    """{code: set of itself and every broader code, transitively}."""
    broader = {c["id"]: set(c.get("broader") or []) for c in codes}
    out = {}
    for c in broader:
        seen, todo = {c}, list(broader[c])
        while todo:
            b = todo.pop()
            if b not in seen:
                seen.add(b)
                todo += list(broader.get(b, ()))
        out[c] = seen
    return out


def _keep(row, source, where):
    for k in ("pass", "unit", "coder"):
        if k in where and str(row.get(k)) != str(where[k]):
            return False
    if "venue" in where and source.get("venue") != where["venue"]:
        return False
    date = str(source.get("published_at") or "")
    if "published_from" in where and not (date and date[:len(where["published_from"])] >= where["published_from"]):
        return False
    if "published_to" in where and not (date and date[:len(where["published_to"])] <= where["published_to"]):
        return False
    return True


def count(snapshot, spec):
    """One count over a codings snapshot. Returns {value, n, missing}: `value` is the count; `n` the number of
    units that qualify before `distinct_by`; `missing` the qualifying units without the property `distinct_by`
    names (no author, no date, no venue), which are not counted."""
    codes = snapshot.get("codes") or []
    ancestors = _ancestors(codes)
    sources = {s["id"]: s for s in snapshot.get("sources") or []}
    wanted = list(dict.fromkeys(spec.get("codes") or []))
    match = spec.get("match", "any")
    unit = spec.get("unit", "source")
    where = spec.get("where") or {}

    def covered(row_codes):
        """The wanted codes a coding's codes are, or are narrower than."""
        have = set()
        for c in row_codes:
            have |= ancestors.get(c, {c})
        return {w for w in wanted if w in have}

    rows = [r for r in snapshot.get("codings") or [] if _keep(r, sources.get(r.get("source"), {}), where)]
    if unit == "coding":
        units = [(r["id"], r["source"]) for r in rows
                 if (covered(r.get("codes") or []) if match == "any" else covered(r.get("codes") or []) == set(wanted))]
    else:
        per_source = {}
        for r in rows:
            per_source.setdefault(r["source"], set()).update(covered(r.get("codes") or []))
        units = [(sid, sid) for sid, got in per_source.items() if (got if match == "any" else got == set(wanted))]
    by = spec.get("distinct_by") or unit
    keys, missing = set(), 0
    for uid, sid in units:
        s = sources.get(sid, {})
        key = {"source": sid, "coding": uid, "author": s.get("author"), "venue": s.get("venue"),
               "year": str(s["published_at"])[:4] if s.get("published_at") else None}[by]
        if key is None:
            missing += 1
        else:
            keys.add(key)
    return {"value": len(keys), "n": len(units), "missing": missing}


def results(snapshot, counts):
    """The results of a set of counts {name: spec}, as a run writes them."""
    out = []
    for name in sorted(counts):
        spec = counts[name]
        c = count(snapshot, spec)
        cond = {"unit": spec.get("unit", "source"), "distinct_by": spec.get("distinct_by") or spec.get("unit", "source")}
        if len(spec.get("codes") or []) == 1:
            cond["code"] = spec["codes"][0]
        out.append({"variable": result_variable(name), "value": c["value"], "n": c["n"], "conditions": cond})
        if c["missing"]:
            out.append({"variable": f"{result_variable(name)}, units not counted for want of "
                                    f"{spec.get('distinct_by')}", "value": c["missing"]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Count coded material from $CLAIMSET_CODINGS into $CLAIMSET_RESULTS.")
    p.add_argument("--counts", required=True, help="JSON object {name: {codes, match, unit, distinct_by, where}}")
    args = p.parse_args(argv)
    counts = json.loads(args.counts)
    probs = [x for name, spec in counts.items() for x in spec_problems(name, spec)]
    if probs:
        print("\n".join(probs), file=sys.stderr)
        return 2
    path = os.environ.get("CLAIMSET_CODINGS")
    snapshot = json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else {}
    out = results(snapshot, counts)
    with open(os.environ["CLAIMSET_RESULTS"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for r in out:
        print(f"{r['variable']} = {r['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
