#!/usr/bin/env python3
"""Claimset v0.11 reference validator (draft).

Part of the `claimset` package (https://github.com/claimset-org/claimset); also
bundled as a standalone script with the claimset-submit skill.

Usage:
    python3 validate.py claimset.json              # check a submission (default)
    python3 validate.py claimset.json --published  # check a published file
    python3 validate.py claimset.json --json       # machine-readable report

A submission omits platform-assigned fields (identifier, version, date, status,
classification IDs, verifiability). The report lists the verifiability values the
platform will compute, so the agent can review them.

Exit code 0: no errors (warnings allowed). Exit code 1: errors found.
Implements the rules in spec section 10, plus computed verifiability (section 5)
and heuristic checks of statement text (section 8). Standard library only.
"""
import json
from pathlib import Path
import math
import re
import sys

SCHEMA_VERSION = "0.19"

ENUM = {
    "status": {"active", "corrected", "retracted", "withdrawn", "expression_of_concern"},
    "origin": {"native", "converted"},
    "autonomy": {"autonomous", "human_supervised", "human_assisted", "unknown"},
    "contribution_record": {"complete", "partial", "unknown"},
    "claim_type": {"descriptive", "associational", "causal", "predictive", "interpretive", "formal", "methodological"},
    "role": {"principal", "supporting", "secondary"},
    "novelty": {"new", "replication", "extension", "refutation"},
    "provenance": {"author_stated", "extracted", "inferred"},
    "verification": {"machine_checked", "agent_checked", "peer_checked", "disputed", "unchecked"},
    "check_outcome": {"confirmed", "disputed", "inconclusive"},
    "check_extent": {"full", "partial"},
    "verifiability": {"open", "audit_only", "unverifiable", "not_applicable"},
    "flag": {"api_model_dependency", "figure_without_data", "unpublished_prompts",
             "unrecorded_model_settings", "unstated_support", "lab_record_withheld",
             "lab_events_withheld", "unanchored_plan"},
    "outcome": {"supported", "not_supported", "inconclusive"},
    "planning": {"confirmatory", "confirmatory_with_deviations", "exploratory"},
    "anchor": {"platform", "rfc3161", "git_remote", "none"},
    "framework": {"frequentist", "equivalence", "bayesian", "deterministic", "formal", "qualitative"},
    "verdict": {"computed", "judged"},
    "test_flag": {"code_changed_after_plan", "holdout_used_elsewhere"},
    "assumption_category": {"theoretical", "statistical", "methodological", "data", "axiom",
                            "theorem_condition", "paradigmatic"},
    "check": {"tested", "sensitivity_analysed", "unchecked", "untestable"},
    "profile": {"quant", "qual", "simulation", "formal"},
    "data_origin": {"primary", "secondary"},
    "design": {"experiment", "quasi_experiment", "observational", "survey", "benchmark", "other"},
    "data_sources": {"interview", "focus_group", "ethnography", "observation", "document", "other"},
    "analytic_approach": {"thematic", "grounded_theory", "ipa", "discourse", "framework", "case_study", "other"},
    "design_model_role": {"subject", "instrument", "annotator", "generator", "baseline"},
    "reasoning": {"deductive", "inductive", "abductive", "analogical", "interpretive"},
    "artefact_kind": {"code", "data", "protocol", "proof", "model", "software", "figure", "table",
                      "prompt", "trace", "manuscript", "lab_record", "other"},
    "relation": {"supports", "contradicts", "extends", "refines", "replicates", "fails_to_replicate", "supersedes",
                 "addresses"},
    "limitation_category": {"sampling", "sample_size", "measurement", "confounding", "data_quality",
                            "model_specification", "analytic_flexibility", "researcher_position",
                            "computational", "other"},
    "impact": {"could_reverse", "could_bias_magnitude", "reduces_precision", "reduces_transferability"},
    "mitigation": {"none", "partial", "addressed"},
    "level": {"public", "platform", "audit", "author_only", "unavailable"},
    "reason": {"personal_data", "commercial", "legal", "ethics", "security", "other"},
    "contributor_type": {"agent", "human", "organisation"},
    "position": {"author", "supervisor", "guarantor", "conversion_guarantor", "contributor", "converter"},
    "credit": {"conceptualization", "data_curation", "formal_analysis", "funding_acquisition", "investigation",
               "methodology", "project_administration", "resources", "software", "supervision", "validation",
               "visualization", "writing_original_draft", "writing_review_editing"},
    "task": {"ideation", "literature_search", "analysis", "code", "writing", "json_preparation", "review", "submission"},
    "extent": {"full", "substantial", "assisted"},
    "tool": {"web_search", "code_execution", "file_access", "retrieval", "other", "none"},
    "record_source": {"harness", "user_reported", "self_reported"},
    "agent_model_role": {"sole", "orchestrator", "subagent"},
    "weights": {"open", "api_only"},
    "support_kind": {"empirical_quant", "empirical_qual", "simulation", "demonstration", "formal_proof",
                     "argument", "cited", "consensus", "expert_judgement", "assumption", "unstated"},
    "id_type": {"doi", "arxiv", "handle", "pmid", "isbn", "url", "physical", "reference", "claimset_local"},
    "rel_id_type": {"claimset", "doi", "arxiv", "handle", "pmid", "isbn", "url", "physical", "claimset_local"},
    "by": {"author", "reviewer"},
    "reviewer_type": {"ai", "human"},
    "strength": {"high", "moderate", "low", "very_low"},
}

PREFIX = {"definitions": "D", "claims": "C", "evidence": "E", "warrants": "W",
          "limitations": "L", "artefacts": "A", "contributors": "P", "models": "M", "references": "B"}

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TERM_RE = re.compile(r"\[\[(D\d+):[^\]]+\]\]")

# Rule 31: source licences that permit conversion → the licences the claimset may carry (spec §3.1).
_BY = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"}
_SA = {"CC-BY-SA-3.0", "CC-BY-SA-4.0"}
_NC = {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}
_NC_SA = {"CC-BY-NC-SA-3.0", "CC-BY-NC-SA-4.0"}
CONVERSION_LICENCES = {
    "CC0-1.0": _BY, "CC-BY-2.0": _BY | {"CC-BY-2.0"}, "CC-BY-3.0": _BY | {"CC-BY-3.0"}, "CC-BY-4.0": _BY,
    "CC-BY-SA-3.0": _SA, "CC-BY-SA-4.0": _SA,
    "CC-BY-NC-3.0": _NC | {"CC-BY-NC-3.0"}, "CC-BY-NC-4.0": _NC,
    "CC-BY-NC-SA-3.0": _NC_SA, "CC-BY-NC-SA-4.0": _NC_SA,
}

ANAPHORA_RE = re.compile(r"^(this|these|those|it|they|such)\b|\b(as shown|the above|aforementioned|see section|"
                         r"see figure|see table|as mentioned|as discussed|previously described)\b", re.I)
EVALUATIVE_RE = re.compile(r"\b(novel|significant(ly)?|remarkabl[ey]|groundbreaking|state-of-the-art|"
                           r"impressive|striking|important|crucial|first ever|unprecedented)\b", re.I)
INSTRUCTION_RE = re.compile(r"\b(ignore (all|any|previous|prior)|you (should|must|are)|as an ai|language model,|"
                            r"system prompt|disregard|follow these|do not tell|assistant:|respond with)\b", re.I)


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.computed = {}

    def err(self, rule, where, msg):
        self.errors.append({"rule": rule, "where": where, "message": msg})

    def warn(self, rule, where, msg):
        self.warnings.append({"rule": rule, "where": where, "message": msg})


def as_list(x):
    return x if isinstance(x, list) else []


def as_dict(x):
    return x if isinstance(x, dict) else {}


def check_enum(r, rule, where, value, enum_name, required=False):
    if value is None:
        if required:
            r.err(rule, where, "missing required value")
        return
    if value not in ENUM[enum_name]:
        r.err(rule, where, f"'{value}' is not one of {sorted(ENUM[enum_name])}")


PLATFORM_PAPER_FIELDS = ("version", "previous_version", "changes", "date", "status", "status_notice")


NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,59}$")
LOCAL_RE = re.compile(r"^([a-z][a-z0-9-]{1,59})/([a-z][a-z0-9-]{1,59})\.claimset\.json(?:#(C\d+))?$")


def local_targets(doc):
    """[(where, target)] for every claimset_local reference: references, cited support and relations."""
    out = []
    for b in as_list(doc.get("references")):
        b = as_dict(b)
        if b.get("id_type") == "claimset_local":
            out.append((f"references:{b.get('id')}.target", b.get("target")))
    for c in as_list(doc.get("claims")):
        c = as_dict(c)
        for i, s_ in enumerate(as_list(c.get("support"))):
            s_ = as_dict(s_)
            if s_.get("id_type") == "claimset_local":
                out.append((f"claims:{c.get('id')}.support[{i}].target", s_.get("target")))
        for i, rel in enumerate(as_list(c.get("relations"))):
            rel = as_dict(rel)
            if rel.get("id_type") == "claimset_local":
                out.append((f"claims:{c.get('id')}.relations[{i}].target", rel.get("target")))
    return out


def check_local(r, doc, published, local_root):
    """Rule 57: references to another paper of the same repository (claimset/<study>/<name>.claimset.json)."""
    for where, target in local_targets(doc):
        m = LOCAL_RE.match(target) if isinstance(target, str) else None
        if not m:
            r.err("57", where, "a local claimset reference is '<study>/<paper>.claimset.json' or "
                               "'<study>/<paper>.claimset.json#C3', relative to the repository's claimset/ folder")
            continue
        if published:
            r.err("57", where, "published files cite published claimsets by their global claim IDs "
                               "(id_type claimset); local references are replaced at publication")
            continue
        if local_root is None:
            continue
        path = Path(local_root) / m.group(1) / f"{m.group(2)}.claimset.json"
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            r.err("57", where, f"{m.group(1)}/{m.group(2)}.claimset.json is not a readable claimset in "
                               f"{local_root}")
            continue
        if m.group(3) and m.group(3) not in {as_dict(c).get("id") for c in as_list(as_dict(other).get("claims"))}:
            r.err("57", where, f"{m.group(1)}/{m.group(2)}.claimset.json has no claim {m.group(3)}")


def claimset_folder(path):
    """The nearest folder named claimset/ above a file (the repository's studies), or None."""
    for d in Path(path).resolve().parents:
        if d.name == "claimset":
            return d
    return None


def check_covers(r, exp):
    """Rule 55: a paper that reports some of its lab's questions says which, with the lab's totals."""
    cov = exp.get("covers")
    if cov is None:
        return
    cov = as_dict(cov)

    def nonneg(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0
    qs = cov.get("questions")
    if not (isinstance(qs, list) and qs and all(isinstance(q, str) and q for q in qs)):
        r.err("55", "paper.exploration.covers.questions", "must be a non-empty list of lab question IDs")
        qs = []
    elif exp.get("questions") != len(qs):
        r.err("55", "paper.exploration.questions", "must equal the number of questions in covers.questions")
    lab = as_dict(cov.get("lab"))
    if not nonneg(lab.get("questions")):
        r.err("55", "paper.exploration.covers.lab.questions", "must be a non-negative integer")
    for part, keys in (("hypotheses", ("total", "supported", "not_supported", "inconclusive", "revised",
                                       "abandoned", "open")),
                       ("tests", ("total", "supported", "not_supported", "inconclusive"))):
        mine, total = as_dict(exp.get(part)), as_dict(lab.get(part))
        for k in keys:
            if not nonneg(total.get(k)):
                r.err("55", f"paper.exploration.covers.lab.{part}.{k}", "must be a non-negative integer")
            elif nonneg(mine.get(k)) and mine[k] > total[k]:
                r.err("55", f"paper.exploration.{part}.{k}", f"cannot exceed the lab's count ({total[k]})")
    if nonneg(lab.get("questions")) and len(qs) > lab["questions"]:
        r.err("55", "paper.exploration.covers.questions", "cannot exceed the lab's count of questions")
    seen = set(qs)
    for i, o in enumerate(as_list(cov.get("other_papers"))):
        o = as_dict(o)
        if not (isinstance(o.get("name"), str) and NAME_RE.match(o["name"])):
            r.err("55", f"paper.exploration.covers.other_papers[{i}].name", "must be a paper name")
        oq = o.get("questions")
        if not (isinstance(oq, list) and oq and all(isinstance(q, str) for q in oq)):
            r.err("55", f"paper.exploration.covers.other_papers[{i}].questions", "must be a non-empty list")
            continue
        if seen & set(oq):
            r.err("55", f"paper.exploration.covers.other_papers[{i}].questions",
                  f"{', '.join(sorted(seen & set(oq)))} reported in more than one paper")
        seen |= set(oq)


def validate(doc, published=False, local_root=None):
    """Check a claimset. `local_root`: the repository's claimset/ folder, when the file lies in one, so that
    local references to other papers can be resolved (rule 57)."""
    r = Report()
    if not isinstance(doc, dict):
        r.err("2", "$", "top level must be a JSON object")
        return r

    if doc.get("schema_version") != SCHEMA_VERSION:
        got = doc.get("schema_version")
        r.err("2", "schema_version", f"must be '{SCHEMA_VERSION}'" + (
            f"; the file follows specification {got}, an earlier version with different rules. Read the "
            f"current specification (`claimset spec`) and update the skills (`claimset skill install`)"
            if isinstance(got, str) and got else ""))

    for key in ("paper", "contributors", "models", "claims"):
        if key not in doc:
            r.err("19", key, "missing required top-level section")
    paper = as_dict(doc.get("paper"))
    coll = {k: as_list(doc.get(k)) for k in PREFIX}

    # ---------- IDs ----------
    ids = {}
    for k, items in coll.items():
        for i, obj in enumerate(items):
            where = f"{k}[{i}]"
            if not isinstance(obj, dict):
                r.err("1", where, "must be an object")
                continue
            oid = obj.get("id")
            if not isinstance(oid, str) or not re.match(rf"^{PREFIX[k]}\d+$", oid):
                r.err("1", where, f"id must look like {PREFIX[k]}1, {PREFIX[k]}2, …")
                continue
            if oid in ids:
                r.err("1", where, f"duplicate id {oid}")
            ids[oid] = (k, obj)

    def by_id(oid, kind):
        hit = ids.get(oid)
        return hit[1] if hit and hit[0] == kind else None

    def ref(rule, where, oid, kind):
        if by_id(oid, kind) is None:
            r.err(rule, where, f"reference '{oid}' does not resolve to an object in {kind}")
            return None
        return by_id(oid, kind)

    claims = {c["id"]: c for c in coll["claims"] if isinstance(c, dict) and "id" in c}
    origin = paper.get("origin")
    converted = origin == "converted"

    # ---------- Paper ----------
    for f in ("title", "licence", "contributions", "contribution_record", "origin"):
        if f not in paper:
            r.err("19", f"paper.{f}", "missing required field")
    if published:
        for f in ("id", "version", "status", "date", "classification"):
            if f not in paper:
                r.err("27", f"paper.{f}", "platform-assigned field missing from published file")
    else:
        for f in PLATFORM_PAPER_FIELDS:
            if f in paper:
                r.err("24", f"paper.{f}", "assigned by the platform at publication; remove it from the submission")
        if "id" in paper and "correction_reason" not in paper:
            r.err("25", "paper.id", "the platform mints the DOI; remove paper.id (keep it only when submitting "
                                    "a correction, together with correction_reason)")
        if "correction_reason" in paper and "id" not in paper:
            r.err("26", "paper.correction_reason", "a correction must name the existing claimset in paper.id")
    check_enum(r, "1", "paper.status", paper.get("status"), "status")
    check_enum(r, "1", "paper.origin", origin, "origin")
    check_enum(r, "1", "paper.contribution_record", paper.get("contribution_record"), "contribution_record")
    if "autonomy" in paper:
        check_enum(r, "1", "paper.autonomy", paper.get("autonomy"), "autonomy")
    if "date" in paper and not DATE_RE.match(str(paper.get("date"))):
        r.err("1", "paper.date", "must be an ISO date YYYY-MM-DD")
    version = paper.get("version")
    if "version" in paper and (not isinstance(version, int) or version < 1):
        r.err("28", "paper.version", "must be an integer >= 1")
    if isinstance(version, int) and version > 1:
        if not paper.get("previous_version") or not paper.get("changes"):
            r.err("28", "paper", "version > 1 requires previous_version and changes")
        for i, ch in enumerate(as_list(paper.get("changes"))):
            if not isinstance(ch, dict) or "op" not in ch or "path" not in ch or "reason" not in ch:
                r.err("28", f"paper.changes[{i}]", "each change needs op, path and reason")
    if paper.get("status") not in (None, "active"):
        notice = as_dict(paper.get("status_notice"))
        missing = [f for f in ("date", "reason_code", "statement", "issued_by") if f not in notice]
        if missing:
            r.err("29", "paper.status_notice", f"required when status is not active; missing {missing}")
    cls = as_dict(paper.get("classification"))
    if published and not cls.get("subfield"):
        r.err("27", "paper.classification.subfield", "platform-assigned OpenAlex subfield missing")
    if not published and not cls.get("keywords"):
        r.warn("W5", "paper.classification.keywords", "recommended: keywords help the platform classify the paper")
    topics = as_list(cls.get("topics"))
    if len(topics) > 3:
        r.err("1", "paper.classification.topics", "at most 3 topics")

    contributions = as_list(paper.get("contributions"))
    for cid in contributions:
        c = ref("5", f"paper.contributions:{cid}", cid, "claims")
        if c is not None and c.get("role") != "principal":
            r.err("5", f"paper.contributions:{cid}", "contributions may list only principal claims")
    for cid, c in claims.items():
        if c.get("role") == "principal" and cid not in contributions:
            r.err("5", f"claims:{cid}", "principal claim missing from paper.contributions")

    # ---------- Corrections in the conversion record (rule 49) ----------
    corrections = as_dict(paper.get("converted_from")).get("corrections")
    if corrections is not None:
        cw = "paper.converted_from.corrections"
        if not converted:
            r.err("49", cw, "corrections appear only in converted papers")
        elif not isinstance(corrections, list):
            r.err("49", cw, "must be a list")
        else:
            for i, corr in enumerate(corrections):
                w = f"{cw}[{i}]"
                if not isinstance(corr, dict):
                    r.err("49", w, "must be an object")
                    continue
                for f in ("field", "where", "printed", "recorded", "reason"):
                    if not isinstance(corr.get(f), str) or not corr[f].strip():
                        r.err("49", w, f"missing {f}")
                fld = corr.get("field")
                if isinstance(fld, str) and fld.split(".")[0] not in ids:
                    r.err("49", f"{w}.field", f"'{fld}' does not begin with the ID of an object in the file")

    # ---------- References (rule 51) ----------
    for b in coll["references"]:
        if not isinstance(b, dict):
            continue
        bw = f"references:{b.get('id')}"
        if not isinstance(b.get("target"), str) or not b["target"].strip():
            r.err("51", bw, "missing target")
        if b.get("id_type") not in ENUM["id_type"] | ENUM["rel_id_type"]:
            r.err("51", f"{bw}.id_type", f"'{b.get('id_type')}' is not an identifier type for cited works")
        if b.get("id_type") == "url" and not b.get("archive_uri"):
            r.err("51", bw, "url targets require archive_uri")
        if b.get("id_type") == "reference" and not converted:
            r.warn("W14", f"{bw}.id_type", "a printed reference without identifier: give the cited work's identifier (doi, arxiv, …)")
    if coll["references"]:
        pointed = set()
        for c in coll["claims"]:
            c = as_dict(c)
            pointed |= {as_dict(s).get("reference") for s in as_list(c.get("support"))}
            pointed |= {as_dict(x).get("reference") for x in as_list(c.get("relations"))}
        pointed |= {as_dict(d).get("source") for d in coll["definitions"] if isinstance(as_dict(d).get("source"), str)}
        pointed |= {as_dict(a).get("reference") for a in coll["artefacts"]}
        for b in coll["references"]:
            if isinstance(b, dict) and b.get("id") and b["id"] not in pointed:
                r.err("51", f"references:{b['id']}", "listed but never cited: list only works that are cited")

    # ---------- Roles and what the contributions rest on (warnings 15 and 16) ----------
    def rests_on(c):
        """Claims this claim rests on: depends_on, assumptions, and the assumptions of the evidence it cites."""
        out = list(as_list(c.get("depends_on"))) + list(as_list(c.get("assumptions")))
        for s in as_list(c.get("support")):
            e = by_id(as_dict(s).get("evidence"), "evidence")
            if e is not None:
                out += list(as_list(e.get("assumptions")))
        return [x for x in out if isinstance(x, str)]

    reached, stack = set(), [cid for cid, c in claims.items() if c.get("role") == "principal"]
    while stack:
        cur = stack.pop()
        for nxt in rests_on(claims.get(cur) or {}):
            if nxt in claims and nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    for cid, c in claims.items():
        if c.get("role") == "supporting" and cid not in reached:
            r.warn("W15", f"claims:{cid}", "no principal claim rests on this supporting claim: link it "
                                           "(depends_on, assumptions) or make it secondary")
        if c.get("role") in ("principal", "supporting"):
            for nxt in rests_on(c):
                if (claims.get(nxt) or {}).get("role") == "secondary":
                    r.warn("W16", f"claims:{cid}", f"rests on {nxt}, which is secondary: a claim that a "
                                                   "contribution rests on is supporting")

    # ---------- Corrections marked inferred (rule 52) ----------
    if converted:
        for i, corr in enumerate(as_list(as_dict(paper.get("converted_from")).get("corrections"))):
            fld = as_dict(corr).get("field")
            if not isinstance(fld, str) or "." not in fld:
                continue
            oid, path = fld.split(".", 1)
            target = by_id(oid, "claims")
            if target is not None and target.get("provenance") != "inferred" \
                    and as_dict(target.get("field_provenance")).get(path) != "inferred":
                r.err("52", f"paper.converted_from.corrections[{i}]",
                      f"{fld} is listed as corrected but its provenance is not inferred (add field_provenance)")

    # ---------- Definitions ----------
    for d in coll["definitions"]:
        if isinstance(d, dict):
            for f in ("term", "definition"):
                if not d.get(f):
                    r.err("1", f"definitions:{d.get('id')}", f"missing {f}")
            if isinstance(d.get("source"), str) and re.match(r"^B\d+$", d["source"]):
                ref("51", f"definitions:{d.get('id')}.source", d["source"], "references")
            if "source" in d:
                src = d["source"]
                ok = (isinstance(src, str) and src.strip()) or (
                    isinstance(src, dict) and isinstance(src.get("target"), str) and src["target"].strip()
                    and src.get("id_type") in ENUM["id_type"])
                if not ok:
                    r.err("50", f"definitions:{d.get('id')}.source",
                          "must be a non-empty string or {target, id_type} with id_type from the list for cited support")

    # ---------- Artefacts ----------
    for a in coll["artefacts"]:
        if not isinstance(a, dict):
            continue
        w = f"artefacts:{a.get('id')}"
        check_enum(r, "1", f"{w}.kind", a.get("kind"), "artefact_kind", required=True)
        if a.get("kind") in ("figure", "table") and not a.get("description"):
            r.err("1", w, "figures and tables require a description")
        if a.get("underlying_data"):
            ref("1", f"{w}.underlying_data", a["underlying_data"], "artefacts")
        if a.get("reference") is not None:
            ref("54", f"{w}.reference", a["reference"], "references")
        check_availability(r, w, a.get("availability"))
        if not a.get("licence"):
            r.warn("W9", f"{w}.licence", "recommended: state the artefact's licence (SPDX identifier)")

    # ---------- Models ----------
    for m in coll["models"]:
        if not isinstance(m, dict):
            continue
        w = f"models:{m.get('id')}"
        for f in ("provider", "name", "version"):
            if not m.get(f):
                r.err("19", f"{w}.{f}", "missing required field")
        for f in ("identifier", "weights", "accessed"):
            if not m.get(f):
                r.warn("W1", f"{w}.{f}", "recommended field missing")
        if m.get("weights") is not None:
            check_enum(r, "1", f"{w}.weights", m.get("weights"), "weights")
        if m.get("weights") == "open" and not m.get("weights_uri"):
            r.warn("W1", f"{w}.weights_uri", "recommended for open-weights models")

    # ---------- Evidence ----------
    for e in coll["evidence"]:
        if not isinstance(e, dict):
            continue
        w = f"evidence:{e.get('id')}"
        check_enum(r, "1", f"{w}.profile", e.get("profile"), "profile", required=True)
        if e.get("data_origin") is not None:
            check_enum(r, "1", f"{w}.data_origin", e.get("data_origin"), "data_origin")
        if e.get("design") is not None:
            check_enum(r, "1", f"{w}.design", e.get("design"), "design")
        for ds in as_list(e.get("data_sources")):
            check_enum(r, "1", f"{w}.data_sources", ds, "data_sources")
        if e.get("analytic_approach") is not None:
            check_enum(r, "1", f"{w}.analytic_approach", e.get("analytic_approach"), "analytic_approach")
        for aid in as_list(e.get("artefacts")):
            ref("1", f"{w}.artefacts", aid, "artefacts")
        if "availability" not in e:
            r.warn("W-avail", w, "availability not stated; verifiability cannot be confirmed")
        check_availability(r, w, e.get("availability"))
        for i, dm in enumerate(as_list(e.get("models"))):
            dw = f"{w}.models[{i}]"
            dm = as_dict(dm)
            ref("18", f"{dw}.model", dm.get("model"), "models")
            check_enum(r, "1", f"{dw}.role", dm.get("role"), "design_model_role", required=True)
            if dm.get("record_source") is not None:
                check_enum(r, "1", f"{dw}.record_source", dm.get("record_source"), "record_source")
            if dm.get("prompts"):
                p = ref("1", f"{dw}.prompts", dm["prompts"], "artefacts")
                if p is not None and p.get("kind") != "prompt":
                    r.err("1", f"{dw}.prompts", "must point to an artefact of kind prompt")
        for lr in as_list(e.get("lab_runs")):
            if not re.match(r"^(AC\d+\.)?RN\d+$", str(lr)):
                r.err("1", f"{w}.lab_runs", f"'{lr}' is not a lab-record run ID (RN1, RN2, …)")
        if e.get("profile") in ("quant", "simulation"):
            res = as_list(e.get("results"))
            if e.get("estimate") is None and not res and not as_list(e.get("tables")):
                r.warn("W8", w, "no estimate and no results: add a structured results list for the numbers this evidence produced")
            seen_r = set()
            for i, rs in enumerate(res):
                rs = as_dict(rs)
                rw = f"{w}.results[{i}]"
                rid = rs.get("id")
                if not isinstance(rid, str) or not re.match(r"^R\d+$", rid):
                    r.err("38", rw, "result id must look like R1, R2, …")
                elif rid in seen_r:
                    r.err("38", rw, f"duplicate result id {rid}")
                else:
                    seen_r.add(rid)
                if not rs.get("quantity"):
                    r.err("38", rw, "missing quantity")
                if not isinstance(rs.get("value"), (int, float)) or isinstance(rs.get("value"), bool):
                    r.err("38", rw, "value must be a number")
                if rs.get("artefact"):
                    ref("38", f"{rw}.artefact", rs["artefact"], "artefacts")
            check_tables(r, w, e, ref)
        elif e.get("results"):
            r.err("38", w, "results are used only in quant and simulation evidence")
        if e.get("profile") not in ("quant", "simulation") and e.get("tables"):
            r.err("48", w, "result tables are used only in quant and simulation evidence")
        if "source_spans" in e and not (isinstance(e["source_spans"], list)
                                        and all(isinstance(x, str) and x for x in e["source_spans"])):
            r.err("1", f"{w}.source_spans", "must be a list of non-empty strings")
        if e.get("profile") == "formal":
            if "theorem" in e:
                r.err("37", f"{w}.theorem", "removed in v0.9: the formal claim's statement is the theorem")
            steps = as_list(e.get("steps"))
            seen = set()
            for i, s in enumerate(steps):
                s = as_dict(s)
                sw = f"{w}.steps[{i}]"
                for f in ("id", "statement", "by", "rule"):
                    if f not in s:
                        r.err("7", sw, f"proof step missing {f}")
                for b in as_list(s.get("by")):
                    if b not in seen and b not in ids:
                        r.err("1", sw, f"step rests on '{b}', which is neither an earlier step nor an object id")
                if s.get("id"):
                    seen.add(s["id"])
            if e.get("proof_artefact"):
                ref("1", f"{w}.proof_artefact", e["proof_artefact"], "artefacts")
            for aid in as_list(e.get("assumptions")):
                c = ref("1", f"{w}.assumptions", aid, "claims")
                if c is not None and not is_assumption(c):
                    r.err("3", f"{w}.assumptions", f"{aid} is not an assumption claim")

    # ---------- Warrants ----------
    for wa in coll["warrants"]:
        if not isinstance(wa, dict):
            continue
        w = f"warrants:{wa.get('id')}"
        check_enum(r, "1", f"{w}.reasoning", wa.get("reasoning"), "reasoning", required=True)
        if not wa.get("text"):
            r.err("1", w, "missing text")
        for p in as_list(wa.get("premises")):
            ref("1", f"{w}.premises", p, "claims")
        text_checks(r, f"{w}.text", wa.get("text"), ids)

    # ---------- Limitations ----------
    for li in coll["limitations"]:
        if not isinstance(li, dict):
            continue
        w = f"limitations:{li.get('id')}"
        check_enum(r, "1", f"{w}.category", li.get("category"), "limitation_category", required=True)
        check_enum(r, "1", f"{w}.impact", li.get("impact"), "impact", required=True)
        check_enum(r, "1", f"{w}.mitigation", li.get("mitigation"), "mitigation", required=True)
        affects = as_list(li.get("affects"))
        if not affects:
            r.err("11", w, "must affect at least one claim")
        for cid in affects:
            ref("11", f"{w}.affects", cid, "claims")
        if li.get("mitigation") in ("partial", "addressed"):
            if not li.get("mitigation_evidence"):
                r.err("11", w, "mitigation partial/addressed requires mitigation_evidence")
            else:
                ref("11", f"{w}.mitigation_evidence", li["mitigation_evidence"], "evidence")
        text_checks(r, f"{w}.statement", li.get("statement"), ids)

    # ---------- Claims ----------
    author_models = set()
    for p in coll["contributors"]:
        if isinstance(p, dict) and p.get("type") == "agent" and "author" in as_list(p.get("positions")):
            for m in as_list(p.get("models")):
                author_models.add(model_key(by_id(as_dict(m).get("model"), "models"), as_dict(m).get("model")))
    for cid, c in claims.items():
        validate_claim(r, c, cid, ids, by_id, ref, converted, author_models)

    # identical scopes (warning W6)
    groups = {}
    for cid, c in claims.items():
        if c.get("scope"):
            groups.setdefault(json.dumps(c["scope"], sort_keys=True), []).append(cid)
    for g in groups.values():
        if len(g) >= 3:
            r.warn("W6", "claims:" + ",".join(g), "these claims share an identical scope; check that each scope fits its own claim")

    # depends_on cycles (rule 8)
    graph = {cid: [d for d in as_list(c.get("depends_on")) if d in claims] for cid, c in claims.items()}
    state = {}

    def visit(n, stack):
        state[n] = 1
        for m in graph.get(n, []):
            if state.get(m) == 1:
                r.err("8", f"claims:{n}", f"depends_on cycle: {' -> '.join(stack + [n, m])}")
            elif state.get(m) is None:
                visit(m, stack + [n])
        state[n] = 2

    for n in graph:
        if state.get(n) is None:
            visit(n, [])

    # ---------- Lab record and exploration (rules 42-46) ----------
    lab = {"level": None, "withheld": set()}
    hyp_claims = [c for c in claims.values() if c.get("hypothesis")]
    exp = paper.get("exploration")
    if hyp_claims and not exp:
        r.err("44", "paper.exploration", "required when claims carry a hypothesis from a lab record")
    if exp is not None:
        exp = as_dict(exp)
        la = ref("44", "paper.exploration.lab_record", exp.get("lab_record"), "artefacts")
        if la is not None:
            if la.get("kind") != "lab_record":
                r.err("44", "paper.exploration.lab_record", "must point to an artefact of kind lab_record")
            lab["level"] = as_dict(la.get("availability")).get("level")
            lab["withheld"] = {as_dict(x).get("event") for x in as_list(la.get("withheld_events"))}
            for i, x in enumerate(as_list(la.get("withheld_events"))):
                x = as_dict(x)
                if not x.get("event"):
                    r.err("1", f"artefacts:{la.get('id')}.withheld_events[{i}]", "missing event")
                check_enum(r, "1", f"artefacts:{la.get('id')}.withheld_events[{i}].reason", x.get("reason"), "reason", required=True)
        def nonneg(v):
            return isinstance(v, int) and not isinstance(v, bool) and v >= 0
        hy, rn = as_dict(exp.get("hypotheses")), as_dict(exp.get("runs"))
        hy = dict(hy, open=hy.get("open", 0))
        hkeys = ("supported", "not_supported", "inconclusive", "revised", "abandoned", "open")
        for k in ("total",) + hkeys:
            if not nonneg(hy.get(k)):
                r.err("45", f"paper.exploration.hypotheses.{k}", "must be a non-negative integer")
        ts = exp.get("tests")
        tkeys = ("supported", "not_supported", "inconclusive")
        if ts is not None:
            ts = as_dict(ts)
            for k in ("total",) + tkeys:
                if not nonneg(ts.get(k)):
                    r.err("45", f"paper.exploration.tests.{k}", "must be a non-negative integer")
            if all(nonneg(ts.get(k)) for k in ("total",) + tkeys) and ts["total"] != sum(ts[k] for k in tkeys):
                r.err("45", "paper.exploration.tests.total", "must equal the sum of the verdict counts")
        for k in ("total", "confirmatory", "exploratory", "failed"):
            if not nonneg(rn.get(k)):
                r.err("45", f"paper.exploration.runs.{k}", "must be a non-negative integer")
        if all(nonneg(hy.get(k)) for k in ("total",) + hkeys) and hy["total"] != sum(hy[k] for k in hkeys):
            r.err("45", "paper.exploration.hypotheses.total", "must equal the sum of the other hypothesis counts")
        if all(nonneg(rn.get(k)) for k in ("total", "confirmatory", "exploratory", "failed")):
            if rn["total"] != rn["confirmatory"] + rn["exploratory"]:
                r.err("45", "paper.exploration.runs.total", "must equal confirmatory + exploratory")
            if rn["failed"] > rn["total"]:
                r.err("45", "paper.exploration.runs.failed", "cannot exceed total")
        for oc in ("supported", "not_supported", "inconclusive"):
            n = len({as_dict(c["hypothesis"]).get("lab_event") for c in hyp_claims
                     if as_dict(c["hypothesis"]).get("outcome") == oc and not c.get("test")})
            if nonneg(hy.get(oc)) and n > hy[oc]:
                r.err("46", "paper.exploration.hypotheses", f"{n} distinct hypotheses reported as {oc} outside "
                                                            f"planned tests, but the count is {hy[oc]}")
            nt = sum(1 for c in hyp_claims if c.get("test") and as_dict(c["hypothesis"]).get("outcome") == oc)
            limit = as_dict(ts).get(oc) if ts is not None else None
            if nt and not nonneg(limit):
                r.err("46", "paper.exploration.tests", f"{nt} planned tests reported as {oc}, but exploration has no tests counts")
            elif nt and nt > limit:
                r.err("46", "paper.exploration.tests", f"{nt} planned tests reported as {oc}, but the count is {limit}")

    # ---------- Studies and papers in a repository (rules 55-57) ----------
    if paper.get("study") is not None and not (isinstance(paper["study"], str) and NAME_RE.match(paper["study"])):
        r.err("56", "paper.study", "must be the study's name: lower-case letters, digits and hyphens, starting "
                                   "with a letter")
    if isinstance(paper.get("exploration"), dict):
        check_covers(r, paper["exploration"])
    check_local(r, doc, published, local_root)

    # ---------- Contributors ----------
    validate_contributors(r, coll["contributors"], paper, ref, converted)

    # ---------- Computed verifiability (rule 13) ----------
    for cid, c in claims.items():
        exp_v, exp_flags, notes = compute_verifiability(c, by_id)
        if c.get("hypothesis") and not is_assumption(c):
            if lab["level"] not in (None, "public"):
                exp_flags.add("lab_record_withheld")
            linked = {as_dict(c["hypothesis"]).get("lab_event")}
            for s_ in as_list(c.get("support")):
                ev = by_id(as_dict(s_).get("evidence"), "evidence") if as_dict(s_).get("evidence") else None
                if ev is not None:
                    linked |= set(as_list(ev.get("lab_runs")))
            if linked & lab["withheld"]:
                exp_flags.add("lab_events_withheld")
            if c.get("planning") in ("confirmatory", "confirmatory_with_deviations") and c.get("planned_at_anchor") == "none":
                exp_flags.add("unanchored_plan")
        r.computed[cid] = {"verifiability": exp_v, "verifiability_flags": sorted(exp_flags)}
        for n in notes:
            r.warn("W-verif", f"claims:{cid}", n)
        if published or "verifiability" in c:
            got_v = c.get("verifiability")
            if got_v != exp_v:
                r.err("13", f"claims:{cid}.verifiability", f"expected '{exp_v}', found '{got_v}'")
        if published or "verifiability_flags" in c:
            got_flags = set(as_list(c.get("verifiability_flags")))
            if got_flags != exp_flags:
                r.err("13", f"claims:{cid}.verifiability_flags",
                      f"expected {sorted(exp_flags)}, found {sorted(got_flags)}")
    return r


def is_assumption(c):
    return any(as_dict(s).get("kind") == "assumption" for s in as_list(c.get("support")))


def check_availability(r, where, av):
    if av is None:
        return
    av = as_dict(av)
    check_enum(r, "12", f"{where}.availability.level", av.get("level"), "level", required=True)
    lvl = av.get("level")
    if lvl and lvl != "public":
        if not av.get("reason"):
            r.err("12", f"{where}.availability", "reason required unless level is public")
        else:
            check_enum(r, "12", f"{where}.availability.reason", av.get("reason"), "reason")
    if lvl in ("platform", "audit", "author_only") and not av.get("content_hash"):
        r.err("12", f"{where}.availability", f"content_hash required for level {lvl}")


def text_checks(r, where, text, ids):
    if not isinstance(text, str):
        return
    for m in TERM_RE.finditer(text):
        if ids.get(m.group(1), (None,))[0] != "definitions":
            r.err("1", where, f"defined-term marker refers to unknown definition {m.group(1)}")
    if INSTRUCTION_RE.search(text):
        r.err("safe-ingest", where, "text looks like an instruction addressed to a reader or model; remove it")


def statement_checks(r, where, text):
    if not isinstance(text, str) or not text.strip():
        r.err("1", where, "missing statement")
        return
    if ANAPHORA_RE.search(text):
        r.warn("S3", where, "possible cross-reference or anaphora; statements must be self-contained")
    if EVALUATIVE_RE.search(text):
        r.warn("S7", where, "possible evaluative rhetoric (e.g. 'novel', 'significant'); state the finding plainly")
    if len(re.findall(r"[.;]\s+[A-Z]", text)) >= 1:
        r.warn("S1", where, "statement may contain more than one sentence or assertion")


def model_key(m, fallback):
    if m is None:
        return fallback
    if m.get("identifier"):
        return ("id", str(m["identifier"]).lower())
    return ("pnv", str(m.get("provider", "")).lower(), str(m.get("name", "")).lower(), str(m.get("version", "")).lower())


NUM_RE = re.compile(r"(?<![\w.^_{])([-+]?\d+(?:\.(\d+))?)(?:[eE]([-+]?\d+))?(\s*%|\s+percentage points?|\s+per ?cent)?")


def number_matches(value, statement):
    """True if value appears in statement at the precision written there (W12 heuristic)."""
    if value == 0 and re.search(r"\b(no|zero|none)\b", statement, re.I):
        return True
    for m in NUM_RE.finditer(statement):
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        dec = len(m.group(2) or "")
        exp = int(m.group(3)) if m.group(3) else 0
        tv = t * (10 ** exp)
        tol = 0.5 * 10 ** (exp - dec)
        if m.group(4):
            tv, tol = tv / 100, tol / 100
        for v in (value, abs(value)):
            if abs(v - tv) <= tol * (1 + 1e-9) + 1e-15:
                return True
    return False


TEST_PARAMETERS = {
    "frequentist": ("alpha", "test", "sides", "correction", "sample"),
    "equivalence": ("alpha", "test", "lower_bound", "upper_bound", "sample"),
    "bayesian": ("prior", "threshold", "sample"),
}


def check_test(r, test, w):
    """Rule 47: a confirmatory claim carries the decision rule it was planned with."""
    if not isinstance(test, dict):
        r.err("47", w, "required for confirmatory claims: the decision rule fixed before the test")
        return
    fw = test.get("framework")
    check_enum(r, "47", f"{w}.framework", fw, "framework", required=True)
    for outcome in ("supported", "not_supported"):
        if not isinstance(test.get(outcome), str) or not test[outcome].strip():
            r.err("47", f"{w}.{outcome}", "the rule for this outcome is required")
    params = test.get("parameters") if isinstance(test.get("parameters"), dict) else {}
    for key in TEST_PARAMETERS.get(fw, ()):
        if params.get(key) in (None, ""):
            r.err("47", f"{w}.parameters.{key}", f"required by the {fw} framework")
    check_enum(r, "47", f"{w}.verdict", test.get("verdict"), "verdict", required=True)
    for i, f in enumerate(as_list(test.get("flags"))):
        check_enum(r, "47", f"{w}.flags[{i}]", f, "test_flag", required=True)
    if test.get("verdict") == "computed" and not test.get("quantities"):
        r.err("47", f"{w}.quantities", "a computed verdict needs the quantities the rules were evaluated on")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_tables(r, w, e, ref):
    """Rule 48: result tables (spec §3.4)."""
    tables = e.get("tables")
    if tables is None:
        return
    if not isinstance(tables, list):
        r.err("48", f"{w}.tables", "must be a list")
        return
    seen = set()
    for i, t in enumerate(tables):
        tw = f"{w}.tables[{i}]"
        if not isinstance(t, dict):
            r.err("48", tw, "must be an object")
            continue
        tid = t.get("id")
        if not isinstance(tid, str) or not re.match(r"^T\d+$", tid):
            r.err("48", tw, "table id must look like T1, T2, …")
        elif tid in seen:
            r.err("48", tw, f"duplicate table id {tid}")
        else:
            seen.add(tid)
        if not isinstance(t.get("description"), str) or not t.get("description").strip():
            r.err("48", tw, "missing description")
        if "conditions" in t and not isinstance(t["conditions"], dict):
            r.err("48", f"{tw}.conditions", "must be an object")
        labels, cols = t.get("row_labels"), t.get("columns")
        if not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels):
            r.err("48", f"{tw}.row_labels", "must be a list of names (possibly empty)")
            labels = None
        if not isinstance(cols, list) or not cols or not all(isinstance(x, str) and x for x in cols):
            r.err("48", f"{tw}.columns", "must be a non-empty list of names")
            cols = None
        elif len(set(cols)) != len(cols):
            r.err("48", f"{tw}.columns", "column names must be unique")
        if cols is not None and "units" in t:
            u = t["units"]
            if not isinstance(u, list) or len(u) != len(cols) or not all(x is None or isinstance(x, str) for x in u):
                r.err("48", f"{tw}.units", "must have one unit (or null) per column")
        rows = t.get("rows")
        if not isinstance(rows, list) or not rows:
            r.err("48", f"{tw}.rows", "must be a non-empty list")
            rows = []
        seen_rows, any_number = set(), False
        for j, row in enumerate(rows):
            rw = f"{tw}.rows[{j}]"
            if not isinstance(row, dict):
                r.err("48", rw, "must be an object")
                continue
            rid = row.get("id")
            if not isinstance(rid, str) or not re.match(r"^r\d+$", rid):
                r.err("48", rw, "row id must look like r1, r2, …")
            elif rid in seen_rows:
                r.err("48", rw, f"duplicate row id {rid}")
            else:
                seen_rows.add(rid)
            lab = row.get("label")
            if labels is not None and (not isinstance(lab, list) or len(lab) != len(labels)
                                       or not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in lab)):
                r.err("48", f"{rw}.label", f"must have one entry per row label ({len(labels)})")
            vals = row.get("values")
            if cols is not None and (not isinstance(vals, list) or len(vals) != len(cols)):
                r.err("48", f"{rw}.values", f"must have one value per column ({len(cols)})")
            elif isinstance(vals, list):
                for v in vals:
                    if v is None:
                        continue
                    if _is_number(v):
                        any_number = True
                    else:
                        r.err("48", f"{rw}.values", f"'{v}' is not a number or null")
        if rows and not any_number:
            r.err("48", tw, "the table has no numeric value")
        if t.get("artefact"):
            ref("48", f"{tw}.artefact", t["artefact"], "artefacts")
        if "source_spans" in t and not (isinstance(t["source_spans"], list)
                                        and all(isinstance(x, str) and x for x in t["source_spans"])):
            r.err("48", f"{tw}.source_spans", "must be a list of non-empty strings")


def result_ref_resolves(ref_id, evidence_id, e):
    """Rule 38: E<n>.R<m>, E<n>.T<m> or E<n>.T<m>.r<k> within the evidence a support entry points to."""
    if e is None:
        return False
    m = re.match(r"^(E\d+)\.(R\d+|T\d+)(?:\.(r\d+))?$", ref_id)
    if not m or m.group(1) != evidence_id:
        return False
    if m.group(2).startswith("R"):
        return m.group(3) is None and m.group(2) in {as_dict(x).get("id") for x in as_list(e.get("results"))}
    table = next((t for t in as_list(e.get("tables")) if isinstance(t, dict) and t.get("id") == m.group(2)), None)
    if table is None:
        return False
    return m.group(3) is None or m.group(3) in {as_dict(x).get("id") for x in as_list(table.get("rows"))}


def validate_claim(r, c, cid, ids, by_id, ref, converted, author_models=frozenset()):
    w = f"claims:{cid}"
    statement_checks(r, f"{w}.statement", c.get("statement"))
    text_checks(r, f"{w}.statement", c.get("statement"), ids)
    check_enum(r, "1", f"{w}.type", c.get("type"), "claim_type", required=True)
    if c.get("role") == "background" or c.get("novelty") == "restatement":
        r.err("4", w, "claims are the paper's own (removed in v0.16: background, restatement); "
                      "cite prior work as support, relate to it, or define borrowed terms (spec §3.3.4)")
    else:
        check_enum(r, "1", f"{w}.role", c.get("role"), "role", required=True)
        check_enum(r, "1", f"{w}.novelty", c.get("novelty"), "novelty", required=True)
    if c.get("provenance") is not None:
        check_enum(r, "1", f"{w}.provenance", c.get("provenance"), "provenance")
    if c.get("verification") is not None:
        check_enum(r, "1", f"{w}.verification", c.get("verification"), "verification")
    if c.get("verifiability") is not None:
        check_enum(r, "1", f"{w}.verifiability", c.get("verifiability"), "verifiability")
    for f in as_list(c.get("verifiability_flags")):
        check_enum(r, "1", f"{w}.verifiability_flags", f, "flag")

    support = [as_dict(s) for s in as_list(c.get("support"))]
    if not support:
        r.err("2", w, "at least one support entry required")
    kinds = [s.get("kind") for s in support]
    for i, s in enumerate(support):
        sw = f"{w}.support[{i}]"
        k = s.get("kind")
        check_enum(r, "10", f"{sw}.kind", k, "support_kind", required=True)
        for aid in as_list(s.get("artefacts")):
            ref("10", f"{sw}.artefacts", aid, "artefacts")
        profile_for = {"empirical_quant": "quant", "empirical_qual": "qual",
                       "simulation": "simulation", "formal_proof": "formal"}
        if k in profile_for:
            e = ref("10", f"{sw}.evidence", s.get("evidence"), "evidence")
            if e is not None and e.get("profile") != profile_for[k]:
                r.err("10", f"{sw}.evidence", f"{k} must point to evidence with profile {profile_for[k]}")
            if k == "simulation" and e is not None and not (as_list(s.get("artefacts")) or as_list(e.get("artefacts"))):
                r.err("10", sw, "simulation support requires an artefact")
            if s.get("results"):
                if k not in ("empirical_quant", "simulation"):
                    r.err("38", f"{sw}.results", "results references only on empirical_quant and simulation support")
                for ref_id in as_list(s.get("results")):
                    if not result_ref_resolves(str(ref_id), s.get("evidence"), e):
                        r.err("38", f"{sw}.results", f"'{ref_id}' does not resolve to a result, table or table row of {s.get('evidence')}")
        elif k == "demonstration":
            if not as_list(s.get("artefacts")):
                r.err("10", sw, "demonstration support requires artefacts")
        elif k == "argument":
            ref("10", f"{sw}.warrant", s.get("warrant"), "warrants")
        elif k == "cited" and "reference" in s:
            ref("51", f"{sw}.reference", s.get("reference"), "references")
        elif k == "cited":
            if not s.get("target"):
                r.err("10", sw, "cited support requires reference, or target with id_type")
            check_enum(r, "10", f"{sw}.id_type", s.get("id_type"), "id_type", required=True)
            if s.get("id_type") == "url" and not s.get("archive_uri"):
                r.err("10", sw, "url citations require archive_uri")
            if s.get("id_type") == "reference" and not converted:
                r.warn("W14", f"{sw}.id_type", "a printed reference without identifier: give the cited work's identifier (doi, arxiv, …)")
            if "self" in s and not isinstance(s.get("self"), bool):
                r.err("10", f"{sw}.self", "must be true or false")
        elif k == "expert_judgement":
            if not s.get("holder"):
                r.err("10", sw, "expert_judgement requires holder")
        elif k == "unstated":
            if not converted:
                r.err("22", sw, "unstated support is allowed only in converted papers")

    if "assumption" in kinds:
        if len(support) != 1:
            r.err("3", w, "assumption must be the sole support entry of its claim")
        if c.get("assessments"):
            r.err("3", w, "assumption claims carry no strength assessments")
        check_enum(r, "3", f"{w}.assumption_category", c.get("assumption_category"), "assumption_category", required=True)
        check_enum(r, "3", f"{w}.check", c.get("check"), "check", required=True)
        if c.get("assumption_category") == "axiom" and c.get("check") != "untestable":
            r.err("3", w, "axiom assumptions must have check: untestable")
        if c.get("check") in ("tested", "sensitivity_analysed"):
            ref("3", f"{w}.check_evidence", c.get("check_evidence"), "evidence")
    if "unstated" in kinds and len(support) != 1:
        r.err("22", w, "unstated must be the sole support entry of its claim")


    if c.get("type") == "interpretive":
        if not c.get("warrant"):
            r.err("6", w, "interpretive claims require a warrant")
    if c.get("warrant"):
        ref("6", f"{w}.warrant", c.get("warrant"), "warrants")

    if "formal_proof" in kinds:
        if not c.get("verification"):
            r.err("7", w, "formal_proof support requires verification")
        for s in support:
            if s.get("kind") == "formal_proof":
                e = by_id(s.get("evidence"), "evidence")
                if e is not None:
                    if not e.get("foundation"):
                        r.err("7", f"evidence:{e.get('id')}", "formal evidence requires foundation")
                    if not (e.get("steps") or e.get("proof_artefact") or (converted and e.get("source_spans"))):
                        r.err("7", f"evidence:{e.get('id')}", "formal evidence requires steps or proof_artefact"
                              + (", or source_spans locating the derivation in the source" if converted else ""))

    for aid in as_list(c.get("assumptions")):
        a = ref("1", f"{w}.assumptions", aid, "claims")
        if a is not None and not is_assumption(a):
            r.err("3", f"{w}.assumptions", f"{aid} is not an assumption claim")
    for lid in as_list(c.get("limitations")):
        ref("11", f"{w}.limitations", lid, "limitations")
    for d in as_list(c.get("depends_on")):
        ref("1", f"{w}.depends_on", d, "claims")
    rel_types = []
    for i, rel in enumerate(as_list(c.get("relations"))):
        rel = as_dict(rel)
        rw = f"{w}.relations[{i}]"
        check_enum(r, "34", f"{rw}.type", rel.get("type"), "relation", required=True)
        rel_types.append(rel.get("type"))
        if "note" in rel and not (isinstance(rel["note"], str) and rel["note"].strip()):
            r.err("34", f"{rw}.note", "must be a non-empty sentence")
        if "reference" in rel:
            ref("51", f"{rw}.reference", rel.get("reference"), "references")
            continue
        if not rel.get("target"):
            r.err("34", rw, "missing reference, or target with id_type")
        check_enum(r, "34", f"{rw}.id_type", rel.get("id_type"), "rel_id_type", required=True)
        if rel.get("id_type") == "url" and not rel.get("archive_uri"):
            r.err("34", rw, "url targets require archive_uri")
        if "self" in rel and not isinstance(rel.get("self"), bool):
            r.err("34", f"{rw}.self", "must be true or false")
    nov = c.get("novelty")
    if nov == "refutation" and not ({"contradicts", "fails_to_replicate"} & set(rel_types)):
        r.err("33", w, "a refutation needs a relation of type contradicts or fails_to_replicate naming what it refutes")
    if nov == "replication" and not ({"replicates", "fails_to_replicate"} & set(rel_types)):
        r.err("33", w, "a replication needs a relation of type replicates or fails_to_replicate naming what it replicates")
    if nov == "extension" and "extends" not in rel_types:
        r.warn("W7", w, "extension without an extends relation; name the work it extends")

    if c.get("type") == "formal":
        if not c.get("verification") and ({"formal_proof", "argument"} & set(kinds)):
            r.err("35", w, "formal claims proved or argued in the paper require verification")
        if c.get("assessments"):
            r.err("35", w, "formal claims carry verification, not strength assessments; remove assessments")
    hyp = c.get("hypothesis")
    if hyp is not None:
        hyp = as_dict(hyp)
        if not hyp.get("lab_event"):
            r.err("42", f"{w}.hypothesis.lab_event", "missing")
        check_enum(r, "42", f"{w}.hypothesis.outcome", hyp.get("outcome"), "outcome", required=True)
        if not c.get("planning"):
            r.err("42", f"{w}.planning", "required when the claim carries a hypothesis")
    if c.get("planning") is not None:
        check_enum(r, "42", f"{w}.planning", c.get("planning"), "planning")
        if c.get("planning") in ("confirmatory", "confirmatory_with_deviations"):
            if not c.get("planned_at"):
                r.err("43", f"{w}.planned_at", "required for confirmatory claims")
            check_enum(r, "43", f"{w}.planned_at_anchor", c.get("planned_at_anchor"), "anchor", required=True)
            check_test(r, c.get("test"), f"{w}.test")
        if c.get("planning") == "exploratory" and c.get("role") == "principal":
            r.warn("W13", w, "exploratory principal claim: say so in the assessment rationale")
    ver = c.get("verification")
    if "verified_by" in c:
        r.err("40", f"{w}.verified_by", "removed in v0.10: record the check in checks instead")
    checks = [as_dict(x) for x in as_list(c.get("checks"))]
    if checks and is_assumption(c):
        r.err("41", w, "checks are not used on assumption claims")
    confirmed_full = {"agent": False, "human": False}
    unresolved_dispute = False
    for i, ck in enumerate(checks):
        kw = f"{w}.checks[{i}]"
        check_enum(r, "39", f"{kw}.outcome", ck.get("outcome"), "check_outcome", required=True)
        check_enum(r, "39", f"{kw}.extent", ck.get("extent"), "check_extent", required=True)
        p = by_id(ck.get("by"), "contributors")
        if p is None:
            r.err("39", f"{kw}.by", "must name a contributor")
            continue
        if "author" in as_list(p.get("positions")):
            r.err("39", f"{kw}.by", "a check must be made by someone who is not an author")
        if ck.get("outcome") == "disputed":
            if not ck.get("notes"):
                r.err("39", kw, "a disputed check needs notes saying what failed")
            if not ck.get("resolution"):
                unresolved_dispute = True
        if ck.get("outcome") == "confirmed" and ck.get("extent") == "full" and p.get("type") in confirmed_full:
            confirmed_full[p["type"]] = True
        if p.get("type") == "agent":
            for m in as_list(p.get("models")):
                k = model_key(by_id(as_dict(m).get("model"), "models"), as_dict(m).get("model"))
                if k in author_models:
                    r.warn("W11", kw, f"checker {p.get('id')} runs on the same model as an author agent; "
                                      "a different model would make the check more independent")
                    break
    if ver is not None:
        expected_ok = {
            "agent_checked": confirmed_full["agent"] and not unresolved_dispute,
            "peer_checked": confirmed_full["human"] and not unresolved_dispute,
            "disputed": unresolved_dispute,
            "unchecked": not (confirmed_full["agent"] or confirmed_full["human"]) and not unresolved_dispute,
        }
        if ver in expected_ok and not expected_ok[ver]:
            r.err("36", f"{w}.verification", f"'{ver}' is inconsistent with the recorded checks (§5)")

    # numbers in the statement vs cited results (W12)
    stmt = c.get("statement") or ""
    for s in support:
        e = by_id(s.get("evidence"), "evidence") if s.get("evidence") else None
        if e is None:
            continue
        rmap = {as_dict(x).get("id"): as_dict(x) for x in as_list(e.get("results"))}
        for ref_id in as_list(s.get("results")):
            rid = str(ref_id).split(".")[-1]
            val = rmap.get(rid, {}).get("value")
            if isinstance(val, (int, float)) and not isinstance(val, bool) and not number_matches(val, stmt):
                r.warn("W12", f"{w}.statement", f"cited result {ref_id} = {val:.6g} does not appear in the statement at the stated precision; check the number, or cite only the results the statement reports")
    if ver == "machine_checked":
        fe = [by_id(s.get("evidence"), "evidence") for s in support if s.get("kind") == "formal_proof"]
        if not any(e is not None and e.get("proof_artefact") and e.get("checker") for e in fe):
            r.err("36", w, "machine_checked requires formal evidence with proof_artefact and checker")

    provs = [c.get("provenance")] + list(as_dict(c.get("field_provenance")).values())
    for p in provs[1:]:
        check_enum(r, "1", f"{w}.field_provenance", p, "provenance")
    if not converted and any(p in ("extracted", "inferred") for p in provs):
        r.err("9", w, "extracted/inferred provenance only in converted papers")
    if converted and not as_list(c.get("source_spans")):
        r.err("9", w, "converted papers require source_spans on every claim")

    for i, a in enumerate(as_list(c.get("assessments"))):
        a = as_dict(a)
        aw = f"{w}.assessments[{i}]"
        if a.get("by") == "converter" or (converted and a.get("by") == "reviewer"):
            r.err("53", f"{aw}.by", "a converted paper records only strengths its source states (by: author, with "
                                    "source_span); judgements of the conversion are platform assessments")
        else:
            check_enum(r, "1", f"{aw}.by", a.get("by"), "by", required=True)
        check_enum(r, "1", f"{aw}.strength", a.get("strength"), "strength", required=True)
        if a.get("by") == "reviewer":
            check_enum(r, "1", f"{aw}.reviewer_type", a.get("reviewer_type"), "reviewer_type", required=True)
        if converted and a.get("by") == "author" and not a.get("source_span"):
            r.err("23", aw, "in converted papers, by: author assessments require source_span")


def compute_verifiability(c, by_id):
    """Return (verifiability, flags, notes) per spec section 5."""
    notes = []
    support = [as_dict(s) for s in as_list(c.get("support"))]
    kinds = [s.get("kind") for s in support]
    flags = set()
    if "assumption" in kinds:
        return "not_applicable", flags, notes
    if "unstated" in kinds:
        return "unverifiable", {"unstated_support"}, notes

    levels = []
    artefact_ids = []
    for s in support:
        artefact_ids += as_list(s.get("artefacts"))
        e = by_id(s.get("evidence"), "evidence") if s.get("evidence") else None
        if e is not None:
            av = as_dict(e.get("availability"))
            levels.append(av.get("level"))
            artefact_ids += as_list(e.get("artefacts"))
            if e.get("proof_artefact"):
                artefact_ids.append(e["proof_artefact"])
            for dm in as_list(e.get("models")):
                dm = as_dict(dm)
                m = by_id(dm.get("model"), "models")
                if m is not None:
                    if m.get("weights") == "api_only":
                        flags.add("api_model_dependency")
                    elif m.get("weights") is None:
                        notes.append(f"model {m.get('id')} has no weights value; api_model_dependency "
                                     "cannot be determined (the platform will derive it)")
                if not dm.get("prompts"):
                    flags.add("unpublished_prompts")
                if not dm.get("settings") or dm.get("record_source") in (None, "self_reported"):
                    flags.add("unrecorded_model_settings")
    seen = set()
    queue = list(artefact_ids)
    while queue:
        aid = queue.pop()
        if aid in seen:
            continue
        seen.add(aid)
        a = by_id(aid, "artefacts")
        if a is None:
            continue
        levels.append(as_dict(a.get("availability")).get("level"))
        if a.get("kind") == "figure":
            if not a.get("underlying_data"):
                flags.add("figure_without_data")
            else:
                queue.append(a["underlying_data"])

    if any(l == "unavailable" for l in levels):
        return "unverifiable", flags, notes
    if kinds and all(k == "expert_judgement" for k in kinds):
        return "unverifiable", flags, notes
    if any(l in ("audit", "author_only") for l in levels):
        return "audit_only", flags, notes
    if any(l is None for l in levels):
        notes.append("some supporting evidence or artefacts state no availability; treated as public")
    return "open", flags, notes


def validate_contributors(r, contributors, paper, ref, converted):
    positions_all = []
    for p in contributors:
        if not isinstance(p, dict):
            continue
        w = f"contributors:{p.get('id')}"
        t = p.get("type")
        check_enum(r, "19", f"{w}.type", t, "contributor_type", required=True)
        pos = as_list(p.get("positions"))
        if not pos:
            r.err("19", f"{w}.positions", "at least one position required")
        for x in pos:
            check_enum(r, "1", f"{w}.positions", x, "position")
            positions_all.append((x, p))
        for cr in as_list(p.get("credit_roles")):
            check_enum(r, "1", f"{w}.credit_roles", cr, "credit")
        if not p.get("credit_roles"):
            r.warn("W1", f"{w}.credit_roles", "recommended field missing")

        if t in ("human", "organisation") and not p.get("name"):
            r.err("19", f"{w}.name", "required for humans and organisations")
        if t == "agent" and not p.get("name"):
            r.warn("W1", f"{w}.name", "recommended for agents")
        if t == "human":
            orcid = p.get("orcid")
            if orcid and not ORCID_RE.match(str(orcid)):
                r.err("14", f"{w}.orcid", "must be an ORCID iD like 0000-0002-1825-0097")
            if not orcid and ("guarantor" in pos or "conversion_guarantor" in pos):
                r.err("14", f"{w}.orcid", "required for human guarantors")
            elif not orcid:
                r.warn("W1", f"{w}.orcid", "recommended field missing")
        if t == "organisation" and not (p.get("ror") or p.get("registration")):
            r.warn("W1", w, "ror or registration recommended for organisations")

        if ("guarantor" in pos or "conversion_guarantor" in pos) and t == "agent":
            r.err("14", w, "guarantors must be humans or organisations")
        if ("author" in pos or "converter" in pos) and t != "agent":
            r.err("15", w, "author and converter positions are held only by agents")
        if "supervisor" in pos and t != "human":
            r.err("15", w, "supervisor is held only by humans")

        if t == "agent":
            models = [as_dict(m) for m in as_list(p.get("models"))]
            if not models:
                r.err("18", f"{w}.models", "agents must list at least one model")
            roles = []
            for i, m in enumerate(models):
                ref("18", f"{w}.models[{i}].model", m.get("model"), "models")
                if m.get("role") is not None:
                    check_enum(r, "20", f"{w}.models[{i}].role", m.get("role"), "agent_model_role")
                    roles.append(m.get("role"))
            if "sole" in roles and len(models) != 1:
                r.err("20", w, "role sole requires exactly one model")
            if roles.count("orchestrator") > 1:
                r.err("20", w, "at most one orchestrator per agent")
            for i, tk in enumerate(as_list(p.get("ai_tasks"))):
                tk = as_dict(tk)
                check_enum(r, "1", f"{w}.ai_tasks[{i}].task", tk.get("task"), "task", required=True)
                check_enum(r, "1", f"{w}.ai_tasks[{i}].extent", tk.get("extent"), "extent", required=True)
            if "author" not in pos and any(as_dict(tk).get("task") == "writing" for tk in as_list(p.get("ai_tasks"))):
                r.warn("W10", w, "this agent has the task writing but is not an author; agents that produced the content are authors")
            for tool in as_list(p.get("tools")):
                check_enum(r, "1", f"{w}.tools", tool, "tool")
            for f in ("ai_tasks", "tools"):
                if not p.get(f):
                    r.warn("W1", f"{w}.{f}", "recommended field missing")
            rs = p.get("record_source")
            if rs is not None:
                check_enum(r, "1", f"{w}.record_source", rs, "record_source")
            if rs in (None, "self_reported"):
                r.warn("W2", f"{w}.record_source", "absent or self_reported; details are the agent's own account")
            if p.get("trace"):
                a = ref("21", f"{w}.trace", p["trace"], "artefacts")
                if a is not None and a.get("kind") != "trace":
                    r.err("21", f"{w}.trace", "must point to an artefact of kind trace")

    def has(position):
        return [p for x, p in positions_all if x == position]

    origin = paper.get("origin")
    if origin == "native":
        if not has("author"):
            r.err("16", "contributors", "native papers need at least one author (an agent)")
        if not has("guarantor"):
            r.err("16", "contributors", "native papers need at least one guarantor (ask the responsible human)")
        if has("conversion_guarantor"):
            r.err("16", "contributors", "conversion_guarantor appears only in converted papers")
        if "autonomy" not in paper:
            r.err("16", "paper.autonomy", "required for native papers")
    elif origin == "converted":
        if len(has("converter")) != 1:
            r.err("17", "contributors", "converted papers need exactly one converter")
        if len(has("conversion_guarantor")) != 1:
            r.err("17", "contributors", "converted papers need exactly one conversion_guarantor")
        if "autonomy" in paper:
            r.err("17", "paper.autonomy", "not used for converted papers")
        cf = as_dict(paper.get("converted_from"))
        for f in ("source_id", "id_type", "source_version", "source_licence", "source_authors",
                  "source_status_at_conversion", "converter", "date"):
            if f not in cf:
                r.err("17", f"paper.converted_from.{f}", "missing required field")
        source_orcids = {as_dict(a).get("orcid") for a in as_list(cf.get("source_authors"))} - {None}
        for g in has("guarantor"):
            if g.get("orcid") not in source_orcids:
                r.err("17", f"contributors:{g.get('id')}", "in converted papers a guarantor must be a source author (by ORCID)")
        if cf.get("source_id") and cf.get("source_id") == paper.get("id"):
            r.err("30", "paper.id", "must differ from converted_from.source_id")
        src, lic = cf.get("source_licence"), paper.get("licence")
        if src and src not in CONVERSION_LICENCES:
            r.err("31", "paper.converted_from.source_licence",
                  f"{src} does not permit conversion (it must permit derivatives; see spec §3.1)")
        elif src and lic and lic not in CONVERSION_LICENCES[src]:
            r.err("31", "paper.licence", f"{lic} is not allowed for a claimset of a {src} source; "
                                         f"use one of {', '.join(sorted(CONVERSION_LICENCES[src]))}")

    cr = paper.get("contribution_record")
    if cr in ("partial", "unknown"):
        r.warn("W3", "paper.contribution_record", f"contribution record is {cr}")
    humans = [p for p in contributors if isinstance(p, dict) and p.get("type") == "human"]
    if paper.get("autonomy") == "human_supervised" and not has("supervisor"):
        r.warn("W4", "paper.autonomy", "human_supervised but no supervisor listed")
    if paper.get("autonomy") == "human_assisted" and not any("contributor" in as_list(h.get("positions")) for h in humans):
        r.warn("W4", "paper.autonomy", "human_assisted but no human contributor listed")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {path}: {exc}")
        return 1
    published = "--published" in argv
    rep = validate(doc, published=published, local_root=claimset_folder(path))
    print_report(rep, published=published, as_json="--json" in argv)
    return 1 if rep.errors else 0


def print_report(rep, published=False, as_json=False):
    """Print a validation report as text or JSON."""
    mode = "published file" if published else "submission"
    if as_json:
        print(json.dumps({"mode": mode, "valid": not rep.errors, "errors": rep.errors,
                          "warnings": rep.warnings, "computed": rep.computed}, indent=2))
    else:
        print(f"{'VALID' if not rep.errors else 'INVALID'} {mode}: "
              f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        for e in rep.errors:
            print(f"  ERROR   [rule {e['rule']}] {e['where']}: {e['message']}")
        for w in rep.warnings:
            print(f"  WARNING [{w['rule']}] {w['where']}: {w['message']}")
        if rep.computed:
            print("  Verifiability the platform will compute:")
            for cid, v in rep.computed.items():
                flags = ", ".join(v["verifiability_flags"]) or "no flags"
                print(f"    {cid}: {v['verifiability']} ({flags})")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
