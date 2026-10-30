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

SCHEMA_VERSION = "0.34"

ENUM = {
    "status": {"active", "corrected", "retracted", "withdrawn", "expression_of_concern"},
    "origin": {"native", "converted"},
    "autonomy": {"autonomous", "human_supervised", "human_assisted", "unknown"},
    "contribution_record": {"complete", "partial", "unknown"},
    "claim_type": {"descriptive", "associational", "causal", "predictive", "interpretive", "formal", "methodological"},
    "role": {"principal", "supporting", "secondary"},
    "novelty": {"no_prior_found", "replication", "extension", "refutation"},
    "provenance": {"author_stated", "extracted", "inferred"},
    "verification": {"machine_checked", "agent_checked", "peer_checked", "disputed", "unchecked"},
    "check_outcome": {"confirmed", "disputed", "inconclusive"},
    "check_extent": {"full", "partial"},
    "verifiability": {"open", "audit_only", "unverifiable", "not_applicable"},
    "flag": {"api_model_dependency", "figure_without_data", "unpublished_prompts",
             "unrecorded_model_settings", "unstated_support", "lab_record_withheld",
             "lab_events_withheld", "unanchored_plan", "availability_not_stated"},
    "outcome": {"supported", "not_supported", "inconclusive"},
    "planning": {"confirmatory", "confirmatory_with_deviations", "exploratory"},
    "anchor": {"platform", "rfc3161", "git_remote", "none"},
    "framework": {"frequentist", "equivalence", "bayesian", "deterministic", "formal", "qualitative"},
    "verdict": {"computed", "judged"},
    "test_flag": {"code_changed_after_plan", "holdout_used_elsewhere", "reproduces_prompting_result"},
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
                      "prompt", "trace", "manuscript", "lab_record", "instrument", "stimulus", "codebook",
                      "recording", "other"},
    "figure_function": {"data_plot", "specimen", "diagram"},
    "derived_how": {"reanalysis", "subset", "pooled", "extension"},
    "ethics_decision": {"approved", "exempt", "not_required"},
    "relation": {"supports", "contradicts", "extends", "refines", "replicates", "fails_to_replicate", "supersedes",
                 "addresses", "explains", "qualifies"},
    "limitation_category": {"sampling", "sample_size", "measurement", "confounding", "data_quality",
                            "model_specification", "analytic_flexibility", "researcher_position",
                            "computational", "other"},
    "impact": {"could_reverse", "could_bias_magnitude", "reduces_precision", "reduces_transferability"},
    "mitigation": {"none", "partial", "addressed"},
    "level": {"public", "platform", "audit", "author_only", "unavailable", "not_stated"},
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
    "id_type": {"doi", "arxiv", "handle", "pmid", "isbn", "url", "curie", "physical", "reference", "claimset_local"},
    "rel_id_type": {"claimset", "doi", "arxiv", "handle", "pmid", "isbn", "url", "curie", "physical", "claimset_local",
                    "local"},
    "by": {"author", "reviewer"},
    "reviewer_type": {"ai", "human"},
    "strength": {"high", "moderate", "low", "very_low"},
    "definition_kind": {"method", "intervention", "population", "data", "setting", "design", "variable", "concept",
                        "theory", "other"},
}

PREFIX = {"definitions": "D", "claims": "C", "evidence": "E", "warrants": "W",
          "limitations": "L", "artefacts": "A", "contributors": "P", "models": "M", "references": "B",
          "open_questions": "O"}

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TERM_RE = re.compile(r"\[\[(D\d+):[^\]]+\]\]")
NESTED_MARK_RE = re.compile(r"\[\[D\d+:[^\]]*\[\[")

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
EVALUATIVE_RE = re.compile(r"\b(novel|(?<!statistically )(?<!statistically in)significant(ly)?|remarkabl[ey]|groundbreaking|state-of-the-art|"
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


PLATFORM_PAPER_FIELDS = ("version", "previous_version", "changes", "date", "status", "status_notice", "answers")

# Global IDs of review comments and responses (revision spec §2.2; rule 59)
COMMENT_ID_RE = re.compile(r"^\S+@v[1-9]\d*/RV[1-9]\d*\.c[1-9]\d*$")
RESPONSE_ID_RE = re.compile(r"^(\S+)@v([1-9]\d*)/RP[1-9]\d*$")


NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,59}$")
LOCAL_RE = re.compile(r"^([a-z][a-z0-9-]{1,59})/([a-z][a-z0-9-]*)\.claimset\.json(?:#(C\d+))?$")
SOURCE_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|https?://arxiv\.org/abs/|doi:|arxiv:)", re.I)


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


def source_name(converted_from):
    """The name of a converted paper's files, made from its source's identifier (§2.4), or None."""
    cf = as_dict(converted_from)
    kind, sid = cf.get("id_type"), cf.get("source_id")
    if not (isinstance(kind, str) and kind and isinstance(sid, str) and sid.strip()):
        return None
    sid = SOURCE_PREFIX_RE.sub("", sid.strip())
    version = cf.get("source_version")
    if kind == "arxiv" and isinstance(version, str) and version and not sid.lower().endswith(version.lower()):
        sid += version
    name = re.sub(r"[^a-z0-9]+", "-", f"{kind}-{sid}".lower()).strip("-")
    return name or None


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


GIT_HOSTS = re.compile(r"^https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|"
                       r"gitlab\.[\w.-]+|git\.[\w.-]+)/", re.I)
FULL_COMMIT = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")


def process_warnings(r, doc, claims, coll, by_id, converted):
    """Warnings W18, W20 and W21: duplicated evidence, unmitigated reversal risks of contributions, artefact
    links that may not last."""
    # W18: two evidence objects from the same code with the same results
    seen = {}
    for e in coll["evidence"]:
        if not isinstance(e, dict):
            continue
        code = as_dict(e.get("execution")).get("code_version")
        results = sorted(json.dumps([as_dict(x).get("variable"), as_dict(x).get("value"), as_dict(x).get("conditions")],
                                    sort_keys=True) for x in as_list(e.get("results")))
        if not code or not results:
            continue
        key = (code, tuple(results))
        if key in seen:
            r.warn("W18", f"evidence:{e.get('id')}", f"same code and the same results as {seen[key]}: one evidence "
                                                    f"object, cited by every claim it supports (and one plan for "
                                                    f"several hypotheses)")
        else:
            seen[key] = e.get("id")
    # W20: a contribution that a limitation could reverse, unmitigated, with its strength not calibrated to it
    if not converted:
        for cid, c in claims.items():
            if c.get("role") != "principal":
                continue
            risky = [lid for lid in as_list(c.get("limitations"))
                     if (by_id(lid, "limitations") or {}).get("impact") == "could_reverse"
                     and (by_id(lid, "limitations") or {}).get("mitigation") == "none"]
            authored = [as_dict(a) for a in as_list(c.get("assessments")) if as_dict(a).get("by") == "author"]
            if any(a.get("strength") in ("low", "very_low") for a in authored):
                continue
            rationale = " ".join(str(a.get("rationale") or "") for a in authored)
            missing = [lid for lid in risky if not re.search(rf"\b{re.escape(str(lid))}\b", rationale)]
            if missing:
                r.warn("W20", f"claims:{cid}", f"a contribution that {', '.join(missing)} could reverse, with no "
                                               f"mitigation: say in the strength rationale how that risk was weighed "
                                               f"(naming {', '.join(missing)}), or lower the strength")
    # W21: a public artefact on a git host whose link is not pinned to a commit
    for a in coll["artefacts"]:
        if not isinstance(a, dict):
            continue
        level = as_dict(a.get("availability")).get("level", "public")
        uri = a.get("uri") or as_dict(a.get("availability")).get("uri") or ""
        if level == "public" and isinstance(uri, str) and GIT_HOSTS.match(uri) and not FULL_COMMIT.search(uri):
            r.warn("W21", f"artefacts:{a.get('id')}.uri", "the link names a branch or a moving reference, not a "
                                                           "commit: what it shows will change; pin it to the full "
                                                           "commit hash")


def validate(doc, published=False, local_root=None, revision=False, file_name=None):
    """Check a claimset. `local_root`: the repository's claimset/ folder, when the file lies in one, so that
    local references to other papers can be resolved (rule 57); `file_name`: the file's name there (warning 31). `revision`: the file is a revision submission,
    which names the claimset it revises in paper.id (rule 25; revision spec §5.1)."""
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
        if "id" in paper and "correction_reason" not in paper and not revision:
            r.err("25", "paper.id", "the platform mints the DOI; remove paper.id (keep it only when submitting "
                                    "a correction, together with correction_reason, or a revision, with its "
                                    "response: `claimset revision check`)")
        if revision and "correction_reason" in paper:
            r.err("25", "paper.correction_reason", "a revision answers reviews with a response; a correction "
                                                   "is submitted on its own (revision spec §5.1)")
        if revision and not paper.get("id"):
            r.err("25", "paper.id", "a revision names the claimset it revises")
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
            for a in as_list(as_dict(ch).get("addresses")):
                if not (isinstance(a, str) and COMMENT_ID_RE.match(a)):
                    r.err("59", f"paper.changes[{i}].addresses", f"'{a}' is not a review comment's global ID "
                                                                 f"(<claimset>@v<n>/RV<k>.c<m>)")
    if "answers" in paper:
        m = RESPONSE_ID_RE.match(str(paper.get("answers")))
        if not m or m.group(1) != paper.get("id") or str(paper.get("version")) != m.group(2):
            r.err("59", "paper.answers", "must be the global ID of this version's response, "
                                         "<paper.id>@v<version>/RP<k>")
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
    defs_by_id = {d.get("id"): d for d in coll["definitions"] if isinstance(d, dict)}
    evidence_by_id = {e.get("id"): e for e in coll["evidence"] if isinstance(e, dict)}

    def rests_on(c):
        """Claims this claim rests on: depends_on, assumptions, and the assumptions of the evidence it cites."""
        out = list(as_list(c.get("depends_on"))) + list(as_list(c.get("assumptions")))
        for s in as_list(c.get("support")):
            e = by_id(as_dict(s).get("evidence"), "evidence")
            if e is not None:
                out += list(as_list(e.get("assumptions")))
        out += inherited_assumptions(c, defs_by_id, evidence_by_id)
        return [x for x in out if isinstance(x, str)]

    reached, stack = set(), [cid for cid, c in claims.items() if c.get("role") == "principal"]
    while stack:
        cur = stack.pop()
        for nxt in rests_on(claims.get(cur) or {}):
            if nxt in claims and nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    # W23: a claim the paper neither shows nor uses
    used = {x for c in claims.values() for x in rests_on(c)}
    for cid, c in claims.items():
        if c.get("role") == "principal" or cid in used or c.get("type") == "formal" or is_assumption(c):
            continue
        support = [as_dict(s) for s in as_list(c.get("support"))]
        if not support or any(s.get("warrant") for s in support) or \
                not {s.get("kind") for s in support} <= {"argument", "consensus", "expert_judgement", "unstated"}:
            continue
        words = rhetoric_cues(c.get("statement"))
        why = f"; it also uses {', '.join(words)}" if words else ""
        if converted:
            r.warn("W23", f"claims:{cid}", f"no evidence or citation supports it and no claim rests on it{why}: "
                                           f"check that the source's argument uses it; leave it out if not")
        else:
            r.warn("W23", f"claims:{cid}", f"no evidence or citation supports it and no claim rests on it{why}: "
                                           f"support it and link it, record it as an open question if it is one, "
                                           f"or leave it out")
    # W26: an own definition that a formal claim names but that has no mathematics (§3.2)
    marked_formal = set()
    for c in claims.values():
        if c.get("type") == "formal" and isinstance(c.get("statement"), str):
            marked_formal |= set(re.findall(r"\[\[(D\d+):", c["statement"]))
    for dfn in coll["definitions"]:
        dfn = as_dict(dfn)
        did, text = dfn.get("id"), dfn.get("definition")
        if did in marked_formal and not dfn.get("source") and isinstance(text, str) and "$" not in text \
                and "```" not in text:
            r.warn("W26", f"definitions:{did}", "a term of the paper's own that a formal claim names, with no "
                                              "mathematics: if it is the paper's method or model, give its "
                                              "defining equations as the paper writes them")
    # W25: a secondary result on which nothing rests (§3.3, Secondary claims in converted papers)
    named = set(used)
    for w in coll["warrants"]:
        named |= {x for x in as_list(as_dict(w).get("premises")) if isinstance(x, str)}
    for q in coll["open_questions"]:
        named |= {x for x in as_list(as_dict(q).get("claims")) if isinstance(x, str)}
    for cid, c in claims.items():                   # relations within the paper, either end (§3.7)
        for rel in as_list(c.get("relations")):
            rel = as_dict(rel)
            if rel.get("id_type") == "local" and isinstance(rel.get("target"), str):
                named |= {cid, rel["target"]}
    for cid, c in claims.items():
        if c.get("role") != "secondary" or c.get("type") != "descriptive" or cid in named:
            continue
        if any(as_dict(s).get("kind") in ("empirical_quant", "simulation") for s in as_list(c.get("support"))):
            r.warn("W25", f"claims:{cid}", "a secondary result on which no claim, warrant or open question rests, and that "
                                           "stands in no relation to another claim: keep it "
                                           "only if it states a loss, limit or null result of the paper's own method; "
                                           "otherwise the result table of its evidence holds it")
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

    check_frame(r, coll, claims, by_id)
    check_v032(r, paper, coll, claims, by_id, ids, converted, published)

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
        check_artefact_v032(r, w, a, ref, by_id, coll)
        if a.get("reference") is not None:
            ref("54", f"{w}.reference", a["reference"], "references")
        if "archive_uri" in a and not (isinstance(a["archive_uri"], str) and a["archive_uri"].strip()):
            r.err("1", f"{w}.archive_uri", "must be a persistent identifier or archive URI (a SWHID, DOI or archived URL)")
        if "availability" not in a:
            r.err("75", w, "every artefact has an availability; not_stated when the paper does not say")
        check_availability(r, w, a.get("availability"))
        cited = a.get("reference") and not is_relative_uri(a.get("uri"))     # another work's, described there
        if not a.get("licence") and not cited:
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

    # ---------- Open questions (rule 62) ----------
    for q in coll["open_questions"]:
        if not isinstance(q, dict):
            continue
        w = f"open_questions:{q.get('id')}"
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            r.err("62", f"{w}.question", "the question, as a self-contained sentence")
        else:
            text_checks(r, f"{w}.question", q["question"], ids)
        for cid in as_list(q.get("claims")):
            ref("62", f"{w}.claims", cid, "claims")
        extra = set(q) - {"id", "question", "claims", "source_spans"}
        if extra:
            r.err("62", w, f"an open question has only id, question, claims and source_spans, not {sorted(extra)}")

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
        if "setup" in e:
            setup = e["setup"]
            if not isinstance(setup, dict) or not setup:
                r.err("61", f"{w}.setup", "an object {studied, cases, procedure, lens, context} of lists of "
                                          "definition IDs")
            elif e.get("profile") == "formal":
                r.err("61", f"{w}.setup", "formal evidence has no setup: its foundation and assumptions say what "
                                          "the derivation rests on")
            else:
                for role, ids_ in setup.items():
                    if role in OLD_SETUP_ROLES:
                        r.err("61", f"{w}.setup.{role}", f"renamed in v0.30: {role} is now "
                                                         f"{OLD_SETUP_ROLES[role]} (`claimset migrate` renames it)")
                    elif role not in SETUP_ROLES:
                        r.err("61", f"{w}.setup.{role}", f"the roles are {', '.join(SETUP_ROLES)}")
                    elif not isinstance(ids_, list) or not ids_:
                        r.err("61", f"{w}.setup.{role}", "a non-empty list of definition IDs")
                    else:
                        for d in ids_:
                            ref("61", f"{w}.setup.{role}", d, "definitions")
        elif e.get("profile") in ENUM["profile"] - {"formal"}:
            r.err("61", w, "evidence other than formal has a setup: the parts of the study (what is studied, on what "
                           "cases, by what procedure, through what lens) as definitions (§3.4)")
        if "availability" not in e:
            r.err("75", w, "every evidence object has an availability; not_stated when the paper does not say")
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
            if e.get("estimate") is None and not as_list(e.get("results")) and not as_list(e.get("tables")):
                r.warn("W8", w, "no estimate and no results: add a structured results list for the numbers this evidence produced")
        if e.get("profile") != "formal":
            seen_r = set()
            for i, rs in enumerate(as_list(e.get("results"))):
                rs = as_dict(rs)
                rw = f"{w}.results[{i}]"
                rid = rs.get("id")
                if not isinstance(rid, str) or not re.match(r"^R\d+$", rid):
                    r.err("38", rw, "result id must look like R1, R2, …")
                elif rid in seen_r:
                    r.err("38", rw, f"duplicate result id {rid}")
                else:
                    seen_r.add(rid)
                check_result(r, rw, rs, ref)
                if rs.get("artefact"):
                    ref("38", f"{rw}.artefact", rs["artefact"], "artefacts")
            check_tables(r, w, e, ref)
        else:
            if e.get("results"):
                r.err("66", w, "formal evidence has no results: its claims are proved, not observed")
            if e.get("tables"):
                r.err("48", w, "formal evidence has no tables")
        if "exemplars" in e:
            r.err("66", f"{w}.exemplars", "removed in v0.32: each excerpt is a result whose value is the excerpt, "
                                          "{text, locator} (§3.4, Excerpts and qualitative evidence)")
        for j, dv in enumerate(as_list(e.get("derived_from"))):
            dv = as_dict(dv)
            dw = f"{w}.derived_from[{j}]"
            if dv.get("evidence") == e.get("id"):
                r.err("72", dw, "evidence is not derived from itself")
            else:
                ref("72", f"{dw}.evidence", dv.get("evidence"), "evidence")
            check_enum(r, "72", f"{dw}.how", dv.get("how"), "derived_how", required=True)
        if "derived_from" in e and (not isinstance(e["derived_from"], list) or not e["derived_from"]):
            r.err("72", f"{w}.derived_from", "a non-empty list of {evidence, how}")
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

    process_warnings(r, doc, claims, coll, by_id, converted)

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
        for where, counts in (("paper.exploration.hypotheses", hy),
                              ("paper.exploration.covers.lab.hypotheses",
                               as_dict(as_dict(as_dict(exp.get("covers")).get("lab")).get("hypotheses")))):
            if "from_exploration" in counts:
                fe, tot = counts["from_exploration"], counts.get("total")
                if not nonneg(fe) or (nonneg(tot) and fe > tot):
                    r.err("58", f"{where}.from_exploration", "must be an integer from 0 to the total of hypotheses")
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
    if local_root is not None and file_name and paper.get("origin") == "converted":
        name = source_name(paper.get("converted_from"))
        if name and file_name != f"{name}.claimset.json":
            r.warn("W31", "paper.converted_from", f"a converted paper in a claimset/ folder is named from its source: "
                                                  f"{name}.claimset.json, not {file_name} (§2.4)")

    # ---------- Contributors ----------
    validate_contributors(r, coll["contributors"], paper, ref, converted)

    # ---------- Computed verifiability (rule 13) ----------
    for cid, c in claims.items():
        exp_v, exp_flags, notes = compute_verifiability(c, by_id, read_from_artefacts(coll))
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
    if lvl and lvl not in ("public", "not_stated"):
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
    for m in NESTED_MARK_RE.finditer(text):
        r.err("1", where, f"a defined-term marker holds plain text, not another marker: '{m.group(0)}…' (§8 rule 5)")
    if INSTRUCTION_RE.search(text):
        r.err("safe-ingest", where, "text looks like an instruction addressed to a reader or model; remove it")


def statement_checks(r, where, text):
    if not isinstance(text, str) or not text.strip():
        r.err("1", where, "missing statement")
        return
    if ANAPHORA_RE.search(text):
        r.warn("S3", where, "possible cross-reference or anaphora; statements must be self-contained")
    if EVALUATIVE_RE.search(text):
        r.warn("S7", where, "possible evaluative rhetoric (e.g. 'novel', 'remarkable'; 'significant' only as 'statistically significant'); state the finding plainly")
    if len(re.findall(r"\.\s+[A-Z]", NUMBER_SKIP.sub(" ", text))) >= 1:
        r.warn("S1", where, "statement may contain more than one sentence or assertion")
    cues = compound_cues(text)
    if cues:
        r.warn("W22", where, f"may hold more than one claim ({'; '.join(cues)}): if one part could be true and "
                             f"another false, split it into claims, each self-contained; a comparison that states "
                             f"where a method is better and where it is not stays one claim")


SETUP_ROLES = ("studied", "cases", "procedure", "lens", "context")
OLD_SETUP_ROLES = {"compared": "studied", "data": "cases", "design": "procedure", "measures": "lens"}
ABOUT_SLOTS = ("subject", "relation", "object", "comparator", "conditions")
DEF_ID_RE = re.compile(r"^D\d+$")
# Kinds of the parts a setup should name (warning 28): a defined term of these kinds in a claim's `about` is a part
# of the study the claim's evidence should list.
SETUP_PART_KINDS = {"method", "intervention", "population", "data", "design", "variable"}
CURIE_RE = re.compile(r"^[a-z][a-z0-9._-]*:\S+$")
ROW_REF_RE = re.compile(r"^(A\d+)#row=(\d+)(?:-(\d+))?$")
EXCERPT_FIELDS = {"text", "locator", "language", "gloss", "translation"}
ENCODING_KEYS = {"x", "y", "colour", "shape", "size", "panel"}
LOCAL_RELATION_TYPES = {"supports", "contradicts", "refines", "explains", "qualifies"}
#: Relation types for other works only (§3.7)
EXTERNAL_ONLY_RELATIONS = {"replicates", "fails_to_replicate", "supersedes", "addresses"}


def check_frame(r, coll, claims, by_id):
    """Rules 63 and 64, warnings 27 and 28: definition kinds, a claim's `about` and its fit with the setups of its
    evidence (spec §3.2, §3.3.5, §3.4)."""
    framed = set()                                   # definitions named in a setup or an `about`
    in_setup = {}                                    # evidence ID -> definition IDs its setup names
    for e in coll["evidence"]:
        e = as_dict(e)
        names = set()
        for ids_ in as_dict(e.get("setup")).values():
            names |= {d for d in as_list(ids_) if isinstance(d, str)}
        in_setup[e.get("id")] = names
        framed |= names
    terms = {}
    for d in coll["definitions"]:
        d = as_dict(d)
        if isinstance(d.get("term"), str):
            terms[d["term"].strip().lower()] = d.get("id")
        if d.get("kind") == "measure":
            r.err("68", f"definitions:{d.get('id')}.kind", "renamed in v0.32: the kind measure is now variable "
                                                          "(`claimset migrate` renames it)")
        elif "kind" in d and d["kind"] not in ENUM["definition_kind"]:
            r.err("63", f"definitions:{d.get('id')}.kind",
                  f"'{d['kind']}' is not a kind: {', '.join(sorted(ENUM['definition_kind']))}")

    for cid, c in claims.items():
        w = f"claims:{cid}"
        about = c.get("about")
        if is_assumption(c):
            if about is not None:
                r.err("64", f"{w}.about", "an assumption claim has no about: it is taken as given, not found")
            continue
        if about is None:
            if c.get("role") == "principal":
                r.err("64", w, "a principal claim needs about: its atom as subject, relation, object, comparator "
                               "and conditions (§3.3.5)")
            elif c.get("role") in ("supporting", "secondary"):
                r.warn("W27", w, "no about: record the claim's atom (subject, relation, object, comparator, "
                                 "conditions) so the claim can be found by what it is about")
            continue
        if not isinstance(about, dict):
            r.err("64", f"{w}.about", "an object {subject, relation, object, comparator, conditions}")
            continue
        extra = set(about) - set(ABOUT_SLOTS)
        if extra:
            r.err("64", f"{w}.about", f"the slots are {', '.join(ABOUT_SLOTS)}, not {sorted(extra)}")
        slots = []                                   # (path, value) of every slot value
        for f in ("subject", "relation"):
            v = about.get(f)
            if not isinstance(v, str) or not v.strip():
                r.err("64", f"{w}.about.{f}", "a non-empty string")
            elif f == "subject":
                slots.append((f, v))
        rel = about.get("relation")
        if isinstance(rel, str) and DEF_ID_RE.match(rel.strip()):
            r.err("64", f"{w}.about.relation", "the relation is text, what is asserted of the subject; never an ID")
        for f in ("object", "comparator"):
            if f not in about:
                r.err("64", f"{w}.about", f"missing {f} (null when there is none)")
                continue
            v = about[f]
            if v is None:
                continue
            if f == "object" and isinstance(v, list):
                if not v:
                    r.err("64", f"{w}.about.object", "a list of variables is not empty")
                for i, x in enumerate(v):
                    d = by_id(x, "definitions") if isinstance(x, str) and DEF_ID_RE.match(x) else None
                    if d is None or d.get("kind") != "variable":
                        r.err("64", f"{w}.about.object[{i}]", "a list in object holds only IDs of definitions of "
                                                              "kind variable (atoms grouped by variable, §8 rule 2)")
                    slots.append((f"object[{i}]", x))
            elif isinstance(v, str) and v.strip():
                slots.append((f, v))
            else:
                r.err("64", f"{w}.about.{f}", "null, a definition ID or a short text" +
                      (", or a list of variable definition IDs" if f == "object" else ""))
        conds = about.get("conditions")
        if not isinstance(conds, list):
            r.err("64", f"{w}.about.conditions", "a list (empty when the claim holds without conditions)")
        else:
            for i, v in enumerate(conds):
                if isinstance(v, str) and v.strip():
                    slots.append((f"conditions[{i}]", v))
                else:
                    r.err("64", f"{w}.about.conditions[{i}]", "a definition ID or a short text")
        marked = set(TERM_RE.findall(c.get("statement") or "")) if isinstance(c.get("statement"), str) else set()
        cited = {as_dict(s).get("evidence") for s in as_list(c.get("support"))} - {None}
        # only evidence that can have a setup counts: formal evidence has none (rule 61)
        cited = {eid for eid in cited if as_dict(by_id(eid, "evidence")).get("profile") not in ("formal", None)}

        def members_of(g, seen=()):
            out = set()
            for m in as_list(as_dict(by_id(g, "definitions")).get("members")):
                if isinstance(m, str) and m not in seen:
                    out |= {m} | members_of(m, seen + (g,))
            return out

        def named_in(v, eid, seen=()):
            """A part is named in a setup directly, as a member of a group named there, or as a group whose members
            are all named there (§3.2, Groups)."""
            names = in_setup.get(eid, set())
            if v in names or any(v in members_of(g) for g in names):
                return True
            members = as_list(as_dict(by_id(v, "definitions")).get("members"))
            return bool(members) and all(isinstance(m, str) and m not in seen and named_in(m, eid, seen + (v,))
                                         for m in members)
        for path, v in slots:
            v = v.strip()
            if DEF_ID_RE.match(v):
                d = by_id(v, "definitions")
                if d is None:
                    r.err("64", f"{w}.about.{path}", f"'{v}' does not resolve to a definition")
                    continue
                framed.add(v)
                if v not in marked:
                    r.err("64", f"{w}.about.{path}", f"{v} is not marked in the statement: write [[{v}:{d.get('term')}]] "
                                                     f"where the statement names it")
                if (cited and d.get("kind") in SETUP_PART_KINDS
                        and not any(named_in(v, eid) for eid in cited)):
                    r.warn("W28", f"{w}.about.{path}", f"{v} ({d.get('kind')}) is part of the study the claim rests on "
                                                       f"but no setup of its evidence ({', '.join(sorted(cited))}) "
                                                       f"names it")
            elif v.lower() in terms:
                r.warn("W27", f"{w}.about.{path}", f"'{v}' is the term of {terms[v.lower()]}: give the ID instead")

    for d in coll["definitions"]:
        d = as_dict(d)
        if "kind" not in d:
            r.err("63", f"definitions:{d.get('id')}", "every definition has a kind: what sort of thing the term names "
                                                      "(§3.2)")

    # Groups (rule 65): members are other definitions of the group's kind; no group contains itself
    defs = {as_dict(d).get("id"): as_dict(d) for d in coll["definitions"]}
    for d in defs.values():
        if "members" not in d:
            continue
        did, members = d.get("id"), d.get("members")
        if not isinstance(members, list) or not members or not all(isinstance(m, str) for m in members):
            r.err("65", f"definitions:{did}.members", "a non-empty list of definition IDs")
            continue
        if not d.get("kind"):
            r.err("65", f"definitions:{did}.kind", "a group needs a kind, the kind of its members")
        for m in members:
            if m == did:
                r.err("65", f"definitions:{did}.members", "a group is not a member of itself")
            elif m not in defs:
                r.err("65", f"definitions:{did}.members", f"{m} does not resolve to a definition")
            elif d.get("kind") and defs[m].get("kind") != d.get("kind"):
                r.err("65", f"definitions:{did}.members", f"{m} is of kind {defs[m].get('kind')}, the group of kind "
                                                          f"{d.get('kind')}: members have the group's kind")
    def contains(group, target, seen):
        for m in as_list(defs.get(group, {}).get("members")):
            if m == target or (m not in seen and contains(m, target, seen | {m})):
                return True
        return False
    for did, d in defs.items():
        if isinstance(d.get("members"), list) and any(m != did and contains(m, did, {m}) for m in d["members"]
                                                      if isinstance(m, str)):
            r.err("65", f"definitions:{did}.members", "a group is a member of itself through other groups")

# Wording typical of rhetoric (spec §8 rule 9); reported only with warning W23, never alone
RHETORIC_CUES = [
    ("importance framing", re.compile(r"\b(crucial|paramount|vital|of (?:great|utmost|paramount) (?:importance|interest)|"
                                      r"increasingly (?:important|popular|relevant))\b", re.I)),
    ("promise", re.compile(r"\b(promising|paves? the way|opens? (?:up )?new (?:avenues|possibilities|directions)|"
                           r"has the potential to|holds? (?:great )?promise)\b", re.I)),
    ("sweeping generalisation", re.compile(r"\b(always|never|all practitioners|everyone|widely (?:recognised|recognized|"
                                           r"accepted|known)|it is well known|universally)\b", re.I)),
    ("speculation", re.compile(r"\b(may|might|could|potentially|possibly)\b", re.I)),
]


def rhetoric_cues(text):
    """Rhetorical wording in a statement: [`kind 'word'`]."""
    if not isinstance(text, str):
        return []
    plain = NUMBER_SKIP.sub(" ", text)
    return [f"{kind} '{m.group(1).lower()}'" for kind, rx in RHETORIC_CUES for m in [rx.search(plain)] if m]


# Signs that a statement holds more than one claim (spec §8 rule 1; warning W22)
COMPOUND_CUES = [
    ("semicolon", re.compile(r";")),
    ("contrast", re.compile(r"\b(but|while|whereas|although|however)\b", re.I)),
    ("addition", re.compile(r"\b(as well as|also|in addition|moreover|furthermore|not only)\b", re.I)),
    ("explanation", re.compile(r"\b(because|due to|thus|therefore|hence|which explains|since(?!\s+(?:the\s+)?\d))\b",
                               re.I)),
    ("side clause", re.compile(r",\s*which\b", re.I)),
    ("parenthesis", re.compile(r"\([^()]*\b(and|also|is|are)\b[^()]*\)", re.I)),
    ("recommendation", re.compile(r"\b(should|recommends?|recommended)\b", re.I)),
]


def compound_cues(text):
    """The signs, in a statement, that it holds more than one claim: [`kind 'word'`]."""
    if not isinstance(text, str):
        return []
    plain = NUMBER_SKIP.sub(" ", text)
    out = []
    for kind, rx in COMPOUND_CUES:
        m = rx.search(plain)
        if m:
            out.append(kind if kind in ("semicolon", "parenthesis", "side clause") else f"{kind} '{m.group(1).lower()}'")
    return out


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


NUMBER_SKIP = re.compile(
    r"\$[^$]*\$"                                                      # LaTeX
    r"|\[\[D\d+:[^\]]*\]\]"                                            # defined-term markers
    r"|\b[A-Za-z]{1,8}\([^()]*\)"                                       # model notation: ETS(A,N,A), AR(1)
    r"|\b(?:Sections?|Sect\.|Tables?|Fig(?:ure)?s?\.?|Eq(?:uation)?s?\.?|Theorems?|Lemmas?|Propositions?|"
    r"Corollar(?:y|ies)|Chapters?|Appendi(?:x|ces)|Algorithms?|§)\s*[A-Z]?\d+(?:\.\d+)*")


def _numbers_in(x, out):
    """Every number in a value: numbers themselves, and numbers written in strings, lists and objects."""
    if isinstance(x, bool):
        return out
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, str):
        for m in NUM_RE.finditer(NUMBER_SKIP.sub(" ", x)):
            try:
                out.append(float(m.group(1)) * 10 ** (int(m.group(3)) if m.group(3) else 0))
            except ValueError:
                pass
    elif isinstance(x, dict):
        for v in x.values():
            _numbers_in(v, out)
    elif isinstance(x, list):
        for v in x:
            _numbers_in(v, out)
    return out


def unsourced_numbers(c, support, by_id):
    """W17: numbers in the statement of an empirical or simulation claim that no structured source holds: the
    cited evidence's results (values, intervals, conditions) and table cells and conditions, the test's
    parameters, the claim's scope, and the definitions the statement references. Evidence descriptions do not
    count: a number a statement reports should be a recorded result."""
    if not {s.get("kind") for s in support} & {"empirical_quant", "simulation"}:
        return []
    sources = []
    for s in support:
        e = by_id(s.get("evidence"), "evidence") if s.get("evidence") else None
        if e is None:
            continue
        for x in as_list(e.get("results")):
            x = as_dict(x)
            _numbers_in([x.get("value"), x.get("interval"), x.get("conditions")], sources)
        for t in as_list(e.get("tables")):
            t = as_dict(t)
            _numbers_in([t.get("conditions"), [as_dict(row).get("values") for row in as_list(t.get("rows"))],
                         [as_dict(row).get("label") for row in as_list(t.get("rows"))]], sources)
        _numbers_in([e.get("estimate"), e.get("interval"), e.get("n")], sources)
    _numbers_in([as_dict(c.get("test")).get("parameters"), c.get("scope")], sources)
    stmt = c.get("statement") or ""
    for m in re.finditer(r"\[\[(D\d+):", stmt):
        d = by_id(m.group(1), "definitions")
        if d is not None:
            _numbers_in(d.get("definition"), sources)
    out = []
    for m in NUM_RE.finditer(NUMBER_SKIP.sub(" ", stmt)):
        token = m.group(0)
        if not any(number_matches(v, token) for v in sources) and token.strip() not in out:
            out.append(token.strip())
    return out


def table_echo(c, support, by_id):
    """W24: values of the cited result tables that the statement repeats. The tables are those of the evidence an
    empirical or simulation support entry points to, narrowed to the tables its `results` name when it names any.
    Numbers are found as for W17; each distinct number in the statement that equals a cell counts once."""
    cells = []
    for s in support:
        if s.get("kind") not in ("empirical_quant", "simulation"):
            continue
        e = by_id(s.get("evidence"), "evidence") if s.get("evidence") else None
        if e is None:
            continue
        named = {str(x).split(".")[1] for x in as_list(s.get("results")) if str(x).count(".") >= 1}
        for t in as_list(e.get("tables")):
            t = as_dict(t)
            if named and t.get("id") not in named:
                continue
            for row in as_list(t.get("rows")):
                _numbers_in(as_dict(row).get("values"), cells)
    if not cells:
        return []
    out = []
    for m in NUM_RE.finditer(NUMBER_SKIP.sub(" ", c.get("statement") or "")):
        token = m.group(0).strip()
        if token not in out and not m.group(4) and any(number_matches(v, token) for v in cells):
            out.append(token)
    return out


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
    if "addresses" in test:
        if not isinstance(test["addresses"], list) or not test["addresses"]:
            r.err("59", f"{w}.addresses", "a non-empty list of review comments' global IDs")
        for a in as_list(test.get("addresses")):
            if not (isinstance(a, str) and COMMENT_ID_RE.match(a)):
                r.err("59", f"{w}.addresses", f"'{a}' is not a review comment's global ID "
                                              f"(<claimset>@v<n>/RV<k>.c<m>)")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def value_ok(v):
    """A value of an observation (rule 66): a number, a non-empty string (a category: a definition ID or a code),
    a boolean, or an excerpt {text, locator, language, gloss, translation}."""
    if isinstance(v, bool) or _is_number(v):
        return True
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, dict):
        return (set(v) <= EXCERPT_FIELDS and isinstance(v.get("text"), str) and v["text"].strip() != ""
                and isinstance(v.get("locator"), str) and v["locator"].strip() != ""
                and all(isinstance(v[k], str) for k in v))
    return False


def _coord_ok(k, v):
    return isinstance(k, str) and k.strip() and (isinstance(v, bool) or _is_number(v)
                                                 or (isinstance(v, str) and v.strip()))


def check_result(r, rw, rs, ref):
    """Rule 66: one result entry (an observation)."""
    if "quantity" in rs:
        r.err("66", f"{rw}.quantity", "renamed in v0.32: what is observed is the result's variable "
                                      "(`claimset migrate` renames it)")
    if not isinstance(rs.get("variable"), str) or not rs["variable"].strip():
        r.err("66", rw, "missing variable: what is observed (a definition ID or text)")
    if not value_ok(rs.get("value")):
        r.err("66", f"{rw}.value", "a number, a category, a boolean, or an excerpt {text, locator}")
    if "unit" in rs and rs["unit"] is not None and not (isinstance(rs["unit"], str) and rs["unit"].strip()):
        r.err("66", f"{rw}.unit", "a UCUM code or text")
    if "n" in rs and not (isinstance(rs["n"], int) and not isinstance(rs["n"], bool) and rs["n"] >= 0):
        r.err("66", f"{rw}.n", "a non-negative integer")
    if "interval" in rs and not isinstance(rs["interval"], dict):
        r.err("66", f"{rw}.interval", "{level, lower, upper}")
    conds = rs.get("conditions")
    if conds is not None and (not isinstance(conds, dict) or not all(_coord_ok(k, v) for k, v in conds.items())):
        r.err("66", f"{rw}.conditions", "an object of dimension: value, each a definition ID, text, number or boolean")
    rf = rs.get("read_from")
    if rf is not None:
        rf = as_dict(rf)
        a = ref("66", f"{rw}.read_from.artefact", rf.get("artefact"), "artefacts")
        if a is not None and a.get("kind") not in ("figure", "recording"):
            r.err("66", f"{rw}.read_from", "values are read from a figure or a recording")
        if rf.get("precision") in (None, ""):
            r.err("66", f"{rw}.read_from", "give the precision of the reading")


def check_tables(r, w, e, ref):
    """Rule 48: tables, compact forms of observations (spec §3.4)."""
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
        if "conditions" in t and (not isinstance(t["conditions"], dict)
                                  or not all(_coord_ok(k, v) for k, v in t["conditions"].items())):
            r.err("48", f"{tw}.conditions", "an object of dimension: value")
        labels, cols = t.get("row_labels"), t.get("columns")
        if not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels):
            r.err("48", f"{tw}.row_labels", "must be a list of dimensions (possibly empty)")
            labels = None

        def col_ok(c):
            if isinstance(c, str):
                return bool(c.strip())
            return (isinstance(c, dict) and set(c) <= {"variable", "conditions"} and isinstance(c.get("variable"), str)
                    and c["variable"].strip() != "" and isinstance(c.get("conditions", {}), dict)
                    and all(_coord_ok(k, v) for k, v in c.get("conditions", {}).items()))
        if not isinstance(cols, list) or not cols or not all(col_ok(x) for x in cols):
            r.err("48", f"{tw}.columns", "a non-empty list, each a variable or {variable, conditions}")
            cols = None
        elif len({json.dumps(x, sort_keys=True) for x in cols}) != len(cols):
            r.err("48", f"{tw}.columns", "columns must be distinct")
        if cols is not None and "units" in t:
            u = t["units"]
            if not isinstance(u, list) or len(u) != len(cols) or not all(x is None or isinstance(x, str) for x in u):
                r.err("48", f"{tw}.units", "must have one unit (or null) per column")
        rows = t.get("rows")
        if not isinstance(rows, list) or not rows:
            r.err("48", f"{tw}.rows", "must be a non-empty list")
            rows = []
        seen_rows, any_value = set(), False
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
                                       or not all(_coord_ok("x", x) for x in lab)):
                r.err("48", f"{rw}.label", f"must have one entry per row label ({len(labels)})")
            vals = row.get("values")
            if cols is not None and (not isinstance(vals, list) or len(vals) != len(cols)):
                r.err("48", f"{rw}.values", f"must have one value per column ({len(cols)})")
            elif isinstance(vals, list):
                for v in vals:
                    if v is None:
                        continue
                    if value_ok(v):
                        any_value = True
                    else:
                        r.err("48", f"{rw}.values", f"'{v}' is not a value (number, category, boolean or excerpt) "
                                                    f"or null")
        if rows and not any_value:
            r.err("48", tw, "the table has no value")
        if t.get("artefact"):
            ref("48", f"{tw}.artefact", t["artefact"], "artefacts")
        if "source_spans" in t and not (isinstance(t["source_spans"], list)
                                        and all(isinstance(x, str) and x for x in t["source_spans"])):
            r.err("48", f"{tw}.source_spans", "must be a list of non-empty strings")


def observations(e):
    """Every observation of an evidence object as (where, variable, conditions, value, unit): its results and the
    cells of its tables, each cell with the table's, the row's and the column's conditions (§3.4)."""
    out = []
    for i, rs in enumerate(as_list(e.get("results"))):
        rs = as_dict(rs)
        out.append((f"results[{i}]", rs.get("variable"), as_dict(rs.get("conditions")), rs.get("value"), rs.get("unit")))
    for i, t in enumerate(as_list(e.get("tables"))):
        t = as_dict(t)
        labels = [x for x in as_list(t.get("row_labels")) if isinstance(x, str)]
        cols = as_list(t.get("columns"))
        units = as_list(t.get("units"))
        for j, row in enumerate(as_list(t.get("rows"))):
            row = as_dict(row)
            base = dict(as_dict(t.get("conditions")))
            base.update({k: v for k, v in zip(labels, as_list(row.get("label")))})
            for k, (c, v) in enumerate(zip(cols, as_list(row.get("values")))):
                cond = dict(base)
                var = c if isinstance(c, str) else as_dict(c).get("variable")
                if isinstance(c, dict):
                    cond.update(as_dict(c.get("conditions")))
                out.append((f"tables[{i}].rows[{j}].values[{k}]", var, cond, v,
                            units[k] if k < len(units) else None))
    return out


def inherited_assumptions(c, defs, evid):
    """The assumptions a claim inherits from the terms it rests on (§3.2, Assumptions of a term): those its
    evidence's setup names in the roles cases, procedure, lens and context, and, of the terms studied, only its own
    subject (each member when the subject is a group) when the setup names it; a claim without about inherits
    from every term its setup names. A narrower term passes on the assumptions of its broader terms. `defs` and
    `evid` map IDs to definitions and evidence objects."""
    def members(g, seen=()):
        out = []
        for m in as_list(as_dict(defs.get(g)).get("members")):
            if isinstance(m, str) and m not in seen:
                out += [m] + members(m, seen + (g,))
        return out

    about = c.get("about") if isinstance(c.get("about"), dict) else None
    subject = about.get("subject") if about else None
    subjects = [subject.strip()] + members(subject.strip()) if isinstance(subject, str) else []
    start = []
    for s in as_list(c.get("support")):
        e = evid.get(as_dict(s).get("evidence"))
        for role, ids in as_dict(as_dict(e).get("setup")).items():
            ids = [i for i in as_list(ids) if isinstance(i, str)]
            if about is None or role != "studied":
                start += ids
                continue
            named = set(ids) | {m for g in ids for m in members(g)}
            start += [x for x in subjects if x in named or (members(x) and set(members(x)) <= named)]
    out, seen, stack = [], set(), list(start)
    while stack:
        did = stack.pop(0)
        if did in seen or did not in defs:
            continue
        seen.add(did)
        out += [a for a in as_list(defs[did].get("assumptions")) if isinstance(a, str) and a not in out]
        stack += [b for b in as_list(defs[did].get("broader")) if isinstance(b, str)]
    return out


def read_from_artefacts(coll):
    """Artefacts some observation is read from (§3.4, read_from)."""
    out = set()
    for e in coll["evidence"]:
        for rs in as_list(as_dict(e).get("results")):
            a = as_dict(as_dict(rs).get("read_from")).get("artefact")
            if isinstance(a, str):
                out.add(a)
    return frozenset(out)


def is_relative_uri(u):
    return isinstance(u, str) and u.strip() != "" and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", u) \
        and not u.startswith("/")


def check_artefact_v032(r, w, a, ref, by_id, coll):
    """Rule 67: figures by function, what a data plot shows, its encoding, and the columns of bulk data."""
    kind = a.get("kind")
    if kind == "figure":
        check_enum(r, "67", f"{w}.function", a.get("function"), "figure_function", required=True)
    elif "function" in a:
        r.err("67", f"{w}.function", "only figures have a function")
    if "shows" in a:
        shows = a["shows"]
        if a.get("function") != "data_plot":
            r.err("67", f"{w}.shows", "only a data_plot shows observations")
        elif not isinstance(shows, list) or not shows:
            r.err("67", f"{w}.shows", "a non-empty list of result, table or row references, or bulk data artefacts")
        else:
            for x in shows:
                x = str(x)
                if re.match(r"^A\d+$", x):
                    ref("67", f"{w}.shows", x, "artefacts")
                    continue
                m = re.match(r"^(E\d+)\.", x)
                e = by_id(m.group(1), "evidence") if m else None
                if not result_ref_resolves(x, m.group(1) if m else None, e, by_id):
                    r.err("67", f"{w}.shows", f"'{x}' does not resolve to a result, table or row")
    if "encoding" in a:
        enc = a["encoding"]
        if not isinstance(enc, dict) or not set(enc) <= ENCODING_KEYS \
                or not all(isinstance(v, str) and v.strip() for v in enc.values()):
            r.err("67", f"{w}.encoding", f"an object of {', '.join(sorted(ENCODING_KEYS))}, each a definition ID or text")
        else:
            for k, v in enc.items():
                if DEF_ID_RE.match(v):
                    ref("67", f"{w}.encoding.{k}", v, "definitions")
    if "columns" in a:
        cols = a["columns"]
        if kind not in ("data", "table"):
            r.err("67", f"{w}.columns", "columns are given for bulk data (kind data or table)")
        if not isinstance(cols, list) or not cols:
            r.err("67", f"{w}.columns", "a non-empty list of {name, variable or dimension, unit}")
        else:
            for i, c in enumerate(cols):
                c = as_dict(c)
                cw = f"{w}.columns[{i}]"
                if not isinstance(c.get("name"), str) or not c["name"].strip():
                    r.err("67", cw, "missing name")
                if ("variable" in c) == ("dimension" in c):
                    r.err("67", cw, "exactly one of variable and dimension")
                term = c.get("variable", c.get("dimension"))
                if isinstance(term, str) and DEF_ID_RE.match(term):
                    ref("67", cw, term, "definitions")
                elif not (isinstance(term, str) and term.strip()):
                    r.err("67", cw, "the variable or dimension, a definition ID or text")
    if a.get("locator") is not None and not (isinstance(a["locator"], str) and a["locator"].strip()):
        r.err("1", f"{w}.locator", "where in the cited work, as text")


def check_v032(r, paper, coll, claims, by_id, ids, converted, published):
    """Rules 66 (terms of observations), 68, 69, 70 (published files), 71 (cycles), 72 (cycles), 73, 74;
    warnings 29 and 30 (spec v0.32)."""
    defs = {as_dict(d).get("id"): as_dict(d) for d in coll["definitions"]}

    def members_of(g, seen=()):
        out = set()
        for m in as_list(defs.get(g, {}).get("members")):
            if isinstance(m, str) and m not in seen:
                out |= {m} | members_of(m, seen + (g,))
        return out

    # ----- definitions: part_of, broader, identifiers, assumptions, marks (rules 68, 73) -----
    for did, d in defs.items():
        dw = f"definitions:{did}"
        for link in ("part_of", "broader"):
            if link not in d:
                continue
            lst = d[link]
            if not isinstance(lst, list) or not lst or not all(isinstance(x, str) for x in lst):
                r.err("68", f"{dw}.{link}", "a non-empty list of definition IDs")
                continue
            for x in lst:
                if x == did:
                    r.err("68", f"{dw}.{link}", "a term is not linked to itself")
                elif x not in defs:
                    r.err("68", f"{dw}.{link}", f"{x} does not resolve to a definition")
                elif link == "broader" and d.get("kind") and defs[x].get("kind") \
                        and defs[x]["kind"] != d["kind"]:
                    r.err("68", f"{dw}.broader", f"{x} is of kind {defs[x]['kind']}: a narrower term has the kind of "
                                                 f"its broader term")
        if "identifiers" in d:
            idl = d["identifiers"]
            if not isinstance(idl, list) or not idl or not all(isinstance(x, str) and CURIE_RE.match(x) for x in idl):
                r.err("68", f"{dw}.identifiers", "a non-empty list of compact identifiers prefix:local "
                                                 "(e.g. chebi:15377, ncbitaxon:9606, doi:10.…)")
        if "assumptions" in d:
            al = d["assumptions"]
            if not isinstance(al, list) or not al:
                r.err("73", f"{dw}.assumptions", "a non-empty list of assumption claim IDs")
            else:
                for x in al:
                    c = claims.get(x)
                    if c is None or not is_assumption(c):
                        r.err("73", f"{dw}.assumptions", f"{x} is not an assumption claim")
        text_checks(r, f"{dw}.definition", d.get("definition"), ids)
    for link in ("part_of", "broader"):
        def reaches(a, target, seen):
            for x in as_list(defs.get(a, {}).get(link)):
                if x == target or (isinstance(x, str) and x not in seen and reaches(x, target, seen | {x})):
                    return True
            return False
        for did, d in defs.items():
            if isinstance(d.get(link), list) and any(isinstance(x, str) and x != did and reaches(x, did, {x})
                                                     for x in d[link]):
                r.err("68", f"definitions:{did}.{link}", f"{link} forms a cycle: a term is not linked to itself "
                                                        f"through other terms")
    for b in coll["references"]:
        b = as_dict(b)
        if b.get("id_type") == "curie" and not CURIE_RE.match(str(b.get("target") or "")):
            r.err("68", f"references:{b.get('id')}.target", "a compact identifier prefix:local")

    # ----- terms of observations (rules 66, 74; warning 30) -----
    terms = {as_dict(d).get("term", "").strip().lower(): did for did, d in defs.items()
             if isinstance(as_dict(d).get("term"), str)}

    def variable_ok(v):
        d = defs.get(v)
        if d is None:
            return False
        if d.get("kind") in ("variable", "concept"):
            return True
        ms = members_of(v)
        return bool(ms) and all(defs.get(m, {}).get("kind") in ("variable", "concept") for m in ms)

    for e in coll["evidence"]:
        e = as_dict(e)
        if e.get("profile") == "formal":
            continue
        w = f"evidence:{e.get('id')}"
        for t in as_list(e.get("tables")):
            for x in as_list(as_dict(t).get("row_labels")):
                if isinstance(x, str) and DEF_ID_RE.match(x) and x not in defs:
                    r.err("66", f"{w}.tables:{as_dict(t).get('id')}.row_labels", f"{x} does not resolve to a definition")
        for where, var, cond, value, _unit in observations(e):
            ow = f"{w}.{where}"
            if isinstance(var, str) and DEF_ID_RE.match(var.strip()):
                if var not in defs:
                    r.err("66", f"{ow}.variable", f"{var} does not resolve to a definition")
                elif not variable_ok(var):
                    r.err("66", f"{ow}.variable", f"{var} is of kind {defs[var].get('kind')}: a variable names a "
                                                  f"definition of kind variable or concept, or a group of them")
            elif isinstance(var, str) and var.strip().lower() in terms:
                r.warn("W30", f"{ow}.variable", f"'{var}' is the term of {terms[var.strip().lower()]}: give the ID")
            groups = []
            for k, v in cond.items():
                for x in (k, v):
                    if isinstance(x, str) and DEF_ID_RE.match(x) and x not in defs:
                        r.err("66", f"{ow}.conditions", f"{x} does not resolve to a definition")
                if k == "comparator":
                    continue
                if k in defs and members_of(k):
                    groups.append(k)
                    if isinstance(v, str) and DEF_ID_RE.match(v) and v in defs and v not in members_of(k):
                        r.err("66", f"{ow}.conditions", f"{v} is not a member of {k}, the dimension it lies on")
            comp = cond.get("comparator")
            if isinstance(comp, str) and DEF_ID_RE.match(comp) and comp in defs and groups \
                    and not any(comp in members_of(g) for g in groups):
                r.err("74", f"{ow}.conditions.comparator", f"{comp} is a member of none of the dimensions "
                                                           f"{', '.join(groups)}: a comparator lies on the dimension "
                                                           f"whose value it replaces")
            if isinstance(value, str) and DEF_ID_RE.match(value) and value not in defs:
                r.err("66", f"{ow}.value", f"{value} does not resolve to a definition")

    # ----- figures without observations (warning 30) -----
    read_from = read_from_artefacts(coll)
    for a in coll["artefacts"]:
        a = as_dict(a)
        if a.get("kind") != "figure":
            continue
        if a.get("function") == "data_plot" and not a.get("shows") and not a.get("underlying_data"):
            r.warn("W30", f"artefacts:{a.get('id')}", "a data plot that names neither the observations it shows nor "
                                                      "its underlying data: record its values as observations")
        if a.get("function") == "specimen" and a.get("id") not in read_from:
            r.warn("W30", f"artefacts:{a.get('id')}", "a specimen from which no observation is read: record what "
                                                      "the paper reads from it, with read_from")
    if published:
        for a in coll["artefacts"]:
            a = as_dict(a)
            if is_relative_uri(a.get("uri")):
                r.err("70", f"artefacts:{a.get('id')}.uri", "a published file names no relative paths: the platform "
                                                            "assigns the location of each packaged file")

    # ----- relations within the paper: explains forms no cycle (rule 71) -----
    explains = {cid: [rel.get("target") for rel in (as_dict(x) for x in as_list(c.get("relations")))
                      if rel.get("id_type") == "local" and rel.get("type") == "explains"]
                for cid, c in claims.items()}

    def cyc(graph, start):
        stack, seen = list(graph.get(start, [])), set()
        while stack:
            x = stack.pop()
            if x == start:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack += graph.get(x, [])
        return False
    for cid in claims:
        if cyc(explains, cid):
            r.err("71", f"claims:{cid}.relations", "explains forms a cycle among the paper's claims")
    derived = {as_dict(e).get("id"): [as_dict(x).get("evidence") for x in as_list(as_dict(e).get("derived_from"))]
               for e in coll["evidence"]}
    for eid in derived:
        if cyc(derived, eid):
            r.err("72", f"evidence:{eid}.derived_from", "derived_from forms a cycle")

    # ----- declarations (rule 69, warning 29) -----
    decl = paper.get("declarations")
    if decl is not None:
        if not isinstance(decl, dict):
            r.err("69", "paper.declarations", "an object {funding, competing_interests, ethics, consent}")
        else:
            extra = set(decl) - {"funding", "competing_interests", "ethics", "consent"}
            if extra:
                r.err("69", "paper.declarations", f"only funding, competing_interests, ethics and consent, not "
                                                  f"{sorted(extra)}")
            if "funding" in decl:
                if not isinstance(decl["funding"], list):
                    r.err("69", "paper.declarations.funding", "a list (empty when there is no funding)")
                for i, f in enumerate(as_list(decl.get("funding"))):
                    f = as_dict(f)
                    if not isinstance(f.get("funder"), str) or not f["funder"].strip():
                        r.err("69", f"paper.declarations.funding[{i}]", "missing funder")
            if "ethics" in decl:
                if not isinstance(decl["ethics"], list):
                    r.err("69", "paper.declarations.ethics", "a list (empty when no ethics review was needed)")
                for i, x in enumerate(as_list(decl.get("ethics"))):
                    x = as_dict(x)
                    if not isinstance(x.get("body"), str) or not x["body"].strip():
                        r.err("69", f"paper.declarations.ethics[{i}]", "missing body")
                    check_enum(r, "69", f"paper.declarations.ethics[{i}].decision", x.get("decision"),
                               "ethics_decision", required=True)
            for f in ("competing_interests", "consent"):
                if f in decl and not (isinstance(decl[f], str) and decl[f].strip()):
                    r.err("69", f"paper.declarations.{f}", "non-empty text")
    if not converted and (not isinstance(decl, dict) or "funding" not in decl or "competing_interests" not in decl):
        r.warn("W29", "paper.declarations", "declare funding and competing interests, as the guarantor states them "
                                            "(an empty funding list and \"None declared.\" are declarations)")


def validate_package(path):
    """Rule 70: a submission package (BagIt, RFC 8493): a folder or a zip holding bagit.txt, manifest-sha256.txt
    and data/ with one claimset file and the files its artefacts name. Returns (report, claimset document)."""
    import hashlib
    import tempfile
    import zipfile
    root = Path(path)
    tmp = None
    if root.is_file() and zipfile.is_zipfile(root):
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(root) as z:
            z.extractall(tmp.name)
        inner = [p for p in Path(tmp.name).iterdir() if p.is_dir()]
        root = inner[0] if len(inner) == 1 and not (Path(tmp.name) / "bagit.txt").exists() else Path(tmp.name)
    r = Report()
    manifest = {}
    mpath = root / "manifest-sha256.txt"
    if not (root / "bagit.txt").is_file() or not mpath.is_file():
        r.err("70", str(path), "a package has bagit.txt and manifest-sha256.txt (BagIt, RFC 8493)")
        return r, None
    for line in mpath.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.strip().partition(" ")
            manifest[name.strip().lstrip("*")] = digest.strip().lower()
    files = sorted((root / "data").glob("*.claimset.json")) or sorted((root / "data").glob("*.json"))
    if len(files) != 1:
        r.err("70", str(path), "the package's data/ holds exactly one claimset file")
        return r, None
    doc = json.loads(files[0].read_text(encoding="utf-8"))

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    for name, digest in manifest.items():
        f = root / name
        if not f.is_file():
            r.err("70", name, "listed in the manifest but not in the package")
        elif sha(f) != digest:
            r.err("70", name, "checksum differs from the manifest")
    for a in as_list(doc.get("artefacts")):
        a = as_dict(a)
        u = a.get("uri")
        if not is_relative_uri(u):
            continue
        aw = f"artefacts:{a.get('id')}.uri"
        rel = f"data/{u}"
        f = root / rel
        if not f.is_file():
            r.err("70", aw, f"{u} is not in the package")
            continue
        digest = sha(f)
        if manifest.get(rel) != digest:
            r.err("70", aw, f"{u} is not in the manifest with its checksum")
        if a.get("version") != f"sha256:{digest}":
            r.err("70", f"artefacts:{a.get('id')}.version", f"sha256:{digest}, the checksum of {u}")
    if tmp is not None:
        tmp.cleanup()
    return r, doc


def result_ref_resolves(ref_id, evidence_id, e, by_id=None):
    """Rule 38: E<n>.R<m>, E<n>.T<m> or E<n>.T<m>.r<k> within the evidence a support entry points to, or rows of a
    bulk data artefact of that evidence, A<n>#row=<k>[-<l>] (RFC 7111)."""
    if e is None:
        return False
    rm = ROW_REF_RE.match(ref_id)
    if rm:
        held = set(as_list(e.get("artefacts"))) | {as_dict(x).get("artefact") for x in as_list(e.get("results"))} \
            | {as_dict(t).get("artefact") for t in as_list(e.get("tables"))}
        if rm.group(1) not in held:
            return False
        a = by_id(rm.group(1), "artefacts") if by_id else None
        if by_id and (a is None or a.get("kind") not in ("data", "table")):
            return False
        return rm.group(3) is None or int(rm.group(3)) >= int(rm.group(2))
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
        if is_assumption(c):
            if "novelty" in c:
                r.err("60", f"{w}.novelty", "an assumption is taken as given, not found: it has no novelty")
        elif c.get("novelty") == "new":
            r.err("60", f"{w}.novelty", "'new' is renamed 'no_prior_found' (v0.22): the authors found no earlier "
                                        "result stating the claim")
        else:
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
            art = ref("10", f"{sw}.artefacts", aid, "artefacts")
            if art is not None and art.get("kind") == "figure" and art.get("function") == "diagram" \
                    and k != "demonstration":
                r.err("67", f"{sw}.artefacts", f"{aid} is a diagram: it illustrates and is never evidence; cite the "
                                               f"observations or the definition it shows")
        profile_for = {"empirical_quant": "quant", "empirical_qual": "qual",
                       "simulation": "simulation", "formal_proof": "formal"}
        if k in profile_for:
            e = ref("10", f"{sw}.evidence", s.get("evidence"), "evidence")
            if e is not None and e.get("profile") != profile_for[k]:
                r.err("10", f"{sw}.evidence", f"{k} must point to evidence with profile {profile_for[k]}")
            if k == "simulation" and e is not None and not (as_list(s.get("artefacts")) or as_list(e.get("artefacts"))):
                r.err("10", sw, "simulation support requires an artefact")
            if s.get("results"):
                if k == "formal_proof":
                    r.err("38", f"{sw}.results", "results references only on empirical and simulation support")
                for ref_id in as_list(s.get("results")):
                    if not result_ref_resolves(str(ref_id), s.get("evidence"), e, by_id):
                        r.err("38", f"{sw}.results", f"'{ref_id}' does not resolve to a result, table, table row or "
                                                     f"bulk data rows of {s.get('evidence')}")
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
            if s.get("id_type") == "curie" and not CURIE_RE.match(str(s.get("target") or "")):
                r.err("68", f"{sw}.target", "a compact identifier prefix:local")
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
        if rel.get("id_type") == "local":
            tgt = rel.get("target")
            if tgt == cid or ids.get(tgt, (None,))[0] != "claims":
                r.err("71", f"{rw}.target", f"'{tgt}' is not another claim of this paper")
            if rel.get("type") not in LOCAL_RELATION_TYPES:
                r.err("71", f"{rw}.type", f"within the paper a relation is one of {', '.join(sorted(LOCAL_RELATION_TYPES))}")
            if rel.get("type") == "explains" and tgt in as_list(c.get("depends_on")):
                r.err("71", rw, f"explains {tgt} and lists it in depends_on: an explanation is not a premise of the "
                                f"result it explains, nor the result of the explanation")
        if rel.get("id_type") == "curie" and not CURIE_RE.match(str(rel.get("target") or "")):
            r.err("68", f"{rw}.target", "a compact identifier prefix:local")
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
    unsourced = unsourced_numbers(c, support, by_id)
    if unsourced:
        r.warn("W17", f"{w}.statement", f"{', '.join(unsourced)} in the statement match no recorded result, table "
                                        f"cell, test parameter, scope condition or referenced definition: record the "
                                        f"number as a result (or the setting in the scope), or check it")
    echoed = table_echo(c, support, by_id)
    if len(echoed) >= 3:
        r.warn("W24", f"{w}.statement", f"repeats {len(echoed)} values of the tables it cites ({', '.join(echoed)}): "
                                        f"state the pattern the paper concludes and cite the table or rows, repeating a "
                                        f"value only when the conclusion turns on it; a claim that only retells a table "
                                        f"is removed and the table cited where it is used (§3.4)")
    with_results = sorted({str(s.get("evidence")) for s in support if s.get("evidence") and as_list(s.get("results"))})
    if len(with_results) > 1:
        r.warn("W22", f"{w}.statement", f"cites results of {', '.join(with_results)}: if the statement reports a "
                                        f"result of each, it may hold more than one claim; split it if one part could "
                                        f"be true and another false")
    kinds = {s.get("kind") for s in support}
    if c.get("novelty") == "no_prior_found" and kinds == {"consensus"}:
        r.warn("W19", w, "novelty no_prior_found, but the only support is established knowledge (consensus): if the claim is "
                         "itself known, cite it or relate to it; if it is new, give the support that shows it")
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


def compute_verifiability(c, by_id, read_from=frozenset()):
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
            if a.get("underlying_data"):
                queue.append(a["underlying_data"])
            if a.get("function") == "specimen":
                if aid not in read_from:
                    flags.add("figure_without_data")
            elif not a.get("underlying_data") and not as_list(a.get("shows")):
                flags.add("figure_without_data")

    if any(l == "unavailable" for l in levels):
        return "unverifiable", flags, notes
    if kinds and all(k == "expert_judgement" for k in kinds):
        return "unverifiable", flags, notes
    if any(l in ("audit", "author_only") for l in levels):
        return "audit_only", flags, notes
    if any(l is None for l in levels):
        notes.append("some supporting evidence or artefacts state no availability; treated as public")
    if any(l == "not_stated" for l in levels):
        flags.add("availability_not_stated")
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
    published = "--published" in argv
    rep = validate_path(path, published=published, revision="--revision" in argv)
    if rep is None:
        return 1
    print_report(rep, published=published, as_json="--json" in argv)
    return 1 if rep.errors else 0


def validate_path(path, published=False, revision=False):
    """Validate a claimset file, or a submission package (a BagIt folder or zip, rule 70). None if unreadable."""
    import zipfile
    p = Path(path)
    if p.is_dir() or (p.is_file() and zipfile.is_zipfile(p)):
        try:
            prep, doc = validate_package(p)
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
            return None
        if doc is None:
            return prep
        rep = validate(doc, published=published, revision=revision)
        rep.errors[:0] = prep.errors
        return rep
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        return None
    return validate(doc, published=published, local_root=claimset_folder(path), revision=revision,
                    file_name=Path(path).name)


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
