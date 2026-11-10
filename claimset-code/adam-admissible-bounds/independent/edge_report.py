"""Breakdown of the oracle's per-model output (oracle_models.json of a run):
the models in the band of 1e-8 around 1, the moduli on admissible models and the
two examples.  Usage: edge_report.py ORACLE_MODELS.json PERLAG_MODELS.json"""

import json
import os
import sys
from fractions import Fraction

EDGE = 1e-8


def near(r):
    return r["hi"] >= 1 - EDGE and r["lo"] <= 1 + EDGE


def mid(r):
    return (r["lo"] + r["hi"]) / 2


def main():
    oracle = json.load(open(sys.argv[1]))
    grid = {m["index"]: m for m in json.load(open(sys.argv[2]))["models"]}
    res = []

    def put(variable, value, **cond):
        res.append({"variable": variable, "value": float(value), "conditions": cond})

    for s in oracle["sets"]:
        ms = [m for m in s["models"] if "error" not in m]
        edge = [m for m in ms if near(m["gcd"]) or near(m["cancelled"])]
        for fam in sorted({m["family"] for m in edge}):
            sub = [m for m in edge if m["family"] == fam]
            put("models in the edge band", len(sub), set=s["set"], family=fam)
            put("max distance of the certified largest mode modulus from 1, edge models",
                max(max(abs(m["gcd"]["lo"] - 1), abs(m["gcd"]["hi"] - 1)) for m in sub), set=s["set"], family=fam)
            put("edge models with roots proved on the unit circle", sum(m["gcd"]["unit_roots"] > 0 for m in sub),
                set=s["set"], family=fam)
            for key, name in (("pbh_max", "PBH reference"), ("max_lambda_cancelled", "prototype's cancellation")):
                if key not in sub[0]["prototype"]:
                    continue
                ref = "gcd" if key == "pbh_max" else "cancelled"
                diff = sum((m["prototype"][key] < 1) != (m[ref]["status"] == "admissible") for m in sub)
                put(f"edge models whose decision by the {name} differs from the exact oracle", diff,
                    set=s["set"], family=fam)
                put(f"max abs difference of the largest mode modulus, {name} against the exact oracle, edge models",
                    max(abs(m["prototype"][key] - mid(m[ref])) for m in sub), set=s["set"], family=fam)
            if s["set"] == "perlag" and fam == "ETS(A,N,A)":
                line = 0
                for m in sub:
                    g = grid[m["index"]]["g"]
                    if abs(Fraction(g[0]) + Fraction(g[1]) - 2) < Fraction(1, 10 ** 12):
                        line += 1
                put("edge models with alpha + gamma = 2 within 1e-12", line, set=s["set"], family=fam)
        # admissible models outside the band: moduli of the reference against the oracle
        kept = [m for m in ms if not (near(m["gcd"]) or near(m["cancelled"]))]
        for key, ref, name in (("pbh_max", "gcd", "PBH reference"),
                               ("max_lambda_cancelled", "cancelled", "prototype's roots after cancellation")):
            adm = [m for m in kept if key in m["prototype"] and m[ref]["status"] == "admissible"]
            if adm:
                put(f"max abs difference of the largest mode modulus, {name} against the exact oracle, "
                    "admissible models", max(abs(m["prototype"][key] - mid(m[ref])) for m in adm), set=s["set"])
                put("admissible models compared", len(adm), set=s["set"], reference=name)
        if s["set"] == "examples":
            for m in ms:
                p = m["prototype"]
                put("largest mode modulus, exact oracle (theta_G)", mid(m["gcd"]), model=m["family"])
                put("abs difference of the largest mode modulus, PBH reference against the exact oracle",
                    abs(p["pbh_max"] - mid(m["gcd"])), model=m["family"])
                put("abs difference of the largest mode modulus, prototype's roots after cancellation "
                    "against the exact oracle", abs(p["max_lambda_cancelled"] - mid(m["cancelled"])), model=m["family"])
                put("degree of theta, exact", m["degree_theta"], model=m["family"])
                put("degree of the shared cyclotomic factor, exact", m["degree_cyclotomic"], model=m["family"])
                put("degree of the exact gcd of theta and phi", m["degree_gcd"], model=m["family"])
    for r in res:
        print(r["variable"], r["conditions"], r["value"])
    out = os.environ.get("CLAIMSET_RESULTS")
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
