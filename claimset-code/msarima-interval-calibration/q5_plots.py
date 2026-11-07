"""Calibration curves of the full predictive distribution (the 99 quantiles) from recorded
outputs: the share of observations at or below each quantile against its level (top row), and
that share minus the level (bottom row)."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1_simulate import TAUS

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
RESULTS = []


def figure(panels, title, path):
    # panels: list of (panel title, {line label: 99 coverages})
    fig, ax = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.4), sharex=True, sharey="row", squeeze=False)
    for j, (name, lines) in enumerate(panels):
        ax[0, j].plot([0, 1], [0, 1], color="#888888", lw=1, ls="--")
        ax[1, j].axhline(0, color="#888888", lw=1, ls="--")
        for k, (label, c) in enumerate(lines.items()):
            ax[0, j].plot(TAUS, c, color=COLOURS[k], lw=2, label=label)
            ax[1, j].plot(TAUS, c - TAUS, color=COLOURS[k], lw=2)
            RESULTS.append(dict(quantity="mean absolute calibration error", value=float(np.abs(c - TAUS).mean()),
                                conditions=dict(figure=os.path.basename(path), panel=name, line=label)))
        ax[0, j].set_title(name, fontsize=10)
        ax[1, j].set_xlabel("nominal level of the quantile")
        for a in ax[:, j]:
            a.grid(color="#e5e5e5", lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].set_ylabel("share of observations at or below")
    ax[1, 0].set_ylabel("share minus nominal")
    ax[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sim_curves(files, cases, T):
    d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["dgp", "T", "case", "h"])
    cols = [f"q{t:.2f}" for t in TAUS]
    d = d[d["T"] == T]
    return {label: d[d["case"] == case][cols].mean().values for label, case in cases.items() if (d["case"] == case).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs="+", required=True)
    ap.add_argument("--sim108", required=True)
    ap.add_argument("--v108", required=True)
    ap.add_argument("--v130", required=True)
    ap.add_argument("--saved-ets", required=True)
    ap.add_argument("--q5", required=True)
    ap.add_argument("--m4")
    a = ap.parse_args()
    out = os.environ.get("CLAIMSET_RUN_DIR", ".")

    cases = {"true parameters": "T_backcasting", "estimated parameters": "E_backcasting", "selected orders": "S_backcasting"}
    figure([(f"T={T}", sim_curves(a.sim, cases, T)) for T in (15, 25, 50, 100, 300)],
           "Simulation, 11 Gaussian DGPs, AutoMSARIMA 1.3.0 with backcasting", os.path.join(out, "calibration_simulation.png"))

    old, new = np.load(a.v108)["cov"], np.load(a.v130)["cov"]
    panels = [(f"benchmark, {ini}", {"1.0.8": np.nanmean(old[j], axis=0), "1.3.0": np.nanmean(new[j], axis=0)})
              for j, ini in enumerate(("backcasting", "optimal"))]
    for case, label in (("T_backcasting", "true parameters"), ("E_backcasting", "estimated"), ("S_backcasting", "selected")):
        c108, c130 = sim_curves([a.sim108], {"1.0.8": case}, 100), sim_curves(a.sim, {"1.3.0": case}, 100)
        panels.append((f"simulation T=100, {label}", {**c108, **c130}))
    figure(panels, "AutoMSARIMA 1.0.8 against 1.3.0", os.path.join(out, "calibration_versions.png"))

    for data, path in (("benchmark", a.q5), ("m4", a.m4)):
        if not path:
            continue
        res = pd.read_csv(os.path.join(path, f"q5_{data}_series.csv.gz"))
        cov = np.load(os.path.join(path, f"q5_{data}_coverage.npy"))
        res["tercile"] = pd.qcut(res["level"], 3, labels=["low", "mid", "high"])
        names = {"msarima": "auto.msarima", "adam_select": "auto.adam NNN, dnorm or dlnorm"}
        lines = {names[m]: np.nanmean(cov[res["method"] == m], axis=0) for m in names}
        if data == "benchmark":
            lines["ETS (AutoADAM ZXZ, saved)"] = np.nanmean(np.load(a.saved_ets)[1], axis=0)
        sel = (res["method"] == "adam_select").values
        panels = [("all series", lines)]
        panels += [(f"{t} level tercile", {names[m]: np.nanmean(cov[(res["method"] == m).values & (res["tercile"] == t).values], axis=0)
                                           for m in names}) for t in ("low", "mid", "high")]
        panels.append(("auto.adam by selected distribution",
                       {f"chose {d}": np.nanmean(cov[sel & (res["distribution"] == d).values], axis=0) for d in ("dnorm", "dlnorm")}))
        label = "M1, M3 and Tourism (5,315 series)" if data == "benchmark" else "M4 sample"
        figure(panels, f"{label}, backcasting", os.path.join(out, f"calibration_{data}.png"))

    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(RESULTS, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
