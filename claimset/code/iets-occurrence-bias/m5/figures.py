"""MCB plots and the table of aggregate error measures from an evaluation run. Usage:
    figures.py <series_measures.csv> <summary_measures.csv>
Writes to $CLAIMSET_RUN_DIR: rmcb.png (greybox rmcb() at 99% on RMSSE and on the scaled
pinball at each level, drawn by its own plot method) and aggregate_measures.csv (the mean
of each measure per method, from summary_measures.csv).
"""

import csv
import os
import sys

import matplotlib
import pandas as pd
from greybox import rmcb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALPHA = 0.01
MEASURES = ["RMSSE"] + [f"pinball{t}" for t in (0.8, 0.9, 0.95, 0.99)]
TABLE = ["RMSSE", "sCE", "sAPIS"] + [f"{k}{t}" for k in ("pinball", "coverage")
                                     for t in (0.8, 0.9, 0.95, 0.99)]


def main(series_file, summary_file):
    outdir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    series = pd.read_csv(series_file)
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(11, 5.2 * len(MEASURES)))
    for ax, measure in zip(axes, MEASURES):
        wide = series.pivot(index="row", columns="method", values=measure).dropna(axis=1)
        res = rmcb(wide, level=1 - ALPHA, outplot="none")
        res.plot(ax=ax, main=f"{measure}: mean ranks, {100 * (1 - ALPHA):g}% MCB intervals"
                 f" (p-value {res.p_value:.2g})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmcb.png"), dpi=130)

    summary = pd.read_csv(summary_file)
    table = summary.pivot(index="method", columns="measure", values="mean")
    table[[c for c in TABLE if c in table]].to_csv(os.path.join(outdir, "aggregate_measures.csv"),
                                                    quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main(*sys.argv[1:3])
