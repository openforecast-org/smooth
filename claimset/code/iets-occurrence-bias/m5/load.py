"""M5 series as SB23 section 5 uses them: days 1-1,969 of the evaluation files, leading
zeros removed, the last 14 days held out."""

import csv

H = 14


def read_series(train, test, rows):
    # {row: (item_id, store_id, values without leading zeros)} for the given rows
    wanted, out = set(rows), {}
    with open(train) as ftr, open(test) as fte:
        rtr, rte = csv.reader(ftr), csv.reader(fte)
        next(rtr), next(rte)
        for i, (a, b) in enumerate(zip(rtr, rte)):
            if i in wanted:
                y = [float(v) for v in a[5:] + b[5:]]
                first = next((k for k, v in enumerate(y) if v != 0), len(y))
                out[i] = (a[0], a[3], y[first:])
    return out
