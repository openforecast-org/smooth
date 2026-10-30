"""The pilot series: 20 rows outside the pre-registered subset (seed 11). Writes pilot.csv."""

import csv
import random
import sys

taken = {int(r["row"]) for r in csv.DictReader(open(sys.argv[1]))}
rows = sorted(random.Random(11).sample([i for i in range(30490) if i not in taken], 20))
with open("pilot.csv", "w", newline="") as f:
    csv.writer(f).writerows([["row"], *[[r] for r in rows]])
