"""The pre-registered random subset of M5 series: 3,000 of the 30,490 rows of
sales_train_evaluation.csv, drawn uniformly without replacement with a fixed seed, before
any value of the series is read. Writes subset.csv (row index, item_id, store_id)."""

import csv
import random
import sys

SEED, SIZE = 20261029, 3000
path = sys.argv[1]
with open(path) as f:
    reader = csv.reader(f)
    header = next(reader)
    keys = [(row[0], row[3]) for row in reader]  # item_id, store_id only
rows = sorted(random.Random(SEED).sample(range(len(keys)), SIZE))
with open("subset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "item_id", "store_id"])
    writer.writerows([r, *keys[r]] for r in rows)
print(len(keys), "series;", len(rows), "drawn with seed", SEED)
