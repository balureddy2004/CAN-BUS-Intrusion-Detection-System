# write the script
import os
import pandas as pd
from pathlib import Path

SRC = Path("datasets/ocs-lab")
DST = Path("datasets/ocs-lab-dev")
DST.mkdir(parents=True, exist_ok=True)

# how many rows to sample from each big file
N = 10000

for f in SRC.iterdir():
    if f.is_file() and f.suffix.lower()==".csv":
        print("Sampling", f.name)
        try:
            # read first N rows safely
            df = pd.read_csv(f, nrows=N, low_memory=True)
            out = DST / f.name
            df.to_csv(out, index=False)
            print(" -> wrote", out, "rows:", len(df))
        except Exception as e:
            print("ERROR reading", f, ":", e)
print("done.")

