# scripts/make_labeled_sample.py
import sys, os, math
from pathlib import Path
import pandas as pd

# Uses your dataset_loader parsing for raw lines
try:
    from backend.dataset_loader import _parse_raw_line, _parse_data_field
except Exception:
    # fallback minimal parser if import fails
    def _parse_raw_line(s):
        return {"ts": None, "can_id": None, "dlc": None, "data": []}
    def _parse_data_field(s):
        return []

def parse_file_to_df(path, max_rows=None, treat_as_raw=True):
    path = Path(path)
    rows = []
    if treat_as_raw:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if max_rows is not None and i >= max_rows:
                    break
                parsed = _parse_raw_line(line)
                # keep original raw line for debugging
                parsed["raw_line"] = line.strip()
                rows.append(parsed)
    else:
        # try pandas
        try:
            df = pd.read_csv(path, nrows=max_rows)
            return df
        except Exception:
            # fallback to raw parsing
            return parse_file_to_df(path, max_rows=max_rows, treat_as_raw=True)
    return pd.DataFrame(rows)

def main():
    if len(sys.argv) < 4:
        print("Usage: python make_labeled_sample.py <attack_file> <benign_file> <out_csv> [max_rows_per_file]")
        sys.exit(2)
    attack_file = sys.argv[1]
    benign_file = sys.argv[2]
    out_csv = sys.argv[3]
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

    print("Reading attack file:", attack_file)
    df_a = parse_file_to_df(attack_file, max_rows=max_rows)
    print("attack rows:", len(df_a))
    df_a["attack"] = 1

    print("Reading benign file:", benign_file)
    df_b = parse_file_to_df(benign_file, max_rows=max_rows)
    print("benign rows:", len(df_b))
    df_b["attack"] = 0

    # concat and shuffle a little
    df_all = pd.concat([df_a, df_b], ignore_index=True, sort=False)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # persist a compact CSV: keep raw_line and attack to be safe for loader
    # dataset_loader will handle single-column raw lines and parse them if needed
    df_all_to_write = pd.DataFrame({"raw_line": df_all.get("raw_line", [""] * len(df_all)), "attack": df_all["attack"]})
    df_all_to_write.to_csv(out_csv, index=False, encoding="utf-8")
    print("Wrote labeled sample:", out_csv, "rows:", len(df_all_to_write))

if __name__ == "__main__":
    main()
