"""
Cohen's kappa per criterion (a-d) between two filled copies of data/v2/annotation/annotation_sheet.csv.
Values: 1/0 (also accepts yes/no, так/ні, true/false). Prints kappa and the professions they disagree on.

    python scripts/annotation_kappa.py annotator1.csv annotator2.csv
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

CRITERIA = ["a_relevant_ok", "b_irrelevant_ok", "c_feminitive_ok", "d_grammatical_ok"]
TRUE = {"1", "yes", "так", "true", "y", "+"}


def load(path):
    df = pd.read_csv(path, dtype=str).set_index("profession")
    for c in CRITERIA:
        df[c] = df[c].str.strip().str.lower().map(lambda v: int(v in TRUE) if isinstance(v, str) and v else None)
    return df


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = a.index.intersection(b.index)
    for c in CRITERIA:
        both = a.loc[common, c].notna() & b.loc[common, c].notna()
        x, y = a.loc[common][both][c].astype(int), b.loc[common][both][c].astype(int)
        disagree = list(x.index[x != y])
        kappa = cohen_kappa_score(x, y) if x.nunique() > 1 or y.nunique() > 1 else float("nan")
        print(f"{c}: n={both.sum()} agreement={(x == y).mean():.3f} kappa={kappa:.3f} disagreements={disagree}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
