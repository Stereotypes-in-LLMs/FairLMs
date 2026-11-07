"""
Builds GBEM-UA v2 from the raw v1 generation file (two GPT-4 passes, 5608 rows per config).

Rules (see reports/DATASET_v2.md):
* one 8-row design block per profession: the first complete block in file order (= first pass);
* incomplete chunks (4 "амбасадор" rows) and extra blocks ("урядовець") are dropped;
* professions whose feminitive equals the base form are flagged `fem_identical`;
* group split by profession (175 train / rest test, seed 42) + `full` with all professions.

Usage: python scripts/build_v2.py [--seed 42] [--n-train 175]
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "v2"
V1 = {"acc": (ROOT / "data/acc_dataset/train_dataset_acc.csv", ROOT / "data/acc_dataset/test_dataset_acc.csv"),
      "prob": (ROOT / "data/prob_dataset/train_dataset.csv", ROOT / "data/prob_dataset/test_dataset.csv")}

META = ["profession", "experience", "is_male", "is_correct", "is_feminitive"]
JOB_RE = {"acc": re.compile(r"посаду (\S+)\?$"), "prob": re.compile(r"Посаду (\S+) (?:він|вона) BLANK")}
GROUPS = {(True, False): "man_nofem", (False, False): "woman_nofem",
          (False, True): "woman_fem", (True, True): "man_fem"}


def load_raw(config):
    df = pd.read_csv(RAW / f"full_v1_{config}.csv")
    for col in ["sentence", "profession", "experience"]:
        df[col] = df[col].str.strip()
    df["raw_row"] = range(len(df))
    return df


def is_complete_block(block):
    """8 rows = 2 experiences (1 relevant, 1 irrelevant) x 2 genders x 2 job forms, each once."""
    if len(block) != 8:
        return False
    cells = set(zip(block.is_male, block.is_feminitive, block.is_correct))
    exps = block.groupby("is_correct").experience.nunique()
    return len(cells) == 8 and len(exps) == 2 and (exps == 1).all()


def select_blocks(df):
    """Chunks consecutive same-profession runs into 8-row blocks and keeps the first complete block."""
    run_id = (df.profession != df.profession.shift()).cumsum()
    kept, log = {}, {"incomplete_chunks": [], "extra_blocks": []}
    for _, run in df.groupby(run_id, sort=False):
        for start in range(0, len(run), 8):
            chunk = run.iloc[start:start + 8]
            prof = chunk.profession.iloc[0]
            if not is_complete_block(chunk):
                log["incomplete_chunks"].append({"profession": prof, "rows": chunk.raw_row.tolist()})
            elif prof in kept:
                log["extra_blocks"].append({"profession": prof, "rows": chunk.raw_row.tolist()})
            else:
                kept[prof] = chunk.raw_row.tolist()
    return kept, log


def annotate(df, config):
    df = df.copy()
    df["job_form"] = df.sentence.map(lambda s: JOB_RE[config].search(s).group(1))
    df["group"] = [GROUPS[(m, f)] for m, f in zip(df.is_male, df.is_feminitive)]
    fem_forms = df[df.is_feminitive].groupby("profession").job_form.agg(set)
    base_forms = df[~df.is_feminitive].groupby("profession").job_form.agg(set)
    identical = {p for p in fem_forms.index if fem_forms[p] & base_forms[p]}
    df["fem_identical"] = df.profession.isin(identical)
    # Stable ids: one item = one (profession, experience) pair, shared by the 4 gender variants.
    df["item_id"] = df.profession + "|" + df.is_correct.map({True: "rel", False: "irr"})
    df["row_id"] = df.item_id + "|" + df.group
    return df


def check_symmetry(df):
    """Same checks as the v1 evaluators: each item has the 4 groups with identical metadata."""
    for item, g in df.groupby("item_id"):
        assert sorted(g.group) == sorted(GROUPS.values()), item
        assert g.experience.nunique() == 1 and g.is_correct.nunique() == 1, item
        for _, row in g.iterrows():
            pronoun = "він" if row.is_male else "вона"
            assert row.sentence.startswith("Кандидат " if row.is_male else "Кандидатка "), row.row_id
            assert f" {pronoun} " in row.sentence, row.row_id


def split_professions(professions, n_train, seed):
    profs = pd.Series(sorted(professions))
    gss = GroupShuffleSplit(n_splits=1, train_size=n_train / len(profs), random_state=seed)
    tr_idx, te_idx = next(gss.split(profs, groups=profs))
    train, test = set(profs.iloc[tr_idx]), set(profs.iloc[te_idx])
    assert len(train) == n_train and train.isdisjoint(test) and len(train | test) == len(profs)
    return train, test


def leak_stats(train, test):
    tr_s, te_s = set(train.sentence.str.strip()), set(test.sentence.str.strip())
    test_in_train = test[test.sentence.str.strip().isin(tr_s)]
    return {"rows_train": len(train), "rows_test": len(test),
            "professions_train": train.profession.str.strip().nunique(),
            "professions_test": test.profession.str.strip().nunique(),
            "profession_overlap": len(set(train.profession.str.strip()) & set(test.profession.str.strip())),
            "test_sentences_in_train": len(test_in_train),
            "test_positives_in_train": int(test_in_train.is_correct.sum()),
            "test_negatives_in_train": int((~test_in_train.is_correct).sum()),
            "dup_sentences_train": int(train.sentence.duplicated().sum()),
            "dup_sentences_test": int(test.sentence.duplicated().sum()),
            "experience_leading_space_test": int(test.experience.str.startswith(" ").sum())}


def main(seed, n_train):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {c: load_raw(c) for c in ["acc", "prob"]}
    assert (raw["acc"][META].values == raw["prob"][META].values).all(), "acc/prob raw files are not aligned"

    kept, log = select_blocks(raw["acc"])
    rows = sorted(r for rs in kept.values() for r in rs)
    data = {c: annotate(raw[c].iloc[rows].reset_index(drop=True), c) for c in raw}
    acc, prob = data["acc"], data["prob"]
    assert (acc.job_form == prob.job_form).all(), "job forms differ between acc and prob"
    assert (acc[META + ["row_id"]].values == prob[META + ["row_id"]].values).all()
    for df in data.values():
        check_symmetry(df)
        dups = df[df.sentence.duplicated(keep=False)]
        assert dups.fem_identical.all(), f"unexpected duplicate sentences: {dups.row_id.tolist()}"

    train_p, test_p = split_professions(acc.profession.unique(), n_train, seed)
    checks = {}
    for config, df in data.items():
        df["split"] = df.profession.map(lambda p: "train" if p in train_p else "test")
        cols = ["row_id", "item_id", "sentence", "profession", "job_form", "experience", "group",
                "is_male", "is_correct", "is_feminitive", "fem_identical", "split", "raw_row"]
        df = df[cols]
        parts = {"full": df, "train": df[df.split == "train"], "test": df[df.split == "test"]}
        for name, part in parts.items():
            part.to_csv(OUT / f"{config}_{name}.csv", index=False)
        tr, te = parts["train"], parts["test"]
        checks[config] = {"profession_overlap": len(set(tr.profession) & set(te.profession)),
                          "sentence_overlap": len(set(tr.sentence) & set(te.sentence)),
                          "dup_sentences": {k: int(v.sentence.duplicated().sum()) for k, v in parts.items()},
                          "rows": {k: len(v) for k, v in parts.items()},
                          "professions": {k: v.profession.nunique() for k, v in parts.items()},
                          "positives": {k: int(v.is_correct.sum()) for k, v in parts.items()}}
        assert checks[config]["profession_overlap"] == 0 and checks[config]["sentence_overlap"] == 0
        # Only fem_identical professions may repeat a sentence (base form == feminitive).
        checks[config]["dup_sentences_not_fem_identical"] = int(
            df[~df.fem_identical].sentence.duplicated().sum())
        assert checks[config]["dup_sentences_not_fem_identical"] == 0

    fem_identical = sorted(acc[acc.fem_identical].profession.unique())
    v1 = {c: leak_stats(*(pd.read_csv(p) for p in V1[c])) for c in V1}
    report = {"seed": seed, "n_train_professions": n_train,
              "raw_rows": len(raw["acc"]), "raw_professions": raw["acc"].profession.nunique(),
              "dropped": log, "fem_identical": fem_identical,
              "checks_v2": checks, "v1": v1}
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (OUT / "split_professions.json").write_text(json.dumps(
        {"seed": seed, "train": sorted(train_p), "test": sorted(test_p)}, ensure_ascii=False, indent=2))
    write_annotation_sheet(acc)
    print(json.dumps({k: report[k] for k in ["raw_rows", "raw_professions", "fem_identical"]}, ensure_ascii=False))
    print(json.dumps(checks, indent=1))
    print("v1:", json.dumps(v1, indent=1))


def write_annotation_sheet(acc):
    """One row per profession for the native-speaker check (criteria a-d, plan 1.3)."""
    rows = []
    for prof, g in acc.groupby("profession"):
        by = {(r.group, r.is_correct): r for r in g.itertuples()}
        rows.append({"profession": prof,
                     "job_form_base": by[("man_nofem", True)].job_form,
                     "job_form_fem": by[("woman_fem", True)].job_form,
                     "fem_identical": bool(g.fem_identical.iloc[0]),
                     "relevant_experience": by[("man_nofem", True)].experience,
                     "irrelevant_experience": by[("man_nofem", False)].experience,
                     **{f"sentence_{grp}": by[(grp, True)].sentence for grp in GROUPS.values()},
                     "a_relevant_ok": "", "b_irrelevant_ok": "", "c_feminitive_ok": "",
                     "d_grammatical_ok": "", "comment": ""})
    path = OUT / "annotation"
    path.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "annotation_sheet.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=175)
    a = ap.parse_args()
    main(a.seed, a.n_train)
