"""
Follow-up analyses of the zero-shot QA runs (method `none`, split `full`) that go beyond RESULTS_v2.md:

  A. grammatical-agreement effect: does a job title whose grammatical gender matches the candidate raise «так»?
  B. flips by item type (relevant vs irrelevant experience) and their direction;
  C. professions with the most consistent gender effect across models.

Writes results/tables/insights_*.csv and prints the tables. Usage: python -m scripts.insights [--B 2000]
"""
import argparse

import numpy as np
import pandas as pd

from scripts.stats import ROOT, bootstrap, ci, load_runs

AGREEMENT = {  # name: (agreeing group, disagreeing group) for the same candidate
    "woman": ("woman_fem", "woman_nofem"),
    "man": ("man_nofem", "man_fem"),
}


def agreement_metrics(w):
    out = {}
    for who, (agree, disagree) in AGREEMENT.items():
        out[f"yes_{who}"] = (w[f"pred_{agree}"].astype(int) - w[f"pred_{disagree}"].astype(int)).mean()
        out[f"margin_{who}"] = (w[f"margin_{agree}"] - w[f"margin_{disagree}"]).mean()
    return out


def flip_metrics(w):
    out = {}
    for rel, sub in (("relevant", w[w.is_correct.astype(bool)]), ("irrelevant", w[~w.is_correct.astype(bool)])):
        for comp, other in (("nofem", "woman_nofem"), ("fem", "woman_fem")):
            m, f = sub.pred_man_nofem.astype(int), sub[f"pred_{other}"].astype(int)
            out[f"qadiff_{comp}_{rel}"] = (m != f).mean()
            out[f"net_man_{comp}_{rel}"] = (m - f).mean()  # > 0: man gets «так» more often
    return out


def boot_table(runs, fn, B):
    rows = []
    for (model, method, seed, metric, split), w in sorted(runs.items()):
        w = w[~w.fem_identical.astype(bool)]
        point, reps = bootstrap({0: w}, fn, B)
        for k, v in point[0].items():
            lo, hi = ci(reps[0][k])
            rows.append({"model": model, "metric": k, "value": v, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t["cell"] = [f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if lo > 0 or hi < 0 else "")
                 for v, lo, hi in zip(t.value, t.lo, t.hi)]
    return t


def professions(runs):
    """Per profession: net man-favoured flips (base-form titles) summed over models and items, and how many models
    show a net effect in each direction; mean margin_diff (man - woman) across models."""
    per = []
    for (model, *_), w in runs.items():
        d = w.assign(net=w.pred_man_nofem.astype(int) - w.pred_woman_nofem.astype(int),
                     mdiff=w.margin_man_nofem - w.margin_woman_nofem)
        per.append(d.groupby("profession").agg(net=("net", "sum"), mdiff=("mdiff", "mean")).assign(model=model))
    p = pd.concat(per).reset_index()
    out = p.groupby("profession").agg(net_flips=("net", "sum"),
                                      models_man=("net", lambda x: int((x > 0).sum())),
                                      models_woman=("net", lambda x: int((x < 0).sum())),
                                      margin_diff=("mdiff", "mean")).reset_index()
    return out.sort_values("margin_diff", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    runs = {k: v for k, v in load_runs(ROOT / "results" / "raw").items()
            if k[1] == "none" and k[2] is None and k[3] == "qa" and k[4] == "full"}
    out = ROOT / "results" / "tables"
    agree = boot_table(runs, agreement_metrics, a.B)
    flips = boot_table(runs, flip_metrics, a.B)
    prof = professions(runs)
    agree.to_csv(out / "insights_agreement.csv", index=False)
    flips.to_csv(out / "insights_flips_by_relevance.csv", index=False)
    prof.to_csv(out / "insights_professions.csv", index=False)
    pd.set_option("display.width", 250)
    print(agree.pivot(index="model", columns="metric", values="cell").to_string())
    print(flips.pivot(index="model", columns="metric", values="cell").T.to_string())
    print(prof.head(15).to_string(index=False))
    print(prof.tail(15).to_string(index=False))
    print("models with net man-favoured flips per profession:",
          (prof.models_man - prof.models_woman).describe().round(2).to_dict())


if __name__ == "__main__":
    main()
