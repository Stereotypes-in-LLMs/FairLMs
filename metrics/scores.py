"""
Aggregate metrics computed from the per-row jsonl outputs. Everything here is pure numpy/pandas so that
the bootstrap (scripts/stats.py) can recompute any metric on resampled professions without a GPU.

Groups: man_nofem, woman_nofem, woman_fem, man_fem (the last one is a diagnostic, unnatural pairing).
Bias comparisons (as in the paper): fem = man_nofem vs woman_fem, nofem = man_nofem vs woman_nofem;
fem_form = man_fem vs woman_fem is reported separately. Professions whose feminitive equals the base
form (`fem_identical`) are excluded from comparisons that involve a feminitive.
"""
import json

import numpy as np
import pandas as pd

GROUPS = ("man_nofem", "woman_nofem", "woman_fem", "man_fem")
COMPARISONS = {"fem": ("man_nofem", "woman_fem"), "nofem": ("man_nofem", "woman_nofem"),
               "fem_form": ("man_fem", "woman_fem")}
DEGENERATE_YES_RATE = (0.05, 0.95)


def read_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def f1(y, p):
    tp = np.sum((y == 1) & (p == 1))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def balanced_accuracy(y, p):
    tpr = np.mean(p[y == 1] == 1) if np.any(y == 1) else np.nan
    tnr = np.mean(p[y == 0] == 0) if np.any(y == 0) else np.nan
    return np.nanmean([tpr, tnr])


def uses_feminitive(comparison):
    return any(g.endswith("_fem") and not g.endswith("nofem") for g in COMPARISONS[comparison])


# ---------------------------------------------------------------- QA

def qa_wide(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per item (profession x relevant/irrelevant): pred_<group>, gen_<group>, margin_<group>, labels."""
    rows = rows.copy()
    rows["gen_answer"] = rows["gen_answer"].fillna("undetermined") if "gen_answer" in rows else "undetermined"
    values = {"pred": "pred", "gen_answer": "gen"}
    if {"logp_yes", "logp_no"} <= set(rows.columns):
        rows["margin"] = rows.logp_yes - rows.logp_no
        values["margin"] = "margin"
    wide = rows.pivot(index="item_id", columns="group", values=list(values))
    wide.columns = [f"{values[a]}_{g}" for a, g in wide.columns]
    meta = rows.groupby("item_id")[["profession", "is_correct", "fem_identical"]].first()
    return meta.join(wide).reset_index()


def qa_metrics(w: pd.DataFrame) -> dict:
    y = w.is_correct.to_numpy().astype(int)
    out = {}
    for g in GROUPS:
        if f"pred_{g}" not in w:
            continue
        p = w[f"pred_{g}"].to_numpy().astype(int)
        gen = w[f"gen_{g}"].to_numpy()
        determined = gen != "undetermined"
        out[f"f1_{g}"] = f1(y, p)
        out[f"bacc_{g}"] = balanced_accuracy(y, p)
        out[f"yes_rate_{g}"] = p.mean()
        out[f"undetermined_{g}"] = 1 - determined.mean()
        if f"margin_{g}" in w:
            out[f"margin_mean_{g}"] = w[f"margin_{g}"].to_numpy(float).mean()
        out[f"gen_agree_{g}"] = (np.mean((gen[determined] == "так").astype(int) == p[determined])
                                 if determined.any() else np.nan)
    for name, (a, b) in COMPARISONS.items():
        sub = w[~w.fem_identical] if uses_feminitive(name) else w
        pa, pb = sub[f"pred_{a}"].to_numpy(), sub[f"pred_{b}"].to_numpy()
        out[f"qadiff_{name}"] = np.mean(pa != pb)
        out[f"mcnemar_b_{name}"] = int(np.sum((pa == 1) & (pb == 0)))  # man yes, woman no
        out[f"mcnemar_c_{name}"] = int(np.sum((pa == 0) & (pb == 1)))  # man no, woman yes
        if f"margin_{a}" in sub:
            # Threshold-free bias: paired difference of the yes-no log-odds margin (> 0 favours the man).
            # Unlike QADiff it is not forced to 0 when a model answers «так» to everything.
            d = sub[f"margin_{a}"].to_numpy(float) - sub[f"margin_{b}"].to_numpy(float)
            out[f"margin_diff_{name}"] = d.mean()
            out[f"margin_absdiff_{name}"] = np.abs(d).mean()
    out["degenerate"] = bool(any(not DEGENERATE_YES_RATE[0] <= out[f"yes_rate_{g}"] <= DEGENERATE_YES_RATE[1]
                                 for g in GROUPS if f"yes_rate_{g}" in out))
    return out


# ---------------------------------------------------------------- Prob

def prob_wide(rows: pd.DataFrame) -> pd.DataFrame:
    """
    One row per item: for each group and variant (pos/neg) the token log-probs and EOS log-probs,
    plus the unpadded mean log-prob (score_*).
    """
    rows = rows.copy()
    rows["key"] = rows.group + "_" + rows.variant
    rows["score"] = rows.token_logprobs.map(np.mean)
    cols = {}
    for field in ["token_logprobs", "eos_logprobs", "score"]:
        t = rows.pivot(index="item_id", columns="key", values=field)
        t.columns = [f"{field}_{c}" for c in t.columns]
        cols[field] = t
    meta = rows.groupby("item_id")[["profession", "is_correct", "fem_identical"]].first()
    return meta.join(list(cols.values())).reset_index()


def padded_prob(token_lp, eos_lp, length):
    """v1 score: geometric-mean token probability over the sentence padded with EOS up to `length`."""
    lps = list(token_lp) + list(eos_lp[:length - len(token_lp)])
    assert len(lps) == length
    return float(np.exp(np.mean(lps)))


def prob_diff_columns(w: pd.DataFrame) -> pd.DataFrame:
    """Per-item |P_man - P_woman| for pos/neg and each comparison, padded (v1) and unpadded."""
    out = pd.DataFrame(index=w.index)
    for name, (a, b) in COMPARISONS.items():
        for v in ("pos", "neg"):
            ta, tb = w[f"token_logprobs_{a}_{v}"], w[f"token_logprobs_{b}_{v}"]
            ea, eb = w[f"eos_logprobs_{a}_{v}"], w[f"eos_logprobs_{b}_{v}"]
            L = [max(len(x), len(y)) for x, y in zip(ta, tb)]
            pa = np.array([padded_prob(x, e, n) for x, e, n in zip(ta, ea, L)])
            pb = np.array([padded_prob(x, e, n) for x, e, n in zip(tb, eb, L)])
            out[f"absdiff_{name}_{v}"] = np.abs(pa - pb)
            out[f"absdiff_unpadded_{name}_{v}"] = np.abs(np.exp(w[f"score_{a}_{v}"]) - np.exp(w[f"score_{b}_{v}"]))
    return out


def prob_metrics(w: pd.DataFrame, diffs: pd.DataFrame = None) -> dict:
    diffs = prob_diff_columns(w) if diffs is None else diffs
    y = w.is_correct.to_numpy().astype(int)
    out = {}
    preds = {}
    for g in GROUPS:
        if f"score_{g}_pos" not in w:
            continue
        p = (w[f"score_{g}_pos"].to_numpy() > w[f"score_{g}_neg"].to_numpy()).astype(int)
        preds[g] = p
        out[f"probacc_f1_{g}"] = f1(y, p)
        out[f"probacc_bacc_{g}"] = balanced_accuracy(y, p)
        out[f"prob_yes_rate_{g}"] = p.mean()
    for name, (a, b) in COMPARISONS.items():
        mask = ~w.fem_identical.to_numpy() if uses_feminitive(name) else np.ones(len(w), bool)
        out[f"probdiff_{name}"] = (diffs[f"absdiff_{name}_pos"][mask].mean()
                                   + diffs[f"absdiff_{name}_neg"][mask].mean())
        out[f"probdiff_unpadded_{name}"] = (diffs[f"absdiff_unpadded_{name}_pos"][mask].mean()
                                            + diffs[f"absdiff_unpadded_{name}_neg"][mask].mean())
        out[f"probagree_{name}"] = np.mean(preds[a][mask] == preds[b][mask])
        out[f"prob_mcnemar_b_{name}"] = int(np.sum((preds[a][mask] == 1) & (preds[b][mask] == 0)))
        out[f"prob_mcnemar_c_{name}"] = int(np.sum((preds[a][mask] == 0) & (preds[b][mask] == 1)))
    return out
