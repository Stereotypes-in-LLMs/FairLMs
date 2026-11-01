"""
R7/R8: statistics, tables, figures and the results report, from results/raw/*.jsonl (CPU only).

* cluster bootstrap over professions (B=2000, percentile 95% CI) for every metric;
* paired bootstrap of differences method - none on the same resampled professions;
* exact McNemar (man vs woman) with Holm correction;
* fine-tuning: mean +- sd over seeds, CI from the bootstrap with metrics averaged over seeds.

    python scripts/stats.py [--B 2000] [--results results] [--report reports/RESULTS_v2.md]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.scores import (COMPARISONS, GROUPS, prob_diff_columns, prob_metrics, prob_wide,  # noqa: E402
                            qa_metrics, qa_wide, read_jsonl)

ZERO_SHOT = ["none", "prompt", "soft", "hard"]
FT_METHODS = ("ft", "ftctrl")  # LoRA adapters; ftctrl = control trained on male base-form sentences only
METHOD_STYLE = {"prompt": ("#2a78d6", "o"), "soft": ("#eb6834", "s"), "hard": ("#1baf7a", "^"),
                "ft": ("#eda100", "D"), "ftctrl": ("#e87ba4", "v")}
V1_MODELS = {"qwen3": "qwen2.5-3b-instruct", "qwen7": "qwen2.5-7b-instruct", "gemma-2b": "gemma-2-2b-it",
             "gemma-9b": "gemma-2-9b-it", "llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
             "llama-3.1-8B-Instruct": "llama-3.1-8b-instruct"}
NAME_RE = re.compile(r"^(?P<model>.+)_(?P<method>none|prompt|soft|hard)(?:_(?P<tag>(?:ft|ftctrl)_s\d+|pv_[a-z0-9]+))?_(?P<metric>qa|prob)_"
                     r"(?P<split>full|train|test)$")


# ---------------------------------------------------------------- loading

def load_runs(raw_dir):
    runs = {}
    for path in sorted(glob.glob(str(Path(raw_dir) / "*.jsonl"))):
        m = NAME_RE.match(Path(path).stem)
        if not m:  # pilots and anything else
            continue
        method, seed = m["method"], None
        if m["tag"] and m["tag"].startswith("pv_"):  # prompt-variant run: "<method>@<variant>"
            method = f"{m['method']}@{m['tag'][3:]}"
        elif m["tag"]:
            method, seed = m["tag"].split("_s")[0], int(m["tag"].split("_s")[1])
        rows = read_jsonl(path)
        wide = qa_wide(rows) if m["metric"] == "qa" else prob_wide(rows)
        if m["metric"] == "prob":
            wide = pd.concat([wide, prob_diff_columns(wide)], axis=1)
        wide["split"] = rows.groupby("item_id").split.first().reindex(wide.item_id).to_numpy()
        runs[(m["model"], method, seed, m["metric"], m["split"])] = wide
    return runs


def metric_fn(metric):
    if metric == "qa":
        return lambda w: {k: v for k, v in qa_metrics(w).items() if not isinstance(v, bool)}
    return lambda w: prob_metrics(w, diffs=w)


# ---------------------------------------------------------------- bootstrap

def bootstrap(wides, fn, B, seed=0):
    """Point estimates and B cluster-bootstrap replicates for every config, on shared resamples."""
    items = sorted(set.intersection(*(set(w.item_id) for w in wides.values())))
    wides = {k: w.set_index("item_id").loc[items].reset_index() for k, w in wides.items()}
    prof = next(iter(wides.values())).profession.to_numpy()
    profs = np.unique(prof)
    idx_of = {p: np.flatnonzero(prof == p) for p in profs}
    point = {k: fn(w) for k, w in wides.items()}
    reps = {k: [] for k in wides}
    rng = np.random.default_rng(seed)
    for _ in range(B):
        idx = np.concatenate([idx_of[p] for p in rng.choice(profs, len(profs))])
        for k, w in wides.items():
            reps[k].append(fn(w.iloc[idx]))
    return point, {k: pd.DataFrame(v) for k, v in reps.items()}


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (np.nan, np.nan) if not len(x) else tuple(np.percentile(x, [2.5, 97.5]))


def fmt(v, lo=None, hi=None, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = f"{v:.{d}f}"
    return s if lo is None else f"{s} [{lo:.{d}f}, {hi:.{d}f}]"


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_p(b, c):
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


# ---------------------------------------------------------------- analysis

def analyse(runs, B):
    """Returns long tables: estimates (per config), diffs (vs none), ft seeds, memorization."""
    est, diffs, ft_rows, mem_rows = [], [], [], []
    models = sorted({k[0] for k in runs})
    for metric in ["qa", "prob"]:
        fn = metric_fn(metric)
        for model in models:
            for scope in ["full", "test"]:
                # zero-shot configs are run on `full`; `test` = the unseen professions only.
                configs = {}
                for method in ZERO_SHOT:
                    w = runs.get((model, method, None, metric, "full"))
                    if w is not None:
                        configs[(method, None)] = w if scope == "full" else w[w.split == "test"]
                if scope == "test":
                    for (mo, me, seed, mt, sp), w in runs.items():
                        if mo == model and me in FT_METHODS and mt == metric:
                            configs[(me, seed)] = w[w.split == "test"]
                if not configs:
                    continue
                point, reps = bootstrap(configs, fn, B)
                for ft_method in FT_METHODS:
                    ft_keys = [k for k in configs if k[0] == ft_method]
                    if not ft_keys:
                        continue
                    # pooled over seeds: average each replicate's metrics across seeds
                    point[(ft_method, "pooled")] = {m: np.mean([point[k][m] for k in ft_keys]) for m in point[ft_keys[0]]}
                    reps[(ft_method, "pooled")] = sum(reps[k] for k in ft_keys) / len(ft_keys)
                    for m in point[ft_keys[0]]:
                        vals = [point[k][m] for k in ft_keys]
                        ft_rows.append({"metric_family": metric, "model": model, "method": ft_method, "metric": m,
                                        "mean": np.mean(vals), "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                                        "n_seeds": len(vals), **{f"seed_{k[1]}": point[k][m] for k in ft_keys}})
                if all((f, "pooled") in point for f in FT_METHODS):
                    # does the gender-augmented adapter beat the male-only control? (paired, same resamples)
                    for m in point[("ft", "pooled")]:
                        if m.startswith(("mcnemar", "prob_mcnemar")):
                            continue
                        lo, hi = ci(reps[("ft", "pooled")][m] - reps[("ftctrl", "pooled")][m])
                        diffs.append({"family": metric, "scope": scope, "model": model, "method": "ft_vs_ftctrl",
                                      "metric": m, "delta": point[("ft", "pooled")][m] - point[("ftctrl", "pooled")][m],
                                      "lo": lo, "hi": hi})
                for (method, seed), p in point.items():
                    if method in FT_METHODS and seed != "pooled":
                        continue
                    for m, v in p.items():
                        lo, hi = ci(reps[(method, seed)][m])
                        est.append({"family": metric, "scope": scope, "model": model, "method": method,
                                    "metric": m, "value": v, "lo": lo, "hi": hi})
                    if method != "none" and ("none", None) in point:
                        for m in p:
                            if m.startswith(("mcnemar", "prob_mcnemar")):
                                continue
                            d = reps[(method, seed)][m] - reps[("none", None)][m]
                            lo, hi = ci(d)
                            diffs.append({"family": metric, "scope": scope, "model": model, "method": method,
                                          "metric": m, "delta": p[m] - point[("none", None)][m], "lo": lo, "hi": hi})
                # fem vs non-fem gap (F3) for every config
                if metric == "qa":
                    for key in point:
                        if key[0] in FT_METHODS and key[1] != "pooled":
                            continue
                        g = reps[key]["qadiff_fem"] - reps[key]["qadiff_nofem"]
                        lo, hi = ci(g)
                        est.append({"family": metric, "scope": scope, "model": model, "method": key[0],
                                    "metric": "gap_fem_minus_nofem",
                                    "value": point[key]["qadiff_fem"] - point[key]["qadiff_nofem"], "lo": lo, "hi": hi})
        # memorization: fine-tuned QA on seen (train) vs unseen (test) professions
        if metric == "qa":
            for (mo, me, seed, mt, sp), w in runs.items():
                if me in FT_METHODS and mt == "qa" and sp == "full":
                    for part in ["train", "test"]:
                        m = qa_metrics(w[w.split == part])
                        mem_rows.append({"model": mo, "method": me, "seed": seed, "professions": "seen" if part == "train" else "unseen",
                                         **{k: m[k] for k in ["f1_man_nofem", "f1_woman_nofem", "f1_woman_fem",
                                                              "qadiff_fem", "qadiff_nofem"]}})
    pv = prompt_sensitivity(runs, models, B)
    est, diffs = pd.DataFrame(est), pd.DataFrame(diffs)
    # McNemar with Holm per family (QA predictions / Prob predictions), on the full zero-shot set + ft test.
    mc = []
    for (family, prefix) in [("qa", "mcnemar"), ("prob", "prob_mcnemar")]:
        sub = est[(est.family == family) & est.metric.str.startswith(prefix + "_b_")]
        for r in sub.itertuples():
            if not ((r.scope == "full" and r.method not in FT_METHODS) or
                    (r.scope == "test" and r.method in FT_METHODS)):
                continue
            comp = r.metric[len(prefix) + 3:]
            c = est[(est.family == family) & (est.scope == r.scope) & (est.model == r.model) &
                    (est.method == r.method) & (est.metric == f"{prefix}_c_{comp}")].value.iloc[0]
            if r.method in FT_METHODS:  # pooled over seeds -> report the mean counts, test on rounded counts
                b, c = int(round(r.value)), int(round(c))
            else:
                b, c = int(r.value), int(c)
            mc.append({"family": family, "scope": r.scope, "model": r.model, "method": r.method,
                       "comparison": comp, "b_man_yes_woman_no": b, "c_man_no_woman_yes": c,
                       "p": mcnemar_p(b, c)})
    mc = pd.DataFrame(mc)
    if len(mc):
        mc["p_holm"] = np.nan
        for fam in mc.family.unique():
            sel = mc.family == fam
            mc.loc[sel, "p_holm"] = holm(mc.loc[sel, "p"])
    return est, diffs, pd.DataFrame(ft_rows), pd.DataFrame(mem_rows), mc, pv


PV_METRICS = ["f1_man_nofem", "yes_rate_man_nofem", "undetermined_man_nofem", "qadiff_fem", "qadiff_nofem",
              "margin_diff_fem", "margin_diff_nofem"]


def prompt_sensitivity(runs, models, B):
    """T8: QA metrics of the base instruction vs its paraphrases / answer-order swap (none and prompt methods)."""
    rows = []
    for model in models:
        for base in ("none", "prompt"):
            configs = {}
            if (model, base, None, "qa", "full") in runs:
                configs["base"] = runs[(model, base, None, "qa", "full")]
            for (mo, me, seed, mt, sp), w in runs.items():
                if mo == model and mt == "qa" and sp == "full" and me.startswith(base + "@"):
                    configs[me.split("@")[1]] = w
            if len(configs) < 2:
                continue
            point, reps = bootstrap(configs, metric_fn("qa"), B)
            for variant, p in point.items():
                for m in PV_METRICS:
                    if m in p:
                        lo, hi = ci(reps[variant][m])
                        rows.append({"model": model, "method": base, "variant": variant, "metric": m,
                                     "value": p[m], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- v1 (published) values

def load_v1():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "acc" / "*" / "*.json"))):
        v = json.load(open(f))["acc"]
        base = Path(f).name.lower()
        method = next((m for k, m in [("fine", "ft"), ("hard", "hard"), ("prompt", "prompt"), ("soft", "soft")]
                       if k in base), "none")
        rows.append({"model": V1_MODELS[Path(f).parent.name], "method": method,
                     "f1_man_nofem": v["acc_man_no_feminitive"], "qadiff_fem": 1 - v["acc_diff_feminitive"],
                     "qadiff_nofem": 1 - v["acc_diff_no_feminitive"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

def pivot_ci(est, family, scope, metrics, methods=None):
    sub = est[(est.family == family) & (est.scope == scope) & est.metric.isin(metrics)]
    if methods:
        sub = sub[sub.method.isin(methods)]
    sub = sub.assign(cell=[fmt(v, lo, hi) for v, lo, hi in zip(sub.value, sub.lo, sub.hi)])
    t = sub.pivot_table(index=["model", "method"], columns="metric", values="cell", aggfunc="first")
    order = {m: i for i, m in enumerate(ZERO_SHOT + list(FT_METHODS))}
    t = t.reset_index().sort_values(["model", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    return t[["model", "method"] + [m for m in metrics if m in t.columns]]


def figures(est, diffs, mem, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5"})

    def height(models):
        return 1.6 + 0.6 * len(models)

    def rows_axis(ax, models):
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)

    def bottom_legend(fig, ax):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0.12, 1, 1))

    def forest(df, value, title, xlabel, path):
        if df.empty:
            return
        models = sorted(df.model.unique())
        fig, axes = plt.subplots(1, 2, figsize=(9, height(models)), sharey=True)
        for ax, comp in zip(axes, ["fem", "nofem"]):
            for mi, model in enumerate(models):
                for j, method in enumerate(["prompt", "soft", "hard", "ft", "ftctrl"]):
                    r = df[(df.model == model) & (df.method == method) & (df.comp == comp)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    y = mi + (j - 2) * 0.15
                    color, marker = METHOD_STYLE[method]
                    ax.plot([r.lo, r.hi], [y, y], color=color, lw=2, solid_capstyle="round")
                    ax.plot(r[value], y, marker=marker, color=color, ms=7, mec="white", mew=1.5,
                            label=method if mi == 0 else None)
            ax.axvline(0, color="#555", lw=1)
            ax.set_title(f"{title}: {'feminitive' if comp == 'fem' else 'non-feminitive'}")
            ax.set_xlabel(xlabel)
            rows_axis(ax, models)
        bottom_legend(fig, axes[0])
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)

    # F1: Delta QADiff (method - none) with CI; zero-shot on full, ft on unseen professions
    d = diffs[(diffs.family == "qa") & diffs.metric.isin(["qadiff_fem", "qadiff_nofem"])]
    d = d[((d.scope == "full") & ~d.method.isin(FT_METHODS)) | ((d.scope == "test") & d.method.isin(FT_METHODS))]
    forest(d.assign(comp=d.metric.str.replace("qadiff_", "")), "delta", "ΔQADiff",
           "method − no-debias (95% CI; < 0 = less bias)", out / "F1_forest_delta_qadiff.pdf")

    # F3: fem - nofem QADiff gap per config
    g = est[(est.family == "qa") & (est.metric == "gap_fem_minus_nofem") & (est.scope == "full")]
    if not g.empty:
        models = sorted(g.model.unique())
        fig, ax = plt.subplots(figsize=(6, height(models)))
        for mi, model in enumerate(models):
            for j, method in enumerate(ZERO_SHOT):
                r = g[(g.model == model) & (g.method == method)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = mi + (j - 1.5) * 0.18
                color, marker = METHOD_STYLE.get(method, ("#555", "o"))
                ax.plot([r.lo, r.hi], [y, y], color=color, lw=2)
                ax.plot(r.value, y, marker=marker, color=color, ms=7, mec="white", mew=1.5,
                        label=("none" if method == "none" else method) if mi == 0 else None)
        ax.axvline(0, color="#555", lw=1)
        rows_axis(ax, models)
        ax.set_xlabel("QADiff(fem) − QADiff(non-fem), 95% CI")
        bottom_legend(fig, ax)
        fig.savefig(out / "F3_fem_gap.pdf")
        fig.savefig(out / "F3_fem_gap.png", dpi=150)
        plt.close(fig)

    # F2: yes_rate per model x method (dot plot, man_nofem / woman_nofem / woman_fem)
    y = est[(est.family == "qa") & (est.scope == "full") & est.metric.str.startswith("yes_rate_")]
    if not y.empty:
        models = sorted(y.model.unique())
        fig, ax = plt.subplots(figsize=(7, height(models)))
        ax.axvspan(0, 0.05, color="#f2f2f2")
        ax.axvspan(0.95, 1, color="#f2f2f2")
        for mi, model in enumerate(models):
            for j, method in enumerate(ZERO_SHOT):
                r = y[(y.model == model) & (y.method == method) & (y.metric == "yes_rate_man_nofem")]
                if r.empty:
                    continue
                color, marker = METHOD_STYLE.get(method, ("#555", "o"))
                ax.plot(r.value.iloc[0], mi + (j - 1.5) * 0.18, marker=marker, color=color, ms=7, mec="white",
                        mew=1.5, label=method if mi == 0 else None, clip_on=False)
        ax.set_xlim(-0.02, 1.02)
        rows_axis(ax, models)
        ax.set_xlabel("share of «так» (man, non-feminitive); shaded = degenerate (<0.05 or >0.95)")
        bottom_legend(fig, ax)
        fig.savefig(out / "F2_yes_rate.pdf")
        fig.savefig(out / "F2_yes_rate.png", dpi=150)
        plt.close(fig)

    # F4: seen vs unseen professions after fine-tuning, one point per seed
    if not mem.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for ax, m, label in zip(axes, ["f1_man_nofem", "qadiff_fem"], ["F1 (man, non-fem)", "QADiff (fem)"]):
            for mi, model in enumerate(sorted(mem.model.unique())):
                sub = mem[mem.model == model]
                for x, part in enumerate(["seen", "unseen"]):
                    vals = sub[sub.professions == part][m]
                    ax.plot([x + mi * 0.08] * len(vals), vals, "o", color=list(METHOD_STYLE.values())[mi][0],
                            ms=6, label=model if x == 0 else None)
            ax.set_xticks([0.08, 1.08], ["seen", "unseen"])
            ax.set_title(label)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "F4_seen_unseen.pdf")
        fig.savefig(out / "F4_seen_unseen.png", dpi=150)
        plt.close(fig)


def md_table(df):
    if df is None or df.empty:
        return "_no runs yet_\n"
    return df.to_markdown(index=False) + "\n"


def claims(est, diffs, v1):
    """T7: published claim -> corrected evidence."""
    rows = []
    q = est[(est.family == "qa") & (est.scope == "full")]
    gaps = q[(q.metric == "gap_fem_minus_nofem") & (q.method == "none")]
    if not gaps.empty:
        pos = gaps[gaps.lo > 0]
        rows.append({"claim (paper)": "QADiff is consistently larger in feminitive contexts (p. 67)",
                     "published": f"holds for {(v1[v1.method == 'none'].qadiff_fem > v1[v1.method == 'none'].qadiff_nofem).sum()}/6 models (point estimates)",
                     "corrected": f"gap CI > 0 for {len(pos)}/{len(gaps)} models",
                     "still holds?": "yes" if len(pos) == len(gaps) else "no"})
    hard = diffs[(diffs.family == "qa") & (diffs.scope == "full") & (diffs.method == "hard") &
                 diffs.metric.isin(["qadiff_fem", "qadiff_nofem"])]
    if not hard.empty:
        sig = hard[hard.hi < 0]
        rows.append({"claim (paper)": "hard debiasing: average relative improvement of 18.9%",
                     "published": "18.9% (2.7% without Qwen2.5-3B, whose QADiff is 0–3 items of 700)",
                     "corrected": f"ΔQADiff CI < 0 in {len(sig)}/{len(hard)} model × context cells",
                     "still holds?": "yes" if len(sig) > len(hard) / 2 else "no"})
    for method in ["prompt", "soft"]:
        d = diffs[(diffs.family == "qa") & (diffs.scope == "full") & (diffs.method == method) &
                  diffs.metric.isin(["qadiff_fem", "qadiff_nofem"])]
        if not d.empty:
            rows.append({"claim (paper)": f"{method} debiasing reduces bias",
                         "published": "mixed (see Table 1)",
                         "corrected": f"ΔQADiff CI < 0 in {(d.hi < 0).sum()}/{len(d)}, > 0 in {(d.lo > 0).sum()}/{len(d)} cells",
                         "still holds?": "yes" if (d.hi < 0).sum() > len(d) / 2 else "no"})
    ft = diffs[(diffs.family == "qa") & (diffs.scope == "test") & (diffs.method == "ft") &
               diffs.metric.isin(["qadiff_fem", "qadiff_nofem"])]
    if not ft.empty:
        rows.append({"claim (paper)": "fine-tuning: disparity decreased substantially",
                     "published": "measured on a test set sharing 1368/1400 positives with train",
                     "corrected": f"on unseen professions ΔQADiff CI < 0 in {(ft.hi < 0).sum()}/{len(ft)} cells",
                     "still holds?": "yes" if (ft.hi < 0).sum() == len(ft) else "partly" if (ft.hi < 0).any() else "no"})
    vs = diffs[(diffs.family == "qa") & (diffs.scope == "test") & (diffs.method == "ft_vs_ftctrl") &
               diffs.metric.isin(["qadiff_fem", "qadiff_nofem"])]
    if not vs.empty:
        rows.append({"claim (paper)": "fine-tuning debiases (beyond learning the task)",
                     "published": "not tested (no control)",
                     "corrected": f"ft − male-only control: ΔQADiff CI < 0 in {(vs.hi < 0).sum()}/{len(vs)} cells",
                     "still holds?": "yes" if (vs.hi < 0).sum() == len(vs) else "partly" if (vs.hi < 0).any() else "no"})
    deg = q[(q.metric == "yes_rate_man_nofem") & (q.model == "qwen2.5-3b-instruct")]
    if not deg.empty:
        rows.append({"claim (paper)": "Qwen2.5-3B: zero difference after debiasing",
                     "published": "QADiff 0.000–0.004 (0–3 of 700 items), F1 = 2/3",
                     "corrected": "yes_rate (man, non-fem): " + ", ".join(f"{r.method} {r.value:.3f}" for r in deg.itertuples()),
                     "still holds?": "no (degenerate)" if ((deg.value > 0.95) | (deg.value < 0.05)).any() else "re-check"})
    f1d = diffs[(diffs.family == "qa") & (diffs.scope == "full") & diffs.metric.str.startswith("f1_") &
                ~diffs.method.isin(FT_METHODS)]
    if not f1d.empty:
        rows.append({"claim (paper)": "debiasing comes at the cost of lower QA accuracy",
                     "published": "F1 drops ≤ 0.002 for several configs",
                     "corrected": f"ΔF1 CI < 0 in {(f1d.hi < 0).sum()}/{len(f1d)} model × method × group cells",
                     "still holds?": "see T2"})
    return pd.DataFrame(rows)


def write_report(est, diffs, ft, mem, mc, pv, v1, path, fig_dir, B):
    qa_metrics_cols = [f"{p}_{g}" for g in GROUPS[:3] for p in ("f1", "bacc")] + \
                      [f"yes_rate_{g}" for g in GROUPS[:3]] + [f"undetermined_{g}" for g in GROUPS[:3]]
    t2 = pivot_ci(est, "qa", "full", qa_metrics_cols, ZERO_SHOT)
    t3 = pivot_ci(est, "qa", "full", ["qadiff_fem", "qadiff_nofem", "qadiff_fem_form", "gap_fem_minus_nofem"], ZERO_SHOT)
    t3m = pivot_ci(est, "qa", "full", ["margin_diff_fem", "margin_diff_nofem", "margin_absdiff_fem",
                                       "margin_absdiff_nofem"], ZERO_SHOT)
    t5m = pivot_ci(est, "qa", "test", ["margin_diff_fem", "margin_diff_nofem"])
    t8 = None
    if len(pv):
        t8 = pv.assign(cell=[fmt(v, lo, hi) for v, lo, hi in zip(pv.value, pv.lo, pv.hi)]).pivot_table(
            index=["model", "method", "variant"], columns="metric", values="cell", aggfunc="first").reset_index()
        t8 = t8[["model", "method", "variant"] + [m for m in PV_METRICS if m in t8.columns]]
    t4 = pivot_ci(est, "prob", "full", ["probdiff_fem", "probdiff_nofem", "probdiff_unpadded_fem",
                                        "probdiff_unpadded_nofem", "probagree_fem", "probagree_nofem",
                                        "probacc_f1_man_nofem", "probacc_f1_woman_nofem", "probacc_f1_woman_fem"],
                  ZERO_SHOT)
    t5 = pivot_ci(est, "qa", "test", ["f1_man_nofem", "f1_woman_nofem", "f1_woman_fem", "qadiff_fem", "qadiff_nofem"])
    t5p = pivot_ci(est, "prob", "test", ["probdiff_fem", "probdiff_nofem", "probacc_f1_man_nofem"])
    dq = diffs[(diffs.family == "qa") & diffs.metric.isin(["qadiff_fem", "qadiff_nofem", "f1_man_nofem"])]
    dq = dq[((dq.scope == "full") & ~dq.method.isin(FT_METHODS)) |
            ((dq.scope == "test") & dq.method.isin(FT_METHODS + ("ft_vs_ftctrl",)))]
    dq = dq.assign(cell=[fmt(v, lo, hi) + (" *" if lo > 0 or hi < 0 else "") for v, lo, hi in zip(dq.delta, dq.lo, dq.hi)])
    dq = dq.pivot_table(index=["model", "method"], columns="metric", values="cell", aggfunc="first").reset_index()
    dp = diffs[(diffs.family == "prob") & (diffs.scope == "full") & diffs.metric.isin(["probdiff_fem", "probdiff_nofem"])]
    dp = dp.assign(cell=[fmt(v, lo, hi, 4) + (" *" if lo > 0 or hi < 0 else "") for v, lo, hi in zip(dp.delta, dp.lo, dp.hi)])
    dp = dp.pivot_table(index=["model", "method"], columns="metric", values="cell", aggfunc="first").reset_index()
    degenerate = est[(est.family == "qa") & (est.scope == "full") & est.metric.str.startswith("yes_rate_") &
                     ((est.value > 0.95) | (est.value < 0.05))]
    mcq = mc[mc.family == "qa"] if len(mc) else mc
    t7 = claims(est, diffs, v1)
    ftq = ft[(ft.metric_family == "qa") & ft.metric.isin(["f1_man_nofem", "qadiff_fem", "qadiff_nofem"])] if len(ft) else ft

    text = f"""# GBEM-UA v2: results

_Generated by `scripts/stats.py` from `results/raw/*.jsonl` (B = {B} cluster-bootstrap resamples over professions, percentile 95% CI).
Do not edit by hand; re-run the script._

Zero-shot methods are evaluated on `full` (all 349 professions); fine-tuning on `test` (174 professions never seen in training).
`fem` = man (base form) vs woman (feminitive); `nofem` = man vs woman, both base form; `fem_form` = man vs woman, both feminitive (diagnostic).
QADiff = share of items where the man and woman answers differ. The profession whose feminitive equals the base form («рибалка») is excluded from comparisons involving a feminitive.

## Validity check: degenerate configurations

A configuration whose share of «так» is outside [0.05, 0.95] for some group answers (almost) constantly; its QADiff is near zero by construction and must not be read as "no bias".

{md_table(degenerate[["model", "method", "metric", "value"]] if len(degenerate) else None)}
## T2. QA quality (zero-shot, full)

{md_table(t2)}
## T3. QA bias (zero-shot, full)

{md_table(t3)}
### Threshold-free bias: log-odds margin

`margin = log P(«так») − log P(«ні»)` per sentence. `margin_diff` = mean paired difference man − woman
(> 0: the man is pushed towards «так»); `margin_absdiff` = mean absolute paired difference. Unlike QADiff, this is
not forced to 0 when a model answers «так» to everything, so it also measures bias in degenerate configurations.

{md_table(t3m)}
### McNemar (exact, man vs woman), Holm-corrected within the QA family

`b` = man «так» / woman «ні», `c` = the opposite. QADiff counts disagreements; McNemar tests whether they systematically favour one gender.

{md_table(mcq)}
## F1. Does a method reduce QA bias? ΔQADiff vs no-debias (paired bootstrap)

`*` = 95% CI excludes 0. Negative ΔQADiff = less disagreement between man and woman.

{md_table(dq)}
![F1]({fig_dir}/F1_forest_delta_qadiff.png)

![F2]({fig_dir}/F2_yes_rate.png)

![F3]({fig_dir}/F3_fem_gap.png)

## T4. Probability metrics (zero-shot, full)

`probdiff` = v1 ProbDiffMetric (EOS-padded, unchanged formula); `probdiff_unpadded` = same without padding (sensitivity check);
`probagree` = share of items where man and woman get the same pos/neg decision (this is what v1 Table 6 called "Acc Prob Metric");
`probacc_f1` = the actual ProbAccMetric: F1 of [P(positive) > P(negated)] against the label.

{md_table(t4)}
### ΔProbDiff vs no-debias

{md_table(dp)}
## T5. Fine-tuning on unseen professions (test split)

All methods restricted to the 174 test professions; `ft` = mean over seeds with the CI of the seed-averaged metric.
`ftctrl` = control adapter trained on the same professions and answers but on male base-form sentences only
(same number of steps). If `ftctrl` reduces QADiff as much as `ft`, the reduction comes from learning the task,
not from gender-specific debiasing. Row `ft_vs_ftctrl` in the ΔQADiff table above gives the paired difference.

{md_table(t5)}
{md_table(t5m)}
{md_table(t5p)}
### Per-seed values

{md_table(ftq)}
## T6. Memorization: seen vs unseen professions after fine-tuning

{md_table(mem)}
![F4]({fig_dir}/F4_seen_unseen.png)

## T8. Prompt sensitivity (QA, full)

`base` = «Дай відповідь одним словом: так або ні.»; variants in `configs/prompt_variants.json`
(`order` swaps «так»/«ні» in the instruction). Conclusions that change across variants are prompt-dependent.

{md_table(t8)}
## T7. Published vs corrected

{md_table(t7)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"T2": t2, "T3": t3, "T4": t4, "T5": t5, "T7": t7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--results", default="results")
    ap.add_argument("--report", default="reports/RESULTS_v2.md")
    a = ap.parse_args()
    res = ROOT / a.results
    runs = load_runs(res / "raw")
    if not runs:
        sys.exit(f"no runs in {res / 'raw'}")
    print(f"{len(runs)} runs:", sorted({(k[0], k[1], k[3]) for k in runs}))
    est, diffs, ft, mem, mc, pv = analyse(runs, a.B)
    tables = res / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, df in [("estimates", est), ("diffs", diffs), ("finetune_seeds", ft), ("memorization", mem),
                     ("mcnemar", mc), ("prompt_sensitivity", pv)]:
        df.to_csv(tables / f"{name}.csv", index=False)
    figures(est, diffs, mem, res / "figures")
    report = ROOT / a.report
    fig_rel = Path("..") / a.results / "figures" if report.parent.name == "reports" else res / "figures"
    out = write_report(est, diffs, ft, mem, mc, pv, load_v1(), report, fig_rel.as_posix(), a.B)
    for name, df in out.items():
        if not df.empty:
            df.to_latex(tables / f"{name}.tex", index=False)
    print("written:", report, tables, res / "figures")


if __name__ == "__main__":
    main()
