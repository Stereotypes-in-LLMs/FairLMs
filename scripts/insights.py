"""
Follow-up analyses of the zero-shot QA runs (method `none`, split `full`) that go beyond RESULTS_v2.md:

  A. grammatical-agreement effect: does a job title whose grammatical gender matches the candidate raise «так»?
  B. flips by item type (relevant vs irrelevant experience) and their direction;
  C. professions with the most consistent gender effect across models;
  D. how each mitigation method changes A and B (paired bootstrap vs `none`; fine-tuning on `test`, pooled over seeds);
  E. robustness of A and B to the instruction wording (prompt-variant runs of `none`).

Writes results/tables/insights_*.csv and prints the tables. Usage: python -m scripts.insights [--B 2000]
"""
import argparse

import numpy as np
import pandas as pd

from scripts.stats import METHOD_STYLE, ROOT, bootstrap, ci, load_runs

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


MITIGATION_METRICS = ["yes_woman", "yes_man", "margin_woman", "margin_man",
                      "net_man_nofem_irrelevant", "net_man_nofem_relevant", "net_man_fem_irrelevant",
                      "net_man_fem_relevant", "qadiff_nofem_irrelevant", "qadiff_nofem_relevant",
                      "qadiff_fem_irrelevant", "qadiff_fem_relevant"]


def mitigation(all_runs, B):
    """Value and paired difference vs `none` for every method. Zero-shot methods on `full`; ft/ftctrl on the test
    professions, averaged over seeds (each replicate averages the seeds on the same resample)."""
    fn = lambda w: {**agreement_metrics(w), **flip_metrics(w)}
    rows = []
    for model in sorted({k[0] for k in all_runs}):
        qa = {(me, se, sp): w[~w.fem_identical.astype(bool)] for (mo, me, se, mt, sp), w in all_runs.items()
              if mo == model and mt == "qa" and "@" not in me}
        for scope, methods in (("full", ["prompt", "soft", "hard"]), ("test", ["ft", "ftctrl"])):
            base = qa.get(("none", None, "full"))
            if base is None:
                continue
            configs = {"none": base if scope == "full" else base[base.split == "test"]}
            for (me, se, sp), w in qa.items():
                if me in methods and (scope == "full" and se is None and sp == "full" or scope == "test" and se):
                    configs[(me, se)] = w if scope == "full" else w[w.split == "test"]
            if len(configs) < 2:
                continue
            point, reps = bootstrap(configs, fn, B)
            for method in methods:
                keys = [k for k in configs if k != "none" and k[0] == method]
                if not keys:
                    continue
                for m in MITIGATION_METRICS:
                    v = np.mean([point[k][m] for k in keys])
                    r = np.mean([reps[k][m].to_numpy() for k in keys], axis=0)
                    d = r - reps["none"][m].to_numpy()
                    rows.append({"model": model, "scope": scope, "method": method, "metric": m, "value": v,
                                 "none": point["none"][m], "delta": v - point["none"][m],
                                 "lo": ci(d)[0], "hi": ci(d)[1]})
    t = pd.DataFrame(rows)
    t["cell"] = [f"{v:+.3f} (Δ {d:+.3f}{' *' if lo > 0 or hi < 0 else ''})"
                 for v, d, lo, hi in zip(t.value, t.delta, t.lo, t.hi)]
    return t


ROBUSTNESS_METRICS = ["yes_woman", "yes_man", "net_man_nofem_irrelevant", "net_man_nofem_relevant"]


def prompt_robustness(all_runs, B):
    fn = lambda w: {**agreement_metrics(w), **flip_metrics(w)}
    rows = []
    for (model, method, seed, metric, split), w in sorted(all_runs.items()):
        if metric != "qa" or split != "full" or seed is not None or method.split("@")[0] != "none":
            continue
        variant = method.split("@")[1] if "@" in method else "base"
        point, reps = bootstrap({0: w[~w.fem_identical.astype(bool)]}, fn, B)
        for m in ROBUSTNESS_METRICS:
            lo, hi = ci(reps[0][m])
            rows.append({"model": model, "variant": variant, "metric": m, "value": point[0][m], "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t["cell"] = [f"{v:+.3f}" + ("*" if lo > 0 or hi < 0 else "") for v, lo, hi in zip(t.value, t.lo, t.hi)]
    return t


NOFEM, FEM, INK = "#2a78d6", "#eb6834", "#333333"


def figures(rob, flips, mit, out):
    """F5 agreement effect (+ instruction variants), F6 net flips by qualification, F7 direction after fine-tuning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5"})

    def rows_axis(ax, models):
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#555", lw=1)

    def save(fig, name):
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    models = sorted(rob.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 1.6 + 0.6 * len(models)), sharey=True)
    for ax, (metric, title) in zip(axes, [("yes_woman", "Woman: feminitive − masculine title"),
                                          ("yes_man", "Man: masculine − feminitive title")]):
        for y, model in enumerate(models):
            d = rob[(rob.model == model) & (rob.metric == metric)]
            var = d[d.variant != "base"]
            ax.plot(var.value, [y + 0.18] * len(var), "o", mfc="none", mec="#999", ms=6,
                    label="other instructions" if y == 0 else None)
            b = d[d.variant == "base"].iloc[0]
            ax.plot([b.lo, b.hi], [y, y], color=INK, lw=2, solid_capstyle="round")
            ax.plot(b.value, y, "o", color=INK, ms=7, mec="white", mew=1.5,
                    label="main instruction (95% CI)" if y == 0 else None)
        rows_axis(ax, models)
        ax.set_title(title)
        ax.set_xlabel("Δ share of «так» when the title agrees with the candidate")
    save(fig, "F5_agreement_effect")

    fig, axes = plt.subplots(1, 2, figsize=(9, 1.6 + 0.6 * len(models)), sharey=True)
    for ax, (rel, title) in zip(axes, [("irrelevant", "Unqualified candidate (irrelevant experience)"),
                                       ("relevant", "Qualified candidate (relevant experience)")]):
        for y, model in enumerate(models):
            for k, (comp, color, label) in enumerate([("nofem", NOFEM, "both masculine title"),
                                                      ("fem", FEM, "woman with feminitive")]):
                r = flips[(flips.model == model) & (flips.metric == f"net_man_{comp}_{rel}")].iloc[0]
                yy = y + (k - 0.5) * 0.25
                ax.plot([r.lo, r.hi], [yy, yy], color=color, lw=2, solid_capstyle="round")
                ax.plot(r.value, yy, "o", color=color, ms=7, mec="white", mew=1.5, label=label if y == 0 else None)
        rows_axis(ax, models)
        ax.set_title(title)
        ax.set_xlabel("net flips (> 0: man favoured; 95% CI)")
    save(fig, "F6_flips_by_qualification")

    m = mit[mit.scope == "test"]
    ft_models = sorted(m.model.unique())
    panels = [("net_man_nofem_irrelevant", "Masculine title, unqualified"),
              ("net_man_nofem_relevant", "Masculine title, qualified"),
              ("net_man_fem_irrelevant", "Woman with feminitive, unqualified"),
              ("net_man_fem_relevant", "Woman with feminitive, qualified")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 2.2 + 1.2 * len(ft_models)), sharey=True)
    styles = [("none", INK, "o", "no debiasing")] + [(k, *METHOD_STYLE[k], lab) for k, lab in
                                                     [("ft", "fine-tuned"), ("ftctrl", "male-only control")]]
    for ax, (metric, title) in zip(axes.flat, panels):
        for y, model in enumerate(ft_models):
            d = m[(m.model == model) & (m.metric == metric)].set_index("method")
            for k, (method, color, marker, label) in enumerate(styles):
                v = d.none.iloc[0] if method == "none" else d.value[method]
                sig = method == "none" or d.lo[method] > 0 or d.hi[method] < 0
                ax.plot(v, y + (k - 1) * 0.2, marker, color=color, ms=7, mec=color, mfc=color if sig else "white",
                        mew=1.5)
        rows_axis(ax, ft_models)
        ax.set_title(title)
        ax.set_xlabel("net flips (> 0: man favoured)")
    for method, color, marker, label in styles:  # legend in the filled style, independent of the first row
        axes.flat[0].plot([], [], marker, color=color, ms=7, label=label)
    fig.text(0.5, 0.06, "hollow marker: change vs no debiasing not significant (paired 95% CI)", ha="center",
             color="#555")
    save(fig, "F7_direction_after_finetuning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    all_runs = load_runs(ROOT / "results" / "raw")
    runs = {k: v for k, v in all_runs.items() if k[1] == "none" and k[2] is None and k[3] == "qa" and k[4] == "full"}
    out = ROOT / "results" / "tables"
    agree = boot_table(runs, agreement_metrics, a.B)
    flips = boot_table(runs, flip_metrics, a.B)
    prof = professions(runs)
    agree.to_csv(out / "insights_agreement.csv", index=False)
    flips.to_csv(out / "insights_flips_by_relevance.csv", index=False)
    prof.to_csv(out / "insights_professions.csv", index=False)
    mit = mitigation(all_runs, a.B)
    mit.to_csv(out / "insights_mitigation.csv", index=False)
    rob = prompt_robustness(all_runs, a.B)
    rob.to_csv(out / "insights_prompt_robustness.csv", index=False)
    figures(rob, flips, mit, ROOT / "results" / "figures")
    pd.set_option("display.width", 250)
    print(agree.pivot(index="model", columns="metric", values="cell").to_string())
    print(flips.pivot(index="model", columns="metric", values="cell").T.to_string())
    print(prof.head(15).to_string(index=False))
    print(prof.tail(15).to_string(index=False))
    for scope in ("full", "test"):
        print(mit[mit.scope == scope].pivot_table(index=["model", "method"], columns="metric", values="cell",
                                                  aggfunc="first")[MITIGATION_METRICS].T.to_string())
    print(rob.pivot_table(index=["metric", "model"], columns="variant", values="cell", aggfunc="first").to_string())
    print("models with net man-favoured flips per profession:",
          (prof.models_man - prof.models_woman).describe().round(2).to_dict())


if __name__ == "__main__":
    main()
