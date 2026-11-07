import json
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.stats import holm, mcnemar_p

ACC = pd.read_csv("data/v2/acc_full.csv")
PROB = pd.read_csv("data/v2/prob_full.csv")


def fake_qa(path, rng, bias, yes_bias=0.0, margin_bias=0.0):
    u = {item: rng.random() for item in ACC.item_id.unique()}  # shared draw: no bias -> same answer
    with open(path, "w") as f:
        for r in ACC.itertuples():
            p = 0.8 if r.is_correct else 0.2
            p += yes_bias + (bias if r.group == "man_nofem" else 0)
            pred = int(u[r.item_id] < min(max(p, 0), 1))
            margin = 3 * (pred - 0.5) + u[r.item_id] + (margin_bias if r.group == "man_nofem" else 0)
            f.write(json.dumps({"row_id": r.row_id, "item_id": r.item_id, "profession": r.profession,
                                "group": r.group, "is_correct": bool(r.is_correct),
                                "fem_identical": bool(r.fem_identical), "split": r.split, "pred": pred,
                                "gen_answer": "так" if pred else "ні", "logp_yes": margin, "logp_no": 0.0},
                               ensure_ascii=False) + "\n")


def fake_prob(path, rng, df):
    with open(path, "w") as f:
        for r in df.itertuples():
            for v in ["pos", "neg"]:
                n = int(rng.integers(8, 12))
                f.write(json.dumps({"row_id": r.row_id, "item_id": r.item_id, "variant": v, "profession": r.profession,
                                    "group": r.group, "is_correct": bool(r.is_correct),
                                    "fem_identical": bool(r.fem_identical), "split": r.split,
                                    "token_logprobs": list(rng.normal(-3, 1, n)),
                                    "eos_logprobs": list(rng.normal(-2, 1, 6))}, ensure_ascii=False) + "\n")


def test_holm_and_mcnemar():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert mcnemar_p(0, 0) == 1.0 and mcnemar_p(30, 5) < 0.001


def test_stats_pipeline_detects_planted_effect(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "results" / "raw"
    raw.mkdir(parents=True)
    for model in ["m-a"]:
        fake_qa(raw / f"{model}_none_qa_full.jsonl", rng, bias=0.3)       # strong man-yes bias
        fake_qa(raw / f"{model}_prompt_qa_full.jsonl", rng, bias=0.0)     # removes it
        fake_qa(raw / f"{model}_hard_qa_full.jsonl", rng, bias=0.0, yes_bias=0.9, margin_bias=0.5)  # degenerate,
        # but the man still gets a higher «так» margin: QADiff = 0, margin_diff > 0
        fake_qa(raw / f"{model}_none_pv_order_qa_full.jsonl", rng, bias=0.0)
        fake_qa(raw / f"{model}_none_ftctrl_s13_qa_full.jsonl", rng, bias=0.3)  # control keeps the bias
        fake_qa(raw / f"{model}_none_ft_s13_qa_full.jsonl", rng, bias=0.0)
        fake_qa(raw / f"{model}_none_ft_s42_qa_full.jsonl", rng, bias=0.0)
        for m in ["none", "prompt"]:
            fake_prob(raw / f"{model}_{m}_prob_full.jsonl", rng, PROB)
        fake_prob(raw / f"{model}_none_ft_s13_prob_test.jsonl", rng, PROB[PROB.split == "test"])
        fake_qa(raw / f"{model}_none_qa_full_pilot5.jsonl", rng, bias=0)  # ignored
    report = tmp_path / "reports" / "RESULTS_v2.md"
    subprocess.run([sys.executable, "scripts/stats.py", "--B", "60", "--results", str(tmp_path / "results"),
                    "--report", str(report)], check=True)
    diffs = pd.read_csv(tmp_path / "results" / "tables" / "diffs.csv")
    d = diffs[(diffs.method == "prompt") & (diffs.metric == "qadiff_nofem") & (diffs.scope == "full")].iloc[0]
    assert d.hi < 0  # planted reduction is detected
    text = report.read_text()
    assert "hard" in text.split("## T2")[0]  # degenerate config flagged
    assert (tmp_path / "results" / "figures" / "F1_forest_delta_qadiff.pdf").exists()
    mc = pd.read_csv(tmp_path / "results" / "tables" / "mcnemar.csv")
    none = mc[(mc.method == "none") & (mc.comparison == "nofem")].iloc[0]
    assert none.b_man_yes_woman_no > none.c_man_no_woman_yes and none.p_holm < 0.01

    est = pd.read_csv(tmp_path / "results" / "tables" / "estimates.csv")
    hard = est[(est.method == "hard") & (est.scope == "full")].set_index("metric")
    assert hard.loc["qadiff_nofem", "value"] == 0 and hard.loc["margin_diff_nofem", "lo"] > 0
    vs = diffs[(diffs.method == "ft_vs_ftctrl") & (diffs.metric == "qadiff_nofem")].iloc[0]
    assert vs.hi < 0  # main adapter beats the biased control
    pv = pd.read_csv(tmp_path / "results" / "tables" / "prompt_sensitivity.csv")
    assert set(pv.variant) == {"base", "order"}
    assert "## T8" in text and "ft_vs_ftctrl" in text
