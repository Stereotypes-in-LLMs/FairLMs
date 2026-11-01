"""CPU dry run: every row of acc/prob full, every tokenizer, soft/hard - target words found and tokenized."""
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from debias.wrapper import DebiasWrapper  # noqa: E402
from metrics.acc_metric import build_prompt  # noqa: E402
from scripts.answer_tokens import MODELS  # noqa: E402

acc, prob = (pd.read_csv(ROOT / f"data/v2/{c}_full.csv") for c in ("acc", "prob"))
for m in MODELS:
    tok = AutoTokenizer.from_pretrained(m)
    stats = {}
    for method in ("soft", "hard"):
        w = DebiasWrapper(model=None, tokenizer=tok, method="none")
        w.method = method  # no basis needed for the lookup
        n = []
        for r in acc.itertuples():
            n.append(len(w.encode(build_prompt(w, r.sentence), r.job_form, r.sentence)[2]))
        for r in prob.itertuples():
            for fill in ("", "не "):
                s = r.sentence.replace("BLANK ", fill)
                n.append(len(w.encode(s, r.job_form, s)[2]))
        stats[method] = (min(n), max(n))
    print(m, "debiased tokens per row (min, max):", stats)
