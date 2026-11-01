"""
Probability protocol (v2).

Each prob-dataset sentence gives two variants: positive ("BLANK " -> "") and negated ("BLANK " -> "не ").
Every variant is scored as a whole sentence, conditioned on a document-start token (BOS if the tokenizer
has one, otherwise <|endoftext|>/EOS, as in lm-evaluation-harness) and, for the prompt method, on the
debias phrase; only the sentence tokens are scored. v1 fed the literal text "<s>" as the start token
(not a BOS for Qwen/Gemma) and, for the prompt method, read the logits at the prompt positions instead
of the sentence positions.

To keep the v1 ProbDiffMetric formula, `pad` EOS tokens are appended after the sentence, where `pad`
is the largest length difference to any gender variant of the same item; their log-probs are stored
separately so both padded (v1) and unpadded scores can be computed offline (metrics/scores.py).
"""
import json

import pandas as pd
import torch
from tqdm import tqdm

VARIANTS = {"pos": "", "neg": "не "}


def doc_prefix_id(tokenizer):
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        return eot
    return tokenizer.eos_token_id


class ProbEvaluator:
    def __init__(self, wrapper, batch_size=16):
        self.wrapper, self.tok = wrapper, wrapper.tokenizer
        self.batch_size = batch_size
        self.prefix_id = doc_prefix_id(self.tok)

    def encode(self, sentence, job_form):
        text = self.wrapper.with_prompt(sentence)
        ids, offsets, debias_idx = self.wrapper.encode(text, job_form, sentence)
        start = text.rindex(sentence)
        # First token that belongs to the sentence (may include the separating space).
        first = next(i for i, (s, e) in enumerate(offsets) if e > start)
        shift = 1  # the prefix token
        return [self.prefix_id] + ids, first + shift, [p + shift for p in debias_idx]

    def score_batch(self, items):
        T = max(len(it["ids"]) + it["pad"] for it in items)
        eos = self.tok.eos_token_id
        ids = torch.full((len(items), T), eos, dtype=torch.long)
        mask = torch.zeros((len(items), T), dtype=torch.long)
        for i, it in enumerate(items):
            seq = it["ids"] + [eos] * it["pad"]
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        logits = self.wrapper.forward(ids, mask, [it["debias_idx"] for it in items]).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        target_lp = logprobs.gather(-1, ids[:, 1:].to(logprobs.device).unsqueeze(-1)).squeeze(-1).cpu()
        out = []
        for i, it in enumerate(items):
            n = len(it["ids"])
            lp = target_lp[i]
            # token at position p is predicted at p-1
            out.append({"token_logprobs": lp[it["first"] - 1:n - 1].tolist(),
                        "eos_logprobs": lp[n - 1:n - 1 + it["pad"]].tolist()})
        return out

    def run(self, df: pd.DataFrame, out_path, meta: dict):
        items = []
        for r in df.itertuples():
            for variant, fill in VARIANTS.items():
                sentence = r.sentence.replace("BLANK ", fill)
                ids, first, debias_idx = self.encode(sentence, r.job_form)
                items.append({"row_id": r.row_id, "item_id": r.item_id, "variant": variant, "sentence": sentence,
                              "ids": ids, "first": first, "debias_idx": debias_idx,
                              "n_tokens": len(ids) - first,
                              "record": {"profession": r.profession, "group": r.group,
                                         "is_correct": bool(r.is_correct), "fem_identical": bool(r.fem_identical),
                                         "split": r.split}})
        # pad = max length difference to the other gender variants of the same item and variant
        longest = {}
        for it in items:
            key = (it["item_id"], it["variant"])
            longest[key] = max(longest.get(key, 0), it["n_tokens"])
        for it in items:
            it["pad"] = longest[(it["item_id"], it["variant"])] - it["n_tokens"]

        with open(out_path, "w") as f:
            for start in tqdm(range(0, len(items), self.batch_size), desc=f"Prob {meta.get('method')}"):
                batch = items[start:start + self.batch_size]
                for it, res in zip(batch, self.score_batch(batch)):
                    assert len(res["token_logprobs"]) == it["n_tokens"]
                    rec = {"row_id": it["row_id"], "item_id": it["item_id"], "variant": it["variant"],
                           **it["record"], **meta, "sentence": it["sentence"], "n_tokens": it["n_tokens"],
                           "pad": it["pad"], "n_debiased_tokens": len(it["debias_idx"]), **res}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
