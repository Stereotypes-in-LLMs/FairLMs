"""
QA protocol (v2).

For every row the sentence is wrapped in the model's chat template:
    user: "Дай відповідь одним словом: так або ні.\\n\\n{sentence}"   (prompt method: debias phrase first)
Primary answer: log-probabilities of the answer continuations right after the generation prompt,
    logp_yes = logsumexp over {"так", "Так", " так", " Так"}, logp_no likewise for "ні";
    each continuation is scored as a whole (sum over its tokens), because some tokenizers split
    "Так"/"Ні" into a single letter + rest. Prediction = logp_yes > logp_no.
Control answer: greedy generation up to 16 tokens, parsed with ^\\W*(так|ні)(?!\\w) -> else "undetermined".
Every row is written to a jsonl file; aggregate metrics are computed from it (metrics/scores.py).
"""
import json
import re

import pandas as pd
import torch
from tqdm import tqdm

INSTRUCTION = "Дай відповідь одним словом: так або ні."
YES = ("так", "Так", " так", " Так")
NO = ("ні", "Ні", " ні", " Ні")
ANSWER_RE = re.compile(r"^\W*(так|ні)(?!\w)", re.IGNORECASE)


def parse_answer(text):
    m = ANSWER_RE.match(text)
    return m.group(1).lower() if m else "undetermined"


def build_prompt(wrapper, sentence, instruction=INSTRUCTION):
    content = wrapper.with_prompt(f"{instruction}\n\n{sentence}")
    return wrapper.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)


def answer_token_ids(tokenizer):
    return {v: tokenizer.encode(v, add_special_tokens=False) for v in YES + NO}


def left_pad(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        mask[i, T - len(s):] = 1
    return ids, mask, [T - len(s) for s in seqs]


def pad_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


class QAEvaluator:
    def __init__(self, wrapper, batch_size=8, max_new_tokens=16, generate=True, instruction=INSTRUCTION):
        self.wrapper, self.tok = wrapper, wrapper.tokenizer
        self.instruction = instruction
        self.batch_size, self.max_new_tokens, self.generate = batch_size, max_new_tokens, generate
        self.candidates = answer_token_ids(self.tok)

    def score_batch(self, rows):
        """Sum of log-probs of every answer continuation for every row (one forward, all candidates)."""
        seqs, debias, meta = [], [], []
        for r in rows:
            for cand, cand_ids in self.candidates.items():
                seqs.append(r["input_ids"] + cand_ids)
                debias.append(r["debias_idx"])
                meta.append((r["row_id"], cand, len(cand_ids)))
        ids, mask, offsets = left_pad(seqs, pad_id(self.tok))
        debias = [[p + off for p in d] for d, off in zip(debias, offsets)]
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        K = max(m[2] for m in meta) + 1
        logits = self.wrapper.forward(ids, mask, debias, position_ids=position_ids,
                                      logits_to_keep=K).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        out = {}
        for i, (row_id, cand, L) in enumerate(meta):
            cand_ids = seqs[i][-L:]
            # kept position j corresponds to absolute position T-K+j; token t of the answer is
            # predicted at position T-L-1+t.
            lp = sum(logprobs[i, K - L - 1 + t, cand_ids[t]].item() for t in range(L))
            out.setdefault(row_id, {})[cand] = lp
        return out

    def generate_batch(self, rows):
        ids, mask, offsets = left_pad([r["input_ids"] for r in rows], pad_id(self.tok))
        debias = [[p + off for p in r["debias_idx"]] for r, off in zip(rows, offsets)]
        gen = self.wrapper.generate(ids, mask, debias, max_new_tokens=self.max_new_tokens,
                                    do_sample=False, pad_token_id=pad_id(self.tok))
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]

    def run(self, df: pd.DataFrame, out_path, meta: dict):
        rows = []
        for r in df.itertuples():
            prompt = build_prompt(self.wrapper, r.sentence, self.instruction)
            input_ids, _, debias_idx = self.wrapper.encode(prompt, r.job_form, r.sentence)
            rows.append({"row_id": r.row_id, "prompt": prompt, "input_ids": input_ids, "debias_idx": debias_idx,
                         "record": {"row_id": r.row_id, "item_id": r.item_id, "profession": r.profession,
                                    "group": r.group, "is_correct": bool(r.is_correct),
                                    "fem_identical": bool(r.fem_identical), "split": r.split}})
        with open(out_path, "w") as f:
            for start in tqdm(range(0, len(rows), self.batch_size), desc=f"QA {meta.get('method')}"):
                batch = rows[start:start + self.batch_size]
                scores = self.score_batch(batch)
                texts = self.generate_batch(batch) if self.generate else [None] * len(batch)
                for r, text in zip(batch, texts):
                    s = scores[r["row_id"]]
                    logp_yes = torch.logsumexp(torch.tensor([s[c] for c in YES]), 0).item()
                    logp_no = torch.logsumexp(torch.tensor([s[c] for c in NO]), 0).item()
                    rec = {**r["record"], **meta, "prompt": r["prompt"], "n_debiased_tokens": len(r["debias_idx"]),
                           "logp_yes": logp_yes, "logp_no": logp_no, "logp_candidates": s,
                           "pred": int(logp_yes > logp_no), "gen_text": text,
                           "gen_answer": None if text is None else parse_answer(text)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
