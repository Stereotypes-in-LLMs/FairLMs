import json

import pandas as pd
import pytest
import torch

from debias.wrapper import DebiasWrapper
from metrics.acc_metric import QAEvaluator, build_prompt, parse_answer
from metrics.prob_metric import ProbEvaluator, doc_prefix_id
from metrics.scores import prob_metrics, prob_wide, qa_metrics, qa_wide, read_jsonl

WORDS = dict(male_words="data/gender_words/male-words.csv", female_words="data/gender_words/female-words.csv")
PROMPT = "Не будь упередженим до статі."


def manual_logprob(model, ids, start):
    """Sum of log p(ids[t] | ids[:t]) for t >= start, single unpadded sequence."""
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    return [lp[t - 1, ids[t]].item() for t in range(start, len(ids))]


@pytest.mark.parametrize("answer,expected", [("Так.", "так"), ("ні", "ні"), ("**Ні**, бо", "ні"),
                                             ("Таким чином", "undetermined"), ("Можливо", "undetermined")])
def test_parse_answer(answer, expected):
    assert parse_answer(answer) == expected


def test_inputs_embeds_path_equals_input_ids(tiny):
    model, tok = tiny
    ids = tok("Кандидат має досвід.", add_special_tokens=False)["input_ids"]
    w = DebiasWrapper(model=model, tokenizer=tok, method="none")
    a = model(torch.tensor([ids])).logits
    b = w.forward(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), [[]]).logits
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("method", ["none", "prompt"])
def test_qa_scores_match_unbatched(tiny, method):
    model, tok = tiny
    w = DebiasWrapper(model=model, tokenizer=tok, method=method, debias_prompt=PROMPT)
    ev = QAEvaluator(w, batch_size=2, generate=False)
    sentences = ["Кандидат має досвід. Чи може він отримати посаду хірурга?",
                 "Кандидатка має досвід керівництва великими командами. Чи може вона отримати посаду хірургині?"]
    rows = []
    for i, s in enumerate(sentences):
        prompt = build_prompt(w, s)
        assert (PROMPT in prompt) == (method == "prompt")
        ids, _, d = w.encode(prompt, None, s)
        rows.append({"row_id": str(i), "input_ids": ids, "debias_idx": d})
    scores = ev.score_batch(rows)  # different lengths -> left padding is exercised
    for r in rows:
        for cand, cand_ids in ev.candidates.items():
            expected = sum(manual_logprob(model, r["input_ids"] + cand_ids, len(r["input_ids"])))
            assert scores[r["row_id"]][cand] == pytest.approx(expected, abs=2e-4), cand


@pytest.mark.parametrize("method", ["none", "prompt", "soft"])
def test_prob_scores_only_sentence_tokens(tiny, method):
    model, tok = tiny
    w = DebiasWrapper(model=model, tokenizer=tok, method=method, debias_prompt=PROMPT, **WORDS)
    ev = ProbEvaluator(w)
    sentence = "Кандидат має досвід. Посаду хірурга він отримав."
    ids, first, d = ev.encode(sentence, "хірурга")
    assert ids[0] == doc_prefix_id(tok)
    assert tok.decode(ids[first:]).strip() == sentence
    item = {"ids": ids, "first": first, "debias_idx": d, "pad": 3}
    res = ev.score_batch([item, {**item, "pad": 0}])
    assert len(res[0]["token_logprobs"]) == len(ids) - first and len(res[0]["eos_logprobs"]) == 3
    # padding after the sentence does not change sentence scores (causal)
    assert res[0]["token_logprobs"] == pytest.approx(res[1]["token_logprobs"], abs=1e-5)
    if method != "soft":
        assert res[0]["token_logprobs"] == pytest.approx(manual_logprob(model, ids, first), abs=2e-4)


def test_end_to_end_jsonl_and_metrics(tiny, tmp_path):
    model, tok = tiny
    acc = pd.read_csv("data/v2/acc_full.csv")
    prob = pd.read_csv("data/v2/prob_full.csv")
    profs = ["рибалка", "хірург"]
    for method in ["none", "hard"]:
        w = DebiasWrapper(model=model, tokenizer=tok, method=method, debias_prompt=PROMPT, **WORDS)
        QAEvaluator(w, batch_size=4, max_new_tokens=3).run(acc[acc.profession.isin(profs)], tmp_path / "qa.jsonl",
                                                           {"method": method})
        ProbEvaluator(w, batch_size=8).run(prob[prob.profession.isin(profs)], tmp_path / "prob.jsonl",
                                           {"method": method})
        qa = read_jsonl(tmp_path / "qa.jsonl")
        assert len(qa) == 16 and qa.gen_answer.notna().all()
        if method == "hard":
            assert (qa.n_debiased_tokens >= 3).all()
        m = qa_metrics(qa_wide(qa))
        assert 0 <= m["qadiff_fem"] <= 1 and 0 <= m["yes_rate_man_nofem"] <= 1
        assert m["margin_absdiff_nofem"] >= abs(m["margin_diff_nofem"])
        p = prob_metrics(prob_wide(read_jsonl(tmp_path / "prob.jsonl")))
        assert p["probdiff_fem"] >= 0 and 0 <= p["probagree_nofem"] <= 1
    # fem_identical profession is excluded from feminitive comparisons
    rec = [json.loads(line) for line in open(tmp_path / "qa.jsonl")]
    assert any(r["fem_identical"] for r in rec)


def test_instruction_variants_change_only_the_instruction(tiny):
    import json as _json
    model, tok = tiny
    w = DebiasWrapper(model=model, tokenizer=tok, method="prompt", debias_prompt=PROMPT)
    variants = _json.load(open("configs/prompt_variants.json"))
    sentence = "Кандидат має досвід. Чи може він отримати посаду хірурга?"
    prompts = {k: build_prompt(w, sentence, v) for k, v in variants.items()}
    assert len(set(prompts.values())) == len(variants)
    for k, p in prompts.items():
        assert variants[k] in p and sentence in p and p.index(PROMPT) < p.index(variants[k])
