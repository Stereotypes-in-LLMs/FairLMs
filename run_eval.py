"""
Runs one evaluation configuration and writes per-row outputs + a summary.

    python run_eval.py --config configs/config_soft.json --model Qwen/Qwen2.5-7B-Instruct --metric qa --split full
    python run_eval.py --config configs/config_no_debias.json --model Qwen/Qwen2.5-3B-Instruct \
        --adapter checkpoints/qwen2.5-3b-instruct_s42 --tag ft_s42 --metric both --split test

Outputs: results/raw/<model>_<method>[_<tag>]_<metric>_<split>.jsonl and results/summary/<same>.json
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from debias import DebiasWrapper
from metrics import ProbEvaluator, QAEvaluator
from metrics.acc_metric import INSTRUCTION
from metrics.scores import prob_metrics, prob_wide, qa_metrics, qa_wide, read_jsonl
from parser.parser import model_revision, parse_config

ROOT = Path(__file__).resolve().parent


def model_short(model_path):
    return Path(model_path.rstrip("/")).name.lower()


def load_model(model_path, adapter=None, load_in_8bit=False, dtype="bfloat16"):
    """Loads model and tokenizer (bf16 by default; v1 used 8-bit, kept as an option for the R0 check)."""
    revision = model_revision(model_path)
    kwargs = {"device_map": "cuda" if torch.cuda.is_available() else "cpu", "dtype": getattr(torch, dtype),
              "revision": revision}
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if "gemma" in model_path.lower():
        kwargs["attn_implementation"] = "eager"  # recommended for Gemma-2 (logit soft-capping)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload() if not load_in_8bit else model
    model.eval()
    return model, tokenizer


def select_split(params, metric, split, limit_professions=None):
    df = pd.read_csv(ROOT / params["data_dir"] / f"{'acc' if metric == 'qa' else 'prob'}_{split}.csv")
    if limit_professions:
        keep = sorted(df.profession.unique())[:limit_professions]
        df = df[df.profession.isin(keep)]
    return df


def run(params, metric, split, adapter=None, tag=None, limit_professions=None, out_dir="results",
        model=None, tokenizer=None, batch_size=None, instruction_id=None):
    out_dir = ROOT / out_dir
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    if model is None:
        model, tokenizer = load_model(params["model_path"], adapter, params.get("load_in_8bit", False))
    wrapper = DebiasWrapper(model=model, tokenizer=tokenizer, method=params["debias"],
                            debias_prompt=params.get("debias_prompt"), male_words=params["male_words_ukr"],
                            female_words=params["female_words_ukr"], pca_s=params["pca_s"], pca_e=params["pca_e"])
    method = params["debias"] + (f"_{tag}" if tag else "")
    instruction = INSTRUCTION
    if instruction_id and instruction_id != "base":
        # prompt-sensitivity runs (QA only): the variant becomes part of the run name, e.g. none_pv_order
        instruction = json.loads((ROOT / "configs" / "prompt_variants.json").read_text())[instruction_id]
        method += f"_pv_{instruction_id}"
        assert metric == "qa", "prompt variants only apply to the QA metric"
    meta = {"model": params["model_path"], "method": method, "adapter": adapter,
            "revision": model_revision(params["model_path"]), "instruction": instruction}

    results = {}
    for m in (["qa", "prob"] if metric == "both" else [metric]):
        name = f"{model_short(params['model_path'])}_{method}_{m}_{split}"
        if limit_professions:
            name += f"_pilot{limit_professions}"
        path = out_dir / "raw" / f"{name}.jsonl"
        df = select_split(params, m, split, limit_professions)
        t0 = time.time()
        if m == "qa":
            QAEvaluator(wrapper, batch_size=batch_size or params.get("qa_batch_size", 8),
                        instruction=instruction).run(df, path, meta)
            summary = qa_metrics(qa_wide(read_jsonl(path)))
        else:
            ProbEvaluator(wrapper, batch_size=batch_size or params.get("prob_batch_size", 16)).run(df, path, meta)
            summary = prob_metrics(prob_wide(read_jsonl(path)))
        summary = {**meta, "metric": m, "split": split, "rows": len(df), "seconds": round(time.time() - t0, 1),
                   **{k: (float(v) if not isinstance(v, (bool, int, str)) else v) for k, v in summary.items()}}
        (out_dir / "summary" / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        results[m] = summary
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", help="overrides model_path from the config")
    ap.add_argument("--method", choices=["none", "prompt", "soft", "hard"], help="overrides debias")
    ap.add_argument("--metric", choices=["qa", "prob", "both"], default="both")
    ap.add_argument("--split", choices=["full", "train", "test"], default="full")
    ap.add_argument("--adapter", help="LoRA adapter directory (fine-tuned model)")
    ap.add_argument("--tag", help="suffix for the method name, e.g. ft_s42")
    ap.add_argument("--limit-professions", type=int, help="pilot run on the first N professions")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--instruction", help="prompt variant id from configs/prompt_variants.json (QA only)")
    a = ap.parse_args()
    params = parse_config(a.config)
    if a.model:
        params["model_path"] = a.model
    if a.method:
        params["debias"] = a.method
    if a.load_in_8bit:
        params["load_in_8bit"] = True
    run(params, a.metric, a.split, a.adapter, a.tag, a.limit_professions, a.out_dir, batch_size=a.batch_size,
        instruction_id=a.instruction)


if __name__ == "__main__":
    main()
