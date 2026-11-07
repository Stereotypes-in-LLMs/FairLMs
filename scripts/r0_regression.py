"""
R0: regression check - the published (v1) code on the v1 test set, in the current environment.
Expected for Qwen2.5-7B-Instruct, no debias (Table 1): QADiff fem = 0.10429, non-fem = 0.07143
(acc_diff_feminitive = 0.89571, acc_diff_no_feminitive = 0.92857).

The legacy code is taken verbatim from the tag v1.0-unlp2025; only the model loader is replaced
(transformers 5 removed `load_in_8bit=` in from_pretrained) and "cuda" is made configurable.

    python scripts/r0_regression.py                      # full check, GPU, 8-bit as in v1
    python scripts/r0_regression.py --percentage 0.02    # quick pilot
"""
import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "v1.0-unlp2025"
FILES = ["utils.py", "debias/__init__.py", "debias/no_debias.py", "debias/prompt_debias.py",
         "debias/soft_debias.py", "metrics/__init__.py", "metrics/acc_metric.py", "metrics/prob_metric.py"]
EXPECTED = {"acc_diff_feminitive": 0.8957142857142857, "acc_diff_no_feminitive": 0.9285714285714286,
            "acc_man_no_feminitive": 0.7913148371531966}


def extract_legacy(dest, device):
    for f in FILES:
        src = subprocess.run(["git", "show", f"{TAG}:{f}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        out = dest / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(src.replace('"cuda"', repr(device)))


def run_legacy(model, tokenizer, percentage, device, embeddings_model="intfloat/multilingual-e5-large",
               dataset=ROOT / "data/acc_dataset/test_dataset_acc.csv", output_file=None):
    tmp = Path(tempfile.mkdtemp(prefix="legacy_v1_"))
    extract_legacy(tmp, device)
    sys.path.insert(0, str(tmp))
    for name in [m for m in sys.modules if m.split(".")[0] in ("debias", "metrics", "utils")]:
        del sys.modules[name]
    try:
        debias = importlib.import_module("debias")
        metrics = importlib.import_module("metrics")
        from sentence_transformers import SentenceTransformer
        wrapper = debias.PlainModelWrapper(model_name="legacy", model=model, tokenizer=tokenizer)
        evaluator = metrics.AccEvaluator(file_path=str(dataset), embeddings_model=SentenceTransformer(
            embeddings_model, device=device), tokenizer=tokenizer, model=wrapper, dataset_percentage=percentage,
            output_file=str(output_file or tmp / "model_accuracy.json"))
        return evaluator.predict()
    finally:
        sys.path.remove(str(tmp))
        for name in [m for m in sys.modules if m.split(".")[0] in ("debias", "metrics", "utils")]:
            del sys.modules[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--percentage", type=float, default=1.0)
    ap.add_argument("--no-8bit", action="store_true", help="bf16 instead of the 8-bit loading used in v1")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    sys.path.insert(0, str(ROOT))
    from parser.parser import model_revision
    revision = model_revision(a.model)
    kwargs = {"device_map": "cuda", "revision": revision}
    if a.no_8bit:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    with torch.no_grad():
        result = run_legacy(model, tokenizer, a.percentage, "cuda")
    report = {"model": a.model, "percentage": a.percentage, "8bit": not a.no_8bit,
              **{k: float(v) for k, v in result.items()}}
    if a.percentage == 1.0 and a.model == "Qwen/Qwen2.5-7B-Instruct":
        report["expected"] = EXPECTED
        report["max_abs_dev"] = max(abs(report[k] - v) for k, v in EXPECTED.items())
        report["reproduced"] = report["max_abs_dev"] < 2 / 700
    out = ROOT / "results" / "r0_regression.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
