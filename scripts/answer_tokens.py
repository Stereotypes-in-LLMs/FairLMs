"""Records how each tokenizer splits the answer variants and the document-start token (configs/answer_tokens.json)."""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics.acc_metric import NO, YES  # noqa: E402
from metrics.prob_metric import doc_prefix_id  # noqa: E402

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-2b-it", "google/gemma-2-9b-it",
          "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]


def main():
    out = {}
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        out[m] = {"answers": {v: {"ids": (ids := tok.encode(v, add_special_tokens=False)),
                                  "tokens": [tok.decode([i]) for i in ids]} for v in YES + NO},
                  "doc_prefix": tok.decode([doc_prefix_id(tok)]), "eos": tok.eos_token, "bos": tok.bos_token}
    (ROOT / "configs" / "answer_tokens.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for m, v in out.items():
        multi = [k for k, a in v["answers"].items() if len(a["ids"]) > 1]
        print(f"{m}: prefix={v['doc_prefix']!r} multi-token answers={multi}")


if __name__ == "__main__":
    main()
