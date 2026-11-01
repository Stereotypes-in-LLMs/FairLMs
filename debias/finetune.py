"""
LoRA fine-tuning (paper, Table 4): r=16, alpha=8, dropout 0.1, q/k/v_proj, 3 epochs, lr 2.5e-4,
batch 2 x grad-accum 8, paged_adamw_8bit.

Training data = train-split professions only (data/v2/acc_train.csv); ~10% of them are held out
(group split) to choose the checkpoint by validation loss. Both classes are used: relevant -> "Так",
irrelevant -> "Ні", for every gender variant that enters the evaluation. The input is the same chat
prompt as in evaluation (metrics/acc_metric.build_prompt); the loss is computed on the answer tokens only.

    python debias/finetune.py --model Qwen/Qwen2.5-3B-Instruct --seed 42
    python debias/finetune.py --build-data-only
    python debias/finetune.py --config configs/finetune_control.json --model ... --seed 42   # control (see below)

Control (configs/finetune_control.json): same professions, same answers and the same number of gradient steps, but
male base-form sentences only (each repeated 3x). Accuracy on the task rises the same way, but the model never sees
the female variants. If QADiff on unseen professions falls as much as with the main adapter, the drop comes from
learning the task (an always-correct model never disagrees with itself), not from gender-specific debiasing.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from debias.wrapper import DebiasWrapper  # noqa: E402
from metrics.acc_metric import build_prompt  # noqa: E402
from parser.parser import model_revision  # noqa: E402

ANSWERS = {True: "Так", False: "Ні"}


def build_data(cfg):
    """Writes data/finetune/{train,val}.jsonl (model-agnostic: sentence + answer) and returns them."""
    df = pd.read_csv(ROOT / cfg["train_csv"])
    assert (df.split == "train").all(), "fine-tuning must only use train-split professions"
    df = df[df.group.isin(cfg["groups"])]
    profs = pd.Series(sorted(df.profession.unique()))
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["val_fraction"], random_state=cfg["val_seed"])
    tr_idx, va_idx = next(gss.split(profs, groups=profs))
    val_profs = set(profs.iloc[va_idx])
    out = ROOT / cfg.get("data_dir", "data/finetune")
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, part in [("train", df[~df.profession.isin(val_profs)]), ("val", df[df.profession.isin(val_profs)])]:
        records = [{"row_id": r.row_id, "profession": r.profession, "group": r.group, "sentence": r.sentence,
                    "answer": ANSWERS[bool(r.is_correct)]} for r in part.itertuples()]
        if name == "train":
            records = records * cfg.get("repeat", 1)
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        data[name] = records
    assert not {r["profession"] for r in data["train"]} & {r["profession"] for r in data["val"]}
    print({k: (len(v), len({r['profession'] for r in v})) for k, v in data.items()})
    return data


class AnswerDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer):
        wrapper = DebiasWrapper(model=None, tokenizer=tokenizer, method="none")
        self.items = []
        for r in records:
            prompt_ids = tokenizer(build_prompt(wrapper, r["sentence"]), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
            self.items.append({"input_ids": prompt_ids + answer_ids,
                               "labels": [-100] * len(prompt_ids) + answer_ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def fn(batch):
        T = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], labels[i, :n], mask[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


def pick_optim(name):
    if "8bit" in name:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print(f"bitsandbytes is not available, falling back from {name} to adamw_torch")
            return "adamw_torch"
    return name


def train(model, tokenizer, cfg, data, out_dir, seed, max_steps=-1, use_cpu=False):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    t = cfg["training"]
    args = TrainingArguments(
        output_dir=str(out_dir), seed=seed, data_seed=seed, num_train_epochs=t["epochs"], max_steps=max_steps,
        learning_rate=t["lr"], per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"], gradient_accumulation_steps=t["grad_accum"],
        optim=pick_optim(t["optim"]), bf16=t["precision"] == "bf16", fp16=t["precision"] == "fp16",
        eval_strategy="epoch" if max_steps < 0 else "steps", save_strategy="epoch" if max_steps < 0 else "steps",
        eval_steps=max_steps if max_steps > 0 else None, save_steps=max_steps if max_steps > 0 else None,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=1, logging_steps=10, report_to=[], remove_unused_columns=False, use_cpu=use_cpu)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=AnswerDataset(data["train"], tokenizer),
                      eval_dataset=AnswerDataset(data["val"], tokenizer), data_collator=collate(pad))
    trainer.train()
    adapter_dir = Path(out_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    (Path(out_dir) / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (Path(out_dir) / "finetune_config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2, ensure_ascii=False))
    return adapter_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "finetune.json"))
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build-data-only", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    data = build_data(cfg)
    if a.build_data_only:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["training"]["precision"], torch.float32)
    revision = model_revision(a.model)
    kwargs = {"dtype": dtype, "device_map": "cuda", "revision": revision}
    if "gemma" in a.model.lower():
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    out_dir = ROOT / cfg["output_dir"] / f"{Path(a.model).name.lower()}_s{a.seed}"
    print("adapter saved to", train(model, tokenizer, cfg, data, out_dir, a.seed))


if __name__ == "__main__":
    main()
