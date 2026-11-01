import json

from conftest import tiny_model
from debias.finetune import AnswerDataset, build_data, train

CFG = json.load(open("configs/finetune.json"))


def test_training_data_is_train_split_with_both_classes():
    data = build_data(CFG)
    assert {r["answer"] for r in data["train"]} == {"Так", "Ні"}
    test_profs = set(__import__("pandas").read_csv("data/v2/acc_test.csv").profession)
    assert not test_profs & {r["profession"] for part in data.values() for r in part}


def test_loss_only_on_answer_tokens():
    _, tok = tiny_model("Qwen/Qwen2.5-3B-Instruct")
    ds = AnswerDataset([{"sentence": "Кандидат має досвід. Чи може він отримати посаду хірурга?", "answer": "Так"}], tok)
    item = ds[0]
    supervised = [t for t, l in zip(item["input_ids"], item["labels"]) if l != -100]
    assert tok.decode(supervised) == "Так"


def test_lora_smoke(tmp_path):
    model, tok = tiny_model("Qwen/Qwen2.5-3B-Instruct")
    data = build_data(CFG)
    data = {k: v[:8] for k, v in data.items()}
    cfg = {**CFG, "training": {**CFG["training"], "optim": "adamw_torch", "precision": "fp32", "grad_accum": 1}}
    adapter = train(model, tok, cfg, data, tmp_path, seed=13, max_steps=2, use_cpu=True)
    assert (adapter / "adapter_config.json").exists()
