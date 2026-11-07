# Runbook: GPU runs for v2

Everything up to here ran on CPU. The steps below need the GPU and have **not been run yet**.
Environment: `.venv` (Python 3.12, torch 2.14 cu130, transformers 5.22; `requirements.txt`).
All 6 tokenizers (including the gated Gemma/Llama) are already accessible with the cached HF token.

```bash
cd FairLMs
nvidia-smi                                  # make sure the GPU is free
scripts/run_all.sh r0         # ~ minutes  R0: v1 code reproduces Table 1 for Qwen2.5-7B? -> results/r0_regression.json
scripts/run_all.sh pilot      # ~ minutes  6 professions × 6 models × 4 methods; prints yes_rate / undetermined / sample answer
#   -> read the sample answers in results/raw/*_pilot6.jsonl before continuing (chat template, «так»/«ні» tokens)
scripts/run_all.sh zeroshot   # R3+R4: 24 configs × {qa, prob} on full
scripts/run_all.sh finetune   # R5: 3 models × 3 seeds
scripts/run_all.sh ft_eval    # R6
scripts/run_all.sh ft_control        # control adapters (male-only data, same steps): 3 models × 3 seeds
scripts/run_all.sh ft_control_eval   # their QA on full + Prob on test
scripts/run_all.sh prompt_variants   # QA, 6 models × {none, prompt} × {p1, p2, order} = 36 runs
scripts/run_all.sh stats      # R7/R8 (CPU): results/tables, results/figures, reports/RESULTS_v2.md
```

Every stage skips outputs that already exist, so it can be interrupted and resumed. Logs go to `logs/`.
Each `results/summary/*.json` records the wall time of its run. Use the pilot timings to extrapolate before
starting `zeroshot`.

## What to check at each step

| step | look at | pass condition |
|---|---|---|
| R0 | `results/r0_regression.json` | `reproduced: true` (within 2/700 of 0.89571 / 0.92857 / F1 0.79131). If not, the environment differs from v1 (8-bit kernels, transformers version) and the comparison in T7 must say so. Does not block v2. |
| pilot | printed table + `gen_text` in the jsonl | answers start with «Так»/«Ні»; `undetermined` small; `yes_rate` not 0/1 for most models |
| zeroshot | `results/summary/*_qa_full.json` | `degenerate: false` expected for most; `undetermined_*` small; Qwen2.5-3B may legitimately be degenerate (that is a finding) |
| finetune | `checkpoints/*/log_history.json` | eval_loss decreases; best epoch recorded |
| ft_eval | `*_none_ft_s*_qa_full.json` | — |
| ft_control | `checkpoints/control/*/log_history.json` | same step count as the main adapters |
| prompt_variants | `*_pv_*_qa_full.json` | `undetermined` small for every variant |
| stats | `reports/RESULTS_v2.md` | T7 and T8 filled in |

## Manual equivalents

```bash
# one configuration
.venv/bin/python run_eval.py --config configs/config_soft.json --model google/gemma-2-9b-it --metric both --split full
# one fine-tuning run and its evaluation
.venv/bin/python debias/finetune.py --model Qwen/Qwen2.5-3B-Instruct --seed 42
.venv/bin/python run_eval.py --config configs/config_no_debias.json --model Qwen/Qwen2.5-3B-Instruct \
    --adapter checkpoints/qwen2.5-3b-instruct_s42/adapter --tag ft_s42 --metric qa --split full
```

Batch sizes are set in the configs (`qa_batch_size` = rows per batch; each row is 8 candidate sequences,
`prob_batch_size` = sentences per batch). They are conservative and can be raised for the 2–3B models.

Priority if GPU time is short: r0 → pilot → zeroshot → finetune → ft_eval → ft_control(+eval) → prompt_variants.
Every stage is optional for the stats script, which reports whatever exists.

## Optional extensions (plan 3.3)

To add 3 larger fine-tuned models, append them to `FT_MODELS` in `scripts/run_all.sh` and to `models` in
`configs/finetune.json`. The stats script picks them up automatically.
