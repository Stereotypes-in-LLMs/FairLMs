#!/usr/bin/env bash
# GPU runs for GBEM-UA v2 (R0, R3-R7). Stages are idempotent: finished outputs are skipped.
#   scripts/run_all.sh r0        # regression: v1 code reproduces Table 1 (Qwen2.5-7B, no debias)
#   scripts/run_all.sh pilot     # R3/R4 on 6 professions per model x method; inspect raw answers
#   scripts/run_all.sh zeroshot  # R3+R4: 6 models x {none,prompt,soft,hard}, QA+Prob on full
#   scripts/run_all.sh finetune  # R5: 3 models x 3 seeds LoRA
#   scripts/run_all.sh ft_eval   # R6: QA on full (seen+unseen), Prob on test
#   scripts/run_all.sh ft_control      # control adapters: male base-form sentences only, same steps
#   scripts/run_all.sh ft_control_eval # their QA on full + Prob on test
#   scripts/run_all.sh prompt_variants # QA with 3 instruction variants, methods none + prompt, 6 models
#   scripts/run_all.sh stats     # R7/R8: CIs, McNemar, tables, figures, reports/RESULTS_v2.md
#   scripts/run_all.sh all       # everything in order
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS=(Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct google/gemma-2-2b-it google/gemma-2-9b-it
        meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct)
FT_MODELS=(Qwen/Qwen2.5-3B-Instruct google/gemma-2-2b-it meta-llama/Llama-3.2-3B-Instruct)
SEEDS=(13 42 2025)
METHODS=(none prompt soft hard)
mkdir -p logs

short() { basename "$1" | tr '[:upper:]' '[:lower:]'; }
done_run() { [[ -f "results/summary/$1.json" ]]; }

r0() { $PY scripts/r0_regression.py 2>&1 | tee logs/r0.log; }

pilot() {
  for m in "${MODELS[@]}"; do for me in "${METHODS[@]}"; do
    $PY run_eval.py --config configs/config_no_debias.json --model "$m" --method "$me" --metric both \
      --split full --limit-professions 6 2>&1 | tee "logs/pilot_$(short "$m")_${me}.log" | tail -3
  done; done
  $PY - <<'PY'
import glob, json
for f in sorted(glob.glob("results/raw/*_qa_full_pilot6.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    und = sum(r["gen_answer"] == "undetermined" for r in rows) / len(rows)
    yes = sum(r["pred"] for r in rows) / len(rows)
    print(f"{f.split('/')[-1]:60s} yes_rate={yes:.2f} undetermined={und:.2f} e.g. {rows[0]['gen_text']!r}")
PY
}

zeroshot() {
  for m in "${MODELS[@]}"; do for me in "${METHODS[@]}"; do
    for metric in qa prob; do
      name="$(short "$m")_${me}_${metric}_full"
      done_run "$name" && { echo "skip $name"; continue; }
      $PY run_eval.py --config configs/config_no_debias.json --model "$m" --method "$me" --metric "$metric" \
        --split full 2>&1 | tee "logs/${name}.log" | tail -2
    done
  done; done
}

finetune() {
  for m in "${FT_MODELS[@]}"; do for s in "${SEEDS[@]}"; do
    dir="checkpoints/$(short "$m")_s${s}"
    [[ -f "$dir/adapter/adapter_config.json" ]] && { echo "skip $dir"; continue; }
    $PY debias/finetune.py --model "$m" --seed "$s" 2>&1 | tee "logs/ft_$(short "$m")_s${s}.log" | tail -3
  done; done
}

ft_eval() {  # $1 = checkpoint dir, $2 = tag prefix
  local root="${1:-checkpoints}" prefix="${2:-ft}"
  for m in "${FT_MODELS[@]}"; do for s in "${SEEDS[@]}"; do
    adapter="${root}/$(short "$m")_s${s}/adapter"
    for spec in "qa full" "prob test"; do
      read -r metric split <<<"$spec"
      name="$(short "$m")_none_${prefix}_s${s}_${metric}_${split}"
      done_run "$name" && { echo "skip $name"; continue; }
      $PY run_eval.py --config configs/config_no_debias.json --model "$m" --adapter "$adapter" \
        --tag "${prefix}_s${s}" --metric "$metric" --split "$split" 2>&1 | tee "logs/${name}.log" | tail -2
    done
  done; done
}

ft_control() {
  for m in "${FT_MODELS[@]}"; do for s in "${SEEDS[@]}"; do
    dir="checkpoints/control/$(short "$m")_s${s}"
    [[ -f "$dir/adapter/adapter_config.json" ]] && { echo "skip $dir"; continue; }
    $PY debias/finetune.py --config configs/finetune_control.json --model "$m" --seed "$s" 2>&1 \
      | tee "logs/ftctrl_$(short "$m")_s${s}.log" | tail -3
  done; done
}

ft_control_eval() { ft_eval checkpoints/control ftctrl; }

prompt_variants() {
  for m in "${MODELS[@]}"; do for me in none prompt; do for v in p1 p2 order; do
    name="$(short "$m")_${me}_pv_${v}_qa_full"
    done_run "$name" && { echo "skip $name"; continue; }
    $PY run_eval.py --config configs/config_no_debias.json --model "$m" --method "$me" --metric qa \
      --split full --instruction "$v" 2>&1 | tee "logs/${name}.log" | tail -2
  done; done; done
}

stats() { $PY scripts/stats.py --B 2000 && $PY -m scripts.insights --B 2000; }

case "${1:-}" in
  r0|pilot|zeroshot|finetune|ft_eval|ft_control|ft_control_eval|prompt_variants|stats) "$1" ;;
  all) r0; pilot; zeroshot; finetune; ft_eval; ft_control; ft_control_eval; prompt_variants; stats ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
