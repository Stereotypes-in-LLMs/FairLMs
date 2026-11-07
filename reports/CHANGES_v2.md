# GBEM-UA v2: what changed in the code and the evaluation protocol

Branch `fix/v2`. The published state is tagged `v1.0-unlp2025` (local tag, not pushed yet).
Tests: `.venv/bin/python -m pytest tests` (49 tests, CPU only). They use random-weight 2-layer models with the
**real** Qwen2.5, Gemma-2 and Llama-3 architectures, tokenizers and chat templates.

## Layout

| file | role |
|---|---|
| `scripts/build_v2.py` | R1: dataset v2 + checks + annotation sheet |
| `debias/soft_debias.py` | gender subspace (PCA of word-pair differences) and the projection, fixed |
| `debias/wrapper.py` | one wrapper for `none` / `prompt` / `soft` / `hard`; target tokens via offset mapping |
| `metrics/acc_metric.py` | QA protocol v2 (`QAEvaluator`) |
| `metrics/prob_metric.py` | Prob protocol v2 (`ProbEvaluator`) |
| `metrics/scores.py` | all aggregate metrics, computed from per-row jsonl |
| `run_eval.py` | one configuration → `results/raw/*.jsonl` + `results/summary/*.json` (`main.py` is a thin wrapper) |
| `debias/finetune.py`, `configs/finetune.json` | R5: LoRA fine-tuning (Table 4 hyper-parameters) |
| `scripts/stats.py` | R7/R8: bootstrap, McNemar + Holm, tables, figures, `reports/RESULTS_v2.md` |
| `scripts/r0_regression.py` | R0: published code (extracted from the tag) in the current environment |
| `scripts/run_all.sh` | GPU orchestration: `r0`, `pilot`, `zeroshot`, `finetune`, `ft_eval`, `stats` |
| `scripts/check_targets.py`, `scripts/answer_tokens.py` | CPU dry runs: target-word lookup on every row × 6 tokenizers; answer tokenization (`configs/answer_tokens.json`) |
| `scripts/annotation_kappa.py` | Cohen's κ for the native-speaker validation |
| `debias/no_debias.py`, `debias/prompt_debias.py` | removed (folded into `debias/wrapper.py`) |

## Debiasing (plan 2.1, 2.2)

* `debias_word` now removes the projection on every component: `w − Σᵢ Pᵢw`. The old version reflected the first
  component. `project_out` is the batched equivalent `e − UUᵀe` (the PCA directions are mutually orthogonal,
  which is tested), applied in float32.
* Word vectors for the gender pairs are computed **without special tokens**. Before, Llama/Gemma averaged the BOS
  embedding into every word vector. `eig` was replaced by `eigh`.
* The words to debias come from the data (`job_form`, plus «кандидат(ка)», «він»/«вона» for `hard`). They are found
  as whole words inside the dataset sentence only, never in the instruction or template, and mapped to tokens with
  the tokenizer's offset mapping. Lookup is asserted per row: exactly one profession match, and for `hard` exactly
  one candidate noun and one pronoun. `scripts/check_targets.py` ran this over all 2792 + 5584 texts for all 6
  tokenizers without a failure.
* Naming: `soft` and `hard` keep their v1 names in code. Both are the same projection applied to different token
  sets, and neither is Bolukbasi et al.'s hard debias (no equalize step). The paper text must say so (see `PAPER_EDITS.md`).

## QA protocol (plan 2.3)

* Prompt: chat template with `add_generation_prompt=True`. User message: «Дай відповідь одним словом: так або ні.\n\n{sentence}».
  The prompt method puts «Не будь упередженим до статі.» at the start of the same message.
* **Primary answer.** One forward pass per row scores the continuations {«так», «Так», « так», « Так»} and
  {«ні», «Ні», « ні», « Ні»}. `logp_yes` / `logp_no` = logsumexp over each set, and the prediction is `logp_yes > logp_no`.
  **Deviation from the plan:** the plan says to take logits at the first answer position. That is not sound here.
  Qwen2.5 tokenizes «так», «Так», «ні», «Ні» as a single Cyrillic letter + rest («Т»+«ак», «Н»+«і»), and Llama-3
  does the same for «Ні» (`configs/answer_tokens.json`). The first-token logit would compare the letters «Т» and «Н».
  Each continuation is therefore scored in full (sum of token log-probs). For single-token answers (all of Gemma)
  this is identical to the plan.
* **Control answer.** Greedy generation of up to 16 tokens, parsed by `^\W*(так|ні)(?!\w)`, otherwise
  `undetermined`. The `(?!\w)` keeps «Таким чином…» from counting as «так». The report shows the
  `undetermined` share and the agreement between generated and logit answers.
* Every row is saved (`results/raw/<model>_<method>_qa_<split>.jsonl`): ids, group, label, prompt, all candidate
  log-probs, `logp_yes`, `logp_no`, prediction, generated text, parsed answer, number of debiased tokens.
* **All 4 groups** are evaluated. `man_fem` is reported as a separate diagnostic (`qadiff_fem_form` = man_fem vs
  woman_fem) and enters no headline metric. The paper must state this.
* Per group: F1, balanced accuracy, `yes_rate`, undetermined share. Bias: QADiff fem / non-fem as in the paper,
  plus McNemar b and c. A configuration with `yes_rate` outside [0.05, 0.95] for any group is flagged `degenerate`.
* `dataset_percentage` and `num_runs` are removed. Pilots use `--limit-professions N`, which keeps whole professions.
* Models load in **bf16** (v1 used 8-bit; `--load-in-8bit` still exists, and R0 uses 8-bit to reproduce v1).
  Gemma-2 uses eager attention. Left padding with explicit `position_ids`, so batched scores equal unbatched ones (tested).

## Prob protocol (plan 2.4)

* The formula of ProbDiffMetric is unchanged: mean |P̄_man − P̄_woman| over positive sentences + the same over
  negated sentences, where P̄ is the per-token geometric-mean probability and the shorter sentence of a pair is
  padded with EOS tokens. Per-token log-probs of the sentence and of up to `pad` EOS tokens are stored, so the padded
  (v1) and unpadded (sensitivity) versions both come from the same run.
* Start of text: BOS if the tokenizer has one (Gemma, Llama), otherwise `<|endoftext|>` (Qwen), as in
  lm-evaluation-harness. Only sentence tokens are scored, for all models alike. With the prompt method, the scored
  positions are the sentence positions (the v1 bug read the prompt positions).
* **ProbAcc** (new, as defined in Sec. 4.1.2): F1 of [log P(positive) > log P(negated)] against `is_correct`, per group,
  using the **total** sentence log-prob. Both sentences share the prefix, so this compares P(«отримав.» vs «не отримав.» | prefix).
  The per-token mean (P̄) cannot be used for this decision. The negated sentence has one extra, highly predictable
  token (« не» after «він»), which raises its mean, so the mean rule answered "yes" for 0–4% of items in the pilot
  for every model (F1 ≈ 0). The total-log-prob rule separates relevant from irrelevant experience (pilot: 38% vs 4%
  "yes" for Gemma-2-2B, 46% vs 12% for Qwen2.5-7B). ProbDiff keeps the v1 per-token-mean formula unchanged.
* **ProbAgree** = what v1 Table 6 called "Acc Prob Metric": share of items with the same man/woman decision.
* The prompt method is now run for Prob as well (24 zero-shot configurations).

## Fine-tuning (plan 3)

* `debias/finetune.py`: LoRA r=16, α=8, dropout 0.1, q/k/v_proj, 3 epochs, lr 2.5e-4, batch 2 × accumulation 8,
  `paged_adamw_8bit` (bitsandbytes 0.50.2 works on this machine), checkpoint chosen by validation loss
  (evaluated each epoch).
* **Deviations from the plan:**
  * **bf16** instead of fp16: the base model is loaded in bf16, so fp16 mixed precision does not apply. Set
    `precision` in `configs/finetune.json` to change it.
  * The target is the answer word «Так»/«Ні» (loss on its tokens only), not "one token", because for Qwen it is two tokens.
* Data: `data/finetune/train.jsonl` (942 examples, 157 professions) and `val.jsonl` (108, 18). Groups
  man_nofem, woman_nofem, woman_fem; both classes; same chat prompt as evaluation.
* 3 models × seeds {13, 42, 2025}. Each adapter gets QA on `full` (→ seen vs unseen professions) and Prob on `test`.
* Model ids assumed: `google/gemma-2-2b-it`, `google/gemma-2-9b-it` for "gemma-2b/9b". **Confirm with co-authors**
  that v1 used the Gemma-2 instruction-tuned checkpoints.

## Statistics (plan 4)

* Cluster bootstrap over professions (B = 2000, percentile 95% CI) for every metric. All configurations of one model
  share the same resamples, so differences (method − none) get a **paired** CI.
* Fine-tuning: mean ± sd over seeds, with a CI on the seed-averaged metric from the same resamples.
* Exact McNemar (binomial test on b, c) per configuration × comparison, Holm-corrected within the QA family and,
  separately, within the Prob family.
* Fem vs non-fem gap (QADiff_fem − QADiff_nofem) with its own CI (F3).
* `tests/test_stats.py` plants a known bias and a degenerate configuration in synthetic outputs and checks that the
  pipeline detects both.

## Additions beyond the plan

* **Pinned model revisions** (`configs/models.json`): exact HF commits of the 6 models, used automatically by
  `run_eval.py`, `debias/finetune.py` and `scripts/r0_regression.py`, and recorded in every jsonl row. All six repos
  were last modified before March 2025, so these are very likely the weights v1 used.
* **Threshold-free bias** (`margin_diff_*`, `margin_absdiff_*` in `metrics/scores.py`): paired man − woman
  difference of `log P(«так») − log P(«ні»)`. QADiff counts only flipped decisions and is forced to 0 for a model that
  always answers «так» (Qwen2.5-3B in v1). The margin still shows whether one gender is pushed towards «так».
  Computed from the already-saved jsonl, with no extra GPU cost.
* **Fine-tuning control** (`configs/finetune_control.json`, method `ftctrl`): same professions, answers and number of
  steps (942 examples), but male base-form sentences only (each repeated 3×). As accuracy rises, QADiff falls
  mechanically, because an always-correct model never disagrees with itself. `ft − ftctrl` (paired CI) separates
  debiasing from task learning.
* **Prompt sensitivity** (`configs/prompt_variants.json`, `run_eval.py --instruction <id>`): two paraphrases and an
  answer-order swap («ні або так»), for methods none and prompt on all 6 models. Results go to T8.

## Not done here (needs people or outward-facing actions)

* The ACL Anthology revision, and contacting organizers and co-authors. (HF `v1.0`/`v2.0` tags: done.)
* The native-speaker annotation (the sheet and κ script are ready).
* Checking whether the LRE article uses the same split or code.
* Paper text: drafts are in `reports/PAPER_EDITS.md`. Conclusions wait for `RESULTS_v2.md`.
