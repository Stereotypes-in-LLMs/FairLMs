# Audit of the published (v1, UNLP 2025) results

Everything below was re-measured from the repository at tag `v1.0-unlp2025` (commit `6b5187a`) and the raw
generation file of commit `7d158a6` (`data/raw/full_v1_{acc,prob}.csv`). The scripts that reproduce each number are
named next to it. Items marked **new** were not in the correction plan and were found while implementing it.

## 1. Data

| Check | v1 value | Source |
|---|---|---|
| Professions | 349 (paper and README say 351) | `scripts/build_v2.py` → `data/v2/build_report.json` |
| Rows in raw file | 5608 = two GPT-4 generation passes of 2804 rows | same |
| How train/test were made | test ≈ pass 1 (2772/2800 rows), train ≈ pass 2 (2776/2804 rows) | same |
| Profession overlap train ∩ test | 349 / 349 | `v1.acc.profession_overlap` |
| Test sentences verbatim in train | **1368 / 2800 (48.9%)**: 1368 of the 1400 positives, 0 negatives | `v1.acc.test_sentences_in_train` |
| Why | relevant experience is identical in both passes for 341/349 professions; irrelevant experiences differ | audit script, see `reports/DATASET_v2.md` |
| «урядовець» | 2 blocks per pass (32 rows instead of 16) | `dropped.extra_blocks` |
| «амбасадор» | 2 extra half-blocks of 4 rows (feminitive rows only) | `dropped.incomplete_chunks` |
| Feminitive = base form | «рибалка»: 4 pairs of identical sentences with different `is_feminitive` | `fem_identical` |
| Leading space in `experience` | 1324 test rows, 28 train rows | `v1.acc.experience_leading_space_test` |
| acc / prob files aligned row by row | yes (profession, experience, labels identical) | assert in `build_v2.py` |

Only the **Finetuning** column is affected by the leak. Zero-shot methods never use train.

## 2. Code

| # | Problem | Effect | Fixed in |
|---|---|---|---|
| 2.1 | `debias_word` adds `projectors[0] @ word` twice → returns `w − 2P₀w − P₁w − …` | the first gender component is **reflected**, not removed; all soft/hard numbers | `debias/soft_debias.py`, test `test_legacy_debias_word_reflects_first_component` |
| 2.2 | **new**: `get_word_embedding` tokenizes with special tokens, so for Llama/Gemma every word vector averages in the BOS embedding | gender direction for Llama/Gemma is contaminated by BOS whenever two words have different token counts | `add_special_tokens=False` |
| 2.3 | **new**: `np.linalg.eig` on a symmetric matrix | eigenpairs may come out complex/unsorted; replaced by `eigh` (same subspace) | `debias/soft_debias.py` |
| 2.4 | Debias target word is parsed from the text (`split(" ")`, word after «посаду», drop last char) | breaks with chat templates; silently wrong if generation appends text without a space | target = `job_form` column, located by offset mapping |
| 2.5 | QA answer = 5 argmax tokens, no chat template, cosine to e5 embeddings of «так»/«ні»; no "undetermined"; raw generations not saved | any output (including refusals and empty strings) is forced to yes/no | new protocol, `metrics/acc_metric.py` |
| 2.6 | **new**: Prob metric with the prompt method reads logits at the *prompt* positions but scores the *sentence* tokens | v1 ProbDiff for "prompt" is meaningless (Qwen-3B: 6·10⁻⁶, Qwen-7B: 2·10⁻⁶, about 10⁴ times smaller than any other cell). This is probably why it is missing from the paper | `metrics/prob_metric.py`, test `test_prob_scores_only_sentence_tokens` |
| 2.7 | `UNCONDITIONAL_START_TOKEN = "<s>"` is fed as text: not a BOS for Qwen/Gemma; **new**: for Llama/Gemma the scored token list itself starts with BOS (tokenizer adds it), so the "first-token probability" is P(BOS \| "<s>") | first-token term differs by architecture | BOS if present, else `<|endoftext|>` (as lm-evaluation-harness); only sentence tokens scored |
| 2.8 | Table 6 "Acc Prob Metric" is `prob_f1_diff_*` = share of items where man and woman get the same pos/neg decision, not accuracy against labels | mislabeled table | both quantities reported under correct names (`probagree_*`, `probacc_*`) |
| 2.9 | Configs: `male_words_ukr` points to `female-words.csv` and vice versa | none on projectors (sign of a direction is irrelevant) | fixed |
| 2.10 | `main.py` requires `dataset_percentage`, missing from all configs; `num_runs` unused; `OUTPUT_FILE` placeholder | the published configs do not run as committed | removed; `run_eval.py` CLI |
| 2.11 | Fine-tuning code, training data and the 175-profession list are not in the repository (README says otherwise) | Finetuning column not reproducible | `debias/finetune.py`, `data/finetune/*.jsonl` |

## 3. Published numbers

### 3.1 QADiff as item counts (n = 700 per group)

QADiff = 1 − `acc_diff_*`. `n_*` = QADiff × 700. Non-integer counts mean the value **cannot** be a share of 700 items.

| model | method | F1 man | QADiff fem | QADiff non-fem | n fem | n non-fem | fem > non-fem |
|:--|:--|--:|--:|--:|--:|--:|:--|
| gemma-2-2b-it | none | 0.7891 | 0.2448 | 0.2509 | 171.4 | 175.6 | no |
| gemma-2-2b-it | prompt | 0.6110 | 0.4190 | 0.4091 | 293.3 | 286.4 | yes |
| gemma-2-2b-it | soft | 0.6910 | 0.2870 | 0.2400 | 200.9 | 168.0 | yes |
| gemma-2-2b-it | hard | 0.6780 | 0.2795 | 0.2311 | 195.7 | 161.7 | yes |
| gemma-2-2b-it | ft | 0.9064 | 0.9076* | 0.9118* | 635.3 | 638.3 | — |
| gemma-2-9b-it | none | 0.8103 | 0.1444 | 0.1320 | 101.1 | 92.4 | yes |
| gemma-2-9b-it | prompt | 0.7156 | 0.1930 | 0.1510 | 135.1 | 105.7 | yes |
| gemma-2-9b-it | soft | 0.8342 | 0.1482 | 0.1170 | 103.7 | 81.9 | yes |
| gemma-2-9b-it | hard | 0.8255 | 0.1110 | 0.1105 | 77.7 | 77.3 | yes |
| llama-3.1-8b | none | 0.7259 | 0.3490 | 0.3516 | 244.3 | 246.1 | no |
| llama-3.1-8b | prompt | 0.6329 | 0.3329 | 0.3318 | 233.1 | 232.3 | yes |
| llama-3.1-8b | soft | 0.7091 | 0.7130 | 0.7600 | 499.1 | 532.0 | no |
| llama-3.1-8b | hard | 0.7909 | 0.3291 | 0.2909 | 230.4 | 203.6 | yes |
| llama-3.2-3b | none | 0.6525 | 0.2457 | 0.2271 | 172 | 159 | yes |
| llama-3.2-3b | prompt | 0.5168 | 0.4743 | 0.4714 | 332 | 330 | yes |
| llama-3.2-3b | soft | 0.5810 | 0.2587 | 0.2410 | 181.1 | 168.7 | yes |
| llama-3.2-3b | hard | 0.6091 | 0.2371 | 0.2297 | 166.0 | 160.8 | yes |
| llama-3.2-3b | ft | 0.8950 | 0.0500 | 0.0314 | 35 | 22 | yes |
| qwen2.5-3b | none | 0.6667 | 0.0014 | 0.0043 | **1** | **3** | no |
| qwen2.5-3b | prompt | 0.6686 | 0.0129 | 0.0214 | 9 | 15 | no |
| qwen2.5-3b | soft | 0.6667 | 0.0000 | 0.0014 | **0** | **1** | no |
| qwen2.5-3b | hard | 0.6667 | 0.0000 | 0.0000 | **0** | **0** | no |
| qwen2.5-3b | ft | 0.8999 | 0.0786 | 0.0629 | 55 | 44 | yes |
| qwen2.5-7b | none | 0.7913 | 0.1043 | 0.0714 | 73 | 50 | yes |
| qwen2.5-7b | prompt | 0.7628 | 0.0586 | 0.0514 | 41 | 36 | yes |
| qwen2.5-7b | soft | 0.7753 | 0.1100 | 0.0800 | 77 | 56 | yes |
| qwen2.5-7b | hard | 0.7819 | 0.1286 | 0.0771 | 90 | 54 | yes |

\* The gemma-2-2b fine-tuned file stores `acc_diff_*` ≈ 0.09, apparently already as QADiff and not as agreement like every other file.

**New provenance finding.** For 15 of the 27 QA result files (all Gemma files, all Llama-3.1-8B files, and Llama-3.2-3B soft/hard), `acc_diff_*` is not k/n for any n ≤ 30,000. The repository evaluator returns `matches / len(labels)`, which is always k/700, so these files were produced by some other procedure (possibly averaged runs or an external notebook). The procedure cannot be recovered from the repository. The v2 runs replace all of these numbers.

### 3.2 Qwen2.5-3B is degenerate

F1 = 0.6667 = 2/3 exactly is what an F1 against balanced labels gives when every answer is «так» (precision 1/2, recall 1). After hard debiasing all three groups have F1 = 2/3 and 100% agreement, so the model answers «так» to every item. Its "zero difference" means 0–3 items out of 700 change. That is a constant classifier, not an unbiased one.

### 3.3 "Average relative improvement of 18.9%"

The number reproduces exactly as the mean over 6 models × {fem, non-fem} of (QADiff_none − QADiff_hard) / QADiff_none = **18.94%**. Two of the 12 terms are Qwen2.5-3B going from 1 or 3 items to 0, which counts as +100% each. Without Qwen2.5-3B the mean is **2.73%**. For soft and prompt the same average is negative (−7.2% and −125%).

### 3.4 Other claims

* *"Consistently larger [QADiff] in feminitive contexts"* (no debias): holds for 3/6 models (gemma-9b, llama-3.2-3b, qwen-7b). It does not hold for gemma-2b, llama-8b or qwen-3b.
* *"Cost of lower QA accuracy"*: several cited drops are ≤ 0.002 F1, which is about one item at n = 700.
* No CI, test or seed exists anywhere. 1/700 = 0.00143 is the resolution of every QADiff in the table.

### 3.5 R0: the published code does not reproduce exactly (run 2026-11-01)

`scripts/r0_regression.py` ran the v1 code (taken verbatim from the tag) on the v1 test set with Qwen2.5-7B-Instruct
in 8-bit, as in v1. Only the environment differs: transformers 5.22 vs 4.50, bitsandbytes 0.50 vs 0.42
(`results/r0_regression.json`).

| | published | R0 | difference |
|---|---|---|---|
| QADiff fem | 0.1043 | 0.1000 | 3 / 700 items |
| QADiff non-fem | 0.0714 | 0.0814 | 7 / 700 items |
| F1 man, non-fem | 0.7913 | 0.7966 | |

The pipeline runs and lands in the same range, but changing the numerical environment alone moves QADiff by up to
0.010 (7 items). The v1 protocol takes the argmax of 5 generated tokens and maps the text to yes/no by embedding
similarity, so near-tie items flip. Several v1 between-method differences are of this size or smaller
(e.g. Qwen2.5-7B soft vs none: +4 / +6 items), so they cannot be distinguished from environment noise.

## 4. What this changes

| Result | Leak | Data defects | QA protocol | Projection bug | No statistics | Action |
|---|---|---|---|---|---|---|
| QA none / prompt | – | ✓ | ✓ | – | ✓ | re-run on v2 |
| QA soft / hard | – | ✓ | ✓ | ✓ | ✓ | re-run on v2 |
| QA finetuning | ✓ | ✓ | ✓ | – | ✓ | retrain on train v2, evaluate on unseen professions |
| Prob none / soft / hard | – | ✓ | – | ✓ (soft/hard) | ✓ | re-run on v2 |
| Prob prompt | – | ✓ | – | – | ✓ | v1 value invalid (2.6); computed for the first time |
| Prob finetuning | – | – | – | – | – | computed for the first time |
