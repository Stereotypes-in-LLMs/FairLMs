# GBEM-UA v2: conclusions

Hand-written reading of [RESULTS_v2.md](RESULTS_v2.md) (generated; every number below is taken from it or from
`results/tables/*.csv`). Runs: 6 models × 4 zero-shot methods × {QA, Prob} on `full` (349 professions);
3 models × 3 seeds × {ft, male-only control} on `test` (174 unseen professions); 6 models × 2 methods × 3 prompt
variants. All 168 runs finished. Undetermined control generations: 0 everywhere except Gemma-2-9B
none/soft/hard (≤ 0.3%, man group).
95% CIs: cluster bootstrap over professions, B = 2000. "Significant" = paired CI excludes 0 (McNemar: Holm p < 0.05).

## 1. Validity

* **No main configuration is degenerate.** Every model answers «так» to 56–89% of items in every group
  (T2, F2). In particular, Qwen2.5-3B answers «так» to 70% (v1: 100%, F1 = 2/3), so its v1 "zero difference" was an
  artifact of the old answer extraction.
* The only degenerate cells are Llama-3.2-3B under paraphrase `p1` (none and prompt): it answers «ні» to
  ≥ 99% of items. Its QADiff there (0.004 / 0.027) must not be read as "unbiased".
* Fine-tuning converged for all 18 adapters (train loss < 0.03). Validation loss rises after epoch 1 for Llama-3.2-3B
  (0.20–0.31 → 0.28–0.47) in both arms, so both arms overfit equally; the comparison between them is unaffected.

## 2. Zero-shot bias (T3)

* QADiff (share of items whose answer flips with the candidate's gender) is **0.02–0.12** with feminitives and
  **0.02–0.08** without: Llama-3.1-8B is the least affected (0.019 / 0.017), Llama-3.2-3B the most (0.124 / 0.076).
  These are 6–43 items out of 349 per comparison.
* **Feminitive gap.** QADiff is larger with a feminitive job title than with the base form in 4/6 models
  (gap CI > 0: Gemma-2-2B, Gemma-2-9B, Llama-3.2-3B, Qwen2.5-3B; +0.025 to +0.048). It is not significant for
  Llama-3.1-8B and Qwen2.5-7B. "Consistently larger" (paper, p. 67) → "larger in 4 of 6 models".
* **Direction (new; McNemar and margin_diff).** The direction depends on the job-title form:
  * *Both base form* (`nofem`): when the answers differ, the man gets «так» more often. This is significant in
    3/6 models (Gemma-2-9B 28 vs 5, Qwen2.5-3B 35 vs 5, Qwen2.5-7B 31 vs 8 discordant pairs), and the threshold-free
    margin favours the man in 4/6 models (margin_diff_nofem > 0, CI excludes 0).
  * *Woman with feminitive vs man with base form* (`fem`): the woman is favoured or the result is not significant.
    margin_diff_fem < 0 (CI excludes 0) in 5/6 models; QA McNemar significant only for Llama-3.2-3B (13 vs 73).
  * So the bias is **not** "feminitives hurt women". In these models a female candidate described with the
    masculine job title is disadvantaged relative to a male one, while the feminitive form shifts the model towards
    «так» for her. The QADiff gap above measures *more flips*, not *more flips against women*.

## 3. Zero-shot debiasing (F1, T7)

* **Hard (proj-ext): no reduction.** ΔQADiff CI < 0 in 0/12 model × context cells. The v1 "18.9% average relative
  improvement" does not survive; it came from Qwen2.5-3B's degenerate 0–3 items and from averaging relative changes.
* **Soft (proj-prof): no effect on bias.** 0/12 cells change in either direction.
* The projections are not no-ops: they change 1–5% of answers relative to `none` (soft 0.7–4.3%, hard 1.6–5.5%),
  but the changed answers do not reduce the gender disagreement. ΔProbDiff for soft is ≤ 3·10⁻⁴ everywhere.
* **Prompt: inconsistent.** Lower QADiff in 2/12 cells (Gemma-2-9B fem −0.032, Qwen2.5-7B nofem −0.014), higher in
  2/12 (Llama-3.1-8B fem +0.019, Qwen2.5-3B nofem +0.020). For Gemma-2-9B the reduction coincides with F1 −0.05
  (the model says «так» to 89% instead of 79% of items), i.e. it comes from answering more uniformly.
* **Cost in accuracy.** 17/72 ΔF1 cells are significantly negative, mostly small (−0.004 to −0.017). The exception is
  prompt debiasing on Gemma-2-9B (−0.05 to −0.07 in all four groups).

## 4. Fine-tuning (T5, T6, F4)

* On the 174 unseen professions fine-tuning raises F1 by 0.10–0.14 and lowers QADiff significantly in 4/6 cells
  (Gemma-2-2B −0.031 / −0.023, Llama-3.2-3B −0.080 / −0.060). Qwen2.5-3B: not significant.
* **The male-only control does the same.** Training on male base-form sentences only (same professions, labels and
  steps) gives the same F1 and statistically indistinguishable QADiff: ft − control is not significantly negative in
  any of 6 cells. For Qwen2.5-3B with feminitives the gender-balanced adapter is *worse* (+0.019, CI excludes 0).
  So the QADiff reduction after fine-tuning comes from learning the task (answers become more confident and
  correct), not from seeing female or feminitive examples.
* **Threshold-free gap: a real effect only for Gemma-2-2B.** The mean absolute man − woman log-odds difference
  (margin_absdiff) is lower for the gender-balanced adapter than for the control in 6/6 cells (−0.10 to −0.41 nats,
  paired CI excludes 0). But the gender-balanced adapters are also less confident overall (mean |margin| on test:
  Gemma 6.9 vs 7.1, Llama 4.2 vs 4.9, Qwen 4.2 vs 6.5), which by itself shrinks absolute gaps. Relative to that
  scale (point estimates, no CI), the gap shrinks clearly for Gemma-2-2B (fem 0.091 vs 0.140, nofem 0.059 vs 0.082),
  only slightly for Llama-3.2-3B (0.134 vs 0.143, 0.083 vs 0.092), and not for Qwen2.5-3B (0.225 vs 0.204,
  0.159 vs 0.166). So the evidence that gender-balanced data adds debiasing beyond task learning is limited to
  one of three models. A scale-normalized metric with a CI is needed before this goes into the paper.
* **Memorization (T6).** F1 (man, base form) is 0.91–1.00 on training professions vs 0.89–0.96 on unseen ones
  (all seeds, both arms).
  v1 evaluated fine-tuning on a test set sharing 1368/1400 positives with train, so its numbers measured this
  "seen" regime.
* Seeds differ noticeably (Qwen2.5-3B ft QADiff on unseen professions: 0.037–0.086 with feminitives), so single-seed
  fine-tuning results are not reliable.

## 5. Probability metric (T4)

* Prompt debiasing **increases** ProbDiff for 5/6 models (e.g. Qwen2.5-3B 0.007 → 0.041 with feminitives). The debias
  phrase shifts sentence probabilities and does so unequally for the two genders. Only Qwen2.5-7B decreases.
* Hard debiasing lowers ProbDiff for Llama-3.1-8B and Qwen2.5-7B and raises it for Llama-3.2-3B and Qwen2.5-3B;
  for the Gemma models the change is < 3·10⁻⁴. Soft does nothing measurable.
* ProbAcc (does the model prefer «отримав» over «не отримав» when the experience is relevant?) is low:
  F1 0.02–0.80, often near or below chance. The statement format barely reflects the relevance of the experience, so
  ProbDiff measures gender sensitivity of the sentence probability, not of a hiring decision. It should be reported as
  a secondary metric.

## 6. Robustness to the instruction (T8)

* For 5/6 models the qualitative picture (QADiff range, feminitive gap sign, no consistent method effect) is the same
  for the base instruction, two paraphrases and the «так»/«ні» order swap.
* Individual numbers move by up to ±0.04 QADiff (Qwen2.5-3B prompt: 0.075–0.131 with feminitives), comparable to
  the method effects in F1. Single-instruction method comparisons of that size are therefore prompt-dependent.
* Qwen2.5-3B's «так» rate goes from 0.70 to 0.89 when the order is swapped, and Llama-3.2-3B collapses to «ні» under
  `p1`: small models are fragile to the wording.

## 7. What the revised paper can claim

1. Ukrainian instruction-tuned LLMs change their hiring answer with the candidate's gender for 2–12% of items. The
   effect is significant and in base-form contexts it favours men in 3–4/6 models.
2. Feminitive job titles increase the number of gender-dependent answers in 4/6 models, but shift the answers
   towards «так» for the female candidate. They do not disadvantage her.
3. Embedding projection (soft/hard) and a debiasing prompt do not reduce gender disagreement reliably. The v1
   18.9% claim is withdrawn.
4. LoRA fine-tuning reduces disagreement on unseen professions, but a male-only control reduces it equally. The
   gain comes from learning the task; a gender-specific effect is visible only in Gemma-2-2B's margin gap.
5. Results depend on the instruction wording at the ±0.04 level, and small models can become degenerate.

## 8. Open caveats

* Native-speaker validation of the data is pending (κ not available). Conclusions about feminitives assume the
  feminitive forms are correct.
* Fine-tuning was run on three small models only (2–3B). The 7–9B models were not fine-tuned.
* R0 showed that v1 numbers move by up to 0.010 from the software environment alone. v2 numbers are pinned to
  model revisions, but the same caveat applies across library versions.
* The provenance of 15/27 v1 QA result files is unknown (AUDIT 3.1), so for those cells the T7 comparison is with
  numbers whose procedure cannot be checked.
