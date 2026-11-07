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

## 2a. New insights (not in the v1 paper)

From `scripts/insights.py` (method `none`, `full`, «рибалка» excluded; cluster bootstrap B = 2000; tables
`results/tables/insights_*.csv`).

**I1. Grammatical agreement drives the answers.** For the same candidate, a job title whose grammatical gender
matches the candidate raises the share of «так». The effect is significant in **6/6 models for both genders**:

| model | woman: feminitive − base title | man: base − feminitive title |
|---|---|---|
| gemma-2-2b-it | +0.042 [+0.023, +0.060] | +0.066 [+0.046, +0.088] |
| gemma-2-9b-it | +0.047 [+0.029, +0.068] | +0.045 [+0.024, +0.065] |
| llama-3.1-8b-instruct | +0.016 [+0.004, +0.027] | +0.019 [+0.006, +0.033] |
| llama-3.2-3b-instruct | +0.096 [+0.069, +0.124] | +0.167 [+0.135, +0.198] |
| qwen2.5-3b-instruct | +0.050 [+0.030, +0.072] | +0.135 [+0.103, +0.170] |
| qwen2.5-7b-instruct | +0.056 [+0.037, +0.076] | +0.063 [+0.034, +0.091] |

(Δ share of «так»; the log-odds margin gives the same picture, +0.48 to +3.29.) This one mechanism explains
all three comparisons in §2:
* with a masculine title for both, only the man agrees → the man is favoured (`nofem`);
* with a feminitive for the woman, both agree → the residual favours the woman slightly (`fem`);
* with a feminitive for both, only the woman agrees → the largest gaps (`fem_form`, McNemar 2–7 vs 21–180,
  always for the woman).

The models therefore behave as if the job title's grammatical gender stated which gender the employer wants.
For Ukrainian, a masculine generic job title is the default in job advertisements, so this is a concrete
mechanism by which masculine-generic phrasing disadvantages female candidates. Mismatch is penalized more for men
in 5/6 models (point estimates), but the man-with-feminitive pairing is itself unnatural (diagnostic only).

**I2. Gender flips happen mostly when the candidate is unqualified.** QADiff on items with an irrelevant
experience is 0.03–0.14, vs 0.003–0.10 on items with a relevant one (e.g. Gemma-2-9B, base-form titles: 0.078 vs 0.014;
Qwen2.5-7B 0.080 vs 0.029). The direction of the flips also differs:
* irrelevant experience, base-form title: the unqualified **man** gets «так» more often than the unqualified woman,
  significantly in 5/6 models (net +0.020 to +0.075 of items; Llama-3.2-3B n.s.);
* relevant experience: no significant net direction in any model.

So the bias is mostly **leniency towards unqualified men**, not rejection of qualified women. Both are hiring
harms, but they call for different fixes and the paper should name the first.

**I3. Professions with the strongest effect follow occupational stereotypes** (mean man − woman margin over 6
models, base-form titles; one experience pair per profession, so these are indicative and have no CI):
* man favoured: basketball/football/hockey player, circus performer, mortar gunner, beekeeper, shepherd, director,
  phthisiatrician (the only profession where 5/6 models flip in the man's favour);
* woman favoured: mammologist, gynecologist, manicurist, obstetrician, assistant, accountant clerk («обліковець»).

Across professions the effect is diffuse: 145/349 professions have a net flip in at least one model, the median
has none, and 9 have a net man-favoured flip in ≥ 3 models (basketball, football, hockey player, boxer, circus
performer, mortar gunner, phthisiatrician, oculist, spokesperson); no profession has a woman-favoured flip in ≥ 3. The aggregate QADiff therefore reflects many small item-level effects plus a
stereotype-aligned tail, not a few outlier professions. «байкар» (woman favoured in 2 models) is an exception to the
stereotype pattern and worth checking by annotators.

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

## 4a. Mitigation through the lens of I1–I2

From `scripts/insights.py` → `results/tables/insights_mitigation.csv`: the agreement effect (I1) and directional
flips by relevance (I2) for every method, paired bootstrap difference vs `none` (zero-shot on `full`; ft/ftctrl on
`test`, averaged over 3 seeds, compared with `none` on the same professions).

**M1. No zero-shot method removes the agreement effect.** It stays positive in 36/36 model × method × gender cells;
the mean over models is 0.067 for `none`, 0.063 for hard, 0.065 for soft and 0.073 for prompt. It decreases
significantly in only 3/36 cells (hard: Llama-3.2-3B woman −0.043, Gemma-2-2B man −0.019; prompt: Gemma-2-9B man
−0.034). It increases significantly in 2 (prompt, man: Llama-3.2-3B +0.037, Qwen2.5-7B +0.036), and hard *raises* the
woman's margin effect for both Qwen models (+0.42, +0.45). Hard projection removes the gender subspace from the input
embeddings of the profession, candidate and pronoun tokens, yet the title-gender cue survives. The cue is therefore
carried beyond the 5-direction input-embedding subspace (e.g. by the feminitive suffix tokens once contextualized),
so input-embedding projection works at the wrong level for a morphological cue.

**M2. Only the prompt touches the leniency towards unqualified men**, in 2/6 models: Gemma-2-9B (net +0.055 →
0.000) and Llama-3.1-8B (+0.021 → −0.011). For Gemma-2-9B this coincides with F1 −0.05 and more «так» overall
(§3), so it is not a targeted fix. Hard and soft change it in no model.

**M3. Fine-tuning weakens the agreement effect, and the male-only control does so equally.** In both arms the effect
drops significantly in 5/6 cells (share of «так») and 6/6 (margin): e.g. Llama-3.2-3B man 0.144 → 0.057 (ft) /
0.055 (control); Gemma-2-2B woman 0.040 → −0.008 / 0.009. The control never saw a woman or a feminitive. Learning to
judge the experience makes the model rely less on the form of the title, without any gendered data.

**M4. What fine-tuning removes is mostly the advantage of the woman with a feminitive, not the male advantage.**
* Woman with feminitive vs man with base title: the woman's advantage disappears. Net man-favoured flips rise to
  ≈ 0 (Gemma-2-2B unqualified −0.052 → 0.000, Llama-3.2-3B −0.103 → −0.002; significant), and for Qwen2.5-3B they turn
  positive (+0.021 in both item types).
* Base-form titles, unqualified candidates: the man's advantage is not reduced significantly in any model
  (Qwen2.5-3B: +0.086 → +0.050 / +0.042, n.s.).
* Qwen2.5-3B, qualified candidates: after fine-tuning (both arms), almost all remaining gender flips favour the man
  (net +0.052 / +0.054 vs +0.017 before, significant). QADiff on these items barely changes (0.056).

So a lower QADiff after fine-tuning does not mean fairer outcomes for women. Part of the reduction is the removal of a
pro-female effect, and for one model the residual flips become more male-favouring. QADiff is symmetric and cannot
show this, so mitigation must be reported with directional metrics (net flips / McNemar b vs c, margin_diff).
Caveats: 174 test professions, 3 small models, and the Qwen2.5-3B overall QADiff change is itself not significant.

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
2a. The common mechanism is grammatical agreement: a job title whose gender matches the candidate raises «так» in
   6/6 models for both genders (I1), so masculine-generic titles disadvantage women.
2b. The gender flips concentrate on unqualified candidates, where unqualified men are accepted more often (I2).
   The most affected professions follow occupational stereotypes (I3).
3. Embedding projection (soft/hard) and a debiasing prompt do not reduce gender disagreement reliably. The v1
   18.9% claim is withdrawn.
4. LoRA fine-tuning reduces disagreement on unseen professions, but a male-only control reduces it equally. The
   gain comes from learning the task; a gender-specific effect is visible only in Gemma-2-2B's margin gap.
4a. No zero-shot method removes the agreement cue (M1). Fine-tuning weakens it, through task learning (M3).
   But the QADiff reduction after fine-tuning mainly removes the advantage of women with feminitive titles, while
   the advantage of unqualified men remains (M4). Mitigation must be evaluated with directional metrics.
5. Results depend on the instruction wording at the ±0.04 level, and small models can become degenerate.

## 8. Open caveats

* Native-speaker validation of the data is pending (κ not available). Conclusions about feminitives assume the
  feminitive forms are correct.
* Fine-tuning was run on three small models only (2–3B). The 7–9B models were not fine-tuned.
* R0 showed that v1 numbers move by up to 0.010 from the software environment alone. v2 numbers are pinned to
  model revisions, but the same caveat applies across library versions.
* The provenance of 15/27 v1 QA result files is unknown (AUDIT 3.1), so for those cells the T7 comparison is with
  numbers whose procedure cannot be checked.
