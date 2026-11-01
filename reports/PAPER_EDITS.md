# Paper revision: text that can be written now

Drafts for the revised UNLP paper and the thesis chapter. Numbers that depend on the GPU runs are marked
**[R]**; take them from `reports/RESULTS_v2.md`. Conclusions (Sec. 5, 7, abstract) are rewritten only after that.

## Sec. 3 Dataset (replaces "351 professions", "manually reviewed and verified", Male+Fem "for completeness")

> GBEM-UA covers 349 Ukrainian professions. For each profession, GPT-4 generated one relevant and one irrelevant
> experience. Each experience is combined with a male or female candidate («Кандидат … він» / «Кандидатка … вона»)
> and with the base or feminitive job title, which gives 8 sentences per profession (2,792 in total) in each of
> the two formats (question for QA, statement for the probability metric). The profession «рибалка» has no
> distinct feminitive and is excluded from all comparisons involving feminitives. Sentences pairing a male candidate
> with a feminitive title are grammatically unnatural; we release them but they enter no bias metric
> (we report them separately as a diagnostic).
>
> We split the data **by profession** (175 train / 174 test, seed 42), so no profession, experience or sentence
> occurs in both splits. Zero-shot methods are evaluated on all 349 professions (`full`); fine-tuned models are
> evaluated on the 174 test professions they never saw. *[After annotation:]* Two native speakers independently
> checked every profession for (a) relevance of the relevant experience, (b) irrelevance of the irrelevant one,
> (c) correctness of the feminitive (gendergid.org.ua) and (d) grammaticality; Cohen's κ was **[R]** / **[R]** /
> **[R]** / **[R]**; **[R]** professions were removed.

If the annotation is not finished before submission, replace the last two sentences with: "The data has not yet
been validated by native speakers."

## Sec. 4.1.1 QA metric (replaces the e5 cosine-similarity answer extraction)

> Each sentence is presented as a single user turn in the model's chat template: «Дай відповідь одним словом: так
> або ні.» followed by the sentence. We score the answer from the model's next-token distribution: the log-probability
> of «так» is the log-sum-exp of the full-sequence log-probabilities of the variants «так», «Так», « так», « Так»
> (likewise for «ні»). We score whole variants, not first tokens, because some tokenizers split «Так»/«Ні» into a
> letter and a suffix. The prediction is «так» if its log-probability is higher. As a control, we also generate up
> to 16 tokens greedily and parse the first word. We report the share of generations that are neither «так» nor «ні»
> and their agreement with the scored answer. Per group, we report F1, balanced accuracy and the share of «так»
> answers. A configuration that answers «так» (or «ні») to more than 95% of the items of any group is flagged as
> degenerate: its QADiff is close to 0 by construction.
>
> QADiff is the share of items whose answer differs between the male and the female variant (feminitive vs base
> title for the female candidate). An exact McNemar test on the discordant pairs (b = only the male candidate gets
> «так», c = only the female one) shows whether the disagreements systematically favour one gender.

Add after QADiff:

> Because QADiff counts only decisions that flip, it is close to 0 for a model that gives the same answer to almost
> everything. We therefore also report a threshold-free measure: the mean paired difference, male minus female
> variant, of the log-odds margin log P(«так») − log P(«ні»). Positive values mean the male candidate is pushed
> towards «так».

## Sec. 4.1.2 Probability metrics (and the Table 6 caption)

> The sentence probability is the per-token geometric mean of the model's probabilities. The sentence is
> conditioned on the model's beginning-of-text token, and only sentence tokens are scored.
> ProbDiff = mean |P_m − P_f| over positive sentences + mean |P_m − P_f| over negated sentences
> (the shorter sentence of a pair is padded with end-of-text tokens; results without padding are in the appendix).
> ProbAcc is the F1 of the decision P(positive) > P(negated) against the label. ProbAgree is the share of items
> where the male and female variants receive the same decision.

**Table 6 caption:** "Acc Prob Metric" → **"ProbAgree: agreement between male and female decisions"**.
Add ProbAcc as a separate column. v1's Table 6 values are ProbAgree, not accuracy.

## Sec. 4.2.2 Embedding debiasing (replaces "hard debiasing")

> We estimate a gender subspace from the first five principal directions of the difference vectors of 37
> gender-defining word pairs (input embeddings, special tokens excluded) and remove the projection of the
> profession tokens onto it (**proj-prof**). The variant **proj-ext** additionally projects the tokens of the
> candidate noun and the pronoun. Both are the projection ("neutralize") step only. Neither includes the equalize
> step of Bolukbasi et al. (2016), so neither is their hard debiasing.

*(If the old names are kept for continuity: "soft" = proj-prof, "hard" = proj-ext, with the sentence above.)*
Erratum note: the v1 implementation reflected the first principal component instead of removing it.

## Sec. 4.2.3 Fine-tuning (replaces "175 professions … gender-neutral and relevant combinations")

> We fine-tune LoRA adapters (r = 16, α = 8, dropout 0.1, q/k/v projections, 3 epochs, learning rate 2.5·10⁻⁴,
> effective batch 16, paged 8-bit AdamW, bf16) on the 175 training professions, holding out 18 of them for
> checkpoint selection by validation loss. Training examples use the evaluation prompt, include both classes
> (relevant → «Так», irrelevant → «Ні») and all three evaluated gender variants with the same label (942 training
> examples). The loss is computed on the answer tokens only. We train three seeds (13, 42, 2025) per model and
> report mean ± sd. We evaluate on the 174 unseen professions; results on the training professions measure
> memorization (Appendix). As accuracy approaches 1, QADiff necessarily approaches 0. To separate debiasing from
> learning the task, we train a control adapter on the same professions, answers and number of steps, but with male
> base-form sentences only. Fine-tuning counts as debiasing only where it reduces QADiff more than this control.

## Statistics paragraph (new, end of Sec. 4)

> All metrics are reported with 95% confidence intervals from a cluster bootstrap over professions (2,000
> resamples). The 8 sentences of a profession are not independent, so we resample professions rather than
> sentences. Differences between a debiasing method and the undebiased model use paired resamples. We call a
> method effective only if the CI of its difference excludes zero. McNemar p-values are Holm-corrected.

## Limitations (add)

* One instruction for the main results; robustness to two paraphrases and to swapping «так»/«ні» in the
  instruction is reported in Table **[R]** (T8). Wording effects beyond these are untested.
* Experiences were generated by GPT-4 and may carry its associations; one relevant and one irrelevant experience per profession.
* Binary gender and a binary decision; the setting does not cover non-binary candidates or graded suitability.
* Instruction-tuned models only; no base models.

## Introduction and related work

Unsupported statements → sources (or soften):

* AI in hiring as high-risk → Regulation (EU) 2024/1689 (AI Act), Annex III, point 4 (employment, workers management).
* Under-representation of feminitives → corpus counts (e.g. frequencies in UberText 2.0; Chaplynskyi, UNLP 2023) or soften.
* "Largely untested in other languages" → cite multilingual bias work (below) or soften.

Benchmarks to add: WinoBias (Zhao et al., NAACL 2018), WinoGender (Rudinger et al., NAACL 2018),
CrowS-Pairs (Nangia et al., EMNLP 2020), BOLD (Dhamala et al., FAccT 2021), BBQ (Parrish et al., Findings of ACL
2022), French CrowS-Pairs (Névéol et al., ACL 2022, as an example of adapting a benchmark to another language),
JobFair (Wang et al., 2024; closest in domain; *verify venue*). Zmigrod et al. (ACL 2019) is especially relevant:
counterfactual data augmentation for morphologically rich languages. Describe it as **mitigation**, not measurement.

Reference fixes:

| as cited | fix |
|---|---|
| Zafar et al. 2004 | Zafar, Valera, Gomez-Rodriguez, Gummadi. *Fairness Constraints: Mechanisms for Fair Classification.* AISTATS 2017 (arXiv 2015) |
| Nazi & Peng 2023 | 2024 (*Large Language Models in Healthcare and Medical Domain: A Review*, Informatics; *verify volume/issue*) |
| Dalvi et al. 2004 | Dalvi, Domingos, Mausam, Sanghai, Verma. *Adversarial Classification.* KDD 2004: about adversarial learning. Fix authors and reconsider citing it as fairness-in-training |
| Yadav et al. 2023 "CDA: Contrastive-adversarial domain adaptation" cited for relabeling | replace: Kamiran & Calders, *Data preprocessing techniques for classification without discrimination*, KAIS 33(1), 2012 ("massaging" = relabeling); for counterfactual augmentation: Zmigrod et al. 2019 |
| Zmigrod et al. 2019 | describe as a mitigation method |
| Bolukbasi et al. | NeurIPS (NIPS) 2016 |
| Caliskan et al. | Science 356(6334):183–186, 2017 |
| LoRA (Hu et al.) | ICLR 2022 |
| Kiulian et al., Guo et al. | check author order against the original papers; add venues |

## Conclusions to re-check after the runs (do not edit before)

| claim | evidence in RESULTS_v2.md |
|---|---|
| "consistently larger" QADiff with feminitives (p. 67) | F3 / T3 `gap_fem_minus_nofem` |
| "average relative improvement of 18.9%" | replace with ΔQADiff + CI (F1); never average relative changes (zeros in the denominator) |
| "disparity … decreased substantially" after fine-tuning | T5 (unseen professions), T6, and `ft_vs_ftctrl` (vs control) |
| "zero difference … Qwen2.5-3B" | degenerate check (F2) + `margin_diff` (T3); remove or explain via yes_rate |
| "cost of lower QA accuracy" | ΔF1 CIs (T2) |
| abstract, Sec. 7 Discussion | align with the above |

Draft revision note for ACL Anthology (from the plan, unchanged):

> Revised version. We rebuilt the train/test split so that no profession appears in both splits (the original split
> shared 1,368 positive test items with train), fixed a projection error in the embedding-debiasing code, replaced the
> similarity-based answer extraction with token-probability scoring, corrected the label of Table 6, added bootstrap
> confidence intervals, and updated the results and conclusions accordingly. Code and data: v2.0.

Consider adding: "…and fixed the probability metric for the prompt-debiasing condition."
