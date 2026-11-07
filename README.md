# GBEM-UA: Gender Bias Evaluation and Mitigation for Ukrainian Large Language Models

This repository contatains the source code of the paper "GBEM-UA: Gender Bias Evaluation and Mitigation for Ukrainian Large Language Models".

## Overview

This project studies gender bias in large language models (LLMs) within the Ukrainian language, focusing on the use of femininitives and the hiring domain. We propose a benchmark dataset and evaluate multiple debiasing strategies, including:

- Prompt-based debiasing
- Embedding debiasing
- Fine-tuning with LoRA

We introduce both Q&A-based and probability-based evaluation metrics to quantify bias and model performance.

## Repository Structure

```bash
FairLMs/
├── configs/       # Configuration files for running experiments
├── data/          # v2 dataset (data/v2: 349 professions × 8 variants, profession-level split); v1 files kept for reference
├── debias/        # Prompt and embedding-projection debiasing (wrapper.py), LoRA fine-tuning (finetune.py)
├── metrics/       # Evaluation metrics (QAAccMetric, ProbDiffMetric, etc.)
├── parser/        # Config parser
├── scripts/       # Dataset build, statistics/reports, GPU run orchestration
├── reports/       # Audit of v1, v2 dataset/method changes, results
├── tests/         # Unit tests (CPU)
├── results/       # Saved resulting metrics 
├── utils.py       # Utility functions
├── run_eval.py    # Runs one model × method × metric configuration
├── main.py        # Config-driven entry point
├── requirements.txt
└── README.md
```

## Dataset
The dataset includes:

Two datasets for accuracy and probability measurment

349 Ukrainian professions (the paper says 351; see reports/AUDIT_v1.md)

8 variations per profession:

- Male / Female

- Femininitive / Non-femininitive

- Relevant / Irrelevant experience

📁 See `data/` for full samples and construction logic.

## Evaluation Metrics

Implemented in `metrics/`:

- `QAAccMetric`: QA accuracy (F1) per gender/form

- `QADiffMetric`: Prediction consistency across gendered variants

- `ProbAccMetric`: Classification accuracy via probability

- `ProbDiffMetric`: Distributional bias via likelihood scores

Results are logged in `results/`.

## Changes since the UNLP 2025 version

The published version is preserved at tag `v1.0-unlp2025`. Version 2 fixes problems found after publication
(details: [`reports/AUDIT_v1.md`](reports/AUDIT_v1.md), [`reports/CHANGES_v2.md`](reports/CHANGES_v2.md)):

* **Data:** the v1 test set shared 1,368 of its 1,400 positive items with train. v2 splits by profession
  (175 / 174) and keeps one block of 8 sentences per profession (`data/v2`, `scripts/build_v2.py`).
* **Debiasing:** the projection reflected the first gender component instead of removing it; fixed and tested.
* **QA metric:** answers are scored from token probabilities under the chat template instead of embedding similarity
  of a 5-token generation; every answer is saved.
* **Prob metric:** proper beginning-of-text conditioning, fixed scoring for the prompt method, and a real ProbAcc.
  v1 Table 6 ("Acc Prob Metric") is man/woman agreement.
* **Fine-tuning** code and data are now in the repository (`debias/finetune.py`).
* **Statistics:** cluster-bootstrap CIs over professions, paired differences, exact McNemar with Holm correction.

## Running

```bash
python scripts/build_v2.py                     # dataset v2 + checks
python -m pytest tests                         # CPU unit tests
scripts/run_all.sh all                         # GPU: R0, pilot, zero-shot, fine-tuning, stats (see reports/RUNBOOK.md)
```

## Examples

Example config files are in `configs/`. You can create your own by editing parameters for:

- Model name

- Debiasing method

- Dataset splits

- Output directories


## Citation

```bibtex
@inproceedings{buleshnyi-etal-2025-gbem,
    title = "{GBEM}-{UA}: Gender Bias Evaluation and Mitigation for {U}krainian Large Language Models",
    author = "Buleshnyi, Mykhailo  and
      Buleshnyi, Maksym  and
      Sumyk, Marta  and
      Drushchak, Nazarii",
    editor = "Romanyshyn, Mariana",
    booktitle = "Proceedings of the Fourth Ukrainian Natural Language Processing Workshop (UNLP 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.unlp-1.8/",
    doi = "10.18653/v1/2025.unlp-1.8",
    pages = "64--72",
    ISBN = "979-8-89176-269-5",
    abstract = "Large Language Models (LLMs) have demonstrated remarkable performance across various domains, but they often inherit biases present in the data they are trained on, leading to unfair or unreliable outcomes{---}particularly in sensitive areas such as hiring, medical decision-making, and education. This paper evaluates gender bias in LLMs within the Ukrainian language context, where the gendered nature of the language and the use of feminitives introduce additional complexity to bias analysis. We propose a benchmark for measuring bias in Ukrainian and assess several debiasing methods, including prompt debiasing, embedding debiasing, and fine-tuning, to evaluate their effectiveness. Our results suggest that embedding debiasing alone is insufficient for a morphologically rich language like Ukrainian, whereas fine-tuning proves more effective in mitigating bias for domain-specific tasks."
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

