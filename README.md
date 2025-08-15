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
├── data/          # Synthetic hiring dataset (351 professions × 8 variants)
├── debias/        # Implementations of prompt, embedding, and fine-tuning methods
├── metrics/       # Evaluation metrics (QAAccMetric, ProbDiffMetric, etc.)
├── parser/        # Wrapper for running debiasing methods
├── results/       # Saved resulting metrics 
├── utils.py       # Utility functions
├── main.py        # Main script for running experiments
├── requirements.txt
└── README.md
```

## Dataset
The dataset includes:

Two datasets for accuracy and probability measurment

351 Ukrainian professions

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

## Examples

Example config files are in `configs/`. You can create your own by editing parameters for:

- Model name

- Debiasing method

- Dataset splits

- Output directories


## Citation
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
## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

