# FairLMs

This repository contatains the source code of the paper {name}.

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
├── results/       # Saved outputs, logs, and results
├── utils.py       # Utility functions
├── main.py        # Main script for running experiments
├── requirements.txt
└── README.md
```
