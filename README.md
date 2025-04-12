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
├── data/                # Synthetic hiring dataset (351 professions x 8 variations)
├── models/              # Fine-tuning scripts and LoRA configs
├── evaluation/          # QA and probabilistic evaluation metrics
├── debiasing/           # Prompting, embedding debiasing implementations
├── results/             # Saved results (CSV/JSON)
├── utils/               # Helper functions
├── requirements.txt     # Python dependencies
└── main.py              # Entry point for experiments
```
