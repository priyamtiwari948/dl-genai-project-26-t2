# Smart MCQ Solver Challenge

This is a Deep Learning & Generative AI Project.

An end-to-end machine learning system that solves multiple-choice questions by ranking the top-3 most probable answers (out of 5 options), evaluated using Mean Average Precision @ 3 (MAP@3).

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/priyamtiwari948/smart-mcq-solver)
🔗 **Competition:** [Smart MCQ Solver Challenge on Kaggle](https://www.kaggle.com/competitions/smart-mcq-solver-challenge)

---

## Problem Statement

Each question consists of a prompt and five options (A–E). The task is to predict the top 3 most likely correct answers in ranked order. Models are scored higher when the correct answer appears earlier in the prediction list.

```
Example:
Correct answer: A
Prediction "A B C" → highest score
Prediction "B A C" → lower score
Prediction "C D A" → lowest score
```

---

## Repository Structure

```
├── Kaggle_notebook.ipynb              # Complete end-to-end notebook (EDA → baseline → all 3 models → comparison)
├── EDA.ipynb                          # Standalone exploratory data analysis
├── baseline model.ipynb               # Baseline: TF-IDF + Cosine Similarity (no training)
├── Logistic Regression.ipynb          # Model 1: TF-IDF + Logistic Regression
├── Scratch model.ipynb                # Model 2: BiLSTM + Attention (built from scratch)
├── ELECTRA Base Discriminator.ipynb   # Model 3: Fine-tuned ELECTRA-base-discriminator
└── README.md
```

---

## Models Built

I developed, trained, and evaluated three unique models — meeting the requirement of one from-scratch model, one pretrained model, and one additional model of choice.

| # | Model | Type | Kaggle MAP@3 Score |
|---|-------|------|---------------------|
| 1 | BiLSTM + Attention (from scratch) | Custom architecture, no pretrained weights | 0.71238 |
| 2 | Logistic Regression | Classical ML baseline | 0.73815 |
| 3 | ELECTRA-base-discriminator | Pretrained transformer (fine-tuned) | 0.75519 |

A simple **TF-IDF + Cosine Similarity baseline** (no training) was also built for reference, scoring 0.3119 on training data — confirming that all three trained models learn genuine patterns rather than relying on simple keyword overlap.

### 1. From-Scratch Model — BiLSTM + Attention
- Custom vocabulary built directly from training data (no external tokenizer)
- Word embeddings trained from random initialization — no pretrained weights anywhere
- Shared bidirectional LSTM encodes both the question prompt and each answer option
- A custom attention mechanism lets each option attend over the most relevant part of the prompt
- Match features (concat, |diff|, product) feed into an MLP scorer that ranks all 5 options
- Trained end-to-end with cross-entropy loss; evaluated using MAP@3

### 2. Logistic Regression (Classical ML)
- TF-IDF feature representation (unigrams to trigrams) of combined prompt + options text
- Fast, interpretable baseline for comparison against deep learning approaches
- Crosses the competition's qualifying MAP@3 cutoff (0.73)

### 3. ELECTRA-base-discriminator (Pretrained Transformer)
- Fine-tuned `google/electra-base-discriminator` using `AutoModelForMultipleChoice`
- Trained on a Kaggle T4 GPU
- Highest-scoring model overall, demonstrating the strength of transfer learning for this task

---

## Exploratory Data Analysis

Key checks performed before modeling (see `EDA.ipynb`):
- No missing values found in the dataset
- Duplicate prompts checked for label consistency
- Correct-answer class distribution is mildly imbalanced (16%–24.5% across A–E), not severe
- Prompt length (~17–19 words average) shows no correlation with the correct answer, ruling out a length-based shortcut
- Vocabulary size calculated to inform embedding layer sizing for the from-scratch model

---

## Experiment Tracking (Weights & Biases)

All three models were tracked using W&B, logging training/validation loss, accuracy, F1 score, and MAP@3 per epoch — enabling direct comparison across architectures.

- Project: `smart-mcq-solver`
- Runs compared: from-scratch BiLSTM, Logistic Regression, ELECTRA fine-tuning

---

## Deployment

The from-scratch model is deployed as an interactive web app using Gradio on Hugging Face Spaces (ZeroGPU hardware).

Try it live: [huggingface.co/spaces/priyamtiwari948/smart-mcq-solver](https://huggingface.co/spaces/priyamtiwari948/smart-mcq-solver)

Users can input any question and 5 options and get real-time top-3 ranked predictions with confidence scores.

---

## Tech Stack

- Python, PyTorch — from-scratch model implementation
- scikit-learn — Logistic Regression baseline
- Hugging Face Transformers — ELECTRA fine-tuning
- Weights & Biases — experiment tracking
- Gradio + Hugging Face Spaces — deployment
- Kaggle — training compute (T4 GPU) and competition submissions

---

## Evaluation Metric

Mean Average Precision @ 3 (MAP@3) — rewards models for ranking the correct answer as high as possible among the top 3 predictions.

---

## Author

**Priyam Tiwari**
BS in Data Science and Applications, IIT Madras
