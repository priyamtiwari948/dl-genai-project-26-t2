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

## Models Built

In this project I developed three unique models, trained, and evaluated as part of this project — meeting the requirement of one from-scratch model, one pretrained model, and one additional model of choice.

| # | Model | Type | Kaggle MAP@3 Score |
|---|-------|------|---------------------|
| 1 | BiLSTM + Attention (from scratch) | Custom architecture, no pretrained weights | 0.71238 |
| 2 | Logistic Regression | Classical ML baseline | 0.73815 |
| 3 | ELECTRA-base-discriminator | Pretrained transformer (fine-tuned) | 0.75519 |

### 1. From-Scratch Model — BiLSTM + Attention
- Custom vocabulary built directly from training data (no external tokenizer)
- Word embeddings trained from random initialization — no pretrained weights anywhere
- Shared bidirectional LSTM encodes both the question prompt and each answer option
- A custom attention mechanism lets each option attend over the most relevant part of the prompt
- Match features (concat, |diff|, product) feed into an MLP scorer that ranks all 5 options
- Trained end-to-end with cross-entropy loss; evaluated using MAP@3

### 2. Logistic Regression (Classical ML)
- TF-IDF / feature-based representation of prompt–option pairs
- Fast, interpretable baseline for comparison against deep learning approaches

### 3. ELECTRA-base-discriminator (Pretrained Transformer)
- Fine-tuned `google/electra-base-discriminator` on the competition dataset
- Best results achieved training on a T4 GPU (Tesla P100 had compatibility issues)
- Highest-scoring model overall, demonstrating the strength of transfer learning for this task

---

## Experiment Tracking (Weights & Biases)

All three models were tracked using W&B, logging training/validation loss, accuracy, F1 score, and MAP@3 per epoch — enabling direct comparison across architectures.

- Project: `smart-mcq-solver`
- Runs compared: from-scratch BiLSTM, Logistic Regression, ELECTRA fine-tuning

---

## Deployment

The from-scratch model is deployed as an interactive web app using Gradio on Hugging Face Spaces (ZeroGPU hardware).

Try it live: [huggingface.co/spaces/priyamtiwari948/smart-mcq-solver](https://huggingface.co/spaces/priyamtiwari948/smart-mcq-solver)

---

## Tech Stack

- Python, PyTorch — from-scratch model implementation
- scikit-learn — Logistic Regression baseline
- Hugging Face Transformers — ELECTRA fine-tuning
- Weights & Biases — experiment tracking
- Gradio + Hugging Face Spaces — deployment
- Kaggle — training compute (T4 / T4x2 GPU) and competition submissions

---

## Evaluation Metric

Mean Average Precision @ 3 (MAP@3) — rewards models for ranking the correct answer as high as possible among the top 3 predictions.

---

## Author

**Priyam Tiwari**
BS in Data Science and Applications, IIT Madras
