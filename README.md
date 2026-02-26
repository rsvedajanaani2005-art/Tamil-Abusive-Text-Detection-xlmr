# Tamil-Abusive-Text-Detection-xlmr

Binary Classification of Tamil YouTube Comments
(Abusive – அவதூறு vs Non-Abusive – சாதாரணம்) using XLM-RoBERTa

# 1. Overview 

This repository presents a transformer-based approach for detecting abusive Tamil YouTube comments targeting women. The task is formulated as a binary text classification problem and implemented using XLM-RoBERTa-base fine-tuning with the Hugging Face Transformers library and PyTorch backend.

The goal is to automatically classify Tamil comments into:

Abusive (அவதூறு)

Non-Abusive (சாதாரணம்)

The system is evaluated using macro-averaged precision, recall, and F1-score, following the official shared task evaluation protocol.

# 2. Task Description

Online platforms have increasingly become spaces where Tamil-speaking women face derogatory and abusive language. Automated detection of such harmful content is critical for building safer digital environments.

This task focuses on binary classification of Tamil YouTube comments.

Dataset Statistics
Split	Samples
Train	3652
Test	913
Total	4565
Labels

Abusive (அவதூறு) – Comments containing offensive, harmful, or insulting language

Non-Abusive (சாதாரணம்) – Neutral or non-harmful comments

Evaluation metrics:

Macro Precision

Macro Recall

Macro F1-score

Metrics are computed using Scikit-learn’s classification_report.

# 3. Model Architecture

The system fine-tunes XLM-RoBERTa-base, a multilingual transformer model pretrained on large-scale multilingual corpora.

Architecture components:

- Pretrained XLM-RoBERTa encoder

- Classification head (linear layer)

- Softmax output layer for binary prediction

- The model is fine-tuned end-to-end using cross-entropy loss.

Implementation stack:

- PyTorch

- Hugging Face Transformers

- Hugging Face Datasets

- Scikit-learn
