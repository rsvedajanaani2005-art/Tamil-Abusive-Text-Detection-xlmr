# Tamil Abusive Text Detection using XLM-RoBERTa

Binary Classification of Tamil YouTube Comments  (Abusive – அவதூறு vs Non-Abusive – சாதாரணம்)

This work is done as part of the task **Abusive Tamil Text Targeting Women on Social Media - DravidianLangTech@ACL2026**

---

## Overview

This repository presents a transformer-based approach for detecting abusive Tamil YouTube comments targeting women. The task is formulated as a binary text classification problem and implemented using **XLM-RoBERTa-base** fine-tuning with the Hugging Face Transformers library and PyTorch.

The system classifies Tamil comments into:

- **Abusive (அவதூறு)**
- **Non-Abusive (சாதாரணம்)**

---

## Dataset

The dataset contains annotated Tamil YouTube comments.

| Split | Samples |
|--------|---------|
| Train | 3652 |
| Test  | 913 |
| Total | 4565 |

The training set is split into 90% training and 10% validation using stratified sampling.

### Columns

- `Text` – Tamil comment  
- `Class` – Label (0 = Non-Abusive, 1 = Abusive)

---

## Model

The model used is **xlm-roberta-base**, a multilingual transformer pretrained on large-scale multilingual corpora.

### Training Configuration

- Max sequence length: 128  
- Learning rate: 2e-5  
- Batch size: 16  
- Epochs: 3  
- Weight decay: 0.01  
- Optimizer: AdamW (via Hugging Face Trainer)  
- Evaluation metric: Macro Precision, Macro Recall, Macro F1  

---

## Notes

- Dataset files are not included due to distribution restrictions.
- GPU is recommended for faster training.
- The best model checkpoint is automatically selected based on validation Macro-F1.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Tamil-Abusive-Text-Detection-xlmr.git
cd Tamil-Abusive-Text-Detection-xlmr

