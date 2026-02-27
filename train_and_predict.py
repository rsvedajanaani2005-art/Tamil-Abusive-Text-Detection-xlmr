import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load Training Data

df = pd.read_csv("data/train.csv")
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["Class"],
    random_state=42
)
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))


# Tokenization

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(batch):
    return tokenizer(
        batch["Text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "Class"]
)
val_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "Class"]
)
train_ds = train_ds.rename_column("Class", "labels")
val_ds = val_ds.rename_column("Class", "labels")

# Model

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Training

trainer.train()

# Validation Evaluation

preds = trainer.predict(val_ds)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(
    y_true,
    y_pred,
    target_names=["Non-Abusive", "Abusive"]
))

# Test Prediction

test_df = pd.read_csv("data/test.csv")
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

def tokenize_test(batch):
    return tokenizer(
        batch["Text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

test_ds = test_ds.map(tokenize_test, batched=True)

test_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)
test_predictions = trainer.predict(test_ds)
pred_labels = np.argmax(test_predictions.predictions, axis=1)
test_df["Class"] = pred_labels
test_df[["Class"]].to_csv("results/predictions.csv", index=False)
