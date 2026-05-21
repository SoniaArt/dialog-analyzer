import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# --------------------------------------------------

print("Загрузка данных тем...")

df = pd.read_csv(
    "training_data_topics.csv"
)

# --------------------------------------------------
# LABELS
# --------------------------------------------------

topics = sorted(
    df['topic'].unique()
)

topic_to_id = {
    t: i
    for i, t in enumerate(topics)
}

id_to_topic = {
    i: t
    for t, i in topic_to_id.items()
}

df['label'] = df['topic'].map(
    topic_to_id
)

print(f"Тем: {len(topics)}")
print(f"Диалогов: {len(df)}")

# --------------------------------------------------
# TRAIN / VAL
# --------------------------------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['dialog'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model_name = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

class TopicDataset(Dataset):

    def __init__(self, texts, labels):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):

        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

    def __len__(self):

        return len(self.labels)

# --------------------------------------------------

train_dataset = TopicDataset(
    train_texts,
    train_labels
)

val_dataset = TopicDataset(
    val_texts,
    val_labels
)

# --------------------------------------------------

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_name,
        num_labels=len(topics),
        ignore_mismatched_sizes=True
    )
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    return {

        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "f1": f1_score(
            labels,
            predictions,
            average="weighted"
        )
    }

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir="./topic_model_results",

    num_train_epochs=5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_steps=100,

    logging_steps=20,

    report_to="none"
)

# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# --------------------------------------------------

print("Обучение topic model...")

trainer.train()

# --------------------------------------------------

metrics = trainer.evaluate()

print("\nРезультаты:")
print(metrics)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

model.save_pretrained(
    "./topic_model"
)

tokenizer.save_pretrained(
    "./topic_model"
)

with open(
    "./topic_model/topics_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "topic_to_id": topic_to_id,
            "id_to_topic": id_to_topic
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nTopic model сохранена!")