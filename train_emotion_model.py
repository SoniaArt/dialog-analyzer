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

print("Загрузка данных эмоций...")

df = pd.read_csv(
    "training_data_emotions.csv"
)

# --------------------------------------------------
# LABELS
# --------------------------------------------------

emotions = sorted(
    df['emotion'].unique()
)

emotion_to_id = {
    e: i
    for i, e in enumerate(emotions)
}

id_to_emotion = {
    i: e
    for e, i in emotion_to_id.items()
}

df['label'] = df['emotion'].map(
    emotion_to_id
)

print(f"Эмоции: {emotions}")
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

class EmotionDataset(Dataset):

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

train_dataset = EmotionDataset(
    train_texts,
    train_labels
)

val_dataset = EmotionDataset(
    val_texts,
    val_labels
)

# --------------------------------------------------

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_name,
        num_labels=len(emotions),
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

    output_dir="./emotion_model_results",

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

print("Обучение emotion model...")

trainer.train()

# --------------------------------------------------

metrics = trainer.evaluate()

print("\nРезультаты:")
print(metrics)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

model.save_pretrained(
    "./emotion_model"
)

tokenizer.save_pretrained(
    "./emotion_model"
)

with open(
    "./emotion_model/emotion_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "emotion_to_id": emotion_to_id,
            "id_to_emotion": id_to_emotion
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nEmotion model сохранена!")