import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DialogAnalyzer:

    def __init__(self):

        self.use_topic_model = False
        self.use_emotion_model = False

        # модель тем

        if os.path.exists("./topic_model"):

            self.topic_model = (
                AutoModelForSequenceClassification
                .from_pretrained("./topic_model")
            )

            self.topic_tokenizer = (
                AutoTokenizer
                .from_pretrained("./topic_model")
            )

            self.topic_model.eval()

            with open(
                "./topic_model/topics_mapping.json",
                "r",
                encoding="utf-8"
            ) as f:

                mapping = json.load(f)

                self.id_to_topic = {
                    int(k): v
                    for k, v in mapping["id_to_topic"].items()
                }

            self.use_topic_model = True

            print("Модель тем загружена")

        # модель эмоций

        if os.path.exists("./emotion_model"):

            self.emotion_model = (
                AutoModelForSequenceClassification
                .from_pretrained("./emotion_model")
            )

            self.emotion_tokenizer = (
                AutoTokenizer
                .from_pretrained("./emotion_model")
            )

            self.emotion_model.eval()

            with open(
                "./emotion_model/emotion_mapping.json",
                "r",
                encoding="utf-8"
            ) as f:

                mapping = json.load(f)

                self.id_to_emotion = {
                    int(k): v
                    for k, v in mapping["id_to_emotion"].items()
                }

            self.use_emotion_model = True

            print("Модель эмоций загружена")

    def preprocess_text(self, dialog):

        return dialog.strip()[:512]

    def classify_topic(self, dialog):

        text = self.preprocess_text(dialog)

        if not self.use_topic_model:
            return "другое"

        inputs = self.topic_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.topic_model(**inputs)

            pred = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

        topic = self.id_to_topic.get(
            pred,
            "другое"
        )

        return topic

    def analyze_sentiment(self, dialog):

        text = self.preprocess_text(dialog)

        if not self.use_emotion_model:
            return "нейтральный"

        inputs = self.emotion_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.emotion_model(**inputs)

            pred = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

        emotion = self.id_to_emotion.get(
            pred,
            "нейтральный"
        )

        #правила постобработки

        lower = text.lower()

        strong_negative = [
            "недопустимо",
            "разочарован",
            "верните деньги",
            "грубый",
            "хамство",
            "ужас",
            "кошмар"
        ]

        strong_positive = [
            "5 звёзд",
            "отлично",
            "буду заказывать",
            "отличное качество",
            "спасибо за быструю доставку",
            "доставка вовремя"
        ]

        problem_words = [
            "не пришёл",
            "сломался",
            "брак",
            "повреждённый",
            "не соответствует",
            "не могу",
            "ошибка",
            "опоздал",
            "списали"
        ]

        #негатив
        if any(w in lower for w in strong_negative):
            return "негативный"

        #позитив
        if any(w in lower for w in strong_positive):
            return "позитивный"

        #проблема + спасибо = нейтрально
        if any(w in lower for w in problem_words):

            if "спасибо" in lower:
                return "нейтральный"

            if "хорошо" in lower:
                return "нейтральный"

            if "нашёл" in lower:
                return "нейтральный"

        return emotion

    def is_problem_dialog(self, dialog):

        lower = dialog.lower()

        problem_words = [
            "не пришёл",
            "сломался",
            "брак",
            "повреждённый",
            "не работает",
            "не могу",
            "ошибка",
            "опоздал",
            "верните деньги",
            "грубый",
            "хамство",
            "списали"
        ]

        return any(
            word in lower
            for word in problem_words
        )

    def analyze_dialog(self, dialog):

        emotion = self.analyze_sentiment(dialog)

        return {
            "topic": self.classify_topic(dialog),
            "emotion": emotion,
            "is_problem": self.is_problem_dialog(dialog)
        }