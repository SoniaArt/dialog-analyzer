import re
import json
import os

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception as exc:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    print(f"ML-модели недоступны, будет использован быстрый режим правил: {exc}")

class DialogAnalyzer:
    def __init__(self):
        self.use_topic_model = False
        self.use_emotion_model = False
        self.use_ml_models = os.getenv("USE_ML_MODELS", "0") == "1"
        self.problem_keywords = {
            "critical": [
                "мошенничество", "обман", "угроза", "суд", "полиция", "жалоба",
                "претензия", "не вернули деньги", "не возвращают деньги", "списали деньги",
                "двойное списание", "заблокировали", "не работает совсем"
            ],
            "high": [
                "брак", "сломался", "сломана", "не работает", "задержка", "опоздал",
                "не доставили", "не пришел", "не пришёл", "возврат", "ошибка",
                "отменили", "потеряли", "поврежден", "повреждён", "не отвечает",
                "долго", "недоволен", "недовольна", "ужасно", "плохо"
            ],
            "medium": [
                "почему", "когда", "сколько ждать", "не могу", "не получается",
                "проблема", "вопрос", "помогите", "разберитесь", "поддержка"
            ],
        }
        self.deescalation_keywords = [
            "спасибо", "решено", "помогли", "вопрос закрыт", "получилось", "разобрались"
        ]
        self.topic_rules = {
            "доставка": ["доставка", "курьер", "заказ", "не пришел", "не пришёл", "опоздал", "привез", "потеряли"],
            "возврат": ["возврат", "верните", "вернуть", "не вернули деньги", "деньги"],
            "качество": ["брак", "сломался", "сломана", "дефект", "поврежден", "повреждён", "не работает"],
            "оплата": ["оплата", "платеж", "платёж", "списали", "карта", "чек"],
            "аккаунт": ["аккаунт", "пароль", "войти", "логин", "заблокировали"],
            "сервис": ["оператор", "поддержка", "жалоба", "претензия", "хамил", "не отвечает"],
        }
        self.negative_words = [
            "плохо", "ужасно", "недоволен", "недовольна", "проблема", "не работает",
            "брак", "задержка", "опоздал", "не пришел", "не пришёл", "обман", "жалоба",
            "верните", "сломался", "сломана", "ошибка"
        ]
        self.positive_words = ["спасибо", "отлично", "хорошо", "помогли", "решено", "получилось"]

        if self.use_ml_models and torch is not None and AutoModelForSequenceClassification is not None and os.path.exists("./topic_model"):
            try:
                self.topic_model = AutoModelForSequenceClassification.from_pretrained("./topic_model")
                self.topic_tokenizer = AutoTokenizer.from_pretrained("./topic_model")
                self.topic_model.eval()
                with open("./topic_model/topics_mapping.json", "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                    self.id_to_topic = {int(k): v for k, v in mapping["id_to_topic"].items()}
                self.use_topic_model = True
                print("Модель тем загружена")
            except Exception as exc:
                print(f"Модель тем не загружена, используется быстрый режим правил: {exc}")

        if self.use_ml_models and torch is not None and AutoModelForSequenceClassification is not None and os.path.exists("./emotion_model"):
            try:
                self.emotion_model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
                self.emotion_tokenizer = AutoTokenizer.from_pretrained("./emotion_model")
                self.emotion_model.eval()
                with open("./emotion_model/emotion_mapping.json", "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                    self.id_to_emotion = {int(k): v for k, v in mapping["id_to_emotion"].items()}
                self.use_emotion_model = True
                print("Модель эмоций загружена")
            except Exception as exc:
                print(f"Модель эмоций не загружена, используется быстрый режим правил: {exc}")

    def get_client_text(self, dialog):
        match = re.search(r'Клиент:\s*(.*?)(?=Оператор:|$)', dialog, re.DOTALL)
        return (match.group(1).strip() if match else dialog)[:500]

    def classify_topic(self, text):
        client_text = self.get_client_text(text)
        if not self.use_topic_model or not client_text:
            return self.classify_topic_by_rules(client_text)
        inputs = self.topic_tokenizer(client_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.topic_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return self.id_to_topic.get(pred, "другое")

    def classify_topic_by_rules(self, text):
        text = str(text).lower()
        scores = {
            topic: sum(1 for keyword in keywords if keyword in text)
            for topic, keywords in self.topic_rules.items()
        }
        best_topic, best_score = max(scores.items(), key=lambda item: item[1])
        return best_topic if best_score > 0 else "другое"

    def analyze_sentiment(self, dialog):
        client_text = self.get_client_text(dialog)
        if not self.use_emotion_model or not client_text:
            return self.analyze_sentiment_by_rules(client_text)
        inputs = self.emotion_tokenizer(client_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return self.id_to_emotion.get(pred, "нейтральный")

    def analyze_sentiment_by_rules(self, text):
        text = str(text).lower()
        negative_score = sum(1 for word in self.negative_words if word in text)
        positive_score = sum(1 for word in self.positive_words if word in text)
        if negative_score > positive_score:
            return "негативный"
        if positive_score > negative_score:
            return "позитивный"
        return "нейтральный"
    
    def analyze_problem_scenario(self, dialog, topic=None, emotion=None):
        text = str(dialog).lower()
        score = 0
        reasons = []

        if emotion == "негативный":
            score += 3
            reasons.append("негативная эмоция клиента")
        elif emotion == "нейтральный":
            score += 1

        for level, keywords in self.problem_keywords.items():
            matched = [keyword for keyword in keywords if keyword in text]
            if not matched:
                continue

            if level == "critical":
                score += 5
            elif level == "high":
                score += 3
            else:
                score += 1

            reasons.append(f"ключевые слова: {', '.join(matched[:3])}")

        question_count = text.count("?")
        if question_count >= 3:
            score += 1
            reasons.append("много уточняющих вопросов")

        if len(text) > 700:
            score += 1
            reasons.append("длинный диалог, требуется внимание")

        risk_topic_markers = ["возврат", "доставка", "качество", "оплата", "жалоба", "брак", "сломалось"]
        if any(marker in str(topic).lower() for marker in risk_topic_markers) and emotion == "негативный":
            score += 2
            reasons.append(f"рискованная тема: {topic}")

        if any(keyword in text for keyword in self.deescalation_keywords):
            score = max(0, score - 2)
            reasons.append("есть признаки решения вопроса")

        if score >= 7:
            severity = "критический"
        elif score >= 4:
            severity = "высокий"
        elif score >= 2:
            severity = "средний"
        else:
            severity = "низкий"

        return {
            "is_problem": score >= 4,
            "problem_score": score,
            "problem_severity": severity,
            "problem_type": self.detect_problem_type(text, topic),
            "problem_reason": "; ".join(dict.fromkeys(reasons)) if reasons else "явных признаков проблемы не найдено"
        }

    def detect_problem_type(self, text, topic=None):
        type_rules = [
            ("Задержка доставки", ["задерж", "не доставили", "не пришел", "не пришёл", "где мой заказ", "когда привез", "потеряли"]),
            ("Проблема с курьером", ["курьер", "опоздал", "нагруб", "хамил"]),
            ("Брак или поломка товара", ["брак", "сломался", "сломана", "не работает", "поврежден", "повреждён", "дефект"]),
            ("Возврат денег", ["возврат", "верните", "не вернули деньги", "не возвращают деньги"]),
            ("Проблема с оплатой", ["оплата", "платеж", "платёж", "списали", "двойное списание"]),
            ("Проблема с аккаунтом", ["аккаунт", "не могу войти", "пароль", "заблокировали"]),
            ("Жалоба на сервис", ["жалоба", "претензия", "ужасно", "плохо", "обман"]),
        ]

        matched_types = [name for name, keywords in type_rules if any(keyword in text for keyword in keywords)]
        if matched_types:
            return "; ".join(dict.fromkeys(matched_types))
        if topic and topic != "другое":
            return str(topic)
        return "Не определено"

    def analyze_dialog(self, dialog):
        topic = self.classify_topic(dialog)
        emotion = self.analyze_sentiment(dialog)
        problem = self.analyze_problem_scenario(dialog, topic, emotion)
        return {
            'topic': topic,
            'emotion': emotion,
            **problem
        }
