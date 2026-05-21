import pandas as pd
import re
import uuid
import chromadb
from sentence_transformers import SentenceTransformer


class DialogDatabase:

    def __init__(self):

        # embedding модель (E5 — хорошо подходит для search)
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )

        # persistent vector DB
        self.client = chromadb.PersistentClient(
            path="./chroma_data"
        )

        self.collection = self.client.get_or_create_collection(
            name="dialogs"
        )

        self.texts = []

    # -----------------------------

    def extract_client_text(self, dialog):

        matches = re.findall(
            r"Клиент:\s*(.*?)(?=Оператор:|$)",
            dialog,
            re.DOTALL
        )

        return " ".join(matches).strip().lower()

    # -----------------------------

    def load_dialogs(self, csv_file):

        df = pd.read_csv(csv_file)

        if "dialog" not in df.columns:
            raise Exception("CSV должен содержать колонку 'dialog'")

        # только клиентский текст
        self.texts = df["dialog"].apply(
            self.extract_client_text
        ).tolist()

        ids = [str(uuid.uuid4()) for _ in self.texts]

        # E5 форматирование (ВАЖНО для качества поиска)
        documents = [
            f"passage: {text}"
            for text in self.texts
        ]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # добавляем в Chroma
        self.collection.add(
            ids=ids,
            documents=self.texts,
            embeddings=embeddings
        )

        print(f"Loaded dialogs: {len(self.texts)}")

        return len(self.texts)

    # -----------------------------

    def find_similar(self, query_text, top_k=5):

        if not self.texts:
            return []

        query_text = query_text.lower()

        # важная нормализация запроса
        query = f"query: жалоба клиента: {query_text}"

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        output = []

        for doc, dist in zip(
            results["documents"][0],
            results["distances"][0]
        ):

            score = round((1 - dist) * 100, 2)
            output.append((doc, score))

        return output