import pandas as pd
import re
import uuid
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_data")
COLLECTION_NAME = "dialogs"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_PATH = str(BASE_DIR / "sentence_transformers_cache")


class DialogDatabase:

    def __init__(self):

        self.texts = []

        self.full_texts = []

        self.embedding_model = None

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self._restore_cached_dialogs()

    # модель эмбиддингов

    def _get_embedding_model(self):

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=EMBEDDING_CACHE_PATH,
                local_files_only=self._is_embedding_model_cached()
            )

        return self.embedding_model

    def _is_embedding_model_cached(self):

        repo_cache_name = f"models--{EMBEDDING_MODEL.replace('/', '--')}"
        snapshots_path = (
            Path(EMBEDDING_CACHE_PATH)
            / repo_cache_name
            / "snapshots"
        )

        if not snapshots_path.exists():
            return False

        required_files = {
            "config.json",
            "model.safetensors",
            "modules.json",
            "tokenizer.json"
        }

        for snapshot_path in snapshots_path.iterdir():
            if not snapshot_path.is_dir():
                continue

            if all(
                (snapshot_path / file_name).exists()
                for file_name in required_files
            ):
                return True

        return False

    # восстановление данных из бд

    def _restore_cached_dialogs(self):

        if self.collection.count() == 0:
            return

        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.texts = stored.get("documents", [])

        self.full_texts = [
            (metadata or {}).get("full_text", document)
            for document, metadata in zip(
                self.texts,
                stored.get("metadatas", [])
            )
        ]

    # очистка коллекции перед новой загрузкой
    def _reset_collection(self):

        try:
            self.client.delete_collection(
                name=COLLECTION_NAME
            )
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    # извлечение клиентского текста
    def extract_client_text(self, dialog):

        matches = re.findall(
            r"Клиент:\s*(.*?)(?=Оператор:|$)",
            dialog,
            re.DOTALL
        )

        if matches:
            return " ".join(matches).strip().lower()

        return str(dialog).strip().lower()

    # загрузка диалогов
    def load_dialogs(self, csv_file):

        df = pd.read_csv(csv_file)

        if "dialog" not in df.columns:
            raise Exception("CSV должен содержать колонку 'dialog'")

        #убираем полные дубликаты диалогов
        df = df.drop_duplicates(subset=["dialog"])

        #полный текст для вывода
        self.full_texts = df["dialog"].astype(str).tolist()

        #клиентский текст для поиска
        self.texts = (
            df["dialog"]
            .apply(self.extract_client_text)
            .tolist()
        )

        self._reset_collection()

        if self.texts:
            embeddings = (
                self._get_embedding_model()
                .encode(
                    self.texts,
                    normalize_embeddings=True
                )
                .tolist()
            )

            ids = [
                str(uuid.uuid5(uuid.NAMESPACE_URL, full_text))
                for full_text in self.full_texts
            ]

            metadatas = [
                {
                    "full_text": full_text,
                    "client_text": client_text
                }
                for full_text, client_text in zip(
                    self.full_texts,
                    self.texts
                )
            ]

            self.collection.add(
                ids=ids,
                documents=self.texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

        print(
            "Уникальных диалогов загружено в ChromaDB: "
            f"{len(self.texts)}"
        )

        return len(self.texts)

    #поиск похожих обращений
    def find_similar(self, query_text, top_k=5):

        if not self.texts or self.collection.count() == 0:
            return []

        query_text = str(query_text).lower().strip()

        if not query_text:
            return []

        query_embedding = (
            self._get_embedding_model()
            .encode(
                [query_text],
                normalize_embeddings=True
            )[0]
            .tolist()
        )

        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, self.collection.count()),
            include=["metadatas", "distances"]
        )

        results = []
        used_texts = set()

        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        for metadata, distance in zip(metadatas, distances):
            full_text = metadata.get("full_text", "").strip()
            score = round(max(0, 1 - distance) * 100, 2)

            if score <= 0:
                continue

            if full_text in used_texts:
                continue

            used_texts.add(full_text)

            results.append((
                full_text,
                score
            ))

            if len(results) >= top_k:
                break

        return results
