import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

class DialogDatabase:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.client = chromadb.PersistentClient(path="./chroma_data")
        self.collection = self.client.get_or_create_collection(name="dialogs")
        self.texts = []
    
    def load_dialogs(self, csv_file):
        df = pd.read_csv(csv_file)
        if 'dialog' not in df.columns:
            raise Exception("В CSV нет колонки 'dialog'")
        self.texts = df['dialog'].astype(str).tolist()
        ids = [str(uuid.uuid4()) for _ in self.texts]
        embeddings = self.model.encode(self.texts).tolist()
        self.collection.add(ids=ids, documents=self.texts, embeddings=embeddings)
        return len(self.texts)
    
    def find_similar(self, query_text, top_k=5):
        query_lower = query_text.lower()
        
        # Сначала ищем точное совпадение по ключевым словам
        keyword_matches = []
        for text in self.texts:
            if query_lower in text.lower():
                keyword_matches.append((text, 100))
        
        if keyword_matches:
            return keyword_matches[:top_k]
        
        # Если нет точных совпадений, ищем эмбеддингами
        query_embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        similar = []
        if results['documents'][0]:
            for doc, dist in zip(results['documents'][0], results['distances'][0]):
                similar.append((doc, round((1 - dist) * 100, 2)))
        return similar