# api/embedder.py
from sentence_transformers import SentenceTransformer
import os

class Embedder:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
            print(f"🔄 Chargement du modèle {model_name}...")
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(model_name)
            print(f"✅ Modèle {model_name} chargé avec succès")
        return cls._instance
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

if __name__ == "__main__":
    embedder = Embedder()
    test_vector = embedder.encode(["test"])
    print(f"✅ Embedding généré, dimension: {len(test_vector[0])}")