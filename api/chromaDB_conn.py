import chromadb
import os 

class chromaDB_conn:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "chromadb"),
            port=int(os.getenv("CHROMA_PORT", 8000))
        )
        self.collection = self.chroma_client.get_or_create_collection(name="rag_doc")

    def add_embeddings(self, chunks, embeddings):
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embeddings[i]],
                metadatas=[{"text": chunk}]
            )
    def query(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [meta['text'] for meta in results['metadatas'][0]]
    
    def delete_collection(self):
        self.chroma_client.delete_collection("rag_doc")
        self.collection = self.chroma_client.get_or_create_collection(name="rag_doc")