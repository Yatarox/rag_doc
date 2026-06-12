# api/ingest.py - Version modifiée
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import Embedder 
from chromaDB_conn import chromaDB_conn
import os

class Ingestor:
    def __init__(self, chunk_size=os.getenv("CHUNK_SIZE", default=1000)):
        self.chunk_size = int(chunk_size)
        self.chroma_db = chromaDB_conn()
        self.embedder = Embedder()
        self.source = os.getenv("SOURCE_PATH", "./data")
    
    def load_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.1),
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    
    def ingest_from_file(self, file_path):
        if file_path.endswith(".pdf"):
            text = self.load_pdf(file_path)
            chunks = self.chunk_text(text)
            embeddings = self.embedder.encode(chunks)  # ← Plus simple
            self.chroma_db.add_embeddings(chunks, embeddings)
            print(f"✅ {len(chunks)} chunks ingérés depuis {file_path}")
            return chunks, embeddings
        else:
            print(f"❌ Format non supporté: {file_path}")
            return None, None
    
    def ingest_init(self):
        if not os.path.exists(self.source):
            os.makedirs(self.source)
            print(f"Dossier {self.source} créé")
            return
        
        pdf_files = [f for f in os.listdir(self.source) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"Aucun PDF trouvé dans {self.source}")
            return
        
        for file in pdf_files:
            file_path = os.path.join(self.source, file)
            self.ingest_from_file(file_path)