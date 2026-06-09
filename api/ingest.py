from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import os 
from chromaDB_conn import chromaDB_conn

class Ingestor:

    def __init__(self,chunk_size=os.getenv("CHUNK_SIZE", default=1000) , model_name=os.getenv("MODEL_NAME"), source=os.getenv("SOURCE_PATH")):
        self.chunk_size = int(chunk_size)
        self.chroma_db = chromaDB_conn()
        self.model_name = model_name
        self.source = source
        self.model = self.load_model()


    def load_model(self):
        model = SentenceTransformer(self.model_name, trust_remote_code=True)
        print(f"Model {self.model_name} loaded successfully.")
        return model

    def load_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        print(f"PDF loaded successfully from {file_path}.")
        return text
    
    def chunk_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.1),
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"Text chunked into {len(chunks)} chunks.")
        return chunks

    def ingest_init(self):
        dir_list = os.listdir(self.source)
        for file in dir_list:
            if file.endswith(".pdf"):
                file_path = os.path.join(self.source, file)
                text = self.load_pdf(file_path)
                chunks = self.chunk_text(text)
                embeddings = self.model.encode(chunks).tolist() 
                self.chroma_db.add_embeddings(chunks, embeddings)
                print(f"Data from {file} ingested successfully.")
        print(f"Ingesting data from {self.source}")