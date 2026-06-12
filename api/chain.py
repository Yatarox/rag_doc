import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from chromaDB_conn import chromaDB_conn

load_dotenv()

class Chain:
    def __init__(self):
        self.chroma_db = chromaDB_conn()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
        )
        
        self.llm = ChatOpenRouter(
            model=os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2-pro:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.3
        )
        
        self.prompt = PromptTemplate.from_template(
            """Tu es un assistant utile. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
            Si la réponse n'est pas dans le contexte, dis "Je ne peux pas répondre à cette question."

            Contexte : {context}

            Question : {question}

            Réponse :"""
        )
    
    def _search_chroma(self, question: str, top_k: int = 3) -> list:
        query_embedding = self.embeddings.embed_query(question)
        
        results = self.chroma_db.query(query_embedding, top_k=top_k)
        
        return results
    
    def run(self, question: str) -> str:
        chunks = self._search_chroma(question)
        
        if not chunks:
            return "Aucun document pertinent trouvé pour répondre à cette question."
        
        context = "\n\n---\n\n".join(chunks)
        
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        response = self.llm.invoke(formatted_prompt)
        
        return response.content