import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from ingest import Ingestor
from chain import Chain

load_dotenv()

app = FastAPI(title="DocChat API")
chain = Chain()

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        temp_dir = os.getenv("TEMP", "/tmp")
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        ingestor = Ingestor()
        ingestor.ingest_from_file(temp_path)
        
        os.remove(temp_path)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"{file.filename} ingéré avec succès"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/chat")
async def chat(question: str):
    try:
        response = chain.run(question)
        return JSONResponse(
            status_code=200,
            content={"response": response}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health():
    return {"status": "ok"}