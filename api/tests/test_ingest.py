# test_ingest.py
from dotenv import load_dotenv
load_dotenv()

from api.ingest import Ingestor

ingestor = Ingestor()
ingestor.ingest_init()