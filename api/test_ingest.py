# test_ingest.py
from dotenv import load_dotenv
load_dotenv()

from ingest import Ingestor

ingestor = Ingestor()
ingestor.ingest_init()