from fastapi import FastAPI
import inngest.fast_api

from inngest_functions import rag_ingest_pdf, rag_query_pdf_ai
from inngest_functions.client import inngest_client

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])