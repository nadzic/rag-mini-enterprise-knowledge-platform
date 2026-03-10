import inngest

from rag_types import RAGChunkAndSource
from services import load_and_chunk_pdf


def load_chunks_and_source(context: inngest.Context) -> RAGChunkAndSource:
    """Load a PDF from event data, chunk it, and attach the resolved source id."""
    pdf_path = context.event.data["pdf_path"]
    source_id = context.event.data.get("source_id", str(pdf_path))
    chunks = load_and_chunk_pdf(str(pdf_path))
    return RAGChunkAndSource(chunks=chunks, source_id=str(source_id))
