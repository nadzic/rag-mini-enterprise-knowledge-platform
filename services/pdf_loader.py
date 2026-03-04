from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str) -> list[str]:
    texts: list[str] = []
    try:
        from llama_index.readers.file import PDFReader

        docs = PDFReader().load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]
    except ModuleNotFoundError:
        # Fallback when optional llama-index reader package is missing.
        pdf = PdfReader(path)
        texts = [page.extract_text() or "" for page in pdf.pages]

    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
