import os

from dotenv import load_dotenv
from openai import OpenAI
from services.openai_api_key import resolve_openai_api_key

load_dotenv()

EMBED_MODEL = "text-embedding-3-large"
client: OpenAI | None = None

def _get_openai_client() -> OpenAI:
    global client
    if client is not None:
        return client

    api_key = resolve_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)
    return client


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_openai_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
