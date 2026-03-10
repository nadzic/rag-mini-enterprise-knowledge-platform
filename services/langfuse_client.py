import os

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

_langfuse_client: Langfuse | None = None
_langfuse_initialized = False


def get_langfuse_client() -> Langfuse | None:
    """Return a singleton Langfuse client when configured, otherwise None."""
    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    _langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    return _langfuse_client


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()
