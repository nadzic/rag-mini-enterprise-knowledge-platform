import os


def resolve_openai_api_key() -> str:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_AI_KEY")
        or os.getenv("OPENAI_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY (or OPEN_AI_KEY / OPENAI_KEY)."
        )
    return api_key
