from services.openai_api_key import resolve_openai_api_key


def get_openai_api_key() -> str:
    return resolve_openai_api_key()
