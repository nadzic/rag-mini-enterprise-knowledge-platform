import inngest
from inngest.experimental import ai

from ..env import get_openai_api_key


def build_user_prompt(question: str, contexts: list[str]) -> str:
    """Build the retrieval-augmented user prompt consumed by the chat model."""
    context_block = "\n\n".join(f"- {context}" for context in contexts)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )


async def infer_answer(ctx: inngest.Context, user_content: str):
    """Run the LLM step through Inngest AI with a fixed answer configuration."""
    adapter = ai.openai.Adapter(
        auth_key=get_openai_api_key(),
        model="gpt-4o-mini",
    )
    return await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions based on the provided context.",
                },
                {"role": "user", "content": user_content},
            ],
        },
    )


def extract_answer(response: object) -> str:
    """Safely extract the assistant text from the OpenAI chat response payload."""
    data = response if isinstance(response, dict) else {}
    raw_choices = data.get("choices", [])
    choices = raw_choices if isinstance(raw_choices, list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    return str(message.get("content", "")).strip()
