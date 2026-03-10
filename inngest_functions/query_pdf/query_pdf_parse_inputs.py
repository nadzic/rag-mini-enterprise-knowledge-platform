import inngest


def parse_query_inputs(ctx: inngest.Context) -> tuple[str, int]:
    """Read and normalize question/top_k from the incoming query event."""
    question = str(ctx.event.data["question"])
    top_k_raw = ctx.event.data.get("top_k", 5)
    top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) else 5
    return question, top_k
