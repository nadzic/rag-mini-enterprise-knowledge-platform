from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

_reranker_model = None


def rerank_enabled() -> bool:
    raw = os.getenv("RERANK_ENABLED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def rerank_top_n(default: int = 20) -> int:
    raw = os.getenv("RERANK_TOP_N", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _reranker_model_name() -> str:
    return os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def _get_reranker():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    from sentence_transformers import CrossEncoder

    _reranker_model = CrossEncoder(_reranker_model_name())
    return _reranker_model


def rerank_records(
    question: str,
    records: Sequence[dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    if not records:
        return []
    if not rerank_enabled():
        return list(records)[:top_k]

    model = _get_reranker()
    pairs = [(question, record.get("text", "")) for record in records]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(records, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [record for record, _score in ranked[:top_k]]
