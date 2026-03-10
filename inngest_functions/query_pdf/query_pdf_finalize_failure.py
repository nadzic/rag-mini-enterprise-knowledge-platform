from typing import Any

from services.langfuse_client import flush_langfuse


def finalize_failure(root_span: Any, step_span: Any, exc: Exception) -> None:
    """Mark spans as failed and flush Langfuse so error traces are persisted."""
    if step_span is not None:
        step_span.update(level="ERROR", status_message=str(exc))
        step_span.end()
    if root_span is not None:
        root_span.update(level="ERROR", status_message=str(exc))
        root_span.end()
    flush_langfuse()
