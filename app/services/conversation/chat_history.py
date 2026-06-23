from __future__ import annotations

from typing import Any

from app.schemas import ChatMessageRecord, ChatResponse, InsightCandidate
from app.services.runs.duckdb_store import append_chat_message, list_chat_messages


def _insights_to_metadata(insights: list[InsightCandidate]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in insights]


def load_history(*, run_id: str, user_id: str | None, limit: int = 200) -> list[ChatMessageRecord]:
    rows = list_chat_messages(run_id=run_id, user_id=user_id, limit=limit)
    messages: list[ChatMessageRecord] = []
    for row in rows:
        insights_raw = row.get("insights") or []
        insights = [InsightCandidate.model_validate(item) for item in insights_raw]
        messages.append(
            ChatMessageRecord(
                id=str(row["id"]),
                role=row["role"],
                text=row["text"],
                insights=insights,
                llm_used=row.get("llm_used"),
                llm_detail=row.get("llm_detail"),
                created_at=row["created_at"],
            )
        )
    return messages


def persist_exchange(
    *,
    run_id: str,
    user_id: str | None,
    question: str,
    response: ChatResponse,
) -> None:
    append_chat_message(
        run_id,
        user_id=user_id,
        role="user",
        text=question,
    )
    append_chat_message(
        run_id,
        user_id=user_id,
        role="assistant",
        text=response.answer,
        metadata={
            "insights": _insights_to_metadata(response.insights),
            "llm_used": response.llm_used,
            "llm_detail": response.llm_detail,
            "llm_mode": response.llm_mode,
        },
    )


def persist_note(
    *,
    run_id: str,
    user_id: str | None,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessageRecord:
    row = append_chat_message(
        run_id,
        user_id=user_id,
        role="assistant",
        text=text,
        metadata=metadata,
    )
    return ChatMessageRecord(
        id=str(row["id"]),
        role="assistant",
        text=text,
        insights=[],
        llm_used=None,
        llm_detail=None,
        created_at=row["created_at"],
    )
