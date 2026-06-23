from app.services.runs.duckdb_store import append_chat_message, clear_run_data, list_chat_messages


def test_append_and_list_chat_messages_by_user():
    run_id = "test-chat-history-run"
    user_id = "user-chat-1"
    clear_run_data(run_id)

    append_chat_message(
        run_id,
        user_id=user_id,
        role="user",
        text="¿Qué grupo tiene peor SLA?",
    )
    append_chat_message(
        run_id,
        user_id=user_id,
        role="assistant",
        text="El grupo 2 concentra el mayor incumplimiento.",
        metadata={
            "insights": [
                {
                    "id": "insight-1",
                    "title": "SLA cluster 2",
                    "description": "Incumplimiento alto",
                }
            ],
            "llm_used": True,
            "llm_detail": "Azure OpenAI",
            "llm_mode": "llm",
        },
    )

    messages = list_chat_messages(run_id=run_id, user_id=user_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["insights"][0]["id"] == "insight-1"
    assert messages[1]["llm_used"] is True

    other_user_messages = list_chat_messages(run_id=run_id, user_id="other-user")
    assert other_user_messages == []

    cleared = clear_run_data(run_id)
    assert cleared["chat_messages"] >= 2
    assert list_chat_messages(run_id=run_id, user_id=user_id) == []
