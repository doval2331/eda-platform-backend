from unittest.mock import patch

import pandas as pd

from app.services.conversation.conversation import build_chat_response


@patch("app.services.conversation.conversation.load_run_evidences")
def test_unmatched_message_returns_short_prompt_without_catalog(mock_load):
    mock_load.return_value = pd.DataFrame(
        [
            {
                "run_id": "run-fallback",
                "evidence_index": 0,
                "sla_breach_rate": 0.627,
                "cluster_label": 1,
                "affected_service": "VPN",
            }
        ]
    )

    response = build_chat_response("run-fallback", "Hola")

    assert "elige una pregunta sugerida" in response.answer.lower()
    assert response.llm_used is False
    assert response.insights == []
    assert response.suggested_questions
    assert "preguntas que la app puede responder" not in response.answer.lower()
    assert "2330" not in response.answer
    assert "62.7" not in response.answer


@patch("app.services.conversation.conversation.load_run_evidences")
def test_explicit_dynamic_questions_request_returns_catalog(mock_load):
    mock_load.return_value = pd.DataFrame(
        [
            {
                "run_id": "run-catalog",
                "evidence_index": 0,
                "cluster_label": 1,
            }
        ]
    )

    response = build_chat_response("run-catalog", "que preguntas puedo hacer")

    assert "preguntas que la app puede responder" in response.answer.lower()
    assert any(insight.id == "available-analytic-tools" for insight in response.insights)


@patch("app.services.conversation.conversation.load_run_evidences")
def test_explicit_overview_request_still_returns_overview(mock_load):
    mock_load.return_value = pd.DataFrame(
        [
            {
                "run_id": "run-overview",
                "evidence_index": 0,
                "sla_breach_rate": 0.5,
                "cluster_label": 0,
            }
        ]
    )

    response = build_chat_response("run-overview", "dame un resumen general")

    assert any(insight.id.startswith("overview") for insight in response.insights)
