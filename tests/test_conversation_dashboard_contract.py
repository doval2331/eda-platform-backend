from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from app.schemas import ConversationDashboardSpec
from app.services.conversation.chart_data import build_conversation_chart_data
from app.services.conversation.dashboard_spec import _sanitize_llm_spec


def _fallback_dashboard_spec() -> ConversationDashboardSpec:
    return ConversationDashboardSpec.model_validate(
        {
            "semantic_variables": [
                {
                    "name": "affected_service",
                    "label": "Servicio afectado",
                    "role": "business",
                    "can_chart": True,
                },
                {
                    "name": "priority",
                    "label": "Prioridad",
                    "role": "business",
                    "can_chart": True,
                },
                {
                    "name": "avg_resolution_hours",
                    "label": "Tiempo promedio de resolucion",
                    "role": "metric",
                    "can_chart": True,
                },
            ],
            "agent_recommendations": [
                {
                    "id": "rec-valid",
                    "title": "Revisar servicios",
                    "action_type": "chart",
                    "linked_visualization_id": "viz-valid",
                }
            ],
            "suggested_visualizations": [
                {
                    "id": "viz-valid",
                    "title": "Servicios por prioridad",
                    "chart_type": "bar",
                    "x": "affected_service",
                    "metric": "count",
                    "group_by": "affected_service",
                }
            ],
            "active_chart_default": {"visualization_id": "viz-valid"},
            "conclusions": [
                {
                    "id": "conclusion-valid",
                    "conclusion": "Hay servicios para revisar.",
                    "evidence": "Hallazgos guardados.",
                    "related_chart": "viz-valid",
                }
            ],
        }
    )


def test_llm_dashboard_contract_flags_unknown_variables_and_unlinked_charts() -> None:
    fallback = _fallback_dashboard_spec()
    raw_spec = {
        "agent_recommendations": [
            {
                "id": "rec-invented",
                "title": "Preparar una lectura narrativa",
                "action_type": "chart",
                "linked_visualization_id": "viz-missing",
            }
        ],
        "suggested_visualizations": [
            {
                "id": "viz-invented",
                "title": "Variable inventada",
                "chart_type": "bar",
                "x": "variable_que_no_existe",
                "metric": "count",
            }
        ],
        "conclusions": [
            {
                "id": "conclusion-invented",
                "conclusion": "Conclusion sin grafico real.",
                "evidence": "Evidencia textual.",
                "related_chart": "viz-missing",
            }
        ],
    }

    spec = _sanitize_llm_spec(raw_spec, fallback)

    assert spec.schema_version == "conversation-dashboard/v1"
    assert spec.contract_status == "warning"
    assert "missing_variable" in spec.llm_risk_flags
    assert any("variable_que_no_existe" in warning for warning in spec.contract_warnings)
    assert spec.agent_recommendations[0].action_type == "chat"
    assert spec.agent_recommendations[0].linked_visualization_id == ""
    assert spec.conclusions[0].related_chart == ""


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_response_marks_missing_dimension_as_not_buildable(load_run_evidences_mock) -> None:
    load_run_evidences_mock.return_value = pd.DataFrame(
        [
            {
                "incident_id": "INC001",
                "preview": "Incidencia sin variables de negocio disponibles",
            }
        ]
    )

    response = build_conversation_chart_data(
        run_id="run-1",
        visualization={
            "id": "viz-missing",
            "title": "Variable no disponible",
            "chart_type": "bar",
            "x": "variable_que_no_existe",
            "metric": "count",
        },
        limit=5,
        evidence_limit=5,
    )

    assert response.schema_version == "conversation-chart-data/v1"
    assert response.validation.chart_is_buildable is False
    assert response.validation.requires_data is True
    assert "dimension" in response.validation.missing
