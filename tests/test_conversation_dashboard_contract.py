from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app.schemas import ConversationDashboardSpec
from app.services.conversation.chart_data import (
    build_conversation_chart_data,
    build_conversation_chart_error_response,
)
from app.services.conversation.dashboard_spec import (
    build_dashboard_context,
    _feedback_summary,
    _fallback_spec,
    _operational_readiness,
    _refresh_dashboard_contract,
    _run_ids_from_context,
    _sanitize_llm_spec,
    _usage_summary,
)
from app.services.conversation.semantic_dictionary import (
    get_semantic_dictionary,
    load_configured_semantic_variables,
    reload_semantic_dictionary,
    save_configured_semantic_variables,
    semantic_dictionary_status,
)


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
    spec.llm_used = True
    spec = _refresh_dashboard_contract(spec)

    assert spec.schema_version == "conversation-dashboard/v1"
    assert spec.contract_status == "warning"
    assert "missing_variable" in spec.llm_risk_flags
    assert any("variable_que_no_existe" in warning for warning in spec.contract_warnings)
    assert spec.agent_recommendations[0].action_type == "chat"
    assert spec.agent_recommendations[0].linked_visualization_id == ""
    assert spec.conclusions[0].related_chart == ""
    assert spec.operational_readiness.llm_validated is False
    assert "Propuesta LLM ajustada por validacion backend" in spec.operational_readiness.blocking_reasons
    assert any("Revisa las advertencias" in action for action in spec.operational_readiness.required_actions)


def test_fallback_dashboard_splits_questions_by_profile() -> None:
    spec = _fallback_spec(
        {
            "dataset_summary": [{"dataset_name": "Incidencias IT", "rows": 120, "columns": 5}],
            "columns": {
                "column_count": 5,
                "business": ["affected_service", "priority"],
                "numeric": ["no_of_reassignments"],
            },
            "metrics": {"records_count": 120, "key_metrics": ["count"]},
            "clusters": [],
            "semantic_variables": [
                {"name": "affected_service", "label": "Servicio afectado", "role": "business", "can_chart": True},
                {"name": "priority", "label": "Prioridad", "role": "business", "can_chart": True},
            ],
            "operational_readiness": {"status": "operational", "run_scope": "single_run", "run_ids": ["run-1"]},
        },
        [
            {
                "id": "insight-1",
                "title": "Servicio con prioridad alta",
                "description": "Evidencia guardada.",
                "metric_value": 10,
                "metric_label": "Prioridad",
            }
        ],
    )

    assert spec.llm_used is False
    assert spec.suggested_questions.functional_user
    assert spec.suggested_questions.expert_user
    assert spec.suggested_questions.functional_user != spec.suggested_questions.expert_user


def test_llm_spec_resolves_semantic_aliases_before_contract_validation() -> None:
    fallback = _fallback_dashboard_spec()
    raw_spec = {
        "agent_recommendations": [
            {
                "id": "rec-alias",
                "title": "Ver servicios afectados",
                "action_type": "chart",
                "linked_visualization_id": "viz-alias",
            }
        ],
        "suggested_visualizations": [
            {
                "id": "viz-alias",
                "title": "Servicios por volumen",
                "chart_type": "bar",
                "x": "Servicio afectado",
                "metric": "Incidencias",
            }
        ],
    }

    spec = _sanitize_llm_spec(raw_spec, fallback)

    assert spec.suggested_visualizations[0].x == "affected_service"
    assert spec.suggested_visualizations[0].metric == "count"
    assert "missing_variable" not in spec.llm_risk_flags
    assert spec.agent_recommendations[0].action_type == "chart"


@patch("app.services.conversation.dashboard_spec.list_chat_messages")
def test_feedback_summary_reads_persisted_dashboard_feedback(list_chat_messages_mock) -> None:
    list_chat_messages_mock.return_value = [
        {
            "id": "msg-1",
            "created_at": "2026-07-08T01:00:00",
            "metadata": {
                "kind": "conversation_dashboard_feedback",
                "target_id": "rec-1",
                "target_title": "Revisar servicios",
                "helpful": True,
                "chart_validated": True,
                "chart_generated": True,
                "action_taken": True,
                "reason": "action_taken",
                "reason_label": "Termino en accion",
                "final_state": "action_taken",
                "drilldown_used": True,
                "tickets_analyzed": 12,
                "evidence_materialized": True,
                "evidence_records": 42,
            },
        },
        {
            "id": "msg-2",
            "created_at": "2026-07-08T01:02:00",
            "metadata": {
                "kind": "conversation_dashboard_feedback",
                "recommendation_id": "rec-2",
                "recommendation_title": "Variable confusa",
                "helpful": False,
                "reason": "wrong_variable",
                "reason_label": "Variable incorrecta",
                "has_warning": True,
            },
        },
        {
            "id": "msg-3",
            "metadata": {"kind": "conversation_dashboard_event", "event_type": "chart_opened"},
        },
    ]

    summary = _feedback_summary(run_ids=["run-1"], user_id="user-1")

    assert summary["total"] == 2
    assert summary["useful"] == 1
    assert summary["not_useful"] == 1
    assert summary["useful_recommendation_ids"] == ["rec-1"]
    assert summary["not_useful_recommendation_ids"] == ["rec-2"]
    assert "Variable confusa" in summary["not_useful_titles"]
    assert summary["reason_counts"]["action_taken"] == 1
    assert summary["reason_counts"]["wrong_variable"] == 1
    assert summary["operational_outcomes"]["action_taken"] == 1
    assert summary["operational_outcomes"]["drilldown_used"] == 1
    assert summary["operational_outcomes"]["tickets_analyzed"] == 12
    assert summary["requires_attention"][0]["reason"] == "wrong_variable"


def test_dashboard_spec_preserves_recommendation_feedback_contract() -> None:
    spec = _fallback_spec(
        {
            "dataset_summary": [{"dataset_name": "Incidencias IT", "rows": 120, "columns": 5}],
            "columns": {
                "column_count": 5,
                "business": ["affected_service", "priority"],
                "numeric": ["no_of_reassignments"],
            },
            "metrics": {"records_count": 120, "key_metrics": ["count"]},
            "clusters": [],
            "semantic_variables": [
                {"name": "affected_service", "label": "Servicio afectado", "role": "business", "can_chart": True},
            ],
            "recommendation_feedback": {
                "total": 1,
                "useful": 1,
                "useful_recommendation_ids": ["rec-a"],
            },
            "operational_readiness": {"status": "operational", "run_scope": "single_run", "run_ids": ["run-1"]},
        },
        [],
    )

    assert spec.recommendation_feedback["total"] == 1
    assert spec.recommendation_feedback["useful_recommendation_ids"] == ["rec-a"]


@patch("app.services.conversation.dashboard_spec.list_chat_messages")
def test_usage_summary_reads_dashboard_events(list_chat_messages_mock) -> None:
    list_chat_messages_mock.return_value = [
        {
            "metadata": {
                "kind": "conversation_dashboard_event",
                "event_type": "recommendation_graph_opened",
                "recommendation_id": "rec-1",
                "visualization_id": "viz-1",
                "visualization_title": "Servicios por prioridad",
            },
            "created_at": "2026-07-08T01:05:00",
        },
        {
            "metadata": {
                "kind": "conversation_dashboard_event",
                "event_type": "tickets_sent_to_agent",
                "recommendation_id": "rec-1",
                "ticket_count": 12,
            },
            "created_at": "2026-07-08T01:06:00",
        },
        {
            "metadata": {
                "kind": "conversation_dashboard_event",
                "event_type": "recommendations_presented",
                "recommendation_ids": ["rec-1", "rec-2"],
            },
            "created_at": "2026-07-08T01:04:00",
        },
        {"metadata": {"kind": "conversation_dashboard_feedback", "target_id": "rec-1"}},
    ]

    summary = _usage_summary(run_ids=["run-1"], user_id="user-1")

    assert summary["total"] == 3
    assert summary["charts_opened"] == 1
    assert summary["tickets_sent_to_agent"] == 1
    assert summary["events_by_type"]["recommendation_graph_opened"] == 1
    assert summary["operational_funnel"]["tickets_sent_to_agent"] == 1
    assert summary["by_recommendation"][0]["recommendation_id"] == "rec-1"
    assert summary["ignored_recommendation_ids"] == ["rec-2"]


def test_dashboard_context_uses_explicit_run_id_over_mixed_insights() -> None:
    run_ids = _run_ids_from_context(
        "run-selected",
        [
            {"run_id": "run-old"},
            {"run_id": "run-other"},
        ],
    )

    assert run_ids == ["run-selected"]


@patch("app.services.conversation.dashboard_spec.list_chat_messages")
@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
@patch("app.services.conversation.dashboard_spec._safe_agent_payload")
@patch("app.services.conversation.dashboard_spec.get_dataset_meta")
@patch("app.services.conversation.dashboard_spec._load_evidences")
@patch("app.services.conversation.dashboard_spec._load_run_rows")
def test_dashboard_context_scopes_evidence_to_selected_run(
    load_run_rows_mock,
    load_evidences_mock,
    get_dataset_meta_mock,
    safe_agent_payload_mock,
    load_configured_semantic_variables_mock,
    list_chat_messages_mock,
) -> None:
    load_configured_semantic_variables_mock.return_value = []
    list_chat_messages_mock.return_value = []
    safe_agent_payload_mock.return_value = []
    get_dataset_meta_mock.return_value = {
        "filename": "Dataset seleccionado.csv",
        "n_rows": 1,
        "n_cols": 4,
    }
    load_run_rows_mock.return_value = [
        SimpleNamespace(
            id="run-selected",
            project_id="project-selected",
            dataset_id="dataset-selected",
            source_id="dataset-selected",
            source_name="Dataset seleccionado",
            project_name="Prueba seleccionada",
            source_type="csv",
            n_samples=1,
            created_at=None,
            reduction_method="PCA",
            seed=42,
            outliers_count=0,
            silhouette=None,
            davies_bouldin=None,
            result_json="{}",
        )
    ]
    load_evidences_mock.return_value = {
        "run-selected": pd.DataFrame(
            [
                {
                    "incident_id": "INC-SELECTED",
                    "affected_service": "App",
                    "priority": "Alta",
                    "preview": "Numero=INC-SELECTED | Affected_Service=App | Priority=Alta",
                }
            ]
        )
    }

    db = object()
    context = build_dashboard_context(
        db=db,
        user_id="user-1",
        run_id="run-selected",
        insights=[
            {"id": "old", "run_id": "run-old", "metric_label": "Prioridad", "metric_value": 10},
            {"id": "other", "run_id": "run-other", "metric_label": "Prioridad", "metric_value": 20},
        ],
    )

    load_run_rows_mock.assert_called_once_with(db, ["run-selected"])
    assert load_evidences_mock.call_args.args[0] == ["run-selected"]
    assert context["run_ids"] == ["run-selected"]
    assert context["semantic_project_id"] == "project-selected"
    assert context["dataset_summary"][0]["run_id"] == "run-selected"
    assert context["evidence_summary"]["records_count"] == 1
    assert context["operational_readiness"]["active_run_id"] == "run-selected"
    assert context["operational_readiness"]["run_scope"] == "single_run"


@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
def test_operational_readiness_reports_single_run_scope(load_configured_semantic_variables_mock) -> None:
    load_configured_semantic_variables_mock.return_value = []

    readiness = _operational_readiness(
        run_ids=["run-selected"],
        evidence_by_run={
            "run-selected": pd.DataFrame([{"incident_id": "INC001", "affected_service": "App"}]),
            "run-other": pd.DataFrame([{"incident_id": "INC999", "affected_service": "DB"}]),
        },
        evidence_summary={"records_count": 1},
        semantic_variables=[{"name": "affected_service", "role": "business"}],
        insights=[{"id": "insight-1", "run_id": "run-selected"}],
    )

    assert readiness["run_scope"] == "single_run"
    assert readiness["active_run_id"] == "run-selected"
    assert readiness["run_ids"] == ["run-selected"]
    assert readiness["evidence_runs"] == 1
    assert readiness["semantic_dictionary_total"] == 1
    assert readiness["semantic_dictionary_configured_count"] == 0
    assert readiness["status"] == "operational"
    assert readiness["decision_level"] == "operational"
    assert readiness["evidence_mode"] == "materialized"
    assert readiness["trust_level"] == "media"
    assert "casos reales" in readiness["functional_message"]
    assert readiness["recommended_next_step"]
    assert readiness["blocking_reasons"] == []
    assert any("diccionario semantico" in action for action in readiness["required_actions"])


@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
def test_operational_readiness_warns_when_multiple_runs_are_combined(load_configured_semantic_variables_mock) -> None:
    load_configured_semantic_variables_mock.return_value = []

    readiness = _operational_readiness(
        run_ids=["run-a", "run-b"],
        evidence_by_run={
            "run-a": pd.DataFrame([{"incident_id": "INC001", "affected_service": "App"}]),
            "run-b": pd.DataFrame([{"incident_id": "INC002", "affected_service": "DB"}]),
        },
        evidence_summary={"records_count": 2},
        semantic_variables=[{"name": "affected_service", "role": "business"}],
        insights=[{"id": "insight-1", "run_id": "run-a"}],
    )

    assert readiness["run_scope"] == "multi_run"
    assert readiness["active_run_id"] == ""
    assert readiness["run_ids"] == ["run-a", "run-b"]
    assert readiness["status"] == "interpretive"
    assert readiness["decision_level"] == "assisted_review"
    assert readiness["evidence_mode"] == "partial"
    assert readiness["trust_level"] == "media"
    assert "Ejecuciones combinadas" in readiness["blocking_reasons"]
    assert any("Selecciona una sola ejecucion" in action for action in readiness["required_actions"])
    assert any("combina varias ejecuciones" in warning for warning in readiness["warnings"])


@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
def test_operational_readiness_reports_governed_semantic_dictionary(load_configured_semantic_variables_mock) -> None:
    load_configured_semantic_variables_mock.return_value = [
        {"name": "affected_service", "label": "Servicio afectado", "role": "business"},
        {"name": "no_of_reassignments", "label": "Cantidad de reasignaciones", "role": "metric"},
    ]

    readiness = _operational_readiness(
        run_ids=["run-governed"],
        evidence_by_run={
            "run-governed": pd.DataFrame([{"incident_id": "INC001", "affected_service": "App"}]),
        },
        evidence_summary={"records_count": 1},
        semantic_variables=[
            {"name": "affected_service", "role": "business"},
            {"name": "no_of_reassignments", "role": "metric"},
        ],
        insights=[{"id": "insight-1", "run_id": "run-governed"}],
    )

    assert readiness["semantic_dictionary_configured"] is True
    assert readiness["semantic_dictionary_total"] == 2
    assert readiness["semantic_dictionary_configured_count"] == 2
    assert readiness["trust_level"] == "alta"
    assert readiness["expert_message"]


@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
def test_operational_readiness_marks_partial_context_as_assisted_review(
    load_configured_semantic_variables_mock,
) -> None:
    load_configured_semantic_variables_mock.return_value = []

    readiness = _operational_readiness(
        run_ids=["run-partial"],
        evidence_by_run={"run-partial": pd.DataFrame()},
        evidence_summary={"records_count": 0},
        semantic_variables=[{"name": "affected_service", "role": "business"}],
        insights=[{"id": "insight-1", "run_id": "run-partial"}],
    )

    assert readiness["status"] == "interpretive"
    assert readiness["decision_level"] == "assisted_review"
    assert readiness["evidence_mode"] == "partial"
    assert readiness["trust_level"] == "media"
    assert readiness["evidence_materialized"] is False
    assert "Sin evidencias reales materializadas" in readiness["blocking_reasons"]
    assert any("Materializa evidencias" in action for action in readiness["required_actions"])
    assert any("evidencias" in warning for warning in readiness["warnings"])
    assert "orientar" in readiness["functional_message"]
    assert readiness["recommended_next_step"]


@patch("app.services.conversation.dashboard_spec.load_configured_semantic_variables")
def test_operational_readiness_marks_empty_context_as_limited(
    load_configured_semantic_variables_mock,
) -> None:
    load_configured_semantic_variables_mock.return_value = []

    readiness = _operational_readiness(
        run_ids=[],
        evidence_by_run={},
        evidence_summary={"records_count": 0},
        semantic_variables=[],
        insights=[],
    )

    assert readiness["status"] == "limited"
    assert readiness["run_scope"] == "empty"
    assert readiness["decision_level"] == "interpretive"
    assert readiness["evidence_mode"] == "interpretive"
    assert readiness["trust_level"] == "baja"
    assert readiness["evidence_materialized"] is False
    assert "Sin evidencias reales materializadas" in readiness["blocking_reasons"]
    assert "Sin hallazgos guardados" in readiness["blocking_reasons"]
    assert "Ejecuta el analisis" in readiness["recommended_next_step"]
    assert readiness["required_actions"]


def test_dashboard_spec_contract_defaults_keep_operational_fields() -> None:
    spec = ConversationDashboardSpec.model_validate({})

    assert spec.schema_version == "conversation-dashboard/v1"
    assert spec.operational_readiness.evidence_mode == "interpretive"
    assert spec.operational_readiness.trust_level == "baja"
    assert spec.operational_readiness.blocking_reasons == []
    assert spec.operational_readiness.required_actions == []
    assert spec.operational_readiness.semantic_dictionary_active_count == 0
    assert spec.operational_readiness.semantic_dictionary_inactive_count == 0


def test_semantic_dictionary_status_exposes_governance_metadata() -> None:
    status = semantic_dictionary_status(
        {
            "affected_service": {"label": "Servicio afectado", "role": "business"},
            "priority": {"label": "Prioridad", "role": "business"},
        }
    )

    assert status["configurable"] is True
    assert status["env_var"] == "CONVERSATION_SEMANTIC_DICTIONARY_PATH"
    assert status["scope"] in {"default_file", "environment_file"}
    assert "source" in status
    assert "writable" in status


def test_semantic_dictionary_can_be_governed_from_config_file(tmp_path, monkeypatch) -> None:
    dictionary_path = tmp_path / "semantic_dictionary.json"
    monkeypatch.setenv("CONVERSATION_SEMANTIC_DICTIONARY_PATH", str(dictionary_path))
    reload_semantic_dictionary()

    try:
        result = save_configured_semantic_variables(
            [
                {
                    "name": "No Of Reassignments",
                    "aliases": ["reassignments"],
                    "label": "Cantidad de reasignaciones",
                    "role": "metric",
                    "type": "numeric",
                    "can_chart": True,
                    "avoid_as_metric": False,
                    "avoid_as_dimension": False,
                    "enabled_profiles": ["funcional", "experto"],
                    "domain": "incidencias-it",
                    "owner": "mesa-servicio",
                    "version": "2026.07",
                    "max_cardinality": 100,
                    "max_null_ratio": 0.4,
                    "source": "catalogo-it",
                    "confidence": "alta",
                    "active": True,
                },
                {
                    "name": "No_Of_Reassignments",
                    "label": "Reasignaciones duplicadas",
                    "role": "business",
                },
                {
                    "name": "cluster_label",
                    "label": "Grupo tecnico",
                    "role": "not-a-role",
                    "type": "not-a-type",
                    "avoid_as_metric": True,
                    "confidence": "invalida",
                },
                {
                    "name": "affected_service",
                    "label": "Servicio afectado desactivado",
                    "role": "business",
                    "type": "categorical",
                    "active": False,
                    "source": "gobierno-proyecto",
                },
            ]
        )
        entries = load_configured_semantic_variables()
        status = semantic_dictionary_status({"affected_service": {"label": "Servicio", "role": "business"}})
        governed_dictionary = get_semantic_dictionary(
            {"affected_service": {"label": "Servicio", "role": "business"}}
        )

        assert result["total"] == 3
        assert result["active_total"] == 2
        assert result["inactive_total"] == 1
        assert dictionary_path.exists()
        assert entries[0]["label"] == "Cantidad de reasignaciones"
        assert entries[0]["source"] == "catalogo-it"
        assert entries[0]["confidence"] == "alta"
        assert entries[0]["active"] is True
        assert entries[0]["avoid_as_dimension"] is False
        assert entries[0]["enabled_profiles"] == ["funcional", "experto"]
        assert entries[0]["domain"] == "incidencias-it"
        assert entries[0]["owner"] == "mesa-servicio"
        assert entries[0]["version"] == "2026.07"
        assert entries[0]["max_cardinality"] == 100
        assert entries[0]["max_null_ratio"] == 0.4
        assert entries[1]["role"] == "unknown"
        assert entries[1]["type"] == ""
        assert entries[1]["avoid_as_metric"] is True
        assert entries[1]["confidence"] == "media"
        assert entries[2]["active"] is False
        assert "affected_service" not in governed_dictionary
        assert governed_dictionary["reassignments"]["label"] == "Cantidad de reasignaciones"
        assert status["scope"] == "environment_file"
        assert status["governed"] is True
        assert status["configured_total"] == 3
        assert status["active_configured_total"] == 2
        assert status["inactive_configured_total"] == 1
    finally:
        reload_semantic_dictionary()


def test_semantic_dictionary_can_be_governed_per_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR", str(tmp_path))
    reload_semantic_dictionary()

    try:
        global_status = semantic_dictionary_status({"affected_service": {"label": "Servicio", "role": "business"}})
        project_result = save_configured_semantic_variables(
            [
                {
                    "name": "custom_business_axis",
                    "label": "Eje funcional del proyecto",
                    "role": "business",
                    "type": "categorical",
                    "can_chart": True,
                }
            ],
            project_id="project-123",
        )
        project_entries = load_configured_semantic_variables(project_id="project-123")
        project_status = semantic_dictionary_status(
            {"affected_service": {"label": "Servicio", "role": "business"}},
            project_id="project-123",
        )

        assert global_status["scope"] in {"default_file", "environment_file"}
        assert project_result["scope"] == "project_file"
        assert project_result["project_id"] == "project-123"
        assert project_entries[0]["label"] == "Eje funcional del proyecto"
        assert project_status["scope"] == "project_file"
        assert project_status["project_id"] == "project-123"
        assert project_status["governed"] is True
    finally:
        reload_semantic_dictionary()


def test_operational_readiness_flags_project_dictionary_without_active_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR", str(tmp_path))
    reload_semantic_dictionary()

    try:
        save_configured_semantic_variables(
            [
                {
                    "name": "affected_service",
                    "label": "Servicio afectado",
                    "role": "business",
                    "type": "categorical",
                    "active": False,
                }
            ],
            project_id="project-without-active-dictionary",
        )

        readiness = _operational_readiness(
            run_ids=["run-active"],
            evidence_by_run={"run-active": pd.DataFrame([{"incident_id": "INC001"}])},
            evidence_summary={"records_count": 1},
            semantic_variables=[{"name": "affected_service", "role": "business"}],
            insights=[{"id": "insight-1"}],
            project_id="project-without-active-dictionary",
        )

        assert readiness["semantic_dictionary_configured"] is True
        assert readiness["semantic_dictionary_active_count"] == 0
        assert readiness["semantic_dictionary_inactive_count"] == 1
        assert readiness["trust_level"] == "media"
        assert "Diccionario semantico sin variables activas" in readiness["blocking_reasons"]
        assert any("Activa variables semanticas" in action for action in readiness["required_actions"])
    finally:
        reload_semantic_dictionary()

@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_uses_project_semantic_dictionary_for_business_axis(
    load_run_evidences_mock,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR", str(tmp_path))
    reload_semantic_dictionary()
    try:
        save_configured_semantic_variables(
            [
                {
                    "name": "custom_business_axis",
                    "label": "Eje funcional del proyecto",
                    "role": "business",
                    "type": "categorical",
                    "can_chart": True,
                }
            ],
            project_id="project-chart",
        )
        load_run_evidences_mock.return_value = pd.DataFrame(
            [
                {"incident_id": "INC001", "custom_business_axis": "Aplicacion", "preview": "Aplicacion"},
                {"incident_id": "INC002", "custom_business_axis": "Base de datos", "preview": "Base"},
                {"incident_id": "INC003", "custom_business_axis": "Aplicacion", "preview": "Aplicacion"},
            ]
        )

        response = build_conversation_chart_data(
            run_id="run-project",
            project_id="project-chart",
            visualization={
                "id": "viz-project-axis",
                "title": "Eje funcional por volumen",
                "chart_type": "bar",
                "x": "custom_business_axis",
                "metric": "count",
            },
            limit=5,
            evidence_limit=5,
        )

        assert response.run_id == "run-project"
        assert response.validation.chart_is_buildable is True
        assert response.x == "custom_business_axis"
        assert any(item.name == "custom_business_axis" and item.label == "Eje funcional del proyecto" for item in response.semantic_dictionary)
        assert response.series[0].key == "Aplicacion"
        assert response.samples_by_key["Aplicacion"]
    finally:
        reload_semantic_dictionary()


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_does_not_use_blocked_semantic_dimension(
    load_run_evidences_mock,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR", str(tmp_path))
    reload_semantic_dictionary()
    try:
        save_configured_semantic_variables(
            [
                {
                    "name": "custom_business_axis",
                    "label": "Identificador tecnico no decisional",
                    "role": "business",
                    "type": "categorical",
                    "can_chart": True,
                    "avoid_as_dimension": True,
                    "enabled_profiles": ["experto"],
                },
                {
                    "name": "affected_service",
                    "label": "Servicio afectado",
                    "role": "business",
                    "type": "categorical",
                    "can_chart": True,
                    "enabled_profiles": ["funcional", "experto"],
                },
            ],
            project_id="project-chart-blocked-axis",
        )
        load_run_evidences_mock.return_value = pd.DataFrame(
            [
                {
                    "incident_id": "INC001",
                    "custom_business_axis": "ID-1",
                    "affected_service": "Aplicacion",
                    "preview": "Affected_Service=Aplicacion",
                },
                {
                    "incident_id": "INC002",
                    "custom_business_axis": "ID-2",
                    "affected_service": "Base de datos",
                    "preview": "Affected_Service=Base de datos",
                },
                {
                    "incident_id": "INC003",
                    "custom_business_axis": "ID-3",
                    "affected_service": "Aplicacion",
                    "preview": "Affected_Service=Aplicacion",
                },
            ]
        )

        response = build_conversation_chart_data(
            run_id="run-project",
            project_id="project-chart-blocked-axis",
            visualization={
                "id": "viz-blocked-axis",
                "title": "Servicios por volumen",
                "chart_type": "bar",
                "x": "custom_business_axis",
                "metric": "count",
            },
            limit=5,
            evidence_limit=5,
        )

        assert response.validation.chart_is_buildable is True
        assert response.x == "affected_service"
        assert response.x != "custom_business_axis"
        assert response.validation.chose_interpretable_variables is True
        assert any("se uso Servicio afectado" in warning for warning in response.validation.warnings)
    finally:
        reload_semantic_dictionary()


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


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_without_materialized_evidence_is_not_operational(load_run_evidences_mock) -> None:
    load_run_evidences_mock.return_value = pd.DataFrame()

    response = build_conversation_chart_data(
        run_id="run-empty",
        visualization={
            "id": "viz-empty",
            "title": "Vista sin evidencia real",
            "chart_type": "bar",
            "x": "affected_service",
            "metric": "count",
        },
        limit=5,
        evidence_limit=5,
    )

    load_run_evidences_mock.assert_called_once_with("run-empty")
    assert response.schema_version == "conversation-chart-data/v1"
    assert response.run_id == "run-empty"
    assert response.series == []
    assert response.evidence_samples == []
    assert response.validation.chart_is_buildable is False
    assert response.validation.uses_real_data is False
    assert response.validation.operation_ready is False
    assert response.validation.requires_data is True
    assert any("evidencias materializadas" in warning for warning in response.validation.warnings)


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_returns_drilldown_samples_by_segment(load_run_evidences_mock) -> None:
    load_run_evidences_mock.return_value = pd.DataFrame(
        [
            {
                "incident_id": "INC001",
                "affected_service": "App",
                "priority": "Alta",
                "preview": "Numero=INC001 | Affected_Service=App | Priority=Alta",
            },
            {
                "incident_id": "INC002",
                "affected_service": "App",
                "priority": "Media",
                "preview": "Numero=INC002 | Affected_Service=App | Priority=Media",
            },
            {
                "incident_id": "INC003",
                "affected_service": "DB",
                "priority": "Alta",
                "preview": "Numero=INC003 | Affected_Service=DB | Priority=Alta",
            },
        ]
    )

    response = build_conversation_chart_data(
        run_id="run-1",
        visualization={
            "id": "viz-service",
            "title": "Servicios por volumen",
            "chart_type": "bar",
            "x": "affected_service",
            "metric": "count",
        },
        limit=5,
        evidence_limit=5,
    )

    assert response.schema_version == "conversation-chart-data/v1"
    load_run_evidences_mock.assert_called_once_with("run-1")
    assert response.validation.chart_is_buildable is True
    assert response.series[0].key == "App"
    assert response.series[0].count == 2
    assert response.series[0].filter == {"column": "affected_service", "operator": "eq", "value": "App"}
    assert len(response.samples_by_key["App"]) == 2
    assert {sample.incident_id for sample in response.samples_by_key["App"]} == {"INC001", "INC002"}


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_is_scoped_per_run_when_execution_changes(load_run_evidences_mock) -> None:
    def load_for_run(run_id: str) -> pd.DataFrame:
        if run_id == "run-a":
            return pd.DataFrame(
                [
                    {
                        "incident_id": "INC-A1",
                        "affected_service": "App A",
                        "priority": "Alta",
                        "preview": "Numero=INC-A1 | Affected_Service=App A | Priority=Alta",
                    },
                    {
                        "incident_id": "INC-A2",
                        "affected_service": "App A",
                        "priority": "Alta",
                        "preview": "Numero=INC-A2 | Affected_Service=App A | Priority=Alta",
                    },
                ]
            )
        if run_id == "run-b":
            return pd.DataFrame(
                [
                    {
                        "incident_id": "INC-B1",
                        "affected_service": "DB B",
                        "priority": "Baja",
                        "preview": "Numero=INC-B1 | Affected_Service=DB B | Priority=Baja",
                    }
                ]
            )
        return pd.DataFrame()

    load_run_evidences_mock.side_effect = load_for_run
    visualization = {
        "id": "viz-service",
        "title": "Servicios por volumen",
        "chart_type": "bar",
        "x": "affected_service",
        "metric": "count",
    }

    response_a = build_conversation_chart_data(
        run_id="run-a",
        visualization=visualization,
        limit=5,
        evidence_limit=5,
    )
    response_b = build_conversation_chart_data(
        run_id="run-b",
        visualization=visualization,
        limit=5,
        evidence_limit=5,
    )

    assert [call.args[0] for call in load_run_evidences_mock.call_args_list] == ["run-a", "run-b"]
    assert response_a.run_id == "run-a"
    assert response_b.run_id == "run-b"
    assert response_a.series[0].key == "App A"
    assert response_b.series[0].key == "DB B"
    assert {sample.incident_id for sample in response_a.samples_by_key["App A"]} == {
        "INC-A1",
        "INC-A2",
    }
    assert {sample.incident_id for sample in response_b.samples_by_key["DB B"]} == {"INC-B1"}
    assert "INC-B1" not in {sample.incident_id for sample in response_a.samples_by_key["App A"]}


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_uses_clear_alternative_when_llm_dimension_is_missing(load_run_evidences_mock) -> None:
    load_run_evidences_mock.return_value = pd.DataFrame(
        [
            {"incident_id": "INC001", "affected_service": "App", "preview": "Affected_Service=App"},
            {"incident_id": "INC002", "affected_service": "DB", "preview": "Affected_Service=DB"},
        ]
    )

    response = build_conversation_chart_data(
        run_id="run-1",
        visualization={
            "id": "viz-invented-axis",
            "title": "Servicios por volumen de incidencias",
            "chart_type": "bar",
            "x": "Incidencias",
            "metric": "count",
        },
        limit=5,
        evidence_limit=5,
    )

    assert response.validation.chart_is_buildable is True
    assert response.x == "affected_service"
    assert any("se uso" in warning for warning in response.validation.warnings)


@patch("app.services.conversation.chart_data.load_run_evidences")
def test_chart_data_sanitizes_non_finite_metric_values(load_run_evidences_mock) -> None:
    load_run_evidences_mock.return_value = pd.DataFrame(
        [
            {
                "incident_id": "INC001",
                "affected_service": "App",
                "avg_resolution_hours": float("nan"),
                "preview": "Numero=INC001 | Affected_Service=App | Duracion=NaN",
            },
            {
                "incident_id": "INC002",
                "affected_service": "App",
                "avg_resolution_hours": float("inf"),
                "preview": "Numero=INC002 | Affected_Service=App | Duracion=Infinity",
            },
            {
                "incident_id": "INC003",
                "affected_service": "DB",
                "avg_resolution_hours": 4.0,
                "preview": "Numero=INC003 | Affected_Service=DB | Duracion=4",
            },
        ]
    )

    response = build_conversation_chart_data(
        run_id="run-1",
        visualization={
            "id": "viz-resolution",
            "title": "Tiempo de resolucion por servicio",
            "chart_type": "bar",
            "x": "affected_service",
            "metric": "avg_resolution_hours",
            "aggregation": "mean",
        },
        limit=5,
        evidence_limit=5,
    )

    assert response.validation.chart_is_buildable is True
    assert all(math.isfinite(point.value) for point in response.series)
    for samples in response.samples_by_key.values():
        assert all(sample.metric_value is None or math.isfinite(sample.metric_value) for sample in samples)


def test_chart_data_error_response_is_controlled() -> None:
    response = build_conversation_chart_error_response(
        run_id="run-1",
        visualization={
            "id": "viz-failed",
            "title": "Mapa de clusters",
            "chart_type": "scatter",
            "x": "x",
            "metric": "count",
        },
        warning="No se pudo calcular el grafico real.",
    )

    assert response.schema_version == "conversation-chart-data/v1"
    assert response.validation.chart_is_buildable is False
    assert response.validation.requires_data is True
    assert "backend_error" in response.validation.missing
    assert response.validation.warnings == ["No se pudo calcular el grafico real."]
