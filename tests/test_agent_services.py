import json

import pandas as pd

from app.services.agent_sampling import sample_cluster_records
from app.services.agent_service import run_interpretation_agent, run_strategy_agent
from app.services.agent_traceability import TraceCollector
from app.services.duckdb_store import (
    append_agent_decisions,
    list_agent_cluster_insights,
    list_agent_decisions,
    list_agent_recommendations,
    save_agent_cluster_insights,
    save_agent_recommendations,
)


def _evidences() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "evidence_index": 0,
                "evidence_id": "IT-1",
                "preview": "VPN login bloqueado por acceso sospechoso",
                "cluster_label": 0,
                "category": "Security",
                "severity": "critical",
                "affected_service": "VPN",
                "assignment_group": "SecOps",
                "sla_breached": True,
                "resolution_minutes": 1200,
                "business_impact_score": 95,
            },
            {
                "run_id": "run-1",
                "evidence_index": 1,
                "evidence_id": "IT-2",
                "preview": "VPN error de login recurrente",
                "cluster_label": 0,
                "category": "Security",
                "severity": "high",
                "affected_service": "VPN",
                "assignment_group": "SecOps",
                "sla_breached": True,
                "resolution_minutes": 900,
                "business_impact_score": 80,
            },
            {
                "run_id": "run-1",
                "evidence_index": 2,
                "evidence_id": "IT-3",
                "preview": "Backup restore lento en servidores",
                "cluster_label": 1,
                "category": "Backup",
                "severity": "medium",
                "affected_service": "Backup Servidores",
                "assignment_group": "CloudOps",
                "sla_breached": False,
                "resolution_minutes": 300,
                "business_impact_score": 55,
            },
        ]
    )


def test_sample_cluster_records_is_reproducible_and_bounded():
    first = sample_cluster_records(_evidences(), sample_size=1, random_state=7, criteria="mixed")
    second = sample_cluster_records(_evidences(), sample_size=1, random_state=7, criteria="mixed")

    assert first["evidence_id"].tolist() == second["evidence_id"].tolist()
    assert first.groupby("cluster_label").size().max() == 1
    assert "selection_score" in first.columns


def test_strategy_agent_returns_recommendations_and_trace():
    tracer = TraceCollector(run_id="run-1")

    result, meta = run_strategy_agent(
        run_id="run-1",
        evidences=_evidences(),
        metrics={"n_clusters": 2, "silhouette": 0.4},
        sample_size=2,
        sample_criteria="priority",
        model_name="deterministic-local",
        tracer=tracer,
    )

    assert not result.empty
    assert {"strategy_id", "recommendation", "trace_id"}.issubset(result.columns)
    assert meta.llm_used is False
    assert meta.model_name == "deterministic-local"
    assert tracer.to_frame().iloc[0]["agent_name"] == "strategy_agent"


def test_interpretation_agent_uses_samples_and_records_trace():
    tracer = TraceCollector(run_id="run-1")

    samples, insights, meta = run_interpretation_agent(
        run_id="run-1",
        evidences=_evidences(),
        sample_size=2,
        sample_criteria="priority",
        random_state=42,
        model_name="deterministic-local",
        tracer=tracer,
    )

    assert not samples.empty
    assert not insights.empty
    assert meta.llm_used is False
    first = insights[insights["cluster_label"] == 0].iloc[0]
    assert "Security" in first["summary"]
    assert json.loads(first["sample_evidence_ids"]) == ["IT-1", "IT-2"]
    assert tracer.to_frame()["agent_name"].tolist() == ["interpretation_agent", "interpretation_agent"]


def test_agent_results_persist_and_list():
    run_id = "test-persist-run"
    recommendations = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "strategy_id": "feature_mix",
                "strategy_type": "segmentation",
                "recommendation": "Usar variables operativas",
                "justification": "Test",
                "variables_used": "[]",
                "metric_or_criterion": "perfil",
                "priority": "high",
                "created_at": "2026-01-01T00:00:00+00:00",
                "trace_id": "trace-1",
            }
        ]
    )
    insights = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "cluster_insight_id": f"{run_id}-cluster-0",
                "cluster_label": 0,
                "cluster_name": "cluster_0",
                "summary": "Cluster de prueba",
                "main_characteristics": "test",
                "highlighted_variables": "[]",
                "possible_causes": "causa test",
                "recommendations": "recomendacion test",
                "business_conclusion": "conclusion test",
                "sample_evidence_ids": '["IT-1"]',
                "sample_size": 1,
                "risk_level": "high",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "trace_id": "trace-2",
            }
        ]
    )
    traces = pd.DataFrame(
        [
            {
                "trace_id": "trace-1",
                "run_id": run_id,
                "agent_name": "strategy_agent",
                "decision_type": "segmentation_strategy",
                "model_name": "deterministic-local",
                "parameters": "{}",
                "variables_used": "[]",
                "input_artifacts": "[]",
                "prompt": "prompt test",
                "response": "response test",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )

    save_agent_recommendations(run_id, recommendations)
    save_agent_cluster_insights(run_id, insights)
    append_agent_decisions(traces)

    listed_recs = list_agent_recommendations(run_id)
    listed_insights = list_agent_cluster_insights(run_id)
    listed_traces = list_agent_decisions(run_id)

    assert len(listed_recs) == 1
    assert listed_recs[0]["strategy_id"] == "feature_mix"
    assert len(listed_insights) == 1
    assert listed_insights[0]["cluster_label"] == 0
    assert len(listed_traces) == 1
    assert listed_traces[0]["agent_name"] == "strategy_agent"
