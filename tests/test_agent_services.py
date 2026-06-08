import json

import pandas as pd

from app.services.agent_sampling import sample_cluster_records
from app.services.agent_service import run_interpretation_agent, run_strategy_agent
from app.services.agent_traceability import TraceCollector


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

    result = run_strategy_agent(
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
    assert tracer.to_frame().iloc[0]["agent_name"] == "strategy_agent"


def test_interpretation_agent_uses_samples_and_records_trace():
    tracer = TraceCollector(run_id="run-1")

    samples, insights = run_interpretation_agent(
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
    first = insights[insights["cluster_label"] == 0].iloc[0]
    assert "Security" in first["summary"]
    assert json.loads(first["sample_evidence_ids"]) == ["IT-1", "IT-2"]
    assert tracer.to_frame()["agent_name"].tolist() == ["interpretation_agent", "interpretation_agent"]
