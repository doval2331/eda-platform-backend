"""Tests for dataset full profile builder."""

from app.services.datasets.dataset_profile import (
    _business_breakdowns,
    _correlation_pairs,
    _quality_alerts,
    build_dataset_full_profile,
)


def test_correlation_pairs_detects_strong_relation():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 4, 3, 2, 1]})
    pairs = _correlation_pairs(df, ["a", "b", "c"], threshold=0.5)
    labels = {(pair["column_a"], pair["column_b"]) for pair in pairs}
    assert ("a", "b") in labels
    assert ("a", "c") in labels


def test_quality_alerts_flags_high_nulls():
    alerts = _quality_alerts(
        __import__("pandas").DataFrame({"x": [None, None, 1, 2]}),
        numeric=["x"],
        columns=[{"name": "x", "null_pct": 50}],
    )
    assert any("nulos" in alert["message"].lower() for alert in alerts)


def test_quality_alerts_mark_sparse_optional_semantic_metric_as_info():
    alerts = _quality_alerts(
        __import__("pandas").DataFrame({"No_of_Related_Incidents": [None] * 99 + [1]}),
        numeric=["No_of_Related_Incidents"],
        columns=[{"name": "No_of_Related_Incidents", "null_pct": 99}],
    )
    alert = next(item for item in alerts if item.get("column") == "No_of_Related_Incidents")
    assert alert["level"] == "info"
    assert alert["reason"] == "low_coverage_optional_variable"
    assert "No se recomienda" in alert["message"]


def test_business_breakdowns_category_sla():
    import pandas as pd

    df = pd.DataFrame(
        {
            "categoria": ["Red", "Red", "App", "App", "App"],
            "sla_incumplido": [1, 0, 1, 1, 0],
        }
    )
    breakdowns = _business_breakdowns(df)
    assert "category_sla" in breakdowns
    red = next(item for item in breakdowns["category_sla"] if item["category"] == "Red")
    assert red["count"] == 2
    assert red["sla_breach_pct"] == 50.0
