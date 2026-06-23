from app.services.runs.duckdb_store import (
    clear_run_data,
    list_selected_insights,
    save_selected_insight,
    save_selected_insights_bulk,
)


def test_save_selected_insights_bulk_persists_all():
    run_id = "test-insight-batch-run"
    user_id = "user-batch-1"
    clear_run_data(run_id)

    saved = save_selected_insights_bulk(
        run_id,
        [
            {
                "id": "insight-a",
                "title": "Cluster 0",
                "description": "Alto SLA",
            },
            {
                "id": "insight-b",
                "title": "Cluster 1",
                "description": "Riesgo medio",
            },
        ],
        user_id=user_id,
    )

    assert saved == 2
    listed = list_selected_insights(run_id=run_id, user_id=user_id)
    assert {item["id"] for item in listed} == {"insight-a", "insight-b"}


def test_save_selected_insight_delegates_to_bulk():
    run_id = "test-insight-single-run"
    user_id = "user-single-1"
    clear_run_data(run_id)

    save_selected_insight(
        run_id,
        {
            "id": "insight-single",
            "title": "Servicio VPN",
            "description": "Volumen alto",
        },
        user_id=user_id,
    )

    listed = list_selected_insights(run_id=run_id, user_id=user_id)
    assert len(listed) == 1
    assert listed[0]["id"] == "insight-single"

    cleared = clear_run_data(run_id)
    assert cleared["selected_insights"] >= 1
