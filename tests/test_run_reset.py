from datetime import datetime, timezone

from app.db import AnalysisRun, SessionLocal, init_db
from app.services.duckdb_store import clear_all_run_data, save_selected_insight
from app.services.run_reset import reset_all_runs


def test_reset_all_runs_clears_sql_and_duckdb():
    init_db()
    db = SessionLocal()
    try:
        db.add(
            AnalysisRun(
                id="test-reset-run",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                modality="it_ops",
                reduction_method="UMAP",
                seed=42,
                n_samples=10,
                outliers_count=1,
                silhouette="0.1",
                davies_bouldin="1.0",
                result_json='{"metrics": {}}',
            )
        )
        db.commit()
        save_selected_insight(
            run_id="test-reset-run",
            insight={
                "id": "insight-1",
                "title": "Test",
                "description": "desc",
            },
            user_id="user-1",
        )

        result = reset_all_runs(db)

        assert result["deleted_runs"] >= 1
        assert db.query(AnalysisRun).count() == 0
        cleared = clear_all_run_data()
        assert all(count == 0 for count in cleared.values())
    finally:
        db.close()
