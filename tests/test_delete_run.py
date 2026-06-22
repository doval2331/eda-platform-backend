from datetime import datetime, timezone

from app.db import AnalysisRun, SessionLocal, init_db
from app.services.runs.duckdb_store import clear_run_data, persist_run_detail, run_exists
from app.services.runs.run_reset import delete_run


def test_delete_single_run_removes_sql_and_duckdb():
    init_db()
    db = SessionLocal()
    run_id = "test-delete-run"
    try:
        detail = {
            "id": run_id,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "modality": "it_ops",
            "reduction_method": "UMAP",
            "seed": 42,
            "n_samples": 10,
            "outliers_count": 1,
            "metrics": {"n_clusters": 2},
            "result": {
                "X_2d": [[0.0, 0.0]],
                "cluster_labels": [0],
                "outliers_count": 0,
                "metrics": {"n_clusters": 1},
                "metadata": [
                    {
                        "id": "ev-1",
                        "preview": "preview",
                        "source": "it_ops",
                    }
                ],
            },
        }
        db.add(
            AnalysisRun(
                id=run_id,
                created_at=detail["created_at"],
                modality="it_ops",
                reduction_method="UMAP",
                seed=42,
                n_samples=10,
                outliers_count=1,
                silhouette="0.1",
                davies_bouldin="1.0",
                result_json='{"metrics": {"n_clusters": 1}, "cluster_labels": [0], "X_2d": [[0,0]], "outliers_count": 0, "metadata": []}',
            )
        )
        db.commit()
        persist_run_detail(detail)
        assert run_exists(run_id)

        result = delete_run(db, run_id)

        assert result["run_id"] == run_id
        assert db.get(AnalysisRun, run_id) is None
        assert not run_exists(run_id)
        cleared = clear_run_data(run_id)
        assert all(count == 0 for count in cleared.values())
    finally:
        db.close()
