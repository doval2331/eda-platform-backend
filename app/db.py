import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    nombre: Mapped[str] = mapped_column(String(120))
    activo: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ultimo_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    strategy: Mapped[str] = mapped_column(String(32), default="per_source")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProjectSource(Base):
    __tablename__ = "project_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    dataset_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    modality: Mapped[str] = mapped_column(String(32))
    reduction_method: Mapped[str] = mapped_column(String(16))
    seed: Mapped[int] = mapped_column(Integer)
    n_samples: Mapped[int] = mapped_column(Integer)
    outliers_count: Mapped[int] = mapped_column(Integer)
    silhouette: Mapped[str | None] = mapped_column(String(32), nullable=True)
    davies_bouldin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    n_clusters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    noise_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_json: Mapped[str] = mapped_column(Text)
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    project_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


def _engine():
    url = get_settings().database_url
    connect_args = {}
    kwargs = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        kwargs["pool_pre_ping"] = True
    return create_engine(url, connect_args=connect_args, **kwargs)


engine = _engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


RESULT_JSON_SAMPLE_LIMIT = 2_000


def compact_pipeline_result_for_storage(result: dict) -> dict:
    """Keep run history useful without storing huge visualization arrays in SQL."""
    metadata = result.get("metadata") or []
    points = result.get("X_2d") or []
    labels = result.get("cluster_labels") or []
    limit = min(RESULT_JSON_SAMPLE_LIMIT, len(metadata) or len(points) or len(labels))
    compact = dict(result)
    compact["X_2d"] = points[:limit]
    compact["cluster_labels"] = labels[:limit]
    compact["metadata"] = metadata[:limit]
    compact["storage"] = {
        "compact": True,
        "sample_limit": RESULT_JSON_SAMPLE_LIMIT,
        "stored_points": limit,
        "total_points": max(len(metadata), len(points), len(labels)),
        "reason": "Historial SQL compacto; las evidencias completas se materializan en DuckDB.",
    }
    return compact

def _ensure_analysis_run_columns() -> None:
    """Add new columns without a formal migration (SQLite / PostgreSQL)."""
    url = get_settings().database_url
    additions = {
        "n_clusters": "INTEGER",
        "noise_pct": "FLOAT",
        "project_id": "VARCHAR(36)",
        "source_type": "VARCHAR(32)",
        "source_id": "VARCHAR(36)",
        "source_name": "VARCHAR(200)",
        "project_name": "VARCHAR(200)",
        "dataset_id": "VARCHAR(36)",
    }
    with engine.begin() as conn:
        if url.startswith("sqlite"):
            rows = conn.execute(text("PRAGMA table_info(analysis_runs)")).fetchall()
            existing = {row[1] for row in rows}
        else:
            rows = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'analysis_runs'"
                )
            ).fetchall()
            existing = {row[0] for row in rows}
        added_columns: set[str] = set()
        for column, col_type in additions.items():
            if column not in existing:
                conn.execute(
                    text(f"ALTER TABLE analysis_runs ADD COLUMN {column} {col_type}")
                )
                added_columns.add(column)
        if url.startswith("sqlite") and {"n_clusters", "noise_pct"} & added_columns:
            conn.execute(
                text(
                    "UPDATE analysis_runs "
                    "SET n_clusters = CAST(json_extract(result_json, '$.metrics.n_clusters') AS INTEGER), "
                    "noise_pct = CAST(json_extract(result_json, '$.metrics.noise_pct') AS FLOAT) "
                    "WHERE n_clusters IS NULL AND result_json IS NOT NULL"
                )
            )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_analysis_runs_created_at ON analysis_runs (created_at)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_analysis_runs_history_summary "
                "ON analysis_runs ("
                "created_at, id, modality, reduction_method, seed, n_samples, "
                "outliers_count, silhouette, davies_bouldin, n_clusters, noise_pct, "
                "project_id, project_name, source_type, source_id, source_name, dataset_id"
                ")"
            )
        )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_analysis_run_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_run(db: Session, *, payload: dict) -> AnalysisRun:
    result = payload["result"]
    result_storage = payload.get("result_storage") or result
    metrics = result["metrics"]
    row = AnalysisRun(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        modality=payload["modality"],
        reduction_method=payload["reduction_method"],
        seed=payload["seed"],
        n_samples=payload["n_samples"],
        outliers_count=result["outliers_count"],
        silhouette=(
            str(metrics["silhouette"]) if metrics.get("silhouette") is not None else None
        ),
        davies_bouldin=(
            str(metrics["davies_bouldin"])
            if metrics.get("davies_bouldin") is not None else None
        ),
        n_clusters=metrics.get("n_clusters"),
        noise_pct=metrics.get("noise_pct"),
        result_json=json.dumps(result_storage, ensure_ascii=False),
        project_id=payload.get("project_id"),
        source_type=payload.get("source_type"),
        source_id=payload.get("source_id"),
        source_name=payload.get("source_name"),
        project_name=payload.get("project_name"),
        dataset_id=payload.get("dataset_id"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run_to_detail(row: AnalysisRun) -> dict:
    result = json.loads(row.result_json)
    metrics = dict(result.get("metrics") or {})
    if metrics.get("silhouette") is None and row.silhouette is not None:
        metrics["silhouette"] = float(row.silhouette)
    if metrics.get("davies_bouldin") is None and row.davies_bouldin is not None:
        metrics["davies_bouldin"] = float(row.davies_bouldin)
    return {
        "id": row.id,
        "created_at": row.created_at,
        "modality": row.modality,
        "reduction_method": row.reduction_method,
        "seed": row.seed,
        "n_samples": row.n_samples,
        "outliers_count": row.outliers_count,
        "metrics": metrics,
        "result": result,
        "project_id": row.project_id,
        "source_type": row.source_type,
        "source_id": row.source_id,
        "source_name": row.source_name,
        "project_name": row.project_name,
        "dataset_id": row.dataset_id,
    }
