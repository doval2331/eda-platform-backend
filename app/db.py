import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, text
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    modality: Mapped[str] = mapped_column(String(32))
    reduction_method: Mapped[str] = mapped_column(String(16))
    seed: Mapped[int] = mapped_column(Integer)
    n_samples: Mapped[int] = mapped_column(Integer)
    outliers_count: Mapped[int] = mapped_column(Integer)
    silhouette: Mapped[str | None] = mapped_column(String(32), nullable=True)
    davies_bouldin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    result_json: Mapped[str] = mapped_column(Text)
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    project_name: Mapped[str | None] = mapped_column(String(200), nullable=True)


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


def _ensure_analysis_run_columns() -> None:
    """Añade columnas nuevas sin migración formal (SQLite / PostgreSQL)."""
    url = get_settings().database_url
    additions = {
        "project_id": "VARCHAR(36)",
        "source_type": "VARCHAR(32)",
        "project_name": "VARCHAR(200)",
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
        for column, col_type in additions.items():
            if column not in existing:
                conn.execute(
                    text(f"ALTER TABLE analysis_runs ADD COLUMN {column} {col_type}")
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
        result_json=json.dumps(result, ensure_ascii=False),
        project_id=payload.get("project_id"),
        source_type=payload.get("source_type"),
        project_name=payload.get("project_name"),
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
        "project_name": row.project_name,
    }
