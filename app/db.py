import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine
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


class SelectedInsight(Base):
    __tablename__ = "selected_insights"

    selected_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    id: Mapped[str] = mapped_column(String(160))
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    selected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    metric_label: Mapped[str] = mapped_column(String(120))
    metric_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dimension: Mapped[str | None] = mapped_column(String(120), nullable=True)
    filter_kind: Mapped[str | None] = mapped_column(String(120), nullable=True)
    filter_value: Mapped[str | None] = mapped_column(String(120), nullable=True)


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


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


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
            if metrics.get("davies_bouldin") is not None
            else None
        ),
        result_json=json.dumps(result, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run_to_detail(row: AnalysisRun) -> dict:
    result = json.loads(row.result_json)
    sil = float(row.silhouette) if row.silhouette is not None else None
    dbi = float(row.davies_bouldin) if row.davies_bouldin is not None else None
    return {
        "id": row.id,
        "created_at": row.created_at,
        "modality": row.modality,
        "reduction_method": row.reduction_method,
        "seed": row.seed,
        "n_samples": row.n_samples,
        "outliers_count": row.outliers_count,
        "metrics": {
            "silhouette": sil,
            "davies_bouldin": dbi,
        },
        "result": result,
    }
