"""Gestión de proyectos multifuente (escenarios de análisis)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db import Project, ProjectSource
from app.services.dataset_store import get_dataset_meta, save_text_upload, save_upload

CSV_SOURCE_TYPES = frozenset({"incidents", "change_mgmt", "software", "hardware"})
TEXT_SOURCE_TYPES = frozenset({"dictionary", "notes"})
ALL_SOURCE_TYPES = CSV_SOURCE_TYPES | TEXT_SOURCE_TYPES

SOURCE_TYPE_LABELS = {
    "incidents": "Incidencias principales",
    "change_mgmt": "Gestión del cambio",
    "software": "Problemas software",
    "hardware": "Problemas hardware",
    "dictionary": "Diccionario de datos",
    "notes": "Notas / transcripción",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _source_to_summary(row: ProjectSource) -> dict:
    meta = json.loads(row.meta_json or "{}")
    return {
        "id": row.id,
        "source_type": row.source_type,
        "filename": row.filename,
        "dataset_id": row.dataset_id,
        "n_rows": row.n_rows,
        "char_count": meta.get("char_count"),
    }


def _project_counts(sources: list[ProjectSource]) -> tuple[int, int, int]:
    csv_sources = [s for s in sources if s.source_type in CSV_SOURCE_TYPES]
    total_rows = sum(s.n_rows or 0 for s in csv_sources)
    return len(sources), len(csv_sources), total_rows


def _project_to_summary(row: Project, sources: list[ProjectSource]) -> dict:
    source_count, csv_count, total_rows = _project_counts(sources)
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description or "",
        "strategy": row.strategy,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "source_count": source_count,
        "csv_source_count": csv_count,
        "total_rows": total_rows,
    }


def create_project(
    db: Session,
    *,
    user_id: str,
    name: str,
    description: str = "",
    strategy: str = "per_source",
) -> dict:
    now = _now()
    row = Project(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=name.strip(),
        description=(description or "").strip(),
        strategy=strategy,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _project_to_summary(row, [])


def list_projects(db: Session, *, user_id: str, limit: int = 50) -> list[dict]:
    limit = min(max(1, limit), 100)
    rows = (
        db.query(Project)
        .filter(Project.user_id == user_id)
        .order_by(Project.updated_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for row in rows:
        sources = (
            db.query(ProjectSource)
            .filter(ProjectSource.project_id == row.id)
            .order_by(ProjectSource.created_at.asc())
            .all()
        )
        result.append(_project_to_summary(row, sources))
    return result


def get_project_or_404(db: Session, *, project_id: str, user_id: str) -> Project:
    row = db.get(Project, project_id)
    if row is None:
        raise LookupError("Proyecto no encontrado")
    if row.user_id != user_id:
        raise PermissionError("No tienes acceso a este proyecto")
    return row


def get_project_detail(db: Session, *, project_id: str, user_id: str) -> dict:
    row = get_project_or_404(db, project_id=project_id, user_id=user_id)
    sources = (
        db.query(ProjectSource)
        .filter(ProjectSource.project_id == row.id)
        .order_by(ProjectSource.created_at.asc())
        .all()
    )
    summary = _project_to_summary(row, sources)
    summary["sources"] = [_source_to_summary(s) for s in sources]
    return summary


def update_project(
    db: Session,
    *,
    project_id: str,
    user_id: str,
    name: str | None = None,
    description: str | None = None,
    strategy: str | None = None,
) -> dict:
    row = get_project_or_404(db, project_id=project_id, user_id=user_id)
    if name is not None:
        row.name = name.strip()
    if description is not None:
        row.description = description.strip()
    if strategy is not None:
        row.strategy = strategy
    row.updated_at = _now()
    db.commit()
    db.refresh(row)
    sources = (
        db.query(ProjectSource)
        .filter(ProjectSource.project_id == row.id)
        .all()
    )
    summary = _project_to_summary(row, sources)
    summary["sources"] = [_source_to_summary(s) for s in sources]
    return summary


def delete_project_source(
    db: Session,
    *,
    project_id: str,
    source_id: str,
    user_id: str,
) -> dict:
    get_project_or_404(db, project_id=project_id, user_id=user_id)
    row = db.get(ProjectSource, source_id)
    if row is None or row.project_id != project_id:
        raise LookupError("Fuente no encontrada")
    db.delete(row)
    project = db.get(Project, project_id)
    if project:
        project.updated_at = _now()
    db.commit()
    return get_project_detail(db, project_id=project_id, user_id=user_id)


def add_csv_source(
    db: Session,
    *,
    project_id: str,
    user_id: str,
    source_type: str,
    filename: str,
    content: bytes,
) -> dict:
    if source_type not in CSV_SOURCE_TYPES:
        raise ValueError(f"Tipo de fuente CSV no válido: {source_type}")

    project = get_project_or_404(db, project_id=project_id, user_id=user_id)

    existing = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.source_type == source_type,
        )
        .first()
    )
    if existing:
        db.delete(existing)

    meta = save_upload(user_id=user_id, filename=filename, content=content)
    now = _now()
    row = ProjectSource(
        id=str(uuid.uuid4()),
        project_id=project_id,
        source_type=source_type,
        dataset_id=meta["dataset_id"],
        filename=filename,
        n_rows=meta["n_rows"],
        meta_json=json.dumps(
            {
                "numeric_columns": meta["numeric_columns"],
                "categorical_columns": meta["categorical_columns"],
                "excluded_columns": meta.get("excluded_columns", []),
                "suggested_id_column": meta.get("suggested_id_column"),
                "all_columns": meta.get("all_columns", []),
            },
            ensure_ascii=False,
        ),
        created_at=now,
    )
    db.add(row)
    project.updated_at = now
    db.commit()
    return get_project_detail(db, project_id=project_id, user_id=user_id)


def add_text_source(
    db: Session,
    *,
    project_id: str,
    user_id: str,
    source_type: str,
    filename: str,
    content: bytes,
) -> dict:
    if source_type not in TEXT_SOURCE_TYPES:
        raise ValueError(f"Tipo de fuente de texto no válido: {source_type}")

    project = get_project_or_404(db, project_id=project_id, user_id=user_id)

    existing = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.source_type == source_type,
        )
        .first()
    )
    if existing:
        db.delete(existing)

    meta = save_text_upload(user_id=user_id, filename=filename, content=content)
    now = _now()
    row = ProjectSource(
        id=str(uuid.uuid4()),
        project_id=project_id,
        source_type=source_type,
        dataset_id=meta["text_id"],
        filename=filename,
        n_rows=None,
        meta_json=json.dumps(
            {"char_count": meta["char_count"], "preview": meta.get("preview", "")},
            ensure_ascii=False,
        ),
        created_at=now,
    )
    db.add(row)
    project.updated_at = now
    db.commit()
    return get_project_detail(db, project_id=project_id, user_id=user_id)


def list_csv_sources(db: Session, *, project_id: str, user_id: str) -> list[ProjectSource]:
    get_project_or_404(db, project_id=project_id, user_id=user_id)
    return (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.source_type.in_(CSV_SOURCE_TYPES),
        )
        .order_by(ProjectSource.created_at.asc())
        .all()
    )


def get_source_dataset_meta(
    source: ProjectSource,
    *,
    user_id: str,
) -> dict:
    if not source.dataset_id:
        raise ValueError("La fuente no tiene dataset asociado")
    return get_dataset_meta(source.dataset_id, user_id=user_id)


def primary_incidents_source(sources: list[ProjectSource]) -> ProjectSource | None:
    for preferred in ("incidents", "software", "hardware", "change_mgmt"):
        for source in sources:
            if source.source_type == preferred:
                return source
    return sources[0] if sources else None
