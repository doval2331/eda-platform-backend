"""Validación de relación entre fuentes tabulares y documentos (resumen vs resumen)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.services.agents.llm_agent import complete_json_with_llm, complete_with_llm
from app.services.datasets.dataset_store import (
    get_dataset_csv_path,
    get_dataset_meta,
    get_text_content,
)
from app.services.projects.project_service import (
    list_csv_sources,
    source_display_name,
    TEXT_SOURCE_TYPES,
)

RELATIONSHIP_ACCEPTED = "accepted"
RELATIONSHIP_EXCLUDED = "excluded"
RELATIONSHIP_PENDING = "pending"
RELATIONSHIP_SKIPPED = "skipped"

DEFAULT_SCORE_THRESHOLD = 0.55

COMPARE_SYSTEM_PROMPT = """
Compara dos resumenes de fuentes de datos de un mismo escenario de analisis IT.
Responde SOLO JSON valido sin markdown:
{"related": true|false, "score": 0.0-1.0, "reason": "explicacion breve en espanol"}
related=true solo si ambos describen el mismo dominio, entidades o variables de negocio.
"""

SUMMARIZE_SYSTEM_PROMPT = """
Resume el contenido en 1-2 parrafos claros en espanol para analisis de incidencias IT.
No inventes datos que no esten en el texto de entrada.
"""


@dataclass(frozen=True)
class RelationshipResult:
    status: str
    score: float | None
    reason: str
    tabular_summary: str | None = None
    text_summary: str | None = None


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if token not in {"the", "and", "para", "con", "los", "las", "del", "una", "por"}
    }


def _overlap_score(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def summarize_tabular_programmatic(meta: dict, df: pd.DataFrame | None = None) -> str:
    cols = meta.get("all_columns") or []
    numeric = meta.get("numeric_columns") or []
    categorical = meta.get("categorical_columns") or []
    n_rows = meta.get("n_rows") or (len(df) if df is not None else 0)
    parts = [
        f"Dataset tabular con {n_rows} filas y {len(cols)} columnas.",
        f"Columnas numericas: {', '.join(numeric[:12]) or 'ninguna'}.",
        f"Columnas categoricas: {', '.join(categorical[:12]) or 'ninguna'}.",
    ]
    if df is not None and not df.empty:
        for col in (categorical[:3] or cols[:3]):
            if col not in df.columns:
                continue
            try:
                top = df[col].astype(str).value_counts().head(3)
                parts.append(
                    f"Valores frecuentes en {col}: "
                    + ", ".join(f"{idx} ({cnt})" for idx, cnt in top.items())
                )
            except (TypeError, ValueError):
                continue
    return " ".join(parts)[:2500]


def summarize_text_programmatic(text: str, *, max_chars: int = 4000) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _llm_summarize(content: str, *, kind: str) -> str | None:
    result = complete_with_llm(
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        user_payload={"tipo": kind, "contenido": content[:6000]},
        temperature=0.1,
    )
    if result.used and result.answer.strip():
        return result.answer.strip()[:2500]
    return None


def compare_summaries(tabular_summary: str, text_summary: str) -> tuple[float, bool, str]:
    llm_result, payload = complete_json_with_llm(
        system_prompt=COMPARE_SYSTEM_PROMPT,
        user_payload={
            "resumen_tabular": tabular_summary[:2500],
            "resumen_documento": text_summary[:2500],
        },
        temperature=0.0,
    )
    if llm_result.used and isinstance(payload, dict):
        score = float(payload.get("score", 0))
        related = bool(payload.get("related"))
        reason = str(payload.get("reason") or "").strip() or "Evaluacion LLM."
        return score, related, reason

    score = _overlap_score(tabular_summary, text_summary)
    related = score >= DEFAULT_SCORE_THRESHOLD
    reason = (
        "Coincidencia lexica entre resumenes."
        if related
        else "No se detecto relacion entre los resumenes."
    )
    return score, related, reason


def validate_text_source_against_project(
    db: Session,
    *,
    project_id: str,
    user_id: str,
    text_source_id: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> RelationshipResult:
    from app.db import ProjectSource

    text_source = db.get(ProjectSource, text_source_id)
    if text_source is None or text_source.project_id != project_id:
        raise LookupError("Fuente de texto no encontrada")

    csv_sources = list_csv_sources(db, project_id=project_id, user_id=user_id)
    if not csv_sources:
        return RelationshipResult(
            status=RELATIONSHIP_SKIPPED,
            score=None,
            reason="Sin fuente tabular de referencia todavia.",
        )

    primary = csv_sources[0]
    if not primary.dataset_id:
        return RelationshipResult(
            status=RELATIONSHIP_PENDING,
            score=None,
            reason="La fuente tabular principal aun no esta lista.",
        )

    meta = get_dataset_meta(primary.dataset_id, user_id=user_id)
    csv_path = get_dataset_csv_path(primary.dataset_id, user_id=user_id)
    df = pd.read_csv(csv_path, nrows=500)
    tabular_prog = summarize_tabular_programmatic(meta, df)
    tabular_summary = _llm_summarize(tabular_prog, kind="tabular") or tabular_prog

    if not text_source.dataset_id:
        return RelationshipResult(
            status=RELATIONSHIP_PENDING,
            score=None,
            reason="Documento sin contenido procesado.",
        )

    raw_text = get_text_content(text_source.dataset_id, user_id=user_id)
    text_prog = summarize_text_programmatic(raw_text)
    text_summary = _llm_summarize(text_prog, kind="documento") or text_prog

    score, related, reason = compare_summaries(tabular_summary, text_summary)
    status = RELATIONSHIP_ACCEPTED if related and score >= score_threshold else RELATIONSHIP_EXCLUDED

    return RelationshipResult(
        status=status,
        score=round(score, 3),
        reason=reason,
        tabular_summary=tabular_summary,
        text_summary=text_summary,
    )


def apply_relationship_result(source, result: RelationshipResult) -> None:
    meta = json.loads(source.meta_json or "{}")
    meta["relationship_status"] = result.status
    meta["relationship_score"] = result.score
    meta["relationship_reason"] = result.reason
    if result.text_summary:
        meta["content_summary"] = result.text_summary[:3000]
    source.meta_json = json.dumps(meta, ensure_ascii=False)


def validate_project_text_sources(
    db: Session,
    *,
    project_id: str,
    user_id: str,
) -> list[RelationshipResult]:
    from app.db import ProjectSource

    rows = (
        db.query(ProjectSource)
        .filter(ProjectSource.project_id == project_id)
        .order_by(ProjectSource.created_at.asc())
        .all()
    )
    results: list[RelationshipResult] = []
    for row in rows:
        if row.source_type not in TEXT_SOURCE_TYPES:
            continue
        try:
            result = validate_text_source_against_project(
                db,
                project_id=project_id,
                user_id=user_id,
                text_source_id=row.id,
            )
        except (LookupError, FileNotFoundError, PermissionError) as exc:
            result = RelationshipResult(
                status=RELATIONSHIP_PENDING,
                score=None,
                reason=str(exc),
            )
        apply_relationship_result(row, result)
        results.append(result)
    db.commit()
    return results


def build_project_document_context(
    db: Session,
    *,
    project_id: str,
    user_id: str,
    max_chars: int = 6000,
) -> tuple[str, bool]:
    from app.db import ProjectSource

    rows = (
        db.query(ProjectSource)
        .filter(ProjectSource.project_id == project_id)
        .order_by(ProjectSource.created_at.asc())
        .all()
    )
    chunks: list[str] = []
    used = False
    for row in rows:
        if row.source_type not in TEXT_SOURCE_TYPES:
            continue
        meta = json.loads(row.meta_json or "{}")
        if meta.get("relationship_status") != RELATIONSHIP_ACCEPTED:
            continue
        summary = meta.get("content_summary") or meta.get("preview") or ""
        if not summary:
            continue
        used = True
        label = source_display_name(row)
        chunks.append(f"[{label}]\n{summary.strip()}")
        if sum(len(c) for c in chunks) >= max_chars:
            break
    text = "\n\n".join(chunks)[:max_chars]
    return text, used
