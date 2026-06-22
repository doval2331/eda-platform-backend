"""Validación de escenarios multifuente antes de ejecutar el pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.orm import Session

from app.services.datasets.dataset_store import get_dataset_csv_path, get_dataset_meta
from app.services.projects.project_service import (
    list_csv_sources,
    primary_incidents_source,
    source_display_name,
)
from app.services.datasets.tabular_preprocess import profile_dataframe

MIN_ROWS = 30
MIN_FEATURE_COLUMNS = 2

# Tipos de fuente que suelen ocupar un único hueco semántico en el escenario.
SINGLE_SLOT_SOURCE_TYPES = frozenset(
    {"incidents", "change_mgmt", "software", "hardware", "dictionary"}
)


@dataclass(frozen=True)
class SourceValidationIssue:
    source_name: str
    message: str


def _normalize_column(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _feature_columns(meta: dict) -> set[str]:
    numeric = meta.get("numeric_columns") or []
    categorical = meta.get("categorical_columns") or []
    return {_normalize_column(column) for column in [*numeric, *categorical]}


def _validate_source_meta(source_name: str, meta: dict) -> SourceValidationIssue | None:
    n_rows = int(meta.get("n_rows") or 0)
    if n_rows < MIN_ROWS:
        return SourceValidationIssue(
            source_name=source_name,
            message=f"tiene {n_rows} filas; se requieren al menos {MIN_ROWS}.",
        )

    feature_count = len(meta.get("numeric_columns") or []) + len(
        meta.get("categorical_columns") or []
    )
    if feature_count < MIN_FEATURE_COLUMNS:
        return SourceValidationIssue(
            source_name=source_name,
            message=(
                "no tiene suficientes columnas numéricas o categóricas para el clustering "
                f"(detectadas {feature_count}, mínimo {MIN_FEATURE_COLUMNS})."
            ),
        )
    return None


def _load_source_frame(source, *, user_id: str) -> tuple[pd.DataFrame, dict]:
    if not source.dataset_id:
        raise ValueError(f"La fuente {source_display_name(source)} no tiene dataset asociado.")
    meta = get_dataset_meta(source.dataset_id, user_id=user_id)
    csv_path = get_dataset_csv_path(source.dataset_id, user_id=user_id)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"La fuente {source_display_name(source)} está vacía.")
    return df, meta


def _preview_merged_profile(
    sources,
    *,
    user_id: str,
) -> tuple[set[str], int]:
    """Devuelve columnas compartidas entre fuentes y recuento de features tras unir."""
    frames: list[pd.DataFrame] = []
    feature_sets: list[set[str]] = []

    for source in sources:
        df, meta = _load_source_frame(source, user_id=user_id)
        feature_sets.append(_feature_columns(meta))
        label = source.source_type
        display = source_display_name(source)
        part = df.copy()
        part["_fuente_tipo"] = label
        part["_fuente_nombre"] = display
        id_col = meta.get("suggested_id_column")
        if isinstance(id_col, str) and id_col in part.columns:
            part["_registro_id"] = part[id_col].astype(str).map(
                lambda value, sid=source.id: f"{sid}:{value}"
            )
        else:
            part["_registro_id"] = [f"{source.id}:{index}" for index in range(len(part))]
        frames.append(part)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    profile = profile_dataframe(merged)
    feature_count = len(profile.numeric_columns) + len(profile.categorical_columns)

    shared = set.intersection(*feature_sets) if feature_sets else set()
    shared -= {"_fuente_tipo", "_fuente_nombre", "_registro_id"}

    pairwise_shared: set[str] = set()
    for left in range(len(feature_sets)):
        for right in range(left + 1, len(feature_sets)):
            pairwise_shared |= feature_sets[left] & feature_sets[right]
    pairwise_shared -= {"_fuente_tipo", "_fuente_nombre", "_registro_id"}

    return pairwise_shared, feature_count


def validate_project_before_run(
    db: Session,
    *,
    project,
    user_id: str,
) -> None:
    """Valida que el escenario pueda ejecutarse con la estrategia elegida."""
    strategy = project.strategy
    csv_sources = list_csv_sources(db, project_id=project.id, user_id=user_id)

    if not csv_sources:
        raise ValueError(
            "El escenario necesita al menos una fuente tabular válida para analizar."
        )

    if strategy == "unified":
        primary = primary_incidents_source(csv_sources)
        if primary is None:
            raise ValueError("No hay fuente tabular principal para analizar.")
        primary_name = source_display_name(primary)
        primary_meta = get_dataset_meta(primary.dataset_id, user_id=user_id)
        primary_issue = _validate_source_meta(primary_name, primary_meta)
        if primary_issue:
            raise ValueError(
                f"La fuente principal ({primary_name}) {primary_issue.message}"
            )
        if primary.source_type != "incidents":
            incidents = next((s for s in csv_sources if s.source_type == "incidents"), None)
            if incidents is None:
                raise ValueError(
                    "La estrategia «Solo fuente principal» espera una fuente de incidencias. "
                    f"Se usaría «{primary_name}», que no parece ser el registro principal de tickets. "
                    "Añade incidencias o cambia a «Un análisis por fuente» / «Unificado multifuente»."
                )
        return

    issues: list[SourceValidationIssue] = []
    for source in csv_sources:
        meta = get_dataset_meta(source.dataset_id, user_id=user_id)
        issue = _validate_source_meta(source_display_name(source), meta)
        if issue:
            issues.append(issue)

    if strategy == "merged":
        if len(csv_sources) < 2:
            raise ValueError(
                "La estrategia unificada multifuente requiere al menos dos fuentes tabulares."
            )

        total_rows = sum(
            int(get_dataset_meta(source.dataset_id, user_id=user_id).get("n_rows") or 0)
            for source in csv_sources
        )
        if total_rows < MIN_ROWS:
            raise ValueError(
                f"Tras combinar las fuentes solo hay {total_rows} filas; se requieren al menos {MIN_ROWS}."
            )

        shared_columns, merged_features = _preview_merged_profile(csv_sources, user_id=user_id)
        if merged_features < MIN_FEATURE_COLUMNS:
            raise ValueError(
                "Tras combinar las fuentes no quedan suficientes columnas analizables "
                f"({merged_features} detectadas, mínimo {MIN_FEATURE_COLUMNS}). "
                "Revisa que los archivos tengan datos numéricos o categóricos útiles."
            )
        if not shared_columns and len(csv_sources) >= 2:
            weak_sources = [
                source_display_name(source)
                for source in csv_sources
                if len(_feature_columns(get_dataset_meta(source.dataset_id, user_id=user_id)))
                < MIN_FEATURE_COLUMNS
            ]
            if weak_sources:
                joined = ", ".join(f"«{name}»" for name in weak_sources)
                raise ValueError(
                    "Las fuentes no comparten columnas y algunas parecen insuficientes para "
                    f"un análisis conjunto: {joined}. Usa «Un análisis por fuente» o sube archivos "
                    "con columnas compatibles del mismo dominio IT."
                )

    if issues:
        detail = "; ".join(f"«{issue.source_name}» {issue.message}" for issue in issues)
        raise ValueError(
            "Algunas fuentes tabulares no pueden ejecutarse: "
            f"{detail}"
        )
