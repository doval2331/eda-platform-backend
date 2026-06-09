"""Orquestación del pipeline por modalidad de datos."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas import EvidenceMetadata, PipelineMetrics, PipelineResult
from app.services.dataset_store import get_dataset_csv_path, get_dataset_meta, meta_to_profile
from app.services.it_ops_preprocess import (
    build_record_preview,
    dataframe_to_features,
    load_it_ops_dataframe,
)
from app.services.incidents_schema import default_exclude_columns, reference_segment_series
from app.services.tabular_preprocess import (
    build_row_preview,
    dataframe_to_features_generic,
    load_tabular_csv,
    resolve_feature_columns,
)
from app.services.pipeline_config import load_pipeline_config
from app.services.pipeline_core import (
    cluster_hdbscan,
    compute_metrics,
    reduce_2d,
    scale_features,
)
from app.services.synthetic_data import (
    _preview,
    generate_high_dim_features,
    seed_for_modality,
)

Modality = Literal["texto", "imagen", "multimodal", "it_ops", "tabular"]
ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]


def _build_legacy_metadata(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    modality: Literal["texto", "imagen", "multimodal"],
    seed: int,
) -> list[EvidenceMetadata]:
    py_rng = random.Random(seed + 7)
    items: list[EvidenceMetadata] = []
    for i, true_c in enumerate(true_labels):
        preview = _preview(modality, int(true_c), py_rng)
        if cluster_labels[i] == -1:
            preview = (
                "Evidencia atípica (patrón no consistente con los clusters principales)"
            )
        items.append(
            EvidenceMetadata(
                id=f"e{i + 1:03d}",
                preview=preview,
                source=modality,
            )
        )
    return items


def _cluster_count(labels: np.ndarray) -> int:
    return len({int(x) for x in labels.tolist() if int(x) >= 0})


def _pipeline_metrics(
    *,
    X_scaled: np.ndarray,
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    cfg: dict,
    df: pd.DataFrame | None,
    n_samples: int,
) -> PipelineMetrics:
    reference = None
    if df is not None:
        ref_series = reference_segment_series(df)
        if ref_series is not None:
            reference = ref_series.to_numpy()
    return compute_metrics(
        X_2d,
        cluster_labels,
        X_features=X_scaled,
        reference_labels=reference,
        hdbscan_config=cfg,
        n_samples=n_samples,
    )


def _num_or_none(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _str_or_none(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    text = str(val).strip()
    return text or None


def _bool_or_none(val) -> bool | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, bool):
        return val
    text = str(val).strip().lower()
    if text in {"true", "1", "si", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _severity_from_risk(risk: float | None, critical_incidents: float | None) -> str | None:
    if risk is None and critical_incidents is None:
        return None
    risk_value = risk or 0.0
    incidents = critical_incidents or 0.0
    if risk_value >= 70 or incidents >= 8:
        return "critical"
    if risk_value >= 50 or incidents >= 5:
        return "high"
    if risk_value >= 30 or incidents >= 2:
        return "medium"
    return "low"


def _build_it_ops_metadata(
    df,
    cluster_labels: np.ndarray,
) -> list[EvidenceMetadata]:
    items: list[EvidenceMetadata] = []
    for i in range(len(df)):
        row = df.iloc[i]
        preview = build_record_preview(row)
        if cluster_labels[i] == -1:
            preview = f"[Outlier] {preview}"

        risk = _num_or_none(row.get("operational_risk_score"))
        critical_incidents = _num_or_none(row.get("critical_incidents"))
        avg_resolution_hours = _num_or_none(row.get("avg_resolution_hours"))
        sla_breach_rate = _num_or_none(row.get("sla_breach_rate"))
        service_line = _str_or_none(row.get("service_line") or row.get("servicio_afectado"))
        sector = _str_or_none(row.get("sector") or row.get("categoria"))
        support_channel = _str_or_none(row.get("support_channel") or row.get("canal_entrada"))
        segment = _str_or_none(row.get("segment") or row.get("synthetic_segment"))
        short_description = _str_or_none(row.get("descripcion_corta"))
        root_cause = _str_or_none(row.get("causa_raiz_simulada"))

        items.append(
            EvidenceMetadata(
                id=str(row.get("incident_id") or row.get("client_id", f"INC{i + 1:05d}")),
                preview=preview,
                source="it_ops",
                incident_id=str(row.get("client_id", f"C{i + 1:05d}")),
                categoria=sector,
                prioridad=_severity_from_risk(risk, critical_incidents),
                servicio_afectado=service_line,
                canal_entrada=support_channel,
                tiempo_resolucion_horas=avg_resolution_hours,
                sla_incumplido=(
                    bool(sla_breach_rate >= 0.12) if sla_breach_rate is not None else None
                ),
                satisfaccion_usuario=_num_or_none(row.get("customer_satisfaction")),
                sector=sector,
                service_line=service_line,
                support_channel=support_channel,
                segment=segment,
                synthetic_segment=segment,
                category=sector,
                subcategory=_str_or_none(row.get("subcategoria")),
                priority=_str_or_none(row.get("prioridad")),
                severity=_severity_from_risk(risk, critical_incidents),
                status="active",
                assignment_group=support_channel,
                affected_service=service_line,
                short_description=short_description,
                root_cause_simulated=root_cause,
                monthly_tickets=_num_or_none(row.get("monthly_tickets")),
                critical_incidents=critical_incidents,
                avg_resolution_hours=avg_resolution_hours,
                resolution_minutes=(
                    avg_resolution_hours * 60 if avg_resolution_hours is not None else None
                ),
                reopenings=_num_or_none(row.get("reaperturas") or row.get("reopen_rate")),
                escalations=_num_or_none(row.get("escalados") or row.get("escalation_rate")),
                sla_breach_rate=sla_breach_rate,
                sla_breached=(
                    bool(sla_breach_rate >= 0.12) if sla_breach_rate is not None else None
                ),
                operational_risk_score=risk,
                business_impact_score=risk,
                security_incidents=_num_or_none(row.get("security_incidents")),
                downtime_hours=_num_or_none(row.get("downtime_hours")),
                customer_satisfaction=_num_or_none(row.get("customer_satisfaction")),
                estimated_cost=_num_or_none(row.get("coste_estimado") or row.get("contract_value")),
            )
        )
    return items


def _build_tabular_metadata(
    df,
    cluster_labels: np.ndarray,
    id_column: str | None,
) -> list[EvidenceMetadata]:
    items: list[EvidenceMetadata] = []
    for i in range(len(df)):
        row = df.iloc[i]
        preview = build_row_preview(row, id_column)
        if cluster_labels[i] == -1:
            preview = f"[Outlier] {preview}"
        row_id = (
            str(row[id_column])
            if id_column and id_column in row.index
            else f"row_{i + 1}"
        )
        sla_incumplido = _bool_or_none(row.get("sla_incumplido"))
        sla_rate = _num_or_none(row.get("sla_breach_rate"))
        if sla_rate is None and sla_incumplido is not None:
            sla_rate = 1.0 if sla_incumplido else 0.0
        tiempo_resolucion_horas = _num_or_none(row.get("tiempo_resolucion_horas"))
        items.append(
            EvidenceMetadata(
                id=row_id,
                preview=preview,
                source="tabular",
                incident_id=_str_or_none(row.get("incident_id")) or row_id,
                categoria=_str_or_none(row.get("categoria")) or _str_or_none(row.get("category")),
                subcategoria=_str_or_none(row.get("subcategoria")),
                prioridad=_str_or_none(row.get("prioridad")) or _str_or_none(row.get("severity")),
                servicio_afectado=(
                    _str_or_none(row.get("servicio_afectado"))
                    or _str_or_none(row.get("affected_service"))
                ),
                canal_entrada=_str_or_none(row.get("canal_entrada")),
                tiempo_resolucion_horas=tiempo_resolucion_horas,
                sla_incumplido=sla_incumplido,
                reaperturas=_num_or_none(row.get("reaperturas")),
                escalados=_num_or_none(row.get("escalados")),
                satisfaccion_usuario=_num_or_none(row.get("satisfaccion_usuario")),
                coste_estimado=_num_or_none(row.get("coste_estimado")),
                descripcion_corta=_str_or_none(row.get("descripcion_corta")),
                causa_raiz_simulada=_str_or_none(row.get("causa_raiz_simulada")),
                synthetic_segment=_str_or_none(row.get("synthetic_segment")),
                category=_str_or_none(row.get("categoria")) or _str_or_none(row.get("category")),
                severity=_str_or_none(row.get("prioridad")) or _str_or_none(row.get("severity")),
                affected_service=(
                    _str_or_none(row.get("servicio_afectado"))
                    or _str_or_none(row.get("affected_service"))
                ),
                avg_resolution_hours=tiempo_resolucion_horas,
                resolution_minutes=(
                    tiempo_resolucion_horas * 60
                    if tiempo_resolucion_horas is not None
                    else None
                ),
                sla_breach_rate=sla_rate,
                sla_breached=sla_incumplido,
                customer_satisfaction=_num_or_none(row.get("satisfaccion_usuario")),
            )
        )
    return items


def run_pipeline(
    *,
    modality: Modality = "it_ops",
    reduction_method: ReductionMethod = "UMAP",
    seed: int = 42,
    n_samples: int | None = None,
    n_features: int = 48,
    n_true_clusters: int = 6,
    dataset_path: Path | str | None = None,
    dataset_id: str | None = None,
    user_id: str | None = None,
    id_column: str | None = None,
    exclude_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> PipelineResult:
    effective_seed = (
        seed
        if modality in ("it_ops", "tabular")
        else seed_for_modality(modality, seed)
    )

    if modality == "tabular":
        if not dataset_id or not user_id:
            raise ValueError("dataset_id y usuario son obligatorios para modalidad tabular")
        meta = get_dataset_meta(dataset_id, user_id=user_id)
        csv_path = get_dataset_csv_path(dataset_id, user_id=user_id)
        profile = meta_to_profile(meta)
        df = load_tabular_csv(csv_path, n_samples=n_samples, seed=effective_seed)
        num_cols, cat_cols = resolve_feature_columns(
            profile,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            exclude_columns=list(
                dict.fromkeys([*(exclude_columns or []), *default_exclude_columns()])
            ),
        )
        X, _ = dataframe_to_features_generic(df, num_cols, cat_cols)
        cfg = load_pipeline_config()
        X_scaled = scale_features(X)
        X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
        cluster_labels = cluster_hdbscan(X_2d, config=cfg)
        outliers_count = int(np.sum(cluster_labels == -1))
        metrics = _pipeline_metrics(
            X_scaled=X_scaled,
            X_2d=X_2d,
            cluster_labels=cluster_labels,
            cfg=cfg,
            df=df,
            n_samples=len(df),
        )
        id_col = id_column or profile.suggested_id_column
        metadata = _build_tabular_metadata(df, cluster_labels, id_col)
        return PipelineResult(
            X_2d=X_2d.tolist(),
            cluster_labels=cluster_labels.astype(int).tolist(),
            outliers_count=outliers_count,
            metrics=metrics,
            metadata=metadata,
        )

    if modality == "it_ops":
        df = load_it_ops_dataframe(
            dataset_path,
            n_samples=n_samples,
            seed=effective_seed,
        )
        X, _, _meta, _groups = dataframe_to_features(df)
        cfg = load_pipeline_config()
        X_scaled = scale_features(X)
        X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
        cluster_labels = cluster_hdbscan(X_2d, config=cfg)
        outliers_count = int(np.sum(cluster_labels == -1))
        metrics = _pipeline_metrics(
            X_scaled=X_scaled,
            X_2d=X_2d,
            cluster_labels=cluster_labels,
            cfg=cfg,
            df=df,
            n_samples=len(df),
        )
        metadata = _build_it_ops_metadata(df, cluster_labels)
        return PipelineResult(
            X_2d=X_2d.tolist(),
            cluster_labels=cluster_labels.astype(int).tolist(),
            outliers_count=outliers_count,
            metrics=metrics,
            metadata=metadata,
        )

    effective_seed = seed_for_modality(modality, seed)
    legacy_n_samples = n_samples if n_samples is not None else 2000
    X, true_labels, _ = generate_high_dim_features(
        n_samples=legacy_n_samples,
        n_features=n_features,
        n_clusters=n_true_clusters,
        seed=effective_seed,
    )
    cfg = load_pipeline_config()
    X_scaled = scale_features(X)
    X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
    cluster_labels = cluster_hdbscan(X_2d, config=cfg)
    outliers_count = int(np.sum(cluster_labels == -1))
    metrics = _pipeline_metrics(
        X_scaled=X_scaled,
        X_2d=X_2d,
        cluster_labels=cluster_labels,
        cfg=cfg,
        df=None,
        n_samples=legacy_n_samples,
    )
    metadata = _build_legacy_metadata(
        true_labels, cluster_labels, modality, effective_seed
    )

    return PipelineResult(
        X_2d=X_2d.tolist(),
        cluster_labels=cluster_labels.astype(int).tolist(),
        outliers_count=outliers_count,
        metrics=metrics,
        metadata=metadata,
    )
