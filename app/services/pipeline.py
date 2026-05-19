"""Orquestación del pipeline por modalidad de datos."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.schemas import EvidenceMetadata, PipelineResult
from app.services.dataset_store import get_dataset_csv_path, get_dataset_meta, meta_to_profile
from app.services.it_ops_preprocess import (
    build_record_preview,
    dataframe_to_features,
    load_it_ops_dataframe,
)
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
        def _num(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        items.append(
            EvidenceMetadata(
                id=str(row.get("client_id", f"C{i + 1:05d}")),
                preview=preview,
                source="it_ops",
                sector=str(row.get("sector", "")) or None,
                service_line=str(row.get("service_line", "")) or None,
                monthly_tickets=_num(row.get("monthly_tickets")),
                sla_breach_rate=_num(row.get("sla_breach_rate")),
                operational_risk_score=_num(row.get("operational_risk_score")),
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
        items.append(
            EvidenceMetadata(
                id=row_id,
                preview=preview,
                source="tabular",
            )
        )
    return items


def run_pipeline(
    *,
    modality: Modality = "it_ops",
    reduction_method: ReductionMethod = "UMAP",
    seed: int = 42,
    n_samples: int = 2000,
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
            exclude_columns=exclude_columns,
        )
        X, _ = dataframe_to_features_generic(df, num_cols, cat_cols)
        cfg = load_pipeline_config()
        X_scaled = scale_features(X)
        X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
        cluster_labels = cluster_hdbscan(X_2d, config=cfg)
        outliers_count = int(np.sum(cluster_labels == -1))
        metrics = compute_metrics(X_2d, cluster_labels).model_copy(
            update={"n_clusters": _cluster_count(cluster_labels)}
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
        X, _, _meta = dataframe_to_features(df)
        # Features ya escaladas por columna; escalado global adicional opcional
        cfg = load_pipeline_config()
        X_scaled = scale_features(X)
        X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
        cluster_labels = cluster_hdbscan(X_2d, config=cfg)
        outliers_count = int(np.sum(cluster_labels == -1))
        metrics = compute_metrics(X_2d, cluster_labels).model_copy(
            update={"n_clusters": _cluster_count(cluster_labels)}
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
    X, true_labels, _ = generate_high_dim_features(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_true_clusters,
        seed=effective_seed,
    )
    cfg = load_pipeline_config()
    X_scaled = scale_features(X)
    X_2d = reduce_2d(X_scaled, reduction_method, effective_seed, config=cfg)
    cluster_labels = cluster_hdbscan(X_2d, config=cfg)
    outliers_count = int(np.sum(cluster_labels == -1))
    metrics = compute_metrics(X_2d, cluster_labels).model_copy(
        update={"n_clusters": _cluster_count(cluster_labels)}
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
