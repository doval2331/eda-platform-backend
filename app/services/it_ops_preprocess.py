"""Carga y preprocesado del dataset de incidencias IT (legacy IT Ops + esquema incidentes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.incidents_schema import (
    IncidentColumnGroups,
    reference_segment_series,
    resolve_incident_column_groups,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "it_ops_synthetic_10000.csv"

# Compatibilidad con imports existentes
METADATA_COLS = ["client_id", "client_name", "contract_reference", "segment"]
CATEGORICAL_COLS = ["sector", "service_line", "support_channel"]
NUMERIC_COLS = [
    "active_users",
    "monthly_tickets",
    "critical_incidents",
    "avg_resolution_hours",
    "sla_breach_rate",
    "reopen_rate",
    "escalation_rate",
    "platform_usage_score",
    "change_requests",
    "project_complexity",
    "customer_satisfaction",
    "contract_value",
    "monthly_cost",
    "operational_risk_score",
    "account_tenure_months",
    "incidents_last_quarter",
    "automation_rate",
    "knowledge_base_usage",
    "training_hours_delivered",
    "fte_assigned",
    "security_incidents",
    "downtime_hours",
    "data_volume_tb",
    "integration_count",
    "license_utilization",
    "patch_compliance_rate",
    "first_contact_resolution",
    "nps_score",
    "backup_frequency_score",
]


def resolve_dataset_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    return DEFAULT_CSV


def load_it_ops_dataframe(
    path: Path | str | None = None,
    *,
    n_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    csv_path = resolve_dataset_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecuta: python scripts/generate_it_ops_dataset.py"
        )
    df = pd.read_csv(csv_path)
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).sort_index().reset_index(drop=True)
    return df


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No hay columnas numéricas ni categóricas para el preprocesado.")
    return ColumnTransformer(transformers, remainder="drop")


def dataframe_to_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    groups: IncidentColumnGroups | None = None,
) -> tuple[np.ndarray, ColumnTransformer, pd.DataFrame, IncidentColumnGroups]:
    """Devuelve matriz X, preprocessor, metadata alineada y grupos de columnas."""
    groups = groups or resolve_incident_column_groups(df)
    numeric_cols = groups.numeric
    categorical_cols = groups.categorical

    if not numeric_cols and not categorical_cols:
        raise ValueError("No se encontraron columnas numéricas o categóricas para modelar.")

    missing_num = [c for c in numeric_cols if c not in df.columns]
    missing_cat = [c for c in categorical_cols if c not in df.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Columnas faltantes en CSV: {missing_num + missing_cat}")

    meta_cols = [
        c
        for c in dict.fromkeys(
            groups.metadata + groups.text + groups.evaluation + ([groups.identifier] if groups.identifier else [])
        )
        if c in df.columns
    ]
    meta_df = df[meta_cols].copy() if meta_cols else df.iloc[:, :1].copy()

    if preprocessor is None:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X = preprocessor.fit_transform(df)
    else:
        X = preprocessor.transform(df)

    return np.asarray(X, dtype=np.float64), preprocessor, meta_df, groups


def build_record_preview(row: pd.Series) -> str:
    record_id = row.get("incident_id") or row.get("client_id") or "?"
    category = row.get("categoria") or row.get("sector") or "—"
    service = row.get("servicio_afectado") or row.get("service_line") or "—"
    description = row.get("descripcion_corta")
    sla = row.get("sla_breach_rate", row.get("sla_incumplido", "—"))
    resolution = row.get("tiempo_resolucion_horas", row.get("avg_resolution_hours", "—"))

    if description and not pd.isna(description):
        return f"{record_id} | {category} | {service} | {description}"

    tickets = row.get("monthly_tickets", "—")
    risk = row.get("operational_risk_score", "—")
    return (
        f"{record_id} | {category} | {service} | "
        f"resolución {resolution}h | SLA {sla} | riesgo {risk} | {tickets} tickets/mes"
    )


def get_reference_segments(df: pd.DataFrame) -> pd.Series | None:
    return reference_segment_series(df)
