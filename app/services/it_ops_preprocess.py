"""Carga y preprocesado del dataset sintético IT Ops (tabular)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "it_ops_synthetic_10000.csv"

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


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def dataframe_to_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[np.ndarray, ColumnTransformer, pd.DataFrame]:
    """Devuelve matriz X, preprocessor ajustado y filas alineadas."""
    missing_num = [c for c in NUMERIC_COLS if c not in df.columns]
    missing_cat = [c for c in CATEGORICAL_COLS if c not in df.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Columnas faltantes en CSV: {missing_num + missing_cat}")

    meta_df = df[METADATA_COLS].copy() if all(c in df.columns for c in METADATA_COLS) else df[
        ["client_id"]
    ].copy()

    if preprocessor is None:
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(df)
    else:
        X = preprocessor.transform(df)

    return np.asarray(X, dtype=np.float64), preprocessor, meta_df


def build_record_preview(row: pd.Series) -> str:
    sector = row.get("sector", "—")
    service = row.get("service_line", "—")
    sla = row.get("sla_breach_rate", "—")
    tickets = row.get("monthly_tickets", "—")
    risk = row.get("operational_risk_score", "—")
    return (
        f"{row.get('client_id', '?')} | {sector} | {service} | "
        f"{tickets} tickets/mes | SLA breach {sla} | riesgo {risk}"
    )
