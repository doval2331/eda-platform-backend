"""
Esquema canónico del dataset de incidencias IT.

Define qué columnas entran al modelado, cuáles son metadata/UI y cuáles solo
sirven para evaluación posterior (ARI/NMI vs segmentos sintéticos).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Nunca usar en clustering / reducción
EVALUATION_COLUMNS = ("synthetic_segment", "segment")

# Solo metadata, tooltips y explicación (no features)
TEXT_METADATA_COLUMNS = (
    "descripcion_corta",
    "causa_raiz_simulada",
    "client_name",
    "contract_reference",
)

IDENTIFIER_COLUMNS = (
    "incident_id",
    "client_id",
    "id",
    "record_id",
    "_registro_id",
)

# Dataset de incidencias (esquema objetivo del TFM)
INCIDENT_NUMERIC_COLUMNS = (
    "tiempo_resolucion_horas",
    "reaperturas",
    "escalados",
    "coste_estimado",
    "satisfaccion_usuario",
    "sla_breach_rate",
    "reopen_rate",
    "escalation_rate",
    "avg_resolution_hours",
    "critical_incidents",
    "monthly_tickets",
    "operational_risk_score",
    "customer_satisfaction",
    "security_incidents",
    "downtime_hours",
)

INCIDENT_CATEGORICAL_COLUMNS = (
    "categoria",
    "subcategoria",
    "prioridad",
    "servicio_afectado",
    "canal_entrada",
    "sector",
    "service_line",
    "support_channel",
    "severity",
    "status",
    "assignment_group",
)

# Dataset legacy IT Ops (cliente/cuenta) — compatible hasta llegar CSV del profesor
LEGACY_NUMERIC_COLUMNS = (
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
)

LEGACY_CATEGORICAL_COLUMNS = ("sector", "service_line", "support_channel")

LEGACY_METADATA_COLUMNS = ("client_id", "client_name", "contract_reference", "segment")


@dataclass(frozen=True)
class IncidentColumnGroups:
    numeric: list[str]
    categorical: list[str]
    metadata: list[str]
    text: list[str]
    evaluation: list[str]
    identifier: str | None


def _pick_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def _pick_identifier(df: pd.DataFrame) -> str | None:
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            return col
    return None


def resolve_incident_column_groups(df: pd.DataFrame) -> IncidentColumnGroups:
    """Resuelve grupos de columnas según columnas presentes en el CSV."""
    incident_numeric = _pick_existing(df, INCIDENT_NUMERIC_COLUMNS)
    incident_categorical = _pick_existing(df, INCIDENT_CATEGORICAL_COLUMNS)
    legacy_numeric = _pick_existing(df, LEGACY_NUMERIC_COLUMNS)
    legacy_categorical = _pick_existing(df, LEGACY_CATEGORICAL_COLUMNS)

    numeric = incident_numeric or legacy_numeric
    categorical = incident_categorical or legacy_categorical

    metadata = list(
        dict.fromkeys(
            _pick_existing(df, LEGACY_METADATA_COLUMNS)
            + _pick_existing(df, ("incident_id", "categoria", "prioridad", "servicio_afectado"))
        )
    )
    text = _pick_existing(df, TEXT_METADATA_COLUMNS)
    evaluation = _pick_existing(df, EVALUATION_COLUMNS)

    return IncidentColumnGroups(
        numeric=numeric,
        categorical=categorical,
        metadata=metadata,
        text=text,
        evaluation=evaluation,
        identifier=_pick_identifier(df),
    )


def default_exclude_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [*EVALUATION_COLUMNS, *TEXT_METADATA_COLUMNS, *IDENTIFIER_COLUMNS]
        )
    )


def reference_segment_series(df: pd.DataFrame) -> pd.Series | None:
    """Etiquetas de referencia para ARI/NMI (solo evaluación posterior)."""
    for col in EVALUATION_COLUMNS:
        if col in df.columns:
            return df[col].astype(str)
    return None
