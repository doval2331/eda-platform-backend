"""
Genera it_ops_synthetic_10000.csv — operaciones IT y clientes corporativos (TFM).

Uso:
    python scripts/generate_it_ops_dataset.py
    python scripts/generate_it_ops_dataset.py --n 50000 --output data/it_ops_synthetic_50000.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ImportError:
    Faker = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "it_ops_synthetic_10000.csv"

SEGMENTS = [
    "stable",
    "high_volume",
    "critical_incidents",
    "complex_projects",
    "risk_anomaly",
]
SEGMENT_PROBS = [0.35, 0.25, 0.18, 0.17, 0.05]

SECTORS = ["banking", "public_sector", "telecom", "energy", "transport", "insurance"]
SERVICES = ["software_dev", "it_support", "cloud", "cybersecurity", "systems_integration"]
CHANNELS = ["email", "portal", "phone", "chat", "api"]

# Columnas numéricas donde se inyectan missing (5–8 % del dataset)
MISSING_NUMERIC_COLS = [
    "reopen_rate",
    "escalation_rate",
    "platform_usage_score",
    "knowledge_base_usage",
    "automation_rate",
    "first_contact_resolution",
]


def _segment_params(seg: str, rng: np.random.Generator) -> dict:
    if seg == "stable":
        return dict(
            users=rng.normal(800, 200),
            tickets=rng.normal(25, 8),
            resolution=rng.normal(8, 3),
            sla=rng.normal(0.03, 0.02),
            satisfaction=rng.normal(8.5, 0.7),
            complexity=rng.normal(3, 1),
        )
    if seg == "high_volume":
        return dict(
            users=rng.normal(2500, 600),
            tickets=rng.normal(120, 30),
            resolution=rng.normal(14, 5),
            sla=rng.normal(0.08, 0.04),
            satisfaction=rng.normal(7.2, 0.9),
            complexity=rng.normal(5, 1.2),
        )
    if seg == "critical_incidents":
        return dict(
            users=rng.normal(1800, 500),
            tickets=rng.normal(80, 25),
            resolution=rng.normal(36, 12),
            sla=rng.normal(0.22, 0.08),
            satisfaction=rng.normal(5.8, 1.0),
            complexity=rng.normal(7, 1.3),
        )
    if seg == "complex_projects":
        return dict(
            users=rng.normal(3200, 900),
            tickets=rng.normal(70, 20),
            resolution=rng.normal(20, 7),
            sla=rng.normal(0.12, 0.05),
            satisfaction=rng.normal(7.0, 0.8),
            complexity=rng.normal(8, 1.0),
        )
    # risk_anomaly
    return dict(
        users=rng.normal(5000, 1500),
        tickets=rng.normal(250, 80),
        resolution=rng.normal(80, 25),
        sla=rng.normal(0.45, 0.15),
        satisfaction=rng.normal(4.0, 1.2),
        complexity=rng.normal(9, 1.0),
    )


def generate_it_ops_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker("es_ES") if Faker else None
    if fake:
        Faker.seed(seed)

    segments = rng.choice(SEGMENTS, size=n, p=SEGMENT_PROBS)
    rows: list[dict] = []

    for i, seg in enumerate(segments, start=1):
        p = _segment_params(seg, rng)
        active_users = max(10, int(p["users"]))
        monthly_tickets = max(0, int(p["tickets"]))
        project_complexity = float(np.clip(p["complexity"], 1, 10))
        sla_breach_rate = float(np.clip(p["sla"], 0, 1))
        resolution = max(0.5, float(p["resolution"]))
        satisfaction = float(np.clip(p["satisfaction"], 1, 10))

        contract_value = active_users * rng.normal(80, 20) + project_complexity * rng.normal(
            10_000, 3000
        )
        operational_risk = float(
            np.clip(
                sla_breach_rate * 40
                + project_complexity * 4
                + resolution * 0.25
                + monthly_tickets * 0.02
                + rng.normal(0, 3),
                0,
                100,
            )
        )

        rows.append(
            {
                "client_id": f"C{i:05d}",
                "client_name": fake.company() if fake else f"Cliente {i:05d}",
                "contract_reference": f"CTR-{rng.integers(10000, 99999)}",
                "sector": rng.choice(SECTORS),
                "service_line": rng.choice(SERVICES),
                "support_channel": rng.choice(CHANNELS),
                "active_users": active_users,
                "monthly_tickets": monthly_tickets,
                "critical_incidents": max(
                    0, int(monthly_tickets * sla_breach_rate * rng.uniform(0.2, 0.6))
                ),
                "avg_resolution_hours": round(resolution, 2),
                "sla_breach_rate": round(sla_breach_rate, 4),
                "reopen_rate": round(
                    float(np.clip(sla_breach_rate * rng.normal(0.8, 0.2), 0, 1)), 4
                ),
                "escalation_rate": round(
                    float(np.clip(sla_breach_rate * rng.normal(0.6, 0.2), 0, 1)), 4
                ),
                "platform_usage_score": round(
                    float(np.clip(rng.normal(active_users / 500, 1), 0, 10)), 2
                ),
                "change_requests": max(0, int(project_complexity * rng.normal(3, 1))),
                "project_complexity": round(project_complexity, 2),
                "customer_satisfaction": round(satisfaction, 2),
                "contract_value": round(max(1000, contract_value), 2),
                "monthly_cost": round(
                    max(500, contract_value / 12 * rng.normal(0.18, 0.04)), 2
                ),
                "operational_risk_score": round(operational_risk, 2),
                "account_tenure_months": max(1, int(rng.normal(48, 24))),
                "incidents_last_quarter": max(0, int(monthly_tickets * rng.uniform(2.5, 4.0))),
                "automation_rate": round(float(np.clip(rng.normal(0.45, 0.2), 0, 1)), 4),
                "knowledge_base_usage": round(float(np.clip(rng.normal(6, 2), 0, 10)), 2),
                "training_hours_delivered": max(0, int(rng.normal(120, 40))),
                "fte_assigned": max(1, int(rng.normal(8, 3))),
                "security_incidents": max(0, int(rng.poisson(2 if seg != "risk_anomaly" else 8))),
                "downtime_hours": round(max(0, rng.exponential(4 if seg == "stable" else 12)), 2),
                "data_volume_tb": round(max(0.1, rng.lognormal(2, 0.8)), 2),
                "integration_count": max(0, int(rng.normal(12, 5))),
                "license_utilization": round(float(np.clip(rng.normal(0.72, 0.15), 0, 1)), 4),
                "patch_compliance_rate": round(float(np.clip(rng.normal(0.88, 0.1), 0, 1)), 4),
                "first_contact_resolution": round(float(np.clip(rng.normal(0.7, 0.15), 0, 1)), 4),
                "nps_score": round(float(np.clip(rng.normal(30, 20), -100, 100)), 1),
                "backup_frequency_score": round(float(np.clip(rng.normal(8, 1.5), 0, 10)), 2),
                "segment": seg,
            }
        )

    df = pd.DataFrame(rows)

    # 5–8 % valores faltantes en columnas seleccionadas
    missing_rate = rng.uniform(0.05, 0.08)
    n_missing_cells = int(len(df) * len(MISSING_NUMERIC_COLS) * missing_rate)
    if n_missing_cells > 0:
        for _ in range(n_missing_cells):
            r_idx = rng.integers(0, len(df))
            col = rng.choice(MISSING_NUMERIC_COLS)
            df.at[r_idx, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar dataset sintético IT Ops")
    parser.add_argument("--n", type=int, default=10_000, help="Número de registros")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_it_ops_dataset(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Guardado: {args.output} ({len(df)} filas, {len(df.columns)} columnas)")


if __name__ == "__main__":
    main()
