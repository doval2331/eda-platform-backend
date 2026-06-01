"""
Genera un dataset sintetico de incidencias IT para probar clustering.

Uso:
    python scripts/generate_it_incidents_dataset.py
    python scripts/generate_it_incidents_dataset.py --n 5000 --output data/it_incidents_synthetic_5000.csv

La columna synthetic_segment se incluye solo para evaluacion posterior.
No debe usarse como feature de clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "it_incidents_synthetic_2000.csv"

SEGMENTS = [
    "Incidencias simples de bajo impacto",
    "Incidencias criticas recurrentes",
    "Incidencias complejas de larga resolucion",
    "Incidencias asociadas a cambios",
    "Incidencias de seguridad",
    "Casos anomalos",
]
SEGMENT_PROBS = [0.32, 0.18, 0.18, 0.15, 0.12, 0.05]

SERVICES = [
    "Portal clientes",
    "ERP financiero",
    "Base de datos transaccional",
    "Correo corporativo",
    "IAM",
    "API pagos",
    "Plataforma cloud",
    "Red corporativa",
    "Mesa de ayuda",
]
CHANNELS = ["portal", "email", "telefono", "monitoreo", "chat", "api"]

SLA_LIMIT_HOURS = {
    "critica": 4,
    "alta": 8,
    "media": 24,
    "baja": 48,
}

SUBCATEGORIES = {
    "Aplicacion": ["Error funcional", "Error 500", "Lentitud", "Integracion"],
    "Base de datos": ["Bloqueos", "Backup", "Consulta lenta", "Conexion"],
    "Infraestructura": ["CPU alta", "Memoria", "Storage", "Red"],
    "Seguridad": ["Acceso", "Phishing", "Vulnerabilidad", "Malware"],
    "Cambios": ["Despliegue", "Rollback", "Configuracion", "Versionado"],
    "Soporte": ["Consulta", "Permisos", "Solicitud", "Configuracion usuario"],
}

ROOT_CAUSES = {
    "Incidencias simples de bajo impacto": [
        "Error de usuario",
        "Configuracion menor",
        "Solicitud operativa",
        "Documentacion insuficiente",
    ],
    "Incidencias criticas recurrentes": [
        "Defecto recurrente",
        "Capacidad insuficiente",
        "Monitoreo tardio",
        "Regla de escalamiento debil",
    ],
    "Incidencias complejas de larga resolucion": [
        "Dependencia de terceros",
        "Deuda tecnica",
        "Arquitectura compleja",
        "Diagnostico incompleto",
    ],
    "Incidencias asociadas a cambios": [
        "Cambio no validado",
        "Despliegue fallido",
        "Rollback incompleto",
        "Configuracion inconsistente",
    ],
    "Incidencias de seguridad": [
        "Credenciales comprometidas",
        "Vulnerabilidad sin parche",
        "Phishing",
        "Politica de acceso debil",
    ],
    "Casos anomalos": [
        "Combinacion inusual de sintomas",
        "Registro incompleto",
        "Falla intermitente no reproducible",
        "Evento extremo",
    ],
}


def _choice(rng: np.random.Generator, values: list[str], probs: list[float] | None = None) -> str:
    return str(rng.choice(values, p=probs))


def _positive_normal(rng: np.random.Generator, mean: float, std: float, minimum: float) -> float:
    return max(minimum, float(rng.normal(mean, std)))


def _segment_profile(segment: str, rng: np.random.Generator) -> dict:
    if segment == "Incidencias simples de bajo impacto":
        return {
            "categoria": _choice(rng, ["Soporte", "Aplicacion"], [0.65, 0.35]),
            "prioridad": _choice(rng, ["baja", "media"], [0.7, 0.3]),
            "resolution": _positive_normal(rng, 3.5, 1.8, 0.2),
            "reaperturas": rng.poisson(0.15),
            "escalados": rng.poisson(0.08),
            "satisfaction": rng.normal(8.6, 0.7),
            "cost": rng.normal(70, 35),
        }
    if segment == "Incidencias criticas recurrentes":
        return {
            "categoria": _choice(rng, ["Aplicacion", "Base de datos", "Infraestructura"]),
            "prioridad": _choice(rng, ["alta", "critica"], [0.55, 0.45]),
            "resolution": _positive_normal(rng, 16, 8, 1),
            "reaperturas": rng.poisson(1.6),
            "escalados": rng.poisson(1.8),
            "satisfaction": rng.normal(5.4, 1.0),
            "cost": rng.normal(1400, 600),
        }
    if segment == "Incidencias complejas de larga resolucion":
        return {
            "categoria": _choice(rng, ["Base de datos", "Infraestructura", "Aplicacion"]),
            "prioridad": _choice(rng, ["media", "alta"], [0.45, 0.55]),
            "resolution": _positive_normal(rng, 56, 22, 8),
            "reaperturas": rng.poisson(2.0),
            "escalados": rng.poisson(2.7),
            "satisfaction": rng.normal(5.8, 1.1),
            "cost": rng.normal(3100, 1200),
        }
    if segment == "Incidencias asociadas a cambios":
        return {
            "categoria": _choice(rng, ["Cambios", "Aplicacion", "Infraestructura"], [0.7, 0.2, 0.1]),
            "prioridad": _choice(rng, ["media", "alta", "critica"], [0.45, 0.45, 0.1]),
            "resolution": _positive_normal(rng, 22, 11, 2),
            "reaperturas": rng.poisson(1.1),
            "escalados": rng.poisson(1.3),
            "satisfaction": rng.normal(6.5, 1.0),
            "cost": rng.normal(1800, 800),
        }
    if segment == "Incidencias de seguridad":
        return {
            "categoria": "Seguridad",
            "prioridad": _choice(rng, ["alta", "critica"], [0.5, 0.5]),
            "resolution": _positive_normal(rng, 12, 10, 0.5),
            "reaperturas": rng.poisson(0.8),
            "escalados": rng.poisson(2.4),
            "satisfaction": rng.normal(6.1, 1.2),
            "cost": rng.normal(4200, 1800),
        }
    return {
        "categoria": _choice(rng, list(SUBCATEGORIES)),
        "prioridad": _choice(rng, ["baja", "media", "alta", "critica"], [0.1, 0.2, 0.35, 0.35]),
        "resolution": _positive_normal(rng, 95, 45, 4),
        "reaperturas": rng.poisson(4.0),
        "escalados": rng.poisson(4.5),
        "satisfaction": rng.normal(3.9, 1.3),
        "cost": rng.normal(9000, 4200),
    }


def _service_for_category(category: str, rng: np.random.Generator) -> str:
    if category == "Seguridad":
        return _choice(rng, ["IAM", "Correo corporativo", "Red corporativa", "Plataforma cloud"])
    if category == "Base de datos":
        return _choice(rng, ["Base de datos transaccional", "ERP financiero", "API pagos"])
    if category == "Cambios":
        return _choice(rng, ["Portal clientes", "ERP financiero", "API pagos", "Plataforma cloud"])
    if category == "Soporte":
        return _choice(rng, ["Mesa de ayuda", "Portal clientes", "Correo corporativo"])
    return _choice(rng, SERVICES)


def _description(category: str, subcategory: str, service: str, root_cause: str) -> str:
    templates = [
        "{service}: {subcategory} asociado a {root_cause}.",
        "Incidencia en {service}; patron observado: {subcategory}.",
        "{category} reporta {subcategory} con posible causa {root_cause}.",
        "Usuario reporta problema en {service}; se detecta {subcategory}.",
    ]
    index = (len(category) + len(subcategory) + len(service) + len(root_cause)) % len(templates)
    return templates[index].format(
        category=category,
        subcategory=subcategory,
        service=service,
        root_cause=root_cause,
    )


def generate_it_incidents_dataset(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segments = rng.choice(SEGMENTS, size=n, p=SEGMENT_PROBS)
    rows: list[dict] = []

    for index, segment in enumerate(segments, start=1):
        profile = _segment_profile(str(segment), rng)
        category = profile["categoria"]
        priority = profile["prioridad"]
        subcategory = _choice(rng, SUBCATEGORIES[category])
        service = _service_for_category(category, rng)
        channel = _choice(rng, CHANNELS, [0.26, 0.2, 0.13, 0.25, 0.1, 0.06])
        resolution = round(float(profile["resolution"]), 2)
        reopenings = int(max(0, profile["reaperturas"]))
        escalations = int(max(0, profile["escalados"]))
        satisfaction = round(float(np.clip(profile["satisfaction"], 1, 10)), 1)
        cost = round(float(max(20, profile["cost"])), 2)
        root_cause = _choice(rng, ROOT_CAUSES[str(segment)])
        sla_limit = SLA_LIMIT_HOURS[priority]
        random_breach = rng.random() < (0.08 if priority in {"baja", "media"} else 0.18)
        sla_breached = bool(resolution > sla_limit or random_breach)

        rows.append(
            {
                "incident_id": f"INC-2026-{index:06d}",
                "categoria": category,
                "subcategoria": subcategory,
                "prioridad": priority,
                "servicio_afectado": service,
                "canal_entrada": channel,
                "tiempo_resolucion_horas": resolution,
                "sla_incumplido": sla_breached,
                "reaperturas": reopenings,
                "escalados": escalations,
                "satisfaccion_usuario": satisfaction,
                "coste_estimado": cost,
                "descripcion_corta": _description(category, subcategory, service, root_cause),
                "causa_raiz_simulada": root_cause,
                "synthetic_segment": str(segment),
            }
        )

    df = pd.DataFrame(rows)

    missing_plan = {
        "subcategoria": 0.015,
        "canal_entrada": 0.02,
        "satisfaccion_usuario": 0.035,
        "coste_estimado": 0.025,
        "causa_raiz_simulada": 0.015,
    }
    for column, rate in missing_plan.items():
        mask = rng.random(len(df)) < rate
        df.loc[mask, column] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar dataset sintetico de incidencias IT")
    parser.add_argument("--n", type=int, default=2_000, help="Numero de incidencias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_it_incidents_dataset(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Guardado: {args.output} ({len(df)} filas, {len(df.columns)} columnas)")
    print("Distribucion synthetic_segment:")
    print(df["synthetic_segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
