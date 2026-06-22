"""
Genera diccionario de datos y datasets de prueba para preparación de datos.

Uso:
    python scripts/generate_test_fixtures.py
    python scripts/generate_test_fixtures.py --output-dir data/fixtures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_it_incidents_dataset import SEGMENTS, generate_it_incidents_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "fixtures"

DATA_DICTIONARY_ROWS = [
    {
        "columna": "incident_id",
        "tipo_dato": "texto (ID)",
        "rol_plataforma": "Identificador",
        "obligatorio": "Sí",
        "modelado": "No",
        "descripcion": "Identificador único de la incidencia. Se usa en tooltips y trazabilidad.",
        "ejemplo": "INC-2026-000123",
        "valores_permitidos": "Formato libre; debe ser único por fila",
    },
    {
        "columna": "categoria",
        "tipo_dato": "categórica",
        "rol_plataforma": "Feature categórica",
        "obligatorio": "Recomendado",
        "modelado": "Sí",
        "descripcion": "Familia funcional de la incidencia.",
        "ejemplo": "Aplicacion",
        "valores_permitidos": "Aplicacion; Base de datos; Infraestructura; Seguridad; Cambios; Soporte",
    },
    {
        "columna": "subcategoria",
        "tipo_dato": "categórica",
        "rol_plataforma": "Feature categórica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Subtipo dentro de la categoría.",
        "ejemplo": "Error funcional",
        "valores_permitidos": "Depende de categoria",
    },
    {
        "columna": "prioridad",
        "tipo_dato": "categórica ordinal",
        "rol_plataforma": "Feature categórica",
        "obligatorio": "Recomendado",
        "modelado": "Sí",
        "descripcion": "Prioridad operativa de la incidencia.",
        "ejemplo": "alta",
        "valores_permitidos": "baja; media; alta; critica (o crítica)",
    },
    {
        "columna": "servicio_afectado",
        "tipo_dato": "categórica",
        "rol_plataforma": "Feature categórica",
        "obligatorio": "Recomendado",
        "modelado": "Sí",
        "descripcion": "Servicio o sistema impactado.",
        "ejemplo": "Portal clientes",
        "valores_permitidos": "Texto corto; ≤40 valores distintos recomendado",
    },
    {
        "columna": "canal_entrada",
        "tipo_dato": "categórica",
        "rol_plataforma": "Feature categórica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Canal por el que se registró la incidencia.",
        "ejemplo": "portal",
        "valores_permitidos": "portal; email; telefono; monitoreo; chat; api",
    },
    {
        "columna": "tiempo_resolucion_horas",
        "tipo_dato": "numérica",
        "rol_plataforma": "Feature numérica",
        "obligatorio": "Recomendado",
        "modelado": "Sí",
        "descripcion": "Horas transcurridas hasta el cierre.",
        "ejemplo": "12.5",
        "valores_permitidos": "Número ≥ 0",
    },
    {
        "columna": "sla_incumplido",
        "tipo_dato": "booleano",
        "rol_plataforma": "Feature numérica/booleana",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Indica si se incumplió el SLA.",
        "ejemplo": "true",
        "valores_permitidos": "true / false",
    },
    {
        "columna": "reaperturas",
        "tipo_dato": "entero",
        "rol_plataforma": "Feature numérica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Número de reaperturas tras el cierre.",
        "ejemplo": "1",
        "valores_permitidos": "Entero ≥ 0",
    },
    {
        "columna": "escalados",
        "tipo_dato": "entero",
        "rol_plataforma": "Feature numérica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Número de escalados a niveles superiores.",
        "ejemplo": "2",
        "valores_permitidos": "Entero ≥ 0",
    },
    {
        "columna": "satisfaccion_usuario",
        "tipo_dato": "numérica",
        "rol_plataforma": "Feature numérica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Puntuación de satisfacción del usuario afectado.",
        "ejemplo": "7.8",
        "valores_permitidos": "Escala 1–10 (recomendado)",
    },
    {
        "columna": "coste_estimado",
        "tipo_dato": "numérica",
        "rol_plataforma": "Feature numérica",
        "obligatorio": "Opcional",
        "modelado": "Sí",
        "descripcion": "Coste operativo estimado en euros.",
        "ejemplo": "850.00",
        "valores_permitidos": "Número ≥ 0",
    },
    {
        "columna": "descripcion_corta",
        "tipo_dato": "texto largo",
        "rol_plataforma": "Texto / UI / LLM",
        "obligatorio": "Opcional",
        "modelado": "No",
        "descripcion": "Resumen legible para tooltips y chat.",
        "ejemplo": "Portal clientes: Error funcional asociado a Error de usuario.",
        "valores_permitidos": "Texto libre",
    },
    {
        "columna": "causa_raiz_simulada",
        "tipo_dato": "texto",
        "rol_plataforma": "Texto / referencia",
        "obligatorio": "Opcional",
        "modelado": "No",
        "descripcion": "Causa raíz documentada (referencia para interpretación).",
        "ejemplo": "Deuda tecnica",
        "valores_permitidos": "Texto libre",
    },
    {
        "columna": "synthetic_segment",
        "tipo_dato": "categórica",
        "rol_plataforma": "Evaluación",
        "obligatorio": "Solo datasets sintéticos",
        "modelado": "No",
        "descripcion": "Segmento latente simulado para medir ARI/NMI. No usar en clustering.",
        "ejemplo": "Incidencias criticas recurrentes",
        "valores_permitidos": "Ver hoja Casos_prueba",
    },
]

TEST_CASES = [
    {
        "id": "caso_01",
        "archivo_base": "caso_01_flujo_basico",
        "descripcion": "Flujo rápido tabular: subida + pipeline + interpretación.",
        "modalidad": "tabular",
        "filas": 500,
        "seed": 101,
        "segment_probs": None,
        "filtro_segmento": None,
    },
    {
        "id": "caso_02",
        "archivo_base": "caso_02_clustering_completo",
        "descripcion": "Dataset equilibrado para probar clusters y agentes LLM.",
        "modalidad": "tabular",
        "filas": 2000,
        "seed": 202,
        "segment_probs": None,
        "filtro_segmento": None,
    },
    {
        "id": "caso_03",
        "archivo_base": "caso_03_minimo_valido",
        "descripcion": "Límite inferior aceptado por la plataforma (30 filas).",
        "modalidad": "tabular",
        "filas": 30,
        "seed": 303,
        "segment_probs": None,
        "filtro_segmento": None,
    },
    {
        "id": "caso_04",
        "archivo_base": "caso_04_criticos_recurrentes",
        "descripcion": "Sesgo a incidencias críticas recurrentes y alta complejidad.",
        "modalidad": "tabular",
        "filas": 600,
        "seed": 404,
        "segment_probs": [0.05, 0.45, 0.25, 0.10, 0.10, 0.05],
        "filtro_segmento": None,
    },
    {
        "id": "caso_05",
        "archivo_base": "caso_05_seguridad",
        "descripcion": "Concentración en incidentes de seguridad.",
        "modalidad": "tabular",
        "filas": 400,
        "seed": 505,
        "segment_probs": [0.05, 0.10, 0.10, 0.10, 0.55, 0.10],
        "filtro_segmento": None,
    },
    {
        "id": "caso_06",
        "archivo_base": "caso_06_cambios_despliegues",
        "descripcion": "Incidencias ligadas a cambios y despliegues.",
        "modalidad": "tabular",
        "filas": 450,
        "seed": 606,
        "segment_probs": [0.08, 0.12, 0.15, 0.50, 0.10, 0.05],
        "filtro_segmento": None,
    },
]

AUX_CHANGE_MGMT_COLUMNS = [
    "change_id",
    "servicio",
    "tipo_cambio",
    "ventana_horas",
    "rollback_requerido",
    "incidencias_post_cambio",
    "tasa_exito",
    "equipo_responsable",
]


def generate_with_probs(n: int, seed: int, segment_probs: list[float] | None) -> pd.DataFrame:
    return generate_it_incidents_dataset(n=n, seed=seed, segment_probs=segment_probs)


def generate_change_mgmt_aux(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    services = [
        "Portal clientes",
        "ERP financiero",
        "API pagos",
        "Plataforma cloud",
        "IAM",
    ]
    change_types = ["estandar", "normal", "urgente", "emergencia"]
    teams = ["Plataforma", "Aplicaciones", "Redes", "Seguridad", "Datos"]

    rows = []
    for index in range(1, n + 1):
        service = str(rng.choice(services))
        change_type = str(rng.choice(change_types, p=[0.45, 0.30, 0.18, 0.07]))
        window = float(max(1, rng.normal(4 if change_type == "estandar" else 8, 2)))
        post_incidents = int(max(0, rng.poisson(2.5 if change_type in {"urgente", "emergencia"} else 0.8)))
        rows.append(
            {
                "change_id": f"CHG-2026-{index:05d}",
                "servicio": service,
                "tipo_cambio": change_type,
                "ventana_horas": round(window, 1),
                "rollback_requerido": bool(rng.random() < (0.35 if change_type == "emergencia" else 0.08)),
                "incidencias_post_cambio": post_incidents,
                "tasa_exito": round(float(np.clip(rng.normal(0.88, 0.08), 0.4, 1.0)), 2),
                "equipo_responsable": str(rng.choice(teams)),
            }
        )
    return pd.DataFrame(rows)


def generate_software_problems_aux(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    modules = ["Facturacion", "CRM", "Inventario", "Reporting", "Integraciones"]
    severities = ["menor", "mayor", "critica"]
    rows = []
    for index in range(1, n + 1):
        severity = str(rng.choice(severities, p=[0.55, 0.30, 0.15]))
        defects = int(max(0, rng.poisson(3 if severity == "critica" else 1.2)))
        rows.append(
            {
                "problem_id": f"PRB-2026-{index:05d}",
                "modulo": str(rng.choice(modules)),
                "severidad": severity,
                "defectos_abiertos": defects,
                "tiempo_medio_resolucion_dias": round(float(max(0.5, rng.normal(5, 2))), 1),
                "usuarios_impactados": int(max(1, rng.lognormal(3, 0.6))),
                "workaround_disponible": bool(rng.random() < 0.4),
            }
        )
    return pd.DataFrame(rows)


def write_dataset(df: pd.DataFrame, base_path: Path, formats: tuple[str, ...] = ("csv", "xlsx")) -> None:
    if "csv" in formats:
        df.to_csv(base_path.with_suffix(".csv"), index=False)
    if "xlsx" in formats:
        df.to_excel(base_path.with_suffix(".xlsx"), index=False, sheet_name="datos")


def build_dictionary_dataframe() -> pd.DataFrame:
    return pd.DataFrame(DATA_DICTIONARY_ROWS)


def build_test_cases_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "caso": case["id"],
                "archivo": f"{case['archivo_base']}.csv / .xlsx",
                "modalidad": case["modalidad"],
                "filas": case["filas"],
                "objetivo_prueba": case["descripcion"],
            }
            for case in TEST_CASES
        ]
        + [
            {
                "caso": "caso_07",
                "archivo": "caso_07_gestion_cambios_aux.csv / .xlsx",
                "modalidad": "project (fuente change_mgmt)",
                "filas": 180,
                "objetivo_prueba": "Fuente auxiliar de gestión de cambios para escenario multifuente.",
            },
            {
                "caso": "caso_08",
                "archivo": "caso_08_problemas_software_aux.csv / .xlsx",
                "modalidad": "project (fuente software)",
                "filas": 150,
                "objetivo_prueba": "Fuente auxiliar de problemas software para escenario multifuente.",
            },
        ]
    )


def build_usage_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paso": 1,
                "accion": "Revisar diccionario",
                "detalle": "Consulte la hoja Diccionario o diccionario_datos_incidencias.md antes de preparar datos.",
            },
            {
                "paso": 2,
                "accion": "Elegir caso de prueba",
                "detalle": "Use caso_01 para smoke test; caso_02 para demo completa; caso_03 valida mínimo 30 filas.",
            },
            {
                "paso": 3,
                "accion": "Subir en Preparar datos",
                "detalle": "Modalidad tabular: suba CSV/XLSX. Modalidad project: caso_02 + fuentes auxiliares 07/08.",
            },
            {
                "paso": 4,
                "accion": "Identificador opcional",
                "detalle": "Seleccione incident_id como columna ID para tooltips en el mapa 2D.",
            },
            {
                "paso": 5,
                "accion": "Ejecutar pipeline",
                "detalle": "Mínimo 30 filas y al menos 2 columnas de features (numéricas + categóricas).",
            },
            {
                "paso": 6,
                "accion": "No modelar texto/evaluación",
                "detalle": "descripcion_corta, causa_raiz_simulada y synthetic_segment no entran en clustering.",
            },
        ]
    )


def write_markdown_dictionary(out_dir: Path, template_df: pd.DataFrame) -> None:
    lines = [
        "# Diccionario de datos — Incidencias IT",
        "",
        "Documento de referencia para **Preparar datos** en la Plataforma EDA.",
        "",
        "## Requisitos mínimos",
        "",
        "- **Formatos**: CSV, TSV, XLSX, JSON, Parquet (máx. 250 MB).",
        "- **Filas mínimas**: 30.",
        "- **Features mínimas**: 2 columnas combinadas (numéricas + categóricas).",
        "- **Cardinalidad categórica**: ≤ 40 valores distintos por columna (recomendado).",
        "",
        "## Campos del esquema objetivo",
        "",
        "| Columna | Tipo | Rol | Modelado | Descripción |",
        "|---------|------|-----|----------|-------------|",
    ]
    for row in DATA_DICTIONARY_ROWS:
        lines.append(
            f"| `{row['columna']}` | {row['tipo_dato']} | {row['rol_plataforma']} | "
            f"{row['modelado']} | {row['descripcion']} |"
        )

    lines.extend(
        [
            "",
            "## Casos de prueba incluidos",
            "",
            "| Caso | Archivo | Uso |",
            "|------|---------|-----|",
        ]
    )
    for _, row in build_test_cases_dataframe().iterrows():
        lines.append(f"| {row['caso']} | `{row['archivo']}` | {row['objetivo_prueba']} |")

    lines.extend(
        [
            "",
            "## Plantilla vacía",
            "",
            "Use `plantilla_incidencias_vacia.xlsx` o la hoja **Plantilla** del Excel del diccionario.",
            "Contiene las columnas recomendadas sin filas de datos.",
            "",
            "## Escenario multifuente (project)",
            "",
            "1. Fuente principal `incidents`: `caso_02_clustering_completo.csv`",
            "2. Fuente `change_mgmt`: `caso_07_gestion_cambios_aux.csv`",
            "3. Fuente `software`: `caso_08_problemas_software_aux.csv`",
            "4. Fuente `dictionary` (texto): este archivo `.md` o el Excel del diccionario",
            "",
        ]
    )

    (out_dir / "diccionario_datos_incidencias.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar diccionario y datasets de prueba")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary_df = build_dictionary_dataframe()
    template_df = pd.DataFrame(columns=[row["columna"] for row in DATA_DICTIONARY_ROWS if row["columna"] != "synthetic_segment"])
    test_cases_df = build_test_cases_dataframe()
    usage_df = build_usage_guide()

    generated: list[tuple[str, int, int]] = []

    for case in TEST_CASES:
        df = generate_with_probs(case["filas"], case["seed"], case["segment_probs"])
        base = out_dir / case["archivo_base"]
        write_dataset(df, base)
        generated.append((case["archivo_base"], len(df), len(df.columns)))

    change_df = generate_change_mgmt_aux(180, seed=707)
    software_df = generate_software_problems_aux(150, seed=808)
    write_dataset(change_df, out_dir / "caso_07_gestion_cambios_aux")
    write_dataset(software_df, out_dir / "caso_08_problemas_software_aux")
    generated.append(("caso_07_gestion_cambios_aux", len(change_df), len(change_df.columns)))
    generated.append(("caso_08_problemas_software_aux", len(software_df), len(software_df.columns)))

    sample = generate_it_incidents_dataset(n=5, seed=909)
    template_with_sample = pd.concat([template_df, sample.drop(columns=["synthetic_segment"])], ignore_index=True)

    dict_xlsx = out_dir / "diccionario_datos_incidencias.xlsx"
    with pd.ExcelWriter(dict_xlsx, engine="openpyxl") as writer:
        dictionary_df.to_excel(writer, sheet_name="Diccionario", index=False)
        template_with_sample.to_excel(writer, sheet_name="Plantilla", index=False)
        test_cases_df.to_excel(writer, sheet_name="Casos_prueba", index=False)
        usage_df.to_excel(writer, sheet_name="Guia_uso", index=False)
        sample.to_excel(writer, sheet_name="Ejemplo_5_filas", index=False)

    template_xlsx = out_dir / "plantilla_incidencias_vacia.xlsx"
    template_df.to_excel(template_xlsx, index=False, sheet_name="plantilla")

    write_markdown_dictionary(out_dir, template_df)

    readme_lines = [
        "# Fixtures de prueba — Preparación de datos",
        "",
        "Generados con `python scripts/generate_test_fixtures.py`.",
        "",
        "## Archivos",
        "",
        "| Archivo | Filas | Columnas | Propósito |",
        "|---------|------:|---------:|-----------|",
    ]
    for name, rows, cols in generated:
        purpose = next(
            (c["descripcion"] for c in TEST_CASES if c["archivo_base"] == name),
            "Fuente auxiliar multifuente" if "aux" in name else "Prueba",
        )
        readme_lines.append(f"| `{name}.csv` / `.xlsx` | {rows} | {cols} | {purpose} |")

    readme_lines.extend(
        [
            "",
            "## Diccionario",
            "",
            "- `diccionario_datos_incidencias.xlsx` — hojas: Diccionario, Plantilla, Casos_prueba, Guia_uso, Ejemplo_5_filas",
            "- `diccionario_datos_incidencias.md` — versión texto para fuente `dictionary` en proyectos",
            "- `plantilla_incidencias_vacia.xlsx` — plantilla sin datos",
            "",
            "## Flujos recomendados",
            "",
            "### Smoke test (5 min)",
            "1. Preparar datos → tabular → `caso_01_flujo_basico.xlsx`",
            "2. Ejecutar pipeline → revisar mapa e interpretación",
            "",
            "### Demo completa",
            "1. `caso_02_clustering_completo.xlsx` (2000 filas)",
            "2. Agentes → estrategia + interpretación",
            "3. Chat + dashboard conversacional",
            "",
            "### Escenario multifuente",
            "1. Crear project → fuente incidents: `caso_02_clustering_completo.csv`",
            "2. Añadir change_mgmt: `caso_07_gestion_cambios_aux.csv`",
            "3. Añadir software: `caso_08_problemas_software_aux.csv`",
            "4. Adjuntar diccionario: `diccionario_datos_incidencias.md`",
            "",
            "### Validación de límites",
            "- Mínimo filas: `caso_03_minimo_valido.csv` (30 filas)",
            "- Sesgos temáticos: casos 04 (críticos), 05 (seguridad), 06 (cambios)",
            "",
        ]
    )

    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Fixtures generados en: {out_dir.resolve()}")
    for name, rows, cols in generated:
        print(f"  - {name}: {rows} filas x {cols} columnas")
    print(f"  - diccionario_datos_incidencias.xlsx")
    print(f"  - diccionario_datos_incidencias.md")
    print(f"  - plantilla_incidencias_vacia.xlsx")


if __name__ == "__main__":
    main()
