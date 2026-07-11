"""Preprocesado tabular genérico: inferencia de columnas y matriz de features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.datasets.incidents_schema import default_exclude_columns

MAX_CATEGORICAL_CARDINALITY = 40
MIN_FEATURE_COLUMNS = 2
DEFAULT_EXCLUDED_COLUMNS = {
    "synthetic_segment",
    "descripcion_corta",
    "description",
    "descripcion_larga",
    "texto",
    "text",
}
KNOWN_NUMERIC_COLUMNS = {
    "tiempo_resolucion_horas",
    "reaperturas",
    "escalados",
    "satisfaccion_usuario",
    "coste_estimado",
}


@dataclass
class TabularColumnProfile:
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]
    suggested_id_column: str | None
    all_columns: list[str]
    column_summaries: list[dict]


HIGH_NULL_RATIO = 0.45
HIGH_CARDINALITY_RATIO = 0.55


def load_tabular_csv(path, *, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El CSV está vacío.")
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).sort_index().reset_index(drop=True)
    return df


def _is_likely_id(series: pd.Series, n_rows: int) -> bool:
    name = str(series.name).lower()
    if name in {"id", "uuid", "client_id", "incident_id", "record_id", "row_id", "_registro_id"}:
        return True
    if series.dtype == object and series.nunique() == n_rows:
        return True
    return False


def profile_dataframe(
    df: pd.DataFrame,
    *,
    exclude_columns: list[str] | None = None,
) -> TabularColumnProfile:
    exclude = set(
        dict.fromkeys(
            [*(exclude_columns or []), *default_exclude_columns(), *DEFAULT_EXCLUDED_COLUMNS]
        )
    )
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    id_candidates: list[str] = []
    n_rows = len(df)
    column_summaries: list[dict] = []

    def add_column_summary(
        *,
        name: str,
        series: pd.Series,
        role: str,
        included: bool,
        reason: str = "",
        kind: str = "unknown",
        can_chart: bool = True,
        avoid_as_metric: bool = False,
        avoid_as_dimension: bool = False,
    ) -> None:
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        row_count = max(n_rows, 1)
        null_ratio = float(series.isna().mean()) if n_rows else 0.0
        cardinality_ratio = float(unique_count / row_count)
        high_nulls = null_ratio >= HIGH_NULL_RATIO
        high_cardinality = (
            unique_count > MAX_CATEGORICAL_CARDINALITY
            and cardinality_ratio >= HIGH_CARDINALITY_RATIO
            and not pd.api.types.is_numeric_dtype(series)
        )
        reasons: list[str] = []
        if reason:
            reasons.append(reason)
        if high_nulls:
            reasons.append("Tiene demasiados valores vacios para una lectura confiable.")
        if high_cardinality:
            reasons.append("Tiene demasiados valores distintos para agrupar o graficar bien.")

        useful_for_analysis = bool(included and not high_nulls and not high_cardinality)
        column_summaries.append(
            {
                "name": name,
                "role": role,
                "kind": kind,
                "included_in_analysis": bool(included),
                "useful_for_analysis": useful_for_analysis,
                "can_chart": bool(can_chart and included and not high_nulls),
                "avoid_as_metric": bool(avoid_as_metric),
                "avoid_as_dimension": bool(avoid_as_dimension),
                "null_ratio": round(null_ratio, 4),
                "null_pct": round(null_ratio * 100, 2),
                "non_null_count": non_null_count,
                "unique_count": unique_count,
                "cardinality_ratio": round(cardinality_ratio, 4),
                "high_nulls": bool(high_nulls),
                "high_cardinality": bool(high_cardinality),
                "not_recommended_reason": " ".join(dict.fromkeys(reasons)),
                "source": "backend_profile",
            }
        )

    for col in df.columns:
        series = df[col]
        if _is_likely_id(series, n_rows):
            id_candidates.append(col)
            excluded.append(col)
            add_column_summary(
                name=col,
                series=series,
                role="identifier",
                kind="identifier",
                included=False,
                reason="Identificador tecnico: sirve para trazabilidad, no como metrica.",
                can_chart=False,
                avoid_as_metric=True,
                avoid_as_dimension=True,
            )
            continue

        if col in exclude:
            excluded.append(col)
            add_column_summary(
                name=col,
                series=series,
                role="excluded",
                kind="technical",
                included=False,
                reason="Variable excluida por reglas de preparacion.",
                can_chart=False,
                avoid_as_metric=True,
                avoid_as_dimension=True,
            )
            continue

        nunique = series.nunique(dropna=False)
        if nunique <= 1:
            excluded.append(col)
            add_column_summary(
                name=col,
                series=series,
                role="constant",
                kind="constant",
                included=False,
                reason="No aporta variacion suficiente para analizar.",
                can_chart=False,
                avoid_as_metric=True,
                avoid_as_dimension=True,
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            if col in KNOWN_NUMERIC_COLUMNS:
                numeric.append(col)
                add_column_summary(
                    name=col,
                    series=series,
                    role="metric",
                    kind="numeric",
                    included=True,
                    can_chart=True,
                    avoid_as_dimension=True,
                )
                continue
            low_card = nunique <= min(20, max(5, int(0.05 * n_rows)))
            if low_card:
                categorical.append(col)
                add_column_summary(
                    name=col,
                    series=series,
                    role="dimension",
                    kind="numeric_category",
                    included=True,
                    can_chart=True,
                    avoid_as_metric=True,
                )
            else:
                numeric.append(col)
                add_column_summary(
                    name=col,
                    series=series,
                    role="metric",
                    kind="numeric",
                    included=True,
                    can_chart=True,
                    avoid_as_dimension=True,
                )
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(col)
            add_column_summary(
                name=col,
                series=series,
                role="dimension",
                kind="boolean",
                included=True,
                can_chart=True,
                avoid_as_metric=True,
            )
        else:
            if nunique <= MAX_CATEGORICAL_CARDINALITY:
                categorical.append(col)
                add_column_summary(
                    name=col,
                    series=series,
                    role="dimension",
                    kind="categorical",
                    included=True,
                    can_chart=True,
                    avoid_as_metric=True,
                )
            else:
                excluded.append(col)
                add_column_summary(
                    name=col,
                    series=series,
                    role="high_cardinality",
                    kind="categorical",
                    included=False,
                    reason="Cardinalidad alta: conviene usarla solo como evidencia o filtro.",
                    can_chart=False,
                    avoid_as_metric=True,
                    avoid_as_dimension=True,
                )

    suggested_id = None
    for preferred in ("_registro_id", "incident_id", "client_id", "id", "record_id", "uuid"):
        if preferred in id_candidates:
            suggested_id = preferred
            break
    if suggested_id is None and id_candidates:
        suggested_id = id_candidates[0]

    return TabularColumnProfile(
        numeric_columns=numeric,
        categorical_columns=categorical,
        excluded_columns=excluded,
        suggested_id_column=suggested_id,
        all_columns=list(df.columns),
        column_summaries=column_summaries,
    )


def resolve_feature_columns(
    profile: TabularColumnProfile,
    *,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    exclude = set(exclude_columns or [])
    num = [c for c in (numeric_columns or profile.numeric_columns) if c not in exclude]
    cat = [c for c in (categorical_columns or profile.categorical_columns) if c not in exclude]
    if len(num) + len(cat) < MIN_FEATURE_COLUMNS:
        raise ValueError(
            f"Se necesitan al menos {MIN_FEATURE_COLUMNS} columnas de features "
            f"(numéricas o categóricas). Revisa exclusiones y tipos."
        )
    return num, cat


# Añadir este import al inicio del archivo si no está
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# Variables que necesitan RobustScaler
ROBUST_SCALE_COLUMNS = {"tiempo_resolucion_horas", "coste_estimado"}

# Variables ordinales con su orden definido
ORDINAL_COLUMNS = {
    "prioridad": [["baja", "media", "alta", "critica", "crítica"]],
}


def build_generic_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: dict[str, list] | None = None,
) -> ColumnTransformer:
    transformers = []

    # ── Numéricas con RobustScaler (outliers extremos) ──────
    robust_cols   = [c for c in numeric_cols if c in ROBUST_SCALE_COLUMNS]
    standard_cols = [c for c in numeric_cols if c not in ROBUST_SCALE_COLUMNS]

    if robust_cols:
        robust_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
        ])
        transformers.append(("robust", robust_pipe, robust_cols))

    if standard_cols:
        standard_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", standard_pipe, standard_cols))

    # ── Ordinales (prioridad) ────────────────────────────────
    if ordinal_cols:
        for col_name, categories in ordinal_cols.items():
            ordinal_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
                ("scaler", StandardScaler()),
            ])
            transformers.append((f"ord_{col_name}", ordinal_pipe, [col_name]))

    # ── Categóricas nominales (one-hot) ─────────────────────
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No hay columnas para transformar.")
    return ColumnTransformer(transformers, remainder="drop")


def dataframe_to_features_generic(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    preprocessor: ColumnTransformer | None = None,
) -> tuple[np.ndarray, ColumnTransformer]:

    # Separar prioridad de las categóricas nominales
    ordinal_cols_found = {}
    cat_cols_clean = []
    for col in categorical_cols:
        if col in ORDINAL_COLUMNS and col in df.columns:
            ordinal_cols_found[col] = ORDINAL_COLUMNS[col]
        else:
            cat_cols_clean.append(col)

    feature_cols = numeric_cols + list(ordinal_cols_found.keys()) + cat_cols_clean
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing}")

    data = df[feature_cols].copy()
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in cat_cols_clean:
        data[col] = data[col].astype("string").fillna("desconocido").astype(str)
    for col in ordinal_cols_found:
        data[col] = data[col].astype("string").fillna("desconocido").astype(str)

    if preprocessor is None:
        preprocessor = build_generic_preprocessor(
            numeric_cols,
            cat_cols_clean,
            ordinal_cols=ordinal_cols_found if ordinal_cols_found else None,
        )
        X = preprocessor.fit_transform(data)
    else:
        X = preprocessor.transform(data)

    return np.asarray(X, dtype=np.float64), preprocessor
    


def build_row_preview(row: pd.Series, id_column: str | None) -> str:
    parts: list[str] = []
    if id_column and id_column in row.index:
        parts.append(str(row[id_column]))
    shown = 0
    for col in row.index:
        if col == id_column:
            continue
        if shown >= 4:
            break
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}={val}")
        shown += 1
    return " | ".join(parts) if parts else str(row.iloc[0])
