"""Preprocesado tabular genérico: inferencia de columnas y matriz de features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.incidents_schema import default_exclude_columns

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


def load_tabular_csv(path, *, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El CSV está vacío.")
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).sort_index().reset_index(drop=True)
    return df


def _is_likely_id(series: pd.Series, n_rows: int) -> bool:
    name = str(series.name).lower()
    if name in {"id", "uuid", "client_id", "incident_id", "record_id", "row_id"}:
        return True
    if series.dtype == object and series.nunique() == n_rows:
        return True
    return False


def profile_dataframe(
    df: pd.DataFrame,
    *,
    exclude_columns: list[str] | None = None,
) -> TabularColumnProfile:
    exclude = set(exclude_columns or []) | DEFAULT_EXCLUDED_COLUMNS
    exclude = set(
        dict.fromkeys([*(exclude_columns or []), *default_exclude_columns()])
    )
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    id_candidates: list[str] = []
    n_rows = len(df)

    for col in df.columns:
        if col in exclude:
            excluded.append(col)
            continue

        series = df[col]
        if _is_likely_id(series, n_rows):
            id_candidates.append(col)
            excluded.append(col)
            continue

        nunique = series.nunique(dropna=False)
        if nunique <= 1:
            excluded.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            if col in KNOWN_NUMERIC_COLUMNS:
                numeric.append(col)
                continue
            low_card = nunique <= min(20, max(5, int(0.05 * n_rows)))
            if low_card:
                categorical.append(col)
            else:
                numeric.append(col)
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(col)
        else:
            if nunique <= MAX_CATEGORICAL_CARDINALITY:
                categorical.append(col)
            else:
                excluded.append(col)

    suggested_id = None
    for preferred in ("incident_id", "client_id", "id", "record_id", "uuid"):
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


def build_generic_preprocessor(
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
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
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
    feature_cols = numeric_cols + categorical_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing}")

    data = df[feature_cols].copy()
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical_cols:
        data[col] = data[col].astype("string").fillna("desconocido").astype(str)

    if preprocessor is None:
        preprocessor = build_generic_preprocessor(numeric_cols, categorical_cols)
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
