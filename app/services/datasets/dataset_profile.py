"""Perfil exploratorio de datasets tabulares (ligero + full con correlaciones/alertas)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.datasets.dataset_store import get_dataset_csv_path, get_dataset_meta


def build_dataset_explore_profile(
    dataset_id: str,
    *,
    user_id: str,
    sample_rows: int = 5000,
) -> dict[str, Any]:
    meta = get_dataset_meta(dataset_id, user_id=user_id)
    csv_path = get_dataset_csv_path(dataset_id, user_id=user_id)
    df = pd.read_csv(csv_path, nrows=sample_rows)

    numeric = [c for c in (meta.get("numeric_columns") or []) if c in df.columns]
    categorical = [c for c in (meta.get("categorical_columns") or []) if c in df.columns]

    columns: list[dict[str, Any]] = []
    for col in numeric[:12]:
        series = pd.to_numeric(df[col], errors="coerce")
        columns.append(
            {
                "name": col,
                "kind": "numeric",
                "null_pct": float(series.isna().mean() * 100),
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "histogram": _histogram(series.dropna()),
            }
        )

    for col in categorical[:12]:
        series = df[col].astype(str)
        counts = series.value_counts().head(8)
        columns.append(
            {
                "name": col,
                "kind": "categorical",
                "null_pct": float((series == "nan").mean() * 100),
                "top_values": [
                    {"label": str(idx), "count": int(cnt)} for idx, cnt in counts.items()
                ],
            }
        )

    return {
        "dataset_id": dataset_id,
        "n_rows_sampled": int(len(df)),
        "n_rows_total": int(meta.get("n_rows") or len(df)),
        "n_cols": int(meta.get("n_cols") or len(df.columns)),
        "columns": columns,
        "breakdowns": _business_breakdowns(df),
    }


def build_dataset_full_profile(
    dataset_id: str,
    *,
    user_id: str,
    sample_rows: int = 5000,
) -> dict[str, Any]:
    """Perfil ampliado: explore + alertas, correlaciones y (opcional) ydata-profiling."""
    base = build_dataset_explore_profile(dataset_id, user_id=user_id, sample_rows=sample_rows)
    meta = get_dataset_meta(dataset_id, user_id=user_id)
    csv_path = get_dataset_csv_path(dataset_id, user_id=user_id)
    df = pd.read_csv(csv_path, nrows=sample_rows)

    numeric = [c for c in (meta.get("numeric_columns") or []) if c in df.columns]
    alerts = _quality_alerts(df, numeric, base.get("columns") or [])
    correlations = _correlation_pairs(df, numeric)
    duplicate_pct = float(df.duplicated().mean() * 100) if len(df) else 0.0

    profiler = "lightweight"
    ydata_alerts: list[dict[str, str]] = []
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            df,
            title=f"Perfil {dataset_id[:8]}",
            minimal=True,
            explorative=False,
            correlations={"calculate": True, "warn_high_correlations": True},
            missing_diagrams={"bar": False, "matrix": False, "heatmap": False},
            interactions=None,
        )
        description = profile.get_description()
        table_stats = description.get("table", {}) or {}
        if table_stats.get("n_duplicates", 0):
            ydata_alerts.append(
                {
                    "level": "warning",
                    "message": f"ydata: {table_stats['n_duplicates']} filas duplicadas detectadas.",
                }
            )
        for alert in (description.get("alerts", []) or [])[:8]:
            if isinstance(alert, dict):
                ydata_alerts.append(
                    {
                        "level": str(alert.get("alert_type", "info")),
                        "message": str(alert.get("alert_fields", alert.get("fields", ""))),
                    }
                )
        profiler = "ydata-profiling"
    except ImportError:
        pass
    except Exception:
        pass

    return {
        **base,
        "profiler": profiler,
        "duplicate_rows_pct": round(duplicate_pct, 2),
        "alerts": alerts + ydata_alerts,
        "correlations": correlations,
    }


def build_dataset_profile_html(
    dataset_id: str,
    *,
    user_id: str,
    sample_rows: int = 5000,
) -> str:
    """Informe HTML completo vía ydata-profiling (requiere dependencia instalada)."""
    from ydata_profiling import ProfileReport

    meta = get_dataset_meta(dataset_id, user_id=user_id)
    csv_path = get_dataset_csv_path(dataset_id, user_id=user_id)
    df = pd.read_csv(csv_path, nrows=sample_rows)
    title = meta.get("filename") or f"Dataset {dataset_id[:8]}"
    profile = ProfileReport(df, title=title, minimal=True, explorative=True)
    return profile.to_html()


def _quality_alerts(
    df: pd.DataFrame,
    numeric: list[str],
    columns: list[dict[str, Any]],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    dup_pct = float(df.duplicated().mean() * 100) if len(df) else 0.0
    if dup_pct >= 5:
        alerts.append(
            {
                "level": "warning",
                "message": f"{dup_pct:.1f}% de filas duplicadas en la muestra.",
            }
        )
    for col_info in columns:
        null_pct = col_info.get("null_pct") or 0
        if null_pct >= 40:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Columna «{col_info['name']}»: {null_pct:.0f}% valores nulos.",
                }
            )
        elif null_pct >= 15:
            alerts.append(
                {
                    "level": "info",
                    "message": f"Columna «{col_info['name']}»: {null_pct:.0f}% valores nulos.",
                }
            )
    if len(numeric) < 2:
        alerts.append(
            {
                "level": "info",
                "message": "Pocas columnas numéricas para correlaciones significativas.",
            }
        )
    return alerts[:12]


def _correlation_pairs(
    df: pd.DataFrame,
    numeric: list[str],
    *,
    threshold: float = 0.55,
    max_pairs: int = 24,
) -> list[dict[str, Any]]:
    if len(numeric) < 2:
        return []
    matrix = df[numeric].apply(pd.to_numeric, errors="coerce").corr()
    pairs: list[dict[str, Any]] = []
    for i, col_a in enumerate(numeric):
        for col_b in numeric[i + 1 :]:
            value = matrix.loc[col_a, col_b]
            if pd.isna(value) or abs(value) < threshold:
                continue
            pairs.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "coefficient": round(float(value), 3),
                }
            )
    pairs.sort(key=lambda item: abs(item["coefficient"]), reverse=True)
    return pairs[:max_pairs]


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_col(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_col(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _sla_series(df: pd.DataFrame) -> pd.Series | None:
    col = _pick_column(
        df,
        (
            "sla_incumplido",
            "sla_breached",
            "sla_breach",
            "incumplimiento_sla",
        ),
    )
    if col is None:
        rate_col = _pick_column(df, ("sla_breach_rate", "tasa_sla", "sla_rate"))
        if rate_col is None:
            return None
        return pd.to_numeric(df[rate_col], errors="coerce")
    raw = df[col]
    if raw.dtype == bool:
        return raw.astype(float)
    mapped = (
        raw.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "1": 1.0, "si": 1.0, "sí": 1.0, "yes": 1.0})
    )
    numeric = pd.to_numeric(raw, errors="coerce")
    return mapped.fillna(numeric)


def _business_breakdowns(df: pd.DataFrame) -> dict[str, Any]:
    breakdowns: dict[str, Any] = {}
    category_col = _pick_column(
        df,
        ("categoria", "category", "subcategoria", "subcategory", "sector", "servicio_afectado"),
    )
    sla = _sla_series(df)
    if category_col and sla is not None:
        work = df[[category_col]].copy()
        work["_sla"] = sla
        work = work.dropna(subset=[category_col])
        work[category_col] = work[category_col].astype(str)
        grouped = (
            work.groupby(category_col, dropna=True)["_sla"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values("count", ascending=False)
            .head(10)
        )
        breakdowns["category_sla"] = [
            {
                "category": str(row[category_col]),
                "count": int(row["count"]),
                "sla_breach_pct": round(float(row["mean"]) * 100, 1)
                if float(row["mean"]) <= 1
                else round(float(row["mean"]), 1),
            }
            for _, row in grouped.iterrows()
        ]

    priority_col = _pick_column(df, ("prioridad", "priority", "severidad", "severity"))
    if priority_col:
        counts = df[priority_col].astype(str).value_counts().head(8)
        breakdowns["priority_volume"] = [
            {"label": str(label), "count": int(count)} for label, count in counts.items()
        ]

    return breakdowns


def _histogram(series: pd.Series, bins: int = 12) -> list[dict[str, float | int]]:
    if series.empty:
        return []
    counts, edges = pd.cut(series, bins=bins, retbins=True, duplicates="drop")
    grouped = series.groupby(counts, observed=False).count()
    out: list[dict[str, float | int]] = []
    for interval, count in grouped.items():
        out.append(
            {
                "bin_start": float(interval.left),
                "bin_end": float(interval.right),
                "count": int(count),
            }
        )
    return out
