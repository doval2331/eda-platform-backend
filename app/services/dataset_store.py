"""Almacenamiento de CSV subidos por usuario (perfil + fichero)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.config import get_settings
from app.services.tabular_preprocess import TabularColumnProfile, load_tabular_csv, profile_dataframe


def uploads_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / get_settings().uploads_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_path(dataset_id: str) -> Path:
    return uploads_dir() / f"{dataset_id}.meta.json"


def _csv_path(dataset_id: str) -> Path:
    return uploads_dir() / f"{dataset_id}.csv"


def save_upload(
    *,
    user_id: str,
    filename: str,
    content: bytes,
    exclude_columns: list[str] | None = None,
) -> dict:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Solo se admiten archivos .csv")
    if len(content) > get_settings().max_upload_bytes:
        raise ValueError(
            f"El archivo supera el límite de {get_settings().max_upload_bytes // (1024 * 1024)} MB"
        )

    dataset_id = str(uuid.uuid4())
    csv_path = _csv_path(dataset_id)
    csv_path.write_bytes(content)

    try:
        df = load_tabular_csv(csv_path)
    except Exception as exc:
        csv_path.unlink(missing_ok=True)
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc

    if len(df) < 30:
        csv_path.unlink(missing_ok=True)
        raise ValueError("El dataset debe tener al menos 30 filas.")

    profile = profile_dataframe(df, exclude_columns=exclude_columns)
    if len(profile.numeric_columns) + len(profile.categorical_columns) < 2:
        csv_path.unlink(missing_ok=True)
        raise ValueError(
            "No se detectaron suficientes columnas numéricas o categóricas para el análisis."
        )

    meta = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "filename": filename,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_columns": profile.numeric_columns,
        "categorical_columns": profile.categorical_columns,
        "excluded_columns": profile.excluded_columns,
        "suggested_id_column": profile.suggested_id_column,
        "all_columns": profile.all_columns,
    }
    _meta_path(dataset_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def get_dataset_meta(dataset_id: str, *, user_id: str) -> dict:
    path = _meta_path(dataset_id)
    if not path.is_file():
        raise FileNotFoundError("Dataset no encontrado")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("user_id") != user_id:
        raise PermissionError("No tienes acceso a este dataset")
    return meta


def get_dataset_csv_path(dataset_id: str, *, user_id: str) -> Path:
    get_dataset_meta(dataset_id, user_id=user_id)
    csv_path = _csv_path(dataset_id)
    if not csv_path.is_file():
        raise FileNotFoundError("Archivo CSV no encontrado")
    return csv_path


def meta_to_profile(meta: dict) -> TabularColumnProfile:
    return TabularColumnProfile(
        numeric_columns=list(meta["numeric_columns"]),
        categorical_columns=list(meta["categorical_columns"]),
        excluded_columns=list(meta.get("excluded_columns", [])),
        suggested_id_column=meta.get("suggested_id_column"),
        all_columns=list(meta["all_columns"]),
    )
