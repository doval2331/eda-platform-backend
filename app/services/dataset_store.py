"""Almacenamiento de fuentes subidas por usuario (perfil + fichero normalizado)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.config import get_settings
from app.services.source_ingestion import ingest_source
from app.services.tabular_preprocess import TabularColumnProfile, profile_dataframe


def uploads_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / get_settings().uploads_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_path(dataset_id: str) -> Path:
    return uploads_dir() / f"{dataset_id}.meta.json"


def _csv_path(dataset_id: str) -> Path:
    return uploads_dir() / f"{dataset_id}.csv"


_BLOCKED_EXTENSIONS = {
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ods",
    ".tsv",
    ".json",
    ".parquet",
}

_CSV_ONLY_MESSAGE = (
    "Solo se admiten archivos CSV. Exporta Excel u otros formatos como "
    "«CSV UTF-8 (delimitado por comas)» antes de subirlos."
)


def _validate_csv_upload(filename: str, content: bytes) -> None:
    lower = filename.lower().strip()
    ext = lower[lower.rfind(".") :] if "." in lower else ""
    if ext in _BLOCKED_EXTENSIONS:
        raise ValueError(_CSV_ONLY_MESSAGE)
    if not lower.endswith(".csv"):
        raise ValueError(_CSV_ONLY_MESSAGE)

    if len(content) >= 4 and content[:4] == b"PK\x03\x04":
        raise ValueError(
            "El archivo parece ser Excel u hoja de cálculo (ZIP). "
            "Guárdalo como CSV UTF-8 y vuelve a subirlo."
        )
    if len(content) >= 4 and content[:4] == b"\xd0\xcf\x11\xe0":
        raise ValueError(
            "El archivo parece ser Excel antiguo (.xls). "
            "Guárdalo como CSV UTF-8 y vuelve a subirlo."
        )


def save_upload(
    *,
    user_id: str,
    filename: str,
    content: bytes,
    exclude_columns: list[str] | None = None,
) -> dict:
    source = ingest_source(filename, content)
    if source.normalized_kind != "tabular" or source.dataframe is None:
        raise ValueError("Esta fuente debe ser tabular: CSV, TSV, Excel, JSON o Parquet.")
    if len(content) > get_settings().max_upload_bytes:
        raise ValueError(
            f"El archivo supera el límite de {get_settings().max_upload_bytes // (1024 * 1024)} MB"
        )

    dataset_id = str(uuid.uuid4())
    csv_path = _csv_path(dataset_id)
    df = source.dataframe
    df.to_csv(csv_path, index=False)

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
        "normalized_kind": source.normalized_kind,
        "original_format": source.original_format,
        "extraction_method": source.extraction_method,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_columns": profile.numeric_columns,
        "categorical_columns": profile.categorical_columns,
        "excluded_columns": profile.excluded_columns,
        "suggested_id_column": profile.suggested_id_column,
        "all_columns": profile.all_columns,
        "ingestion_metadata": source.metadata,
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


def _text_path(text_id: str) -> Path:
    return uploads_dir() / f"{text_id}.txt"


def _text_meta_path(text_id: str) -> Path:
    return uploads_dir() / f"{text_id}.text.meta.json"


def save_text_upload(
    *,
    user_id: str,
    filename: str,
    content: bytes,
    max_chars: int = 500_000,
) -> dict:
    if len(content) > get_settings().max_upload_bytes:
        raise ValueError(
            f"El archivo supera el límite de {get_settings().max_upload_bytes // (1024 * 1024)} MB"
        )

    source = ingest_source(filename, content)
    if source.normalized_kind != "text" or not source.text:
        raise ValueError("Esta fuente debe ser documental o audio: TXT, MD, DOCX, PDF o audio.")

    text = source.text.strip()
    if not text:
        raise ValueError("El archivo de texto está vacío.")
    if len(text) > max_chars:
        raise ValueError(f"El texto supera el límite de {max_chars} caracteres.")

    text_id = str(uuid.uuid4())
    _text_path(text_id).write_text(text, encoding="utf-8")
    preview = text[:400].replace("\n", " ")
    meta = {
        "text_id": text_id,
        "user_id": user_id,
        "filename": filename,
        "normalized_kind": source.normalized_kind,
        "original_format": source.original_format,
        "extraction_method": source.extraction_method,
        "char_count": len(text),
        "word_count": len(text.split()),
        "preview": preview,
        "ingestion_metadata": source.metadata,
    }
    _text_meta_path(text_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def get_text_content(text_id: str, *, user_id: str) -> str:
    path = _text_meta_path(text_id)
    if not path.is_file():
        raise FileNotFoundError("Texto no encontrado")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("user_id") != user_id:
        raise PermissionError("No tienes acceso a este texto")
    text_path = _text_path(text_id)
    if not text_path.is_file():
        raise FileNotFoundError("Archivo de texto no encontrado")
    return text_path.read_text(encoding="utf-8")


def meta_to_profile(meta: dict) -> TabularColumnProfile:
    return TabularColumnProfile(
        numeric_columns=list(meta["numeric_columns"]),
        categorical_columns=list(meta["categorical_columns"]),
        excluded_columns=list(meta.get("excluded_columns", [])),
        suggested_id_column=meta.get("suggested_id_column"),
        all_columns=list(meta["all_columns"]),
    )
