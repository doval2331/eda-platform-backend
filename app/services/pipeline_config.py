"""Carga de hiperparámetros exportados desde el notebook (Colab / Jupyter)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT / "notebooks" / "artifacts"
DEFAULT_CONFIG_PATH = DEFAULT_ARTIFACTS_DIR / "pipeline_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "reduction_method": "UMAP",
    "seed": 42,
    "n_samples_api": 2000,
    "umap": {
        "n_neighbors": 15,
        "min_dist": 0.1,
    },
    "hdbscan": {
        "min_cluster_size": None,
        "min_samples": None,
        "cluster_selection_method": "eom",
    },
    "tsne_max_samples": 3000,
}


def artifacts_dir() -> Path:
    return DEFAULT_ARTIFACTS_DIR


def config_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else DEFAULT_CONFIG_PATH


def load_pipeline_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = config_path(path)
    if not cfg_path.is_file():
        return dict(DEFAULT_CONFIG)
    with cfg_path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in loaded.items() if k in DEFAULT_CONFIG})
    for key in ("umap", "hdbscan"):
        if key in loaded and isinstance(loaded[key], dict):
            merged[key] = {**DEFAULT_CONFIG[key], **loaded[key]}
    return merged


def save_pipeline_config(data: dict[str, Any], path: Path | str | None = None) -> Path:
    out = config_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out
