"""Gestor universal de fuentes para archivos tabulares y documentales."""

from app.services.datasets.source_ingestion.manager import IngestedSource, detect_source_kind, ingest_source

__all__ = ["IngestedSource", "detect_source_kind", "ingest_source"]
