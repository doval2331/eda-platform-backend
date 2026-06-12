import pytest

from app.services.project_service import (
    CSV_SOURCE_TYPES,
    OTHER_SOURCE_TYPES,
    TEXT_SOURCE_TYPES,
    _resolve_source_type_for_kind,
    primary_incidents_source,
    SOURCE_TYPE_LABELS,
)


class _FakeSource:
    def __init__(self, source_type: str):
        self.source_type = source_type


def test_csv_and_text_source_types_are_disjoint():
    assert not CSV_SOURCE_TYPES & TEXT_SOURCE_TYPES
    assert "incidents" in CSV_SOURCE_TYPES
    assert "dictionary" in TEXT_SOURCE_TYPES


def test_primary_incidents_source_prefers_incidents():
    sources = [
        _FakeSource("hardware"),
        _FakeSource("incidents"),
        _FakeSource("software"),
    ]
    primary = primary_incidents_source(sources)
    assert primary.source_type == "incidents"


def test_primary_incidents_source_fallback_order():
    sources = [_FakeSource("change_mgmt"), _FakeSource("hardware")]
    primary = primary_incidents_source(sources)
    assert primary.source_type == "hardware"


def test_source_type_labels_cover_all_types():
    for source_type in CSV_SOURCE_TYPES | TEXT_SOURCE_TYPES | OTHER_SOURCE_TYPES:
        assert source_type in SOURCE_TYPE_LABELS


def test_resolve_source_type_prefers_file_kind_over_semantic_label():
    assert _resolve_source_type_for_kind("dictionary", "tabular") == "other"
    assert _resolve_source_type_for_kind("notes", "tabular") == "other"
    assert _resolve_source_type_for_kind("incidents", "tabular") == "incidents"
    assert _resolve_source_type_for_kind("incidents", "text") == "other"
    assert _resolve_source_type_for_kind("dictionary", "text") == "dictionary"
