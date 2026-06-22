from app.services.projects.project_validation import (
    MIN_FEATURE_COLUMNS,
    MIN_ROWS,
    _feature_columns,
    _validate_source_meta,
)


def test_validate_source_meta_ok():
    meta = {
        "n_rows": 100,
        "numeric_columns": ["a", "b"],
        "categorical_columns": ["c"],
    }
    assert _validate_source_meta("Incidencias", meta) is None


def test_validate_source_meta_too_few_rows():
    meta = {
        "n_rows": 10,
        "numeric_columns": ["a", "b"],
        "categorical_columns": ["c"],
    }
    issue = _validate_source_meta("Cambios", meta)
    assert issue is not None
    assert str(MIN_ROWS) in issue.message


def test_validate_source_meta_too_few_features():
    meta = {
        "n_rows": 100,
        "numeric_columns": ["a"],
        "categorical_columns": [],
    }
    issue = _validate_source_meta("Software", meta)
    assert issue is not None
    assert str(MIN_FEATURE_COLUMNS) in issue.message


def test_feature_columns_normalizes_names():
    meta = {
        "numeric_columns": ["Tiempo-Resolucion"],
        "categorical_columns": ["Prioridad"],
    }
    assert _feature_columns(meta) == {"tiempo_resolucion", "prioridad"}
