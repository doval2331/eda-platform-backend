from app.services.projects.source_relationship import _overlap_score, summarize_tabular_programmatic


def test_overlap_score_detects_shared_domain():
    a = "incidencias IT energia hardware servicios SLA prioridad"
    b = "diccionario de energia hardware tickets SLA incumplimiento"
    assert _overlap_score(a, b) >= 0.2


def test_summarize_tabular_programmatic_includes_columns():
    meta = {
        "n_rows": 100,
        "all_columns": ["prioridad", "categoria"],
        "numeric_columns": [],
        "categorical_columns": ["prioridad", "categoria"],
    }
    text = summarize_tabular_programmatic(meta)
    assert "prioridad" in text
    assert "100" in text
