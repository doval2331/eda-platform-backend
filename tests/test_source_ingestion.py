from app.services.source_ingestion import detect_source_kind, ingest_source


def test_ingest_csv_as_tabular_source():
    source = ingest_source(
        "incidencias.csv",
        b"id,category,severity\n1,software,high\n2,hardware,low\n",
    )

    assert source.normalized_kind == "tabular"
    assert source.original_format == "csv"
    assert source.dataframe is not None
    assert source.dataframe.shape == (2, 3)


def test_ingest_text_as_context_source():
    source = ingest_source(
        "diccionario.md",
        "incident_id: identificador de incidencia\nseverity: prioridad".encode("utf-8"),
    )

    assert source.normalized_kind == "text"
    assert source.original_format == "md"
    assert source.text is not None
    assert "incident_id" in source.text


def test_detect_source_kind_from_extension():
    assert detect_source_kind("datos.xlsx") == "tabular"
    assert detect_source_kind("diccionario.pdf") == "text"
    assert detect_source_kind("reunion.wav") == "text"
