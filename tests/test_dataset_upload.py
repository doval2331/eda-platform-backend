import pytest
import pandas as pd

from app.services.datasets.dataset_store import _validate_csv_upload, save_dataframe_as_dataset


def test_rejects_xlsx_extension():
    with pytest.raises(ValueError, match="Solo se admiten archivos CSV"):
        _validate_csv_upload("incidencias.xlsx", b"id,sla\n1,ok\n")


def test_rejects_xls_extension():
    with pytest.raises(ValueError, match="Solo se admiten archivos CSV"):
        _validate_csv_upload("datos.xls", b"id,sla\n1,ok\n")


def test_rejects_zip_signature_even_with_csv_extension():
    with pytest.raises(ValueError, match="parece ser Excel"):
        _validate_csv_upload("falso.csv", b"PK\x03\x04fake-xlsx")


def test_accepts_csv_filename_and_content():
    _validate_csv_upload("incidencias.csv", b"id,sla\n1,ok\n")


def test_default_upload_limit_is_250_mb():
    from app.config import Settings

    settings = Settings()
    assert settings.max_upload_bytes == 250 * 1024 * 1024


def test_persisted_dataset_includes_column_semantic_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.datasets.dataset_store.uploads_dir",
        lambda: tmp_path,
    )

    n = 50
    df = pd.DataFrame(
        {
            "incident_id": [f"INC-{i:05d}" for i in range(n)],
            "categoria": ["red", "aplicacion", "seguridad", "datos", "cloud"] * 10,
            "tiempo_resolucion_horas": [float(i % 12) for i in range(n)],
            "descripcion_larga": [f"detalle {i}" for i in range(n)],
            "detalle_libre": [f"texto libre unico {i}" for i in range(n)],
            "campo_con_nulos": [None if i < 35 else f"valor {i}" for i in range(n)],
        }
    )

    meta = save_dataframe_as_dataset(
        user_id="user-test",
        filename="metadata.csv",
        df=df,
    )

    summaries = {item["name"]: item for item in meta["column_summaries"]}
    assert set(summaries) == set(df.columns)
    assert summaries["incident_id"]["role"] == "identifier"
    assert summaries["incident_id"]["can_chart"] is False
    assert summaries["categoria"]["useful_for_analysis"] is True
    assert summaries["tiempo_resolucion_horas"]["role"] == "metric"
    assert summaries["descripcion_larga"]["included_in_analysis"] is False
    assert summaries["detalle_libre"]["high_cardinality"] is True
    assert summaries["campo_con_nulos"]["high_nulls"] is True
