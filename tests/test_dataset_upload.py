import pytest

from app.services.dataset_store import _validate_csv_upload


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


def test_default_upload_limit_is_50_mb():
    from app.config import Settings

    settings = Settings()
    assert settings.max_upload_bytes == 50 * 1024 * 1024
