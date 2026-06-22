import pandas as pd
import pytest

from app.services.datasets.dataset_store import save_dataframe_as_dataset


def _sample_incidents(n: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": [f"INC-{i:04d}" for i in range(n)],
            "categoria": ["Red"] * n,
            "prioridad": ["Alta"] * n,
            "tiempo_resolucion_horas": [float(i % 10 + 1) for i in range(n)],
        }
    )


def _sample_changes(n: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "change_id": [f"CHG-{i:04d}" for i in range(n)],
            "tipo_cambio": ["Normal"] * n,
            "riesgo": ["Medio"] * n,
            "duracion_horas": [float(i % 8 + 2) for i in range(n)],
        }
    )


def test_save_dataframe_as_dataset_persists_merged_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.datasets.dataset_store.uploads_dir",
        lambda: tmp_path,
    )

    df1 = _sample_incidents()
    df2 = _sample_changes()
    df1["_fuente_tipo"] = "Incidencias"
    df1["_fuente_nombre"] = "incidencias_q1"
    df1["_registro_id"] = [f"src1:{value}" for value in df1["incident_id"]]
    df2["_fuente_tipo"] = "Cambios"
    df2["_fuente_nombre"] = "cambios_q1"
    df2["_registro_id"] = [f"src2:{value}" for value in df2["change_id"]]

    merged = pd.concat([df1, df2], ignore_index=True, sort=False)
    meta = save_dataframe_as_dataset(
        user_id="user-test",
        filename="escenario_unificado.csv",
        df=merged,
    )

    assert meta["n_rows"] == len(df1) + len(df2)
    assert "_fuente_tipo" in meta["categorical_columns"]
    assert "_fuente_nombre" in meta["categorical_columns"]
    assert meta["suggested_id_column"] == "_registro_id"


def test_save_dataframe_as_dataset_rejects_small_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.datasets.dataset_store.uploads_dir",
        lambda: tmp_path,
    )

    df = _sample_incidents(n=10)
    with pytest.raises(ValueError, match="30 filas"):
        save_dataframe_as_dataset(
            user_id="user-test",
            filename="small.csv",
            df=df,
        )
