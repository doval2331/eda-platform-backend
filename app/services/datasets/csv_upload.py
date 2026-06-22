"""
Selección / subida de CSV para el notebook (Colab, Jupyter local o ruta manual).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_colab() -> bool:
    return "google.colab" in sys.modules


def save_uploaded_bytes(content: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def upload_csv_colab(dest: Path) -> Path:
    from google.colab import files  # type: ignore[import-untyped]

    print("Selecciona un archivo .csv en el diálogo...")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No se subió ningún archivo.")
    name = next(iter(uploaded))
    if not name.lower().endswith(".csv"):
        print(f"Advertencia: '{name}' no termina en .csv")
    return save_uploaded_bytes(uploaded[name], dest)


def upload_csv_jupyter_widget(dest: Path) -> Path:
    import ipywidgets as widgets
    from IPython.display import display

    upload = widgets.FileUpload(accept=".csv", multiple=False, description="CSV")
    btn = widgets.Button(description="Confirmar CSV", button_style="primary")
    out = widgets.Output()
    state: dict[str, Path | None] = {"path": None}

    def on_confirm(_btn):
        with out:
            out.clear_output()
            if not upload.value:
                print("Primero haz clic en «Upload» y elige tu .csv")
                return
            entry = upload.value[0]
            content = entry["content"]
            if hasattr(content, "tobytes"):
                raw = content.tobytes()
            else:
                raw = bytes(content)
            path = save_uploaded_bytes(raw, dest)
            state["path"] = path
            print(f"Listo: {path} ({path.stat().st_size // 1024} KB)")

    btn.on_click(on_confirm)
    display(widgets.VBox([
        widgets.HTML("<b>Sube tu CSV</b> (it_ops_synthetic_10000.csv u otro compatible)"),
        upload,
        btn,
        out,
    ]))
    print("Cuando veas «Listo» arriba, continúa con la siguiente celda.")
    return dest


def pick_csv_tkinter() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(
        title="Seleccionar CSV",
        filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
    )
    root.destroy()
    return Path(chosen) if chosen else None


def resolve_csv_path(
    *,
    mode: str,
    artifacts_dir: Path,
    default_path: Path,
    generate_script: Path | None = None,
) -> Path:
    """
    mode: 'upload' | 'default' | 'generate' | 'path'
    """
    mode = mode.strip().lower()
    upload_dest = artifacts_dir / "dataset_active.csv"

    if mode == "upload":
        if upload_dest.is_file():
            print(f"Reutilizando CSV ya subido: {upload_dest}")
            return upload_dest

        if is_colab():
            return upload_csv_colab(upload_dest)

        # Jupyter / VS Code: widget si hay IPython
        try:
            from IPython import get_ipython  # type: ignore[import-untyped]

            if get_ipython() is not None:
                upload_csv_jupyter_widget(upload_dest)
                if upload_dest.is_file():
                    return upload_dest
                raise FileNotFoundError(
                    "Pulsa «Confirmar CSV» y vuelve a ejecutar esta misma celda."
                )
        except ImportError:
            pass
        picked = pick_csv_tkinter()
        if picked and picked.is_file():
            save_uploaded_bytes(picked.read_bytes(), upload_dest)
            return upload_dest
        manual = input("Ruta absoluta al .csv: ").strip().strip('"')
        p = Path(manual)
        if p.is_file():
            return p
        raise FileNotFoundError(f"No se encontró: {manual}")

    if mode == "generate":
        if generate_script is None or not generate_script.is_file():
            raise FileNotFoundError(f"Script no encontrado: {generate_script}")
        import subprocess

        subprocess.run(
            [sys.executable, str(generate_script)],
            check=True,
            cwd=str(generate_script.parent.parent),
        )
        if not default_path.is_file():
            raise FileNotFoundError("El generador no creó el CSV esperado.")
        return default_path

    if mode == "default":
        if default_path.is_file():
            return default_path
        raise FileNotFoundError(
            f"No existe {default_path}. Usa mode='upload' o mode='generate'."
        )

    if mode == "path":
        manual = input("Ruta al .csv: ").strip().strip('"')
        p = Path(manual)
        if p.is_file():
            return p
        raise FileNotFoundError(manual)

    raise ValueError(f"mode inválido: {mode}. Usa upload | default | generate | path")
