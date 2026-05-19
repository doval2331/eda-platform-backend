# EDA Platform — Backend (FastAPI)



API REST para análisis exploratorio: dataset IT Ops tabular, reducción (PCA, t-SNE, UMAP) y clustering HDBSCAN.



## Requisitos



- Python 3.11+



## Instalación



```bash

cd eda-platform-backend

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env

```



## 1. Generar dataset sintético IT Ops



```bash

python scripts/generate_it_ops_dataset.py

# → data/it_ops_synthetic_10000.csv



# Opcional: 50k para pruebas de rendimiento

python scripts/generate_it_ops_dataset.py --n 50000 --output data/it_ops_synthetic_50000.csv

```



## 2. Notebook de experimentación (recomendado antes de la API)



```bash

jupyter notebook notebooks/01_it_ops_eda.ipynb

```



El notebook importa los mismos módulos que la API (`app/services/`), de modo que notebook y backend comparten lógica.

### Google Colab

1. Sube el proyecto (ZIP con `app/`, `scripts/`, `notebooks/`) o clona el repo.
2. Abre `notebooks/01_it_ops_eda.ipynb` en Colab.
3. Ejecuta **Setup Colab** → **Seleccionar o subir CSV** (`DATA_MODE = "upload"`) y el resto.
4. Al final, **Exportar artefactos** genera `notebooks/artifacts/`:
   - `pipeline_config.json` — la API lo lee automáticamente
   - `fig_*.png`, `tabla_metricas_reduccion.csv`, `tabla_crosstab_segment_cluster.csv`
5. En Colab se descarga un ZIP; descomprímelo en `notebooks/artifacts/` en tu PC.

## 3. Usuario demo (login)

```bash
python scripts/seed_user.py
# Email: analista@tfm.local (o DEMO_USER_EMAIL en .env)
# Password: TfmDemo2026!
```

Configura `JWT_SECRET` en `.env` antes de producción.

## 4. Ejecutar API



```bash

uvicorn app.main:app --reload --port 8000

```



Documentación: http://127.0.0.1:8000/docs



## Endpoints



| Método | Ruta | Descripción |

|--------|------|-------------|

| GET | `/health` | Estado del servicio |

| POST | `/api/runs` | Ejecuta pipeline y persiste resultado |

| GET | `/api/runs` | Lista ejecuciones |

| GET | `/api/runs/{id}` | Detalle |



### Ejemplo IT Ops (principal)



```json

{

  "modality": "it_ops",

  "reduction_method": "UMAP",

  "seed": 42,

  "n_samples": 2000

}

```



Modalidades `texto`, `imagen`, `multimodal`: demo con vectores sintéticos legacy.



## Estructura



```

scripts/generate_it_ops_dataset.py   # Generador CSV

data/it_ops_synthetic_10000.csv      # Dataset (no versionado)

notebooks/01_it_ops_eda.ipynb        # EDA + validación

app/services/it_ops_preprocess.py    # Carga y features

app/services/pipeline_core.py        # Reducción + HDBSCAN

app/services/pipeline.py             # Orquestación por modalidad

app/services/pipeline_config.py      # Lee notebooks/artifacts/pipeline_config.json

notebooks/artifacts/                 # Figuras + JSON exportados desde Colab/Jupyter

```

Tras ejecutar el notebook, copia `pipeline_config.json` a `notebooks/artifacts/`; la API aplicará esos hiperparámetros.

## Frontend



```bash

# Terminal 1

uvicorn app.main:app --reload --port 8000



# Terminal 2

cd eda-platform-frontend

npm run dev

```



## Variables de entorno



| Variable | Default |

|----------|---------|

| `DATABASE_URL` | `sqlite:///./eda_platform.db` |

| `DEFAULT_N_SAMPLES` | `2000` |

| `IT_OPS_DATASET_PATH` | `data/it_ops_synthetic_10000.csv` |



## Base de datos (PostgreSQL con Docker — opción B)

Requisito: [Docker Desktop](https://www.docker.com/products/docker-desktop/) en ejecución.

### Arranque rápido

**Windows (CMD — sin política de ejecución):**

```cmd
cd eda-platform-backend
scripts\start_db.bat
```

**Windows (PowerShell)** — si falla por *execution policy*, usa el `.bat` de arriba o:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_db.ps1
```

**Linux / macOS:**

```bash
chmod +x scripts/start_db.sh
./scripts/start_db.sh
```

**Manual:**

```bash
docker compose up -d
copy .env.docker.example .env    # Windows: Copy-Item .env.docker.example .env
python scripts/wait_for_db.py
python scripts/seed_user.py
uvicorn app.main:app --reload --port 8000
```

### DBeaver

| Campo | Valor |
|--------|--------|
| Host | `127.0.0.1` |
| Puerto | `5432` |
| Base de datos | `eda_platform` |
| Usuario | `eda` |
| Contraseña | `eda_local_dev` (la de `.env.docker.example`) |

Tras login en la app y ejecutar el pipeline, revisa las tablas `users` y `analysis_runs`.

### Comandos útiles

```bash
docker compose ps
docker compose logs -f postgres
docker compose down          # parar
docker compose down -v       # parar y borrar datos (volumen)
```

### SQLite (sin Docker)

Por defecto en `.env.example`: `sqlite:///./eda_platform.db`. Adecuado solo para pruebas locales sin DBeaver/Postgres.

## Despliegue (back + front + Postgres)

Misma idea que en local: API con `DATABASE_URL` apuntando a Postgres gestionado.

| Pieza | Sugerencia |
|--------|------------|
| **PostgreSQL** | [Neon](https://neon.tech) o [Supabase](https://supabase.com) (gratis) — copia la URL en `DATABASE_URL` |
| **Backend** | [Render](https://render.com) / [Railway](https://railway.app) — repo FastAPI, `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Frontend** | [Vercel](https://vercel.com) — build con `VITE_API_BASE=https://tu-api.onrender.com` |

Checklist producción:

1. `DATABASE_URL` = Postgres en la nube (no SQLite).
2. `JWT_SECRET` aleatorio y distinto al de desarrollo.
3. `CORS_ORIGINS` = URL exacta del frontend desplegado.
4. CSV `data/it_ops_synthetic_10000.csv` disponible en el servidor (o regenerar con el script).
5. Tras deploy: `python scripts/wait_for_db.py && python scripts/seed_user.py`.
6. Frontend: `npm run build` con `VITE_API_BASE` apuntando a la API pública.

En un VPS puedes reutilizar el mismo `docker-compose.yml` (solo servicio `postgres`) o levantar Postgres + API con Docker.


