# EDA Platform - Backend FastAPI

API REST para el prototipo TFM de analisis exploratorio de datasets IT. Ejecuta carga de datos, reduccion dimensional, clustering, persistencia de resultados, exploracion conversacional y dashboard de insights seleccionados.

## Requisitos

- Python 3.11 o superior. Recomendado: Python 3.12 estable.
- Git.
- Opcional: Docker Desktop, si se quiere usar PostgreSQL y Metabase para la capa BI.

DuckDB no se instala como servidor. Se usa como libreria embebida de Python y queda instalado con `pip install -r requirements.txt`.

## Instalacion

```powershell
cd "C:\Users\Marisol Altamiranda\eda-platform-backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Si el usuario demo todavia no existe:

```powershell
python scripts/seed_user.py
```

Credenciales demo:

- Email: `analista@tfm.local`
- Password: `TfmDemo2026!`

## Base de datos

La solucion usa tres capas de persistencia cuando se usa la integracion BI completa:

- `eda_platform.db`: SQLite por defecto para usuarios, login e historial de ejecuciones.
- `data/eda_platform.duckdb`: DuckDB para resultados analiticos, evidencias, clusters e insights seleccionados.
- PostgreSQL: capa BI `bi_*` para que Metabase consuma datos con su driver oficial.

No hay que levantar DuckDB manualmente. El archivo se crea automaticamente cuando arranca la API o cuando se ejecuta el pipeline.

Variables principales en `.env`:

| Variable | Valor local recomendado |
|----------|--------------------------|
| `DATABASE_URL` | `sqlite:///./eda_platform.db` |
| `DUCKDB_PATH` | `data/eda_platform.duckdb` |
| `IT_OPS_DATASET_PATH` | `data/it_ops_synthetic_10000.csv` |
| `DEFAULT_N_SAMPLES` | `2000` |
| `BI_SYNC_ENABLED` | `false` sin Docker, `true` con `.env.docker.example` |
| `BI_DATABASE_URL` | `postgresql+psycopg2://eda:eda_local_dev@127.0.0.1:5432/eda_platform` |
| `METABASE_URL` | `http://localhost:3000` |
| `METABASE_USERNAME` | usuario admin de Metabase, por ejemplo `analista@tfm.local` |
| `METABASE_PASSWORD` | password del usuario admin de Metabase |
| `METABASE_DATABASE_NAME` | `TFM IT Analytics` |
| `METABASE_DASHBOARD_NAME` | `Dashboard IT - Evidencias conversacionales` |
| `LLM_ENABLED` | `false` por defecto; `true` para activar explicacion asistida por LLM |
| `LLM_PROVIDER` | `openai_compatible` |
| `LLM_API_BASE` | `https://api.openai.com/v1` o endpoint compatible |
| `LLM_API_KEY` | API key del proveedor LLM |
| `LLM_MODEL` | modelo compatible; para Ollama local se usa `qwen2.5:1.5b`, para OpenAI puede usarse `gpt-4.1-mini` |
| `METABASE_PG_HOST` | `postgres` si Metabase corre en Docker |
| `METABASE_PG_DBNAME` | `eda_platform` |
| `METABASE_PG_USER` | `eda` |
| `METABASE_PG_PASSWORD` | `eda_local_dev` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174` |

## Generar dataset IT de ejemplo

Mientras no haya dataset real de la empresa, se puede generar un dataset sintetico IT Ops:

```powershell
python scripts/generate_it_ops_dataset.py
```

Salida esperada:

```text
data/it_ops_synthetic_10000.csv
```

Opcional para pruebas mas grandes:

```powershell
python scripts/generate_it_ops_dataset.py --n 50000 --output data/it_ops_synthetic_50000.csv
```

## Generar dataset sintetico de incidencias IT

Para probar la propuesta de clustering con incidencias individuales:

```powershell
python scripts/generate_it_incidents_dataset.py
```

Salida esperada:

```text
data/it_incidents_synthetic_2000.csv
```

Tambien se puede cambiar el tamano:

```powershell
python scripts/generate_it_incidents_dataset.py --n 5000 --output data/it_incidents_synthetic_5000.csv
```

Columnas principales:

```text
incident_id
categoria
subcategoria
prioridad
servicio_afectado
canal_entrada
tiempo_resolucion_horas
sla_incumplido
reaperturas
escalados
satisfaccion_usuario
coste_estimado
descripcion_corta
causa_raiz_simulada
synthetic_segment
```

Segmentos simulados incluidos:

```text
Incidencias simples de bajo impacto
Incidencias criticas recurrentes
Incidencias complejas de larga resolucion
Incidencias asociadas a cambios
Incidencias de seguridad
Casos anomalos
```

`synthetic_segment` se incluye solo para evaluacion posterior. El perfilador tabular lo excluye del modelado por defecto para evitar fuga de informacion.

Para probarlo en la interfaz:

1. Levantar backend y frontend.
2. Ir a `Analisis exploratorio`.
3. Subir `data/it_incidents_synthetic_2000.csv`.
4. Usar modalidad `tabular`.
5. Verificar que `incident_id`, `descripcion_corta` y `synthetic_segment` queden excluidas del modelado.
6. Ejecutar el pipeline.
7. Probar el chat con preguntas sobre SLA, prioridades, causas raiz, anomalias y clusters criticos.
## Esquema de incidencias y metricas

El pipeline analiza incidencias IT (no cuentas cliente). Columnas de texto (`descripcion_corta`) y evaluacion (`segment`, `synthetic_segment`) quedan fuera del modelado; los segmentos sinteticos solo sirven para ARI/NMI posteriores.

Documentacion completa: [`docs/dataset_schema.md`](docs/dataset_schema.md).

Metricas devueltas por cada ejecucion (`metrics` en la respuesta):

| Metrica | Descripcion |
|---------|-------------|
| `silhouette` | Cohesion/separacion en el espacio 2D |
| `davies_bouldin` | Dispersion intra/inter cluster (menor es mejor) |
| `calinski_harabasz` | Separacion en el espacio de features original |
| `n_clusters` | Clusters HDBSCAN (sin ruido) |
| `noise_pct` | Porcentaje de puntos marcados como ruido (-1) |
| `ari` | Adjusted Rand Index vs segmento sintetico (si existe) |
| `nmi` | Normalized Mutual Information vs segmento sintetico |
| `cluster_stability` | Acuerdo ARI entre dos HDBSCAN con parametros ligeramente distintos |

Las metricas se persisten en SQLite (`result_json`), DuckDB (`run_registry`) y, si BI esta activo, PostgreSQL (`bi_runs`).

## Ejecutar API

Puerto local recomendado para este proyecto:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Documentacion Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

Si preferis usar otro puerto, tambien funciona, pero el frontend debe apuntar al mismo puerto.

## Endpoints principales

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio |
| `POST` | `/api/auth/login` | Login JWT |
| `POST` | `/api/datasets/upload` | Carga CSV del usuario |
| `GET` | `/api/datasets/{dataset_id}` | Perfil del dataset cargado |
| `POST` | `/api/runs` | Ejecuta pipeline y persiste resultado |
| `GET` | `/api/runs` | Lista ejecuciones |
| `GET` | `/api/runs/{id}` | Detalle de una ejecucion |
| `POST` | `/api/runs/{id}/chat` | Exploracion conversacional del run |
| `POST` | `/api/runs/{id}/insights/select` | Guarda un insight para dashboard |
| `GET` | `/api/conversation-dashboard` | Devuelve dashboard conversacional |
| `GET` | `/api/metabase/status` | Estado de la capa BI/Metabase |
| `POST` | `/api/metabase/dashboard` | Publica tablas BI y crea dashboard base en Metabase |
| `POST` | `/api/bi-sync` | Publica todas las tablas BI en PostgreSQL |
| `POST` | `/api/runs/{id}/bi-sync` | Publica un run en PostgreSQL BI |

## Ejemplo de ejecucion IT Ops

```json
{
  "modality": "it_ops",
  "reduction_method": "UMAP",
  "seed": 42,
  "n_samples": 2000
}
```

Tambien se puede usar modalidad `tabular` subiendo un CSV desde el frontend.

## Exploracion conversacional y agente LLM

El chat funciona como agente controlado por herramientas analiticas internas. Por defecto es local y guiado: no se conecta a un chatbot externo ni envia datos sensibles a internet.

Flujo:

1. El usuario ejecuta el pipeline.
2. El backend persiste evidencias, clusters y metricas en DuckDB.
3. El usuario pregunta desde el frontend: por ejemplo `que servicios incumplen mas SLA`.
4. El backend ejecuta herramientas internas como `analizar_sla`, `analizar_clusters`, `analizar_anomalias`, `analizar_causas_raiz` o `analizar_prioridades`.
5. Las herramientas devuelven resumenes agregados e insights candidatos.
6. Si el usuario pregunta por decisiones, el backend agrega una herramienta `alternativas_decision`, que compara SLA, demora, riesgo, volumen, causas y anomalias para generar opciones de priorizacion.
7. Si `LLM_ENABLED=true`, el backend envia solo esos resumenes agregados al LLM para redactar una explicacion simple y sugerir alternativas interpretativas.
8. Si `LLM_ENABLED=false`, se usa la respuesta local por reglas.
9. El usuario selecciona insights o alternativas.
10. Los insights quedan disponibles en `/api/conversation-dashboard`.
11. Si `BI_SYNC_ENABLED=true`, el backend publica tablas `bi_*` en PostgreSQL para Metabase.

Activacion opcional del LLM:

```env
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=tu_api_key
LLM_MODEL=gpt-4.1-mini
```

Tambien puede usarse un servidor local compatible con OpenAI, por ejemplo Ollama o LM Studio, cambiando `LLM_API_BASE` y `LLM_MODEL`.

Configuracion local con Ollama:

```env
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_API_BASE=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:1.5b
LLM_TIMEOUT_SECONDS=120
```

Instalacion y modelo local:

```powershell
winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
ollama pull qwen2.5:1.5b
```

El LLM no consulta DuckDB directamente, no recibe filas completas, no calcula clusters y no toma decisiones automaticamente. Solo traduce resultados agregados generados por el backend y ayuda a presentar alternativas de priorizacion para que el usuario las evalue.

## Metabase BI con DuckDB como base principal

La arquitectura BI es:

```text
FastAPI / pipeline
  -> DuckDB (base analitica principal)
  -> PostgreSQL (tablas bi_* para serving BI)
  -> Metabase (dashboards con driver oficial PostgreSQL)
```

DuckDB sigue siendo la fuente analitica principal. PostgreSQL se usa como capa de publicacion porque Metabase se conecta de forma oficial y estable a PostgreSQL.

Para levantar PostgreSQL + Metabase:

```powershell
docker compose up -d
Copy-Item .env.docker.example .env
python scripts/wait_for_db.py
python scripts/seed_user.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Si ya existen los contenedores locales del TFM:

```text
tfm-analytics-db  # PostgreSQL en 5432
tfm-metabase      # Metabase en 3000
```

levantar ambos:

```powershell
docker start tfm-analytics-db
docker start tfm-metabase
```

y configurar `.env` del backend asi:

```env
BI_SYNC_ENABLED=true
BI_DATABASE_URL=postgresql+psycopg2://tfm:tfm@127.0.0.1:5432/tfm_it
METABASE_URL=http://localhost:3000
METABASE_DASHBOARD_URL=
METABASE_USERNAME=analista@tfm.local
METABASE_PASSWORD=TfmDemo2026!
METABASE_DATABASE_NAME=TFM IT Analytics
METABASE_DASHBOARD_NAME=Dashboard IT - Evidencias conversacionales
METABASE_PG_HOST=tfm-analytics-db
METABASE_PG_PORT=5432
METABASE_PG_DBNAME=tfm_it
METABASE_PG_USER=tfm
METABASE_PG_PASSWORD=tfm
```

Despues reiniciar la API:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

URLs:

```text
API:      http://127.0.0.1:8000
Metabase: http://localhost:3000
```

Tablas publicadas para Metabase:

```text
bi_runs
bi_evidences
bi_cluster_summary
bi_sla_by_category
bi_service_risk
bi_selected_insights
```

El backend puede crear la conexion PostgreSQL en Metabase automaticamente con
las variables `METABASE_PG_*`. Si se configura manualmente en Metabase, usar:

| Campo | Valor |
|-------|-------|
| Host | `postgres` si Metabase corre en Docker, `127.0.0.1` desde fuera de Docker |
| Puerto | `5432` |
| Base de datos | `eda_platform` |
| Usuario | `eda` |
| Password | `eda_local_dev` |

Para los contenedores locales `tfm-*`, usar:

| Campo | Valor |
|-------|-------|
| Host | `tfm-analytics-db` desde Metabase Docker, `127.0.0.1` desde fuera de Docker |
| Puerto | `5432` |
| Base de datos | `tfm_it` |
| Usuario | `tfm` |
| Password | `tfm` |

Filtros recomendados en dashboards:

```text
run_id
cluster_label
category
severity
affected_service
```

### Crear dashboard automatico en Metabase

La pantalla React `Metabase BI` incluye el boton `Crear dashboard en Metabase`.
Ese boton:

1. Publica las tablas `bi_*` desde DuckDB hacia PostgreSQL.
2. Inicia sesion contra Metabase con `METABASE_USERNAME` y `METABASE_PASSWORD`.
3. Busca o crea la base `METABASE_DATABASE_NAME`.
4. Crea el dashboard `METABASE_DASHBOARD_NAME`.
5. Agrega tarjetas para SLA, severidad, servicios, tiempos, clusters e insights seleccionados.

Tambien se puede crear desde API:

```powershell
# Requiere token Bearer de login
POST http://127.0.0.1:8000/api/metabase/dashboard
```

La respuesta devuelve `dashboard_url`. Si queres dejar un dashboard fijo como acceso directo, copia esa URL en `.env`:

```env
METABASE_DASHBOARD_URL=http://localhost:3000/dashboard/ID_DEL_DASHBOARD
```

Si el volumen de PostgreSQL ya existia antes de agregar Metabase y falla porque no existe la base `metabase`, crearla una vez:

```powershell
docker compose exec postgres psql -U eda -d eda_platform -c "CREATE DATABASE metabase OWNER eda;"
```

### Publicar datos para Metabase

Desde la app React, entrar a `Metabase BI` y presionar `Publicar tablas BI`.

Tambien se puede publicar desde API:

```powershell
# Requiere token Bearer de login
POST http://127.0.0.1:8000/api/bi-sync
POST http://127.0.0.1:8000/api/runs/{run_id}/bi-sync
```

Estado esperado:

```json
{
  "enabled": true,
  "metabase_url": "http://localhost:3000",
  "postgres_status": "ok",
  "detail": "PostgreSQL BI disponible"
}
```

### Troubleshooting Metabase

Si `http://localhost:3000` muestra `ERR_CONNECTION_REFUSED`, revisar que el contenedor este corriendo:

```powershell
docker ps -a --filter "name=tfm-metabase"
docker start tfm-metabase
```

Metabase puede tardar varios minutos la primera vez o despues de una actualizacion porque migra su base interna H2. Revisar logs:

```powershell
docker logs --tail 120 tfm-metabase
```

Cuando termine de iniciar, `http://localhost:3000` debe responder.

Si la pantalla React `Metabase BI` muestra `Not Found`, cerrar sesion, volver a iniciar sesion y refrescar con `Ctrl + F5`. En desarrollo el frontend usa el proxy de Vite hacia `http://127.0.0.1:8000`.

### Backup para mover a otra laptop

Exportar PostgreSQL:

```powershell
docker exec tfm-analytics-db pg_dump -U tfm -Fc -d tfm_it -f /tmp/tfm_it.dump
docker cp tfm-analytics-db:/tmp/tfm_it.dump "C:\tfm-backup\tfm_it.dump"
```

Exportar Metabase H2:

```powershell
docker run --rm `
  -v "tfm2_metabase-data:/metabase-data" `
  -v "C:\tfm-backup:/backup" `
  alpine sh -c "tar czf /backup/metabase-data.tar.gz -C /metabase-data ."
```

Copiar tambien DuckDB si se quiere conservar el estado analitico local:

```powershell
Copy-Item "data\eda_platform.duckdb" "C:\tfm-backup\eda_platform.duckdb"
```

Archivos esperados:

```text
C:\tfm-backup\tfm_it.dump
C:\tfm-backup\metabase-data.tar.gz
C:\tfm-backup\eda_platform.duckdb
```

## Notebook de experimentacion

```powershell
jupyter notebook notebooks/01_it_ops_eda.ipynb
```

El notebook reutiliza modulos de `app/services/`, de modo que notebook y API comparten logica.

## PostgreSQL con Docker opcional

SQLite es suficiente para pruebas locales sin BI. Si se quiere PostgreSQL para usuarios, historial y Metabase:

```powershell
docker compose up -d
Copy-Item .env.docker.example .env
python scripts/wait_for_db.py
python scripts/seed_user.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Datos DBeaver:

| Campo | Valor |
|-------|-------|
| Host | `127.0.0.1` |
| Puerto | `5432` |
| Base de datos | `eda_platform` |
| Usuario | `eda` |
| Password | `eda_local_dev` |

## Estructura relevante

```text
app/api/routes.py                  # Endpoints FastAPI
app/services/pipeline.py           # Orquestacion del pipeline
app/services/pipeline_core.py      # Reduccion dimensional + HDBSCAN
app/services/duckdb_store.py       # Persistencia analitica DuckDB
app/services/conversation.py       # Motor conversacional guiado
app/services/dataset_store.py      # Carga/perfilado de CSV
scripts/generate_it_ops_dataset.py # Generador sintetico IT Ops
scripts/seed_user.py               # Usuario demo
data/                              # CSV, SQLite y DuckDB locales
```

## Checklist para otro ambiente

1. Crear `.venv`.
2. Instalar `requirements.txt`.
3. Copiar `.env.example` a `.env`.
4. Generar dataset sintetico o cargar CSV desde el frontend.
5. Ejecutar `python scripts/seed_user.py`.
6. Levantar API en `127.0.0.1:8000`.
7. Levantar frontend apuntando al mismo puerto.
