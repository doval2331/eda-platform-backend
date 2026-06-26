# API Endpoints — EDA Platform Backend

Base URL local: `http://127.0.0.1:8000`

Documentación interactiva (Swagger): `http://127.0.0.1:8000/docs`

## Autenticación

Todos los endpoints bajo `/api/*` (excepto login) requieren JWT Bearer:

```http
Authorization: Bearer <token>
```

Obtener token: `POST /api/auth/login`

---

## Resumen rápido

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/api/auth/login` | No | Iniciar sesión |
| GET | `/api/auth/me` | Sí | Usuario actual |
| POST | `/api/datasets/upload` | Sí | Subir dataset |
| GET | `/api/datasets/{dataset_id}` | Sí | Perfil de dataset |
| POST | `/api/projects` | Sí | Crear proyecto |
| GET | `/api/projects` | Sí | Listar proyectos |
| GET | `/api/projects/{project_id}` | Sí | Detalle de proyecto |
| PATCH | `/api/projects/{project_id}` | Sí | Actualizar proyecto |
| POST | `/api/projects/{project_id}/sources` | Sí | Añadir fuente al proyecto |
| DELETE | `/api/projects/{project_id}/sources/{source_id}` | Sí | Eliminar fuente |
| POST | `/api/projects/{project_id}/runs` | Sí | Ejecutar pipeline del proyecto |
| POST | `/api/runs` | Sí | Crear ejecución (run) |
| GET | `/api/runs` | Sí | Listar ejecuciones |
| GET | `/api/runs/{run_id}` | Sí | Detalle de ejecución |
| DELETE | `/api/runs/{run_id}` | Sí | Eliminar una ejecución |
| DELETE | `/api/runs` | Sí | Borrar todo el historial |
| GET | `/api/runs/{run_id}/cluster-profiles` | Sí | Perfiles de clusters |
| POST | `/api/runs/{run_id}/chat` | Sí | Chat conversacional |
| GET | `/api/runs/{run_id}/chat/history` | Sí | Historial de chat |
| POST | `/api/runs/{run_id}/chat/messages` | Sí | Añadir mensaje al historial |
| GET | `/api/runs/{run_id}/chat/suggestions` | Sí | Preguntas sugeridas |
| POST | `/api/runs/{run_id}/agents/strategy` | Sí | Agente de estrategia |
| POST | `/api/runs/{run_id}/agents/interpretation` | Sí | Agente de interpretación |
| POST | `/api/runs/{run_id}/agents/human-decision` | Sí | Decisión humana (HITL) |
| GET | `/api/runs/{run_id}/agents/results` | Sí | Resultados de agentes |
| GET | `/api/runs/{run_id}/agents/traces` | Sí | Trazabilidad de agentes |
| GET | `/api/projects/{project_id}/agents/traces` | Sí | Trazas del proyecto |
| POST | `/api/runs/{run_id}/insights/select` | Sí | Seleccionar insight |
| POST | `/api/runs/{run_id}/insights/select/batch` | Sí | Selección masiva |
| GET | `/api/conversation-dashboard` | Sí | Dashboard de insights |
| GET | `/api/metabase/status` | Sí | Estado BI / Metabase |
| GET | `/api/metabase/embed-token` | Sí | Token JWT para embed |
| POST | `/api/metabase/dashboard` | Sí | Crear dashboard Metabase |
| POST | `/api/bi-sync` | Sí | Sincronizar tablas BI |
| POST | `/api/runs/{run_id}/bi-sync` | Sí | Sincronizar BI de un run |

---

## 1. Health

### `GET /health`

Comprueba que la API y la base de datos responden.

**Auth:** no requerida

**Respuesta 200**

```json
{
  "status": "ok",
  "database": "ok"
}
```

---

## 2. Autenticación

### `POST /api/auth/login`

**Body (JSON)**

```json
{
  "email": "analista@tfm.local",
  "password": "TfmDemo2026!"
}
```

**Respuesta 200**

```json
{
  "token": "<jwt>",
  "token_type": "bearer",
  "user": {
    "id": "...",
    "email": "analista@tfm.local",
    "nombre": "Analista TFM",
    "activo": true
  }
}
```

**Errores:** `401` credenciales inválidas

---

### `GET /api/auth/me`

Devuelve el usuario autenticado.

**Respuesta 200:** `UserPublic`

---

## 3. Datasets

### `POST /api/datasets/upload`

Sube un archivo CSV/Excel/texto y devuelve su perfil analítico.

**Content-Type:** `multipart/form-data`

| Campo | Tipo | Requerido |
|-------|------|-----------|
| `file` | archivo | Sí |

**Respuesta 201:** `DatasetProfileResponse`

```json
{
  "dataset_id": "uuid",
  "filename": "datos.csv",
  "n_rows": 1000,
  "n_cols": 12,
  "numeric_columns": ["..."],
  "categorical_columns": ["..."],
  "excluded_columns": [],
  "suggested_id_column": "id",
  "all_columns": ["..."]
}
```

---

### `GET /api/datasets/{dataset_id}`

Perfil de un dataset previamente subido.

**Errores:** `403` sin permiso, `404` no encontrado

---

## 4. Proyectos

Un **proyecto** agrupa varias fuentes (CSV, notas, etc.) y define la estrategia de análisis:

- `per_source` — una ejecución por cada fuente tabular
- `unified` — solo la fuente principal de incidentes
- `merged` — fusiona todas las fuentes tabulares en un solo dataset

### `POST /api/projects`

**Body (JSON)**

```json
{
  "name": "Análisis Q1",
  "description": "Incidentes y cambios",
  "strategy": "per_source"
}
```

**Respuesta 201:** `ProjectDetail`

---

### `GET /api/projects`

**Query:** `limit` (default 50)

**Respuesta 200:** `ProjectSummary[]`

---

### `GET /api/projects/{project_id}`

**Respuesta 200:** `ProjectDetail` (incluye fuentes)

---

### `PATCH /api/projects/{project_id}`

**Body (JSON)** — todos los campos opcionales:

```json
{
  "name": "Nuevo nombre",
  "description": "...",
  "strategy": "merged"
}
```

---

### `POST /api/projects/{project_id}/sources`

Añade una fuente al proyecto.

**Content-Type:** `multipart/form-data`

| Campo | Tipo | Requerido |
|-------|------|-----------|
| `source_type` | enum | Sí |
| `file` | archivo | Sí |
| `source_name` | string | No |

**Valores de `source_type`:** `incidents`, `change_mgmt`, `software`, `hardware`, `dictionary`, `notes`, `other`

---

### `DELETE /api/projects/{project_id}/sources/{source_id}`

Elimina una fuente del proyecto.

---

### `POST /api/projects/{project_id}/runs`

Ejecuta el pipeline sobre las fuentes del proyecto según su estrategia.

**Body (JSON)**

```json
{
  "reduction_method": "UMAP",
  "seed": 42,
  "n_samples": 2000,
  "id_column": "incident_id",
  "exclude_columns": [],
  "numeric_columns": null,
  "categorical_columns": null
}
```

**Valores de `reduction_method`:** `PCA`, `t-SNE`, `UMAP`

**Respuesta 201:** `ProjectRunResponse`

```json
{
  "project_id": "...",
  "project_name": "...",
  "strategy": "per_source",
  "primary_run_id": "...",
  "runs": [ /* RunDetail[] */ ]
}
```

---

## 5. Ejecuciones (Runs)

### `POST /api/runs`

Crea una ejecución individual del pipeline (sin proyecto).

**Body (JSON)**

```json
{
  "modality": "tabular",
  "reduction_method": "UMAP",
  "seed": 42,
  "n_samples": 2000,
  "dataset_id": "uuid-del-upload",
  "id_column": null,
  "exclude_columns": [],
  "numeric_columns": null,
  "categorical_columns": null,
  "project_name": "Demo",
  "source_type": "incidents"
}
```

**Valores de `modality`:** `texto`, `imagen`, `multimodal`, `it_ops`, `tabular`

> Para `tabular` es obligatorio `dataset_id`.

**Respuesta 201:** `RunDetail` (incluye `result` con coordenadas 2D, labels, métricas y metadata)

---

### `GET /api/runs`

**Query:** `limit` (1–100, default 20)

**Respuesta 200:** `RunSummary[]`

---

### `GET /api/runs/{run_id}`

Detalle completo de una ejecución.

**Respuesta 200:** `RunDetail`

---

### `DELETE /api/runs/{run_id}`

Elimina una ejecución y sus datos en DuckDB/BI.

**Respuesta 200:** `RunDeleteResponse`

---

### `DELETE /api/runs`

Borra **todas** las ejecuciones del historial.

**Respuesta 200:** `RunResetResponse`

---

### `GET /api/runs/{run_id}/cluster-profiles`

Perfil operativo de cada cluster (estadísticas, modo de visualización recomendado).

**Respuesta 200**

```json
{
  "run_id": "...",
  "n_clusters": 5,
  "modo_viz": "radar",
  "stats_globales": { },
  "perfiles": [ ]
}
```

---

## 6. Chat conversacional

### `POST /api/runs/{run_id}/chat`

Pregunta sobre los resultados de una ejecución.

**Body (JSON)**

```json
{
  "question": "¿Qué cluster tiene más SLA incumplido?",
  "history": [
    { "role": "user", "text": "..." },
    { "role": "assistant", "text": "..." }
  ]
}
```

**Respuesta 200:** `ChatResponse`

```json
{
  "answer": "...",
  "suggested_questions": ["..."],
  "insights": [ ],
  "llm_used": false,
  "llm_mode": "rules",
  "llm_detail": null
}
```

---

### `GET /api/runs/{run_id}/chat/history`

Historial persistido del chat.

**Respuesta 200:** `ChatHistoryResponse`

---

### `POST /api/runs/{run_id}/chat/messages`

Añade una nota/mensaje al historial (p. ej. respuesta del asistente externo).

**Body (JSON)**

```json
{
  "role": "assistant",
  "text": "Nota del analista",
  "metadata": {}
}
```

---

### `GET /api/runs/{run_id}/chat/suggestions`

Preguntas sugeridas según el contexto del run.

**Respuesta 200:** `ChatSuggestionsResponse`

---

## 7. Agentes (IA)

### `POST /api/runs/{run_id}/agents/strategy`

Agente que recomienda estrategias analíticas sobre las evidencias del run.

**Body (JSON)**

```json
{
  "sample_size": 30,
  "sample_criteria": "priority",
  "model_name": "auto"
}
```

**Valores de `sample_criteria`:** `priority`, `random`, `mixed`

**Respuesta 200:** `AgentServiceResponse`

---

### `POST /api/runs/{run_id}/agents/interpretation`

Agente que interpreta clusters y genera insights.

**Body (JSON)**

```json
{
  "sample_size": 30,
  "sample_criteria": "priority",
  "random_state": 42,
  "model_name": "auto"
}
```

---

### `POST /api/runs/{run_id}/agents/human-decision`

Registra la validación humana (human-in-the-loop).

**Body (JSON)**

```json
{
  "decision_type": "strategy_approval",
  "status": "approved",
  "summary": "Estrategia validada por el analista.",
  "approved_strategy_ids": ["s1", "s2"],
  "parameters": {},
  "model_name": "human-in-the-loop"
}
```

**Valores de `status`:** `approved`, `rejected`, `needs_review`

---

### `GET /api/runs/{run_id}/agents/results`

Recomendaciones, insights y flag de trazas.

**Respuesta 200:** `AgentResultsResponse`

---

### `GET /api/runs/{run_id}/agents/traces`

Trazabilidad de decisiones de agentes.

**Query:** `limit` (1–500, opcional)

**Respuesta 200:** `AgentTraceResponse`

**Errores:** `404` si no hay trazas

---

### `GET /api/projects/{project_id}/agents/traces`

Trazas agregadas de todas las ejecuciones del proyecto.

**Query:** `limit` (1–500, opcional)

---

## 8. Insights seleccionados

### `POST /api/runs/{run_id}/insights/select`

Marca un insight como seleccionado para el dashboard conversacional.

**Body (JSON)**

```json
{
  "insight": {
    "id": "ins-1",
    "title": "Cluster crítico",
    "description": "...",
    "metric_label": "SLA breach",
    "metric_value": 0.42,
    "dimension": "cluster",
    "filter_kind": "cluster_id",
    "filter_value": "2"
  }
}
```

**Respuesta 200:** `{ "status": "ok" }`

---

### `POST /api/runs/{run_id}/insights/select/batch`

Selección masiva (máx. 50 insights).

**Body (JSON)**

```json
{
  "insights": [ /* InsightCandidate[] */ ]
}
```

**Respuesta 200:** `{ "status": "ok", "saved": 3 }`

---

### `GET /api/conversation-dashboard`

Insights seleccionados para el dashboard.

**Query:** `run_id` (opcional — filtra por ejecución)

**Respuesta 200:** `ConversationDashboardResponse`

---

## 9. BI y Metabase

Requiere `BI_SYNC_ENABLED=true` y PostgreSQL configurado (`BI_DATABASE_URL`).

### `GET /api/metabase/status`

Estado de sincronización BI, tablas `bi_*` y enlaces al dashboard.

**Respuesta 200:** `MetabaseStatusResponse`

---

### `GET /api/metabase/embed-token`

Genera token JWT para incrustar el dashboard de Metabase en el frontend.

**Query:** `run_id` (opcional)

**Respuesta 200**

```json
{
  "status": "ok",
  "token": "...",
  "instance_url": "https://metabase.ejemplo.com",
  "embed_url": "https://...",
  "dashboard_id": 2,
  "expires_in_seconds": 600
}
```

**Errores:** `400` si el embedding no está configurado

---

### `POST /api/metabase/dashboard`

Crea (o recrea) el dashboard de conversación en Metabase vía API.

**Respuesta 200:** `MetabaseDashboardCreateResponse`

---

### `POST /api/bi-sync`

Sincroniza todas las tablas BI desde DuckDB → PostgreSQL.

**Respuesta 200:** `BiSyncResponse`

---

### `POST /api/runs/{run_id}/bi-sync`

Sincroniza solo los datos de una ejecución concreta.

**Respuesta 200:** `BiSyncResponse`

---

## Códigos de error habituales

| Código | Significado |
|--------|-------------|
| `400` | Validación (body, estrategia de proyecto, dataset faltante…) |
| `401` | Token ausente, expirado o credenciales inválidas |
| `403` | Sin permiso sobre el recurso |
| `404` | Run, proyecto, dataset o trazas no encontrados |
| `422` | Sin evidencias materializadas (agentes) |
| `500` | Error interno al guardar archivos o sincronizar BI |

---

## Ejemplo de flujo completo

```bash
# 1. Login
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"analista@tfm.local","password":"TfmDemo2026!"}'

# 2. Subir CSV
curl -X POST http://127.0.0.1:8000/api/datasets/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@datos.csv"

# 3. Crear run tabular
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"modality":"tabular","reduction_method":"UMAP","dataset_id":"<uuid>"}'

# 4. Chat sobre el run
curl -X POST http://127.0.0.1:8000/api/runs/<run_id>/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"Resume los clusters principales"}'

# 5. Health check (sin auth)
curl http://127.0.0.1:8000/health
```
