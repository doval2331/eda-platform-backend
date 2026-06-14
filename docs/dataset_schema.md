# Esquema del dataset de incidencias IT

Este documento define las columnas del dataset, su significado y cómo las usa el backend.

## Roles de columna

| Rol | Uso en pipeline | Ejemplos |
|-----|-----------------|----------|
| **Identificador** | Metadata / UI, nunca en modelado | `incident_id`, `client_id` |
| **Numérica** | Imputación mediana + StandardScaler + clustering | `tiempo_resolucion_horas`, `sla_breach_rate` |
| **Categórica** | Imputación moda + one-hot + clustering | `categoria`, `prioridad`, `servicio_afectado` |
| **Texto** | Tooltips, chat y LLM; **no** en modelado | `descripcion_corta`, `causa_raiz_simulada` |
| **Evaluación** | Solo ARI/NMI posteriores; **nunca** en modelado | `synthetic_segment`, `segment` |

## Esquema objetivo (incidencias)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `incident_id` | ID | Identificador único (`INC-000123`) |
| `categoria` | Categórica | Tipo general (red, aplicación, seguridad…) |
| `subcategoria` | Categórica | Subtipo (autenticación, latencia…) |
| `prioridad` | Categórica | baja, media, alta, crítica |
| `servicio_afectado` | Categórica | CRM, ERP, portal cliente… |
| `canal_entrada` | Categórica | email, portal, teléfono… |
| `tiempo_resolucion_horas` | Numérica | Horas hasta cierre |
| `sla_incumplido` / `sla_breach_rate` | Numérica / bool | Incumplimiento SLA |
| `reaperturas` / `reopen_rate` | Numérica | Reaperturas |
| `escalados` / `escalation_rate` | Numérica | Escalados |
| `satisfaccion_usuario` | Numérica | Valoración 1–5 o 1–10 |
| `coste_estimado` | Numérica | Coste operativo |
| `descripcion_corta` | Texto | Resumen para UI/LLM |
| `causa_raiz_simulada` | Texto | Causa raíz de referencia |
| `synthetic_segment` | Evaluación | Segmento latente simulado |

## Dataset legacy actual (`it_ops_synthetic_*.csv`)

Hasta recibir el CSV definitivo del profesor, el pipeline usa el dataset a nivel
**cliente/cuenta** con mapeo operativo:

| Legacy | Interpretación en UI |
|--------|----------------------|
| `client_id` | Identificador de registro |
| `sector` | Categoría / sector |
| `service_line` | Servicio afectado |
| `support_channel` | Canal de entrada |
| `avg_resolution_hours` | Tiempo de resolución |
| `sla_breach_rate` | Tasa SLA incumplido |
| `reopen_rate` | Reaperturas |
| `escalation_rate` | Escalados |
| `segment` | Segmento sintético (**solo evaluación**) |

## Preprocesamiento

1. Carga CSV y submuestreo opcional.
2. Resolución de grupos (`incidents_schema.resolve_incident_column_groups`).
3. Exclusión automática de identificadores, texto y `synthetic_segment`/`segment`.
4. Imputación: mediana (numéricas), moda (categóricas).
5. One-hot encoding + StandardScaler.
6. Reducción 2D (PCA / t-SNE / UMAP) → HDBSCAN (principal) + DBSCAN (baseline comparativo automático).

## Métricas devueltas por `POST /api/runs`

### HDBSCAN (`metrics`)

| Métrica | Descripción |
|---------|-------------|
| `silhouette` | Separación entre clusters |
| `davies_bouldin` | Compactación / separación |
| `calinski_harabasz` | Densidad entre grupos (espacio de features) |
| `n_clusters` | Clusters detectados |
| `noise_pct` | % ruido HDBSCAN (incidencias atípicas) |
| `ari` | ARI vs `synthetic_segment` (si existe) |
| `nmi` | NMI vs segmento sintético (si existe) |
| `cluster_stability` | Acuerdo entre dos HDBSCAN con parámetros cercanos |

### DBSCAN baseline (`baseline_metrics`)

Mismas métricas excepto `cluster_stability`. Se calculan sobre la misma proyección 2D solo para comparación metodológica; no alimentan scatter, metadatos ni agente.

## Endpoints y consumo

- **Frontend**: scatter 2D, KPIs, tooltips (`EvidenceMetadata`), pestaña Interpretación.
- **DuckDB**: `run_registry`, `run_evidences` para chat conversacional.
- **Metabase BI**: tablas `bi_runs`, `bi_evidences`, `bi_sla_by_category`, etc.

Definición en código: `app/services/incidents_schema.py`.
