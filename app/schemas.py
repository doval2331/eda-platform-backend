from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["texto", "imagen", "multimodal", "it_ops", "tabular"]
ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]
ProjectStrategy = Literal["per_source", "unified"]
ProjectSourceType = Literal[
    "incidents",
    "change_mgmt",
    "software",
    "hardware",
    "dictionary",
    "notes",
    "other",
]


class EvidenceMetadata(BaseModel):
    id: str
    preview: str
    source: Modality
    incident_id: str | None = None
    categoria: str | None = None
    subcategoria: str | None = None
    prioridad: str | None = None
    servicio_afectado: str | None = None
    canal_entrada: str | None = None
    tiempo_resolucion_horas: float | None = None
    sla_incumplido: bool | None = None
    reaperturas: float | None = None
    escalados: float | None = None
    satisfaccion_usuario: float | None = None
    coste_estimado: float | None = None
    descripcion_corta: str | None = None
    causa_raiz_simulada: str | None = None
    synthetic_segment: str | None = None
    sector: str | None = None
    service_line: str | None = None
    support_channel: str | None = None
    segment: str | None = None
    category: str | None = None
    subcategory: str | None = None
    priority: str | None = None
    severity: str | None = None
    status: str | None = None
    assignment_group: str | None = None
    affected_service: str | None = None
    short_description: str | None = None
    root_cause_simulated: str | None = None
    monthly_tickets: float | None = None
    critical_incidents: float | None = None
    avg_resolution_hours: float | None = None
    resolution_minutes: float | None = None
    reopenings: float | None = None
    escalations: float | None = None
    sla_breach_rate: float | None = None
    sla_breached: bool | None = None
    operational_risk_score: float | None = None
    business_impact_score: float | None = None
    security_incidents: float | None = None
    downtime_hours: float | None = None
    customer_satisfaction: float | None = None
    estimated_cost: float | None = None
    features: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class PipelineMetrics(BaseModel):
    silhouette: float | None = None
    davies_bouldin: float | None = None
    calinski_harabasz: float | None = None
    n_clusters: int | None = None
    noise_pct: float | None = None
    ari: float | None = None
    nmi: float | None = None
    cluster_stability: float | None = None
    trustworthiness: float | None = None
    pca_variance_explained: float | None = None


class PipelineResult(BaseModel):
    X_2d: list[list[float]]
    cluster_labels: list[int]
    outliers_count: int
    metrics: PipelineMetrics
    metadata: list[EvidenceMetadata]
    baseline_algorithm: str = "DBSCAN"
    baseline_metrics: PipelineMetrics | None = None


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    filename: str
    normalized_kind: str | None = None
    original_format: str | None = None
    extraction_method: str | None = None
    n_rows: int
    n_cols: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]
    suggested_id_column: str | None = None
    all_columns: list[str]


class RunCreateBody(BaseModel):
    """Cuerpo JSON aceptado por el endpoint (snake_case y alias)."""

    modality: Modality = "it_ops"
    reduction_method: ReductionMethod = "UMAP"
    seed: int | None = None
    n_samples: int | None = Field(default=None, ge=30, le=10_000)
    dataset_id: str | None = None
    id_column: str | None = None
    exclude_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    project_name: str | None = Field(default=None, max_length=200)
    source_type: str | None = Field(default=None, max_length=32)


class ProjectCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    strategy: ProjectStrategy = "per_source"


class ProjectUpdateBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    strategy: ProjectStrategy | None = None


class ProjectSourceSummary(BaseModel):
    id: str
    source_type: ProjectSourceType
    source_name: str | None = None
    filename: str
    dataset_id: str | None = None
    processing_status: str = "processed"
    n_rows: int | None = None
    n_cols: int | None = None
    char_count: int | None = None
    word_count: int | None = None
    normalized_kind: str | None = None
    original_format: str | None = None
    extraction_method: str | None = None
    preview: str | None = None
    all_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str
    strategy: ProjectStrategy
    created_at: datetime
    updated_at: datetime
    source_count: int = 0
    csv_source_count: int = 0
    total_rows: int = 0


class ProjectDetail(ProjectSummary):
    sources: list[ProjectSourceSummary] = Field(default_factory=list)


class ProjectRunCreateBody(BaseModel):
    reduction_method: ReductionMethod = "UMAP"
    seed: int | None = None
    n_samples: int | None = Field(default=None, ge=30, le=10_000)
    id_column: str | None = None
    exclude_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None


class RunSummary(BaseModel):
    id: str
    created_at: datetime
    modality: Modality
    reduction_method: ReductionMethod
    seed: int
    n_samples: int
    outliers_count: int
    metrics: PipelineMetrics
    project_id: str | None = None
    project_name: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    source_name: str | None = None


class RunDetail(RunSummary):
    result: PipelineResult


class ProjectRunResponse(BaseModel):
    project_id: str
    project_name: str
    strategy: ProjectStrategy
    primary_run_id: str
    runs: list[RunDetail]


class HealthResponse(BaseModel):
    status: str
    database: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserPublic(BaseModel):
    id: str
    email: str
    nombre: str
    activo: bool = True


class LoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserPublic


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=8)


class InsightCandidate(BaseModel):
    id: str
    title: str
    description: str
    metric_label: str | None = None
    metric_value: float | None = None
    dimension: str | None = None
    filter_kind: str | None = None
    filter_value: str | None = None


class ChatResponse(BaseModel):
    answer: str
    suggested_questions: list[str] = Field(default_factory=list)
    insights: list[InsightCandidate] = Field(default_factory=list)
    llm_used: bool = False
    llm_mode: str = "rules"
    llm_detail: str | None = None


class ChatSuggestionsResponse(BaseModel):
    suggested_questions: list[str] = Field(default_factory=list)


class InsightSelectionBody(BaseModel):
    insight: InsightCandidate


class SelectedInsightDashboardItem(InsightCandidate):
    run_id: str
    selected_at: datetime
    run_created_at: datetime | None = None
    modality: Modality | None = None
    reduction_method: ReductionMethod | None = None
    evidence_count: int | None = None
    avg_sla_breach_rate: float | None = None
    avg_resolution_hours: float | None = None
    avg_risk: float | None = None


class ConversationDashboardResponse(BaseModel):
    total: int
    insights: list[SelectedInsightDashboardItem] = Field(default_factory=list)


class BiSyncResponse(BaseModel):
    status: str
    message: str
    tables: dict[str, int] = Field(default_factory=dict)


class MetabaseStatusResponse(BaseModel):
    enabled: bool
    metabase_url: str
    dashboard_url: str | None = None
    postgres_status: str
    detail: str | None = None
    tables: dict[str, int] = Field(default_factory=dict)


class MetabaseDashboardCard(BaseModel):
    id: int
    name: str
    url: str


class MetabaseDashboardCreateResponse(BaseModel):
    status: str
    message: str
    dashboard_id: int | None = None
    dashboard_url: str | None = None
    database_id: int | None = None
    cards: list[MetabaseDashboardCard] = Field(default_factory=list)


class AgentStrategyRequest(BaseModel):
    sample_size: int = Field(default=30, ge=1, le=500)
    sample_criteria: Literal["priority", "random", "mixed"] = "priority"
    model_name: str = "auto"


class AgentInterpretationRequest(BaseModel):
    sample_size: int = Field(default=30, ge=1, le=500)
    sample_criteria: Literal["priority", "random", "mixed"] = "priority"
    random_state: int = 42
    model_name: str = "auto"


class AgentHumanDecisionRequest(BaseModel):
    decision_type: str = Field(default="strategy_approval", min_length=1)
    status: Literal["approved", "rejected", "needs_review"] = "approved"
    summary: str = Field(default="Estrategia validada por el analista.", min_length=1)
    approved_strategy_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, object] = Field(default_factory=dict)
    model_name: str = "human-in-the-loop"


class AgentHumanDecisionResponse(BaseModel):
    status: str
    run_id: str
    trace_id: str
    message: str


class RunResetResponse(BaseModel):
    status: str
    deleted_runs: int
    duckdb_tables_cleared: dict[str, int] = Field(default_factory=dict)
    bi_tables_cleared: dict[str, int] | None = None
    message: str


class RunDeleteResponse(BaseModel):
    status: str
    run_id: str
    duckdb_tables_cleared: dict[str, int] = Field(default_factory=dict)
    bi_tables_cleared: dict[str, int] | None = None
    message: str


class AgentServiceResponse(BaseModel):
    status: str
    run_id: str
    trace_ids: list[str] = Field(default_factory=list)
    items: list[dict] = Field(default_factory=list)
    llm_used: bool = False
    llm_mode: str = "rules"
    llm_detail: str | None = None
    model_name: str = "deterministic-local"


class AgentTraceResponse(BaseModel):
    run_id: str
    trace_count: int
    traces: list[dict] = Field(default_factory=list)


class AgentResultsResponse(BaseModel):
    run_id: str
    recommendations: list[dict] = Field(default_factory=list)
    insights: list[dict] = Field(default_factory=list)
    has_traces: bool = False
