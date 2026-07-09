from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["texto", "imagen", "multimodal", "it_ops", "tabular"]
ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]
ProjectStrategy = Literal["per_source", "unified", "merged"]
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
    pipeline_tuning_applied: dict[str, float | int] | None = None


class PipelineTuningFields(BaseModel):
    umap_n_neighbors: int | None = Field(default=None, ge=2, le=200)
    umap_min_dist: float | None = Field(default=None, ge=0.0, le=0.99)
    hdbscan_min_cluster_size: int | None = Field(default=None, ge=2, le=5000)
    hdbscan_min_samples: int | None = Field(default=None, ge=1, le=1000)
    dbscan_eps: float | None = Field(default=None, ge=0.001, le=10.0)


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


class RunCreateBody(PipelineTuningFields):
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
    relationship_status: str | None = None
    relationship_score: float | None = None
    relationship_reason: str | None = None
    content_summary: str | None = None


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


class ProjectSourceUploadJobResponse(BaseModel):
    job_id: str
    project_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    message: str
    filename: str
    source_type: ProjectSourceType
    source_name: str | None = None
    uploaded_bytes: int | None = None
    error: str | None = None
    project: ProjectDetail | None = None


class ProjectRunCreateBody(PipelineTuningFields):
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
    dataset_id: str | None = None


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
    question: str = Field(..., min_length=1, max_length=5000)
    display_question: str | None = Field(default=None, max_length=1200)
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
    document_context_used: bool = False


class ChatSuggestionsResponse(BaseModel):
    suggested_questions: list[str] = Field(default_factory=list)


class ChatMessageRecord(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    text: str
    insights: list[InsightCandidate] = Field(default_factory=list)
    llm_used: bool | None = None
    llm_detail: str | None = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessageRecord] = Field(default_factory=list)


class ChatAppendMessageBody(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str = Field(..., min_length=1, max_length=2000)
    metadata: dict[str, Any] | None = None


class InsightSelectionBody(BaseModel):
    insight: InsightCandidate


class InsightBatchSelectionBody(BaseModel):
    insights: list[InsightCandidate] = Field(..., min_length=1, max_length=50)


class InsightBatchSelectionResponse(BaseModel):
    status: str = "ok"
    saved: int


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


class SelectedInsightsResponse(BaseModel):
    total: int
    insights: list[SelectedInsightDashboardItem] = Field(default_factory=list)


DashboardPriority = Literal["alta", "media", "baja"]
DashboardAudience = Literal["funcional", "experto", "ambos"]
DashboardRecommendationAction = Literal["chart", "chat", "conclusion"]
DashboardVariableRole = Literal["business", "metric", "technical", "identifier", "unknown"]
DashboardChartType = Literal[
    "bar",
    "line",
    "scatter",
    "priority_matrix",
    "distribution",
    "ranking",
    "heatmap",
    "boxplot",
]
DashboardConfidence = Literal["alta", "media", "baja"]
DashboardEvidenceSource = Literal["dataset", "pipeline", "cluster", "insight", "llm", "user"]


class ConversationExecutiveSummary(BaseModel):
    title: str = "Resumen ejecutivo"
    dataset_name: str = ""
    analysis_objective: str = ""
    records_count: int = 0
    columns_count: int = 0
    main_variables: list[str] = Field(default_factory=list)
    key_metrics: list[str] = Field(default_factory=list)
    summary: str = ""


class ConversationSemanticVariable(BaseModel):
    name: str
    label: str = ""
    role: DashboardVariableRole = "unknown"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    recommended_use: str = ""
    avoid_as_metric: bool = False
    can_chart: bool = True
    semantic_type: str = ""
    source: str = ""
    confidence: Literal["alta", "media", "baja", ""] = ""
    active: bool = True


class ConversationPriorityFinding(BaseModel):
    id: str
    title: str
    priority: DashboardPriority = "media"
    impact: str = ""
    urgency: str = ""
    evidence: str = ""
    suggested_action: str = ""
    related_variables: list[str] = Field(default_factory=list)
    suggested_question: str = ""


class ConversationAgentRecommendation(BaseModel):
    id: str
    title: str
    why_it_matters: str = ""
    what_to_analyze: str = ""
    recommended_next_step: str = ""
    audience: DashboardAudience = "ambos"
    action_type: DashboardRecommendationAction = "chat"
    linked_visualization_id: str = ""
    evidence_needed: str = ""


class ConversationSuggestedVisualization(BaseModel):
    id: str
    title: str
    chart_type: DashboardChartType = "bar"
    x: str = ""
    y: str = ""
    metric: str = ""
    group_by: str = ""
    aggregation: str = ""
    filters: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    evidence_used: str = ""
    question_answered: str = ""
    audience: DashboardAudience = "ambos"
    what_i_am_seeing: str = ""
    why_it_matters: str = ""
    suggested_action: str = ""
    drilldown: str = ""


class ConversationActiveChartDefault(BaseModel):
    visualization_id: str = ""
    explanation: str = ""


class ConversationDashboardConclusion(BaseModel):
    id: str
    conclusion: str
    evidence: str = ""
    related_chart: str = ""
    related_metric: str = ""
    related_items: list[str] = Field(default_factory=list)
    confidence: DashboardConfidence = "media"
    recommended_action: str = ""
    source: str = ""
    evidence_quality: str = ""


class ConversationEvidenceLineStep(BaseModel):
    step: int
    title: str
    description: str = ""
    source: DashboardEvidenceSource = "dataset"
    related_items: list[str] = Field(default_factory=list)


class ConversationSuggestedQuestionGroups(BaseModel):
    functional_user: list[str] = Field(default_factory=list)
    expert_user: list[str] = Field(default_factory=list)


class ConversationOperationalReadiness(BaseModel):
    status: Literal["operational", "interpretive", "limited"] = "limited"
    run_scope: Literal["single_run", "multi_run", "empty"] = "empty"
    active_run_id: str = ""
    run_ids: list[str] = Field(default_factory=list)
    decision_level: Literal["operational", "assisted_review", "interpretive"] = "interpretive"
    evidence_mode: Literal["materialized", "partial", "interpretive"] = "interpretive"
    trust_level: Literal["alta", "media", "baja"] = "baja"
    evidence_materialized: bool = False
    evidence_records: int = 0
    evidence_runs: int = 0
    selected_insights: int = 0
    semantic_dictionary_configured: bool = False
    semantic_dictionary_source: str = ""
    semantic_dictionary_total: int = 0
    semantic_dictionary_configured_count: int = 0
    semantic_dictionary_active_count: int = 0
    semantic_dictionary_inactive_count: int = 0
    llm_validated: bool = False
    summary: str = ""
    functional_message: str = ""
    expert_message: str = ""
    recommended_next_step: str = ""
    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)


class ConversationDashboardSpec(BaseModel):
    schema_version: str = "conversation-dashboard/v1"
    contract_status: str = "valid"
    contract_warnings: list[str] = Field(default_factory=list)
    llm_risk_flags: list[str] = Field(default_factory=list)
    executive_summary: ConversationExecutiveSummary = Field(default_factory=ConversationExecutiveSummary)
    semantic_variables: list[ConversationSemanticVariable] = Field(default_factory=list)
    priority_findings: list[ConversationPriorityFinding] = Field(default_factory=list)
    agent_recommendations: list[ConversationAgentRecommendation] = Field(default_factory=list)
    suggested_visualizations: list[ConversationSuggestedVisualization] = Field(default_factory=list)
    active_chart_default: ConversationActiveChartDefault = Field(default_factory=ConversationActiveChartDefault)
    conclusions: list[ConversationDashboardConclusion] = Field(default_factory=list)
    evidence_line: list[ConversationEvidenceLineStep] = Field(default_factory=list)
    suggested_questions: ConversationSuggestedQuestionGroups = Field(default_factory=ConversationSuggestedQuestionGroups)
    operational_readiness: ConversationOperationalReadiness = Field(default_factory=ConversationOperationalReadiness)
    recommendation_feedback: dict[str, Any] = Field(default_factory=dict)
    dashboard_usage_summary: dict[str, Any] = Field(default_factory=dict)
    llm_used: bool = False
    llm_mode: str = "rules"
    llm_detail: str | None = None


class ConversationDashboardResponse(BaseModel):
    total: int
    insights: list[SelectedInsightDashboardItem] = Field(default_factory=list)
    dashboard_spec: ConversationDashboardSpec = Field(default_factory=ConversationDashboardSpec)


class ConversationChartVisualizationRequest(BaseModel):
    id: str = ""
    title: str = ""
    chart_type: str = "bar"
    x: str = ""
    y: str = ""
    metric: str = "count"
    group_by: str = ""
    aggregation: str = "count"
    filters: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    evidence_used: str = ""
    question_answered: str = ""
    audience: str = "ambos"


class ConversationChartDataRequest(BaseModel):
    visualization: ConversationChartVisualizationRequest
    limit: int = Field(default=12, ge=1, le=50)
    evidence_limit: int = Field(default=12, ge=1, le=1000)


class ConversationChartSeriesPoint(BaseModel):
    key: str
    label: str
    value: float
    count: int
    metric: str = "count"
    filter: dict[str, Any] = Field(default_factory=dict)


class ConversationChartEvidenceItem(BaseModel):
    evidence_id: str = ""
    incident_id: str = ""
    title: str = ""
    preview: str = ""
    source: str = ""
    group: str = ""
    priority: str = ""
    service: str = ""
    category: str = ""
    metric_value: float | None = None
    fields: dict[str, Any] = Field(default_factory=dict)


class ConversationChartValidation(BaseModel):
    status: str = "ok"
    quality_score: int = 0
    operation_ready: bool = False
    llm_used_available_data: bool = True
    chose_interpretable_variables: bool = True
    chart_is_buildable: bool = True
    uses_real_data: bool = True
    requires_data: bool = False
    uses_technical_variable: bool = False
    possibly_invented: bool = False
    evidence_returned: int = 0
    validation_summary: str = ""
    recommended_action: str = ""
    warnings: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    source: str = "duckdb"


class ConversationChartDataResponse(BaseModel):
    schema_version: str = "conversation-chart-data/v1"
    run_id: str
    visualization_id: str = ""
    title: str = ""
    chart_type: str = "bar"
    x: str = ""
    metric: str = "count"
    aggregation: str = "count"
    total_records: int = 0
    evidence_returned: int = 0
    evidence_truncated: bool = False
    series: list[ConversationChartSeriesPoint] = Field(default_factory=list)
    evidence_samples: list[ConversationChartEvidenceItem] = Field(default_factory=list)
    samples_by_key: dict[str, list[ConversationChartEvidenceItem]] = Field(default_factory=dict)
    semantic_dictionary: list[ConversationSemanticVariable] = Field(default_factory=list)
    validation: ConversationChartValidation = Field(default_factory=ConversationChartValidation)


class BiSyncResponse(BaseModel):
    status: str
    message: str
    tables: dict[str, int] = Field(default_factory=dict)


class MetabaseStatusResponse(BaseModel):
    enabled: bool
    metabase_url: str
    dashboard_id: int | None = None
    dashboard_url: str | None = None
    embed_url: str | None = None
    postgres_status: str
    detail: str | None = None
    tables: dict[str, int] = Field(default_factory=dict)
    embedding_configured: bool = False


class MetabaseEmbedTokenResponse(BaseModel):
    status: str
    message: str | None = None
    token: str | None = None
    instance_url: str | None = None
    embed_url: str | None = None
    dashboard_id: int | None = None
    expires_in_seconds: int | None = None


class MetabaseDashboardCard(BaseModel):
    id: int
    name: str
    url: str


class MetabaseDashboardCreateResponse(BaseModel):
    status: str
    message: str
    dashboard_id: int | None = None
    dashboard_url: str | None = None
    embed_url: str | None = None
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
    document_context_used: bool = False


class AgentTraceResponse(BaseModel):
    run_id: str
    trace_count: int
    traces: list[dict] = Field(default_factory=list)


class AgentResultsResponse(BaseModel):
    run_id: str
    recommendations: list[dict] = Field(default_factory=list)
    insights: list[dict] = Field(default_factory=list)
    has_traces: bool = False
