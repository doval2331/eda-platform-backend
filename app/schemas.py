from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["texto", "imagen", "multimodal", "it_ops", "tabular"]
ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]


class EvidenceMetadata(BaseModel):
    id: str
    preview: str
    source: Modality
    sector: str | None = None
    service_line: str | None = None
    support_channel: str | None = None
    segment: str | None = None
    category: str | None = None
    severity: str | None = None
    status: str | None = None
    assignment_group: str | None = None
    affected_service: str | None = None
    monthly_tickets: float | None = None
    critical_incidents: float | None = None
    avg_resolution_hours: float | None = None
    resolution_minutes: float | None = None
    sla_breach_rate: float | None = None
    sla_breached: bool | None = None
    operational_risk_score: float | None = None
    business_impact_score: float | None = None
    security_incidents: float | None = None
    downtime_hours: float | None = None
    customer_satisfaction: float | None = None


class PipelineMetrics(BaseModel):
    silhouette: float | None = None
    davies_bouldin: float | None = None
    n_clusters: int | None = None


class PipelineResult(BaseModel):
    X_2d: list[list[float]]
    cluster_labels: list[int]
    outliers_count: int
    metrics: PipelineMetrics
    metadata: list[EvidenceMetadata]


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    filename: str
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


class RunSummary(BaseModel):
    id: str
    created_at: datetime
    modality: Modality
    reduction_method: ReductionMethod
    seed: int
    n_samples: int
    outliers_count: int
    metrics: PipelineMetrics


class RunDetail(RunSummary):
    result: PipelineResult


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


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


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
