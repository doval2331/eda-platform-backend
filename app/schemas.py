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
    monthly_tickets: float | None = None
    sla_breach_rate: float | None = None
    operational_risk_score: float | None = None


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
