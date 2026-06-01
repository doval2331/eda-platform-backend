import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import get_settings
from app.db import AnalysisRun, User, get_db, run_to_detail, save_run
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDashboardResponse,
    DatasetProfileResponse,
    HealthResponse,
    InsightSelectionBody,
    BiSyncResponse,
    MetabaseDashboardCreateResponse,
    MetabaseStatusResponse,
    PipelineMetrics,
    RunCreateBody,
    RunDetail,
    RunSummary,
)
from app.services.dataset_store import get_dataset_meta, save_upload
from app.services.conversation import build_chat_response
from app.services.duckdb_store import (
    list_selected_insights,
    persist_run_detail,
    run_exists,
    save_selected_insight,
)
from app.services.bi_postgres_store import (
    get_bi_status,
    sync_bi_tables,
    try_sync_bi_tables,
)
from app.services.metabase_dashboard import (
    MetabaseDashboardError,
    create_conversation_dashboard,
)
from app.services.pipeline import run_pipeline

router = APIRouter()


def _bi_sync_response(run_id: str | None = None) -> BiSyncResponse:
    try:
        result = sync_bi_tables(run_id=run_id, force=True)
    except Exception as exc:
        return BiSyncResponse(
            status="error",
            message=f"No se pudo sincronizar PostgreSQL BI: {exc}",
            tables={},
        )
    return BiSyncResponse(
        status=result.status,
        message=result.message,
        tables=result.tables,
    )


def _metrics_from_payload(row: AnalysisRun) -> dict:
    metrics: dict = {}
    try:
        payload = json.loads(row.result_json)
        raw = payload.get("metrics") or {}
        if isinstance(raw, dict):
            metrics = dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if metrics.get("silhouette") is None and row.silhouette is not None:
        metrics["silhouette"] = float(row.silhouette)
    if metrics.get("davies_bouldin") is None and row.davies_bouldin is not None:
        metrics["davies_bouldin"] = float(row.davies_bouldin)
    if metrics.get("n_clusters") is None:
        try:
            labels = json.loads(row.result_json).get("cluster_labels", [])
            metrics["n_clusters"] = len({int(x) for x in labels if int(x) >= 0})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return metrics


def _metrics_from_row(row: AnalysisRun) -> PipelineMetrics:
    return PipelineMetrics.model_validate(_metrics_from_payload(row))


def _get_run_or_404(db: Session, run_id: str) -> AnalysisRun:
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ejecucion no encontrada")
    return row


def _materialize_run_in_duckdb(row: AnalysisRun) -> None:
    if not run_exists(row.id):
        persist_run_detail(run_to_detail(row))


@router.get("/health", response_model=HealthResponse)
def health(db: Annotated[Session, Depends(get_db)]):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", database=db_status)


@router.post("/api/datasets/upload", response_model=DatasetProfileResponse, status_code=201)
async def upload_dataset(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    del db  # reservado para futura persistencia en BD
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
    content = await file.read()
    try:
        meta = save_upload(
            user_id=user.id,
            filename=file.filename,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error al guardar el dataset") from exc
    return DatasetProfileResponse(**meta)


@router.get("/api/datasets/{dataset_id}", response_model=DatasetProfileResponse)
def get_dataset_profile(
    dataset_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        meta = get_dataset_meta(dataset_id, user_id=user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset no encontrado") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return DatasetProfileResponse(**meta)


@router.post("/api/runs", response_model=RunDetail, status_code=201)
def create_run(
    body: RunCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    seed = body.seed if body.seed is not None else settings.default_seed
    n_samples = body.n_samples or settings.default_n_samples

    if body.modality == "tabular" and not body.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Sube un CSV y proporciona dataset_id para modalidad tabular",
        )

    try:
        result = run_pipeline(
            modality=body.modality,
            reduction_method=body.reduction_method,
            seed=seed,
            n_samples=n_samples,
            dataset_path=settings.it_ops_dataset_path,
            dataset_id=body.dataset_id,
            user_id=user.id,
            id_column=body.id_column,
            exclude_columns=body.exclude_columns or None,
            numeric_columns=body.numeric_columns,
            categorical_columns=body.categorical_columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    payload = {
        "modality": body.modality,
        "reduction_method": body.reduction_method,
        "seed": seed,
        "n_samples": n_samples,
        "result": result.model_dump(),
    }
    row = save_run(db, payload=payload)
    detail = run_to_detail(row)
    persist_run_detail(detail)
    try_sync_bi_tables(row.id)
    return RunDetail(**detail)


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
    limit: int = 20,
):
    limit = min(max(1, limit), 100)
    rows = (
        db.query(AnalysisRun)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RunSummary(
            id=r.id,
            created_at=r.created_at,
            modality=r.modality,  # type: ignore[arg-type]
            reduction_method=r.reduction_method,  # type: ignore[arg-type]
            seed=r.seed,
            n_samples=r.n_samples,
            outliers_count=r.outliers_count,
            metrics=_metrics_from_row(r),
        )
        for r in rows
    ]


@router.post("/api/runs/{run_id}/chat", response_model=ChatResponse)
def chat_with_run(
    run_id: str,
    body: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    return build_chat_response(run_id, body.question)


@router.post("/api/runs/{run_id}/insights/select")
def select_run_insight(
    run_id: str,
    body: InsightSelectionBody,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    save_selected_insight(run_id, body.insight.model_dump(), user_id=_user.id)
    try_sync_bi_tables(run_id)
    return {"status": "ok"}


@router.get("/api/conversation-dashboard", response_model=ConversationDashboardResponse)
def get_conversation_dashboard(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    run_id: str | None = None,
):
    if run_id:
        row = _get_run_or_404(db, run_id)
        _materialize_run_in_duckdb(row)
    insights = list_selected_insights(run_id=run_id, user_id=user.id)
    return ConversationDashboardResponse(total=len(insights), insights=insights)


@router.get("/api/metabase/status", response_model=MetabaseStatusResponse)
def metabase_status(_user: Annotated[User, Depends(get_current_user)]):
    return MetabaseStatusResponse(**get_bi_status())


@router.post("/api/metabase/dashboard", response_model=MetabaseDashboardCreateResponse)
def create_metabase_dashboard(_user: Annotated[User, Depends(get_current_user)]):
    try:
        sync_bi_tables(force=True)
        result = create_conversation_dashboard()
    except MetabaseDashboardError as exc:
        return MetabaseDashboardCreateResponse(status="error", message=str(exc))
    except Exception as exc:
        return MetabaseDashboardCreateResponse(
            status="error",
            message=f"No se pudo crear el dashboard en Metabase: {exc}",
        )
    return MetabaseDashboardCreateResponse(**result)


@router.post("/api/bi-sync", response_model=BiSyncResponse)
def sync_all_bi_tables(_user: Annotated[User, Depends(get_current_user)]):
    return _bi_sync_response()


@router.post("/api/runs/{run_id}/bi-sync", response_model=BiSyncResponse)
def sync_run_bi_tables(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    return _bi_sync_response(run_id)


@router.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    _materialize_run_in_duckdb(row)
    return RunDetail(**run_to_detail(row))
