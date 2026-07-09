import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.orm import Session, load_only

from app.api.deps import get_current_user
from app.config import get_settings
from app.db import AnalysisRun, SessionLocal, User, get_db, run_to_detail, save_run
from app.schemas import (
    AgentHumanDecisionRequest,
    AgentHumanDecisionResponse,
    AgentInterpretationRequest,
    AgentResultsResponse,
    AgentServiceResponse,
    AgentStrategyRequest,
    AgentTraceResponse,
    ChatRequest,
    ChatResponse,
    ChatAppendMessageBody,
    ChatHistoryResponse,
    ChatMessageRecord,
    ChatSuggestionsResponse,
    ConversationChartDataRequest,
    ConversationChartDataResponse,
    ConversationDashboardResponse,
    DatasetProfileResponse,
    HealthResponse,
    InsightSelectionBody,
    InsightBatchSelectionBody,
    InsightBatchSelectionResponse,
    BiSyncResponse,
    MetabaseDashboardCreateResponse,
    MetabaseEmbedTokenResponse,
    MetabaseStatusResponse,
    PipelineMetrics,
    ProjectCreateBody,
    ProjectDetail,
    ProjectRunCreateBody,
    ProjectRunResponse,
    ProjectSourceUploadJobResponse,
    ProjectSourceType,
    ProjectSummary,
    ProjectUpdateBody,
    RunCreateBody,
    RunDeleteResponse,
    RunDetail,
    RunResetResponse,
    RunSummary,
    SelectedInsightsResponse,
)
from app.services.datasets.dataset_store import get_dataset_meta, save_upload, uploads_dir
from app.services.datasets.dataset_profile import (
    build_dataset_explore_profile,
    build_dataset_full_profile,
    build_dataset_profile_html,
)
from app.services.conversation.conversation import build_chat_response, build_suggested_questions_for_run
from app.services.conversation.chart_data import (
    build_conversation_chart_data,
    build_conversation_chart_error_response,
)
from app.services.conversation.dashboard_spec import SEMANTIC_VARIABLES, build_dashboard_spec
from app.services.conversation.semantic_dictionary import (
    get_semantic_dictionary,
    load_configured_semantic_variables,
    reload_semantic_dictionary,
    save_configured_semantic_variables,
    semantic_dictionary_status,
    semantic_dictionary_path,
)
from app.services.conversation.chat_history import load_history, persist_exchange, persist_note
from app.services.runs.duckdb_store import (
    append_agent_decisions,
    list_agent_cluster_insights,
    list_agent_decisions,
    list_agent_recommendations,
    list_selected_insights,
    persist_run_detail,
    run_exists,
    save_agent_cluster_insights,
    save_agent_cluster_samples,
    save_agent_recommendations,
    save_selected_insight,
    save_selected_insights_bulk,
    load_run_evidences,
)
from app.services.agents.agent_service import run_interpretation_agent, run_strategy_agent
from app.services.runs.run_reset import delete_run, reset_all_runs
from app.services.agents.agent_traceability import TraceCollector
from app.services.bi.bi_postgres_store import (
    get_bi_status,
    sync_bi_tables,
    try_sync_bi_tables,
)
from app.services.bi.metabase_embed import MetabaseEmbedError, create_embed_token, embedding_is_configured
from app.services.bi.metabase_dashboard import (
    MetabaseDashboardError,
    create_conversation_dashboard,
    get_conversation_dashboard_links,
)
from app.services.pipeline.pipeline import run_pipeline
from app.services.pipeline.pipeline_config import tuning_overrides_from_body
from app.services.projects.project_service import (
    add_project_source,
    add_project_source_from_path,
    create_project,
    delete_project_source,
    get_project_detail,
    get_project_or_404,
    list_csv_sources,
    list_projects,
    merge_project_tabular_sources,
    primary_incidents_source,
    source_display_name,
    update_project,
)
from app.services.projects.project_validation import validate_project_before_run
from app.services.projects.source_relationship import (
    build_project_document_context,
    validate_project_text_sources,
)

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_CHUNK_SIZE = 1024 * 1024
_upload_jobs: dict[str, dict] = {}
_upload_jobs_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_upload_job(job_id: str, **changes) -> dict:
    with _upload_jobs_lock:
        job = _upload_jobs.setdefault(job_id, {"job_id": job_id})
        job.update(changes)
        job["updated_at"] = _utc_now_iso()
        return dict(job)


def _get_upload_job(job_id: str) -> dict | None:
    with _upload_jobs_lock:
        job = _upload_jobs.get(job_id)
        return dict(job) if job else None


def _public_upload_job(job: dict) -> ProjectSourceUploadJobResponse:
    return ProjectSourceUploadJobResponse(
        job_id=str(job["job_id"]),
        project_id=str(job["project_id"]),
        status=job.get("status", "queued"),
        message=str(job.get("message") or ""),
        filename=str(job.get("filename") or ""),
        source_type=job.get("source_type"),
        source_name=job.get("source_name"),
        uploaded_bytes=job.get("uploaded_bytes"),
        error=job.get("error"),
        project=job.get("project"),
    )


async def _stream_upload_to_temp(file: UploadFile, filename: str, job_id: str) -> tuple[Path, int]:
    incoming_dir = uploads_dir() / "_incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    temp_path = incoming_dir / f"{job_id}{Path(filename).suffix}"
    max_bytes = get_settings().max_upload_bytes
    uploaded = 0

    try:
        with temp_path.open("wb") as dest:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                uploaded += len(chunk)
                if uploaded > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"El archivo supera el límite de {max_bytes // (1024 * 1024)} MB",
                    )
                dest.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return temp_path, uploaded


def _process_project_source_upload_job(job_id: str) -> None:
    job = _get_upload_job(job_id)
    if not job:
        return

    _set_upload_job(
        job_id,
        status="processing",
        message="Procesando archivo y perfilando columnas.",
    )
    temp_path = Path(str(job["temp_path"]))
    db = SessionLocal()
    try:
        detail = add_project_source_from_path(
            db,
            project_id=str(job["project_id"]),
            user_id=str(job["user_id"]),
            source_type=str(job["source_type"]),
            source_name=job.get("source_name"),
            filename=str(job["filename"]),
            path=temp_path,
        )
        if str(job["source_type"]) in {"dictionary", "notes", "other"}:
            validate_project_text_sources(
                db,
                project_id=str(job["project_id"]),
                user_id=str(job["user_id"]),
            )
            detail = get_project_detail(
                db,
                project_id=str(job["project_id"]),
                user_id=str(job["user_id"]),
            )
        _set_upload_job(
            job_id,
            status="completed",
            message="Fuente procesada y agregada al escenario.",
            project=detail,
            error=None,
        )
    except Exception as exc:
        _set_upload_job(
            job_id,
            status="failed",
            message="No se pudo procesar la fuente.",
            error=str(exc),
        )
    finally:
        db.close()
        temp_path.unlink(missing_ok=True)


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


def _metrics_from_summary_row(row: AnalysisRun) -> PipelineMetrics:
    metrics: dict = {}
    if row.silhouette is not None:
        metrics["silhouette"] = float(row.silhouette)
    if row.davies_bouldin is not None:
        metrics["davies_bouldin"] = float(row.davies_bouldin)
    if row.n_clusters is not None:
        metrics["n_clusters"] = row.n_clusters
    if row.noise_pct is not None:
        metrics["noise_pct"] = row.noise_pct
    return PipelineMetrics.model_validate(metrics)


def _get_run_or_404(db: Session, run_id: str) -> AnalysisRun:
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ejecucion no encontrada")
    return row


def _materialize_run_in_duckdb(row: AnalysisRun) -> None:
    if not run_exists(row.id):
        persist_run_detail(run_to_detail(row))


def _document_context_for_run(
    db: Session,
    row: AnalysisRun,
    user_id: str,
) -> tuple[str, bool]:
    if not row.project_id:
        return "", False
    return build_project_document_context(
        db,
        project_id=row.project_id,
        user_id=user_id,
    )


def _run_summary_from_row(row: AnalysisRun) -> RunSummary:
    return RunSummary(
        id=row.id,
        created_at=row.created_at,
        modality=row.modality,  # type: ignore[arg-type]
        reduction_method=row.reduction_method,  # type: ignore[arg-type]
        seed=row.seed,
        n_samples=row.n_samples,
        outliers_count=row.outliers_count,
        metrics=_metrics_from_summary_row(row),
        project_id=row.project_id,
        project_name=row.project_name,
        source_type=row.source_type,
        source_id=row.source_id,
        source_name=row.source_name,
        dataset_id=row.dataset_id,
    )


def _execute_and_persist_run(
    db: Session,
    *,
    user: User,
    modality: str,
    reduction_method: str,
    seed: int,
    n_samples: int | None,
    dataset_id: str | None = None,
    id_column: str | None = None,
    exclude_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    source_name: str | None = None,
    pipeline_overrides: dict | None = None,
) -> RunDetail:
    settings = get_settings()
    result = run_pipeline(
        modality=modality,
        reduction_method=reduction_method,
        seed=seed,
        n_samples=n_samples,
        dataset_path=settings.it_ops_dataset_path,
        dataset_id=dataset_id,
        user_id=user.id,
        id_column=id_column,
        exclude_columns=exclude_columns or None,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        pipeline_overrides=pipeline_overrides,
    )
    analyzed_rows = (
        len(result.metadata)
        if result.metadata
        else (n_samples or settings.default_n_samples)
    )
    payload = {
        "modality": modality,
        "reduction_method": reduction_method,
        "seed": seed,
        "n_samples": analyzed_rows,
        "result": result.model_dump(),
        "project_id": project_id,
        "project_name": project_name,
        "source_type": source_type,
        "source_id": source_id,
        "source_name": source_name,
        "dataset_id": dataset_id,
    }
    row = save_run(db, payload=payload)
    detail = run_to_detail(row)
    persist_run_detail(detail)
    try_sync_bi_tables(row.id)
    return RunDetail(**detail)


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


@router.get("/api/datasets/{dataset_id}/explore-profile")
def get_dataset_explore_profile(
    dataset_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        return build_dataset_explore_profile(dataset_id, user_id=user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset no encontrado") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/api/datasets/{dataset_id}/full-profile")
def get_dataset_full_profile(
    dataset_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        return build_dataset_full_profile(dataset_id, user_id=user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset no encontrado") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/api/datasets/{dataset_id}/profile-report", response_class=HTMLResponse)
def get_dataset_profile_report(
    dataset_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        html = build_dataset_profile_html(dataset_id, user_id=user.id)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="ydata-profiling no está instalado en el servidor.",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset no encontrado") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return HTMLResponse(content=html)


@router.post("/api/projects", response_model=ProjectDetail, status_code=201)
def create_project_route(
    body: ProjectCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    detail = create_project(
        db,
        user_id=user.id,
        name=body.name,
        description=body.description,
        strategy=body.strategy,
    )
    return ProjectDetail(**detail, sources=[])


@router.get("/api/projects", response_model=list[ProjectSummary])
def list_projects_route(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    limit: int = 50,
):
    return [ProjectSummary(**item) for item in list_projects(db, user_id=user.id, limit=limit)]


@router.get("/api/projects/{project_id}", response_model=ProjectDetail)
def get_project_route(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        detail = get_project_detail(db, project_id=project_id, user_id=user.id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return ProjectDetail(**detail)


@router.patch("/api/projects/{project_id}", response_model=ProjectDetail)
def update_project_route(
    project_id: str,
    body: ProjectUpdateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        detail = update_project(
            db,
            project_id=project_id,
            user_id=user.id,
            name=body.name,
            description=body.description,
            strategy=body.strategy,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return ProjectDetail(**detail)


@router.post(
    "/api/projects/{project_id}/sources",
    response_model=ProjectSourceUploadJobResponse,
    status_code=202,
)
async def upload_project_source(
    project_id: str,
    source_type: ProjectSourceType,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    source_name: str | None = Form(None),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
    try:
        get_project_or_404(db, project_id=project_id, user_id=user.id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    temp_path, uploaded_bytes = await _stream_upload_to_temp(file, file.filename, job_id)
    job = _set_upload_job(
        job_id,
        project_id=project_id,
        user_id=user.id,
        source_type=source_type,
        source_name=source_name,
        filename=file.filename,
        temp_path=str(temp_path),
        uploaded_bytes=uploaded_bytes,
        status="queued",
        message="Archivo recibido. Procesamiento en cola.",
        created_at=_utc_now_iso(),
        project=None,
        error=None,
    )
    background_tasks.add_task(_process_project_source_upload_job, job_id)
    return _public_upload_job(job)


@router.get(
    "/api/projects/{project_id}/sources/jobs/{job_id}",
    response_model=ProjectSourceUploadJobResponse,
)
def get_project_source_upload_job(
    project_id: str,
    job_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    job = _get_upload_job(job_id)
    if not job or job.get("project_id") != project_id or job.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="Job de carga no encontrado")
    return _public_upload_job(job)


@router.post("/api/projects/{project_id}/sources/sync", response_model=ProjectDetail)
async def upload_project_source_sync(
    project_id: str,
    source_type: ProjectSourceType,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    source_name: str | None = Form(None),
    file: UploadFile = File(...),
):
    """Compatibilidad para clientes antiguos; el frontend usa la carga asincrona."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
    content = await file.read()
    try:
        detail = add_project_source(
            db,
            project_id=project_id,
            user_id=user.id,
            source_type=source_type,
            source_name=source_name,
            filename=file.filename,
            content=content,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al guardar la fuente: {exc}",
        ) from exc
    return ProjectDetail(**detail)


@router.delete("/api/projects/{project_id}/sources/{source_id}", response_model=ProjectDetail)
def remove_project_source(
    project_id: str,
    source_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        detail = delete_project_source(
            db,
            project_id=project_id,
            source_id=source_id,
            user_id=user.id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return ProjectDetail(**detail)


@router.post("/api/projects/{project_id}/validate-sources", response_model=ProjectDetail)
def validate_project_sources(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        get_project_or_404(db, project_id=project_id, user_id=user.id)
        validate_project_text_sources(db, project_id=project_id, user_id=user.id)
        return ProjectDetail(
            **get_project_detail(db, project_id=project_id, user_id=user.id),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/api/projects/{project_id}/runs", response_model=ProjectRunResponse, status_code=201)
def create_project_runs(
    project_id: str,
    body: ProjectRunCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    seed = body.seed if body.seed is not None else settings.default_seed
    pipeline_overrides = tuning_overrides_from_body(body) or None

    try:
        project = get_project_or_404(db, project_id=project_id, user_id=user.id)
        csv_sources = list_csv_sources(db, project_id=project_id, user_id=user.id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not csv_sources:
        raise HTTPException(
            status_code=400,
            detail="El proyecto necesita al menos una fuente tabular para analizar",
        )

    try:
        validate_project_before_run(db, project=project, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    targets = csv_sources
    merged_dataset_id: str | None = None
    if project.strategy == "merged":
        if len(csv_sources) < 2:
            raise HTTPException(
                status_code=400,
                detail="El modo unificado multifuente requiere al menos dos fuentes tabulares.",
            )
        try:
            merged_meta = merge_project_tabular_sources(
                db,
                project_id=project.id,
                user_id=user.id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        merged_dataset_id = merged_meta["dataset_id"]
        targets = []
    elif project.strategy == "unified":
        primary = primary_incidents_source(csv_sources)
        if primary is None:
            raise HTTPException(status_code=400, detail="No hay fuente CSV válida")
        targets = [primary]

    runs: list[RunDetail] = []
    try:
        if merged_dataset_id:
            run_detail = _execute_and_persist_run(
                db,
                user=user,
                modality="tabular",
                reduction_method=body.reduction_method,
                seed=seed,
                n_samples=body.n_samples,
                dataset_id=merged_dataset_id,
                id_column="_registro_id",
                exclude_columns=body.exclude_columns,
                numeric_columns=body.numeric_columns,
                categorical_columns=body.categorical_columns,
                project_id=project.id,
                project_name=project.name,
                source_type="merged",
                source_id=None,
                source_name="Todas las fuentes (unificado)",
                pipeline_overrides=pipeline_overrides,
            )
            runs.append(run_detail)
        for source in targets:
            run_detail = _execute_and_persist_run(
                db,
                user=user,
                modality="tabular",
                reduction_method=body.reduction_method,
                seed=seed,
                n_samples=body.n_samples,
                dataset_id=source.dataset_id,
                id_column=body.id_column,
                exclude_columns=body.exclude_columns,
                numeric_columns=body.numeric_columns,
                categorical_columns=body.categorical_columns,
                project_id=project.id,
                project_name=project.name,
                source_type=source.source_type,
                source_id=source.id,
                source_name=source_display_name(source),
                pipeline_overrides=pipeline_overrides,
            )
            runs.append(run_detail)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    primary_run = runs[0]
    return ProjectRunResponse(
        project_id=project.id,
        project_name=project.name,
        strategy=project.strategy,  # type: ignore[arg-type]
        primary_run_id=primary_run.id,
        runs=runs,
    )


@router.post("/api/runs", response_model=RunDetail, status_code=201)
def create_run(
    body: RunCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    seed = body.seed if body.seed is not None else settings.default_seed
    pipeline_overrides = tuning_overrides_from_body(body) or None

    if body.modality == "tabular" and not body.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Sube un CSV y proporciona dataset_id para modalidad tabular",
        )

    try:
        return _execute_and_persist_run(
            db,
            user=user,
            modality=body.modality,
            reduction_method=body.reduction_method,
            seed=seed,
            n_samples=body.n_samples,
            dataset_id=body.dataset_id,
            id_column=body.id_column,
            exclude_columns=body.exclude_columns,
            numeric_columns=body.numeric_columns,
            categorical_columns=body.categorical_columns,
            project_name=body.project_name,
            source_type=body.source_type or ("incidents" if body.modality == "tabular" else None),
            pipeline_overrides=pipeline_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/api/runs", response_model=RunResetResponse)
def clear_all_runs(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    result = reset_all_runs(db)
    deleted = int(result["deleted_runs"])
    return RunResetResponse(
        status="ok",
        deleted_runs=deleted,
        duckdb_tables_cleared=result.get("duckdb_tables_cleared") or {},
        bi_tables_cleared=result.get("bi_tables_cleared"),
        message=(
            f"Se eliminaron {deleted} ejecuciones del historial y los datos analiticos asociados."
        ),
    )


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
    limit: int = 20,
):
    limit = min(max(1, limit), 100)
    rows = (
        db.query(AnalysisRun)
        .options(
            load_only(
                AnalysisRun.id,
                AnalysisRun.created_at,
                AnalysisRun.modality,
                AnalysisRun.reduction_method,
                AnalysisRun.seed,
                AnalysisRun.n_samples,
                AnalysisRun.outliers_count,
                AnalysisRun.silhouette,
                AnalysisRun.davies_bouldin,
                AnalysisRun.n_clusters,
                AnalysisRun.noise_pct,
                AnalysisRun.project_id,
                AnalysisRun.project_name,
                AnalysisRun.source_type,
                AnalysisRun.source_id,
                AnalysisRun.source_name,
                AnalysisRun.dataset_id,
            )
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_run_summary_from_row(r) for r in rows]


@router.post("/api/runs/{run_id}/chat", response_model=ChatResponse)
def chat_with_run(
    run_id: str,
    body: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    project_sources: list[dict] = []
    project_strategy: str | None = None
    if row.project_id:
        try:
            project_detail = get_project_detail(db, project_id=row.project_id, user_id=user.id)
            project_sources = project_detail.get("sources", [])
            project_strategy = project_detail.get("strategy")
        except (LookupError, PermissionError):
            project_sources = []
    response = build_chat_response(
        run_id,
        body.question,
        run_context={
            "project_name": row.project_name,
            "project_strategy": project_strategy,
            "source_type": row.source_type,
            "source_name": row.source_name,
            "source_id": row.source_id,
            "sources": project_sources,
        },
        history=[item.model_dump() for item in body.history[-8:]],
        document_context=_document_context_for_run(db, row, user.id)[0],
    )
    persist_exchange(
        run_id=run_id,
        user_id=user.id,
        question=body.display_question or body.question,
        response=response,
    )
    return response


@router.get("/api/runs/{run_id}/chat/history", response_model=ChatHistoryResponse)
def get_chat_history_for_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    messages = load_history(run_id=run_id, user_id=user.id)
    return ChatHistoryResponse(messages=messages)


@router.post("/api/runs/{run_id}/chat/messages", response_model=ChatMessageRecord)
def append_chat_message_route(
    run_id: str,
    body: ChatAppendMessageBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    message = persist_note(
        run_id=run_id,
        user_id=user.id,
        text=body.text,
        metadata=body.metadata,
    )
    return message


@router.get("/api/runs/{run_id}/chat/suggestions", response_model=ChatSuggestionsResponse)
def get_chat_suggestions_for_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    project_sources: list[dict] = []
    project_strategy: str | None = None
    if row.project_id:
        try:
            project_detail = get_project_detail(db, project_id=row.project_id, user_id=user.id)
            project_sources = project_detail.get("sources", [])
            project_strategy = project_detail.get("strategy")
        except (LookupError, PermissionError):
            project_sources = []
    return ChatSuggestionsResponse(
        suggested_questions=build_suggested_questions_for_run(
            run_id,
            run_context={
                "project_name": row.project_name,
                "project_strategy": project_strategy,
                "source_type": row.source_type,
                "source_name": row.source_name,
                "source_id": row.source_id,
                "sources": project_sources,
            },
        )
    )


@router.post("/api/runs/{run_id}/agents/strategy", response_model=AgentServiceResponse)
def run_strategy_agent_for_run(
    run_id: str,
    body: AgentStrategyRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    evidences = load_run_evidences(run_id)
    if evidences.empty:
        raise HTTPException(status_code=422, detail="No hay evidencias materializadas para la corrida")
    document_context, document_used = _document_context_for_run(db, row, user.id)
    tracer = TraceCollector(run_id=run_id)
    recommendations, meta = run_strategy_agent(
        run_id=run_id,
        evidences=evidences,
        metrics=_metrics_from_payload(row),
        sample_size=body.sample_size,
        sample_criteria=body.sample_criteria,
        model_name=body.model_name,
        tracer=tracer,
        document_context=document_context or None,
    )
    traces = tracer.to_frame()
    save_agent_recommendations(run_id, recommendations)
    append_agent_decisions(traces)
    return AgentServiceResponse(
        status="ok",
        run_id=run_id,
        trace_ids=traces["trace_id"].tolist() if not traces.empty else [],
        items=recommendations.where(pd.notnull(recommendations), None).to_dict(orient="records"),
        llm_used=meta.llm_used,
        llm_mode=meta.llm_mode,
        llm_detail=meta.llm_detail,
        model_name=meta.model_name,
        document_context_used=document_used,
    )


@router.post("/api/runs/{run_id}/agents/interpretation", response_model=AgentServiceResponse)
def run_interpretation_agent_for_run(
    run_id: str,
    body: AgentInterpretationRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    evidences = load_run_evidences(run_id)
    if evidences.empty:
        raise HTTPException(status_code=422, detail="No hay evidencias materializadas para la corrida")
    document_context, document_used = _document_context_for_run(db, row, user.id)
    tracer = TraceCollector(run_id=run_id)
    samples, insights, meta = run_interpretation_agent(
        run_id=run_id,
        evidences=evidences,
        sample_size=body.sample_size,
        sample_criteria=body.sample_criteria,
        random_state=body.random_state,
        model_name=body.model_name,
        tracer=tracer,
        document_context=document_context or None,
    )
    traces = tracer.to_frame()
    save_agent_cluster_samples(run_id, samples)
    save_agent_cluster_insights(run_id, insights)
    append_agent_decisions(traces)
    return AgentServiceResponse(
        status="ok",
        run_id=run_id,
        trace_ids=traces["trace_id"].tolist() if not traces.empty else [],
        items=insights.where(pd.notnull(insights), None).to_dict(orient="records"),
        llm_used=meta.llm_used,
        llm_mode=meta.llm_mode,
        llm_detail=meta.llm_detail,
        model_name=meta.model_name,
        document_context_used=document_used,
    )


@router.post("/api/runs/{run_id}/agents/human-decision", response_model=AgentHumanDecisionResponse)
def record_human_agent_decision_for_run(
    run_id: str,
    body: AgentHumanDecisionRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    _get_run_or_404(db, run_id)
    selected_by_strategy = body.parameters.get("selected_variables_by_strategy")
    variables_used: list[str] = []
    if isinstance(selected_by_strategy, dict):
        for selected_variables in selected_by_strategy.values():
            if isinstance(selected_variables, list):
                variables_used.extend(str(variable) for variable in selected_variables)
            elif isinstance(selected_variables, str):
                variables_used.append(selected_variables)
    variables_used = sorted({variable for variable in variables_used if variable.strip()})

    tracer = TraceCollector(run_id=run_id)
    trace_id = tracer.record(
        agent_name="human_in_the_loop",
        decision_type=body.decision_type,
        prompt="Validacion humana de la estrategia sugerida por agentes.",
        response=body.summary,
        model_name=body.model_name,
        variables_used=variables_used,
        input_artifacts=["agent_recommendations", f"run:{run_id}"],
        parameters={
            "status": body.status,
            "approved_strategy_ids": body.approved_strategy_ids,
            **body.parameters,
        },
    )
    append_agent_decisions(tracer.to_frame())
    return AgentHumanDecisionResponse(
        status="ok",
        run_id=run_id,
        trace_id=trace_id,
        message="Decision humana registrada en trazabilidad.",
    )


@router.get("/api/runs/{run_id}/agents/results", response_model=AgentResultsResponse)
def get_agent_results_for_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    _get_run_or_404(db, run_id)
    recommendations = list_agent_recommendations(run_id)
    insights = list_agent_cluster_insights(run_id)
    traces = list_agent_decisions(run_id)
    return AgentResultsResponse(
        run_id=run_id,
        recommendations=recommendations,
        insights=insights,
        has_traces=bool(traces),
    )


@router.get("/api/runs/{run_id}/agents/traces", response_model=AgentTraceResponse)
def get_agent_traces_for_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
):
    _get_run_or_404(db, run_id)
    traces = list_agent_decisions(run_id, limit=limit)
    if not traces:
        raise HTTPException(status_code=404, detail="No hay trazabilidad de agentes para esta corrida")
    return AgentTraceResponse(run_id=run_id, trace_count=len(traces), traces=traces)


@router.get("/api/projects/{project_id}/agents/traces", response_model=AgentTraceResponse)
def get_agent_traces_for_project(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
):
    get_project_or_404(db, project_id=project_id, user_id=user.id)
    run_ids = [
        row.id
        for row in (
            db.query(AnalysisRun.id)
            .filter(AnalysisRun.project_id == project_id)
            .order_by(AnalysisRun.created_at.asc())
            .all()
        )
    ]
    traces: list[dict] = []
    for run_id in run_ids:
        for trace in list_agent_decisions(run_id, limit=limit):
            traces.append({**trace, "source_run_id": run_id})
    if not traces:
        raise HTTPException(status_code=404, detail="No hay trazabilidad de agentes para este proyecto")
    if limit is not None:
        traces = sorted(
            traces,
            key=lambda trace: str(trace.get("created_at") or ""),
            reverse=True,
        )[:limit]
    return AgentTraceResponse(run_id=project_id, trace_count=len(traces), traces=traces)


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
    return {"status": "ok"}


@router.post(
    "/api/runs/{run_id}/insights/select/batch",
    response_model=InsightBatchSelectionResponse,
)
def select_run_insights_batch(
    run_id: str,
    body: InsightBatchSelectionBody,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    saved = save_selected_insights_bulk(
        run_id,
        [insight.model_dump() for insight in body.insights],
        user_id=_user.id,
    )
    return InsightBatchSelectionResponse(saved=saved)


@router.get("/api/runs/{run_id}/insights/selected", response_model=SelectedInsightsResponse)
def get_run_selected_insights(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    insights = list_selected_insights(run_id=run_id, user_id=user.id)
    return SelectedInsightsResponse(total=len(insights), insights=insights)


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
    dashboard_spec = build_dashboard_spec(
        db=db,
        user_id=user.id,
        run_id=run_id,
        insights=insights,
    )
    return ConversationDashboardResponse(
        total=len(insights),
        insights=insights,
        dashboard_spec=dashboard_spec,
    )


@router.get("/api/conversation/semantic-dictionary")
def get_conversation_semantic_dictionary(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    refresh: bool = False,
    run_id: str | None = None,
    project_id: str | None = None,
):
    dictionary_project_id = project_id
    if run_id:
        row = _get_run_or_404(db, run_id)
        dictionary_project_id = row.project_id or dictionary_project_id
    if refresh:
        reload_semantic_dictionary()
    dictionary = get_semantic_dictionary(SEMANTIC_VARIABLES, project_id=dictionary_project_id)
    status = semantic_dictionary_status(SEMANTIC_VARIABLES, project_id=dictionary_project_id)
    configured_variables = load_configured_semantic_variables(project_id=dictionary_project_id)
    seen: set[tuple[str, str, str]] = set()
    variables: list[dict[str, Any]] = []
    for lookup_key, entry in sorted(dictionary.items()):
        identity = (
            str(entry.get("label") or lookup_key),
            str(entry.get("role") or ""),
            str(entry.get("description") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        variables.append(
            {
                "lookup_key": lookup_key,
                "label": entry.get("label") or lookup_key,
                "role": entry.get("role") or "unknown",
                "semantic_type": entry.get("semantic_type") or "",
                "can_chart": bool(entry.get("can_chart", True)),
                "avoid_as_metric": bool(entry.get("avoid_as_metric", False)),
                "avoid_as_dimension": bool(entry.get("avoid_as_dimension", False)),
                "description": entry.get("description") or "",
                "recommended_use": entry.get("recommended_use") or "",
                "aliases": entry.get("aliases") or [],
                "source": entry.get("source") or "base",
                "confidence": entry.get("confidence") or "media",
                "active": entry.get("active", True),
                "enabled_profiles": entry.get("enabled_profiles") or [],
                "domain": entry.get("domain") or "",
                "owner": entry.get("owner") or "",
                "version": entry.get("version") or "",
                "max_cardinality": entry.get("max_cardinality"),
                "max_null_ratio": entry.get("max_null_ratio"),
            }
        )
    return {
        "source": str(semantic_dictionary_path(dictionary_project_id)),
        "exists": status["exists"],
        "scope": status["scope"],
        "project_id": status.get("project_id") or "",
        "env_var": status["env_var"],
        "configurable": status["configurable"],
        "writable": status["writable"],
        "base_total": status["base_total"],
        "configured_total": status["configured_total"],
        "active_configured_total": status.get("active_configured_total", 0),
        "inactive_configured_total": status.get("inactive_configured_total", 0),
        "governed": status["governed"],
        "total": len(variables),
        "variables": variables,
        "configured_variables": configured_variables,
    }


@router.put("/api/conversation/semantic-dictionary")
def update_conversation_semantic_dictionary(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    payload: dict[str, Any] = Body(...),
    run_id: str | None = None,
    project_id: str | None = None,
):
    dictionary_project_id = project_id
    if run_id:
        row = _get_run_or_404(db, run_id)
        dictionary_project_id = row.project_id or dictionary_project_id
    entries = payload.get("variables") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Se esperaba un arreglo 'variables'.")
    result = save_configured_semantic_variables(entries, project_id=dictionary_project_id)
    return {
        "source": result["path"],
        "scope": result["scope"],
        "project_id": result.get("project_id") or "",
        "total": result["total"],
        "active_total": result.get("active_total", 0),
        "inactive_total": result.get("inactive_total", 0),
        "variables": result["variables"],
    }


@router.post(
    "/api/runs/{run_id}/conversation-chart-data",
    response_model=ConversationChartDataResponse,
)
def get_run_conversation_chart_data(
    run_id: str,
    body: ConversationChartDataRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    visualization = body.visualization.model_dump()
    try:
        _materialize_run_in_duckdb(row)
        return build_conversation_chart_data(
            run_id=run_id,
            visualization=visualization,
            limit=body.limit,
            evidence_limit=body.evidence_limit,
            project_id=row.project_id,
        )
    except Exception as exc:
        logger.exception("Error calculating conversation chart data for run_id=%s", run_id)
        return build_conversation_chart_error_response(
            run_id=run_id,
            visualization=visualization,
            warning=(
                "No se pudo calcular el grafico real al consultar o agregar las evidencias. "
                "Actualiza la ejecucion o prueba con otra vista sugerida."
            ),
        )


@router.get("/api/metabase/status", response_model=MetabaseStatusResponse)
def metabase_status(_user: Annotated[User, Depends(get_current_user)]):
    status = get_bi_status()
    settings = get_settings()
    if settings.metabase_username and settings.metabase_password:
        try:
            status.update(get_conversation_dashboard_links())
        except MetabaseDashboardError:
            pass
    return MetabaseStatusResponse(**status)


@router.get("/api/metabase/embed-token", response_model=MetabaseEmbedTokenResponse)
def metabase_embed_token(
    _user: Annotated[User, Depends(get_current_user)],
    run_id: str | None = None,
):
    try:
        return MetabaseEmbedTokenResponse(**create_embed_token(run_id=run_id))
    except MetabaseEmbedError as exc:
        return MetabaseEmbedTokenResponse(status="error", message=str(exc))
    except Exception as exc:
        return MetabaseEmbedTokenResponse(
            status="error",
            message=f"No se pudo generar el token de incrustación: {exc}",
        )


@router.post("/api/metabase/dashboard", response_model=MetabaseDashboardCreateResponse)
def create_metabase_dashboard(_user: Annotated[User, Depends(get_current_user)]):
    settings = get_settings()
    if not settings.metabase_username or not settings.metabase_password:
        return MetabaseDashboardCreateResponse(
            status="error",
            message=(
                "Configura METABASE_USERNAME y METABASE_PASSWORD en el backend "
                "antes de crear el dashboard en Metabase."
            ),
        )

    try:
        status = get_bi_status()
        if not status.get("tables", {}).get("bi_evidences"):
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


@router.delete("/api/runs/{run_id}", response_model=RunDeleteResponse)
def delete_run_route(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    try:
        result = delete_run(db, run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunDeleteResponse(
        status="ok",
        run_id=run_id,
        duckdb_tables_cleared=result.get("duckdb_tables_cleared") or {},
        bi_tables_cleared=result.get("bi_tables_cleared"),
        message="Se eliminó la ejecución y sus datos analíticos asociados.",
    )


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

@router.get("/api/runs/{run_id}/cluster-profiles")
def get_cluster_profiles(
    run_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Devuelve el perfil operativo de cada cluster para un run específico.
    Incluye el modo de visualización recomendado según el número de clusters.
    """
    import numpy as np
    from app.services.pipeline.cluster_profiler import (
        cluster_profiler,
        calcular_stats_globales,
        modo_visualizacion,
    )

    # Obtener el run
    # Por esto:
    run = _get_run_or_404(db, run_id)
    import json
    result = run.result_json
    if isinstance(result, str):
        result = json.loads(result)
    if not result:
        raise HTTPException(status_code=404, detail="El run no tiene resultados")
    # Obtener labels
    labels = np.array(result.get("cluster_labels", []))
    if len(labels) == 0:
        raise HTTPException(status_code=404, detail="Sin etiquetas de cluster")

    # Reconstruir DataFrame desde metadata
    metadata = result.get("metadata", [])
    if not metadata:
        raise HTTPException(status_code=404, detail="Sin metadata disponible")

    df_meta = pd.DataFrame(metadata)

    # Calcular perfiles
    stats_globales = calcular_stats_globales(df_meta)
    perfiles       = cluster_profiler(df_meta, labels, stats_globales)

    # Número de clusters sin ruido
    n_clusters = len([p for p in perfiles if not p["es_ruido"]])
    modo       = modo_visualizacion(n_clusters)

    return {
        "run_id":        run_id,
        "n_clusters":    n_clusters,
        "modo_viz":      modo,
        "stats_globales": stats_globales,
        "perfiles":      perfiles,
    }
@router.get("/api/metabase/embed-token")
def get_metabase_embed_token(
    run_id: str | None = Query(default=None),
    _user: Annotated[User, Depends(get_current_user)] = None,
):
    """
    Genera un token JWT firmado para incrustar el dashboard de Metabase
    de forma segura en el frontend, sin exponer el secret de embedding.
    """
    try:
        return create_embed_token(run_id=run_id)
    except MetabaseEmbedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
