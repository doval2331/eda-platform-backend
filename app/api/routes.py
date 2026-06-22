import json
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import get_settings
from app.db import AnalysisRun, User, get_db, run_to_detail, save_run
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
    ChatSuggestionsResponse,
    ConversationDashboardResponse,
    DatasetProfileResponse,
    HealthResponse,
    InsightSelectionBody,
    BiSyncResponse,
    MetabaseDashboardCreateResponse,
    MetabaseStatusResponse,
    PipelineMetrics,
    ProjectCreateBody,
    ProjectDetail,
    ProjectRunCreateBody,
    ProjectRunResponse,
    ProjectSourceType,
    ProjectSummary,
    ProjectUpdateBody,
    RunCreateBody,
    RunDeleteResponse,
    RunDetail,
    RunResetResponse,
    RunSummary,
)
from app.services.datasets.dataset_store import get_dataset_meta, save_upload
from app.services.conversation.conversation import build_chat_response, build_suggested_questions_for_run
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
from app.services.bi.metabase_dashboard import (
    MetabaseDashboardError,
    create_conversation_dashboard,
)
from app.services.pipeline.pipeline import run_pipeline
from app.services.projects.project_service import (
    add_project_source,
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


def _run_summary_from_row(row: AnalysisRun) -> RunSummary:
    return RunSummary(
        id=row.id,
        created_at=row.created_at,
        modality=row.modality,  # type: ignore[arg-type]
        reduction_method=row.reduction_method,  # type: ignore[arg-type]
        seed=row.seed,
        n_samples=row.n_samples,
        outliers_count=row.outliers_count,
        metrics=_metrics_from_row(row),
        project_id=row.project_id,
        project_name=row.project_name,
        source_type=row.source_type,
        source_id=row.source_id,
        source_name=row.source_name,
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


@router.post("/api/projects/{project_id}/sources", response_model=ProjectDetail)
async def upload_project_source(
    project_id: str,
    source_type: ProjectSourceType,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    source_name: str | None = Form(None),
    file: UploadFile = File(...),
):
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


@router.post("/api/projects/{project_id}/runs", response_model=ProjectRunResponse, status_code=201)
def create_project_runs(
    project_id: str,
    body: ProjectRunCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    seed = body.seed if body.seed is not None else settings.default_seed

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
    return build_chat_response(
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
    )


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
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    evidences = load_run_evidences(run_id)
    if evidences.empty:
        raise HTTPException(status_code=422, detail="No hay evidencias materializadas para la corrida")
    tracer = TraceCollector(run_id=run_id)
    recommendations, meta = run_strategy_agent(
        run_id=run_id,
        evidences=evidences,
        metrics=_metrics_from_payload(row),
        sample_size=body.sample_size,
        sample_criteria=body.sample_criteria,
        model_name=body.model_name,
        tracer=tracer,
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
    )


@router.post("/api/runs/{run_id}/agents/interpretation", response_model=AgentServiceResponse)
def run_interpretation_agent_for_run(
    run_id: str,
    body: AgentInterpretationRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = _get_run_or_404(db, run_id)
    _materialize_run_in_duckdb(row)
    evidences = load_run_evidences(run_id)
    if evidences.empty:
        raise HTTPException(status_code=422, detail="No hay evidencias materializadas para la corrida")
    tracer = TraceCollector(run_id=run_id)
    samples, insights, meta = run_interpretation_agent(
        run_id=run_id,
        evidences=evidences,
        sample_size=body.sample_size,
        sample_criteria=body.sample_criteria,
        random_state=body.random_state,
        model_name=body.model_name,
        tracer=tracer,
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
