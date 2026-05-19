import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import get_settings
from app.db import AnalysisRun, User, get_db, run_to_detail, save_run
from app.schemas import (
    DatasetProfileResponse,
    HealthResponse,
    PipelineMetrics,
    RunCreateBody,
    RunDetail,
    RunSummary,
)
from app.services.dataset_store import get_dataset_meta, save_upload
from app.services.pipeline import run_pipeline

router = APIRouter()


def _metrics_from_row(row: AnalysisRun) -> PipelineMetrics:
    sil = float(row.silhouette) if row.silhouette else None
    dbi = float(row.davies_bouldin) if row.davies_bouldin else None
    n_clusters = None
    try:
        payload = json.loads(row.result_json)
        n_clusters = payload.get("metrics", {}).get("n_clusters")
        if n_clusters is not None:
            n_clusters = int(n_clusters)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if n_clusters is None:
        try:
            labels = json.loads(row.result_json).get("cluster_labels", [])
            n_clusters = len({int(x) for x in labels if int(x) >= 0})
        except (json.JSONDecodeError, TypeError, ValueError):
            n_clusters = None
    return PipelineMetrics(
        silhouette=sil,
        davies_bouldin=dbi,
        n_clusters=n_clusters,
    )


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
    return RunDetail(**run_to_detail(row))


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


@router.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return RunDetail(**run_to_detail(row))
