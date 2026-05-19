"""Núcleo compartido: reducción, clustering y métricas (notebook + API)."""

from __future__ import annotations

import warnings
from typing import Literal

import hdbscan
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from app.schemas import PipelineMetrics
from app.services.pipeline_config import load_pipeline_config

ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]


def scale_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def reduce_2d(
    X: np.ndarray,
    method: ReductionMethod,
    seed: int,
    *,
    config: dict | None = None,
) -> np.ndarray:
    cfg = config or load_pipeline_config()
    n = X.shape[0]
    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    if method == "t-SNE":
        max_n = int(cfg.get("tsne_max_samples", 3000))
        if n > max_n:
            raise ValueError(
                f"t-SNE con más de {max_n} muestras puede ser muy lento. "
                "Reduce n_samples o usa UMAP/PCA."
            )
        perplexity = min(30.0, max(5.0, (n - 1) / 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = TSNE(
                n_components=2,
                random_state=seed,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
            )
        return reducer.fit_transform(X)

    umap_cfg = cfg.get("umap", {})
    n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    min_dist = float(umap_cfg.get("min_dist", 0.1))
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist,
    )
    return reducer.fit_transform(X)


def cluster_hdbscan(X_2d: np.ndarray, *, config: dict | None = None) -> np.ndarray:
    cfg = config or load_pipeline_config()
    hdb = cfg.get("hdbscan", {})
    auto_mcs = max(5, min(15, X_2d.shape[0] // 12))
    min_cluster_size = hdb.get("min_cluster_size")
    min_samples = hdb.get("min_samples")
    if min_cluster_size is None:
        min_cluster_size = auto_mcs
    else:
        min_cluster_size = int(min_cluster_size)
    if min_samples is None:
        min_samples = max(3, min_cluster_size // 3)
    else:
        min_samples = int(min_samples)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=hdb.get("cluster_selection_method", "eom"),
    )
    return clusterer.fit_predict(X_2d)


def compute_metrics(X_2d: np.ndarray, labels: np.ndarray) -> PipelineMetrics:
    mask = labels >= 0
    if not mask.any():
        return PipelineMetrics(silhouette=None, davies_bouldin=None)

    unique = set(labels[mask].tolist())
    if len(unique) < 2 or mask.sum() < len(unique) + 1:
        return PipelineMetrics(silhouette=None, davies_bouldin=None)

    try:
        sil = float(silhouette_score(X_2d[mask], labels[mask]))
    except ValueError:
        sil = None

    try:
        db = float(davies_bouldin_score(X_2d[mask], labels[mask]))
    except ValueError:
        db = None

    return PipelineMetrics(silhouette=sil, davies_bouldin=db)
