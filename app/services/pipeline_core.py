"""Núcleo compartido: reducción, clustering y métricas (notebook + API)."""

from __future__ import annotations

import copy
import warnings
from typing import Literal

import hdbscan
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

    n = X_2d.shape[0]
    min_cluster_size = hdb.get("min_cluster_size")
    min_samples = hdb.get("min_samples")

    if min_cluster_size is None:
        min_cluster_size = _mcs_adaptativo(n)
    else:
        min_cluster_size = int(min_cluster_size)

    if min_samples is None:
        min_samples = max(5, min_cluster_size // 8)
    else:
        min_samples = int(min_samples)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=hdb.get("cluster_selection_method", "eom"),
    )
    return clusterer.fit_predict(X_2d)

def _resolve_hdbscan_min_samples(cfg: dict, n_points: int) -> int:
    hdb = cfg.get("hdbscan", {})
    min_cluster_size = hdb.get("min_cluster_size")
    if min_cluster_size is None:
        min_cluster_size = max(5, min(15, n_points // 12))
    else:
        min_cluster_size = int(min_cluster_size)
    min_samples = hdb.get("min_samples")
    if min_samples is None:
        return max(3, min_cluster_size // 3)
    return int(min_samples)


def cluster_dbscan(X_2d: np.ndarray, *, config: dict | None = None) -> np.ndarray:
    cfg = config or load_pipeline_config()
    db_cfg = cfg.get("dbscan", {})
    eps = float(db_cfg.get("eps", 0.027))
    min_samples = db_cfg.get("min_samples")
    if min_samples is None:
        min_samples = _resolve_hdbscan_min_samples(cfg, X_2d.shape[0])
    else:
        min_samples = int(min_samples)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return clusterer.fit_predict(X_2d)


def _encode_reference_labels(reference: np.ndarray) -> np.ndarray:
    encoder = LabelEncoder()
    return encoder.fit_transform(reference.astype(str))


def _cluster_stability(
    X_2d: np.ndarray,
    *,
    config: dict | None = None,
) -> float | None:
    """Acuerdo ARI entre dos HDBSCAN con min_cluster_size base y +2."""
    cfg = config or load_pipeline_config()
    labels_a = cluster_hdbscan(X_2d, config=cfg)
    cfg_b = copy.deepcopy(cfg)
    hdb = dict(cfg_b.get("hdbscan") or {})
    base_mcs = hdb.get("min_cluster_size")
    if base_mcs is None:
        base_mcs = max(5, min(15, X_2d.shape[0] // 12))
    hdb["min_cluster_size"] = int(base_mcs) + 2
    cfg_b["hdbscan"] = hdb
    labels_b = cluster_hdbscan(X_2d, config=cfg_b)

    mask = (labels_a >= 0) & (labels_b >= 0)
    if mask.sum() < 10:
        return None
    try:
        return float(adjusted_rand_score(labels_a[mask], labels_b[mask]))
    except ValueError:
        return None


def compute_metrics(
    X_2d: np.ndarray,
    labels: np.ndarray,
    *,
    X_features: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
    hdbscan_config: dict | None = None,
    n_samples: int | None = None,
    include_stability: bool = True,
) -> PipelineMetrics:
    mask = labels >= 0
    n_total = n_samples if n_samples is not None else len(labels)
    noise_pct = float(np.sum(labels == -1) / n_total * 100) if n_total else None

    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None
    n_clusters = len({int(x) for x in labels.tolist() if int(x) >= 0}) or None

    if mask.any():
        unique = set(labels[mask].tolist())
        if len(unique) >= 2 and mask.sum() >= len(unique) + 1:
            try:
                silhouette = float(silhouette_score(X_2d[mask], labels[mask]))
            except ValueError:
                silhouette = None
            try:
                davies_bouldin = float(davies_bouldin_score(X_2d[mask], labels[mask]))
            except ValueError:
                davies_bouldin = None

    if X_features is not None and mask.any():
        unique = set(labels[mask].tolist())
        if len(unique) >= 2:
            try:
                calinski_harabasz = float(
                    calinski_harabasz_score(X_features[mask], labels[mask])
                )
            except ValueError:
                calinski_harabasz = None

    ari = None
    nmi = None
    if reference_labels is not None and mask.any():
        ref_encoded = _encode_reference_labels(reference_labels)
        try:
            ari = float(adjusted_rand_score(ref_encoded[mask], labels[mask]))
        except ValueError:
            ari = None
        try:
            nmi = float(normalized_mutual_info_score(ref_encoded[mask], labels[mask]))
        except ValueError:
            nmi = None

    stability = None
    if include_stability:
        stability = _cluster_stability(X_2d, config=hdbscan_config)

    return PipelineMetrics(
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        calinski_harabasz=calinski_harabasz,
        n_clusters=n_clusters,
        noise_pct=noise_pct,
        ari=ari,
        nmi=nmi,
        cluster_stability=stability,
    )
