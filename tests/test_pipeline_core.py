import numpy as np
import pytest

from app.services.pipeline.pipeline import run_pipeline
from app.services.pipeline.pipeline_core import (
    cluster_dbscan,
    cluster_hdbscan,
    compute_metrics,
)


def _blob_coords(n_per_cluster: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    chunks = [rng.normal(loc=center, scale=0.25, size=(n_per_cluster, 2)) for center in centers]
    return np.vstack(chunks)


def test_cluster_dbscan_returns_labels_with_noise():
    coords = _blob_coords()
    cfg = {"dbscan": {"eps": 0.6, "min_samples": 5}, "hdbscan": {}}
    labels = cluster_dbscan(coords, config=cfg)
    assert labels.shape == (coords.shape[0],)
    assert set(np.unique(labels)) <= {-1, 0, 1, 2}


def test_baseline_metrics_no_stability():
    coords = _blob_coords()
    labels = cluster_dbscan(coords, config={"dbscan": {"eps": 0.6, "min_samples": 5}})
    metrics = compute_metrics(coords, labels, include_stability=False)
    assert metrics.cluster_stability is None
    assert metrics.n_clusters is not None
    assert metrics.n_clusters >= 1


def test_compute_metrics_ari_with_reference():
    coords = _blob_coords(n_per_cluster=30)
    labels = cluster_hdbscan(coords, config={"hdbscan": {"min_cluster_size": 10, "min_samples": 3}})
    reference = np.array([0, 1, 2] * 30)
    metrics = compute_metrics(
        coords,
        labels,
        reference_labels=reference,
        include_stability=False,
    )
    assert metrics.ari is not None
    assert metrics.nmi is not None
    assert -1.0 <= metrics.ari <= 1.0
    assert 0.0 <= metrics.nmi <= 1.0


@pytest.mark.parametrize("modality", ["texto", "imagen", "multimodal"])
def test_pipeline_result_includes_baseline(modality):
    result = run_pipeline(
        modality=modality,
        reduction_method="PCA",
        seed=7,
        n_samples=120,
        n_features=12,
        n_true_clusters=3,
    )
    assert result.baseline_algorithm == "DBSCAN"
    assert result.baseline_metrics is not None
    assert result.baseline_metrics.cluster_stability is None
    assert result.metrics.cluster_stability is not None or result.metrics.n_clusters is not None
    assert len(result.cluster_labels) == 120
    assert result.metrics.n_clusters is not None
