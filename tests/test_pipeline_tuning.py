import numpy as np

from app.services.pipeline.pipeline_config import merge_pipeline_config
from app.services.pipeline.pipeline_core import cluster_hdbscan


def test_merge_pipeline_config_applies_hdbscan_override():
    base = {
        "umap": {"n_neighbors": 15, "min_dist": 0.1},
        "hdbscan": {"min_cluster_size": None, "min_samples": None},
        "dbscan": {"eps": 0.027, "min_samples": 5},
    }
    merged = merge_pipeline_config(
        base,
        {"hdbscan_min_cluster_size": 80, "umap_n_neighbors": 25},
    )
    assert merged["hdbscan"]["min_cluster_size"] == 80
    assert merged["umap"]["n_neighbors"] == 25


def test_large_min_cluster_size_reduces_cluster_count():
    coords = np.vstack(
        [
            np.random.default_rng(0).normal(loc=[0, 0], scale=0.3, size=(120, 2)),
            np.random.default_rng(1).normal(loc=[4, 0], scale=0.3, size=(120, 2)),
            np.random.default_rng(2).normal(loc=[2, 3], scale=0.3, size=(120, 2)),
        ]
    )
    small_mcs = cluster_hdbscan(coords, config={"hdbscan": {"min_cluster_size": 8, "min_samples": 3}})
    large_mcs = cluster_hdbscan(coords, config={"hdbscan": {"min_cluster_size": 80, "min_samples": 5}})
    n_small = len({int(x) for x in small_mcs if int(x) >= 0})
    n_large = len({int(x) for x in large_mcs if int(x) >= 0})
    assert n_large <= n_small
