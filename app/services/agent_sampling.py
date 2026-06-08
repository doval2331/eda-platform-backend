from __future__ import annotations

import numpy as np
import pandas as pd


SAMPLE_COLUMNS = [
    "run_id",
    "evidence_index",
    "evidence_id",
    "preview",
    "cluster_label",
    "category",
    "categoria",
    "severity",
    "prioridad",
    "affected_service",
    "servicio_afectado",
    "assignment_group",
    "support_channel",
    "sla_breached",
    "sla_incumplido",
    "sla_breach_rate",
    "resolution_minutes",
    "avg_resolution_hours",
    "business_impact_score",
    "operational_risk_score",
    "causa_raiz_simulada",
]


def sample_cluster_records(
    evidences: pd.DataFrame,
    *,
    sample_size: int = 30,
    random_state: int = 42,
    criteria: str = "priority",
    include_noise: bool = False,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if criteria not in {"priority", "random", "mixed"}:
        raise ValueError("criteria must be one of: priority, random, mixed")
    if evidences.empty or "cluster_label" not in evidences:
        return _empty_samples()

    work = evidences.copy()
    if not include_noise:
        work = work[work["cluster_label"].fillna(-1).astype(int) != -1]
    if work.empty:
        return _empty_samples()

    sampled_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)
    for cluster_label, group in work.groupby("cluster_label", sort=True):
        scored = group.copy()
        scored["selection_score"] = _selection_score(scored)
        if criteria == "random":
            selected = scored.sample(
                n=min(sample_size, len(scored)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("evidence_index")
        elif criteria == "mixed":
            pool = scored.sort_values(
                ["selection_score", "evidence_index"],
                ascending=[False, True],
            ).head(min(len(scored), max(sample_size * 2, sample_size)))
            selected = pool.sample(
                n=min(sample_size, len(pool)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values(["selection_score", "evidence_index"], ascending=[False, True])
        else:
            selected = scored.sort_values(
                ["selection_score", "evidence_index"],
                ascending=[False, True],
            ).head(sample_size)

        selected = selected.copy()
        selected["selection_rank"] = range(1, len(selected) + 1)
        selected["selection_criteria"] = criteria
        selected["sample_size_config"] = int(sample_size)
        selected["cluster_label"] = int(cluster_label)
        sampled_frames.append(selected)

    if not sampled_frames:
        return _empty_samples()
    result = pd.concat(sampled_frames, ignore_index=True)
    keep = [col for col in SAMPLE_COLUMNS if col in result]
    extra = ["selection_rank", "selection_score", "selection_criteria", "sample_size_config"]
    return result[keep + extra].reset_index(drop=True)


def _selection_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col in ("business_impact_score", "operational_risk_score"):
        if col in df:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    for col in ("sla_breached", "sla_incumplido"):
        if col in df:
            score += df[col].fillna(False).astype(bool).astype(float) * 25
            break
    if "sla_breach_rate" in df:
        rate = pd.to_numeric(df["sla_breach_rate"], errors="coerce").fillna(0)
        score += rate.where(rate > 1, rate * 100).clip(0, 100) * 0.25
    for col in ("severity", "prioridad"):
        if col in df:
            severity = df[col].astype(str).str.lower()
            score += severity.map({"critical": 30, "critica": 30, "high": 20, "alta": 20, "medium": 10, "media": 10, "low": 3, "baja": 3}).fillna(0)
            break
    for col in ("resolution_minutes", "avg_resolution_hours", "tiempo_resolucion_horas"):
        if col in df:
            value = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if value.max() > 0:
                score += (value / value.max()).clip(0, 1) * 10
            break
    return score.round(4)


def _empty_samples() -> pd.DataFrame:
    return pd.DataFrame(columns=SAMPLE_COLUMNS + ["selection_rank", "selection_score", "selection_criteria"])
