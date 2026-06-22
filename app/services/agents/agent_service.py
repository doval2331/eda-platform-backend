from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings
from app.services.agents.agent_sampling import sample_cluster_records
from app.services.agents.agent_traceability import TraceCollector, to_json, utc_now_iso
from app.services.agents.llm_agent import (
    INTERPRETATION_SYSTEM_PROMPT,
    STRATEGY_SYSTEM_PROMPT,
    complete_json_with_llm,
    llm_ready,
)


@dataclass(frozen=True)
class AgentRunMeta:
    llm_used: bool
    llm_mode: str
    llm_detail: str
    model_name: str


INTERPRETATION_BATCH_SIZE = 10
MAX_LLM_CLUSTERS = 15


def build_cluster_summary(evidences: pd.DataFrame) -> pd.DataFrame:
    if evidences.empty or "cluster_label" not in evidences:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cluster_label, group in evidences.groupby("cluster_label", sort=True):
        label = int(cluster_label)
        rows.append(
            {
                "run_id": str(group["run_id"].iloc[0]) if "run_id" in group else "",
                "cluster_label": label,
                "cluster_name": "outliers" if label == -1 else f"cluster_{label}",
                "evidence_count": int(len(group)),
                "avg_sla_breach_rate": _mean_first(group, ["sla_breach_rate", "sla_breached", "sla_incumplido"]),
                "avg_resolution_hours": _mean_first(group, ["avg_resolution_hours", "tiempo_resolucion_horas"]),
                "avg_risk": _mean_first(group, ["operational_risk_score", "business_impact_score"]),
                "critical_or_high_count": _critical_count(group),
                "top_category": _top_first(group, ["category", "categoria", "sector"]),
                "top_service": _top_first(group, ["affected_service", "servicio_afectado", "service_line"]),
                "top_assignment": _top_first(group, ["assignment_group", "support_channel", "canal_entrada"]),
                "top_root_cause": _top_first(group, ["causa_raiz_simulada"]),
                "top_terms": _top_terms(group),
            }
        )
    return pd.DataFrame(rows)


def run_strategy_agent(
    *,
    run_id: str,
    evidences: pd.DataFrame,
    metrics: dict[str, Any],
    sample_size: int,
    sample_criteria: str,
    model_name: str,
    tracer: TraceCollector,
) -> tuple[pd.DataFrame, AgentRunMeta]:
    use_llm = llm_ready() and model_name != "deterministic-local"
    effective_model = get_settings().llm_model if use_llm else "deterministic-local"
    fallback_rows = _strategy_rows(evidences, metrics, sample_size, sample_criteria)
    llm_result, llm_payload = (None, None)
    rows = fallback_rows
    mode = "deterministic"
    llm_used = False
    llm_mode = "rules"
    llm_detail = "Modo deterministico local."

    prompt = (
        "Actua como agente de estrategia para una corrida de clustering IT. "
        f"run_id={run_id}; filas={len(evidences)}; columnas={list(evidences.columns)}; "
        f"metricas={json.dumps(metrics, ensure_ascii=False, default=str)}; "
        f"muestra_por_cluster={sample_size}; criterio_muestra={sample_criteria}. "
        "Recomienda variables, metricas y criterios para interpretar clusters sin enviar el dataset completo."
    )

    if use_llm:
        llm_result, llm_payload = complete_json_with_llm(
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            user_payload={
                "run_id": run_id,
                "evidence_count": len(evidences),
                "columns": list(evidences.columns),
                "key_columns": _key_columns(evidences),
                "metrics": metrics,
                "sample_size": sample_size,
                "sample_criteria": sample_criteria,
                "baseline_recommendations": fallback_rows,
            },
        )
        if llm_result.used and isinstance(llm_payload, dict):
            llm_rows = _strategy_rows_from_llm(llm_payload.get("recommendations"))
            if llm_rows:
                rows = llm_rows
                mode = "llm_active"
                llm_used = True
                llm_mode = llm_result.mode
                llm_detail = llm_result.detail
        elif llm_result is not None:
            llm_mode = llm_result.mode
            llm_detail = llm_result.detail

    response = json.dumps(
        {"mode": mode, "llm_used": llm_used, "recommendations": rows},
        ensure_ascii=False,
        default=str,
    )
    trace_id = tracer.record(
        agent_name="strategy_agent",
        decision_type="segmentation_strategy",
        prompt=prompt,
        response=response,
        model_name=effective_model,
        variables_used=_available_columns(
            evidences,
            [
                "category",
                "categoria",
                "severity",
                "prioridad",
                "affected_service",
                "servicio_afectado",
                "assignment_group",
                "sla_breached",
                "sla_incumplido",
                "sla_breach_rate",
                "avg_resolution_hours",
                "business_impact_score",
                "operational_risk_score",
            ],
        ),
        input_artifacts=["run_evidences", "run_registry", "cluster_summary"],
        parameters={"sample_size": sample_size, "sample_criteria": sample_criteria},
    )
    for row in rows:
        row["run_id"] = run_id
        row["trace_id"] = trace_id
    meta = AgentRunMeta(
        llm_used=llm_used,
        llm_mode=llm_mode,
        llm_detail=llm_detail,
        model_name=effective_model,
    )
    return pd.DataFrame(rows), meta


def run_interpretation_agent(
    *,
    run_id: str,
    evidences: pd.DataFrame,
    sample_size: int,
    sample_criteria: str,
    random_state: int,
    model_name: str,
    tracer: TraceCollector,
) -> tuple[pd.DataFrame, pd.DataFrame, AgentRunMeta]:
    use_llm = llm_ready() and model_name != "deterministic-local"
    effective_model = get_settings().llm_model if use_llm else "deterministic-local"
    summary = build_cluster_summary(evidences)
    samples = sample_cluster_records(
        evidences,
        sample_size=sample_size,
        random_state=random_state,
        criteria=sample_criteria,
    )
    rows: list[dict[str, object]] = []
    llm_used = False
    llm_mode = "rules"
    llm_detail = "Modo deterministico local."
    if summary.empty:
        meta = AgentRunMeta(
            llm_used=False,
            llm_mode=llm_mode,
            llm_detail=llm_detail,
            model_name=effective_model,
        )
        return samples, pd.DataFrame(), meta

    cluster_frames = [
        cluster
        for _, cluster in summary.sort_values("cluster_label").iterrows()
        if int(cluster["cluster_label"]) != -1
    ]
    llm_targets = _prioritize_clusters_for_llm(cluster_frames)
    llm_by_cluster = _llm_cluster_interpretations(llm_targets, samples) if use_llm else {}
    if llm_by_cluster:
        llm_used = True
        llm_mode = "llm_active"
        llm_detail = (
            f"Agente LLM activo: {effective_model}. "
            f"Interpretacion enriquecida en {len(llm_by_cluster)} de {len(cluster_frames)} clusters."
        )

    for cluster in cluster_frames:
        cluster_label = int(cluster["cluster_label"])
        cluster_samples = (
            samples[samples["cluster_label"].astype(int) == cluster_label]
            if not samples.empty
            else pd.DataFrame()
        )
        insight = _cluster_interpretation(run_id, cluster, cluster_samples)
        llm_insight = llm_by_cluster.get(cluster_label)
        if llm_insight:
            insight = _merge_cluster_interpretation(insight, llm_insight)
        mode = "llm_active" if llm_insight else "deterministic"
        insight["interpretation_mode"] = mode
        prompt = (
            "Actua como agente de interpretacion de clusters IT. "
            f"Resumen_cluster={json.dumps(cluster.to_dict(), ensure_ascii=False, default=str)}; "
            f"muestras={json.dumps(cluster_samples.head(30).to_dict(orient='records'), ensure_ascii=False, default=str)}. "
            "Explica patrones, diferencias, posibles causas y recomendaciones usando solo agregados y muestras."
        )
        response = json.dumps(
            {"mode": mode, "llm_used": bool(llm_insight), "interpretation": insight},
            ensure_ascii=False,
            default=str,
        )
        trace_id = tracer.record(
            agent_name="interpretation_agent",
            decision_type="cluster_interpretation",
            prompt=prompt,
            response=response,
            model_name=effective_model,
            variables_used=_available_columns(
                cluster_samples,
                [
                    "category",
                    "categoria",
                    "severity",
                    "prioridad",
                    "affected_service",
                    "servicio_afectado",
                    "assignment_group",
                    "sla_breached",
                    "sla_breach_rate",
                    "preview",
                ],
            ),
            input_artifacts=["cluster_summary", "cluster_samples"],
            parameters={"cluster_label": cluster_label, "sample_size": int(len(cluster_samples))},
        )
        insight["trace_id"] = trace_id
        rows.append(insight)

    meta = AgentRunMeta(
        llm_used=llm_used,
        llm_mode=llm_mode,
        llm_detail=llm_detail,
        model_name=effective_model,
    )
    return samples, pd.DataFrame(rows), meta


def _strategy_rows_from_llm(raw_recommendations: object) -> list[dict[str, object]]:
    if not isinstance(raw_recommendations, list):
        return []
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw_recommendations):
        if not isinstance(item, dict):
            continue
        recommendation = str(item.get("recommendation") or "").strip()
        if not recommendation:
            continue
        variables = item.get("variables_used", [])
        if isinstance(variables, str):
            variables_payload = variables
        else:
            variables_payload = to_json(variables if isinstance(variables, list) else [])
        rows.append(
            {
                "strategy_id": str(item.get("strategy_id") or f"llm_strategy_{index + 1}"),
                "strategy_type": str(item.get("strategy_type") or "interpretation"),
                "recommendation": recommendation,
                "justification": str(item.get("justification") or "Recomendacion generada por el agente LLM."),
                "variables_used": variables_payload,
                "metric_or_criterion": str(item.get("metric_or_criterion") or "lectura de clusters"),
                "priority": str(item.get("priority") or "medium"),
                "created_at": utc_now_iso(),
            }
        )
    return rows


def _prioritize_clusters_for_llm(
    cluster_frames: list[pd.Series],
    limit: int = MAX_LLM_CLUSTERS,
) -> list[pd.Series]:
    def score(cluster: pd.Series) -> tuple[float, int]:
        return (
            _safe_float(cluster.get("avg_risk")),
            int(cluster.get("evidence_count", 0) or 0),
        )

    ordered = sorted(cluster_frames, key=score, reverse=True)
    return ordered[:limit]


def _llm_cluster_interpretations(
    cluster_frames: list[pd.Series],
    samples: pd.DataFrame,
) -> dict[int, dict[str, str]]:
    if not cluster_frames:
        return {}
    merged: dict[int, dict[str, str]] = {}
    for start in range(0, len(cluster_frames), INTERPRETATION_BATCH_SIZE):
        batch = cluster_frames[start : start + INTERPRETATION_BATCH_SIZE]
        payload_clusters: list[dict[str, object]] = []
        for cluster in batch:
            cluster_label = int(cluster["cluster_label"])
            cluster_samples = (
                samples[samples["cluster_label"].astype(int) == cluster_label]
                if not samples.empty
                else pd.DataFrame()
            )
            payload_clusters.append(
                {
                    "cluster_label": cluster_label,
                    "cluster_name": str(cluster.get("cluster_name") or f"cluster_{cluster_label}"),
                    "summary": cluster.to_dict(),
                    "samples": cluster_samples.head(8).to_dict(orient="records"),
                }
            )
        llm_result, llm_payload = complete_json_with_llm(
            system_prompt=INTERPRETATION_SYSTEM_PROMPT,
            user_payload={"clusters": payload_clusters},
        )
        if not llm_result.used or not isinstance(llm_payload, dict):
            continue
        for item in llm_payload.get("interpretations", []):
            if not isinstance(item, dict):
                continue
            try:
                cluster_label = int(item.get("cluster_label"))
            except (TypeError, ValueError):
                continue
            merged[cluster_label] = {
                key: str(item.get(key) or "").strip()
                for key in (
                    "summary",
                    "main_characteristics",
                    "possible_causes",
                    "recommendations",
                    "business_conclusion",
                )
                if str(item.get(key) or "").strip()
            }
    return merged


def _merge_cluster_interpretation(
    base: dict[str, object],
    llm_insight: dict[str, str],
) -> dict[str, object]:
    merged = dict(base)
    for key, value in llm_insight.items():
        if value:
            merged[key] = value
    return merged


def _strategy_rows(
    evidences: pd.DataFrame,
    metrics: dict[str, Any],
    sample_size: int,
    sample_criteria: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "strategy_id": "feature_mix",
            "strategy_type": "segmentation",
            "recommendation": "Usar variables textuales/preview y variables operativas para explicar grupos similares.",
            "justification": f"La corrida tiene {len(evidences)} evidencias y columnas disponibles: {', '.join(_key_columns(evidences))}.",
            "variables_used": to_json(_key_columns(evidences)),
            "metric_or_criterion": "perfil operativo + similitud de clusters",
            "priority": "high",
            "created_at": utc_now_iso(),
        },
        {
            "strategy_id": "cluster_explanation_variables",
            "strategy_type": "interpretation",
            "recommendation": "Explicar clusters por servicio, categoria, severidad/prioridad, SLA, tiempos, riesgo y causa raiz si existe.",
            "justification": "Estas dimensiones traducen el cluster tecnico a una lectura comprensible para negocio.",
            "variables_used": to_json(
                _available_columns(
                    evidences,
                    [
                        "affected_service",
                        "servicio_afectado",
                        "category",
                        "categoria",
                        "severity",
                        "prioridad",
                        "sla_breach_rate",
                        "sla_breached",
                        "avg_resolution_hours",
                        "operational_risk_score",
                        "causa_raiz_simulada",
                    ],
                )
            ),
            "metric_or_criterion": "diferencias entre clusters y prioridad operativa",
            "priority": "high",
            "created_at": utc_now_iso(),
        },
        {
            "strategy_id": "cluster_sampling",
            "strategy_type": "sampling",
            "recommendation": f"Seleccionar hasta {sample_size} evidencias por cluster con criterio {sample_criteria}.",
            "justification": "El agente debe trabajar con muestras y agregados; no con todas las filas.",
            "variables_used": to_json(["cluster_label", "evidence_id", "selection_score"]),
            "metric_or_criterion": sample_criteria,
            "priority": "high",
            "created_at": utc_now_iso(),
        },
    ]
    if metrics:
        rows.append(
            {
                "strategy_id": "metric_review",
                "strategy_type": "validation",
                "recommendation": "Validar calidad de agrupamiento antes de tomar decisiones sobre clusters.",
                "justification": f"Metricas disponibles: {json.dumps(metrics, ensure_ascii=False, default=str)}.",
                "variables_used": to_json(["silhouette", "davies_bouldin", "n_clusters", "noise_pct", "cluster_stability"]),
                "metric_or_criterion": "calidad clustering",
                "priority": "medium",
                "created_at": utc_now_iso(),
            }
        )
    return rows


def _cluster_interpretation(
    run_id: str,
    cluster: pd.Series,
    samples: pd.DataFrame,
) -> dict[str, object]:
    cluster_label = int(cluster["cluster_label"])
    count = int(cluster.get("evidence_count", 0) or 0)
    top_category = str(cluster.get("top_category") or "sin categoria dominante")
    top_service = str(cluster.get("top_service") or "sin servicio dominante")
    sla = _safe_float(cluster.get("avg_sla_breach_rate"))
    risk = _safe_float(cluster.get("avg_risk"))
    resolution = _safe_float(cluster.get("avg_resolution_hours"))
    risk_level = _risk_level(sla, risk, int(cluster.get("critical_or_high_count", 0) or 0))
    sample_ids = samples["evidence_id"].astype(str).tolist() if "evidence_id" in samples else []
    sample_values = _sample_top_values(samples)
    characteristics = [
        f"{count} evidencias",
        f"categoria dominante: {top_category}",
        f"servicio dominante: {top_service}",
        f"SLA medio: {_pct(sla)}",
        f"resolucion media: {resolution:.2f} h",
        f"riesgo medio: {risk:.2f}",
    ]
    characteristics.extend(f"{key}: {value}" for key, value in sample_values.items())
    return {
        "run_id": run_id,
        "cluster_insight_id": f"{run_id}-cluster-{cluster_label}",
        "cluster_label": cluster_label,
        "cluster_name": str(cluster.get("cluster_name") or f"cluster_{cluster_label}"),
        "summary": (
            f"Cluster {cluster_label} agrupa principalmente {top_category} en {top_service}, "
            f"con {count} evidencias, SLA medio {_pct(sla)} y riesgo {risk:.2f}."
        ),
        "main_characteristics": "; ".join(characteristics),
        "highlighted_variables": to_json(
            sorted(
                set(
                    [
                        "top_category",
                        "top_service",
                        "avg_sla_breach_rate",
                        "avg_resolution_hours",
                        "avg_risk",
                        "top_root_cause",
                        *sample_values.keys(),
                    ]
                )
            )
        ),
        "possible_causes": _possible_causes(cluster, sample_values),
        "recommendations": _recommendations(risk_level, top_category, top_service),
        "business_conclusion": f"Perfil {risk_level}: revisar recurrencia, responsables y servicios afectados antes de convertirlo en accion operativa.",
        "sample_evidence_ids": to_json(sample_ids),
        "sample_size": int(len(samples)),
        "risk_level": risk_level,
        "generated_at": utc_now_iso(),
    }


def _mean_first(df: pd.DataFrame, columns: list[str]) -> float | None:
    for column in columns:
        if column not in df:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            values = series.astype(float)
        else:
            values = pd.to_numeric(series, errors="coerce")
        if values.notna().any():
            return float(values.mean())
    return None


def _top_first(df: pd.DataFrame, columns: list[str]) -> str:
    for column in columns:
        if column not in df:
            continue
        counts = df[column].dropna().astype(str).value_counts()
        if not counts.empty:
            return str(counts.index[0])
    return ""


def _critical_count(df: pd.DataFrame) -> int:
    for column in ("severity", "prioridad"):
        if column in df:
            values = df[column].astype(str).str.lower()
            return int(values.isin(["critical", "critica", "high", "alta"]).sum())
    return 0


def _top_terms(df: pd.DataFrame) -> str:
    text = " ".join(df.get("preview", pd.Series(dtype=str)).dropna().astype(str).tolist())
    if not text:
        return ""
    words = [
        word.strip(".,:;()[]{}").lower()
        for word in text.split()
        if len(word.strip(".,:;()[]{}")) > 3
    ]
    if not words:
        return ""
    counts = pd.Series(words).value_counts().head(8)
    return ", ".join(counts.index.tolist())


def _available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in df]


def _key_columns(df: pd.DataFrame) -> list[str]:
    return _available_columns(
        df,
        [
            "preview",
            "category",
            "categoria",
            "severity",
            "prioridad",
            "affected_service",
            "servicio_afectado",
            "assignment_group",
            "support_channel",
            "sla_breach_rate",
            "sla_breached",
            "avg_resolution_hours",
            "operational_risk_score",
        ],
    )


def _sample_top_values(samples: pd.DataFrame) -> dict[str, str]:
    if samples.empty:
        return {}
    result: dict[str, str] = {}
    for label, columns in {
        "muestra_servicio_dominante": ["affected_service", "servicio_afectado", "service_line"],
        "muestra_categoria_dominante": ["category", "categoria", "sector"],
        "muestra_prioridad_dominante": ["severity", "prioridad"],
        "muestra_equipo_dominante": ["assignment_group", "support_channel", "canal_entrada"],
    }.items():
        value = _top_first(samples, columns)
        if value:
            result[label] = value
    return result


def _safe_float(value: object) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pct(value: float) -> str:
    pct = value * 100 if abs(value) <= 1 else value
    return f"{pct:.2f}%"


def _risk_level(sla: float, risk: float, critical_count: int) -> str:
    normalized_sla = sla * 100 if abs(sla) <= 1 else sla
    if normalized_sla >= 35 or risk >= 70 or critical_count >= 20:
        return "alto"
    if normalized_sla >= 15 or risk >= 45 or critical_count >= 5:
        return "medio"
    return "bajo"


def _possible_causes(cluster: pd.Series, sample_values: dict[str, str]) -> str:
    signals = [
        str(cluster.get("top_category") or ""),
        str(cluster.get("top_service") or ""),
        str(cluster.get("top_root_cause") or ""),
        str(cluster.get("top_terms") or ""),
        *sample_values.values(),
    ]
    signals = [signal for signal in signals if signal and signal != "nan"]
    if not signals:
        return "No hay senales suficientes; revisar la muestra y las variables disponibles."
    return "Posible recurrencia asociada a " + ", ".join(signals[:5]) + "."


def _recommendations(risk_level: str, top_category: str, top_service: str) -> str:
    if risk_level == "alto":
        return f"Priorizar revision operativa de {top_category}/{top_service}, validar causa raiz y definir accion correctiva."
    if risk_level == "medio":
        return f"Monitorear {top_category}/{top_service}, comparar contra otros clusters y revisar evidencias de muestra."
    return f"Usar {top_category}/{top_service} como perfil descriptivo y mantener seguimiento en el dashboard."
