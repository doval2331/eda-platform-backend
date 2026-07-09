from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

STRATEGY_SYSTEM_PROMPT = """
Eres un agente de estrategia para analisis exploratorio de incidencias IT tras un clustering.
Recibes solo metadatos de columnas, metricas de calidad y parametros de muestreo.
No inventes columnas ni metricas que no esten en la entrada.
Responde SOLO con JSON valido, sin markdown, con esta forma:
{"recommendations":[{"strategy_id":"...","strategy_type":"segmentation|interpretation|sampling|validation",
"recommendation":"...","justification":"...","variables_used":["col1"],"metric_or_criterion":"...",
"priority":"high|medium|low"}]}
Genera entre 3 y 5 recomendaciones breves y accionables en espanol.
"""

INTERPRETATION_SYSTEM_PROMPT = """
Eres un agente de interpretacion de clusters de incidencias IT.
Recibes resumenes agregados por cluster y muestras acotadas; no afirmes causalidad absoluta.
Responde SOLO con JSON valido, sin markdown, con esta forma:
{"interpretations":[{"cluster_label":0,"summary":"...","main_characteristics":"...",
"possible_causes":"...","recommendations":"...","business_conclusion":"..."}]}
Incluye una entrada por cada cluster_label recibido. Texto claro para negocio en espanol.
"""

DASHBOARD_DESIGN_SYSTEM_PROMPT = """
Eres un arquitecto analitico para dashboards conversacionales de incidencias IT.
Tu tarea es disenar una especificacion de dashboard, no escribir HTML ni codigo.
Recibes contexto agregado: resumen del dataset, columnas, metricas, parametros,
clusters, insights seleccionados, evidencias persistidas e historial disponible.
No inventes datos, columnas, metricas, fuentes ni conteos. Usa solo el contexto recibido.
Si falta informacion, dilo dentro de evidencias o recomendaciones como "sin dato".
Usa operational_readiness del contexto para graduar la respuesta:
- operational: puedes proponer decisiones operativas con drill-down y tickets.
- interpretive: presenta lectura asistida y pide validar evidencia antes de decidir.
- limited: evita conclusiones fuertes y explica que faltan evidencias materializadas.
Usa recommendation_feedback si existe:
- prioriza patrones parecidos a recomendaciones marcadas como utiles;
- reformula o baja prioridad de recomendaciones marcadas como no utiles;
- no ocultes una recomendacion con evidencia fuerte, pero explica por que vuelve a aparecer;
- no inventes feedback ni asumas preferencias si no viene en el contexto.
Usa dashboard_usage_summary si existe:
- prioriza graficos, drill-downs y acciones que el usuario realmente abrio o envio al agente;
- si una recomendacion aparece pero nunca se usa, proponla con una pregunta mas clara o una accion mas concreta;
- no uses conteos de uso como evidencia de negocio, solo como senal de experiencia y priorizacion.
Diferencia las recomendaciones para usuario funcional y usuario experto.
Devuelve SOLO JSON valido, sin markdown, con exactamente esta forma general:
{
  "executive_summary": {
    "title": "Resumen ejecutivo",
    "dataset_name": "",
    "analysis_objective": "",
    "records_count": 0,
    "columns_count": 0,
    "main_variables": [],
    "key_metrics": [],
    "summary": ""
  },
  "semantic_variables": [
    {
      "name": "",
      "label": "",
      "role": "business|metric|technical|identifier|unknown",
      "description": "",
      "recommended_use": "",
      "avoid_as_metric": false
    }
  ],
  "priority_findings": [
    {
      "id": "",
      "title": "",
      "priority": "alta|media|baja",
      "impact": "",
      "urgency": "",
      "evidence": "",
      "suggested_action": "",
      "related_variables": [],
      "suggested_question": ""
    }
  ],
  "agent_recommendations": [
    {
      "id": "",
      "title": "",
      "why_it_matters": "",
      "what_to_analyze": "",
      "recommended_next_step": "",
      "audience": "funcional|experto|ambos",
      "action_type": "chart|chat|conclusion",
      "linked_visualization_id": "",
      "evidence_needed": ""
    }
  ],
  "suggested_visualizations": [
    {
      "id": "",
      "title": "",
      "chart_type": "bar|line|scatter|priority_matrix|distribution|ranking|heatmap|boxplot",
      "x": "",
      "y": "",
      "metric": "",
      "group_by": "",
      "aggregation": "",
      "filters": [],
      "reason": "",
      "evidence_used": "",
      "question_answered": "",
      "audience": "funcional|experto|ambos",
      "what_i_am_seeing": "",
      "why_it_matters": "",
      "suggested_action": "",
      "drilldown": ""
    }
  ],
  "active_chart_default": {
    "visualization_id": "",
    "explanation": ""
  },
  "conclusions": [
    {
      "id": "",
      "conclusion": "",
      "evidence": "",
      "related_chart": "",
      "related_metric": "",
      "related_items": [],
      "confidence": "alta|media|baja",
      "recommended_action": "",
      "source": "llm|rules|dataset|pipeline",
      "evidence_quality": ""
    }
  ],
  "evidence_line": [
    {
      "step": 1,
      "title": "",
      "description": "",
      "source": "dataset|pipeline|cluster|insight|llm|user",
      "related_items": []
    }
  ],
  "suggested_questions": {
    "functional_user": [],
    "expert_user": []
  }
}
Genera entre 3 y 6 visualizaciones y entre 3 y 6 hallazgos prioritarios cuando haya datos.
Prioriza graficas interpretables para negocio antes que identificadores tecnicos.
No uses cluster_label como grafica principal si hay variables de negocio mas claras.
En x, y, metric y group_by usa nombres exactos de columnas disponibles o semantic_variables.name.
No uses etiquetas funcionales como si fueran columnas si no aparecen en el contexto.
Si quieres contar registros, usa metric="count"; no inventes una columna llamada Incidencias.
Si falta una variable de negocio, elige la alternativa disponible mas cercana y explicalo en reason.
Traduce nombres tecnicos a lenguaje funcional cuando sea posible:
cluster_label = "Grupo tecnico" y no debe usarse como metrica funcional principal.
No Of Reassignments = "Cantidad de reasignaciones".
affected_service = "Servicio afectado".
Clasifica cada recomendacion:
- action_type="chart" solo si existe una suggested_visualization relacionada y graficable.
- action_type="chat" si requiere interpretacion, hipotesis o explicacion textual.
- action_type="conclusion" si produce una conclusion ejecutiva presentable.
Para cada grafico explica que se ve, por que importa, que evidencia lo respalda,
que accion sugiere y que drill-down operativo conviene abrir.
En conclusions.related_chart usa exactamente el id de una suggested_visualizations existente.
Si una conclusion no tiene grafico aplicable, deja related_chart vacio y explica la evidencia.
En conclusions.related_items incluye ids, grupos, servicios o etiquetas de evidencia presentes en el contexto.
""" 


@dataclass(frozen=True)
class LlmResult:
    answer: str
    used: bool
    mode: str
    detail: str


SYSTEM_PROMPT = """
Eres una capa conversacional controlada para analisis exploratorio de incidencias IT.
Tu funcion es interpretar resultados agregados calculados por herramientas internas del backend.
No tienes acceso directo a DuckDB, no consultas filas completas y no ejecutas acciones sobre datos.
No digas que clasificas incidencias: usa "agrupa", "detecta patrones" o "sugiere perfiles".
No inventes metricas, causas ni recomendaciones. Si falta un dato, dilo como "sin dato".
No afirmes causalidad absoluta; habla de patrones para investigar.
Cuando recibas alternativas de decision, presentalas como opciones de priorizacion para que
el usuario las evalue; no indiques que el sistema toma decisiones automaticamente.
Da una respuesta breve, clara, accionable y en espanol para usuarios no expertos.
"""


def _enabled() -> bool:
    settings = get_settings()
    return bool(settings.llm_enabled)


def _http_error_detail(status_code: int | None) -> str:
    if status_code == 401:
        return "El proveedor LLM rechazo la API key. Revisa LLM_API_KEY."
    if status_code == 403:
        return "El proveedor LLM rechazo permisos de la cuenta o del proyecto."
    if status_code == 404:
        return "El proveedor LLM no encontro el endpoint o modelo configurado."
    if status_code == 429:
        return "El proveedor LLM respondio HTTP 429: cuota, rate limit o facturacion insuficiente."
    if status_code and status_code >= 500:
        return f"El proveedor LLM respondio HTTP {status_code}: error temporal del servicio."
    if status_code:
        return f"El proveedor LLM respondio HTTP {status_code}."
    return "No se pudo contactar el proveedor LLM; se uso respuesta local."


def llm_ready() -> bool:
    settings = get_settings()
    if not _enabled():
        return False
    if settings.llm_provider != "openai_compatible":
        return False
    if not settings.llm_api_key and "api.openai.com" in settings.llm_api_base:
        return False
    return True


def _llm_unavailable_result(*, fallback: str, mode: str, detail: str) -> LlmResult:
    return LlmResult(answer=fallback, used=False, mode=mode, detail=detail)


def _extract_json_payload(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _post_chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> LlmResult:
    settings = get_settings()
    if not _enabled():
        return _llm_unavailable_result(
            fallback="",
            mode="rules",
            detail="LLM desactivado por configuracion.",
        )
    if settings.llm_provider != "openai_compatible":
        logger.warning("Proveedor LLM no soportado: %s", settings.llm_provider)
        return _llm_unavailable_result(
            fallback="",
            mode="llm_fallback",
            detail=f"Proveedor LLM no soportado: {settings.llm_provider}.",
        )
    if not settings.llm_api_key and "api.openai.com" in settings.llm_api_base:
        logger.warning("LLM_ENABLED=true, pero LLM_API_KEY esta vacio.")
        return _llm_unavailable_result(
            fallback="",
            mode="llm_config_pending",
            detail="LLM activado por bandera, pero falta LLM_API_KEY.",
        )

    payload = {
        "model": settings.llm_model,
        "temperature": temperature,
        "messages": messages,
    }
    url = settings.llm_api_base.rstrip("/") + "/chat/completions"
    api_version = settings.llm_api_version.strip()
    if not api_version and "openai.azure.com" in settings.llm_api_base:
        api_version = "2024-12-01-preview"
    if api_version:
        url = f"{url}?{urllib.parse.urlencode({'api-version': api_version})}"
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        if "openai.azure.com" in settings.llm_api_base:
            headers["api-key"] = settings.llm_api_key
        else:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=settings.llm_timeout_seconds
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("No se pudo obtener respuesta LLM: HTTP %s", exc.code)
        return _llm_unavailable_result(
            fallback="",
            mode="llm_fallback",
            detail=_http_error_detail(exc.code),
        )
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("No se pudo obtener respuesta LLM: %s", exc)
        return _llm_unavailable_result(
            fallback="",
            mode="llm_fallback",
            detail=_http_error_detail(None),
        )

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        logger.warning("Respuesta LLM con formato inesperado.")
        return _llm_unavailable_result(
            fallback="",
            mode="llm_fallback",
            detail="El proveedor LLM respondio con un formato inesperado.",
        )
    if not content:
        return _llm_unavailable_result(
            fallback="",
            mode="llm_fallback",
            detail="El proveedor LLM devolvio una respuesta vacia.",
        )
    return LlmResult(
        answer=content,
        used=True,
        mode="llm_active",
        detail=f"Agente LLM activo: {settings.llm_model}.",
    )


def complete_with_llm(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float = 0.2,
) -> LlmResult:
    return _post_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, default=str),
            },
        ],
        temperature=temperature,
    )


def complete_json_with_llm(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float = 0.2,
) -> tuple[LlmResult, Any | None]:
    result = complete_with_llm(
        system_prompt=system_prompt,
        user_payload=user_payload,
        temperature=temperature,
    )
    if not result.used:
        return result, None
    try:
        return result, _extract_json_payload(result.answer)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("No se pudo parsear JSON del LLM: %s", exc)
        return LlmResult(
            answer=result.answer,
            used=False,
            mode="llm_fallback",
            detail="El proveedor LLM no devolvio JSON valido.",
        ), None


def design_dashboard_with_llm(
    *,
    user_payload: dict[str, Any],
) -> tuple[LlmResult, Any | None]:
    return complete_json_with_llm(
        system_prompt=DASHBOARD_DESIGN_SYSTEM_PROMPT,
        user_payload=user_payload,
        temperature=0.1,
    )


def explain_with_llm(
    *,
    question: str,
    tool_summaries: list[dict[str, Any]],
    fallback_answer: str,
    conversation_history: list[dict[str, str]] | None = None,
    document_context: str | None = None,
    dashboard_context: dict[str, Any] | None = None,
) -> LlmResult:
    dashboard_instruction = ""
    if dashboard_context:
        dashboard_instruction = (
            "Hay una seleccion explicita del dashboard conversacional. "
            "Responde exactamente sobre esa accion, visualizacion, hallazgo o recomendacion. "
            "No cambies a otro hallazgo salvo que este directamente relacionado. "
            "Menciona al inicio que elemento se esta analizando y usa la evidencia enviada."
        )
    result = complete_with_llm(
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "pregunta_usuario": question,
            "historial_reciente": conversation_history or [],
            "resumenes_agregados": tool_summaries,
            "respuesta_base": fallback_answer,
            "contexto_documental": document_context or "",
            "contexto_dashboard": dashboard_context or {},
            "instruccion": (
                "Reescribe la respuesta base en lenguaje simple. "
                "Usa solo los resumenes agregados y respeta la intencion de la pregunta. "
                f"{dashboard_instruction} "
                "Si hay contexto_documental relacionado, mencionalo solo cuando aporte al analisis. "
                "No repitas una respuesta generica si hay una herramienta especifica. "
                "Mantene una extension similar. "
                "Si el usuario pregunta por una fuente o archivo, responde con el "
                "nombre de fuente disponible y que revisar dentro de esa fuente. "
                "Si pregunta por datos faltantes, explica conteos, columnas revisadas "
                "y donde comprobarlos; no propongas alternativas generales. "
                "Si pregunta por dashboard, lista hallazgos accionables y por que llevarlos. "
                "Si hay hallazgos de alternativas_decision, incluye hasta tres "
                "opciones priorizadas con: foco, por que importa y proximo paso."
            ),
        },
    )
    if result.used:
        return result
    return LlmResult(
        answer=fallback_answer,
        used=False,
        mode=result.mode,
        detail=result.detail,
    )
