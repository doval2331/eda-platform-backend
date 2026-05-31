from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


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


def explain_with_llm(
    *,
    question: str,
    tool_summaries: list[dict[str, Any]],
    fallback_answer: str,
) -> LlmResult:
    settings = get_settings()
    if not _enabled():
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="rules",
            detail="LLM desactivado por configuracion.",
        )
    if settings.llm_provider != "openai_compatible":
        logger.warning("Proveedor LLM no soportado: %s", settings.llm_provider)
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_fallback",
            detail=f"Proveedor LLM no soportado: {settings.llm_provider}.",
        )
    if not settings.llm_api_key and "api.openai.com" in settings.llm_api_base:
        logger.warning("LLM_ENABLED=true, pero LLM_API_KEY esta vacio.")
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_config_pending",
            detail="LLM activado por bandera, pero falta LLM_API_KEY.",
        )

    payload = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "pregunta_usuario": question,
                        "resumenes_agregados": tool_summaries,
                        "respuesta_base": fallback_answer,
                        "instruccion": (
                            "Reescribe la respuesta base en lenguaje simple. "
                            "Usa solo los resumenes agregados. Mantene una extension similar. "
                            "Si hay hallazgos de alternativas_decision, incluye hasta tres "
                            "opciones priorizadas con: foco, por que importa y proximo paso."
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
    }
    url = settings.llm_api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if settings.llm_api_key:
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
        logger.warning("No se pudo obtener explicacion LLM: HTTP %s", exc.code)
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_fallback",
            detail=_http_error_detail(exc.code),
        )
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("No se pudo obtener explicacion LLM: %s", exc)
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_fallback",
            detail=_http_error_detail(None),
        )

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        logger.warning("Respuesta LLM con formato inesperado.")
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_fallback",
            detail="El proveedor LLM respondio con un formato inesperado.",
        )
    if not content:
        return LlmResult(
            answer=fallback_answer,
            used=False,
            mode="llm_fallback",
            detail="El proveedor LLM devolvio una respuesta vacia.",
        )
    return LlmResult(
        answer=content,
        used=True,
        mode="llm_active",
        detail=f"Agente LLM activo: {settings.llm_model}.",
    )
