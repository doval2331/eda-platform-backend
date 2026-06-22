"""Tokens JWT para incrustar dashboards de Metabase en la app."""

from __future__ import annotations

import re
import time
from typing import Any

from jose import jwt

from app.config import get_settings


class MetabaseEmbedError(RuntimeError):
    pass


def _resolve_dashboard_id() -> int:
    settings = get_settings()
    if settings.metabase_dashboard_id:
        return int(settings.metabase_dashboard_id)
    url = (settings.metabase_dashboard_url or "").strip()
    match = re.search(r"/dashboard/(\d+)", url)
    if match:
        return int(match.group(1))
    raise MetabaseEmbedError(
        "Configura METABASE_DASHBOARD_URL (p. ej. http://localhost:3000/dashboard/2) "
        "o METABASE_DASHBOARD_ID en el backend."
    )


def embedding_is_configured() -> bool:
    settings = get_settings()
    if not settings.metabase_embedding_secret.strip():
        return False
    try:
        _resolve_dashboard_id()
        return True
    except MetabaseEmbedError:
        return False


def create_embed_token(*, run_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    secret = settings.metabase_embedding_secret.strip()
    if not secret:
        raise MetabaseEmbedError(
            "Configura METABASE_EMBEDDING_SECRET en el backend (Admin → Incrustación en Metabase)."
        )

    dashboard_id = _resolve_dashboard_id()
    # El dashboard auto-generado filtra por latest_run en SQL; no expone run_id como
    # parámetro de incrustación. Enviarlo provoca 400 en /api/embed/dashboard/.../dashcard.
    _ = run_id
    params: dict[str, list[str]] = {}

    expires_in = max(1, int(settings.metabase_embed_expire_minutes)) * 60
    payload = {
        "resource": {"dashboard": dashboard_id},
        "params": params,
        "exp": int(time.time()) + expires_in,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    instance_url = settings.metabase_url.rstrip("/")
    embed_url = f"{instance_url}/embed/dashboard/{token}#bordered=true&titled=true&downloads=true"

    return {
        "status": "ok",
        "token": token,
        "instance_url": instance_url,
        "embed_url": embed_url,
        "dashboard_id": dashboard_id,
        "expires_in_seconds": expires_in,
    }
