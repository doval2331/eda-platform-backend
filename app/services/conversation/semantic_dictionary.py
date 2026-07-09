from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_DICTIONARY_PATH = Path(__file__).with_name("semantic_dictionary.json")
DEFAULT_PROJECT_DICTIONARY_DIR = Path(__file__).with_name("semantic_dictionaries")
VALID_ROLES = {"business", "metric", "technical", "identifier", "unknown"}
VALID_TYPES = {"categorical", "numeric", "boolean", "date", "text", ""}


def semantic_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def get_semantic_dictionary(
    base_dictionary: dict[str, dict[str, Any]],
    *,
    project_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    dictionary: dict[str, dict[str, Any]] = {}
    for key, value in base_dictionary.items():
        _register_entry(dictionary, key, value)
    for key, value in _configured_semantic_entries(_project_scope(project_id)).items():
        _register_entry(dictionary, key, value)
    return dictionary


def reload_semantic_dictionary() -> None:
    _configured_semantic_entries.cache_clear()


def semantic_dictionary_path(project_id: str | None = None) -> Path:
    project_scope = _project_scope(project_id)
    if project_scope:
        base_dir = Path(os.getenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR") or DEFAULT_PROJECT_DICTIONARY_DIR)
        return base_dir / f"{project_scope}.json"
    return Path(os.getenv("CONVERSATION_SEMANTIC_DICTIONARY_PATH") or DEFAULT_DICTIONARY_PATH)


def semantic_dictionary_status(
    base_dictionary: dict[str, dict[str, Any]],
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    project_scope = _project_scope(project_id)
    path = semantic_dictionary_path(project_scope)
    configured = load_configured_semantic_variables(project_id=project_scope)
    env_path = os.getenv("CONVERSATION_SEMANTIC_DICTIONARY_PATH")
    env_dir = os.getenv("CONVERSATION_SEMANTIC_DICTIONARY_DIR")
    return {
        "source": str(path),
        "exists": path.exists(),
        "scope": "project_file" if project_scope else "environment_file" if env_path else "default_file",
        "project_id": project_scope,
        "env_var": "CONVERSATION_SEMANTIC_DICTIONARY_DIR" if project_scope else "CONVERSATION_SEMANTIC_DICTIONARY_PATH",
        "configurable": True,
        "writable": path.parent.exists() and os.access(path.parent, os.W_OK),
        "base_total": len(base_dictionary),
        "configured_total": len(configured),
        "governed": bool(configured),
        "project_scope_enabled": bool(project_scope),
        "project_dictionary_dir": str(Path(env_dir or DEFAULT_PROJECT_DICTIONARY_DIR)),
    }


def load_configured_semantic_variables(project_id: str | None = None) -> list[dict[str, Any]]:
    path = semantic_dictionary_path(project_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("variables") if isinstance(payload, dict) else None
    return [item for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []


def save_configured_semantic_variables(
    entries: list[dict[str, Any]],
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    normalized = [_normalize_configured_entry(item) for item in entries if isinstance(item, dict)]
    normalized = [item for item in normalized if item["name"]]
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in normalized:
        key = semantic_key(item["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    path = semantic_dictionary_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"variables": deduped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reload_semantic_dictionary()
    return {
        "path": str(path),
        "scope": "project_file" if _project_scope(project_id) else "global_file",
        "project_id": _project_scope(project_id),
        "total": len(deduped),
        "variables": deduped,
    }


@lru_cache(maxsize=64)
def _configured_semantic_entries(project_scope: str = "") -> dict[str, dict[str, Any]]:
    entries = load_configured_semantic_variables(project_id=project_scope)
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        normalized = _normalize_configured_entry(item)
        name = normalized["name"]
        if not name:
            continue
        aliases = normalized["aliases"]
        entry = {
            "label": normalized["label"],
            "role": normalized["role"],
            "description": normalized["description"],
            "recommended_use": normalized["recommended_use"],
            "avoid_as_metric": normalized["avoid_as_metric"],
            "can_chart": normalized["can_chart"],
            "semantic_type": normalized["type"],
            "aliases": aliases,
        }
        result[name] = entry
        for alias in aliases:
            result[alias] = entry
    return result


def _project_scope(project_id: str | None) -> str:
    text = str(project_id or "").strip()
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)[:120]


def _register_entry(dictionary: dict[str, dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    entry = dict(value)
    aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
    for candidate in [key, *aliases, semantic_key(key), *(semantic_key(alias) for alias in aliases)]:
        if candidate:
            dictionary[candidate] = entry


def _normalize_configured_entry(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "").strip()
    aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
    role = str(item.get("role") or "unknown").strip().lower()
    semantic_type = str(item.get("type") or item.get("semantic_type") or "").strip().lower()
    return {
        "name": name,
        "aliases": aliases,
        "label": str(item.get("label") or name).strip() or name,
        "role": role if role in VALID_ROLES else "unknown",
        "type": semantic_type if semantic_type in VALID_TYPES else "",
        "can_chart": bool(item.get("can_chart", True)),
        "avoid_as_metric": bool(item.get("avoid_as_metric", False)),
        "description": str(item.get("description") or "").strip(),
        "recommended_use": str(item.get("recommended_use") or "").strip(),
    }
