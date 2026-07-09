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
VALID_CONFIDENCE = {"alta", "media", "baja", ""}
VALID_PROFILES = {"funcional", "experto", "ambos"}


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
        _register_entry(dictionary, key, _normalize_base_entry(key, value))

    for item in load_configured_semantic_variables(project_id=_project_scope(project_id)):
        normalized = _normalize_configured_entry(item)
        if not normalized["name"]:
            continue
        if not normalized["active"]:
            _unregister_entry(dictionary, normalized["name"], normalized["aliases"])
            continue
        entry = _semantic_entry_from_normalized(normalized)
        _register_entry(dictionary, normalized["name"], entry)
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
    configured = [_normalize_configured_entry(item) for item in load_configured_semantic_variables(project_id=project_scope)]
    active_total = sum(1 for item in configured if item["active"])
    inactive_total = sum(1 for item in configured if not item["active"])
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
        "active_configured_total": active_total,
        "inactive_configured_total": inactive_total,
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
        "active_total": sum(1 for item in deduped if item["active"]),
        "inactive_total": sum(1 for item in deduped if not item["active"]),
        "variables": deduped,
    }


@lru_cache(maxsize=64)
def _configured_semantic_entries(project_scope: str = "") -> dict[str, dict[str, Any]]:
    entries = load_configured_semantic_variables(project_id=project_scope)
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        normalized = _normalize_configured_entry(item)
        name = normalized["name"]
        if not name or not normalized["active"]:
            continue
        entry = _semantic_entry_from_normalized(normalized)
        result[name] = entry
        for alias in normalized["aliases"]:
            result[alias] = entry
    return result


def _project_scope(project_id: str | None) -> str:
    text = str(project_id or "").strip()
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)[:120]


def _entry_keys(key: str, aliases: list[str]) -> list[str]:
    values = [key, *aliases, semantic_key(key), *(semantic_key(alias) for alias in aliases)]
    return [value for value in values if value]


def _register_entry(dictionary: dict[str, dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    entry = dict(value)
    if entry.get("active") is False:
        return
    aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
    entry.setdefault("aliases", aliases)
    entry.setdefault("active", True)
    entry.setdefault("source", "base")
    entry.setdefault("confidence", "media")
    for candidate in _entry_keys(key, aliases):
        dictionary[candidate] = entry


def _unregister_entry(dictionary: dict[str, dict[str, Any]], key: str, aliases: list[str]) -> None:
    for candidate in _entry_keys(key, aliases):
        dictionary.pop(candidate, None)


def _normalize_base_entry(name: str, item: dict[str, Any]) -> dict[str, Any]:
    entry = dict(item)
    entry.setdefault("label", name)
    entry.setdefault("role", "unknown")
    entry.setdefault("semantic_type", entry.get("type") or "")
    entry.setdefault("aliases", [])
    entry.setdefault("can_chart", True)
    entry.setdefault("avoid_as_metric", False)
    entry.setdefault("avoid_as_dimension", False)
    entry.setdefault("description", "")
    entry.setdefault("recommended_use", "")
    entry.setdefault("active", True)
    entry.setdefault("source", "base")
    entry.setdefault("confidence", "media")
    entry.setdefault("enabled_profiles", [])
    entry.setdefault("domain", "")
    entry.setdefault("owner", "")
    entry.setdefault("version", "")
    entry.setdefault("max_cardinality", None)
    entry.setdefault("max_null_ratio", None)
    return entry


def _semantic_entry_from_normalized(normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": normalized["label"],
        "role": normalized["role"],
        "description": normalized["description"],
        "recommended_use": normalized["recommended_use"],
        "avoid_as_metric": normalized["avoid_as_metric"],
        "avoid_as_dimension": normalized["avoid_as_dimension"],
        "can_chart": normalized["can_chart"],
        "semantic_type": normalized["type"],
        "aliases": normalized["aliases"],
        "source": normalized["source"],
        "confidence": normalized["confidence"],
        "active": normalized["active"],
        "enabled_profiles": normalized["enabled_profiles"],
        "domain": normalized["domain"],
        "owner": normalized["owner"],
        "version": normalized["version"],
        "max_cardinality": normalized["max_cardinality"],
        "max_null_ratio": normalized["max_null_ratio"],
    }


def _normalize_configured_entry(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "").strip()
    aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
    role = str(item.get("role") or "unknown").strip().lower()
    semantic_type = str(item.get("type") or item.get("semantic_type") or "").strip().lower()
    confidence = str(item.get("confidence") or "media").strip().lower()
    source = str(item.get("source") or item.get("configured_by") or "config").strip()
    profiles = [
        str(profile).strip().lower()
        for profile in item.get("enabled_profiles") or item.get("profiles") or []
        if str(profile).strip().lower() in VALID_PROFILES
    ]
    max_cardinality = _optional_int(item.get("max_cardinality"))
    max_null_ratio = _optional_float(item.get("max_null_ratio"))
    return {
        "name": name,
        "aliases": aliases,
        "label": str(item.get("label") or name).strip() or name,
        "role": role if role in VALID_ROLES else "unknown",
        "type": semantic_type if semantic_type in VALID_TYPES else "",
        "can_chart": bool(item.get("can_chart", True)),
        "avoid_as_metric": bool(item.get("avoid_as_metric", False)),
        "avoid_as_dimension": bool(item.get("avoid_as_dimension", False)),
        "description": str(item.get("description") or "").strip(),
        "recommended_use": str(item.get("recommended_use") or "").strip(),
        "source": source[:80] or "config",
        "confidence": confidence if confidence in VALID_CONFIDENCE else "media",
        "active": bool(item.get("active", True)),
        "enabled_profiles": profiles,
        "domain": str(item.get("domain") or "").strip()[:80],
        "owner": str(item.get("owner") or item.get("approved_by") or "").strip()[:80],
        "version": str(item.get("version") or "").strip()[:40],
        "max_cardinality": max_cardinality,
        "max_null_ratio": max_null_ratio,
    }


def _optional_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return min(number, 1.0)
