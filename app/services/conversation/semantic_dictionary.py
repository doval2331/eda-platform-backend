from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_DICTIONARY_PATH = Path(__file__).with_name("semantic_dictionary.json")
VALID_ROLES = {"business", "metric", "technical", "identifier", "unknown"}
VALID_TYPES = {"categorical", "numeric", "boolean", "date", "text", ""}


def semantic_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def get_semantic_dictionary(base_dictionary: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dictionary: dict[str, dict[str, Any]] = {}
    for key, value in base_dictionary.items():
        _register_entry(dictionary, key, value)
    for key, value in _configured_semantic_entries().items():
        _register_entry(dictionary, key, value)
    return dictionary


def reload_semantic_dictionary() -> None:
    _configured_semantic_entries.cache_clear()


def semantic_dictionary_path() -> Path:
    return Path(os.getenv("CONVERSATION_SEMANTIC_DICTIONARY_PATH") or DEFAULT_DICTIONARY_PATH)


def load_configured_semantic_variables() -> list[dict[str, Any]]:
    path = semantic_dictionary_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("variables") if isinstance(payload, dict) else None
    return [item for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []


def save_configured_semantic_variables(entries: list[dict[str, Any]]) -> dict[str, Any]:
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

    path = semantic_dictionary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"variables": deduped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reload_semantic_dictionary()
    return {"path": str(path), "total": len(deduped), "variables": deduped}


@lru_cache(maxsize=1)
def _configured_semantic_entries() -> dict[str, dict[str, Any]]:
    entries = load_configured_semantic_variables()
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
