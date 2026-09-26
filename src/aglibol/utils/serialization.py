"""Safe JSON and serialization utilities."""

from __future__ import annotations

import json
from typing import Any


def safe_json_dumps(obj: Any, indent: int | None = None) -> str:
    """Serialize object to JSON handling common non-serializable objects gracefully."""

    def default_handler(o: Any) -> str:
        return str(o)

    return json.dumps(obj, indent=indent, default=default_handler, ensure_ascii=False)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON text, returning default fallback on parse error."""
    try:
        return json.loads(text)
    except Exception:
        return default
