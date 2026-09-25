"""Reading what comes back: the JSON object inside a model's answer, and the provider's own error details."""
import json
import re
from typing import Optional


def extract_json_object(text: str) -> dict:
    """The JSON object in a model's answer. Tolerates a markdown fence, prose around the object, a doubled opening brace
    and trailing commas; raises ValueError when there is no object (a list, a bare string or number is not one)."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    candidates = [s]
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last > first and (first, last) != (0, len(s) - 1):
        candidates.append(s[first:last + 1])
    for candidate in candidates:
        for variant in (candidate, re.sub(r"^\{\s*\{", "{", candidate), re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                obj = json.loads(variant)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
            raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    raise ValueError(f"no valid JSON object in the answer ({s[:60]!r})")


def upstream_message(response) -> str:
    """The provider's own explanation when a 200 response carries no choices (OpenRouter puts it in `error`)."""
    error = getattr(response, "error", None)
    if isinstance(error, dict):
        return str(error.get("message") or error)[:100]
    return str(error)[:100] if error else "no detail"


def retry_after(exc) -> Optional[float]:
    """Seconds the provider asked us to wait (Retry-After header), if any."""
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None
