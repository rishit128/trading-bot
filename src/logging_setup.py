"""Logging that works for both people (text) and log pipelines (JSON), with a configurable level."""
import json
import logging
import sys
from datetime import datetime, timezone

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
FORMATS = ("text", "json")
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}
NOISY = ("httpx", "httpcore", "openai", "urllib3", "peewee", "websockets")  # per-request chatter, and httpx URLs hold the Telegram token


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, any `extra=` fields (symbol, action, agents, ...), and exc."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a single JSON line."""
        payload = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage(), **_extras(record)}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable line; structured fields are appended as key=value."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a text line with key=value extras appended."""
        line = super().format(record)
        extras = _extras(record)
        return f"{line} | " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else line


def configure_logging(level: str = "INFO", fmt: str = "text", stream=None) -> None:
    """Install one handler in text or JSON format at the given level and quiet chatty libraries."""
    level, fmt = level.strip().upper(), fmt.strip().lower()
    if level not in LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {LEVELS}, got {level!r}")
    if fmt not in FORMATS:
        raise ValueError(f"LOG_FORMAT must be one of {FORMATS}, got {fmt!r}")
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
