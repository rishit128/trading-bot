"""OpenRouter client: schema-enforced JSON signals, model fallback, and an in-memory answer cache."""
import hashlib
import json
import logging
import os
import threading
import time
from typing import Callable, Dict, Literal, Tuple

import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class Signal(BaseModel):
    """One agent's opinion: BUY/SELL/HOLD, confidence 0-1 and one-sentence reasoning; `degraded` marks a fail-safe HOLD."""
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    degraded: bool = False  # set only by our fail-safe HOLD, never by a model


SIGNAL_SCHEMA = {
    "name": "trading_signal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}


class LLMUnavailable(Exception):
    """Raised when every configured model failed."""
    pass


class LLMClient:
    """Asks OpenRouter models, in order, for a schema-enforced Signal; falls through on any failure."""

    def __init__(self, models, client=None, max_tokens: int = 1500, cache_ttl_seconds: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, max_cache_entries: int = 4096):
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self.max_tokens = max_tokens
        self.dead = set()
        # Identical prompt -> reuse the answer. Analyses use completed daily bars, so a stock's prompt is constant all
        # session and this turns ~every re-analysis into a free cache hit. Failures are never cached.
        self.cache_ttl, self.clock, self.max_cache_entries = cache_ttl_seconds, clock, max_cache_entries
        self._cache: Dict[str, Tuple[float, "Signal"]] = {}
        self._cache_lock = threading.Lock()
        self.client = client or OpenAI(
            base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=60.0, max_retries=1
        )

    def _cached(self, key: str):
        if self.cache_ttl <= 0:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit and self.clock() - hit[0] < self.cache_ttl:
                return hit[1]
            self._cache.pop(key, None)
        return None

    def _remember(self, key: str, sig: "Signal") -> None:
        if self.cache_ttl <= 0:
            return
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (self.clock(), sig)

    def signal(self, prompt: str) -> Signal:
        """Ask the models in order for a Signal; identical prompts are answered from cache."""
        key = hashlib.sha256(prompt.encode()).hexdigest()
        cached = self._cached(key)
        if cached is not None:
            return cached
        sig = self._ask(prompt)
        self._remember(key, sig)
        return sig

    def _ask(self, prompt: str) -> Signal:
        errors = []
        for model in self.models:
            if model in self.dead:
                continue
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_schema", "json_schema": SIGNAL_SCHEMA},
                )
                if not response.choices:
                    raise ValueError(f"no choices in response (error={getattr(response, 'error', None)})")
                text = response.choices[0].message.content
                if not text:
                    raise ValueError("empty content")
                raw = json.loads(text)
                if not isinstance(raw, dict):
                    raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                return Signal.model_validate({k: raw[k] for k in ("action", "confidence", "reasoning") if k in raw})
            except openai.NotFoundError as e:
                log.warning("model %s no longer exists, skipping for this session: %s", model, e)
                self.dead.add(model)
                errors.append(f"{model}: NotFound")
            except (openai.APIError, ValueError, ValidationError) as e:
                log.warning("model %s failed: %s: %s", model, type(e).__name__, e)
                errors.append(f"{model}: {type(e).__name__}")
        raise LLMUnavailable("; ".join(errors))
