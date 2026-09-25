"""OpenRouter client: schema-enforced JSON signals, model fallback, and an in-memory answer cache."""
import hashlib
import json
import logging
import os
import threading
import time
from typing import Callable, Dict, Literal, Optional, Tuple, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Sent when a model's first answer could not be parsed or validated: re-ask strictly, once, before giving up.
RETRY_HINT = ("Your previous response was not a valid JSON object for the required schema. Respond with ONLY a single "
              "raw JSON object matching the schema, with no prose, no markdown fences, and no trailing text.")


class Signal(BaseModel):
    """One agent's opinion: BUY/SELL/HOLD, confidence 0-1 and one-sentence reasoning; `degraded` marks a fail-safe HOLD.

    `details` holds the richer reasoning (chain-of-thought steps, risks, edge confidence) some agents return. It is
    populated by the agent code, never read from the model's JSON, and is persisted with the decision for audit/replay."""
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    degraded: bool = False  # set only by our fail-safe HOLD, never by a model
    details: Optional[dict] = None  # optional richer reasoning, for audit and replay; never from the model
    raw_model: Optional[str] = None  # which configured model actually answered, stamped client-side by the LLM client


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

_T = TypeVar("_T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Raised when every configured model failed."""
    pass


class LLMClient:
    """Asks OpenRouter models, in order, for a schema-enforced JSON object; falls through on any failure.

    `signal()` is the classic one-sentence Signal; `structured_call()` is the same machinery for any pydantic model,
    which is what the chain-of-thought agents use."""

    def __init__(self, models, client=None, max_tokens: int = 2500, cache_ttl_seconds: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, max_cache_entries: int = 4096):
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self.max_tokens = max_tokens
        self.dead: set[str] = set()  # models that no longer exist; skipped for the session so we stop paying for them
        self.last_model: Optional[str] = None  # which configured model answered the most recent live call
        # Identical (schema, prompt) -> reuse the answer. Analyses use completed daily bars, so a stock's prompt is
        # constant all session and this turns ~every re-analysis into a free cache hit. Failures are never cached.
        self.cache_ttl, self.clock, self.max_cache_entries = cache_ttl_seconds, clock, max_cache_entries
        self._cache: Dict[str, Tuple[float, BaseModel]] = {}
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

    def _remember(self, key: str, value: BaseModel) -> None:
        if self.cache_ttl <= 0:
            return
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (self.clock(), value)

    def signal(self, prompt: str) -> Signal:
        """Ask the models in order for a Signal; identical prompts are answered from cache."""
        return self.structured_call(prompt, Signal, SIGNAL_SCHEMA,
                                    parse=lambda raw: Signal.model_validate(
                                        {k: raw[k] for k in ("action", "confidence", "reasoning") if k in raw}))

    def structured_call(self, prompt: str, model_class: Type[_T], schema: dict,
                        parse: Optional[Callable[[dict], _T]] = None,
                        max_tokens: Optional[int] = None) -> _T:
        """Ask the models in order for a JSON object matching `schema`, validated into model_class.

        Identical (schema, prompt) pairs are served from cache. `parse` overrides how the raw JSON dict becomes an
        instance (used by `signal` to drop model-controlled fields such as `degraded`)."""
        key = hashlib.sha256(f"{schema['name']}\x00{prompt}".encode()).hexdigest()
        cached = self._cached(key)
        if cached is not None:
            self.last_model = getattr(cached, "raw_model", None)
            return cached
        instance = self._ask(prompt, schema, parse or model_class.model_validate, max_tokens)
        self._remember(key, instance)
        return instance

    def _ask(self, prompt: str, schema: dict, parse: Callable[[dict], BaseModel],
             max_tokens: Optional[int]) -> BaseModel:
        """One model at a time; a model is retried once when its answer is unparseable or fails the schema (that is the
        "retry with fallback" behaviour), skipped for the session when it no longer exists, and moved past on transport
        errors. Every model failing raises LLMUnavailable."""
        errors = []
        for model in self.models:
            if model in self.dead:
                continue
            messages = [{"role": "user", "content": prompt}]
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens or self.max_tokens,
                        messages=messages,
                        response_format={"type": "json_schema", "json_schema": schema},
                    )
                    if not response.choices:
                        raise ValueError(f"no choices in response (error={getattr(response, 'error', None)})")
                    text = response.choices[0].message.content
                    if not text:
                        raise ValueError("empty content")
                    raw = json.loads(text)
                    if not isinstance(raw, dict):
                        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                    instance = parse(raw)
                    if hasattr(instance, "raw_model"):
                        instance.raw_model = model  # tag the answer with who produced it, for per-model accountability
                    self.last_model = model
                    return instance
                except openai.NotFoundError as e:
                    log.warning("model %s no longer exists, skipping for this session: %s", model, e)
                    self.dead.add(model)
                    errors.append(f"{model}: NotFound")
                    break
                except openai.APIError as e:
                    log.warning("model %s failed: %s: %s", model, type(e).__name__, e)
                    errors.append(f"{model}: {type(e).__name__}")
                    break
                except (ValueError, ValidationError) as e:
                    if attempt == 0:
                        log.warning("model %s returned unusable output; retrying once: %s", model, e)
                        messages.append({"role": "user", "content": RETRY_HINT})
                        continue
                    log.warning("model %s failed twice on unusable output: %s", model, e)
                    errors.append(f"{model}: {type(e).__name__}")
        raise LLMUnavailable("; ".join(errors))
