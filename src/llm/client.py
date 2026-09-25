"""OpenRouter client built for unreliable free models: schema-enforced JSON, tolerant parsing, failure-aware retries,
per-model health with a circuit breaker, request pacing, and an in-memory answer cache.

What the free models actually do wrong (measured on ~500 production calls and direct probes, not guessed):
  * they are REASONING models: hidden "thinking" tokens are billed against `max_tokens`, so an answer is often empty
    (finish_reason=length) or cut off mid-JSON. Turning reasoning off cut answers from 1,600-3,200 to 400-650 tokens and
    30-276 s to 12-85 s, all valid. The prompt already asks for step-by-step reasoning inside the JSON, so nothing is lost.
  * OpenRouter answers HTTP 200 even when the upstream is overloaded (the error is in the body, `choices` is empty);
  * some models wrap JSON in prose or code fences, or repeat the opening brace.

Each failure is classified, because each needs a different response:
  TRANSIENT  overloaded / rate limited / timeout / empty answer   -> wait (exponential backoff + jitter), retry same model
  TRUNCATED  the answer hit max_tokens                            -> retry with a larger budget (doubling, capped)
  FORMAT     unparseable or schema-invalid JSON                   -> retry once, quoting the error back to the model
  PERMANENT  model gone / auth / bad request                      -> skip the model (gone models are skipped for the session)
A model that fails several calls in a row is benched for a cool-off so healthy models answer first; if every model is
benched the least recently benched is still tried, so the client can never wedge itself."""
import hashlib
import logging
import os
import random
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, cast

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.engine.enums import ACTION_VALUES
from src.engine.agent_signal import AgentSignal
from src.llm.errors import FORMAT, PERMANENT, TRANSIENT, TRUNCATED, Failure, LLMUnavailable
from src.llm.parsing import extract_json_object, retry_after, upstream_message

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Appended when a model's answer could not be parsed or validated: re-ask strictly before giving up.
RETRY_HINT = ("Your previous response was not a valid JSON object for the required schema. Respond with ONLY a single "
              "raw JSON object matching the schema, with no prose, no markdown fences, and no trailing text.")

SIGNAL_SCHEMA = {
    "name": "trading_signal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTION_VALUES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}

_T = TypeVar("_T", bound=BaseModel)


class LLMClient:
    """Asks OpenRouter models for a schema-enforced JSON object, surviving flaky free models (see the module docstring).

    `signal()` is the classic one-sentence AgentSignal; `structured_call()` is the same machinery for any pydantic model,
    which is what the chain-of-thought agents use. The defaults do no waiting and no pacing so unit tests stay fast;
    `from_settings` wires the production values."""

    def __init__(self, models, client=None, max_tokens: int = 3000, cache_ttl_seconds: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, max_cache_entries: int = 4096,
                 reasoning_off: bool = False, max_tokens_cap: int = 8000, timeout: float = 120.0,
                 transient_retries: int = 2, format_retries: int = 1, truncation_retries: int = 2,
                 backoff_base: float = 0.0, backoff_cap: float = 60.0, call_budget_seconds: float = 180.0,
                 breaker_threshold: int = 3, breaker_cooldown: float = 300.0,
                 min_interval: float = 0.0, max_concurrency: int = 0,
                 sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random):
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self.max_tokens, self.max_tokens_cap, self.reasoning_off = max_tokens, max_tokens_cap, reasoning_off
        self.retries = {TRANSIENT: transient_retries, FORMAT: format_retries, TRUNCATED: truncation_retries}
        self.backoff_base, self.backoff_cap, self.call_budget = backoff_base, backoff_cap, call_budget_seconds
        self.breaker_threshold, self.breaker_cooldown = breaker_threshold, breaker_cooldown
        self.min_interval, self._sleep, self._jitter = min_interval, sleep, jitter
        self.dead: set[str] = set()  # models that no longer exist; skipped for the session so we stop paying for them
        self.last_model: Optional[str] = None  # which configured model answered the most recent live call
        self._no_reasoning_control: set[str] = set()  # models that reject the "reasoning off" switch
        # Identical (schema, prompt) -> reuse the answer. Analyses use completed daily bars, so a stock's prompt is
        # constant all session and this turns ~every re-analysis into a free cache hit. Failures are never cached.
        self.cache_ttl, self.clock, self.max_cache_entries = cache_ttl_seconds, clock, max_cache_entries
        self._cache: Dict[str, Tuple[float, BaseModel]] = {}
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._benched_until: Dict[str, float] = {}
        self._consecutive_failures: Counter = Counter()
        self._window: Dict[str, Counter[str]] = {}  # per-model counts since the last drain_stats()
        self._seconds: Dict[str, float] = {}  # ... and the seconds the successful calls took
        self._slots = threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        self._rate_lock, self._next_slot = threading.Lock(), 0.0
        # The SDK's own retry is off: it would hide rate limits from, and multiply, our own accounted retries.
        self.client = client or OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"],
                                       timeout=timeout, max_retries=0)

    @classmethod
    def from_settings(cls, settings, **overrides):
        """The production client: reasoning off, paced to the free tier's request limit, bounded concurrency, backoff."""
        params = dict(cache_ttl_seconds=settings.llm_cache_hours * 3600, reasoning_off=settings.llm_reasoning_off,
                      min_interval=settings.llm_min_interval, max_concurrency=settings.llm_concurrency,
                      backoff_base=4.0)
        params.update(overrides)
        return cls(settings.models, **params)

    # ------------------------------------------------------------------ cache
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

    # ------------------------------------------------------------------ public calls
    def signal(self, prompt: str) -> AgentSignal:
        """Ask the models in order for a AgentSignal; identical prompts are answered from cache."""
        return self.structured_call(prompt, AgentSignal, SIGNAL_SCHEMA,
                                    parse=lambda raw: AgentSignal.model_validate(
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
        instance = cast(_T, self._ask(prompt, schema, parse or model_class.model_validate, max_tokens))  # `parse` builds a model_class
        self._remember(key, instance)
        return instance

    # ------------------------------------------------------------------ model selection and health
    def _order(self) -> list:
        """Healthy models in configured order. When every model is benched, only the one soonest to recover is tried (a
        single probe: hammering a provider that is down just burns the time budget)."""
        now = self.clock()
        live = [m for m in self.models if m not in self.dead]
        with self._state_lock:
            ready = [m for m in live if self._benched_until.get(m, 0.0) <= now]
            benched = sorted((m for m in live if m not in ready), key=lambda m: self._benched_until[m])
        return ready if ready else benched[:1]

    def _bump(self, model: str, key: str, n: int = 1) -> None:
        with self._state_lock:
            self._window.setdefault(model, Counter())[key] += n

    def _succeeded(self, model: str, seconds: float) -> None:
        with self._state_lock:
            self._consecutive_failures[model] = 0
            self._benched_until.pop(model, None)
            self._window.setdefault(model, Counter())["ok"] += 1
            self._seconds[model] = self._seconds.get(model, 0.0) + seconds

    def _gave_up_on(self, model: str) -> None:
        """A whole call to this model failed: after enough in a row, bench it so others answer first."""
        with self._state_lock:
            self._consecutive_failures[model] += 1
            if self.breaker_threshold > 0 and self._consecutive_failures[model] >= self.breaker_threshold:
                self._benched_until[model] = self.clock() + self.breaker_cooldown
                self._window.setdefault(model, Counter())["benched"] += 1
                log.warning("model %s failed %d calls in a row: benched for %.0fs", model,
                            self._consecutive_failures[model], self.breaker_cooldown)

    def drain_stats(self) -> Dict[str, dict]:
        """Per-model counters since the last drain (ok, failures by kind, benched), for the cycle's health line."""
        with self._state_lock:
            out: Dict[str, dict] = {m: {**c, **({"seconds": self._seconds[m]} if m in self._seconds else {})}
                                    for m, c in self._window.items() if c}
            self._window, self._seconds = {}, {}
        return out

    def health_line(self) -> Optional[str]:
        """One human line about the LLM since the last call to this method, or None when nothing happened."""
        stats = self.drain_stats()
        if not stats:
            return None
        parts = []
        for model, c in stats.items():
            bad = ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k not in ("ok", "benched", "seconds"))
            ok = c.get("ok", 0)
            speed = f" (avg {c['seconds'] / ok:.0f}s)" if ok else ""
            parts.append(f"{model.split('/')[-1]} ok {ok}{speed}" + (f" | {bad}" if bad else "")
                         + (" | BENCHED" if c.get("benched") else ""))
        return "LLM health: " + "; ".join(parts)

    # ------------------------------------------------------------------ one call
    def _ask(self, prompt: str, schema: dict, parse: Callable[[dict], BaseModel],
             max_tokens: Optional[int]) -> BaseModel:
        """Try the models in health order until one answers; every model failing raises LLMUnavailable."""
        deadline = self.clock() + self.call_budget
        errors = []
        for model in self._order():
            if self.clock() >= deadline:
                errors.append("call time budget exhausted")
                break
            try:
                return self._try_model(model, prompt, schema, parse, max_tokens or self.max_tokens, deadline)
            except Failure as f:
                errors.append(f"{model}: {f.kind}: {f.detail}")
        raise LLMUnavailable("; ".join(errors))

    def _try_model(self, model, prompt, schema, parse, tokens, deadline) -> BaseModel:
        """Up to a bounded number of attempts on one model, reacting to each failure kind as the module docstring says."""
        base = [{"role": "user", "content": prompt}]
        messages, spent = base, Counter[str]()
        while True:
            started = self.clock()
            try:
                instance = self._request(model, messages, tokens, schema, parse)
                self._succeeded(model, self.clock() - started)
                self.last_model = model
                return instance
            except Failure as f:
                self._bump(model, f.kind)
                log.warning("model %s %s (attempt %d): %s", model, f.kind, sum(spent.values()) + 1, f.detail)
                spent[f.kind] += 1
                if f.kind == PERMANENT or spent[f.kind] > self.retries.get(f.kind, 0) or self.clock() >= deadline:
                    if f.kind != PERMANENT:  # a rejected request (too long, bad key) says nothing about the model's health
                        self._gave_up_on(model)
                    raise
                if f.kind == TRANSIENT:
                    self._sleep(self._backoff(spent[f.kind], f.retry_after))
                elif f.kind == TRUNCATED:
                    tokens = min(tokens * 2, self.max_tokens_cap)
                elif f.kind == FORMAT:
                    messages = base + [{"role": "user", "content": f"{RETRY_HINT} (Problem: {f.detail})"}]

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        """Exponential backoff with jitter, never shorter than the provider's Retry-After."""
        wait = min(self.backoff_cap, self.backoff_base * (2 ** (attempt - 1))) * (0.75 + 0.5 * self._jitter())
        return max(wait, retry_after or 0.0)

    @contextmanager
    def _paced(self):
        """Bound how many requests are in flight and space request starts by `min_interval` (free-tier rate limit)."""
        if self._slots is not None:
            self._slots.acquire()
        try:
            if self.min_interval > 0:
                with self._rate_lock:
                    now = self.clock()
                    wait = max(0.0, self._next_slot - now)
                    self._next_slot = max(now, self._next_slot) + self.min_interval
                if wait:
                    self._sleep(wait)
            yield
        finally:
            if self._slots is not None:
                self._slots.release()

    def _create(self, model: str, messages: list, tokens: int, schema: dict):
        """The API call, with the reasoning switch that is dropped for any model that rejects it."""
        kwargs: Dict[str, Any] = dict(model=model, max_tokens=tokens, messages=messages,
                                      response_format={"type": "json_schema", "json_schema": schema})
        if self.reasoning_off and model not in self._no_reasoning_control:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        try:
            with self._paced():
                return self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "extra_body" in kwargs and "reasoning" in str(e).lower():
                log.warning("model %s cannot switch reasoning off; asking without it from now on", model)
                self._no_reasoning_control.add(model)
                kwargs.pop("extra_body")
                with self._paced():
                    return self.client.chat.completions.create(**kwargs)
            raise

    def _request(self, model: str, messages: list, tokens: int, schema: dict, parse) -> BaseModel:
        """One API call, classified: returns the validated instance or raises Failure."""
        try:
            response = self._create(model, messages, tokens, schema)
        except openai.NotFoundError as e:
            self.dead.add(model)
            raise Failure(PERMANENT, f"model no longer exists ({str(e)[:60]})")
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError) as e:
            raise Failure(PERMANENT, f"{type(e).__name__}: {str(e)[:100]}")
        except openai.RateLimitError as e:
            raise Failure(TRANSIENT, "rate limited", retry_after=retry_after(e))
        except openai.APIError as e:  # timeouts, connection errors, 5xx: all worth another go
            raise Failure(TRANSIENT, f"{type(e).__name__}: {str(e)[:80]}")
        choices = getattr(response, "choices", None)
        if not choices:
            raise Failure(TRANSIENT, f"no choices in response ({upstream_message(response)})")
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        text = getattr(choice.message, "content", None)
        if not text or not text.strip():
            raise Failure(TRUNCATED if finish == "length" else TRANSIENT,
                           f"empty content (finish_reason={finish})")
        try:
            instance = parse(extract_json_object(text))
        except (ValueError, ValidationError) as e:
            kind = TRUNCATED if finish == "length" else FORMAT
            raise Failure(kind, f"{type(e).__name__}: {str(e)[:100]} (finish_reason={finish}, head={text[:60]!r})")
        if hasattr(instance, "raw_model"):
            instance.raw_model = model  # tag the answer with who produced it, for per-model accountability
        return instance
