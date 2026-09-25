"""The OpenRouter client for unreliable free models and the types around it:

    client.py    LLMClient: schema-enforced JSON calls with classified retries, model health, pacing and a cache
    errors.py    LLMUnavailable (what callers catch) and the failure kinds the client reacts to
    parsing.py   tolerant extraction of the JSON object from an answer, and provider error details

The names callers need are re-exported here (`AgentSignal` too, for the many modules that import it from `src.llm`)."""
from src.engine.agent_signal import AgentSignal
from src.llm.client import RETRY_HINT, SIGNAL_SCHEMA, LLMClient
from src.llm.errors import FORMAT, PERMANENT, TRANSIENT, TRUNCATED, LLMUnavailable
from src.llm.parsing import extract_json_object

__all__ = ["LLMClient", "LLMUnavailable", "AgentSignal", "SIGNAL_SCHEMA", "RETRY_HINT", "extract_json_object",
           "TRANSIENT", "TRUNCATED", "FORMAT", "PERMANENT"]
