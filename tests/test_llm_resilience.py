"""The LLM client must survive what the free models really do: overloaded upstreams, answers eaten by hidden reasoning,
prose-wrapped or malformed JSON, dead or flaky models, and the free tier's request limits."""
import json
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.config import Settings, load_settings
from src.llm import RETRY_HINT, LLMClient, LLMUnavailable, extract_json_object

GOOD = json.dumps({"action": "BUY", "confidence": 0.8, "reasoning": "uptrend"})


def answer(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)])


def overloaded():
    """What OpenRouter really sends when the upstream is busy: HTTP 200, no choices, the reason in `error`."""
    return SimpleNamespace(choices=None, error={"message": "Upstream error from Nvidia: Service temporarily overloaded"})


class Scripted:
    """Per-model queue of responses (a completion or an exception); the last one repeats. Records every call."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        queue = self.script[kw["model"]]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def models_called(self):
        return [c["model"] for c in self.calls]


def make(script, **kw):
    sleeps = []
    client = Scripted(script)
    llm = LLMClient(list(script), client=client, sleep=sleeps.append, jitter=lambda: 0.5, **kw)
    return llm, client, sleeps


# ------------------------------------------------------------------ tolerant JSON
@pytest.mark.parametrize("text", [
    GOOD,
    f"```json\n{GOOD}\n```",
    f"Sure! Here is the JSON:\n{GOOD}\nHope that helps.",
    "{\n" + GOOD,  # a doubled opening brace, seen from nemotron in json_object mode
    GOOD[:-1] + ",}",  # trailing comma
    "Use {curly braces} like so:\n```json\n" + GOOD + "\n```",  # stray braces in the prose: only the fence isolates the JSON
])
def test_json_is_extracted_from_the_shapes_free_models_produce(text):
    assert extract_json_object(text)["action"] == "BUY"


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "42", '"BUY"', "null", '{"action": "BUY", "confid'])
def test_non_objects_and_truncated_json_are_rejected(text):
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_a_fenced_answer_is_used_without_burning_a_retry():
    llm, client, _ = make({"a": [answer(f"```json\n{GOOD}\n```")]})
    assert llm.signal("p").action == "BUY" and len(client.calls) == 1


# ------------------------------------------------------------------ transient: overloaded / rate limited / empty
def test_overloaded_upstream_is_retried_with_exponential_backoff_on_the_same_model():
    llm, client, sleeps = make({"a": [overloaded(), overloaded(), answer(GOOD)]}, backoff_base=4.0)
    assert llm.signal("p").action == "BUY"
    assert client.models_called() == ["a", "a", "a"] and sleeps == [4.0, 8.0]  # 4 * 2^(n-1), jitter factor 1.0


def test_retry_after_from_a_rate_limit_is_honoured():
    limited = openai.RateLimitError("slow down", response=httpx.Response(
        429, headers={"retry-after": "30"}, request=httpx.Request("POST", "http://x")), body=None)
    llm, _, sleeps = make({"a": [limited, answer(GOOD)]}, backoff_base=1.0)
    llm.signal("p")
    assert sleeps == [30.0]  # longer than our own 1s backoff


def test_transient_retries_are_bounded_then_the_next_model_answers():
    llm, client, _ = make({"a": [overloaded()], "b": [answer(GOOD)]})
    assert llm.signal("p").action == "BUY"
    assert client.models_called() == ["a", "a", "a", "b"]  # first try + two retries, then fall through


# ------------------------------------------------------------------ truncated: the reasoning ate the budget
def test_an_empty_answer_that_hit_the_token_limit_is_retried_with_a_bigger_budget():
    llm, client, sleeps = make({"a": [answer("", finish="length"), answer(GOOD)]}, max_tokens=2500)
    assert llm.signal("p").action == "BUY"
    assert [c["max_tokens"] for c in client.calls] == [2500, 5000] and sleeps == []  # no waiting: it is not overload


def test_the_token_budget_doubles_but_never_beyond_the_cap():
    llm, client, _ = make({"a": [answer("", finish="length")]}, max_tokens=3000, max_tokens_cap=8000)
    with pytest.raises(LLMUnavailable):
        llm.signal("p")
    assert [c["max_tokens"] for c in client.calls] == [3000, 6000, 8000]


def test_json_cut_off_at_the_limit_is_a_truncation_not_a_format_error():
    cut = GOOD[:30]
    llm, client, _ = make({"a": [answer(cut, finish="length"), answer(GOOD)]})
    assert llm.signal("p").action == "BUY" and client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]


# ------------------------------------------------------------------ format: the model gets told what was wrong
def test_a_format_error_is_retried_once_quoting_the_problem_back():
    llm, client, _ = make({"a": [answer('{"action": "MAYBE", "confidence": 0.5, "reasoning": "r"}'), answer(GOOD)]})
    assert llm.signal("p").action == "BUY"
    hint = client.calls[1]["messages"][-1]["content"]
    assert hint.startswith(RETRY_HINT) and "Problem:" in hint and len(client.calls) == 2


def test_a_model_that_keeps_answering_badly_costs_two_calls_then_the_next_model_is_used():
    llm, client, _ = make({"a": [answer("garbage")], "b": [answer(GOOD)]})
    assert llm.signal("p").action == "BUY" and client.models_called() == ["a", "a", "b"]


# ------------------------------------------------------------------ health: a flaky model is benched, then given another chance
def test_a_model_that_fails_repeatedly_is_benched_and_the_healthy_one_answers_first():
    now = [0.0]
    llm, client, _ = make({"a": [answer("garbage")], "b": [answer(GOOD)]}, breaker_threshold=2, breaker_cooldown=300)
    llm.clock = lambda: now[0]
    llm.signal("p1")
    llm.signal("p2")  # a has now failed two whole calls in a row: benched
    client.calls.clear()
    llm.signal("p3")
    assert client.models_called() == ["b"]  # a is not even tried while benched
    now[0] = 301.0
    client.calls.clear()
    llm.signal("p4")
    assert client.models_called()[0] == "a"  # cool-off over: a gets another chance


def test_a_rejected_request_does_not_count_against_the_models_health():
    bad = openai.BadRequestError("context too long", response=httpx.Response(
        400, request=httpx.Request("POST", "http://x")), body=None)
    llm, _, _ = make({"a": [bad], "b": [answer(GOOD)]}, breaker_threshold=1)
    llm.signal("p")
    assert not llm._benched_until  # with threshold 1 a counted failure would have benched a


def test_a_success_clears_the_failure_streak():
    llm, _, _ = make({"a": [answer("garbage"), answer("garbage"), answer(GOOD)], "b": [answer(GOOD)]}, breaker_threshold=2)
    llm.signal("p1")  # a fails its call (2 attempts), b answers: streak 1
    llm.signal("p2")  # a answers on its first attempt of this call: streak reset
    assert not llm._benched_until


def test_when_every_model_is_benched_one_probe_is_still_made_per_call():
    llm, client, _ = make({"a": [answer("garbage")], "b": [answer("garbage")]}, breaker_threshold=1, breaker_cooldown=1e9)
    with pytest.raises(LLMUnavailable):
        llm.signal("p1")  # both models tried and benched
    client.calls.clear()
    with pytest.raises(LLMUnavailable):
        llm.signal("p2")
    assert len(set(client.models_called())) == 1 and len(client.calls) == 2  # a single probe, not a hammering of all
    client.script["a"] = [answer(GOOD)]
    client.script["b"] = [answer(GOOD)]
    assert llm.signal("p3").action == "BUY"  # the probed model recovered and is used again


# ------------------------------------------------------------------ reasoning switch
def test_reasoning_is_switched_off_only_when_configured():
    on, client_on, _ = make({"a": [answer(GOOD)]}, reasoning_off=True)
    on.signal("p")
    assert client_on.calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    off, client_off, _ = make({"a": [answer(GOOD)]})
    off.signal("p")
    assert "extra_body" not in client_off.calls[0]


def test_a_model_that_rejects_the_reasoning_switch_is_asked_without_it_from_then_on():
    bad = openai.BadRequestError("Unsupported parameter: reasoning", response=httpx.Response(
        400, request=httpx.Request("POST", "http://x")), body=None)
    llm, client, _ = make({"a": [bad, answer(GOOD), answer(GOOD)]}, reasoning_off=True)
    llm.signal("p1")
    llm.signal("p2")
    assert "extra_body" in client.calls[0] and "extra_body" not in client.calls[1] and "extra_body" not in client.calls[2]
    assert "a" not in llm.dead


def test_an_unrelated_bad_request_skips_the_model_without_retrying():
    bad = openai.BadRequestError("context too long", response=httpx.Response(
        400, request=httpx.Request("POST", "http://x")), body=None)
    llm, client, _ = make({"a": [bad], "b": [answer(GOOD)]})
    assert llm.signal("p").action == "BUY" and client.models_called() == ["a", "b"]


# ------------------------------------------------------------------ pacing and concurrency
def test_requests_are_spaced_by_the_minimum_interval():
    now = [100.0]
    llm, client, sleeps = make({"a": [answer(GOOD)]}, min_interval=3.0)
    llm.clock = lambda: now[0]
    llm._sleep = lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s))
    for prompt in ("p1", "p2", "p3"):
        llm.signal(prompt)
    assert sleeps == [3.0, 3.0] and len(client.calls) == 3  # the first goes at once, then one every 3 seconds


def test_no_more_requests_are_in_flight_than_the_concurrency_limit():
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow_create(**kw):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return answer(GOOD)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=slow_create)))
    llm = LLMClient(["a"], client=client, max_concurrency=2)
    threads = [threading.Thread(target=llm.signal, args=(f"p{i}",)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 2


def test_the_call_time_budget_stops_a_slow_cascade():
    now = [0.0]
    llm, client, _ = make({"a": [overloaded()], "b": [overloaded()], "c": [answer(GOOD)]}, call_budget_seconds=10)
    llm.clock = lambda: now[0]
    llm._sleep = lambda s: now.__setitem__(0, now[0] + 20)  # each wait blows the budget
    with pytest.raises(LLMUnavailable, match="budget"):
        llm.signal("p")
    assert "c" not in client.models_called()


# ------------------------------------------------------------------ observability and wiring
def test_health_line_summarises_what_happened_and_then_resets():
    llm, _, _ = make({"a": [overloaded(), answer(GOOD)]})
    llm.signal("p")
    line = llm.health_line()
    assert "a ok 1" in line and "transient 1" in line and "avg" in line
    assert llm.health_line() is None  # drained


def test_from_settings_wires_the_production_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings = load_settings()
    llm = LLMClient.from_settings(settings)
    assert llm.reasoning_off is True and llm.min_interval == 3.0 and llm._slots is not None
    assert llm.backoff_base > 0 and llm.cache_ttl == settings.llm_cache_hours * 3600
    with pytest.raises(ValueError):
        Settings(llm_min_interval=-1)


# ================================================================== agent level: validation, the fallback ladder, audit tags
from src.agents.agents import TechnicalAgent  # noqa: E402
from src.agents.base import AgentContext  # noqa: E402
from src.data.indicators import Snapshot  # noqa: E402

UP = Snapshot("TESTCO", 250.0, 240.0, 220.0, 58.0, 1_000_000)  # the mechanical filter says BUY
DOWN = Snapshot("TESTCO", 200.0, 240.0, 220.0, 45.0, 1_000_000)  # price below MA50: the filter says HOLD


def cot(**over):
    body = {"action": "BUY", "confidence": 0.8, "edge_confidence": 0.5,
            "step1_trend": "Price is above both averages", "step2_overbought": "RSI 58 leaves room",
            "step3_volume": "Volume is above its average", "step4_confluence": 7,
            "step5_risks": ["earnings gap", "index sell-off"], "final_reasoning": "Clean uptrend with room to run",
            "rule_alignment": "agree", "falsification": "A close below 240"}
    body.update(over)
    return json.dumps(body)


def agent_for(script, snap=UP, **flags):
    client = Scripted(script)
    llm = LLMClient(list(script), client=client, sleep=lambda s: None)
    agent = TechnicalAgent(llm, use_learning=False, use_context=False, use_reflect=False, **flags)
    return agent, client, snap


def run(agent, snap):
    return agent.analyze(AgentContext(snap.symbol, snap))


@pytest.mark.parametrize("bad", [
    dict(final_reasoning=""), dict(final_reasoning="n/a"), dict(step1_trend="  "), dict(step3_volume="-"),
    dict(step5_risks=[]), dict(step5_risks=["", "  "]),
])
def test_blank_or_placeholder_reasoning_is_retried_not_traded_on(bad):
    agent, client, snap = agent_for({"a": [answer(cot(**bad)), answer(cot())]})
    signal = run(agent, snap)
    assert signal.action == "BUY" and not signal.degraded and len(client.calls) == 2
    assert "Problem:" in client.calls[1]["messages"][-1]["content"]  # the model is told what was wrong


def test_a_blank_falsification_is_dropped_not_fatal():
    agent, client, snap = agent_for({"a": [answer(cot(falsification="  "))]})
    signal = run(agent, snap)
    assert signal.action == "BUY" and "falsification" not in signal.details and len(client.calls) == 1


def test_rule_alignment_is_computed_not_taken_from_the_model():
    liar = cot(action="HOLD", rule_alignment="agree")  # says it agrees with a BUY filter while answering HOLD
    up_agent, _, up = agent_for({"a": [answer(liar)]}, snap=UP)
    assert run(up_agent, up).details["rule_alignment"] == "deviate"  # filter says BUY, the model said HOLD
    dn_agent, _, dn = agent_for({"a": [answer(cot(action="HOLD", rule_alignment="deviate"))]}, snap=DOWN)
    assert run(dn_agent, dn).details["rule_alignment"] == "agree"  # filter says HOLD, the model said HOLD


def test_a_normal_answer_is_tagged_as_the_full_chain_of_thought_tier():
    agent, _, snap = agent_for({"a": [answer(cot())]})
    assert run(agent, snap).details["tier"] == "cot"


def test_if_the_full_prompt_fails_everywhere_the_compact_prompt_is_tried():
    compact = json.dumps({"action": "BUY", "confidence": 0.7, "reasoning": "trend intact"})

    def route(**kw):
        return answer("garbage") if "five fixed steps" in kw["messages"][0]["content"] else answer(compact)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=route)))
    agent = TechnicalAgent(LLMClient(["a"], client=client), use_learning=False, use_context=False, use_reflect=False)
    signal = run(agent, UP)
    assert signal.action == "BUY" and not signal.degraded
    assert signal.details["tier"] == "compact" and "cot_failure" in signal.details and signal.details["raw_model"] == "a"


def test_only_when_both_prompts_fail_does_the_agent_stand_down():
    agent, _, snap = agent_for({"a": [overloaded()]})
    signal = run(agent, snap)
    assert signal.action == "HOLD" and signal.degraded is True


def test_a_skipped_refinement_leaves_a_mark_in_the_audit_record():
    from tests.test_learning_context_reflect import make_stats

    reflection_fails = [answer(cot()), overloaded()]
    client = Scripted({"a": reflection_fails})
    agent = TechnicalAgent(LLMClient(["a"], client=client, sleep=lambda s: None), use_learning=False, use_context=False,
                           use_reflect=True)
    signal = run(agent, UP)
    assert signal.action == "BUY" and signal.details["reflection_skipped"] is True and "reflection" not in signal.details

    learn_fails = Scripted({"a": [answer(cot()), overloaded()]})
    agent = TechnicalAgent(LLMClient(["a"], client=learn_fails, sleep=lambda s: None), use_learning=True,
                           use_context=False, use_reflect=False, history_fn=lambda s: make_stats(12, 0.3))
    assert run(agent, UP).details["learning_skipped"] is True


def test_the_pipeline_logs_one_health_line_per_cycle(tmp_path, caplog):
    import logging
    from tests.test_llm_and_pipeline import FakeBroker, make_pipeline
    from src.engine.risk_engine import Portfolio

    llm, _, _ = make({"a": [overloaded(), answer(GOOD)]})
    llm.signal("p")
    pipe, _, _ = make_pipeline(tmp_path, broker=FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)))
    pipe.agents["technical"].llm = llm
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        pipe.run_once()
    assert any("LLM health:" in r.message and "transient 1" in r.message for r in caplog.records)
