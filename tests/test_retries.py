"""Transient network failures are retried once (or twice); definite answers about the data are not."""
import json

import httpx
import pandas as pd
import pytest
from langgraph.types import RetryPolicy

from src.data.indicators import StaleDataError
from src.data.india import IntradayFeed
from src.data.retry import is_transient_data_error, retry_call
from src.monitoring.telegram import Notifier
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph
from tests.test_cycle_services import FakeServices

FAST = RetryPolicy(max_attempts=2, initial_interval=0.0, jitter=False, retry_on=is_transient_data_error)


# ---------------------------------------------------------------- the helper
def flaky(failures, error=ConnectionError("blip")):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return "ok"

    return fn, calls


def test_a_transient_failure_is_retried_with_growing_waits_and_then_succeeds():
    fn, calls = flaky(2)
    sleeps = []
    assert retry_call(fn, attempts=3, delay=1.0, backoff=2.0, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_the_last_failure_is_reraised_unchanged_when_every_attempt_fails():
    fn, calls = flaky(9, error=TimeoutError("slow"))
    with pytest.raises(TimeoutError, match="slow"):
        retry_call(fn, attempts=3, sleep=lambda s: None)
    assert len(calls) == 3


@pytest.mark.parametrize("error", [ValueError("no market data returned"), StaleDataError("last bar is old"),
                                   ValueError("need 200 bars for MA200, got 12")])
def test_a_definite_answer_about_the_data_is_never_retried(error):
    fn, calls = flaky(9, error=error)
    with pytest.raises(type(error)):
        retry_call(fn, attempts=3, sleep=lambda s: None)
    assert len(calls) == 1


def test_a_garbled_reply_is_transient_even_though_a_json_error_is_a_value_error():
    assert is_transient_data_error(json.JSONDecodeError("cut off", "{", 1))
    assert not is_transient_data_error(ValueError("x")) and is_transient_data_error(ConnectionError("x"))
    fn, calls = flaky(1, error=json.JSONDecodeError("cut off", "{", 1))
    assert retry_call(fn, sleep=lambda s: None) == "ok" and len(calls) == 2


def test_a_custom_rule_can_veto_retrying_and_one_attempt_means_no_retry():
    fn, calls = flaky(9)
    with pytest.raises(ConnectionError):
        retry_call(fn, attempts=5, retry_if=lambda e: False, sleep=lambda s: None)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="attempts"):
        retry_call(fn, attempts=0)


# ---------------------------------------------------------------- the intraday price feed
def test_the_intraday_feed_retries_a_failed_download_once():
    idx = pd.DatetimeIndex(["2026-09-21 09:15"], tz="Asia/Kolkata")
    frame = pd.DataFrame({"Close": [101.5], "High": [102.0], "Low": [101.0], "Open": [101.0]}, index=idx)
    calls = []

    def download(ticker, **kw):
        calls.append(ticker)
        if len(calls) == 1:
            raise ConnectionError("yahoo blip")
        return frame

    sleeps = []
    feed = IntradayFeed(download=download, retry_sleep=sleeps.append)
    assert feed.last_price("AAA") == 101.5 and len(calls) == 2 and sleeps == [1.0]


def test_the_intraday_feed_gives_up_after_one_retry_and_the_caller_sees_the_error():
    feed = IntradayFeed(download=lambda t, **kw: (_ for _ in ()).throw(ConnectionError("down")), retry_sleep=lambda s: None)
    with pytest.raises(ConnectionError):
        feed.last_price("AAA")


# ---------------------------------------------------------------- telegram alerts
class Http:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def post(self, url, json):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(status):
    return httpx.Response(status, request=httpx.Request("POST", "http://x"))


def test_an_alert_survives_one_network_blip():
    http = Http(ConnectionError("blip"), reply(200))
    assert Notifier("t", "1", client=http, sleep=lambda s: None).send("hi") is True and http.calls == 2


def test_an_alert_is_retried_on_a_server_error_but_not_on_a_bad_token():
    server = Http(reply(503), reply(200))
    assert Notifier("t", "1", client=server, sleep=lambda s: None).send("hi") is True and server.calls == 2
    bad_token = Http(reply(401), reply(200))
    assert Notifier("bad", "1", client=bad_token, sleep=lambda s: None).send("hi") is False and bad_token.calls == 1


def test_an_alert_that_keeps_failing_returns_false_and_never_raises():
    http = Http(ConnectionError("a"), ConnectionError("b"))
    assert Notifier("t", "1", client=http, sleep=lambda s: None).send("hi") is False and http.calls == 2


# ---------------------------------------------------------------- the graph's snapshot node
class FlakySnapshot(FakeServices):
    def __init__(self, error, failures, **kw):
        super().__init__(["AAA"], approve={"AAA"}, **kw)
        self.error, self.failures, self.attempts = error, failures, 0

    def snapshot(self, symbol):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return super().snapshot(symbol)


def run(services):
    analysis = build_analysis_graph(services, snapshot_retry=FAST)
    return build_cycle_graph(services, analysis, build_decision_graph(services)).invoke(
        {"symbols": []}, config={"max_concurrency": 2})["results"]


def test_a_transient_snapshot_failure_costs_one_retry_not_the_stock_for_the_cycle():
    services = FlakySnapshot(ConnectionError("yahoo blip"), failures=1)
    [result] = run(services)
    assert result.action == "BUY" and result.error is None and services.attempts == 2


def test_a_stale_or_missing_data_answer_is_not_retried_and_becomes_an_error_result():
    for error in (StaleDataError("stale"), ValueError("no market data returned")):
        services = FlakySnapshot(error, failures=9)
        [result] = run(services)
        assert result.action == "ERROR" and services.attempts == 1


def test_a_snapshot_that_keeps_failing_is_reported_after_the_retry_is_used_up():
    services = FlakySnapshot(ConnectionError("down"), failures=9)
    [result] = run(services)
    assert result.action == "ERROR" and services.attempts == 2 and "ConnectionError" in result.error
