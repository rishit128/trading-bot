import pandas as pd

from src.research.shadow import (
    COST_PCT,
    Stats,
    ai_cells,
    ai_verdict,
    catalyst_verdict,
    score_signal,
    stats,
    unrecorded_entries,
    with_catalyst,
)

IST = "Asia/Kolkata"


def t(hm):
    return pd.Timestamp(f"2026-11-04 {hm}", tz=IST)


def session(rows):
    return pd.DataFrame([{"t": t(hm), "high": h, "low": low, "close": c} for hm, h, low, c in rows])


def test_a_signal_is_scored_with_the_stop_first_and_nothing_from_its_own_bar():
    bars = session([("09:30", 500, 1, 100), ("09:35", 101, 99.5, 100), ("09:40", 104, 95, 100)])
    kind, net = score_signal(bars, t("09:30"), 100.0, 98.0, 104.0)
    assert kind == "STOP" and abs(net - (-2.0 - COST_PCT)) < 1e-9  # both touched: the stop
    won = session([("09:35", 104.5, 99, 104)])
    kind, net = score_signal(won, t("09:30"), 100.0, 98.0, 104.0)
    assert kind == "TARGET" and abs(net - (4.0 - COST_PCT)) < 1e-9
    held = session([("09:35", 101, 99, 100), ("15:10", 101, 99, 101), ("15:15", 200, 1, 150)])
    kind, net = score_signal(held, t("09:30"), 100.0, 98.0, 104.0)
    assert kind == "CLOSE" and abs(net - (1.0 - COST_PCT)) < 1e-9  # the square-off bar is not used
    assert score_signal(session([("09:30", 1, 1, 1)]), t("09:30"), 100.0, 98.0, 104.0) is None


def test_only_news_published_before_the_signal_counts():
    signals = pd.DataFrame(
        [
            {"symbol": "A", "day": "2026-11-04", "bar_time": t("09:30")},
            {"symbol": "B", "day": "2026-11-04", "bar_time": t("09:30")},
            {"symbol": "C", "day": "2026-11-04", "bar_time": t("09:30")},
        ]
    )
    filings = pd.DataFrame(
        [
            {"symbol": "A", "session": "2026-11-04", "published": t("09:34")},  # known when the 09:30-09:35 bar closed
            {"symbol": "B", "session": "2026-11-04", "published": t("11:00")},  # after the signal
        ]
    )
    assert with_catalyst(signals, filings).tolist() == [True, False, False]


def test_stats_cluster_by_date():
    frame = pd.DataFrame({"day": ["d1", "d1", "d2", "d3", "d4"], "net": [1.0, 3.0, 2.0, 1.0, 3.0]})
    s = stats(frame)
    assert s.n == 5 and s.dates == 4 and s.first_half == 2.0 and s.second_half == 2.0
    assert stats(frame.iloc[:0]).n == 0


def good(n=120, dates=40, mean=0.5, t=3.0, first=0.4, second=0.6):
    return Stats(n, dates, mean, 0.5, t, first, second)


def test_the_news_gate_needs_enough_evidence_and_every_condition():
    rest = good(mean=-0.1)
    assert catalyst_verdict(good(n=50), rest).startswith("RUNNING")
    assert catalyst_verdict(good(dates=10), rest).startswith("RUNNING")
    assert catalyst_verdict(good(), rest).startswith("PASS")
    for failing in (good(t=1.0), good(first=-0.1), good(mean=-0.2)):
        assert catalyst_verdict(failing, rest).startswith("FAIL")


def test_ai_cells_leave_out_held_stocks_and_keep_the_last_decision_of_a_bar():
    decisions = pd.DataFrame(
        [
            {"symbol": "A", "bar_date": "d1", "rule_action": "BUY", "final_action": "HOLD", "candidate_source": "screener"},
            {"symbol": "A", "bar_date": "d1", "rule_action": "BUY", "final_action": "BUY", "candidate_source": "screener"},
            {"symbol": "B", "bar_date": "d1", "rule_action": "BUY", "final_action": "BUY", "candidate_source": "held"},
        ]
    )
    outcomes = pd.DataFrame(
        [{"symbol": "A", "bar_date": "d1", "gross_return": 0.1}, {"symbol": "B", "bar_date": "d1", "gross_return": 0.2}]
    )
    cells = ai_cells(decisions, outcomes)
    assert cells["symbol"].tolist() == ["A"] and cells["cell"].tolist() == ["buy / AI buy"]


def test_the_ai_gate_scores_each_disagreement_from_the_ais_side():
    def cells(n, ai_right):
        rows = []
        for i in range(n):
            ai_buys = i % 2 == 0
            move = (0.05 if ai_buys else -0.05) * (1 if ai_right else -1) + (0.001 if i % 3 else 0)
            rows.append(
                {
                    "day": f"d{i % 30}",
                    "rule_action": "HOLD" if ai_buys else "BUY",
                    "final_action": "BUY" if ai_buys else "HOLD",
                    "gross_return": move,
                }
            )
        return pd.DataFrame(rows)

    assert ai_verdict(cells(50, True)).startswith("RUNNING")
    assert ai_verdict(cells(120, True)).startswith("PASS")
    assert ai_verdict(cells(120, False)).startswith("FAIL")


def test_entries_without_an_order_are_found():
    assert unrecorded_entries([("A", "d1"), ("B", "d1")], [("A", "d1"), ("B", "d2")]) == [("B", "d1")]
