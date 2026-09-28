"""Phase 2's machinery: every rule picks the trade it describes, the null is a fair null, the statistics are right, and the
whole harness FINDS a planted edge and does NOT invent one in noise (so "no candidates" on real data means something)."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.research import intraday_hypotheses as hyp
from src.research.intraday_audit import FEE_FRACTION, Plan, StockDay, random_entry, simulate_plan, stock_days, trade_returns
from tests.test_intraday_audit import DAY0, minutes_of, session

N = 75


def sday(closes, volumes=None, prev_close=None, open0=100.0, rng=(100.5, 99.5)):
    """A StockDay from closes (padded flat to a full session): bars run open to close, no wicks; the range is `rng`."""
    c = np.array(list(closes) + [closes[-1]] * (N - len(closes)), float)
    o = np.concatenate([[open0], c[:-1]])
    h, low = np.maximum(o, c), np.minimum(o, c)
    h[:3], low[:3] = rng
    v = np.full(N, 1000.0) if volumes is None else np.array(list(volumes) + [1000.0] * (N - len(volumes)), float)
    t = np.array([(DAY0 + timedelta(hours=9, minutes=15 + 5 * i)).time() for i in range(N)])
    return StockDay("S", DAY0.date(), t, minutes_of(t), o, h, low, c, v, prev_close)


def bar_of(plan):
    """The bar a plan enters on, or None when the rule picks no trade."""
    return None if plan is None else plan.index


def path(**moves):
    """closes: 100 for the range bars, then `bar=price` overrides held until the next override."""
    closes, price = [], 100.0
    for i in range(N):
        price = moves.get(f"b{i}", price)
        closes.append(price)
    return closes


# -- trade mechanics, long and short ----------------------------------------------------------------------------------------
def test_trade_returns_charge_slippage_on_both_fills_and_the_fee_once():
    gross, net = trade_returns(100.0, 102.0, 1, 0.001)
    assert gross == pytest.approx(0.02) and net == pytest.approx(102 * 0.999 / (100 * 1.001) - 1 - FEE_FRACTION)
    gross, net = trade_returns(100.0, 98.0, -1, 0.001)                      # a short: sold at 99.9, bought back at 98.098
    assert gross == pytest.approx(0.02) and net == pytest.approx((99.9 - 98.098) / 100 - FEE_FRACTION)
    flat_long, flat_short = trade_returns(100.0, 100.0, 1, 0.0)[1], trade_returns(100.0, 100.0, -1, 0.0)[1]
    assert flat_long == pytest.approx(-FEE_FRACTION) == pytest.approx(flat_short)


def test_a_short_plan_stops_out_above_and_takes_profit_below():
    d = sday(path(b10=100.0))
    entry_bar = 10
    # nothing touched: flat at 15:15
    assert simulate_plan(d, Plan(entry_bar, -1, stop=101.0, target=98.0)) == (100.0, "EOD")
    # bar 11 opens 100, closes 101.4: high 101.4
    up = sday(path(b11=101.4))
    assert simulate_plan(up, Plan(entry_bar, -1, stop=101.0, target=98.0)) == (101.0, "STOP")
    # a spike through it fills AT it
    assert simulate_plan(sday(path(b11=103.0)), Plan(10, -1, stop=101.0, target=98.0)) == (101.0, "STOP")
    down = sday(path(b11=97.0))
    assert simulate_plan(down, Plan(entry_bar, -1, stop=101.0, target=98.0)) == (98.0, "TARGET")
    # no levels: only the square-off
    assert simulate_plan(down, Plan(entry_bar, -1, stop=None, target=None))[1] == "EOD"


def test_a_short_that_gaps_through_its_stop_is_filled_at_the_open_and_open_only_mode_misses_a_wick():
    d = sday(path())
    # bar 12 OPENS at 103, through the 101 stop
    d.o[12], d.h[12], d.c[12] = 103.0, 103.0, 103.0
    assert simulate_plan(d, Plan(10, -1, stop=101.0), "full") == (103.0, "STOP")
    assert simulate_plan(d, Plan(10, -1, stop=101.0), "open_only") == (103.0, "STOP")
    wick = sday(path())
    # bar 11 spikes to 101.4 and closes back at 100
    wick.h[11] = 101.4
    assert simulate_plan(wick, Plan(10, -1, stop=101.0), "full") == (101.0, "STOP")
    assert simulate_plan(wick, Plan(10, -1, stop=101.0), "open_only") == (100.0, "EOD")


# -- the rules pick the trades they describe --------------------------------------------------------------------------------
def test_orb_short_sells_a_volume_backed_breakdown_below_vwap_with_the_stop_above_the_range():
    d = sday(path(b10=99.0), volumes=[1000] * 10 + [4000])                                            # close 99 < range low 99.5
    plan = hyp.orb_short(d)
    assert (plan.index, plan.side, plan.stop) == (10, -1, 100.5) and plan.target == pytest.approx(99.0 - 2 * 1.5)
    assert hyp.orb_short(sday(path(b10=99.0))) is None                                                # no volume surge
    # that is a breakout, not a breakdown
    assert hyp.orb_short(sday(path(b10=101.0), volumes=[1000] * 10 + [4000])) is None
    assert hyp.orb_short(sday(path(b10=99.0), volumes=[1000] * 10 + [4000], rng=(103.0, 97.0))) is None   # range wider than 2.5%
    # a breakdown that is still ABOVE VWAP (the stock spent the morning far lower) is not sold: bars 3-8 at 90 drag VWAP down
    rebound = sday(path(b3=90.0, b9=99.0), volumes=[1000] * 9 + [9000])
    assert hyp.orb_short(rebound) is None
    # the entry window for the short is [09:30, 14:00): bar 56 starts 13:55 (in), bar 57 starts 14:00 (out)
    assert [bar_of(hyp.orb_short(sday(path(**{f"b{i}": 99.0}), volumes=[1000] * i + [4000]))) for i in (56, 57)] == [56, None]


def test_gap_and_go_needs_a_gap_of_at_least_one_percent_and_a_breakout_in_the_first_hour():
    up = path(b4=101.0)
    plan = hyp.gap_and_go(sday(up, prev_close=98.0))                                                  # gap +2.04%
    assert (plan.index, plan.side, plan.stop, plan.target) == (4, 1, 99.5, None)
    assert hyp.gap_and_go(sday(up, prev_close=99.6)) is None                                          # gap 0.4%: too small
    assert hyp.gap_and_go(sday(up, prev_close=99.3)) is None                                          # gap 0.7%: still under 1%
    assert hyp.gap_and_go(sday(up, prev_close=None)) is None                                          # no previous close known
    # breakout after 10:30 is outside the window
    assert hyp.gap_and_go(sday(path(b40=101.0), prev_close=98.0)) is None
    # the window is [09:30, 10:30): bar 3 starts 09:30 (in), bar 14 starts 10:25 (in), bar 15 starts 10:30 (out)
    assert [bar_of(hyp.gap_and_go(sday(path(**{f"b{i}": 101.0}), prev_close=98.0))) for i in (3, 14, 15)] == [3, 14, None]
    # never clears the range high
    assert hyp.gap_and_go(sday(path(b4=100.4), prev_close=98.0)) is None


def test_gap_fade_trades_back_toward_the_previous_close_only_once_the_gap_has_started_to_close():
    # gapped up 2.04%, then trades below the open
    short = hyp.gap_fade(sday(path(b4=99.0), prev_close=98.0 / 1.0 * 0.98, open0=100.0))
    assert (short.side, short.stop, short.index) == (-1, 100.5, 4)
    # gapped down 1.96%, then trades above the open
    long_ = hyp.gap_fade(sday(path(b4=101.0), prev_close=102.0, open0=100.0))
    assert (long_.side, long_.stop, long_.index) == (1, 99.5, 4)
    # gap up but the price never gives any back
    assert hyp.gap_fade(sday(path(), prev_close=98.0)) is None
    # a 0.5% gap is not a big gap
    assert hyp.gap_fade(sday(path(b4=99.0), prev_close=99.5)) is None


def test_first_hour_momentum_needs_a_one_percent_move_and_the_same_side_of_vwap_at_1015():
    up = hyp.first_hour_momentum(sday(path(b11=101.5)))
    assert (up.index, up.side) == (11, 1) and up.stop == pytest.approx(101.5 * 0.985) and up.target is None
    down = hyp.first_hour_momentum(sday(path(b11=98.5)))
    assert (down.index, down.side) == (11, -1) and down.stop == pytest.approx(98.5 * 1.015)
    assert hyp.first_hour_momentum(sday(path(b11=100.9))) is None                                     # +0.9%: below the threshold
    # +2% at 10:15 but the morning spent far above that price, so it is BELOW VWAP: no long
    high_morning = sday(path(b3=110.0, b11=102.0), volumes=[1000] * 3 + [9000] * 8 + [1000])
    assert hyp.first_hour_momentum(high_morning) is None


def test_last_half_hour_momentum_follows_the_days_direction_from_1445_with_no_stop():
    plan = hyp.last_half_hour(sday(path(b65=101.2)))
    assert (plan.index, plan.side, plan.stop, plan.target) == (65, 1, None, None)
    assert hyp.last_half_hour(sday(path(b65=98.5))).side == -1
    assert hyp.last_half_hour(sday(path(b65=100.5))) is None


# -- the null is fair -------------------------------------------------------------------------------------------------------
def test_random_entry_nulls_only_enter_where_the_stop_is_on_the_losing_side_and_keep_the_rules_side():
    d = sday(path(b10=99.0, b20=101.0), volumes=[1000] * 10 + [4000])
    for seed in range(40):
        rng = np.random.default_rng(seed)
        short = hyp.orb_short_null(d, rng)
        assert short.side == -1 and d.c[short.index] < short.stop and short.target < d.c[short.index]
        long_ = hyp.orb_long_null(d, rng)
        assert long_.side == 1 and d.c[long_.index] > long_.stop and long_.target > d.c[long_.index]
    fade = hyp.gap_fade_null(sday(path(b4=99.0), prev_close=98.0 * 0.98), np.random.default_rng(0))
    assert fade.side == -1
    assert hyp.gap_fade_null(sday(path(), prev_close=100.0), np.random.default_rng(0)) is None
    assert hyp.gap_and_go_null(sday(path(), prev_close=100.0), np.random.default_rng(0)) is None


def test_the_orb_random_entry_covers_exactly_the_live_entry_window_and_the_live_range_filter():
    # only bars 2 (09:25, still the range), 3 (09:30), 56 (13:55) and 57 (14:00) are above the range low: 3 and 56 are eligible
    closes = [90.0] * N
    for i in (2, 3, 56, 57):
        closes[i] = 101.0
    d = sday(closes)
    picks = {random_entry(d, np.random.default_rng(seed)).index for seed in range(60)}
    assert picks == {3, 56}
    assert random_entry(sday(closes, rng=(103.0, 97.0)), np.random.default_rng(0)) is None       # range wider than 2.5%: no trade


def test_the_fixed_time_nulls_keep_the_same_days_but_toss_a_coin_for_the_side():
    d = sday(path(b65=101.2))
    sides = [hyp.last_half_hour_null(d, np.random.default_rng(seed)).side for seed in range(200)]
    assert 60 < sides.count(1) < 140 and set(sides) == {1, -1}                                       # ~ a fair coin
    # same eligibility as the rule
    assert hyp.last_half_hour_null(sday(path(b65=100.2)), np.random.default_rng(0)) is None
    flipped = [hyp.first_hour_momentum_null(sday(path(b11=101.5)), np.random.default_rng(s)) for s in range(50)]
    assert {p.side for p in flipped} == {1, -1} and all(p.stop == pytest.approx(101.5 * (1 - p.side * 0.015)) for p in flipped)


def test_random_entries_are_reproducible_from_the_seed():
    d = sday(path(b10=99.0, b20=101.0))
    assert hyp.orb_long_null(d, np.random.default_rng(3)) == hyp.orb_long_null(d, np.random.default_rng(3))


# -- statistics -------------------------------------------------------------------------------------------------------------
def test_cluster_t_is_the_t_statistic_of_the_per_date_means():
    assert hyp.cluster_t(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(2.5 / (np.std([1, 2, 3, 4], ddof=1) / 2))
    assert hyp.cluster_t(pd.Series([1.0])) == 0.0 and hyp.cluster_t(pd.Series([2.0, 2.0, 2.0])) == 0.0


def trades(net_by_date, per_date=10, side=1, exit_=101.0, price=100.0):
    rows = [{"date": d, "net": n, "gross": n + 0.001, "price": price, "exit": exit_, "side": side, "symbol": "S", "why": "EOD"}
            for d, n in net_by_date.items() for _ in range(per_date)]
    return pd.DataFrame(rows)


def dates(n):
    base = pd.Timestamp("2026-01-05").date()
    return [base + timedelta(days=i) for i in range(n)]


def test_a_consistent_edge_that_beats_its_null_passes_every_criterion():
    rng = np.random.default_rng(0)
    signal = trades({d: 0.004 + rng.normal(0, 0.0004) for d in dates(40)})
    null = trades({d: -0.001 + rng.normal(0, 0.0004) for d in dates(40)})
    out = hyp.judge("h", "idea", signal, null, 0.0005)
    assert out.candidate and out.failed == [] and out.trades == 400 and out.dates == 40
    assert out.mean_net == pytest.approx(0.004, abs=3e-4) and out.paired_t > hyp.T_BAR and out.detectable > 0


@pytest.mark.parametrize("case", ["few_trades", "noisy", "one_half", "late_only", "slippage", "null"])
def test_each_criterion_can_veto_a_candidate(case):
    rng = np.random.default_rng(1)
    good = {d: 0.004 + rng.normal(0, 0.0004) for d in dates(40)}
    null = trades({d: -0.001 + rng.normal(0, 0.0004) for d in dates(40)})
    signal = trades(good)
    if case == "few_trades":
        signal = trades(good, per_date=2)
    elif case == "noisy":
        signal = trades({d: 0.004 + rng.normal(0, 0.05) for d in dates(40)})
    elif case == "one_half":
        signal = trades({d: (0.008 if i < 20 else -0.001) for i, d in enumerate(dates(40))})
    elif case == "late_only":
        # good on average, but only in the second half
        signal = trades({d: (-0.001 if i < 20 else 0.008) for i, d in enumerate(dates(40))})
    elif case == "slippage":
        signal = trades(good, exit_=100.05)                              # gross ~0.05%: doubling the slippage turns it negative
    elif case == "null":
        null = trades(good)                                              # the null does just as well: the signal adds nothing
    out = hyp.judge("h", "idea", signal, null, 0.0005)
    assert not out.candidate and out.failed


def test_significance_is_required_even_when_the_edge_beats_a_very_poor_null():
    """Positive, in both halves, robust to slippage, far better than the null: but t is 2.5, under the 2.6 bar, so it fails."""
    wobble = np.tile([1.0, -1.0], 20) * 0.001
    signal = trades({d: 0.0004 + w for d, w in zip(dates(40), wobble)})
    null = trades({d: -0.005 + w / 10 for d, w in zip(dates(40), wobble)})
    out = hyp.judge("h", "idea", signal, null, 0.0005)
    assert out.first_half > 0 and out.second_half > 0 and out.net_double_slippage > 0 and out.paired_t > hyp.T_BAR
    assert 2.0 < out.t_net < hyp.T_BAR and not out.candidate and len(out.failed) == 1 and "t 2.5" in out.failed[0]


def test_a_hypothesis_with_no_trades_fails_with_a_reason():
    out = hyp.judge("h", "idea", pd.DataFrame(), pd.DataFrame(), 0.0005)
    assert not out.candidate and out.failed == ["no trades"]


# -- the harness finds a planted edge and does not invent one --------------------------------------------------------------
def market(edge, n_stocks=25, n_days=45):
    """Random-walk sessions. With `edge`, a stock that is up or down >= 1% at 14:45 keeps going the same way into the close."""
    frames = {}
    for s in range(n_stocks):
        parts = []
        for day in range(n_days):
            frame = session(day, s * 10_000 + day, spike_prob=0.0)
            if edge:
                move = frame["Close"].iloc[65] / frame["Open"].iloc[0] - 1
                if abs(move) >= 0.01:
                    ramp = np.concatenate([np.linspace(0, 1, 6), np.ones(len(frame) - 66 - 6 + 1 + 0)])[: len(frame) - 66]
                    drift = 1 + np.sign(move) * 0.012 * ramp
                    for col in ("Open", "High", "Low", "Close"):
                        frame.iloc[66:, frame.columns.get_loc(col)] = frame[col].iloc[66:].to_numpy() * drift
            parts.append(frame)
        frames[f"S{s}"] = pd.concat(parts)
    return frames


def test_the_harness_flags_a_planted_last_half_hour_edge_as_a_candidate_and_not_the_same_market_without_it():
    planted = hyp.run(stock_days(market(edge=True)), [hyp.HYPOTHESES[4]], seeds=3)[0]
    assert planted.trades >= hyp.MIN_TRADES and planted.mean_net > 0.003, planted
    assert planted.candidate, planted.failed
    noise = hyp.run(stock_days(market(edge=False)), [hyp.HYPOTHESES[4]], seeds=3)[0]
    assert not noise.candidate and noise.mean_net < 0                       # costs make a zero-edge rule a steady loser


def test_report_lists_every_hypothesis_says_why_each_failed_and_states_the_kill_criterion():
    outcomes = hyp.run(stock_days(market(edge=False)), hyp.HYPOTHESES[3:], seeds=1)
    text = hyp.report(outcomes)
    assert "H4 first-hour momentum" in text and "H5 last-half-hour momentum" in text and "fails -" in text
    assert "kill criterion applies" in text and "smallest edge that could be detected" in text
    planted = hyp.run(stock_days(market(edge=True)), [hyp.HYPOTHESES[4]], seeds=3)
    assert "CANDIDATES: H5 last-half-hour momentum" in hyp.report(planted) and "<-- CANDIDATE" in hyp.report(planted)


def test_previous_close_is_only_known_when_the_previous_stored_session_is_adjacent():
    near = pd.concat([session(0, 1), session(1, 2)])                       # consecutive days
    far = pd.concat([session(0, 1), session(20, 2)])                       # twenty days apart: a hole in the archive
    assert stock_days({"S": near})[1].prev_close is not None and stock_days({"S": near})[0].prev_close is None
    assert stock_days({"S": far})[1].prev_close is None
