from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select

from src.database import PaperPositionRecord, PaperTradeRecord, make_session_factory
from src.engine.paper_broker import PaperBroker
from src.engine.costs import india_delivery_fees

T0 = datetime(2026, 9, 21, 4, 0, tzinfo=timezone.utc)  # 09:30 IST, a Monday


class FakeFeed:
    def __init__(self, prices=None):
        self.prices = dict(prices or {})
        self.bars = {}

    def last_price(self, symbol):
        if symbol not in self.prices:
            raise ValueError(f"{symbol}: no price")
        return self.prices[symbol]

    def bars_since(self, symbol, since):
        df = self.bars.get(symbol)
        if df is None:
            return pd.DataFrame()
        return df[df.index > pd.Timestamp(since)]


class FakeClock:
    def __init__(self):
        self.open, self.day = True, "2026-09-21"

    def is_open(self, now=None):
        return self.open

    def today_ist(self, now=None):
        return self.day


def no_fees(side, value):
    return 0.0


def bar(minutes_after_t0, o, h, l, c):
    return pd.DataFrame({"Open": [o], "High": [h], "Low": [l], "Close": [c]},
                        index=pd.DatetimeIndex([T0 + timedelta(minutes=minutes_after_t0)]))


def make(tmp_path, fees=no_fees, slippage=0.0, cash=1_000_000.0, prices=None, name="p.db"):
    now = [T0]
    feed, clock = FakeFeed(prices or {"AAA": 100.0}), FakeClock()
    sessions = make_session_factory(f"sqlite:///{tmp_path / name}")
    broker = PaperBroker(sessions, feed, clock, initial_cash=cash, fees=fees, slippage=slippage, now_fn=lambda: now[0])
    return broker, feed, clock, now, sessions


def test_buy_creates_position_with_bracket_levels_from_signal_price(tmp_path):
    broker, feed, _, _, sessions = make(tmp_path)
    fill = broker.buy_with_bracket("AAA", 100, 102.0, 0.02, 0.05)  # signal price 102, market 100
    assert fill.status == "filled" and fill.broker_order_id.startswith("paper-")
    with sessions() as s:
        pos = s.get(PaperPositionRecord, "AAA")
        assert (pos.qty, pos.avg_price, pos.stop, pos.target) == (100, 100.0, 99.96, 107.1)
    pf = broker.portfolio()
    assert pf.cash == 990_000.0 and pf.equity == 1_000_000.0 and pf.positions == {"AAA": 10_000.0} and pf.position_qty == {"AAA": 100}


def test_open_position_counts_as_open_bracket_order_and_blocks_second_buy(tmp_path):
    broker, *_ = make(tmp_path)
    assert broker.has_open_order("AAA") is False
    broker.buy_with_bracket("AAA", 10, 100.0, 0.02, 0.05)
    assert broker.has_open_order("AAA") is True
    with pytest.raises(ValueError, match="already open"):
        broker.buy_with_bracket("AAA", 10, 100.0, 0.02, 0.05)


def test_insufficient_cash_is_rejected_without_side_effects(tmp_path):
    broker, _, _, _, sessions = make(tmp_path, cash=5_000.0)
    with pytest.raises(ValueError, match="insufficient cash"):
        broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    assert broker.portfolio().cash == 5_000.0 and broker.has_open_order("AAA") is False


def test_slippage_is_adverse_on_both_sides(tmp_path):
    broker, feed, _, _, sessions = make(tmp_path, slippage=0.005)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    with sessions() as s:
        assert s.get(PaperPositionRecord, "AAA").avg_price == pytest.approx(100.5)
    broker.sell("AAA", 100)
    with sessions() as s:
        assert s.scalar(select(PaperTradeRecord.exit_price)) == pytest.approx(99.5)


def test_indian_delivery_fees_are_realistic():
    buy, sell = india_delivery_fees("BUY", 100_000), india_delivery_fees("SELL", 100_000)
    assert 118 < buy < 120      # STT 100 + stamp 15 + exchange/SEBI/GST ~3.6
    assert 119 < sell < 121     # STT 100 + exchange/SEBI/GST ~3.6 + DP 15.93
    assert 0.0022 < (buy + sell) / 100_000 < 0.0026  # ~0.24% round trip


def test_sell_records_trade_with_fees_and_updates_cash(tmp_path):
    broker, feed, _, _, sessions = make(tmp_path, fees=india_delivery_fees)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    feed.prices["AAA"] = 110.0
    broker.sell("AAA", 100)
    with sessions() as s:
        [trade] = s.scalars(select(PaperTradeRecord)).all()
        assert s.get(PaperPositionRecord, "AAA") is None
    gross = 1_000.0
    assert trade.reason == "SIGNAL" and trade.net_pnl == pytest.approx(gross - trade.fees)
    assert 20 < trade.fees < 40
    assert broker.portfolio().cash == pytest.approx(1_000_000.0 + trade.net_pnl)


def test_partial_sell_reduces_quantity_and_oversell_is_rejected(tmp_path):
    broker, _, _, _, sessions = make(tmp_path)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    broker.sell("AAA", 40)
    with sessions() as s:
        assert s.get(PaperPositionRecord, "AAA").qty == 60
    with pytest.raises(ValueError, match="cannot sell"):
        broker.sell("AAA", 61)
    with pytest.raises(ValueError, match="cannot sell"):
        broker.sell("NOPE", 1)


def stop_target_case(tmp_path, bars):
    broker, feed, _, now, sessions = make(tmp_path)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)  # stop 98.00, target 105.00
    feed.bars["AAA"] = bars
    now[0] = T0 + timedelta(hours=1)
    pf = broker.portfolio()
    with sessions() as s:
        trades = s.scalars(select(PaperTradeRecord)).all()
    return pf, trades


def test_stop_hit_intraday_exits_at_stop_price(tmp_path):
    pf, [t] = stop_target_case(tmp_path, bar(10, 100, 101, 97.0, 98.5))
    assert t.reason == "STOP" and t.exit_price == 98.0 and pf.positions == {} and pf.cash == pytest.approx(999_800.0)


def test_target_hit_exits_at_target_price(tmp_path):
    pf, [t] = stop_target_case(tmp_path, bar(10, 101, 106.0, 100.5, 105.5))
    assert t.reason == "TARGET" and t.exit_price == 105.0 and pf.cash == pytest.approx(1_000_500.0)


def test_gap_below_stop_exits_at_the_open_not_the_stop(tmp_path):
    _, [t] = stop_target_case(tmp_path, bar(10, 95.0, 96.0, 94.0, 95.5))
    assert t.reason == "STOP" and t.exit_price == 95.0


def test_gap_above_target_exits_at_the_open(tmp_path):
    _, [t] = stop_target_case(tmp_path, bar(10, 108.0, 109.0, 107.0, 108.5))
    assert t.reason == "TARGET" and t.exit_price == 108.0


def test_stop_wins_when_one_bar_touches_both_levels(tmp_path):
    _, [t] = stop_target_case(tmp_path, bar(10, 100.0, 106.0, 97.0, 101.0))
    assert t.reason == "STOP"


def test_no_exit_when_price_stays_between_levels(tmp_path):
    pf, trades = stop_target_case(tmp_path, bar(10, 100.0, 104.0, 99.0, 101.0))
    assert trades == [] and pf.position_qty == {"AAA": 100}


def test_bars_are_not_rechecked_after_being_processed(tmp_path):
    broker, feed, _, now, sessions = make(tmp_path)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    feed.bars["AAA"] = bar(10, 100.0, 104.0, 99.0, 101.0)
    now[0] = T0 + timedelta(hours=1)
    broker.portfolio()
    with sessions() as s:
        assert s.get(PaperPositionRecord, "AAA").last_checked.replace(tzinfo=timezone.utc) == now[0]


def test_bars_before_entry_cannot_trigger_a_stop(tmp_path):
    broker, feed, _, now, sessions = make(tmp_path)
    feed.bars["AAA"] = bar(-30, 90.0, 91.0, 89.0, 90.0)  # a crash BEFORE we bought
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    now[0] = T0 + timedelta(hours=1)
    assert broker.portfolio().position_qty == {"AAA": 100}


def test_price_feed_failure_values_position_at_cost_and_does_not_crash(tmp_path):
    broker, feed, *_ = make(tmp_path)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    del feed.prices["AAA"]
    assert broker.portfolio().equity == pytest.approx(1_000_000.0)


def test_start_of_day_equity_is_the_previous_days_last_mark(tmp_path):
    broker, feed, clock, now, _ = make(tmp_path)
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    feed.prices["AAA"] = 104.0
    day1 = broker.portfolio()  # first call ever: no previous mark yet
    assert day1.start_of_day_equity == pytest.approx(1_000_400.0)
    clock.day = "2026-09-22"
    feed.prices["AAA"] = 101.0
    day2 = broker.portfolio()
    assert day2.start_of_day_equity == pytest.approx(1_000_400.0) and day2.equity == pytest.approx(1_000_100.0)


def test_market_open_delegates_to_the_clock(tmp_path):
    broker, _, clock, _, _ = make(tmp_path)
    assert broker.is_market_open() is True
    clock.open = False
    assert broker.is_market_open() is False


def test_state_persists_across_restarts_and_initial_cash_is_not_reset(tmp_path):
    broker, feed, clock, now, sessions = make(tmp_path, cash=500_000.0)
    broker.buy_with_bracket("AAA", 10, 100.0, 0.02, 0.05)
    reopened = PaperBroker(sessions, feed, clock, initial_cash=9_999_999.0, fees=no_fees, now_fn=lambda: now[0])
    pf = reopened.portfolio()
    assert pf.cash == 499_000.0 and pf.position_qty == {"AAA": 10}


def test_summary_reports_trades_win_rate_fees_and_exits(tmp_path):
    broker, feed, _, _, _ = make(tmp_path, fees=india_delivery_fees, prices={"AAA": 100.0, "BBB": 200.0})
    broker.buy_with_bracket("AAA", 100, 100.0, 0.02, 0.05)
    feed.prices["AAA"] = 110.0
    broker.sell("AAA", 100)                    # winner
    broker.buy_with_bracket("BBB", 50, 200.0, 0.02, 0.05)
    feed.prices["BBB"] = 190.0
    broker.sell("BBB", 50)                     # loser
    s = broker.summary()
    assert s["closed_trades"] == 2 and s["win_rate"] == 0.5 and s["open_positions"] == 0
    assert s["exits"] == {"STOP": 0, "TARGET": 0, "SIGNAL": 2} and s["fees_paid"] > 0
    assert s["equity"] == pytest.approx(1_000_000.0 + s["realized_net_pnl"])
    assert s["return_pct"] == pytest.approx(s["equity"] / 1_000_000.0 - 1)


def test_summary_with_no_trades(tmp_path):
    broker, *_ = make(tmp_path)
    s = broker.summary()
    assert s["closed_trades"] == 0 and s["win_rate"] is None and s["return_pct"] == 0.0


def test_works_end_to_end_inside_the_trading_pipeline(tmp_path):
    from src.config import RiskLimits, Settings
    from src.data.indicators import Snapshot
    from src.llm import AgentSignal
    from src.pipeline import TradingPipeline
    from tests.test_llm_and_pipeline import StubAgent

    broker, feed, _, _, sessions = make(tmp_path, prices={"AAA": 100.0})
    pipe = TradingPipeline(
        Settings(watchlist=("AAA",), dry_run=False, risk=RiskLimits()),
        [StubAgent(lambda s: AgentSignal(action="BUY", confidence=0.9, reasoning="t"))], broker, sessions,
        lambda sym: Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000), lambda sym: [],
    )
    [r] = pipe.run_once()
    assert r.order_status == "filled" and broker.portfolio().position_qty == {"AAA": 500}  # 5% of 1,000,000 / 100
    [again] = pipe.run_once()  # position now at its 5% cap: the risk engine stops a second buy before cooldown matters
    assert again.order_status is None and "no room" in again.risk.reason


def test_holdings_show_buy_date_live_price_and_profit_or_loss_best_first(tmp_path):
    broker, feed, _, _, _ = make(tmp_path, prices={"UP": 100.0, "DOWN": 200.0})
    broker.buy_with_bracket("UP", 10, 100.0, 0.08, 1.0)
    broker.buy_with_bracket("DOWN", 5, 200.0, 0.08, 1.0)
    feed.prices.update({"UP": 110.0, "DOWN": 180.0})
    up, down = broker.holdings()
    assert (up["symbol"], up["pnl"], round(up["pnl_pct"], 2), up["opened_at"]) == ("UP", 100.0, 0.10, T0)
    assert (down["symbol"], down["pnl"], round(down["pnl_pct"], 2), down["stop"]) == ("DOWN", -100.0, -0.10, 184.0)


def test_trade_history_lists_closed_trades_newest_first_with_both_dates(tmp_path):
    broker, feed, _, now, _ = make(tmp_path, prices={"AAA": 100.0, "BBB": 50.0})
    broker.buy_with_bracket("AAA", 10, 100.0, 0.08, 1.0)
    broker.buy_with_bracket("BBB", 10, 50.0, 0.08, 1.0)
    feed.prices.update({"AAA": 120.0, "BBB": 40.0})
    broker.sell("AAA", 10)
    now[0] = T0 + timedelta(days=1)
    broker.sell("BBB", 10)
    first, second = broker.trade_history()
    assert (first["symbol"], first["net_pnl"], first["opened_at"], first["closed_at"]) == ("BBB", -100.0, T0, T0 + timedelta(days=1))
    assert (second["symbol"], second["net_pnl"], second["reason"]) == ("AAA", 200.0, "SIGNAL")
