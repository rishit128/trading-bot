from types import SimpleNamespace

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

from src.engine.broker import AlpacaBroker


class FakeTrading:
    def __init__(self, open_orders=()):
        self.submitted, self.cancelled, self.open_orders = [], [], list(open_orders)

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="abc", status="accepted")

    def get_orders(self, req):
        return self.open_orders

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)


def test_buy_is_gtc_bracket_with_correct_levels():
    fake = FakeTrading()
    fill = AlpacaBroker(client=fake).buy_with_bracket("AAPL", 10, 100.0, 0.02, 0.05)
    req = fake.submitted[0]
    assert (req.symbol, req.qty, req.side) == ("AAPL", 10, OrderSide.BUY)
    assert req.time_in_force == TimeInForce.GTC and req.order_class == OrderClass.BRACKET
    assert req.stop_loss.stop_price == 98.0 and req.take_profit.limit_price == 105.0
    assert fill.broker_order_id == "abc"


def test_sell_cancels_open_bracket_legs_first():
    fake = FakeTrading(open_orders=[SimpleNamespace(id="leg1"), SimpleNamespace(id="leg2")])
    AlpacaBroker(client=fake).sell("AAPL", 7)
    assert fake.cancelled == ["leg1", "leg2"]
    assert fake.submitted[0].side == OrderSide.SELL and fake.submitted[0].qty == 7


class FlakySubmit(FakeTrading):
    def __init__(self, failures, open_orders=()):
        super().__init__(open_orders)
        self.failures = failures

    def submit_order(self, req):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("blip")
        return super().submit_order(req)


def test_sell_retries_once_after_a_transient_failure():
    fake = FlakySubmit(failures=1, open_orders=[SimpleNamespace(id="leg")])
    fill = AlpacaBroker(client=fake).sell("AAPL", 7)
    assert fill.broker_order_id == "abc" and fake.cancelled == ["leg"]


def test_sell_failure_after_cancelling_legs_raises_a_loud_unprotected_error():
    import pytest

    fake = FlakySubmit(failures=2, open_orders=[SimpleNamespace(id="leg")])
    with pytest.raises(RuntimeError, match="UNPROTECTED"):
        AlpacaBroker(client=fake).sell("AAPL", 7)
