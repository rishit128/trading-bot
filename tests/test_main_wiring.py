
import main
from src.app import kit as app_kit, reports
from src.config import Settings
from src.database import make_session_factory
from src.engine.paper_broker import PaperBroker


def sessions(tmp_path):
    return make_session_factory(f"sqlite:///{tmp_path / 'm.db'}")


def test_india_kit_uses_paper_broker_and_disables_unreliable_news(tmp_path):
    kit = app_kit.build_kit(Settings(market="india", universe="market"), sessions(tmp_path))
    assert isinstance(kit.broker, PaperBroker) and kit.screener is not None
    assert kit.headlines_fn("RELIANCE") == [] and "simulator" in kit.broker_label


def test_india_watchlist_mode_has_no_screener(tmp_path):
    kit = app_kit.build_kit(Settings(market="india", universe="watchlist"), sessions(tmp_path))
    assert kit.screener is None


def test_india_needs_only_the_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("MARKET", "india")
    monkeypatch.setenv("UNIVERSE", "watchlist")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    pipeline = main.build_pipeline(live=False)
    assert pipeline.settings.market == "india" and pipeline.universe_fn is None


def test_india_screener_uses_the_nse_list_and_indian_thresholds(tmp_path):
    settings = Settings(market="india", min_price=250.0, min_traded_value=5e8, max_candidates=7)
    screener = app_kit.build_india_screener(settings)
    assert screener.cfg.min_price == 250.0 and screener.cfg.min_traded_value == 5e8 and screener.cfg.max_candidates == 7
    assert screener.chunk == 200


def test_report_prints_paper_account_summary(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'r.db'}"
    reports.print_report(Settings(market="india", database_url=url, paper_initial_cash=250_000.0))
    out = capsys.readouterr().out
    assert "250,000" in out and "closed trades 0" in out and "₹" in out


def test_india_kit_analyses_completed_sessions_and_quotes_live(tmp_path):
    kit = app_kit.build_kit(Settings(market="india"), sessions(tmp_path))
    assert kit.use_news is False and kit.quote_fn is not None


def test_india_pipeline_registers_only_the_technical_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("MARKET", "india")
    monkeypatch.setenv("UNIVERSE", "watchlist")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    pipeline = main.build_pipeline(live=False)
    assert list(pipeline.agents) == ["technical"] and pipeline.quote_fn is not None
    assert set(pipeline.analysis_graph.get_graph().nodes) == {"__start__", "fetch_data", "agent_technical", "collect", "__end__"}


def test_llm_cache_ttl_comes_from_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("UNIVERSE", "watchlist")
    monkeypatch.setenv("LLM_CACHE_HOURS", "2")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'c.db'}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    assert main.build_pipeline(live=False).agents["technical"].llm.cache_ttl == 2 * 3600


def test_print_graphs_lists_the_agents(capsys):
    reports.print_graphs(Settings(market="india"))
    india = capsys.readouterr().out
    assert "agent_technical" in india and "agent_sentiment" not in india
    assert india.count("%%") == 3 and "analyze_symbol" in india and "decide" in india


def test_holdout_flag_returns_a_callable_hook():
    """Regression: `--holdout` used to leave reserve_callback as None, so the reserve was never marked."""
    import main
    from types import SimpleNamespace

    from src.database import make_session_factory

    hook = main.build_holdout_callback(SimpleNamespace(sessions=make_session_factory("sqlite:///:memory:")))
    assert callable(hook)


def test_the_live_kit_downloads_a_stock_once_per_session_not_every_cycle(monkeypatch, tmp_path):
    """Regression: every 30-minute cycle re-downloaded two years of daily bars per stock."""
    import dataclasses
    from src.app import kit as kit_module
    from src.config import Settings
    from src.data.indicators import Snapshot
    from src.database import make_session_factory

    calls = []
    monkeypatch.setattr(kit_module, "fetch_snapshot",
                        lambda symbol, **kw: calls.append(symbol) or Snapshot(symbol, 1.0, 1.0, 1.0, 50.0, 1))
    settings = dataclasses.replace(Settings(), universe="watchlist")
    built = kit_module.build_kit(settings, make_session_factory(f"sqlite:///{tmp_path / 'k.db'}"))
    built.snapshot_fn("RELIANCE")
    built.snapshot_fn("RELIANCE")
    built.snapshot_fn("TCS")
    assert calls == ["RELIANCE", "TCS"]
