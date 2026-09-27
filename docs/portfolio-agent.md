# Real-portfolio agent: the plan

An information agent for the real Integrated account. It **never places orders**. You decide and place every trade
yourself. It tells you what you hold, what is risky, and, only once the paper bot has earned it, what the bot's rules
would do. It is not a licensed adviser, and its output is information, not a recommendation.

## The flow

```
 /portfolio (you give the OTP)          every trading day (no login needed)          weekly / on demand
 ─────────────────────────────          ───────────────────────────────────          ──────────────────
 log in → read pages + tabs → log out    holdings from the last login                 report on Telegram
   holdings, cost & dates (cap. gains)   + Yahoo prices (ISIN → NSE symbol)           (facts → flags → bot view)
   dividends, transactions, charges      → trend, drawdown, returns vs Nifty
   portal totals (value/invested/gain)   → alerts only when something changes
```

You log in only when holdings change (after you buy or sell), or about weekly. Prices are refreshed daily without a
login, because holdings do not change unless you trade.

## What it tells you, in three layers

| Layer | Available | What it is | Examples |
|---|---|---|---|
| 1. Facts | now | numbers read from the portal and the market | value, invested, gain; weights; sectors; each holding's return since purchase; holding period and tax status (short/long term); dividends received |
| 2. Risk flags | now (rules are fixed and written down) | plain rules on those facts, the same every day | one stock > 20%, one sector > 30%, top five > 60%; a holding 25%+ below cost; a holding below its 200-day average (the trend-exit rule the paper bot uses); 30%+ below its 52-week high; lagging Nifty by 20%+ over a year; "turns long-term in N days" (selling before then costs more tax) |
| 3. Bot view | **only after the paper-trading gate** | what the paper bot's rules would do with each holding and which new stocks it likes, with its live track record next to it | "the bot's rule would exit X (trend broken); its live record is 34 trades, +6% vs Nifty" |

Layers 1 and 2 are facts and fixed rules. They do not claim an edge, so they can start now. Layer 3 is a claim that the
bot's buy/sell rules are worth following, and research so far has not proven that (STRATEGY.md). So it stays switched
off until the paper account proves it on live data.

## The paper-trading gate (set now, before any results)

Layer 3 turns on only when **all** of these hold on the live paper account:

1. At least **6 months** of live paper trading and at least **30 closed trades**.
2. After fees, it **beats Nifty 50** and the equal-weight investable basket over the same period.
3. Worst drawdown is **no worse than Nifty's** over the same period.
4. Its setup stats still hold outside the backtest: live win rate and average win/loss are within the backtest's
   range.

If the gate fails, Layer 3 stays off, and the agent says so in the report rather than quietly showing weaker advice.

## Steps

1. **Done:** read-only login with the OTP on Telegram, step-by-step messages, holdings from the Portfolio Analyzer,
   portal totals, sectors, concentration flags, the guard, and logout.
2. **Done:** read-only tabs (Demat: Holdings, Analyser, Dividend, Transaction, Demat Charges; Capital gains:
   Realised, Current Holdings, Short/Long Term Holdings, Transaction History).
3. **Done (2026-09-27):** the tabs' statements are read, and the report built from them:
   - average buy price, cost, P&L and buy/sell history per stock;
   - short-term lots with their long-term dates;
   - dividends;
   - this year's sales;
   - recent transactions, with mergers and demergers labelled;
   - a checked AI note.

   Known caveat: after demergers, the portal's split of cost can be uneven (e.g. Jio Financial shows almost no cost),
   so those returns are shown as "cost not recorded".
3b. **Done (2026-09-27):** the Telegram report reformatted for readability — bold section headers and key figures
   (Telegram HTML), every card its own visual block, a numbered login progress ("Step 3/5 — ..."), and every run
   ending with a "👉 Next:" line stating what (if anything) the user must do. Every stock/sector/AI-note string is
   HTML-escaped before sending (a name with "&", e.g. "Balmer Lawrie & Company Ltd", must never break the message);
   a regression test covers this. `report.py`'s `sections()` builds the formatted (HTML) version for Telegram;
   `graph.format_report()` strips the tags back to plain text for the file saved to disk.
4. Daily market facts without login: ISIN → NSE symbol (Yahoo), then trend, 52-week drawdown and 1-year return vs
   Nifty. A Telegram alert only when a flag appears or clears.
5. Weekly report: Layers 1 and 2, plus the AI note (explains only; sees symbols and percentages, never amounts or IDs).
6. Layer 3 once the gate passes: the bot's view on each holding, and its top new candidates, always with its live record.
7. Later, maybe: the Equity section opens a separate trading terminal (where orders are placed). Reading positions
   and trade history from it needs its own guarded, read-only design. It is not touched until then.
