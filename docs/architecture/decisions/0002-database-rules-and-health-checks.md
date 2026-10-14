# 0002: The database enforces the rules the code relies on; health means "working", not "running"

**Status:** accepted, 2026-10-15

## Context

- Uniqueness was enforced only in application code (`_done_today`, `remembered_signals`, one position row per symbol).
  That is safe for one process and undefended against a second writer; a second Telegram poller on the same bot token had
  already happened once.
- Production data was checked read-only: 10 orders for 10 decisions, 60 intraday signals over 60 distinct (day, stock)
  pairs. The `decisions` table has 156 repeated (stock, bar, prompt, feature) keys from before decisions were reused per bar.
- A container that is "running" can be stuck, or failing every cycle (Yahoo down, a bug).

## Decision

1. Unique indexes `ux_orders_decision` (one order per decision) and `ux_intraday_signals_day_symbol` (one signal per stock per
   session), declared in the models and added to older databases by an additive migration. If an older database already
   breaks a rule, that index is skipped with a warning: nothing is deleted or rewritten, and the bot still starts.
2. **No** uniqueness rule on `decisions`: the existing data makes it wrong.
3. Each loop writes a heartbeat file with its latest cycle outcome; the container health check calls the bot unhealthy when
   the file is stale or the last three cycles failed.

## Consequences

- A repeated event cannot create a second order or a second recorded signal, even from another process.
- The migration was run against copies of all four production databases first: identical row counts and content hashes,
  and a second start changes nothing.
- Foreign keys are not added: SQLite would need a table rebuild for existing tables, which is not worth the risk here.
