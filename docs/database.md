# Production database architecture

## The two accounts

The bot keeps **two independent ledgers** so results never mix:

| Account | Environment variable | Default | Lives in |
| --- | --- | --- | --- |
| Swing (daily decisions + the paper account driving them) | `DATABASE_URL` | `sqlite:///trading.db` | `decisions`, `orders`, `paper_*`, `equity_history`, `holdout_marks`, `control_flags` |
| Intraday breakout engine | `INTRADAY_DATABASE_URL` | `sqlite:///intraday.db` | its own copy of the same tables, own starting cash |

Every consumer reaches the database through `make_session_factory(database_url)`: it creates the
engine, applies migrations idempotently (create tables, add nullable columns, add indexes), enables
WAL for SQLite, and returns a session factory. **No code writes raw DDL at runtime.**

## What the tables are

- `decisions` — the audit log: one row per analysed stock per cycle, every agent's signal, the final
  call, the risk verdict, the P0 version stamps (`prompt_version`, `feature_version`,
  `strategy_version`), the per-decision universe snapshot (`universe_size`). Append-only, never edited.
- `orders` — each order attempt/skip and the broker's reported fill. Also append-only.
- `paper_account` / `paper_positions` / `paper_trades` — the simulated broker's state: the single
  account, open positions with their stop/target levels, and closed trades with fees (`STOP` /
  `TARGET` / `SIGNAL` exits).
- `equity_history` — account equity at the start of each cycle (drawdown-limit source).
- `holdout_marks` — the independent holdout reserve's valuation history (see `docs/holdout.md`).
- `control_flags` — runtime switches (`/pause`, etc.).

## Migration order

Every upgrade runs the same three steps, in this order:

1. **Create** tables that do not exist (`Base.metadata.create_all`).
2. **Add missing columns** — *nullable only*. Adding a schema-breaking NOT NULL column raises
   `RuntimeError` instead of being guessed.
3. **Add missing indexes** — `IF NOT EXISTS`.

The steps are idempotent, so the bot's startup path can (and does) run them safely against any
existing database.

## The sanctioned tool

`scripts/migrate_db.py` gives the schema an explicit, rootable path:

```
python scripts/migrate_db.py                  # dry run: prints what is out of date, changes nothing
python scripts/migrate_db.py --check          # exit 0 iff the database is current (no changes)
python scripts/migrate_db.py --apply          # applies the idempotent, nullable-only migration
python scripts/migrate_db.py --database-url postgresql://…  # any URL; SQLite and Postgres both work
```

## Live gates

- **Migrations are a real action**: the default is a dry run; `--apply` is required to change anything.
- **Nothing is invented**: a new NOT NULL column is a manual, schema-breaking migration to a chosen
  subset of `Base.metadata`, never an automatic ALTER.
- **Read-only paths stay read-only**: reconciliation, calibration, holdout and the analysis scripts
  never write to `decisions`/`paper_*`.

## Shipped versus production

`make_session_factory` uses `create_engine(database_url)` with `pool_pre_ping` and (SQLite) WAL, so
`DATABASE_URL` is already how a managed Postgres would be attached — change the variable, re-run
`scripts/migrate_db.py --check`, and the SQLite-only WAL block is skipped automatically. Before the
first live-money run:

1. Point both `DATABASE_URL` and `INTRADAY_DATABASE_URL` at real backups (a durable, daily-backup
   store; SQLite files are single-file — copy them under the same journal-mode guard).
2. `scripts/migrate_db.py --check` on each.
3. Exercise the upgrade once against a restore (not the live copy) before the first deploy.