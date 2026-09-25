"""P0, prod-DB: the one sanctioned way to touch a database schema.

    python scripts/migrate_db.py                       # dry run: print what is out of date
    python scripts/migrate_db.py --check               # exit 0 only if the database is current
    python scripts/migrate_db.py --apply               # apply the nullable-only, idempotent migration
    python scripts/migrate_db.py --database-url postgresql://... --apply

Migrations are a real action, never silent: default is a dry run, `--apply` is required for change and
the NOT-NULL guard means a schema-breaking column addition raises instead of being invented. The live
databases (swing = $DATABASE_URL, intraday = $INTRADAY_DATABASE_URL) are upgraded this way, then the
bot's normal startup path sees a current schema and does nothing extra."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import apply_migrations, is_current, schema_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=None, help="defaults to $DATABASE_URL or sqlite:///trading.db")
    ap.add_argument("--apply", action="store_true", help="apply the migration (default is a dry run)")
    ap.add_argument("--check", action="store_true", help="exit 0 only if the schema is current; makes no changes")
    args = ap.parse_args()

    from src.config import load_settings

    url = args.database_url or load_settings().database_url
    if args.check:
        ok = is_current(url)
        print(f"{'CURRENT' if ok else 'OUT OF DATE'}: {url}")
        sys.exit(0 if ok else 1)

    if args.apply:
        report = apply_migrations(url, dry_run=False)
        lines = report["columns_added"]
        print(f"applied: {len(lines)} table(s) gained columns")
        for table, cols in lines:
            print(f"  {table}: {', '.join(cols) or '(indexes only)'}")
        print("done, current" if report["ok"] else "apply did not converge; inspect the database")
        sys.exit(0 if report["ok"] else 1)

    status = schema_status(url)
    print(f"schema report for {url}")
    for table in sorted(status):
        info = status[table]
        if info["missing"]:
            print(f"  {table:<16} MISSING {', '.join(info['missing'])}")
        else:
            print(f"  {table:<16} current ({len(info['existing'])} columns)")
    if not status:
        print("  (no tables exist yet; the bot's startup create_all will build them)")
    print("dry run; use --apply to apply, --check to assert current")


if __name__ == "__main__":
    main()