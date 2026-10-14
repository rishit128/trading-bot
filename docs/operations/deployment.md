# Deployment (AWS) and rollback

One EC2 instance in Mumbai (`ap-south-1`) runs two containers from **one** image (`ai-trading-bot:aws`): `trading-bot` (the
swing bot) and `intraday-bot` (the three intraday paper strategies). Production is deployed by hand from a checkout; nothing
deploys automatically, and a CI run never touches the server.

## Before you deploy (every time)

1. `git fetch` and compare with `origin/main`: another computer also commits and deploys, and a deploy from an old checkout
   would **overwrite newer code on the server**.
2. The quality gates pass on the commit you are deploying: `ruff check`, `ruff format --check`, `mypy`, `pytest`, `bandit`,
   `pip-audit` ([standards](../development/standards.md); CI runs the same).
3. If the change touches a database table, run it against a **copy** of production first (how: below).
4. Know the rollback commit: the commit currently running. `deploy.sh` writes it to `~/ai-trading-bot/.deployed-commit` (`ssh ... "cat ai-trading-bot/.deployed-commit"`); that file exists after the first deploy made with this script version, and until then the last known production commit is `787ef3d` (verified by comparing every source file on the server on 2026-10-15).

## Deploy

```
bash deploy/deploy.sh > deploy.log 2>&1          # packages this checkout, uploads it, rebuilds, restarts (about 5 minutes)
```

What it does that matters:

- **The server's `.env` is never overwritten.** The script lists, by name only, which settings differ from your local `.env`
  (different value / only here / only on the server). Edit the server's file on the server if a setting should change. A
  test (`tests/test_deployment.py`) fails if the script starts shipping `.env` again.
- It clears the old code folders before extracting, so a deleted module cannot linger.
- It builds the image **once** and both containers use it, then installs the 15-minute watchdog cron job.
- Data lives in Docker volumes (`bot_data`, `portfolio_data`), which a rebuild does not touch.

## After you deploy: verify, do not assume

```
ssh -i deploy/ai-trading-bot-key.pem ubuntu@35.154.232.64
cd ai-trading-bot
sudo docker compose -f docker-compose.aws.yml ps               # both "Up", and "(healthy)" once the start period (3 min) has passed
sudo docker logs --since 5m ai-trading-bot-intraday-bot-1      # the banner, then "intraday/gap-go/gap-only cycle: ..." lines
sudo docker logs --since 5m ai-trading-bot-trading-bot-1       # five "OK" startup checks, then "market closed; waiting" or cycle results
grep -E '^(MIN|MAX)_POSITION_PCT' .env                         # the live sizing is what you expect
```

Then send `/status` and `/gapgo` on Telegram. A new database migration runs at startup; its effect is visible in the log as
absent warnings (a skipped index logs `the unique index is NOT created`).

## Rollback

Code is in the image and in git; data is in volumes and in nightly disk snapshots. To go back to a known commit:

```
git checkout <previous-commit>          # the commit that was running before
bash deploy/deploy.sh                   # rebuilds and restarts that code; the server's .env and data stay
```

- **Schema changes are additive** (new nullable columns, new unique indexes), so older code runs against a newer database:
  rolling back code does not require rolling back data.
- **A bad data change** is the one thing code rollback cannot fix. Disk snapshots are taken nightly at 00:00 IST and the last 7
  are kept (AWS Data Lifecycle Manager, set up by `deploy.sh`). To restore: create a volume from the snapshot, attach it, and
  copy the `.db` file back while the containers are stopped (`docker compose down` first).
- To stop trading immediately without redeploying: send `/pause` on Telegram, or `docker compose down` (history is kept).

## Testing a database change on a copy of production

```
# on the server: a consistent read-only copy of each database (sqlite backup API), never the live file
sudo docker exec ai-trading-bot-intraday-bot-1 python -c "import sqlite3; s=sqlite3.connect('file:/data/trading.db?mode=ro', uri=True); d=sqlite3.connect('/tmp/trading.db'); s.backup(d)"
# copy it down, run the new code's make_session_factory() on the copy, and compare row counts and a content hash before/after
```

This is how the unique-index migration was checked (identical hashes on all four databases) before it was deployed.

## Access and secrets

- SSH is allowed only from addresses added to the instance's security group (`sg-0ad3f31c56e63e802`); when your connection
  times out, your IP changed: add the new one (`aws ec2 authorize-security-group-ingress ... --port 22 --cidr <ip>/32`).
- The Integrated login lives in AWS Secrets Manager (one secret, one IAM role scoped to it); the OpenRouter key and Telegram
  token live only in the server's `.env`. Nothing secret is in the image or in git.
- Known risk, accepted by the owner: the AWS account is used with its root credentials. A scoped IAM user is the safer setup.
