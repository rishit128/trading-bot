# Development standards

Python 3.13 (the production image and CI use it). All settings for the tools live in `pyproject.toml`.

## The checks (run these before committing)

```
python -m ruff check .            # lint: pycodestyle, pyflakes, import order, bugbear
python -m ruff format .           # formatting (CI runs `--check`)
python -m mypy                    # types for src/ and main.py
python -m pytest                  # ~900 offline tests, a few minutes
python -m bandit -r src main.py -q
python -m pip_audit -r requirements.txt --no-deps
```

CI (`.github/workflows/ci.yml`) runs all of them plus a build of the production image, and **fails the run if any fails**:
`quality` (ruff, format, mypy), `test` (pytest with a coverage gate of 90%; measured 93%), `security` (bandit, pip-audit),
`docker` (the image builds). Do not silence a check to get a green run: fix the cause, or write down why a rule does not
apply (a `noqa` or `nosec` carries its reason on the line above).

Install the tools with `pip install -r requirements-dev.txt`. `git config blame.ignoreRevsFile .git-blame-ignore-revs` makes
`git blame` skip the one mechanical formatting commit.

## Code

- **Types.** Public functions, protocols and module boundaries are annotated; mypy is deliberately not `strict` (the aim is
  honest interfaces, Optional handling and return types, not annotating every local). Keep `mypy` at zero errors.
- **Structured data.** Use dataclasses or pydantic models for anything that crosses a module boundary (a `Setup`, a
  `Snapshot`, a `Decision`), not bare dicts or tuples.
- **Configuration.** Every setting is read once in `config.py` into `Settings` and documented in `.env.example`
  (`tests/test_env_example.py` fails if they drift). The exceptions are on purpose: the `portfolio` package reads its own
  (it must import nothing from the trading code), and the database paths live together in `app/wiring.py`.
- **Magic numbers.** A threshold with a business meaning is a named constant with a comment saying where the number came from
  (`GAP_MIN`, `HUMILITY`, `MAX_FEE_DRAG_PCT`). Trading thresholds are changed only through a documented, tested decision in
  `STRATEGY.md`, never as part of a cleanup.
- **Errors.** Catch the specific exception. A broad `except Exception` is allowed only at a loop boundary that must not die
  (a cycle, a Telegram poll, an AI call) and it must log or notify; best-effort cleanup logs at debug. A swallowed error that
  hides a failure is a bug. Re-raise with `from e` so the cause survives.
- **Logging.** `logging.getLogger(__name__)`; never log a secret. The Telegram client silences `httpx` logging because the bot
  token is part of every URL.
- **Time and randomness.** Anything that depends on the clock takes it as a parameter (`now_fn`, `clock`, `sleep`) so tests
  control it. No hidden global state.
- **Dead code.** Delete it; git remembers. `python -m vulture main.py src scripts --min-confidence 61` is the quick scan.
- **Comments** explain why, in particular the financial assumptions, not what the line does.

## Commits

Small, focused and reversible; one concern each. Mechanical changes (formatting) get their own commit, listed in
`.git-blame-ignore-revs`. Behaviour changes to trading rules are never mixed into refactors: they are proposed, tested
(`STRATEGY.md` records the pre-declared pass/fail rule and the result), and approved separately.
