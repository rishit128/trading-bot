# 0001: Keep the current package layout instead of `domain / application / infrastructure`

**Status:** accepted, 2026-10-15

## Context

An architecture review asked whether the code should move to a layered `domain/`, `application/`, `infrastructure/`
structure. The measured state of the code:

- 86 modules, about 10,000 lines, one process type, **one** broker implementation (the paper simulator).
- The packages already form a layered, acyclic graph ([dependency map](../dependency-map.md)), and ten layering rules are
  enforced by `tests/test_layering.py` (the engine is a leaf, production never imports research, no cycles, and so on).
- 93% test coverage, including golden tests of whole trading cycles (`tests/test_characterization.py`).

## Decision

Keep the packages as they are. Fix concrete problems found in the audit instead (misplaced fee schedule, a 317-line
`main.py`, an if-chain of Telegram commands, missing database constraints, no health checks, a deploy that overwrote the
server's settings).

## Consequences

- No large rename: `git blame` and history stay useful, and the golden tests keep protecting behaviour.
- The boundaries that matter are already tested, and a new one is a new test in `test_layering.py`.
- Revisit if a second broker is added (a `Broker` protocol already exists in `engine/ports.py`), or a second deployable
  service appears: that is the point where a real infrastructure boundary would exist.
