# Task 2: 事件契约包 (contracts) - Report

## What was implemented

Created the `contracts/` package as the single source of truth for Tier-1 perception event models:

- `contracts/pyproject.toml` -- package definition for `housafe-contracts` v0.1.0
- `contracts/housafe_contracts/__init__.py` -- empty init
- `contracts/housafe_contracts/events.py` -- event models and dispatcher
  - `POSTURES = ("stand","sit","lie","walk","fall")`
  - `EVENT_KINDS = ("presence","posture","vital","occupancy","heartbeat")`
  - `_Base` -- shared fields: ts, radar_id, room, seq
  - `PresenceEvent`, `PostureEvent`, `VitalEvent`, `OccupancyEvent` (all extend `_Base`)
  - `Heartbeat` (extends `BaseModel` directly -- no room/seq)
  - `parse_event(kind, payload)` dispatches by kind string
- `contracts/tests/test_events.py` -- 4 tests
- `backend/pyproject.toml` -- added `"housafe-contracts"` to dependencies

## TDD Evidence

### RED phase

Command: `cd contracts && pytest -q`

```
ERROR tests/test_events.py
ModuleNotFoundError: No module named 'housafe_contracts.events'
1 error in 0.26s
```

### GREEN phase

Command: `cd contracts && pytest -q`

```
....
4 passed in 0.09s
```

## Files changed

- `contracts/pyproject.toml` (created)
- `contracts/housafe_contracts/__init__.py` (created)
- `contracts/housafe_contracts/events.py` (created)
- `contracts/tests/__init__.py` (created)
- `contracts/tests/test_events.py` (created)
- `backend/pyproject.toml` (modified)

## Self-review findings

- `Literal["stand","sit","lie","walk","fall"]` used explicitly (not `Literal[POSTURES]`) as recommended in the brief to avoid Python type-checker issues; `POSTURES` retained as runtime constant
- `Heartbeat` correctly does NOT extend `_Base` -- has no `room` or `seq` fields, only `ts`, `radar_id`, `status`, `fw_version`
- `parse_event` raises `ValueError` for unknown kinds (matches test expectation), while Pydantic `ValidationError` surfaces naturally via model construction
- All 4 tests pass with standard Python 3.12 + pydantic 2.x

## Concerns

None. The contract package is minimal, well-typed, and ready to be imported by backend/simulator/AI modules.
