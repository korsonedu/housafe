# Task 8 Report: 实时推送 WS（realtime）

## Status: DONE

## TDD Evidence

**Step 1: Write test** -- Created `realtime/tests/test_realtime.py` with the `test_ingest_to_app_flow` end-to-end test. The test exercises the full chain: ingest pushes an event via `group_send`, channel layer delivers it, realtime consumer receives `event.push` and forwards to the app client.

**Step 2: Test FAIL** -- Not explicitly captured (implementation written in same pass), but the test exercises a route that didn't exist before implementation. After implementation, verified the test passes with in-memory channel layer + SQLite.

**Step 3: Implement** -- Created all files:
- `realtime/__init__.py` (empty)
- `realtime/apps.py` (standard Django app config)
- `realtime/consumers.py` (AppConsumer with JWT auth, family ownership check, event_push forwarding)
- `realtime/routing.py` (path: `ws/app`)
- `realtime/tests/__init__.py` (empty)
- `realtime/tests/test_realtime.py` (end-to-end test)
- Modified `housafe/asgi.py` (added `import realtime.routing`, merged URLRouter)
- Modified `housafe/settings.py` (added `"realtime"` to INSTALLED_APPS)

**Step 4: Test PASS** -- `pytest realtime -q` passes (1 passed in 0.37s)

## Test command

```
DATABASE_URL=sqlite:///tmp/housafe_test.sqlite3 pytest realtime -q
```

Note: PostgreSQL was not available in the local environment (Docker daemon down, colima startup hung). Used SQLite with in-memory channel layer for testing. The same test should run against PostgreSQL when the full stack is available.

## Coupling Check

| Check | Result |
|-------|--------|
| Import from `ingest.` | None found -- communication is via channel_layer groups only |
| Import from `rest_framework_simplejwt.tokens` | Allowed (lazy import inside `_auth`) |
| Import from `django.contrib.auth.models` | Allowed (lazy import inside `_auth`) |
| Import from `families.models` | Allowed (lazy import inside `_owns`, FK ownership query per contract) |
| Import from `channels.generic.websocket` | Standard Channels dependency |
| Import from `urllib.parse` | Standard library |

All module boundaries respected. realtime is a pure orchestration layer -- it composes JWT validation, family ownership check, and channel_layer groups without crossing into ingest internals.

## Files Changed

| File | Action |
|------|--------|
| `realtime/__init__.py` | Created (empty) |
| `realtime/apps.py` | Created (standard Django AppConfig) |
| `realtime/consumers.py` | Created (AppConsumer: JWT auth + family check + event_push) |
| `realtime/routing.py` | Created (URL: `ws/app`) |
| `realtime/tests/__init__.py` | Created (empty) |
| `realtime/tests/test_realtime.py` | Created (end-to-end test) |
| `housafe/asgi.py` | Modified (added realtime routing to URLRouter) |
| `housafe/settings.py` | Modified (added `"realtime"` to INSTALLED_APPS) |

## Self-Review

- [x] JWT authentication works (test passes)
- [x] Family ownership check works (test passes)
- [x] event_push forwards from ingest via channel layer (test passes)
- [x] Close code 4401 on auth/ownership failure
- [x] `hasattr(self, "group")` guard in disconnect
- [x] No ingest imports (pure channel_layer communication)
- [x] `self.accept()` only called after all checks pass
- [x] Query string decoded from bytes before parsing
- [x] Empty `__init__.py` files in realtime and realtime/tests
- [x] Commit message matches required format
