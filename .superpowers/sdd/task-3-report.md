# Task 3 Report: 账户与 JWT 鉴权（accounts）

## Status: DONE

## TDD Evidence

### RED phase (before implementation):
```
FF                                                                       [100%]
FAILED accounts/tests/test_auth.py::test_register_then_login - assert 404 == 201
FAILED accounts/tests/test_auth.py::test_login_wrong_password - assert 404 == 401
2 failed in 0.25s
```
Both tests returned 404 because routes didn't exist yet -- expected behavior.

### GREEN phase (after implementation):
```
..                                                                       [100%]
2 passed in 1.19s
```
Both tests pass: register returns 201, login returns 200 with access token, wrong password returns 401.

## Files Changed

**Created (7):**
- `backend/accounts/__init__.py` -- empty package marker
- `backend/accounts/apps.py` -- Django AppConfig for accounts
- `backend/accounts/models.py` -- minimal (import only, Django requirement)
- `backend/accounts/serializers.py` -- RegisterSerializer with write-only password, min_length 8
- `backend/accounts/views.py` -- RegisterView (CreateAPIView, allow any)
- `backend/accounts/urls.py` -- routes: register, login, refresh
- `backend/accounts/tests/__init__.py` -- empty package marker
- `backend/accounts/tests/test_auth.py` -- test_register_then_login, test_login_wrong_password
- `backend/pytest.ini` -- pytest config (DJANGO_SETTINGS_MODULE, asyncio_mode)
- `backend/conftest.py` -- api fixture (APIClient with db)

**Modified (2):**
- `backend/housafe/settings.py` -- appended INSTALLED_APPS (rest_framework_simplejwt, accounts), added REST_FRAMEWORK config (JWT auth, IsAuthenticated default)
- `backend/housafe/urls.py` -- appended urlpatterns with api/auth/ include

## Self-Review

- [x] Tests use `pytestmark = pytest.mark.django_db` correctly
- [x] `rest_framework_simplejwt` and `accounts` appended to existing INSTALLED_APPS (not replacing)
- [x] urls.py uses `urlpatterns += [...]` pattern (not replacing existing)
- [x] Permission on RegisterView is AllowAny (public registration)
- [x] Login uses TokenObtainPairView (returns access + refresh)
- [x] Password write_only=True, min_length=8 enforced
- [x] Models.py exists (Django app requirement)
- [x] Commit message follows spec: `feat: accounts with jwt auth (register/login/refresh)`

## Concerns

- Tests run with SQLite (DATABASE_URL override) because PostgreSQL/Docker wasn't available in this environment. In production/CI, ensure PostgreSQL is running.
