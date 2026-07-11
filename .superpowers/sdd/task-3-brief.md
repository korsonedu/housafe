### Task 3: 账户与 JWT 鉴权（accounts）

**Files:**
- Create: `backend/accounts/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`
- Modify: `backend/housafe/settings.py`（加 app、DRF、simplejwt），`backend/housafe/urls.py`
- Test: `backend/accounts/tests/test_auth.py`
- Create: `backend/conftest.py`, `backend/pytest.ini`

**Interfaces:**
- Consumes: 无
- Produces: REST `POST /api/auth/register`、`POST /api/auth/login`（返回 `{access, refresh}`）、`POST /api/auth/refresh`；用 Django 默认 `User`。

- [ ] **Step 1: 配置 pytest**

`backend/pytest.ini`:
```ini
[pytest]
DJANGO_SETTINGS_MODULE = housafe.settings
python_files = test_*.py
asyncio_mode = auto
```
`backend/conftest.py`:
```python
import pytest
@pytest.fixture
def api(db):
    from rest_framework.test import APIClient
    return APIClient()
```

- [ ] **Step 2: 写失败测试**

`backend/accounts/tests/test_auth.py`:
```python
import pytest
pytestmark = pytest.mark.django_db

def test_register_then_login(api):
    r = api.post("/api/auth/register", {"username":"kid","password":"pw12345678"}, format="json")
    assert r.status_code == 201
    r = api.post("/api/auth/login", {"username":"kid","password":"pw12345678"}, format="json")
    assert r.status_code == 200 and "access" in r.data

def test_login_wrong_password(api):
    api.post("/api/auth/register", {"username":"kid","password":"pw12345678"}, format="json")
    r = api.post("/api/auth/login", {"username":"kid","password":"wrong"}, format="json")
    assert r.status_code == 401
```

- [ ] **Step 3: 运行确认失败**

Run: `cd backend && pytest accounts -q`
Expected: FAIL（404，路由未建）。

- [ ] **Step 4: 实现 accounts**

`backend/accounts/serializers.py`:
```python
from django.contrib.auth.models import User
from rest_framework import serializers

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username","password"]
        extra_kwargs = {"password":{"write_only":True,"min_length":8}}
    def create(self, data):
        return User.objects.create_user(**data)
```
`backend/accounts/views.py`:
```python
from rest_framework import generics, permissions
from .serializers import RegisterSerializer

class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]
```
`backend/accounts/urls.py`:
```python
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import RegisterView
urlpatterns = [
    path("register", RegisterView.as_view()),
    path("login", TokenObtainPairView.as_view()),
    path("refresh", TokenRefreshView.as_view()),
]
```
`backend/accounts/apps.py`:
```python
from django.apps import AppConfig
class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "accounts"
```
settings.py 增补：
```python
INSTALLED_APPS += ["rest_framework_simplejwt","accounts"]
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework_simplejwt.authentication.JWTAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
}
```
`housafe/urls.py` 增补：
```python
from django.urls import path, include
urlpatterns += [path("api/auth/", include("accounts.urls"))]
```

- [ ] **Step 5: 运行确认通过并提交**

Run: `cd backend && pytest accounts -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: accounts with jwt auth (register/login/refresh)"
```

## Global Constraints (relevant)

- Python ≥ 3.12；Django ≥ 5.0。
- 严守 PRD「独居假设 + 无身份识别」：数据模型不得出现"识别是谁"的字段。（此任务用 Django 默认 User，符合要求）
- 每个任务结束必须 commit；提交信息用 `feat:` / `test:` / `chore:` 前缀。
