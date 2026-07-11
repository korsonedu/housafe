# 家安 housafe 「地基」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭出家安产品的软件地基——事件契约 + Django 后端骨架（账户/家庭/设备/接入/时序存储/实时推送）+ 最小 RN App 壳，用仿真事件端到端跑通"手机实时看到老人状态"。

**Architecture:** 仿真接缝定在「事件契约」层（Tier-1 感知事件 = "完美 ALG-0"输出），仿真雷达/真实雷达对系统可插拔。设备经 Channels WS 灌事件 → 校验契约 → 落 Timescale → 经 Redis 分发 → Channels WS 实时推到 App。ALG-0/ALG-1~6/通知/完整 App 均不在本计划，留占位接口。

**Tech Stack:** Python 3.12 / Django 5.x / DRF / Channels 4 / channels-redis / djangorestframework-simplejwt / Pydantic 2 / psycopg 3 / PostgreSQL 16 + TimescaleDB / Redis 7 / Node 20 / Expo SDK 51 (React Native).

## Global Constraints

- Python ≥ 3.12；Django ≥ 5.0；Pydantic ≥ 2.6；Channels ≥ 4.0。
- 事件时间轴一律用事件自带的 `ts`（UTC 毫秒 int），服务器**绝不**用"当前时间"覆盖；服务器接收时间单独存 `ts_recv`。
- 事件契约唯一来源是 `contracts/` 包，backend / simulator / ai 三方 import 同一份，禁止各自复制字段定义。
- 姿态枚举固定为 `stand | sit | lie | walk | fall`；房间为自由字符串 label。
- 严守 PRD「独居假设 + 无身份识别」：数据模型不得出现"识别是谁"的字段。
- 每个任务结束必须 commit；提交信息用 `feat:` / `test:` / `chore:` 前缀。
- 测试框架：后端用 `pytest` + `pytest-django` + `pytest-asyncio`；契约用 `pytest`。
- 所有 WS 鉴权失败必须关闭连接并返回明确 close code，不得静默接受。

## 模块解耦契约（每个任务 review 必查）

每个模块单一职责、只通过下列接口互相调用，**禁止跨模块直接 reach-in**（不得跨 app import 对方内部实现、不得跨 app 直接查对方的表）。

| 模块 | 单一职责 | 允许依赖 | 对外暴露的唯一接口 |
|------|---------|---------|-------------------|
| `contracts/` | 事件字段定义与校验 | 无（纯 Pydantic） | `parse_event()` / 事件模型类 |
| `accounts` | 身份与鉴权 | 无 | JWT token；`request.user` |
| `families` | 家庭/老人/联系人 | accounts（仅 `request.user`） | 模型 + REST |
| `devices` | 设备身份与生命周期 | families（仅 FK） | `RadarDevice.verify()` / REST |
| `events` | 时序落库与查询 | contracts、devices（仅 device_id 字符串） | `store_event()` / 查询 REST |
| `ingest` | 收设备事件的编排层 | contracts、devices、events、channel_layer | WS `ws/ingest` |
| `realtime` | 推事件给 App 的编排层 | accounts、families、channel_layer | WS `ws/app` |
| `ai/`（占位） | ALG-1~6 | 只经 channel_layer/Redis 订阅事件 | 子项目③ |

**判据：** ingest / realtime 是仅有的两个「编排层」，允许组合调用下游接口；其余模块之间不得互相 import 内部函数或直接读对方的表。任何任务若出现跨模块 reach-in，review 打回。

---

### Task 1: 项目脚手架与本地基础设施

**Files:**
- Create: `.gitignore`
- Create: `docker-compose.yml`
- Create: `backend/pyproject.toml`
- Create: `backend/housafe/__init__.py`, `backend/housafe/settings.py`, `backend/housafe/urls.py`, `backend/housafe/asgi.py`, `backend/manage.py`
- Create: `backend/.env.example`
- Create: `README.md`

**Interfaces:**
- Produces: 可运行的 Django ASGI 项目 `housafe`；Postgres(Timescale)/Redis 本地服务；settings 读取 `DATABASE_URL`、`REDIS_URL`。

- [ ] **Step 1: 初始化 git 与忽略文件**

```bash
cd /Users/eular/Desktop/housafe
git init
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
.venv/
node_modules/
.expo/
*.sqlite3
.DS_Store
```

- [ ] **Step 2: docker-compose 起 Timescale + Redis**

`docker-compose.yml`:
```yaml
services:
  db:
    image: timescale/timescaledb:2.15.0-pg16
    environment:
      POSTGRES_DB: housafe
      POSTGRES_USER: housafe
      POSTGRES_PASSWORD: housafe
    ports: ["5432:5432"]
    volumes: ["dbdata:/var/lib/postgresql/data"]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
volumes:
  dbdata:
```

Run: `docker-compose up -d` → Expected: `db` 与 `redis` 容器 healthy。

- [ ] **Step 3: 建 Python 环境与依赖**

`backend/pyproject.toml`:
```toml
[project]
name = "housafe-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "django>=5.0",
  "djangorestframework>=3.15",
  "djangorestframework-simplejwt>=5.3",
  "channels>=4.0",
  "channels-redis>=4.2",
  "daphne>=4.1",
  "psycopg[binary]>=3.1",
  "pydantic>=2.6",
  "dj-database-url>=2.1",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-django>=4.8", "pytest-asyncio>=0.23"]
```

```bash
cd backend && python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 4: 生成 Django 项目骨架**

```bash
cd backend && django-admin startproject housafe . --name asgi.py
```

编辑 `housafe/settings.py` 关键项：
```python
import os, dj_database_url
from dotenv import load_dotenv
load_dotenv()

INSTALLED_APPS += ["rest_framework", "channels"]
ASGI_APPLICATION = "housafe.asgi.application"
DATABASES = {"default": dj_database_url.parse(
    os.environ.get("DATABASE_URL", "postgres://housafe:housafe@localhost:5432/housafe"))}
CHANNEL_LAYERS = {"default": {
    "BACKEND": "channels_redis.core.RedisChannelLayer",
    "CONFIG": {"hosts": [os.environ.get("REDIS_URL", "redis://localhost:6379/0")]}}}
```

`.env.example`:
```
DATABASE_URL=postgres://housafe:housafe@localhost:5432/housafe
REDIS_URL=redis://localhost:6379/0
```

- [ ] **Step 5: 验证启动并提交**

Run: `python manage.py migrate && python manage.py runserver`
Expected: 无报错，`http://127.0.0.1:8000/` 返回 Django 欢迎页。

```bash
git add -A && git commit -m "chore: scaffold monorepo, django asgi project, docker infra"
```

---

### Task 2: 事件契约包（contracts）

**Files:**
- Create: `contracts/pyproject.toml`, `contracts/housafe_contracts/__init__.py`
- Create: `contracts/housafe_contracts/events.py`
- Test: `contracts/tests/test_events.py`

**Interfaces:**
- Produces:
  - `POSTURES = ("stand","sit","lie","walk","fall")`
  - Pydantic 模型：`PresenceEvent`、`PostureEvent`、`VitalEvent`、`OccupancyEvent`、`Heartbeat`
  - 每个事件含 `ts:int`（UTC ms）、`radar_id:str`、`room:str`、`seq:int`（心跳无 seq/room）
  - `parse_event(kind:str, payload:dict) -> BaseModel`：按 `kind` 分发校验，非法抛 `ValidationError`
  - `EVENT_KINDS = ("presence","posture","vital","occupancy","heartbeat")`

- [ ] **Step 1: 写失败测试**

`contracts/tests/test_events.py`:
```python
import pytest
from pydantic import ValidationError
from housafe_contracts.events import parse_event, PostureEvent, POSTURES

def test_posture_valid():
    e = parse_event("posture", {"ts":1700000000000,"radar_id":"r1","room":"bedroom","seq":5,"posture":"walk","confidence":0.9})
    assert isinstance(e, PostureEvent) and e.posture == "walk"

def test_posture_rejects_bad_enum():
    with pytest.raises(ValidationError):
        parse_event("posture", {"ts":1,"radar_id":"r1","room":"b","seq":1,"posture":"jump","confidence":0.5})

def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_event("nope", {})

def test_postures_frozen():
    assert POSTURES == ("stand","sit","lie","walk","fall")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd contracts && pip install -e . && pytest -q`
Expected: FAIL（`housafe_contracts.events` 不存在）。

- [ ] **Step 3: 实现契约**

`contracts/housafe_contracts/events.py`:
```python
from typing import Literal
from pydantic import BaseModel, Field

POSTURES = ("stand","sit","lie","walk","fall")
EVENT_KINDS = ("presence","posture","vital","occupancy","heartbeat")

class _Base(BaseModel):
    ts: int = Field(..., description="UTC ms, source-authoritative")
    radar_id: str
    room: str
    seq: int

class PresenceEvent(_Base):
    presence: bool
    moving: bool

class PostureEvent(_Base):
    posture: Literal[POSTURES]
    confidence: float = Field(..., ge=0, le=1)

class VitalEvent(_Base):
    quiet: bool
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float = Field(..., ge=0, le=1)

class OccupancyEvent(_Base):
    count: int = Field(..., ge=0)

class Heartbeat(BaseModel):
    ts: int
    radar_id: str
    status: Literal["online","offline"]
    fw_version: str

_MODELS = {"presence":PresenceEvent,"posture":PostureEvent,"vital":VitalEvent,
           "occupancy":OccupancyEvent,"heartbeat":Heartbeat}

def parse_event(kind: str, payload: dict) -> BaseModel:
    if kind not in _MODELS:
        raise ValueError(f"unknown event kind: {kind}")
    return _MODELS[kind](**payload)
```

`contracts/pyproject.toml`:
```toml
[project]
name = "housafe-contracts"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.6"]
```

> 注：`Literal[POSTURES]` 需 `Literal[*POSTURES]`（Python 3.12 语法）。若报错，改写为 `Literal["stand","sit","lie","walk","fall"]` 并保留 `POSTURES` 常量做校验。

- [ ] **Step 4: 运行确认通过**

Run: `cd contracts && pytest -q`
Expected: PASS（4 passed）。

- [ ] **Step 5: 后端引用契约并提交**

在 `backend/pyproject.toml` dependencies 加 `"housafe-contracts"`，并 `pip install -e ../contracts`。

```bash
git add -A && git commit -m "feat: add event contract package (tier-1 perception events)"
```

---

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

---

### Task 4: 家庭 / 老人 / 紧急联系人（families）

**Files:**
- Create: `backend/families/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`
- Modify: `backend/housafe/settings.py`（INSTALLED_APPS 加 `families`），`backend/housafe/urls.py`
- Test: `backend/families/tests/test_families.py`

**Interfaces:**
- Consumes: `accounts` 的 JWT 用户
- Produces:
  - 模型 `Family(id, name, owner→User)`、`Membership(user, family, role)`、`Elder(id, family, name, note)`、`Contact(id, family, name, phone, order)`
  - REST（均需鉴权，仅本人所属家庭可见）：`GET/POST /api/families`、`GET/PATCH/DELETE /api/families/{id}`、`GET/POST /api/families/{id}/elders`、`GET/POST /api/families/{id}/contacts`（contacts 按 `order` 升序）

- [ ] **Step 1: 写失败测试**

`backend/families/tests/test_families.py`:
```python
import pytest
pytestmark = pytest.mark.django_db

def auth(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")

def test_create_family_and_list_own_only(api):
    auth(api)
    r = api.post("/api/families", {"name":"我家"}, format="json")
    assert r.status_code == 201
    fid = r.data["id"]
    r = api.get("/api/families")
    assert r.status_code == 200 and len(r.data) == 1
    r = api.post(f"/api/families/{fid}/contacts", {"name":"儿子","phone":"13800000000","order":1}, format="json")
    assert r.status_code == 201

def test_family_requires_auth(api):
    assert api.get("/api/families").status_code == 401
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest families -q`
Expected: FAIL（404/路由缺失）。

- [ ] **Step 3: 实现 models**

`backend/families/models.py`:
```python
from django.conf import settings
from django.db import models

class Family(models.Model):
    name = models.CharField(max_length=100)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="owned_families")

class Membership(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="members")
    role = models.CharField(max_length=20, default="member")

class Elder(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="elders")
    name = models.CharField(max_length=50)
    note = models.CharField(max_length=200, blank=True)

class Contact(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="contacts")
    name = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    order = models.PositiveIntegerField(default=0)
    class Meta:
        ordering = ["order"]
```

- [ ] **Step 4: 实现 serializers / views / urls**

`backend/families/serializers.py`:
```python
from rest_framework import serializers
from .models import Family, Elder, Contact

class FamilySerializer(serializers.ModelSerializer):
    class Meta: model = Family; fields = ["id","name"]
class ElderSerializer(serializers.ModelSerializer):
    class Meta: model = Elder; fields = ["id","name","note"]
class ContactSerializer(serializers.ModelSerializer):
    class Meta: model = Contact; fields = ["id","name","phone","order"]
```
`backend/families/views.py`:
```python
from rest_framework import viewsets
from .models import Family, Elder, Contact
from .serializers import FamilySerializer, ElderSerializer, ContactSerializer

class FamilyViewSet(viewsets.ModelViewSet):
    serializer_class = FamilySerializer
    def get_queryset(self):
        return Family.objects.filter(owner=self.request.user)
    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

class _ChildViewSet(viewsets.ModelViewSet):
    child_model = None
    def _family(self):
        return Family.objects.get(id=self.kwargs["family_id"], owner=self.request.user)
    def get_queryset(self):
        return self.child_model.objects.filter(family=self._family())
    def perform_create(self, serializer):
        serializer.save(family=self._family())

class ElderViewSet(_ChildViewSet):
    child_model = Elder; serializer_class = ElderSerializer
class ContactViewSet(_ChildViewSet):
    child_model = Contact; serializer_class = ContactSerializer
```
`backend/families/urls.py`:
```python
from django.urls import path
from .views import FamilyViewSet, ElderViewSet, ContactViewSet
fam = FamilyViewSet.as_view({"get":"list","post":"create"})
fam_d = FamilyViewSet.as_view({"get":"retrieve","patch":"partial_update","delete":"destroy"})
elders = ElderViewSet.as_view({"get":"list","post":"create"})
contacts = ContactViewSet.as_view({"get":"list","post":"create"})
urlpatterns = [
    path("families", fam),
    path("families/<int:pk>", fam_d),
    path("families/<int:family_id>/elders", elders),
    path("families/<int:family_id>/contacts", contacts),
]
```
`apps.py` 同 Task 3 模式（name="families"）。settings 加 `"families"`；`housafe/urls.py` 加 `path("api/", include("families.urls"))`。

- [ ] **Step 5: 迁移、测试、提交**

Run: `python manage.py makemigrations families && python manage.py migrate && pytest families -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: families/elders/contacts with per-owner scoping"
```

---

### Task 5: 雷达设备管理（devices）

**Files:**
- Create: `backend/devices/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`
- Modify: settings（加 `devices`），`housafe/urls.py`
- Test: `backend/devices/tests/test_devices.py`

**Interfaces:**
- Consumes: `families.Family`（设备归属家庭）；JWT 用户
- Produces:
  - 模型 `RadarDevice(device_id:str uniq, secret:str, family, room, online:bool, last_heartbeat, fw_version)`
  - `RadarDevice.verify(device_id, secret) -> RadarDevice | None`（classmethod，供 Task 7 ingest 鉴权用）
  - REST：`POST /api/families/{id}/devices/bind`（生成 device_id+secret 返回一次）、`GET /api/families/{id}/devices`、`PATCH /api/devices/{device_id}`（重命名 room）、`GET /api/devices/{device_id}/status`

- [ ] **Step 1: 写失败测试**

`backend/devices/tests/test_devices.py`:
```python
import pytest
pytestmark = pytest.mark.django_db

def auth_and_family(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    return api.post("/api/families", {"name":"我家"}, format="json").data["id"]

def test_bind_returns_credentials_and_lists(api):
    fid = auth_and_family(api)
    r = api.post(f"/api/families/{fid}/devices/bind", {"room":"bedroom"}, format="json")
    assert r.status_code == 201 and r.data["device_id"] and r.data["secret"]
    r = api.get(f"/api/families/{fid}/devices")
    assert len(r.data) == 1 and r.data[0]["room"] == "bedroom"

def test_verify_credentials():
    from devices.models import RadarDevice
    from families.models import Family
    from django.contrib.auth.models import User
    u = User.objects.create_user("k","","pw12345678")
    fam = Family.objects.create(name="f", owner=u)
    d = RadarDevice.objects.create(device_id="d1", secret="s1", family=fam, room="bath")
    assert RadarDevice.verify("d1","s1") == d
    assert RadarDevice.verify("d1","bad") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest devices -q`
Expected: FAIL。

- [ ] **Step 3: 实现 models**

`backend/devices/models.py`:
```python
import secrets
from django.db import models
from families.models import Family

class RadarDevice(models.Model):
    device_id = models.CharField(max_length=40, unique=True)
    secret = models.CharField(max_length=64)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="devices")
    room = models.CharField(max_length=50)
    online = models.BooleanField(default=False)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    fw_version = models.CharField(max_length=20, default="sim-0.1")

    @classmethod
    def issue(cls, family, room):
        return cls.objects.create(
            device_id="rad_"+secrets.token_hex(6),
            secret=secrets.token_hex(16), family=family, room=room)

    @classmethod
    def verify(cls, device_id, secret):
        return cls.objects.filter(device_id=device_id, secret=secret).first()
```

- [ ] **Step 4: serializers / views / urls**

`backend/devices/serializers.py`:
```python
from rest_framework import serializers
from .models import RadarDevice
class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","room","online","last_heartbeat","fw_version"]
class BindResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","secret","room"]
```
`backend/devices/views.py`:
```python
from rest_framework import viewsets, views, response, status
from families.models import Family
from .models import RadarDevice
from .serializers import DeviceSerializer, BindResultSerializer

class DeviceBindView(views.APIView):
    def post(self, request, family_id):
        fam = Family.objects.get(id=family_id, owner=request.user)
        d = RadarDevice.issue(fam, request.data.get("room","room"))
        return response.Response(BindResultSerializer(d).data, status=status.HTTP_201_CREATED)

class DeviceViewSet(viewsets.ModelViewSet):
    serializer_class = DeviceSerializer
    lookup_field = "device_id"
    def get_queryset(self):
        qs = RadarDevice.objects.filter(family__owner=self.request.user)
        fid = self.kwargs.get("family_id")
        return qs.filter(family_id=fid) if fid else qs
```
`backend/devices/urls.py`:
```python
from django.urls import path
from .views import DeviceBindView, DeviceViewSet
dev_list = DeviceViewSet.as_view({"get":"list"})
dev_detail = DeviceViewSet.as_view({"patch":"partial_update","get":"retrieve"})
urlpatterns = [
    path("families/<int:family_id>/devices/bind", DeviceBindView.as_view()),
    path("families/<int:family_id>/devices", dev_list),
    path("devices/<str:device_id>", dev_detail),
    path("devices/<str:device_id>/status", dev_detail),
]
```
settings 加 `"devices"`；`housafe/urls.py` 加 `path("api/", include("devices.urls"))`。

- [ ] **Step 5: 迁移、测试、提交**

Run: `python manage.py makemigrations devices && python manage.py migrate && pytest devices -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: radar device bind/list/rename + credential verify"
```

---

### Task 6: 时序事件存储与查询（events）

**Files:**
- Create: `backend/events/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`, `store.py`
- Create: `backend/events/migrations/0002_hypertables.py`（RunSQL 转 hypertable）
- Modify: settings（加 `events`），`housafe/urls.py`
- Test: `backend/events/tests/test_store.py`, `backend/events/tests/test_query.py`

**Interfaces:**
- Consumes: 契约模型（Task 2）、`RadarDevice`
- Produces:
  - 表 `PostureRow`、`PresenceRow`、`VitalRow`、`OccupancyRow`（列含 `ts:bigint`, `ts_recv`, `device_id`, `room`, `seq` + 各自字段）
  - `store_event(device_id:str, kind:str, model:BaseModel, ts_recv:int) -> None`（幂等：`(device_id, kind, seq)` 冲突则忽略）
  - REST：`GET /api/families/{id}/events?kind=&room=&since=&until=`（按 `ts` 升序）；`GET /api/families/{id}/today`（每房间最新 presence/posture/vital）

- [ ] **Step 1: 写失败测试（存储幂等 + 以 ts 为轴）**

`backend/events/tests/test_store.py`:
```python
import pytest
pytestmark = pytest.mark.django_db
from events.store import store_event
from events.models import PostureRow
from housafe_contracts.events import parse_event

def test_store_and_idempotent():
    m = parse_event("posture", {"ts":1700000000000,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9})
    store_event("d1","posture",m,ts_recv=1700000000050)
    store_event("d1","posture",m,ts_recv=1700000000060)  # 重复 seq
    rows = PostureRow.objects.filter(device_id="d1")
    assert rows.count() == 1 and rows.first().ts == 1700000000000
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest events/tests/test_store.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 models + store**

`backend/events/models.py`:
```python
from django.db import models

class _Row(models.Model):
    ts = models.BigIntegerField()
    ts_recv = models.BigIntegerField()
    device_id = models.CharField(max_length=40)
    room = models.CharField(max_length=50)
    seq = models.BigIntegerField()
    class Meta:
        abstract = True
        constraints = []
        indexes = [models.Index(fields=["device_id","ts"])]

class PostureRow(_Row):
    posture = models.CharField(max_length=10); confidence = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_posture_seq")]
class PresenceRow(_Row):
    presence = models.BooleanField(); moving = models.BooleanField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_presence_seq")]
class VitalRow(_Row):
    quiet = models.BooleanField(); resp_rate = models.FloatField(null=True)
    heart_rate = models.FloatField(null=True); quality = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_vital_seq")]
class OccupancyRow(_Row):
    count = models.IntegerField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_occ_seq")]
```
`backend/events/store.py`:
```python
from django.db import IntegrityError
from .models import PostureRow, PresenceRow, VitalRow, OccupancyRow

_MAP = {"posture":PostureRow,"presence":PresenceRow,"vital":VitalRow,"occupancy":OccupancyRow}
_FIELDS = {
    "posture": lambda m: {"posture":m.posture,"confidence":m.confidence},
    "presence": lambda m: {"presence":m.presence,"moving":m.moving},
    "vital": lambda m: {"quiet":m.quiet,"resp_rate":m.resp_rate,"heart_rate":m.heart_rate,"quality":m.quality},
    "occupancy": lambda m: {"count":m.count},
}
def store_event(device_id, kind, model, ts_recv):
    Row = _MAP[kind]
    try:
        Row.objects.create(ts=model.ts, ts_recv=ts_recv, device_id=device_id,
                           room=model.room, seq=model.seq, **_FIELDS[kind](model))
    except IntegrityError:
        pass  # 幂等：重复 (device_id, seq) 忽略
```

- [ ] **Step 4: 运行确认通过**

Run: `python manage.py makemigrations events && python manage.py migrate && pytest events/tests/test_store.py -q`
Expected: PASS。

- [ ] **Step 5: Timescale hypertable 迁移**

`backend/events/migrations/0002_hypertables.py`:
```python
from django.db import migrations
SQL = """
SELECT create_hypertable('events_posturerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_presencerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_vitalrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_occupancyrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
"""
class Migration(migrations.Migration):
    dependencies = [("events","0001_initial")]
    operations = [migrations.RunSQL(SQL, reverse_sql=migrations.RunSQL.noop)]
```
> hypertable 要求时间列在主键内。若 `create_hypertable` 报唯一约束错误，将各表唯一约束改为 `("device_id","seq","ts")` 并重生成 0001 迁移。

Run: `python manage.py migrate`
Expected: 无错（hypertable 建成）。

- [ ] **Step 6: 查询 API（today + events）**

`backend/events/tests/test_query.py`:
```python
import pytest
pytestmark = pytest.mark.django_db
from housafe_contracts.events import parse_event
from events.store import store_event

def _setup(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    fid = api.post("/api/families", {"name":"f"}, format="json").data["id"]
    did = api.post(f"/api/families/{fid}/devices/bind", {"room":"bed"}, format="json").data["device_id"]
    return fid, did

def test_today_returns_latest_per_room(api):
    fid, did = _setup(api)
    for seq,p in [(1,"stand"),(2,"walk")]:
        store_event(did,"posture",parse_event("posture",
            {"ts":1700000000000+seq,"radar_id":did,"room":"bed","seq":seq,"posture":p,"confidence":0.9}),ts_recv=0)
    r = api.get(f"/api/families/{fid}/today")
    assert r.status_code == 200 and r.data["bed"]["posture"]["posture"] == "walk"
```
`backend/events/views.py`:
```python
from rest_framework import views, response
from families.models import Family
from .models import PostureRow, PresenceRow, VitalRow

class TodayView(views.APIView):
    def get(self, request, family_id):
        fam = Family.objects.get(id=family_id, owner=request.user)
        dids = list(fam.devices.values_list("device_id", flat=True))
        out = {}
        for did_room in fam.devices.values("device_id","room"):
            room = did_room["room"]; did = did_room["device_id"]
            out.setdefault(room, {})
            for key, Row, fields in [
                ("posture",PostureRow,["posture","confidence"]),
                ("presence",PresenceRow,["presence","moving"]),
                ("vital",VitalRow,["quiet","resp_rate","heart_rate","quality"])]:
                row = Row.objects.filter(device_id=did).order_by("-ts").first()
                if row: out[room][key] = {"ts":row.ts, **{f:getattr(row,f) for f in fields}}
        return response.Response(out)
```
`backend/events/urls.py`:
```python
from django.urls import path
from .views import TodayView
urlpatterns = [path("families/<int:family_id>/today", TodayView.as_view())]
```
settings 加 `"events"`；`housafe/urls.py` 加 `path("api/", include("events.urls"))`。

- [ ] **Step 7: 测试并提交**

Run: `cd backend && pytest events -q`
Expected: PASS。
```bash
git add -A && git commit -m "feat: timescale event storage + today/history query"
```

---

### Task 7: 接入网关 WS（ingest）

**Files:**
- Create: `backend/ingest/__init__.py`, `apps.py`, `consumers.py`, `routing.py`
- Create: `backend/housafe/asgi.py`（改造为 ProtocolTypeRouter）
- Modify: settings（加 `ingest`）
- Test: `backend/ingest/tests/test_ingest.py`

**Interfaces:**
- Consumes: `RadarDevice.verify`（Task 5）、`parse_event`（Task 2）、`store_event`（Task 6）
- Produces:
  - WS 端点 `ws/ingest`：首帧 `{device_id, secret}` 鉴权；后续帧 `{kind, payload}`；校验契约→`store_event`→`group_send("family_<id>", {...})` 供 realtime
  - 鉴权失败 close code 4401；契约非法回 `{error:"invalid"}` 不落库

- [ ] **Step 1: 写失败测试（Channels Communicator）**

`backend/ingest/tests/test_ingest.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

async def _device(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    return await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed"), fam

@pytest.mark.asyncio
async def test_bad_secret_closes(db):
    await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id":"d1","secret":"WRONG"})
    msg = await c.receive_output()
    assert msg["type"] == "websocket.close"

@pytest.mark.asyncio
async def test_valid_event_persists(db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id":"d1","secret":"s1"})
    await c.receive_json_from()  # ack
    await c.send_json_to({"kind":"posture","payload":{"ts":1700000000000,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9}})
    await c.receive_json_from()  # stored ack
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 1
    await c.disconnect()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest ingest -q`
Expected: FAIL（asgi/consumer 缺失）。

- [ ] **Step 3: 实现 consumer + routing + asgi**

`backend/ingest/consumers.py`:
```python
import time
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from pydantic import ValidationError
from housafe_contracts.events import parse_event
from devices.models import RadarDevice
from events.store import store_event

class IngestConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.device = None
        await self.accept()

    @database_sync_to_async
    def _verify(self, did, secret): return RadarDevice.verify(did, secret)
    @database_sync_to_async
    def _family_id(self): return self.device.family_id
    @database_sync_to_async
    def _store(self, kind, model): store_event(self.device.device_id, kind, model, ts_recv=int(time.time()*1000))

    async def receive_json(self, content):
        if self.device is None:
            self.device = await self._verify(content.get("device_id"), content.get("secret"))
            if self.device is None:
                await self.close(code=4401); return
            self.group = f"family_{await self._family_id()}"
            await self.channel_layer.group_add(self.group, self.channel_name)
            await self.send_json({"ack":"auth"}); return
        try:
            model = parse_event(content["kind"], content["payload"])
        except (ValidationError, ValueError, KeyError):
            await self.send_json({"error":"invalid"}); return
        await self._store(content["kind"], model)
        await self.channel_layer.group_send(self.group,
            {"type":"event.push","kind":content["kind"],"payload":content["payload"]})
        await self.send_json({"ack":"stored","seq":model.seq})
```
`backend/ingest/routing.py`:
```python
from django.urls import path
from .consumers import IngestConsumer
websocket_urlpatterns = [path("ws/ingest", IngestConsumer.as_asgi())]
```
`backend/housafe/asgi.py`:
```python
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE","housafe.settings")
from django.core.asgi import get_asgi_application
django_asgi = get_asgi_application()
from channels.routing import ProtocolTypeRouter, URLRouter
import ingest.routing, realtime.routing
application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": URLRouter(ingest.routing.websocket_urlpatterns + realtime.routing.websocket_urlpatterns),
})
```
> `realtime.routing` 在 Task 8 建。为让 Task 7 独立测试通过，先在 asgi 里只引 `ingest.routing`，Task 8 再加 `realtime`。

- [ ] **Step 4: 运行确认通过并提交**

Run: `cd backend && pytest ingest -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: ingest ws gateway (device auth, contract validate, persist, fanout)"
```

---

### Task 8: 实时推送 WS（realtime）

**Files:**
- Create: `backend/realtime/__init__.py`, `apps.py`, `consumers.py`, `routing.py`
- Modify: `backend/housafe/asgi.py`（加 realtime 路由），settings（加 `realtime`）
- Test: `backend/realtime/tests/test_realtime.py`

**Interfaces:**
- Consumes: ingest 的 `group_send("family_<id>", {type:"event.push",...})`；JWT
- Produces: WS 端点 `ws/app?token=<jwt>&family=<id>`：JWT 鉴权 + 校验用户属该家庭 → 加入 `family_<id>` 组 → 收到 `event.push` 转发给 App

- [ ] **Step 1: 写失败测试（端到端：ingest 灌 → app 端收到）**

`backend/realtime/tests/test_realtime.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.mark.asyncio
async def test_ingest_to_app_flow(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    from rest_framework_simplejwt.tokens import AccessToken
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    token = str(AccessToken.for_user(u))

    app = WebsocketCommunicator(application, f"/ws/app?token={token}&family={fam.id}")
    assert (await app.connect())[0]
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id":"d1","secret":"s1"}); await ing.receive_json_from()
    await ing.send_json_to({"kind":"posture","payload":{"ts":1,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9}})
    pushed = await app.receive_json_from()
    assert pushed["kind"] == "posture" and pushed["payload"]["room"] == "bed"
    await app.disconnect(); await ing.disconnect()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest realtime -q`
Expected: FAIL。

- [ ] **Step 3: 实现 realtime consumer + routing**

`backend/realtime/consumers.py`:
```python
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async

class AppConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        qs = parse_qs(self.scope["query_string"].decode())
        token = qs.get("token",[None])[0]; family_id = qs.get("family",[None])[0]
        user = await self._auth(token)
        if user is None or not await self._owns(user, family_id):
            await self.close(code=4401); return
        self.group = f"family_{family_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    @database_sync_to_async
    def _auth(self, token):
        if not token: return None
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth.models import User
            return User.objects.get(id=AccessToken(token)["user_id"])
        except Exception:
            return None
    @database_sync_to_async
    def _owns(self, user, family_id):
        from families.models import Family
        return Family.objects.filter(id=family_id, owner=user).exists()

    async def event_push(self, event):
        await self.send_json({"kind":event["kind"],"payload":event["payload"]})

    async def disconnect(self, code):
        if hasattr(self,"group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
```
`backend/realtime/routing.py`:
```python
from django.urls import path
from .consumers import AppConsumer
websocket_urlpatterns = [path("ws/app", AppConsumer.as_asgi())]
```
确认 `housafe/asgi.py` 已把 `realtime.routing.websocket_urlpatterns` 并入 URLRouter（见 Task 7 Step 3）。settings 加 `"realtime"`。

- [ ] **Step 4: 运行确认通过并提交**

Run: `cd backend && pytest realtime -q`
Expected: PASS。
```bash
git add -A && git commit -m "feat: realtime app ws (jwt auth, family scoping, event fanout)"
```

---

### Task 9: 最小仿真灌数脚本 + 端到端冒烟

**Files:**
- Create: `simulator/__init__.py`, `simulator/feed.py`, `simulator/README.md`
- Test: `backend/tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: `ws/ingest`（Task 7）、契约（Task 2）
- Produces: `feed.py`：给定 `device_id/secret/ws_url`，按脚本发一串事件（presence→posture 序列 + vital），用于手动演示；`test_e2e_smoke.py` 断言全链路。

- [ ] **Step 1: 写端到端冒烟测试**

`backend/tests/test_e2e_smoke.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.mark.asyncio
async def test_full_chain_device_to_query(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id":"d1","secret":"s1"}); await ing.receive_json_from()
    for seq,p in [(1,"stand"),(2,"walk"),(3,"lie")]:
        await ing.send_json_to({"kind":"posture","payload":{"ts":1700000000000+seq,"radar_id":"d1","room":"bed","seq":seq,"posture":p,"confidence":0.9}})
        await ing.receive_json_from()
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 3
    await ing.disconnect()
```

- [ ] **Step 2: 运行确认通过**

Run: `cd backend && pytest tests/test_e2e_smoke.py -q`
Expected: PASS。

- [ ] **Step 3: 写演示用 feed 脚本**

`simulator/feed.py`:
```python
"""最小灌数脚本：python feed.py <device_id> <secret> [ws_url]"""
import asyncio, sys, time, json, websockets

async def main(device_id, secret, url="ws://localhost:8000/ws/ingest"):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"device_id":device_id,"secret":secret}))
        print(await ws.recv())
        seq = 0
        script = [("presence",{"presence":True,"moving":False}),
                  ("posture",{"posture":"stand","confidence":0.95}),
                  ("posture",{"posture":"walk","confidence":0.9}),
                  ("vital",{"quiet":True,"resp_rate":16.0,"heart_rate":72.0,"quality":0.8}),
                  ("posture",{"posture":"lie","confidence":0.92})]
        for kind, fields in script:
            seq += 1
            payload = {"ts":int(time.time()*1000),"radar_id":device_id,"room":"bedroom","seq":seq, **fields}
            await ws.send(json.dumps({"kind":kind,"payload":payload}))
            print(await ws.recv()); await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
```
`simulator/README.md`：说明先 `bind` 拿 device_id/secret，再 `pip install websockets && python feed.py <id> <secret>`，配合 App 观察实时刷新。

- [ ] **Step 4: 提交**

```bash
git add -A && git commit -m "feat: minimal simulator feed script + e2e smoke test"
```

---

### Task 10: 最小 App 壳（RN + Expo，含设计系统）

> **本任务必须先调用 `frontend-design` skill 确立视觉语言，再写屏幕。** 目标是一个「产品级、有设计感、可信温暖」的界面，不是毛坯房、不要 AI 通用紫渐变风。
>
> **设计基调（养老监护 · 面向 35-50 岁子女）：** 安心 / 温暖 / 可信 / 清晰。暖白背景、沉静的青绿主色（健康与安全感）、琥珀色表示需关注、柔珊瑚表示告警、暖灰文字层级分明。留白充足、圆角柔和、卡片有柔和投影、触控目标够大。状态用颜色语义传达（正常=青绿、关注=琥珀、告警=珊瑚），而非只靠文字。

**Files:**
- Create: `app/package.json`, `app/app.json`, `app/App.tsx`
- Create: `app/src/theme.ts`（设计 token 单一来源）
- Create: `app/src/components/ui.tsx`（Screen / Card / StatusBadge / VitalStat / SectionHeader）
- Create: `app/src/api.ts`, `app/src/LoginScreen.tsx`, `app/src/TodayScreen.tsx`
- Create: `app/README.md`

**Interfaces:**
- Consumes: REST（login、families、today）、`ws/app`（realtime）
- Produces: 可在 Expo 跑的 App：登录 → 今日状态屏实时显示设备在线 + 姿态/生命体征，**带完整加载态/空态/错误态**。

- [ ] **Step 1: 调用 frontend-design skill 确立视觉语言**

调用 `frontend-design` skill，产出：配色（主/辅/语义色）、字号层级、间距刻度、圆角/投影规范，写入 `app/src/theme.ts`。基线 token（可被 skill 优化，但不得回退成无设计的默认样式）：

```typescript
// app/src/theme.ts
export const colors = {
  bg: "#F7F5F1",          // 暖白背景
  surface: "#FFFFFF",
  primary: "#2E7D6F",     // 沉静青绿（健康/安全）
  primarySoft: "#E3EFEB",
  attention: "#C98A2B",   // 琥珀（需关注）
  attentionSoft: "#F6ECD9",
  alert: "#D96A5B",       // 柔珊瑚（告警）
  alertSoft: "#F8E4E0",
  text: "#2B2B2B",
  textMuted: "#7A756E",
  border: "#E7E2DA",
};
export const space = { xs:4, sm:8, md:12, lg:16, xl:24, xxl:32 };
export const radius = { sm:8, md:12, lg:20, pill:999 };
export const font = { h1:28, h2:20, body:16, small:13 };
export const shadow = {
  card: { shadowColor:"#000", shadowOpacity:0.06, shadowRadius:12, shadowOffset:{width:0,height:4}, elevation:2 },
};
```

- [ ] **Step 2: 初始化 Expo 工程 + 图标**

```bash
cd app && npx create-expo-app@latest . --template blank-typescript
npx expo install expo-secure-store @expo/vector-icons react-native-safe-area-context
```
`app/README.md`：`API_BASE` 改为本机局域网 IP（真机连不上 localhost），`npx expo start`。

- [ ] **Step 3: 设计系统组件**

`app/src/components/ui.tsx`:
```typescript
import { View, Text, ViewProps } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors, space, radius, font, shadow } from "../theme";

export function Screen({ children }: ViewProps) {
  return (
    <SafeAreaView style={{ flex:1, backgroundColor: colors.bg }}>
      <View style={{ flex:1, paddingHorizontal: space.xl }}>{children}</View>
    </SafeAreaView>
  );
}
export function Card({ children, tone="normal" }: { children: any; tone?: "normal"|"attention"|"alert" }) {
  const bar = { normal: colors.primary, attention: colors.attention, alert: colors.alert }[tone];
  return (
    <View style={{ backgroundColor: colors.surface, borderRadius: radius.lg, padding: space.lg,
      marginBottom: space.md, borderLeftWidth:4, borderLeftColor: bar, ...shadow.card }}>
      {children}
    </View>
  );
}
export function StatusBadge({ online }: { online: boolean }) {
  const c = online ? colors.primary : colors.textMuted;
  const soft = online ? colors.primarySoft : colors.border;
  return (
    <View style={{ flexDirection:"row", alignItems:"center", gap: space.xs, alignSelf:"flex-start",
      backgroundColor: soft, paddingHorizontal: space.md, paddingVertical: space.xs, borderRadius: radius.pill }}>
      <View style={{ width:8, height:8, borderRadius:4, backgroundColor:c }} />
      <Text style={{ color:c, fontSize: font.small, fontWeight:"600" }}>{online ? "在线" : "离线"}</Text>
    </View>
  );
}
export function VitalStat({ icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <View style={{ flex:1, alignItems:"center", gap: space.xs }}>
      <Ionicons name={icon} size={22} color={colors.primary} />
      <Text style={{ fontSize: font.h2, fontWeight:"700", color: colors.text }}>{value}</Text>
      <Text style={{ fontSize: font.small, color: colors.textMuted }}>{label}</Text>
    </View>
  );
}
export function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={{ marginTop: space.lg, marginBottom: space.md }}>
      <Text style={{ fontSize: font.h1, fontWeight:"700", color: colors.text }}>{title}</Text>
      {subtitle ? <Text style={{ fontSize: font.body, color: colors.textMuted, marginTop: space.xs }}>{subtitle}</Text> : null}
    </View>
  );
}
```

- [ ] **Step 4: API 客户端**

`app/src/api.ts`:
```typescript
export const API_BASE = "http://192.168.1.100:8000"; // 改成你本机局域网 IP
export const WS_BASE = "ws://192.168.1.100:8000";

export async function login(username: string, password: string) {
  const r = await fetch(`${API_BASE}/api/auth/login`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) throw new Error("登录失败");
  return (await r.json()).access as string;
}
export async function getFamilies(token: string) {
  const r = await fetch(`${API_BASE}/api/families`, { headers:{ Authorization:`Bearer ${token}` }});
  if (!r.ok) throw new Error("加载家庭失败");
  return r.json();
}
export async function getToday(token: string, familyId: number) {
  const r = await fetch(`${API_BASE}/api/families/${familyId}/today`, { headers:{ Authorization:`Bearer ${token}` }});
  if (!r.ok) throw new Error("加载状态失败");
  return r.json();
}
```

- [ ] **Step 5: 登录屏（设计版）**

`app/src/LoginScreen.tsx`:
```typescript
import { useState } from "react";
import { View, Text, TextInput, Pressable, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Screen } from "./components/ui";
import { colors, space, radius, font, shadow } from "./theme";
import { login } from "./api";

export default function LoginScreen({ onLogin }: { onLogin: (t: string) => void }) {
  const [u, setU] = useState(""); const [p, setP] = useState("");
  const [err, setErr] = useState(""); const [loading, setLoading] = useState(false);
  const input = { backgroundColor: colors.surface, borderWidth:1, borderColor: colors.border,
    borderRadius: radius.md, padding: space.lg, fontSize: font.body, color: colors.text } as const;
  return (
    <Screen>
      <View style={{ flex:1, justifyContent:"center", gap: space.md }}>
        <View style={{ alignItems:"center", marginBottom: space.xl, gap: space.sm }}>
          <View style={{ width:64, height:64, borderRadius:radius.lg, backgroundColor: colors.primarySoft,
            alignItems:"center", justifyContent:"center" }}>
            <Ionicons name="shield-checkmark" size={34} color={colors.primary} />
          </View>
          <Text style={{ fontSize: font.h1, fontWeight:"700", color: colors.text }}>家安</Text>
          <Text style={{ fontSize: font.body, color: colors.textMuted }}>随时掌握父母在家的安好</Text>
        </View>
        <TextInput placeholder="用户名" placeholderTextColor={colors.textMuted} value={u}
          onChangeText={setU} autoCapitalize="none" style={input} />
        <TextInput placeholder="密码" placeholderTextColor={colors.textMuted} value={p}
          onChangeText={setP} secureTextEntry style={input} />
        {err ? <Text style={{ color: colors.alert, fontSize: font.small }}>{err}</Text> : null}
        <Pressable disabled={loading} onPress={async () => {
            setErr(""); setLoading(true);
            try { onLogin(await login(u, p)); } catch { setErr("登录失败，请检查账号密码"); }
            finally { setLoading(false); }
          }}
          style={{ backgroundColor: colors.primary, padding: space.lg, borderRadius: radius.md,
            alignItems:"center", marginTop: space.sm, ...shadow.card }}>
          {loading ? <ActivityIndicator color="#fff" /> :
            <Text style={{ color:"#fff", fontSize: font.body, fontWeight:"600" }}>登录</Text>}
        </Pressable>
      </View>
    </Screen>
  );
}
```

- [ ] **Step 6: 今日状态屏（设计版，含加载/空/错误态）**

`app/src/TodayScreen.tsx`:
```typescript
import { useEffect, useState } from "react";
import { View, Text, ScrollView, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Screen, Card, VitalStat, SectionHeader } from "./components/ui";
import { colors, space, font } from "./theme";
import { getFamilies, getToday, WS_BASE } from "./api";

const POSTURE_CN: Record<string,string> = { stand:"站立", sit:"坐着", lie:"躺卧", walk:"走动", fall:"跌倒" };

export default function TodayScreen({ token }: { token: string }) {
  const [rooms, setRooms] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true); const [err, setErr] = useState("");

  useEffect(() => {
    let ws: WebSocket | undefined;
    (async () => {
      try {
        const fams = await getFamilies(token);
        if (!fams.length) { setLoading(false); return; }
        const fid = fams[0].id;
        setRooms(await getToday(token, fid)); setLoading(false);
        ws = new WebSocket(`${WS_BASE}/ws/app?token=${token}&family=${fid}`);
        ws.onmessage = (e) => {
          const { kind, payload } = JSON.parse(e.data);
          setRooms(prev => ({ ...prev, [payload.room]: { ...prev[payload.room], [kind]: payload } }));
        };
      } catch (e:any) { setErr(e.message || "加载失败"); setLoading(false); }
    })();
    return () => ws?.close();
  }, [token]);

  if (loading) return <Screen><View style={{ flex:1, justifyContent:"center" }}><ActivityIndicator color={colors.primary} size="large" /></View></Screen>;

  const entries = Object.entries(rooms);
  return (
    <Screen>
      <ScrollView showsVerticalScrollIndicator={false}>
        <SectionHeader title="今日状态" subtitle="父母在家的实时安好" />
        {err ? <Card tone="alert"><Text style={{ color: colors.alert }}>{err}</Text></Card> : null}
        {!err && entries.length === 0 ? (
          <Card>
            <View style={{ alignItems:"center", gap: space.sm, paddingVertical: space.lg }}>
              <Ionicons name="wifi-outline" size={30} color={colors.textMuted} />
              <Text style={{ color: colors.textMuted, textAlign:"center" }}>
                暂无数据{"\n"}绑定雷达并运行仿真灌数脚本后，此处会实时刷新
              </Text>
            </View>
          </Card>
        ) : null}
        {entries.map(([room, s]: any) => {
          const fall = s.posture?.posture === "fall";
          return (
            <Card key={room} tone={fall ? "alert" : "normal"}>
              <View style={{ flexDirection:"row", justifyContent:"space-between", alignItems:"center", marginBottom: space.md }}>
                <Text style={{ fontSize: font.h2, fontWeight:"700", color: colors.text }}>{room}</Text>
                {s.presence ? (
                  <Text style={{ color: s.presence.presence ? colors.primary : colors.textMuted, fontWeight:"600" }}>
                    {s.presence.presence ? (s.presence.moving ? "有人・移动中" : "有人・静止") : "无人"}
                  </Text>
                ) : null}
              </View>
              {s.posture ? (
                <Text style={{ fontSize: font.body, color: fall ? colors.alert : colors.text, marginBottom: space.md }}>
                  当前姿态：{POSTURE_CN[s.posture.posture] ?? s.posture.posture}
                </Text>
              ) : null}
              {s.vital ? (
                <View style={{ flexDirection:"row", borderTopWidth:1, borderTopColor: colors.border, paddingTop: space.md }}>
                  <VitalStat icon="pulse" label="心率 bpm" value={s.vital.heart_rate ?? "-"} />
                  <VitalStat icon="water" label="呼吸 次/分" value={s.vital.resp_rate ?? "-"} />
                </View>
              ) : null}
            </Card>
          );
        })}
      </ScrollView>
    </Screen>
  );
}
```

- [ ] **Step 7: 组装 App.tsx**

`app/App.tsx`:
```typescript
import { useState } from "react";
import { SafeAreaProvider } from "react-native-safe-area-context";
import LoginScreen from "./src/LoginScreen";
import TodayScreen from "./src/TodayScreen";

export default function App() {
  const [token, setToken] = useState<string | null>(null);
  return (
    <SafeAreaProvider>
      {token ? <TodayScreen token={token} /> : <LoginScreen onLogin={setToken} />}
    </SafeAreaProvider>
  );
}
```

- [ ] **Step 8: 手动端到端验证并提交**

后端 `daphne housafe.asgi:application` + docker 起 db/redis：
1. App 注册/登录
2. curl bind 一台设备拿 device_id/secret
3. `python simulator/feed.py <id> <secret>`
4. App「今日状态」实时刷新，姿态/心率/呼吸卡片跳动；跌倒时卡片转珊瑚色 → 验证通过
5. 检查：无毛坯感、层级分明、有加载/空/错误态、跌倒有语义色

```bash
git add -A && git commit -m "feat: designed expo app shell (theme system + login + realtime today)"
```

---

## 自审记录（Spec 覆盖核对）

- 契约防腐层 → Task 2 ✓；账户 → Task 3 ✓；家庭/老人/联系人 → Task 4 ✓；设备绑定/在线 → Task 5 ✓；Timescale 时序存储 + today/history 查询 → Task 6 ✓；ingest WS（鉴权/校验/落库/分发）→ Task 7 ✓；realtime WS（JWT/家庭隔离/推送）→ Task 8 ✓；最小 App 壳 → Task 10 ✓；端到端冒烟 + 灌数脚本 → Task 9 ✓。
- `ts` 权威时间轴：契约含 `ts/ts_recv`（Task 2），store 以 `ts` 落库（Task 6），贯穿。
- `seq` 去重：Task 6 唯一约束 + store 幂等 ✓。
- 明确不做项（点云/ALG-0/ALG-1~6/通知/OTA 真实/MQTT）→ 计划中无对应任务，符合范围。
- 占位接口：ai/ 目录留空、Redis 分发通道已由 ingest `group_send` 打通，子项目③可直接消费。
- 类型一致性：`store_event(device_id, kind, model, ts_recv)`、`RadarDevice.verify`、`parse_event(kind, payload)` 在各任务签名一致 ✓。
